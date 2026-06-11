#!/usr/bin/env python3
# 只跑 ZipMap_AR，不跑 ReSplat，输出 pose / depth / ATE / timing
"""
左目图片文件夹
→ ZipMap_AR + checkpoint_online.pt
→ model(images)  # 和官方 Gradio demo 一样，一次性输入全部图片
→ 输出 pose
→ 计算 ATE/RPE
→ 记录时间成本

命令：
TORCH_COMPILE_DISABLE=1 python run_zipmap_official_streaming_demo_cli_eval.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --ckpt_path /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_online.pt \
  --image_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
  --output_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_official_streaming_cli_P000_0_80 \
  --start_index 0 \
  --end_index 80 \
  --device cuda:0 \
  --pose_alignment sim3
"""
"""
Headless CLI version of ZipMap's official streaming Gradio demo.

This script is for SSH/headless machines.

It reproduces the core inference path of demo_gradio_zipmap_streaming.py:
  - ZipMap_AR
  - official streaming model_config
  - checkpoint_online.pt
  - load_and_preprocess_images(...)
  - predictions = model(images)
  - pose_encoding_to_extri_intri(...)
  - optional align_first_view

It does NOT use Gradio.
It does NOT run stateful one-frame-at-a-time inference.
It does NOT load ReSplat.

Outputs:
  output_dir/
    predictions_official_streaming_pose.npz
    poses_c2w_opencv_tum.txt
    poses_w2c_opencv.txt
    poses_c2w_opencv_matrix.txt
    poses_resplat_tartanair_loader.txt
    poses_resplat_tartanair_loader_tum.txt
    intrinsics.txt
    selected_frames.csv/json
    timing_summary.json
    pose_eval/summary.json                 if --gt_pose_file is provided
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Exactly mirrors demo_gradio_zipmap_streaming.py.
MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": False,
    "enable_camera_mlp": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": {
            "bias": True,
            "head_dim": 1024,
            "inter_multi": 2,
            "base_lr": 0.01,
            "muon_update_steps": 5,
            "use_gate_fn": True,
        },
        "window_size": 1,
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def abs_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def log(msg: str) -> None:
    print(msg, flush=True)


def import_module_from_path(name: str, path: Path):
    path = abs_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Module path does not exist: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect_images(root: Path, recursive: bool = False) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image files found under: {root}")
    return paths


def make_sample_indices(
    n: int,
    start_index: int,
    end_index: Optional[int],
    stride: int,
    max_count: Optional[int],
) -> list[int]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    start = max(0, int(start_index))
    end = n if end_index is None else min(n, int(end_index))
    if start >= end:
        raise ValueError(f"Invalid range: start_index={start_index}, end_index={end_index}, length={n}")
    indices = list(range(start, end, stride))
    if max_count is not None:
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")
        indices = indices[:max_count]
    if not indices:
        raise ValueError("Sampling produced zero frames.")
    return indices


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    cuda_sync(device)
    t0 = time.perf_counter()
    out = fn()
    cuda_sync(device)
    return out, time.perf_counter() - t0


def tensor_to_np_no_batch(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr


def load_model(args, device: torch.device):
    sys.path.insert(0, str(abs_path(args.zipmap_repo)))
    from zipmap.models.ZipMap_AR import ZipMap

    model_config = json.loads(json.dumps(MODEL_CONFIG))
    model_config["other_config"]["affine_invariant"] = bool(args.affine_invariant)

    model = ZipMap(**model_config)

    try:
        checkpoint = torch.load(abs_path(args.ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(abs_path(args.ckpt_path), map_location="cpu")

    if args.ema and isinstance(checkpoint, dict) and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        log("[ZipMap] using EMA weights")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log(f"[ZipMap] missing keys: {missing}")
    if unexpected:
        log(f"[ZipMap] unexpected keys: {unexpected}")

    model.eval().to(device)
    return model, model_config


def main() -> None:
    torch.set_float32_matmul_precision("high")

    ap = argparse.ArgumentParser(description="Headless official ZipMap streaming-demo pose evaluator")

    ap.add_argument("--zipmap_repo", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--image_dir", required=True, help="Directory containing images, e.g. image_lcam_front")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=None, help="Exclusive end index.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_frames", type=int, default=None)

    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--affine_invariant", type=str2bool, default=True)
    ap.add_argument("--align_first_view", type=str2bool, default=True)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--gt_pose_file", default=None)
    ap.add_argument("--pose_alignment", choices=["none", "se3", "sim3"], default="sim3")
    ap.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    ap.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    ap.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    ap.add_argument("--rpe_delta", type=int, default=1)

    ap.add_argument(
        "--save_extra_predictions",
        type=str2bool,
        default=False,
        help="Also save pose_enc/depth/depth_conf/local_points/local_points_conf/images if available. Large files.",
    )

    args = ap.parse_args()

    zipmap_repo = abs_path(args.zipmap_repo)
    out_dir = abs_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (zipmap_repo / "zipmap").is_dir():
        raise FileNotFoundError(f"zipmap package not found under --zipmap_repo: {zipmap_repo}")

    export_mod = import_module_from_path(
        "zipmap_export_official_streaming_cli",
        zipmap_repo / "tools" / "export_zipmap_predictions.py",
    )
    eval_mod = import_module_from_path(
        "zipmap_eval_official_streaming_cli",
        zipmap_repo / "tools" / "evaluate_zipmap_pose.py",
    )

    sys.path.insert(0, str(zipmap_repo))
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required. Requested device={args.device}, cuda_available={torch.cuda.is_available()}")

    all_paths = collect_images(abs_path(args.image_dir), recursive=args.recursive)
    indices = make_sample_indices(
        len(all_paths),
        start_index=args.start_index,
        end_index=args.end_index,
        stride=args.stride,
        max_count=args.num_frames,
    )
    image_paths = [all_paths[i] for i in indices]

    log(f"[Input] selected {len(image_paths)} image(s): original {indices[0]}..{indices[-1]}")
    log(f"[Output] {out_dir}")
    log("[Mode] official streaming demo core: ZipMap_AR + checkpoint_online + model(images)")

    frame_records = [
        {
            "local_index": int(local_i),
            "original_index": int(original_i),
            "source_name": image_paths[local_i].name,
            "source_path": str(image_paths[local_i]),
        }
        for local_i, original_i in enumerate(indices)
    ]
    save_json(out_dir / "selected_frames.json", frame_records)
    with (out_dir / "selected_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_records[0].keys()))
        writer.writeheader()
        writer.writerows(frame_records)

    model, model_config = load_model(args, device)

    def preprocess():
        return load_and_preprocess_images(
            [str(p) for p in image_paths],
            target_size=args.target_size,
            mode=args.preprocess_mode,
        ).to(device)

    images, preprocess_sec = timed(device, preprocess)
    log(f"[Preprocess] images={tuple(images.shape)}, time={preprocess_sec:.4f}s")

    def infer():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            return model(images)

    predictions_raw, infer_sec = timed(device, infer)
    log(f"[Infer] time={infer_sec:.4f}s")

    def postprocess():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions_raw["pose_enc"], images.shape[-2:])
        predictions_raw["extrinsic"] = extrinsic
        predictions_raw["intrinsic"] = intrinsic

        extrinsic_np = tensor_to_np_no_batch(extrinsic).astype(np.float32)
        intrinsic_np = tensor_to_np_no_batch(intrinsic).astype(np.float32)

        T_w2c = export_mod.to_homogeneous_4x4(extrinsic_np)
        if args.align_first_view:
            T_w2c = export_mod.align_w2c_to_first_camera(T_w2c).astype(np.float32)
            extrinsic_np = T_w2c[:, :3, :4].astype(np.float32)

        T_c2w = export_mod.invert_se3_np(T_w2c).astype(np.float32)
        return extrinsic_np, intrinsic_np, T_w2c.astype(np.float32), T_c2w

    (extrinsic_np, intrinsic_np, T_w2c, T_c2w), postprocess_sec = timed(device, postprocess)
    log(f"[Postprocess] time={postprocess_sec:.4f}s")

    pred_to_save: dict[str, Any] = {
        "extrinsic": extrinsic_np,
        "intrinsic": intrinsic_np,
        "T_w2c_opencv": T_w2c,
        "T_c2w_opencv": T_c2w,
        "selected_original_indices": np.asarray(indices, dtype=np.int64),
    }

    if args.save_extra_predictions:
        for key in ("pose_enc", "depth", "depth_conf", "local_points", "local_points_conf", "images"):
            if key in predictions_raw and torch.is_tensor(predictions_raw[key]):
                pred_to_save[key] = tensor_to_np_no_batch(predictions_raw[key])

    np.savez_compressed(out_dir / "predictions_official_streaming_pose.npz", **pred_to_save)

    timestamps = [p.stem for p in image_paths]
    export_mod.write_tum_pose_file(out_dir / "poses_c2w_opencv_tum.txt", T_c2w, timestamps=timestamps)
    export_mod.write_matrix_file(out_dir / "poses_w2c_opencv.txt", T_w2c[:, :3, :4])
    export_mod.write_matrix_file(out_dir / "poses_c2w_opencv_matrix.txt", T_c2w)

    T_resplat_loader_pose = export_mod.compute_resplat_loader_pose_from_cv(T_c2w).astype(np.float32)
    export_mod.write_pose7_file(out_dir / "poses_resplat_tartanair_loader.txt", T_resplat_loader_pose)
    export_mod.write_tum_pose_file(
        out_dir / "poses_resplat_tartanair_loader_tum.txt",
        T_resplat_loader_pose,
        timestamps=timestamps,
    )
    export_mod.save_intrinsics(out_dir / "intrinsics.txt", intrinsic_np)

    timing_summary = {
        "num_frames": len(image_paths),
        "preprocess_sec": float(preprocess_sec),
        "infer_sec": float(infer_sec),
        "postprocess_sec": float(postprocess_sec),
        "total_sec": float(preprocess_sec + infer_sec + postprocess_sec),
        "infer_sec_per_frame": float(infer_sec / max(1, len(image_paths))),
        "total_sec_per_frame": float((preprocess_sec + infer_sec + postprocess_sec) / max(1, len(image_paths))),
    }
    save_json(out_dir / "timing_summary.json", timing_summary)

    pose_eval_summary: dict[str, Any] = {"gt_used": False}
    if args.gt_pose_file is not None:
        gt_path = abs_path(args.gt_pose_file)
        T_gt_all = eval_mod.load_gt_trajectory(
            gt_path,
            args.gt_convention,
            args.gt_quat_order,
            args.gt_matrix_convention,
        )
        T_gt = T_gt_all[indices]

        if args.pose_alignment == "none":
            T_eval = T_c2w.astype(np.float64)
            scale = 1.0
            R_align = np.eye(3)
            t_align = np.zeros(3)
        else:
            pred_centers = T_c2w[:, :3, 3].astype(np.float64)
            gt_centers = T_gt[:, :3, 3].astype(np.float64)
            with_scale = args.pose_alignment == "sim3"
            scale, R_align, t_align = eval_mod.umeyama_alignment(pred_centers, gt_centers, with_scale=with_scale)
            T_eval = eval_mod.apply_similarity_to_poses(
                T_c2w.astype(np.float64),
                scale,
                R_align,
                t_align,
            ).astype(np.float64)

        ate = eval_mod.compute_ate(T_eval, T_gt.astype(np.float64))
        rpe_rows, rpe = eval_mod.compute_rpe(T_eval, T_gt.astype(np.float64), delta=args.rpe_delta)

        pose_eval_dir = out_dir / "pose_eval"
        pose_eval_dir.mkdir(parents=True, exist_ok=True)

        eval_mod.write_pose_tum(
            pose_eval_dir / "trajectory_pred_aligned_c2w_opencv.txt",
            T_eval.astype(np.float32),
            names=[p.name for p in image_paths],
        )
        eval_mod.write_pose_tum(
            pose_eval_dir / "trajectory_gt_matched_c2w_opencv.txt",
            T_gt.astype(np.float32),
            names=[p.name for p in image_paths],
        )

        err = np.linalg.norm(T_eval[:, :3, 3] - T_gt[:, :3, 3], axis=1)
        with (pose_eval_dir / "per_frame_errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["local_index", "original_index", "source_name", "ate_translation_error"],
            )
            writer.writeheader()
            for local_i, e in enumerate(err):
                writer.writerow(
                    {
                        "local_index": int(local_i),
                        "original_index": int(indices[local_i]),
                        "source_name": image_paths[local_i].name,
                        "ate_translation_error": float(e),
                    }
                )

        with (pose_eval_dir / "rpe_errors.csv").open("w", newline="", encoding="utf-8") as f:
            if rpe_rows:
                writer = csv.DictWriter(f, fieldnames=list(rpe_rows[0].keys()))
                writer.writeheader()
                writer.writerows(rpe_rows)

        pose_eval_summary = {
            "gt_used": True,
            "gt_pose_file": str(gt_path),
            "alignment": args.pose_alignment,
            "scale": float(scale),
            "R": R_align.tolist(),
            "t": t_align.tolist(),
            "ate": ate,
            "rpe": rpe,
            "per_frame_errors_csv": str(pose_eval_dir / "per_frame_errors.csv"),
        }
        save_json(pose_eval_dir / "summary.json", pose_eval_summary)

        log(
            f"[ATE] alignment={args.pose_alignment}, "
            f"rmse={ate['rmse']:.6f}, mean={ate['mean']:.6f}, "
            f"scale={float(scale):.6f}"
        )

    save_json(
        out_dir / "run_summary.json",
        {
            "mode": "official_streaming_demo_cli",
            "output_dir": str(out_dir),
            "zipmap_repo": str(zipmap_repo),
            "ckpt_path": str(abs_path(args.ckpt_path)),
            "image_dir": str(abs_path(args.image_dir)),
            "num_frames": len(image_paths),
            "selected_original_indices": [int(i) for i in indices],
            "align_first_view": bool(args.align_first_view),
            "pose_eval": pose_eval_summary,
            "timing_summary": timing_summary,
            "model_config": model_config,
        },
    )

    log("[Done]")
    log(f"  predictions: {out_dir / 'predictions_official_streaming_pose.npz'}")
    log(f"  resplat pose: {out_dir / 'poses_resplat_tartanair_loader.txt'}")
    log(f"  timing: {out_dir / 'timing_summary.json'}")
    if args.gt_pose_file is not None:
        log(f"  pose eval: {out_dir / 'pose_eval' / 'summary.json'}")


if __name__ == "__main__":
    main()
