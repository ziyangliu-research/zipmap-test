#!/usr/bin/env python3
# zipmap流式传输，完整版本
# TartanAir stereo → ZipMap pose/depth → ReSplat Gaussian packets → fusion/render
# 添加了选取resplat输出的init高斯进行融合的功能
"""
命令：
TORCH_COMPILE_DISABLE=1 python run_zipmap_resplat_fusion_api_official_streaming_merged_init_fast.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --zipmap_ckpt /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_online.pt \
  --left_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --right_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_rcam_front \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
  --work_dir /home/shiyo/Desktop/ZipMap/outputs/api_official_streaming_resplat_P000_0_50_init_fast \
  --start_index 0 \
  --end_index 50 \
  --scene_name P000 \
  --resplat_experiment tartanair_p000_ft \
  --resplat_packet_stage init \
  --packet_stage_for_fusion init \
  --fusion_probe_mode packet_trajectory \
  --trajectory_render_chunk_size 1 \
  --device cuda:0 \
  --pose_alignment sim3 \
  --resplat_pose_source aligned \
  --save_zipmap_depth false \
  --save_zipmap_points false
"""
"""
API-style official ZipMap-Streaming -> ReSplat packet -> Gaussian prefix-fusion pipeline.

This is intentionally different from the older command-wrapper script:
  - ZipMap is called as Python functions, not as a subprocess.
  - ReSplat is instantiated from its Hydra config once, then fed in-memory batches.
  - ReSplat view_sampler JSON is not used for packet generation.
  - Gaussian packets are generated in memory and optionally saved to .pt.
  - Fusion consumes the in-memory packets directly.

Typical usage:

python run_zipmap_resplat_fusion_api.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --zipmap_ckpt /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_aff_inv.pt \
  --left_dir /path/to/P000/image_lcam_front \
  --right_dir /path/to/P000/image_rcam_front \
  --gt_pose_file /path/to/P000/pose_lcam_front.txt \
  --work_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_p000_20_40 \
  --start_index 20 --end_index 41 \
  --resplat_experiment tartanair_p000_ft \
  --scene_name P000 \
  --device cuda:0

Notes:
  - end_index is exclusive, Python-style. start=20,end=41 selects frames 20..40.
  - ZipMap estimates left-camera poses. Right-camera poses are synthesized from a fixed stereo rig.
  - For final ReSplat packets when encoder.num_refine > 0, the script calls forward_update(..., target=None)
    because the update uses context-view render error; target rendering is not needed for packet creation.
  - The packet still stores target metadata/image for downstream fixed/follow-camera fusion probes.
"""

from __future__ import annotations

SCRIPT_VERSION = "2026-06-05-v4-official-streaming-zipmap-ar"

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as tf

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Generic helpers
# =============================================================================


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


@contextlib.contextmanager
def temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


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


def tensor_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: tensor_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tensor_to_device(v, device) for v in obj)
    return obj


def detach_cpu_tensor(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: detach_cpu_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_cpu_tensor(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_cpu_tensor(v) for v in obj)
    return obj


def save_png_tensor(image: torch.Tensor, path: Path) -> None:
    """Save [3,H,W] image tensor in [0,1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    arr = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


# =============================================================================
# Pose and camera helpers
# =============================================================================


def tartan_from_cv_matrix_torch(device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    return T


def fixed_tartanair_stereo_rig_cv(
    baseline: float = 0.25000006,
    device: torch.device | None = None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return T_left_cv_to_right_cv for TartanAir front stereo."""
    T_tartan_from_cv = tartan_from_cv_matrix_torch(device=device, dtype=dtype)
    T_cv_from_tartan = torch.linalg.inv(T_tartan_from_cv)

    T_left_tartan_to_right_tartan = torch.eye(4, dtype=dtype, device=device)
    T_left_tartan_to_right_tartan[:3, 3] = torch.tensor(
        [0.0, float(baseline), 0.0], dtype=dtype, device=device
    )
    return T_cv_from_tartan @ T_left_tartan_to_right_tartan @ T_tartan_from_cv


def build_pixel_K(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = float(fx)
    K[1, 1] = float(fy)
    K[0, 2] = float(cx)
    K[1, 2] = float(cy)
    return K


def process_image_and_K(
    image_path: Path,
    K_pixel: torch.Tensor,
    image_shape: tuple[int, int],
    normalize_intrinsics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror DatasetTartanAir._process_image_and_K."""
    to_tensor = tf.ToTensor()
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        K = K_pixel.clone().float()
        target_h, target_w = image_shape
        scale = max(target_w / orig_w, target_h / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))
        im = im.resize((resized_w, resized_h), Image.BILINEAR)
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        left = int(round((resized_w - target_w) / 2.0))
        top = int(round((resized_h - target_h) / 2.0))
        im = im.crop((left, top, left + target_w, top + target_h))
        K[0, 2] -= left
        K[1, 2] -= top
        if normalize_intrinsics:
            K[0, 0] /= target_w
            K[1, 1] /= target_h
            K[0, 2] /= target_w
            K[1, 2] /= target_h
        return to_tensor(im), K


def read_pose_json(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("Twc", "extrinsics", "pose"):
            if key in data:
                data = data[key]
                break
    T = torch.tensor(data, dtype=torch.float32)
    if T.numel() == 16:
        T = T.reshape(4, 4)
    if T.shape != (4, 4):
        raise ValueError(f"Pose JSON must contain a 4x4 matrix, got {tuple(T.shape)} from {path}")
    return T


def read_pose_sequence_json(path: Path) -> list[torch.Tensor]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "poses" in data:
        data = data["poses"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of poses or {{'poses': [...]}} in {path}")
    poses = []
    for i, item in enumerate(data):
        T = torch.tensor(item, dtype=torch.float32)
        if T.numel() == 16:
            T = T.reshape(4, 4)
        if T.shape != (4, 4):
            raise ValueError(f"Pose #{i} in {path} must be 4x4, got {tuple(T.shape)}")
        poses.append(T)
    return poses


# =============================================================================
# ZipMap API stage
# =============================================================================


@dataclass
class SelectedStereoFrames:
    indices: list[int]
    left_paths: list[Path]
    right_paths: list[Path]
    frame_records: list[dict[str, Any]]
    stereo_pairs: list[dict[str, Any]]


@dataclass
class ZipMapResult:
    out_dir: Path
    selected: SelectedStereoFrames
    raw_T_c2w_cv: np.ndarray
    aligned_T_c2w_cv: np.ndarray
    intrinsics_zipmap: np.ndarray
    predictions: dict[str, np.ndarray]
    pose_eval_summary: dict[str, Any]


def select_stereo_frames(
    left_dir: Path,
    right_dir: Path,
    start_index: int,
    end_index: Optional[int],
    stride: int,
    num_frames: Optional[int],
    recursive: bool,
) -> SelectedStereoFrames:
    left_all = collect_images(left_dir, recursive=recursive)
    right_all = collect_images(right_dir, recursive=recursive)
    if len(left_all) != len(right_all):
        raise ValueError(f"Left/right image count mismatch: left={len(left_all)}, right={len(right_all)}")

    indices = make_sample_indices(
        len(left_all), start_index=start_index, end_index=end_index, stride=stride, max_count=num_frames
    )
    left_paths = [left_all[i] for i in indices]
    right_paths = [right_all[i] for i in indices]

    frame_records = []
    stereo_pairs = []
    for pred_i, original_i in enumerate(indices):
        frame_records.append(
            {
                "pred_index": pred_i,
                "source_path": str(left_all[original_i]),
                "source_name": left_all[original_i].name,
                "camera_id": "left",
                "sampled_pair_index": pred_i,
                "original_pair_index": original_i,
                "original_index": original_i,
                "gt_index": original_i,
            }
        )
        stereo_pairs.append(
            {
                "sampled_pair_index": pred_i,
                "original_pair_index": original_i,
                "left": str(left_all[original_i]),
                "right": str(right_all[original_i]),
            }
        )

    return SelectedStereoFrames(indices, left_paths, right_paths, frame_records, stereo_pairs)


OFFICIAL_STREAMING_MODEL_CONFIG = {
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


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_cuda(device: torch.device, fn):
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


def load_official_streaming_zipmap_ar(args: argparse.Namespace, zipmap_repo: Path, device: torch.device):
    """Load the exact ZipMap_AR model path used by demo_gradio_zipmap_streaming.py."""
    sys.path.insert(0, str(zipmap_repo))
    from zipmap.models.ZipMap_AR import ZipMap

    # Deep copy without importing copy.
    model_config = json.loads(json.dumps(OFFICIAL_STREAMING_MODEL_CONFIG))
    model_config["other_config"]["affine_invariant"] = bool(args.zipmap_affine_invariant)

    model = ZipMap(**model_config)

    ckpt_path = abs_path(args.zipmap_ckpt)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    if args.zipmap_ema and isinstance(checkpoint, dict) and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        log("[ZipMap-AR] using EMA weights")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log(f"[ZipMap-AR] missing keys: {missing}")
    if unexpected:
        log(f"[ZipMap-AR] unexpected keys: {unexpected}")

    model.eval().to(device)
    return model, model_config


def run_zipmap_api(args: argparse.Namespace, selected: SelectedStereoFrames) -> ZipMapResult:
    """
    Official ZipMap streaming-demo pose stage.

    This replaces the old offline ZipMap stage with:
        ZipMap_AR + checkpoint_online.pt + model(images)

    Important:
      - This is the official Gradio streaming-demo inference path, run headlessly.
      - It is still a single batch forward over selected left images.
      - ReSplat consumes the resulting poses in memory via ZipMapResult.
      - Pose/depth/point files are saved only for inspection and reproducibility.
    """
    zipmap_repo = abs_path(args.zipmap_repo)
    zipmap_out = abs_path(args.work_dir) / args.zipmap_out_name
    zipmap_out.mkdir(parents=True, exist_ok=True)

    zipmap_export_path = abs_path(args.zipmap_export_script) if args.zipmap_export_script else zipmap_repo / "tools" / "export_zipmap_predictions.py"
    zipmap_eval_path = abs_path(args.zipmap_eval_script) if args.zipmap_eval_script else zipmap_repo / "tools" / "evaluate_zipmap_pose.py"
    export_mod = import_module_from_path("zipmap_export_api", zipmap_export_path)
    eval_mod = import_module_from_path("zipmap_eval_api", zipmap_eval_path)

    # Make ZipMap package importable.
    sys.path.insert(0, str(zipmap_repo))
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
    from zipmap.utils.geometry import (
        unproject_depth_map_to_point_map,
        closed_form_inverse_se3,
        homogenize_points,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or args.zipmap_cpu:
        raise RuntimeError(
            f"ZipMap_AR official streaming path expects CUDA. "
            f"Requested device={args.device}, zipmap_cpu={args.zipmap_cpu}, cuda_available={torch.cuda.is_available()}"
        )

    log(f"[1/4] ZipMap-AR official streaming path: loading {len(selected.left_paths)} left images")

    with temporary_cwd(zipmap_repo):
        model, model_config = load_official_streaming_zipmap_ar(args, zipmap_repo, device)

        image_names = [str(p) for p in selected.left_paths]

        def preprocess():
            return load_and_preprocess_images(
                image_names,
                target_size=args.zipmap_target_size,
                mode=args.zipmap_preprocess_mode,
            ).to(device)

        images, preprocess_sec = timed_cuda(device, preprocess)
        log(f"      preprocess: images={tuple(images.shape)}, time={preprocess_sec:.4f}s")

        def infer():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                return model(images)

        predictions_raw, infer_sec = timed_cuda(device, infer)
        log(f"      infer: time={infer_sec:.4f}s")

        def decode_pose():
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions_raw["pose_enc"], images.shape[-2:])
            predictions_raw["extrinsic"] = extrinsic
            predictions_raw["intrinsic"] = intrinsic
            return extrinsic, intrinsic

        (extrinsic, intrinsic), postprocess_sec = timed_cuda(device, decode_pose)
        log(f"      pose decode: time={postprocess_sec:.4f}s")

    # Keep only fields required by downstream ReSplat and optional ZipMap inspection.
    pred: dict[str, np.ndarray] = {
        "extrinsic": tensor_to_np_no_batch(extrinsic).astype(np.float32),
        "intrinsic": tensor_to_np_no_batch(intrinsic).astype(np.float32),
    }

    # Optional depth/point outputs. These are not required by ReSplat packet generation,
    # but they preserve the old script's inspection behavior.
    if args.save_zipmap_depth and "depth" in predictions_raw and torch.is_tensor(predictions_raw["depth"]):
        pred["depth"] = tensor_to_np_no_batch(predictions_raw["depth"]).astype(np.float32)
    if args.save_zipmap_points:
        for key in ("depth", "depth_conf", "local_points", "local_points_conf"):
            if key in predictions_raw and torch.is_tensor(predictions_raw[key]):
                pred[key] = tensor_to_np_no_batch(predictions_raw[key]).astype(np.float32)

    T_w2c_cv = export_mod.to_homogeneous_4x4(pred["extrinsic"].astype(np.float32))
    if args.zipmap_align_first_view:
        T_w2c_cv = export_mod.align_w2c_to_first_camera(T_w2c_cv)
        pred["extrinsic"] = T_w2c_cv[:, :3, :4].astype(np.float32)
    T_c2w_cv = export_mod.invert_se3_np(T_w2c_cv).astype(np.float32)
    intrinsics = pred["intrinsic"].astype(np.float32)

    if args.save_zipmap_points and "depth" in pred:
        pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
            pred["depth"], pred["extrinsic"], intrinsics, if_c2w=False
        ).astype(np.float32)
    if args.save_zipmap_points and "local_points" in pred:
        cam_to_world = closed_form_inverse_se3(pred["extrinsic"]).astype(np.float32)
        local_points = pred["local_points"].astype(np.float32)
        world_points = np.einsum("sij,shwj->shwi", cam_to_world, homogenize_points(local_points))
        pred["world_points"] = world_points[..., :3].astype(np.float32)
        if "local_points_conf" in pred:
            pred["world_points_conf"] = pred["local_points_conf"].astype(np.float32)
        elif "depth_conf" in pred:
            pred["world_points_conf"] = pred["depth_conf"].astype(np.float32)

    pred["T_w2c_opencv"] = T_w2c_cv.astype(np.float32)
    pred["T_c2w_opencv"] = T_c2w_cv.astype(np.float32)
    pred["intrinsic"] = intrinsics

    np.savez_compressed(zipmap_out / "predictions.npz", **pred)
    save_json(zipmap_out / "frame_records.json", selected.frame_records)
    save_json(zipmap_out / "stereo_pairs.json", selected.stereo_pairs)
    (zipmap_out / "selected_gt_indices.txt").write_text(
        "".join(f"{r['pred_index']} {r['gt_index']} {r['camera_id']} {r['source_name']}\n" for r in selected.frame_records),
        encoding="utf-8",
    )

    if args.save_zipmap_input_images:
        export_mod.save_tensor_images_as_png(images.detach().cpu(), zipmap_out / "images_zipmap_input", [p.name for p in selected.left_paths])

    timestamps = [p.stem for p in selected.left_paths]
    export_mod.write_tum_pose_file(zipmap_out / "poses_c2w_opencv_tum.txt", T_c2w_cv, timestamps=timestamps)
    export_mod.write_matrix_file(zipmap_out / "poses_w2c_opencv.txt", T_w2c_cv[:, :3, :4])
    export_mod.write_matrix_file(zipmap_out / "poses_c2w_opencv_matrix.txt", T_c2w_cv)
    T_resplat_loader_pose = export_mod.compute_resplat_loader_pose_from_cv(T_c2w_cv).astype(np.float32)
    export_mod.write_pose7_file(zipmap_out / "poses_resplat_tartanair_loader.txt", T_resplat_loader_pose)
    export_mod.write_tum_pose_file(zipmap_out / "poses_resplat_tartanair_loader_tum.txt", T_resplat_loader_pose, timestamps=timestamps)
    export_mod.save_intrinsics(zipmap_out / "intrinsics.txt", intrinsics)

    if args.save_zipmap_depth and "depth" in pred:
        export_mod.save_depth_frames(pred["depth"], zipmap_out / "depth")
    if args.save_zipmap_points:
        points_dir = zipmap_out / "points"
        points_dir.mkdir(parents=True, exist_ok=True)
        for key in ("world_points_from_depth", "world_points", "world_points_conf"):
            if key in pred:
                np.save(points_dir / f"{key}.npy", pred[key].astype(np.float32))

    timing_summary = {
        "mode": "official_streaming_demo_batch_ZipMap_AR",
        "num_frames": len(selected.left_paths),
        "preprocess_sec": float(preprocess_sec),
        "infer_sec": float(infer_sec),
        "pose_decode_sec": float(postprocess_sec),
        "total_zipmap_sec": float(preprocess_sec + infer_sec + postprocess_sec),
        "infer_sec_per_frame": float(infer_sec / max(1, len(selected.left_paths))),
        "total_zipmap_sec_per_frame": float((preprocess_sec + infer_sec + postprocess_sec) / max(1, len(selected.left_paths))),
    }
    save_json(zipmap_out / "timing_summary.json", timing_summary)

    meta = {
        "left_dir": str(abs_path(args.left_dir)),
        "right_dir": str(abs_path(args.right_dir)),
        "selected_original_indices": [int(i) for i in selected.indices],
        "selected_local_to_original": [
            {"local_index": int(local_i), "original_index": int(original_i)}
            for local_i, original_i in enumerate(selected.indices)
        ],
        "num_images": len(selected.left_paths),
        "preprocessed_shape_SCHW": list(images.shape),
        "target_size": args.zipmap_target_size,
        "preprocess_mode": args.zipmap_preprocess_mode,
        "align_first_view": args.zipmap_align_first_view,
        "affine_invariant": args.zipmap_affine_invariant,
        "ckpt_path": str(abs_path(args.zipmap_ckpt)),
        "zipmap_model_type": "ZipMap_AR",
        "zipmap_model_config": model_config,
        "timing": timing_summary,
        "array_shapes": {k: list(v.shape) for k, v in pred.items() if isinstance(v, np.ndarray)},
        "note": (
            "ZipMap stage has been replaced by the official streaming Gradio demo core path: "
            "ZipMap_AR + checkpoint_online.pt + model(images). "
            "ReSplat consumes raw_T_c2w_cv/aligned_T_c2w_cv directly in memory."
        ),
    }
    save_json(zipmap_out / "meta.json", meta)

    # Optional GT-based alignment for metric scale.
    aligned_T = T_c2w_cv.copy()
    pose_eval_summary: dict[str, Any] = {"alignment": "none", "gt_used": False}
    if args.gt_pose_file is not None and args.pose_alignment != "none":
        gt_path = abs_path(args.gt_pose_file)
        T_gt_all = eval_mod.load_gt_trajectory(
            gt_path,
            args.gt_convention,
            args.gt_quat_order,
            args.gt_matrix_convention,
        )
        gt_indices = [int(r["gt_index"]) for r in selected.frame_records]
        T_gt = T_gt_all[gt_indices]
        pred_centers = T_c2w_cv[:, :3, 3].astype(np.float64)
        gt_centers = T_gt[:, :3, 3].astype(np.float64)
        with_scale = args.pose_alignment == "sim3"
        scale, R_align, t_align = eval_mod.umeyama_alignment(pred_centers, gt_centers, with_scale=with_scale)
        aligned_T = eval_mod.apply_similarity_to_poses(T_c2w_cv.astype(np.float64), scale, R_align, t_align).astype(np.float32)
        ate = eval_mod.compute_ate(aligned_T.astype(np.float64), T_gt.astype(np.float64))
        rpe_rows, rpe = eval_mod.compute_rpe(aligned_T.astype(np.float64), T_gt.astype(np.float64), delta=args.rpe_delta)
        pose_eval_dir = zipmap_out / "pose_eval"
        pose_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_mod.write_pose_tum(pose_eval_dir / "trajectory_pred_aligned_c2w_opencv.txt", aligned_T, names=[p.name for p in selected.left_paths])
        eval_mod.write_pose_tum(pose_eval_dir / "trajectory_gt_matched_c2w_opencv.txt", T_gt, names=[p.name for p in selected.left_paths])
        save_json(
            pose_eval_dir / "summary.json",
            {
                "gt_pose_file": str(gt_path),
                "gt_indices": gt_indices,
                "alignment": args.pose_alignment,
                "scale": float(scale),
                "R": R_align.tolist(),
                "t": t_align.tolist(),
                "ate": ate,
                "rpe": rpe,
                "timing": timing_summary,
            },
        )
        # lightweight CSVs for later inspection
        with (pose_eval_dir / "per_frame_errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["local_index", "gt_index", "ate_translation_error"])
            writer.writeheader()
            err = np.linalg.norm(aligned_T[:, :3, 3] - T_gt[:, :3, 3], axis=1)
            for i, (gt_i, e) in enumerate(zip(gt_indices, err)):
                writer.writerow({"local_index": i, "gt_index": gt_i, "ate_translation_error": float(e)})
        with (pose_eval_dir / "rpe_errors.csv").open("w", newline="", encoding="utf-8") as f:
            if rpe_rows:
                writer = csv.DictWriter(f, fieldnames=list(rpe_rows[0].keys()))
                writer.writeheader()
                writer.writerows(rpe_rows)
        pose_eval_summary = {
            "alignment": args.pose_alignment,
            "gt_used": True,
            "scale": float(scale),
            "ate": ate,
            "rpe": rpe,
            "timing": timing_summary,
        }
        log(
            f"      pose alignment={args.pose_alignment}, scale={scale:.6f}, "
            f"ATE_RMSE={ate['rmse']:.6f}"
        )

    # Release ZipMap model and big tensors before loading/running ReSplat.
    del model, predictions_raw, extrinsic, intrinsic, images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ZipMapResult(
        out_dir=zipmap_out,
        selected=selected,
        raw_T_c2w_cv=T_c2w_cv,
        aligned_T_c2w_cv=aligned_T,
        intrinsics_zipmap=intrinsics,
        predictions=pred,
        pose_eval_summary=pose_eval_summary,
    )


# =============================================================================
# ReSplat API stage
# =============================================================================


@dataclass
class ResplatRuntime:
    model: Any
    cfg: Any
    device: torch.device
    image_shape: tuple[int, int]
    near: float
    far: float
    normalize_intrinsics: bool
    K_pixel: torch.Tensor


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    try:
        return obj.get(key, default)
    except Exception:
        try:
            return obj[key]
        except Exception:
            return getattr(obj, key, default)


def resolve_repo_relative(path_like: Any, repo: Path) -> Optional[str]:
    if path_like is None:
        return None
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return str(p)
    return str((repo / p).resolve())


def compose_resplat_cfg(args: argparse.Namespace):
    resplat_repo = abs_path(args.resplat_repo)
    sys.path.insert(0, str(resplat_repo))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        f"+experiment={args.resplat_experiment}",
        "mode=test",
        "wandb.mode=disabled",
        f"output_dir={str(abs_path(args.work_dir) / args.resplat_out_name)}",
    ]
    overrides.extend(args.resplat_override or [])
    if args.resplat_checkpoint is not None:
        overrides.append(f"checkpointing.pretrained_model={str(abs_path(args.resplat_checkpoint))}")

    with initialize_config_dir(config_dir=str(resplat_repo / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    return cfg_dict


def load_resplat_runtime(args: argparse.Namespace) -> ResplatRuntime:
    resplat_repo = abs_path(args.resplat_repo)
    sys.path.insert(0, str(resplat_repo))

    with temporary_cwd(resplat_repo):
        cfg_dict = compose_resplat_cfg(args)
        from src.config import load_typed_root_config
        from src.global_cfg import set_cfg
        from src.loss import get_losses
        from src.misc.step_tracker import StepTracker
        from src.model.decoder import get_decoder
        from src.model.encoder import get_encoder
        from src.model.model_wrapper import ModelWrapper

        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
        model = ModelWrapper(
            cfg.optimizer,
            cfg.test,
            cfg.train,
            encoder,
            encoder_visualizer,
            get_decoder(cfg.model.decoder, cfg.dataset),
            get_losses(cfg.loss),
            StepTracker(),
            eval_data_cfg=None,
        )

        strict_load = not cfg.checkpointing.no_strict_load
        pretrained_model_path = resolve_repo_relative(cfg.checkpointing.pretrained_model, resplat_repo)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=strict_load)
            log(f"[2/4] ReSplat: loaded pretrained model {pretrained_model_path}")
        else:
            log("[2/4] ReSplat: no checkpointing.pretrained_model configured")

        # Optional depth and update-module loading, matching src/main.py logic.
        pretrained_depth_path = resolve_repo_relative(cfg.checkpointing.pretrained_depth, resplat_repo)
        if pretrained_depth_path is not None:
            depth_ckpt = torch.load(pretrained_depth_path, map_location="cpu")
            if isinstance(depth_ckpt, dict) and "state_dict" in depth_ckpt:
                depth_ckpt = depth_ckpt["state_dict"]
            if isinstance(depth_ckpt, dict) and "model" in depth_ckpt:
                depth_ckpt = depth_ckpt["model"]
            model.encoder.depth_predictor.load_state_dict(depth_ckpt, strict=strict_load)
            log(f"      loaded pretrained depth {pretrained_depth_path}")

        update_path = resolve_repo_relative(cfg.checkpointing.resume_update_module, resplat_repo)
        if update_path is not None:
            update_ckpt = torch.load(update_path, map_location="cpu")
            if isinstance(update_ckpt, dict) and "state_dict" in update_ckpt:
                update_ckpt = update_ckpt["state_dict"]
            if isinstance(update_ckpt, dict) and "model" in update_ckpt:
                update_ckpt = update_ckpt["model"]
            model_state = model.state_dict()
            filtered = {
                k: v for k, v in update_ckpt.items()
                if "encoder.update" in k and k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
            }
            model.load_state_dict(filtered, strict=False)
            log(f"      loaded update-module weights {update_path} ({len(filtered)} tensors)")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"ReSplat expects CUDA. Requested device={args.device}, cuda_available={torch.cuda.is_available()}")
    model.eval().to(device)

    dataset_cfg = cfg.dataset
    image_shape = tuple(int(x) for x in dataset_cfg.image_shape)
    near = float(dataset_cfg.near)
    far = float(dataset_cfg.far)
    normalize_intrinsics = bool(dataset_cfg.normalize_intrinsics)

    # Intrinsics are read from the first sequence when available; otherwise dataset-level fx/fy/cx/cy.
    seq0 = dataset_cfg.sequences[0] if getattr(dataset_cfg, "sequences", None) else dataset_cfg
    fx = float(getattr(seq0, "fx", getattr(dataset_cfg, "fx", 0.0)))
    fy = float(getattr(seq0, "fy", getattr(dataset_cfg, "fy", 0.0)))
    cx = float(getattr(seq0, "cx", getattr(dataset_cfg, "cx", 0.0)))
    cy = float(getattr(seq0, "cy", getattr(dataset_cfg, "cy", 0.0)))
    if args.fx is not None:
        fx = args.fx
    if args.fy is not None:
        fy = args.fy
    if args.cx is not None:
        cx = args.cx
    if args.cy is not None:
        cy = args.cy
    if not all(v > 0 for v in (fx, fy, cx, cy)):
        raise ValueError(f"Invalid ReSplat intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    log(f"      image_shape={image_shape}, near={near}, far={far}, K=({fx},{fy},{cx},{cy})")
    return ResplatRuntime(
        model=model,
        cfg=cfg,
        device=device,
        image_shape=image_shape,
        near=near,
        far=far,
        normalize_intrinsics=normalize_intrinsics,
        K_pixel=build_pixel_K(fx, fy, cx, cy),
    )


def make_resplat_batch_for_frame(
    runtime: ResplatRuntime,
    scene_key: str,
    frame_index: int,
    left_path: Path,
    right_path: Path,
    T_left_c2w_cv: np.ndarray,
    target_camera: str,
    target_offset_frame_index: Optional[int] = None,
    target_left_path: Optional[Path] = None,
    target_right_path: Optional[Path] = None,
    target_T_left_c2w_cv: Optional[np.ndarray] = None,
    stereo_baseline: float = 0.25000006,
) -> dict[str, Any]:
    """Build a batched ReSplat example without using DatasetTartanAir/view_sampler."""
    device_none = None
    T_left = torch.tensor(T_left_c2w_cv, dtype=torch.float32)
    T_lr = fixed_tartanair_stereo_rig_cv(baseline=stereo_baseline, device=device_none, dtype=torch.float32)
    T_right = T_left @ T_lr

    img_l, K_l = process_image_and_K(left_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)
    img_r, K_r = process_image_and_K(right_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)

    context = {
        "extrinsics": torch.stack([T_left, T_right], dim=0).unsqueeze(0),
        "intrinsics": torch.stack([K_l, K_r], dim=0).unsqueeze(0),
        "image": torch.stack([img_l, img_r], dim=0).unsqueeze(0),
        "near": torch.full((1, 2), runtime.near, dtype=torch.float32),
        "far": torch.full((1, 2), runtime.far, dtype=torch.float32),
        "index": torch.tensor([[frame_index, frame_index]], dtype=torch.long),
        "camera_id": torch.tensor([[0, 1]], dtype=torch.long),
    }

    # Target metadata is stored in packets. It is not needed for refinement update if target=None is used.
    if target_offset_frame_index is None:
        target_offset_frame_index = frame_index
    if target_left_path is None:
        target_left_path = left_path
    if target_right_path is None:
        target_right_path = right_path
    if target_T_left_c2w_cv is None:
        target_T_left = T_left
    else:
        target_T_left = torch.tensor(target_T_left_c2w_cv, dtype=torch.float32)
    target_T_right = target_T_left @ T_lr

    target_images: list[torch.Tensor] = []
    target_intrinsics: list[torch.Tensor] = []
    target_extrinsics: list[torch.Tensor] = []
    target_indices: list[int] = []
    target_camera_ids: list[int] = []

    if target_camera in {"left", "both"}:
        img_tl, K_tl = process_image_and_K(target_left_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)
        target_images.append(img_tl)
        target_intrinsics.append(K_tl)
        target_extrinsics.append(target_T_left)
        target_indices.append(target_offset_frame_index)
        target_camera_ids.append(0)
    if target_camera in {"right", "both"}:
        img_tr, K_tr = process_image_and_K(target_right_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)
        target_images.append(img_tr)
        target_intrinsics.append(K_tr)
        target_extrinsics.append(target_T_right)
        target_indices.append(target_offset_frame_index)
        target_camera_ids.append(1)
    if not target_images:
        raise ValueError("target_camera must be left, right, or both")

    target = {
        "extrinsics": torch.stack(target_extrinsics, dim=0).unsqueeze(0),
        "intrinsics": torch.stack(target_intrinsics, dim=0).unsqueeze(0),
        "image": torch.stack(target_images, dim=0).unsqueeze(0),
        "near": torch.full((1, len(target_images)), runtime.near, dtype=torch.float32),
        "far": torch.full((1, len(target_images)), runtime.far, dtype=torch.float32),
        "index": torch.tensor([target_indices], dtype=torch.long),
        "camera_id": torch.tensor([target_camera_ids], dtype=torch.long),
    }

    return {
        "context": context,
        "target": target,
        "scene": [scene_key],
        "scene_name": [scene_key.rsplit("_", 1)[0] if scene_key.rsplit("_", 1)[-1].isdigit() else scene_key],
    }


def make_packet_from_gaussians(
    gaussians: Any,
    batch: dict[str, Any],
    decoder: Any,
    stage: str,
) -> dict[str, Any]:
    scene_value = batch["scene"]
    scene = scene_value[0] if isinstance(scene_value, (list, tuple)) else scene_value
    background_color = getattr(decoder, "background_color", None)
    if background_color is None:
        background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    elif torch.is_tensor(background_color):
        background_color = background_color.detach().cpu()
    else:
        background_color = torch.tensor(background_color, dtype=torch.float32)

    packet = {
        "scene": str(scene),
        "packet_stage": stage,
        "context_index": batch["context"]["index"][0].detach().cpu(),
        "target_index": batch["target"]["index"][0].detach().cpu(),
        "target_camera_id": batch["target"]["camera_id"][0].detach().cpu(),
        "context_extrinsics": batch["context"]["extrinsics"][0].detach().cpu(),
        "context_intrinsics": batch["context"]["intrinsics"][0].detach().cpu(),
        "context_near": batch["context"]["near"][0].detach().cpu(),
        "context_far": batch["context"]["far"][0].detach().cpu(),
        "target_extrinsics": batch["target"]["extrinsics"][0].detach().cpu(),
        "target_intrinsics": batch["target"]["intrinsics"][0].detach().cpu(),
        "target_near": batch["target"]["near"][0].detach().cpu(),
        "target_far": batch["target"]["far"][0].detach().cpu(),
        "target_image": batch["target"]["image"][0].detach().cpu(),
        "image_shape": tuple(int(x) for x in batch["target"]["image"].shape[-2:]),
        "background_color": background_color,
        "means": gaussians.means[0].detach().cpu(),
        "covariances": gaussians.covariances[0].detach().cpu() if gaussians.covariances is not None else None,
        "harmonics": gaussians.harmonics[0].detach().cpu(),
        "opacities": gaussians.opacities[0].detach().cpu(),
        "scales": gaussians.scales[0].detach().cpu() if gaussians.scales is not None else None,
        "rotations": gaussians.rotations[0].detach().cpu() if gaussians.rotations is not None else None,
        "rotations_unnorm": gaussians.rotations_unnorm[0].detach().cpu() if gaussians.rotations_unnorm is not None else None,
    }
    return packet


@torch.no_grad()
def infer_resplat_packets(
    runtime: ResplatRuntime,
    zipmap: ZipMapResult,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    model = runtime.model
    device = runtime.device
    stage_request = args.resplat_packet_stage
    selected = zipmap.selected
    T_left_all = zipmap.aligned_T_c2w_cv if args.resplat_pose_source == "aligned" else zipmap.raw_T_c2w_cv

    packet_groups: dict[str, list[dict[str, Any]]] = {"init": [], "final": []}
    packet_out_root = abs_path(args.work_dir) / args.packet_out_name
    if args.save_packets:
        if stage_request == "both":
            (packet_out_root / "init").mkdir(parents=True, exist_ok=True)
            (packet_out_root / "final").mkdir(parents=True, exist_ok=True)
        else:
            (packet_out_root / stage_request).mkdir(parents=True, exist_ok=True)

    log(f"[3/4] ReSplat: generating packets from {len(selected.indices)} stereo pairs")
    for local_i, original_i in enumerate(selected.indices):
        scene_key = f"{args.scene_name}_{local_i:04d}"
        target_local = local_i + args.resplat_target_offset
        if target_local < 0 or target_local >= len(selected.indices):
            if args.drop_invalid_target_offset:
                continue
            target_local = max(0, min(target_local, len(selected.indices) - 1))
        target_original_i = selected.indices[target_local]

        batch_cpu = make_resplat_batch_for_frame(
            runtime=runtime,
            scene_key=scene_key,
            frame_index=original_i,
            left_path=selected.left_paths[local_i],
            right_path=selected.right_paths[local_i],
            T_left_c2w_cv=T_left_all[local_i],
            target_camera=args.resplat_target_camera,
            target_offset_frame_index=target_original_i,
            target_left_path=selected.left_paths[target_local],
            target_right_path=selected.right_paths[target_local],
            target_T_left_c2w_cv=T_left_all[target_local],
            stereo_baseline=args.stereo_baseline,
        )
        batch = tensor_to_device(batch_cpu, device)
        batch = model.data_shim(batch)

        context = batch["context"]
        with torch.no_grad():
            enc_out = model.encoder(
                context,
                0,
                deterministic=False,
                visualization_dump=None,
            )
            condition_features = None
            if isinstance(enc_out, dict):
                condition_features = enc_out.get("condition_features", None)
                gaussians_init = enc_out["gaussians"]
            else:
                gaussians_init = enc_out

            if stage_request in {"init", "both"}:
                pkt = make_packet_from_gaussians(gaussians_init, batch, model.decoder, stage="init")
                packet_groups["init"].append(pkt)
                if args.save_packets:
                    torch.save(pkt, packet_out_root / ("init" if stage_request == "both" else "init") / f"{scene_key}.pt")

            # Only run ReSplat refinement when final packets are requested.
            # If stage_request == "init", use the raw encoder output directly and skip
            # forward_update completely. This avoids wasting time and avoids saving final packets.
            if stage_request in {"final", "both"}:
                gaussians_final = gaussians_init
                if getattr(model.encoder.cfg, "num_refine", 0) > 0:
                    if condition_features is None:
                        raise RuntimeError(
                            "encoder.num_refine > 0 but encoder output did not contain condition_features. "
                            "Check model.encoder.return_depth / ReSplat config."
                        )
                    # target rendering is not needed to update Gaussians; update uses context-view render error.
                    refine_out = model.encoder.forward_update(
                        batch["context"],
                        None,
                        condition_features,
                        gaussians_init,
                        model.decoder,
                        batch.get("context_remain", None),
                    )
                    if len(refine_out["gaussian"]) == 0:
                        raise RuntimeError("forward_update returned no gaussian outputs.")
                    gaussians_final = refine_out["gaussian"][-1]

                pkt = make_packet_from_gaussians(gaussians_final, batch, model.decoder, stage="final")
                packet_groups["final"].append(pkt)
                if args.save_packets:
                    subdir = "final" if stage_request == "both" else "final"
                    torch.save(pkt, packet_out_root / subdir / f"{scene_key}.pt")

        if (local_i + 1) % max(1, args.log_every) == 0 or local_i + 1 == len(selected.indices):
            log(f"      generated {local_i + 1}/{len(selected.indices)} packet(s)")

    manifest = {
        "packet_stage_requested": stage_request,
        "packet_stage_for_fusion": args.packet_stage_for_fusion,
        "num_init_packets": len(packet_groups["init"]),
        "num_final_packets": len(packet_groups["final"]),
        "selected_original_indices": [int(i) for i in selected.indices],
        "selected_local_to_original": [
            {"local_index": int(local_i), "original_index": int(original_i)}
            for local_i, original_i in enumerate(selected.indices)
        ],
        "target_camera": args.resplat_target_camera,
        "target_offset": args.resplat_target_offset,
    }
    save_json(packet_out_root / "manifest.json", manifest)
    return packet_groups


# =============================================================================
# Fusion API stage
# =============================================================================


@dataclass
class ProbeView:
    label: str
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    near: torch.Tensor
    far: torch.Tensor
    image_shape: tuple[int, int]
    gt_image: Optional[torch.Tensor]
    meta: dict[str, Any]


def packet_base_scene(scene: str) -> str:
    parts = scene.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scene


def packet_sort_key(packet: dict[str, Any]) -> int:
    return int(packet["context_index"].reshape(-1)[0].item())


def concat_packets_to_gaussians(packet_list: list[dict[str, Any]], device: torch.device):
    """Concatenate packet tensors into one batched Gaussians object.

    Important memory detail: do not call the cat helper twice for optional fields.
    With tens of packets, covariances / harmonics / scales can be multi-GB tensors;
    duplicate temporary concatenations can be enough to trigger OOM before rendering.
    """
    from src.model.types import Gaussians

    def cat_optional(key: str):
        vals = [p.get(key, None) for p in packet_list]
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            raise ValueError(f"Mixed None/non-None field for {key}")
        return torch.cat(vals, dim=0).to(device, non_blocking=True).unsqueeze(0).contiguous()

    means = cat_optional("means")
    covariances = cat_optional("covariances")
    harmonics = cat_optional("harmonics")
    opacities = cat_optional("opacities")
    scales = cat_optional("scales")
    rotations = cat_optional("rotations")
    rotations_unnorm = cat_optional("rotations_unnorm")

    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        rotations_unnorm=rotations_unnorm,
    )


def probe_from_packet_target(packet: dict[str, Any], target_view_i: int, label_prefix: str) -> ProbeView:
    idx = int(packet["target_index"][target_view_i].item())
    cam_id = int(packet["target_camera_id"][target_view_i].item())
    cam = "left" if cam_id == 0 else "right"
    return ProbeView(
        label=f"{label_prefix}_{idx:06d}_{cam}",
        extrinsics=packet["target_extrinsics"][target_view_i].clone(),
        intrinsics=packet["target_intrinsics"][target_view_i].clone(),
        near=packet["target_near"][target_view_i].reshape(1).clone(),
        far=packet["target_far"][target_view_i].reshape(1).clone(),
        image_shape=tuple(int(x) for x in packet["image_shape"]),
        gt_image=packet["target_image"][target_view_i].clone(),
        meta={
            "source": "packet_target",
            "packet_scene": packet["scene"],
            "target_index": idx,
            "camera_id": cam_id,
        },
    )


def build_fixed_probe_from_packets(args: argparse.Namespace, packets: list[dict[str, Any]]) -> ProbeView:
    if args.fixed_packet_local_index is not None:
        local = int(args.fixed_packet_local_index)
        if local < 0 or local >= len(packets):
            raise IndexError(f"fixed_packet_local_index={local} out of range 0..{len(packets)-1}")
        # choose first matching target camera if possible
        packet = packets[local]
        cam_ids = packet["target_camera_id"].tolist()
        try:
            view_i = cam_ids.index(int(args.fixed_target_camera_id))
        except ValueError:
            view_i = 0
        return probe_from_packet_target(packet, view_i, label_prefix=f"fixed_packet_{local:04d}")

    if args.fixed_target_index is None:
        raise ValueError("packet_fixed_target requires --fixed_target_index or --fixed_packet_local_index")

    for packet in packets:
        for i in range(packet["target_index"].numel()):
            if int(packet["target_index"][i].item()) == int(args.fixed_target_index) and int(packet["target_camera_id"][i].item()) == int(args.fixed_target_camera_id):
                return probe_from_packet_target(packet, i, label_prefix="fixed_target")
    raise ValueError(
        f"No packet target found for target_index={args.fixed_target_index}, camera_id={args.fixed_target_camera_id}"
    )


def resolve_probe_camera_params(
    runtime: ResplatRuntime,
    args: argparse.Namespace,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve custom-probe image shape, normalized intrinsics, near and far.

    The ReSplat gsplat decoder expects normalized intrinsics because it multiplies
    K[:,0] by image width and K[:,1] by image height before rasterization.
    """
    if args.probe_image_shape is None:
        image_shape = runtime.image_shape
    else:
        image_shape = (int(args.probe_image_shape[0]), int(args.probe_image_shape[1]))
    H, W = image_shape

    if args.probe_intrinsics_norm is not None and args.probe_intrinsics_pixel is not None:
        raise ValueError("Use either --probe_intrinsics_norm or --probe_intrinsics_pixel, not both.")

    if args.probe_intrinsics_norm is not None:
        fx, fy, cx, cy = [float(x) for x in args.probe_intrinsics_norm]
        K_norm = build_pixel_K(fx, fy, cx, cy)
    elif args.probe_intrinsics_pixel is not None:
        fx, fy, cx, cy = [float(x) for x in args.probe_intrinsics_pixel]
        K_norm = build_pixel_K(fx / W, fy / H, cx / W, cy / H)
    else:
        K_norm = runtime.K_pixel.clone().float()
        K_norm[0, 0] /= W
        K_norm[1, 1] /= H
        K_norm[0, 2] /= W
        K_norm[1, 2] /= H

    near = torch.tensor([runtime.near if args.probe_near is None else float(args.probe_near)], dtype=torch.float32)
    far = torch.tensor([runtime.far if args.probe_far is None else float(args.probe_far)], dtype=torch.float32)
    return image_shape, K_norm.float(), near, far


def load_probe_gt_image(
    gt_image_path: Optional[Path],
    image_shape: tuple[int, int],
) -> Optional[torch.Tensor]:
    if gt_image_path is None:
        return None
    to_tensor = tf.ToTensor()
    with Image.open(gt_image_path) as im:
        im = im.convert("RGB")
        H, W = image_shape
        im = im.resize((W, H), Image.BILINEAR)
        return to_tensor(im)


def build_custom_probe(
    runtime: ResplatRuntime,
    args: argparse.Namespace,
    pose: torch.Tensor,
    label: str,
    gt_image_path: Optional[Path] = None,
) -> ProbeView:
    image_shape, K_norm, near, far = resolve_probe_camera_params(runtime, args)
    gt = load_probe_gt_image(gt_image_path, image_shape)
    return ProbeView(
        label=label,
        extrinsics=pose.float().clone(),
        intrinsics=K_norm.float().clone(),
        near=near.clone(),
        far=far.clone(),
        image_shape=image_shape,
        gt_image=gt,
        meta={
            "source": "custom_pose",
            "gt_image_path": None if gt_image_path is None else str(gt_image_path),
            "image_shape": list(image_shape),
            "intrinsics_norm": K_norm.tolist(),
            "near": float(near.item()),
            "far": float(far.item()),
        },
    )


def probes_for_prefix(
    args: argparse.Namespace,
    runtime: ResplatRuntime,
    packets: list[dict[str, Any]],
    prefix: list[dict[str, Any]],
    fixed_probe: Optional[ProbeView],
    custom_probe_sequence: Optional[list[ProbeView]],
) -> list[ProbeView]:
    mode = args.fusion_probe_mode
    if mode == "prefix_last_only":
        last = prefix[-1]
        return [probe_from_packet_target(last, i, label_prefix="prefix_last") for i in range(last["target_image"].shape[0])]
    if mode == "packet_fixed_target":
        assert fixed_probe is not None
        return [fixed_probe]
    if mode == "custom_pose":
        assert fixed_probe is not None
        return [fixed_probe]
    if mode == "custom_pose_sequence":
        assert custom_probe_sequence is not None
        return custom_probe_sequence
    raise ValueError(f"Unsupported fusion_probe_mode={mode}")


def metric_mean(x: torch.Tensor) -> float:
    return float(x.detach().cpu().mean().item())


def plot_curve(x: list[int], y: list[float], title: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of fused packets")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def build_packet_trajectory_probes(packets: list[dict[str, Any]]) -> list[ProbeView]:
    """Use the selected packets' own target cameras as a follow-camera trajectory."""
    probes: list[ProbeView] = []
    for packet_i, packet in enumerate(packets):
        n_views = int(packet["target_image"].shape[0])
        for view_i in range(n_views):
            probe = probe_from_packet_target(packet, view_i, label_prefix=f"traj_{packet_i:04d}")
            probe.meta["trajectory_packet_i"] = packet_i
            probe.meta["trajectory_view_i"] = view_i
            probes.append(probe)
    return probes


@torch.no_grad()
def render_fused_gaussians_to_probes(
    runtime: ResplatRuntime,
    fused,
    probes: list[ProbeView],
    images_dir: Path,
    gt_dir: Path,
    filename_prefix: str,
    should_compute_lpips: bool,
    render_chunk_size: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], list[str]]:
    """Render one fused Gaussian map to a list of probe cameras.

    Trajectory rendering can be very memory-heavy because gsplat projects every
    Gaussian to every camera in the provided camera batch. For example, rendering
    16M Gaussians to 80 cameras in one decoder call is effectively an 80-view
    projection batch. Therefore this function renders probe cameras in small
    chunks; the default chunk size is 1 for OOM safety.
    """
    device = runtime.device
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    render_chunk_size = max(1, int(render_chunk_size))
    image_shapes = {p.image_shape for p in probes}
    if len(image_shapes) != 1:
        raise ValueError(f"All probes must share image_shape, got {sorted(image_shapes)}")
    H, W = next(iter(image_shapes))

    rendered_names: list[str] = []
    gt_names: list[str] = []
    per_view: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for chunk_start in range(0, len(probes), render_chunk_size):
        chunk = probes[chunk_start : chunk_start + render_chunk_size]
        extr = torch.stack([p.extrinsics for p in chunk], dim=0).unsqueeze(0).to(device)
        intr = torch.stack([p.intrinsics for p in chunk], dim=0).unsqueeze(0).to(device)
        near = torch.stack([p.near.reshape(()) for p in chunk], dim=0).unsqueeze(0).to(device)
        far = torch.stack([p.far.reshape(()) for p in chunk], dim=0).unsqueeze(0).to(device)

        output = runtime.model.decoder.forward(fused, extr, intr, near, far, (H, W), depth_mode=None)
        rendered = output.color[0].detach().cpu()

        chunk_gt_local_indices = [i for i, p in enumerate(chunk) if p.gt_image is not None]
        chunk_metrics: dict[str, Any] = {}
        if chunk_gt_local_indices:
            pred = rendered[chunk_gt_local_indices].to(device)
            gt = torch.stack([chunk[i].gt_image for i in chunk_gt_local_indices], dim=0).to(device)
            chunk_metrics["psnr"] = compute_psnr(gt, pred)
            chunk_metrics["ssim"] = compute_ssim(gt, pred)
            if should_compute_lpips:
                chunk_metrics["lpips"] = compute_lpips(gt, pred)

        for local_i, (probe, img) in enumerate(zip(chunk, rendered)):
            global_i = chunk_start + local_i
            # filename = f"{filename_prefix}_{global_i:04d}_{probe.label}.png"
            filename = f"{global_i + 1:03d}.png"
            save_png_tensor(img, images_dir / filename)
            rendered_names.append(filename)
            row = {
                "probe_label": probe.label,
                "gt_available": probe.gt_image is not None,
                "meta": probe.meta,
                "rendered_image": filename,
            }
            if probe.gt_image is not None:
                save_png_tensor(probe.gt_image, gt_dir / filename)
                gt_names.append(filename)
                metric_i = chunk_gt_local_indices.index(local_i)
                psnr = float(chunk_metrics["psnr"][metric_i].detach().cpu().item())
                ssim = float(chunk_metrics["ssim"][metric_i].detach().cpu().item())
                row["psnr"] = psnr
                row["ssim"] = ssim
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                if should_compute_lpips:
                    lpips = float(chunk_metrics["lpips"][metric_i].detach().cpu().item())
                    row["lpips"] = lpips
                    lpips_values.append(lpips)
            per_view.append(row)

        # Release temporary projection/rasterization buffers before the next view chunk.
        del output, rendered, extr, intr, near, far
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    metrics_mean: dict[str, Any] = {}
    if psnr_values:
        metrics_mean["psnr"] = float(sum(psnr_values) / len(psnr_values))
        metrics_mean["ssim"] = float(sum(ssim_values) / len(ssim_values))
        if should_compute_lpips and lpips_values:
            metrics_mean["lpips"] = float(sum(lpips_values) / len(lpips_values))
    return per_view, metrics_mean, rendered_names, gt_names


@torch.no_grad()
def run_fusion_api(
    runtime: ResplatRuntime,
    packet_groups: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    if args.skip_fusion:
        return None

    device = runtime.device
    stage = args.packet_stage_for_fusion
    packets = packet_groups.get(stage, [])
    if not packets:
        raise RuntimeError(f"No packets available for fusion stage='{stage}'.")
    packets = sorted(packets, key=packet_sort_key)

    if args.fusion_packet_ranges is not None:
        selected_indices = parse_packet_ranges(args.fusion_packet_ranges, len(packets))
        packets = [packets[i] for i in selected_indices]
    elif args.fusion_max_packets is not None:
        packets = packets[: args.fusion_max_packets]

    out_dir = abs_path(args.work_dir) / args.fusion_out_name
    images_dir = out_dir / "images"
    gt_dir = out_dir / "gt"
    plots_dir = out_dir / "plots"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fixed_probe = None
    custom_probe_sequence = None
    if args.fusion_probe_mode == "packet_fixed_target":
        fixed_probe = build_fixed_probe_from_packets(args, packets)
        log(f"[4/4] Fusion: fixed packet probe {fixed_probe.label}")
    elif args.fusion_probe_mode == "custom_pose":
        if args.probe_pose_json is None:
            raise ValueError("custom_pose requires --probe_pose_json")
        fixed_probe = build_custom_probe(
            runtime,
            args,
            read_pose_json(abs_path(args.probe_pose_json)),
            label=abs_path(args.probe_pose_json).stem,
            gt_image_path=None if args.gt_image_path is None else abs_path(args.gt_image_path),
        )
        log(f"[4/4] Fusion: custom fixed probe {fixed_probe.label}")
    elif args.fusion_probe_mode == "custom_pose_sequence":
        if args.probe_poses_json is None:
            raise ValueError("custom_pose_sequence requires --probe_poses_json")
        custom_probe_sequence = [
            build_custom_probe(runtime, args, pose, label=f"{abs_path(args.probe_poses_json).stem}_{i:04d}")
            for i, pose in enumerate(read_pose_sequence_json(abs_path(args.probe_poses_json)))
        ]
        log(f"[4/4] Fusion: custom pose sequence with {len(custom_probe_sequence)} poses")
    else:
        log(f"[4/4] Fusion: probe mode {args.fusion_probe_mode}")

    # Final-map trajectory rendering mode.
    # Semantics: first fuse the selected packets once, then render that final map
    # along the selected packets' actual target camera trajectory. This is intended
    # for final-map inspection, not for prefix-diagnostic curves.
    if args.fusion_probe_mode in {"packet_last_only", "packet_trajectory"}:
        fused = concat_packets_to_gaussians(packets, device)
        probes = build_packet_trajectory_probes(packets)
        log(
            f"[4/4] Fusion: final trajectory render with {len(packets)} packet(s), "
            f"{int(fused.means.shape[1])} Gaussians, {len(probes)} probe view(s), "
            f"render_chunk_size={args.trajectory_render_chunk_size}"
        )
        per_view, metrics_mean, rendered_names, gt_names = render_fused_gaussians_to_probes(
            runtime=runtime,
            fused=fused,
            probes=probes,
            images_dir=images_dir,
            gt_dir=gt_dir,
            filename_prefix=f"final_k_{len(packets):03d}",
            should_compute_lpips=args.fusion_compute_lpips,
            render_chunk_size=args.trajectory_render_chunk_size,
        )
        if metrics_mean:
            msg = f"      final trajectory: PSNR={metrics_mean['psnr']:.4f}, SSIM={metrics_mean['ssim']:.4f}"
            if args.fusion_compute_lpips and "lpips" in metrics_mean:
                msg += f", LPIPS={metrics_mean['lpips']:.4f}"
            log(msg)
        else:
            log("      final trajectory: no GT")

        summary = {
            "output_dir": str(out_dir),
            "packet_stage_for_fusion": stage,
            "fusion_probe_mode": args.fusion_probe_mode,
            "semantics": "final_fused_map_rendered_along_selected_packet_target_trajectory",
            "selected_packet_count": len(packets),
            "num_gaussians": int(fused.means.shape[1]),
            "packet_scenes": [p["scene"] for p in packets],
            "context_indices": [p["context_index"].tolist() for p in packets],
            "target_indices": [p["target_index"].tolist() for p in packets],
            "rendered_images": rendered_names,
            "gt_images": gt_names,
            "metrics_mean": metrics_mean,
            "metrics_per_view": per_view,
            "selection": {
                "fusion_packet_ranges": args.fusion_packet_ranges,
                "fusion_max_packets": args.fusion_max_packets,
                "trajectory_render_chunk_size": args.trajectory_render_chunk_size,
            },
        }
        save_json(out_dir / "summary.json", summary)
        return summary

    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    steps: list[dict[str, Any]] = []
    curve_k: list[int] = []
    curve_gaussians: list[float] = []
    curve_psnr: list[float] = []
    curve_ssim: list[float] = []
    curve_lpips: list[float] = []

    for k in range(1, len(packets) + 1):
        prefix = packets[:k]
        fused = concat_packets_to_gaussians(prefix, device)
        probes = probes_for_prefix(args, runtime, packets, prefix, fixed_probe, custom_probe_sequence)

        image_shapes = {p.image_shape for p in probes}
        if len(image_shapes) != 1:
            raise ValueError(f"All probes must share image_shape, got {sorted(image_shapes)}")
        H, W = next(iter(image_shapes))
        extr = torch.stack([p.extrinsics for p in probes], dim=0).unsqueeze(0).to(device)
        intr = torch.stack([p.intrinsics for p in probes], dim=0).unsqueeze(0).to(device)
        near = torch.stack([p.near.reshape(()) for p in probes], dim=0).unsqueeze(0).to(device)
        far = torch.stack([p.far.reshape(()) for p in probes], dim=0).unsqueeze(0).to(device)

        output = runtime.model.decoder.forward(fused, extr, intr, near, far, (H, W), depth_mode=None)
        rendered = output.color[0].detach().cpu()

        rendered_names = []
        gt_names = []
        per_view = []
        gt_indices = [i for i, p in enumerate(probes) if p.gt_image is not None]
        metrics = {}
        if gt_indices:
            pred = rendered[gt_indices].to(device)
            gt = torch.stack([probes[i].gt_image for i in gt_indices], dim=0).to(device)
            metrics["psnr"] = compute_psnr(gt, pred)
            metrics["ssim"] = compute_ssim(gt, pred)
            if args.fusion_compute_lpips:
                metrics["lpips"] = compute_lpips(gt, pred)

        for i, (probe, img) in enumerate(zip(probes, rendered)):
            filename = f"k_{k:03d}_{probe.label}.png"
            save_png_tensor(img, images_dir / filename)
            rendered_names.append(filename)
            row = {
                "probe_label": probe.label,
                "gt_available": probe.gt_image is not None,
                "meta": probe.meta,
                "rendered_image": filename,
            }
            if probe.gt_image is not None:
                save_png_tensor(probe.gt_image, gt_dir / filename)
                gt_names.append(filename)
                local_gt_i = gt_indices.index(i)
                row["psnr"] = float(metrics["psnr"][local_gt_i].detach().cpu().item())
                row["ssim"] = float(metrics["ssim"][local_gt_i].detach().cpu().item())
                if args.fusion_compute_lpips:
                    row["lpips"] = float(metrics["lpips"][local_gt_i].detach().cpu().item())
            per_view.append(row)

        curve_k.append(k)
        curve_gaussians.append(float(fused.means.shape[1]))
        step = {
            "num_packets": k,
            "num_gaussians": int(fused.means.shape[1]),
            "packet_scenes": [p["scene"] for p in prefix],
            "context_indices": [p["context_index"].tolist() for p in prefix],
            "target_indices": [p["target_index"].tolist() for p in prefix],
            "rendered_images": rendered_names,
            "gt_images": gt_names,
            "metrics_per_view": per_view,
        }
        if metrics:
            mean_psnr = metric_mean(metrics["psnr"])
            mean_ssim = metric_mean(metrics["ssim"])
            curve_psnr.append(mean_psnr)
            curve_ssim.append(mean_ssim)
            step["metrics_mean"] = {"psnr": mean_psnr, "ssim": mean_ssim}
            msg = f"      fusion {k}/{len(packets)}: G={int(fused.means.shape[1])}, PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}"
            if args.fusion_compute_lpips:
                mean_lpips = metric_mean(metrics["lpips"])
                curve_lpips.append(mean_lpips)
                step["metrics_mean"]["lpips"] = mean_lpips
                msg += f", LPIPS={mean_lpips:.4f}"
            log(msg)
        else:
            log(f"      fusion {k}/{len(packets)}: G={int(fused.means.shape[1])}, no GT")
        steps.append(step)

    plot_curve(curve_k, curve_gaussians, "Number of Gaussians vs Number of Fused Packets", "Number of Gaussians", plots_dir / "num_gaussians_vs_packets.png")
    if curve_psnr:
        plot_curve(curve_k, curve_psnr, "PSNR vs Number of Fused Packets", "PSNR", plots_dir / "psnr_vs_packets.png")
    if curve_ssim:
        plot_curve(curve_k, curve_ssim, "SSIM vs Number of Fused Packets", "SSIM", plots_dir / "ssim_vs_packets.png")
    if curve_lpips:
        plot_curve(curve_k, curve_lpips, "LPIPS vs Number of Fused Packets", "LPIPS", plots_dir / "lpips_vs_packets.png")

    summary = {
        "output_dir": str(out_dir),
        "packet_stage_for_fusion": stage,
        "fusion_probe_mode": args.fusion_probe_mode,
        "selected_packet_count": len(packets),
        "curves": {
            "num_packets": curve_k,
            "num_gaussians": curve_gaussians,
        },
        "steps": steps,
    }
    if curve_psnr:
        summary["curves"]["psnr"] = curve_psnr
    if curve_ssim:
        summary["curves"]["ssim"] = curve_ssim
    if curve_lpips:
        summary["curves"]["lpips"] = curve_lpips
    save_json(out_dir / "summary.json", summary)
    log(f"      fusion summary: {out_dir / 'summary.json'}")
    return summary


def parse_packet_ranges(spec: str, num_packets: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            indices = range(int(a), int(b) + 1)
        else:
            indices = [int(part)]
        for i in indices:
            if i < 0 or i >= num_packets:
                raise IndexError(f"Packet index {i} out of range 0..{num_packets-1}")
            if i not in seen:
                selected.append(i)
                seen.add(i)
    if not selected:
        raise ValueError(f"No packet selected by range spec: {spec}")
    return selected


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="API-style ZipMap -> ReSplat packets -> fusion pipeline")

    # Repositories and input.
    ap.add_argument("--zipmap_repo", required=True)
    ap.add_argument("--resplat_repo", required=True)
    ap.add_argument("--zipmap_export_script", default=None, help="Optional explicit path to export_zipmap_predictions.py")
    ap.add_argument("--zipmap_eval_script", default=None, help="Optional explicit path to evaluate_zipmap_pose.py")
    ap.add_argument("--zipmap_ckpt", required=True)
    ap.add_argument("--left_dir", required=True)
    ap.add_argument("--right_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=None, help="Exclusive end index. Example: 20 41 selects 20..40.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_frames", type=int, default=None, help="Maximum number of selected stereo pairs after range/stride.")
    ap.add_argument("--scene_name", default="P000")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=1)

    # ZipMap.
    ap.add_argument("--zipmap_out_name", default="zipmap_outputs")
    ap.add_argument("--zipmap_ema", action="store_true")
    ap.add_argument("--zipmap_affine_invariant", type=str2bool, default=True)
    ap.add_argument("--zipmap_align_first_view", type=str2bool, default=True)
    ap.add_argument("--zipmap_target_size", type=int, default=518)
    ap.add_argument("--zipmap_preprocess_mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--zipmap_cpu", action="store_true")
    ap.add_argument("--save_zipmap_input_images", type=str2bool, default=False)
    ap.add_argument("--save_zipmap_depth", type=str2bool, default=True)
    ap.add_argument("--save_zipmap_points", type=str2bool, default=True)

    # Pose alignment.
    ap.add_argument("--gt_pose_file", default=None)
    ap.add_argument("--pose_alignment", choices=["none", "se3", "sim3"], default="sim3")
    ap.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    ap.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    ap.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    ap.add_argument("--rpe_delta", type=int, default=1)

    # ReSplat config/model.
    ap.add_argument("--resplat_experiment", default="tartanair_p000_ft")
    ap.add_argument("--resplat_override", action="append", default=[], help="Hydra override for model/config only. Not used for dataset sampling.")
    ap.add_argument("--resplat_checkpoint", default=None, help="Override checkpointing.pretrained_model.")
    ap.add_argument("--resplat_out_name", default="resplat_runtime")
    ap.add_argument("--resplat_pose_source", choices=["aligned", "raw"], default="aligned")
    ap.add_argument("--resplat_packet_stage", choices=["init", "final", "both"], default="final")
    ap.add_argument("--packet_stage_for_fusion", choices=["init", "final"], default="final")
    ap.add_argument("--resplat_target_camera", choices=["left", "right", "both"], default="left")
    ap.add_argument("--resplat_target_offset", type=int, default=0, help="Target frame offset in selected-frame local index.")
    ap.add_argument("--drop_invalid_target_offset", action="store_true")
    ap.add_argument("--stereo_baseline", type=float, default=0.25000006)
    ap.add_argument("--save_packets", type=str2bool, default=True)
    ap.add_argument("--packet_out_name", default="gaussian_packets_api")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)

    # Fusion.
    ap.add_argument("--skip_fusion", action="store_true")
    ap.add_argument("--fusion_out_name", default="fusion_eval_api")
    ap.add_argument(
        "--fusion_probe_mode",
        choices=["packet_last_only", "packet_trajectory", "prefix_last_only", "packet_fixed_target", "custom_pose", "custom_pose_sequence"],
        default="packet_last_only",
        help=(
            "packet_last_only/packet_trajectory: fuse selected packets once and render the final map along packet target trajectory. "
            "prefix_last_only: old diagnostic behavior, render the latest packet target at every prefix step. "
            "packet_fixed_target/custom_pose/custom_pose_sequence: prefix diagnostics with fixed/custom probes."
        ),
    )
    ap.add_argument("--fixed_packet_local_index", type=int, default=None)
    ap.add_argument("--fixed_target_index", type=int, default=None)
    ap.add_argument("--fixed_target_camera_id", type=int, default=0)
    ap.add_argument("--probe_pose_json", default=None)
    ap.add_argument("--probe_poses_json", default=None)
    ap.add_argument("--gt_image_path", default=None)
    ap.add_argument("--probe_image_shape", nargs=2, type=int, default=None, metavar=("H", "W"), help="Custom probe render size, e.g. --probe_image_shape 540 960. Used for custom_pose/custom_pose_sequence.")
    ap.add_argument("--probe_intrinsics_norm", nargs=4, type=float, default=None, metavar=("FXN", "FYN", "CXN", "CYN"), help="Custom normalized probe intrinsics.")
    ap.add_argument("--probe_intrinsics_pixel", nargs=4, type=float, default=None, metavar=("FX", "FY", "CX", "CY"), help="Custom pixel-space probe intrinsics; converted to normalized intrinsics internally.")
    ap.add_argument("--probe_near", type=float, default=None, help="Optional near plane for custom probe rendering.")
    ap.add_argument("--probe_far", type=float, default=None, help="Optional far plane for custom probe rendering.")
    ap.add_argument("--fusion_packet_ranges", default=None)
    ap.add_argument("--fusion_max_packets", type=int, default=None)
    ap.add_argument("--trajectory_render_chunk_size", type=int, default=1, help="For packet_trajectory/packet_last_only final-map rendering, render this many probe cameras per decoder call. Default 1 is safest for large fused maps.")
    ap.add_argument("--fusion_compute_lpips", action="store_true")
    ap.add_argument("--script_version", action="store_true", help="Print script version and exit.")

    return ap


def main() -> None:
    if "--script_version" in sys.argv:
        print(SCRIPT_VERSION)
        return
    args = build_argparser().parse_args()
    work_dir = abs_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    selected = select_stereo_frames(
        left_dir=abs_path(args.left_dir),
        right_dir=abs_path(args.right_dir),
        start_index=args.start_index,
        end_index=args.end_index,
        stride=args.stride,
        num_frames=args.num_frames,
        recursive=args.recursive,
    )
    log(
        f"Selected {len(selected.indices)} stereo pair(s): "
        f"original_index first={selected.indices[0]}, last={selected.indices[-1]} "
        f"(local_index first=0, last={len(selected.indices) - 1})"
    )

    zipmap_result = run_zipmap_api(args, selected)
    runtime = load_resplat_runtime(args)
    packet_groups = infer_resplat_packets(runtime, zipmap_result, args)
    fusion_summary = run_fusion_api(runtime, packet_groups, args)

    run_summary = {
        "work_dir": str(work_dir),
        "selected_original_indices": [int(i) for i in selected.indices],
        "selected_local_to_original": [
            {"local_index": int(local_i), "original_index": int(original_i)}
            for local_i, original_i in enumerate(selected.indices)
        ],
        "zipmap_out": str(zipmap_result.out_dir),
        "pose_eval": zipmap_result.pose_eval_summary,
        "packet_counts": {k: len(v) for k, v in packet_groups.items()},
        "fusion_summary": None if fusion_summary is None else str(work_dir / args.fusion_out_name / "summary.json"),
        "elapsed_sec": time.time() - t0,
    }
    save_json(work_dir / "run_summary.json", run_summary)
    log("Done.")
    log(f"  run summary: {work_dir / 'run_summary.json'}")
    log(f"  zipmap outputs: {zipmap_result.out_dir}")
    if args.save_packets:
        log(f"  packets: {work_dir / args.packet_out_name}")
    if fusion_summary is not None:
        log(f"  fusion: {work_dir / args.fusion_out_name}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
