#!/usr/bin/env python3
"""
Experiment 5: true prefix-only stereo scale estimation.

For each requested prefix length k, this script independently runs ZipMap on only
[L0,R0,...,L{k-1},R{k-1}]. Therefore no future image can affect the predicted
stereo baselines used at prefix k.

The estimated scale is then transferred to the same saved 50-frame left-only
trajectory from Experiment 1, so the only tested factor is the prefix-only
stereo scale estimate.

This is not yet a state-reusing online implementation: every k is rerun from a
fresh checkpoint. It is a causal availability test, not a runtime-efficient
streaming system.

Example:
TORCH_COMPILE_DISABLE=1 python run_zipmap_stereo_prefix_only_scale_experiment.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --ckpt_path /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_online.pt \
  --left_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --right_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_rcam_front \
  --experiment_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_stereo_scale_P000_0_50 \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
  --prefix_lengths 1,2,3,5,10,15,20,30,40,50 \
  --known_baseline 0.25000006 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    "zipmap_prefix_only_base",
    Path(__file__).resolve().with_name("run_zipmap_official_streaming_demo_cli_eval.py"),
)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_prefix_lengths(text: str, n: int) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if 1 <= k <= n:
            values.append(k)
    if not values:
        raise ValueError("No valid prefix lengths")
    return sorted(set(values))


def scale_translations(T: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(T, dtype=np.float64).copy()
    p0 = out[0, :3, 3].copy()
    out[:, :3, 3] = p0 + float(scale) * (out[:, :3, 3] - p0)
    return out.astype(np.float32)


def evaluate_se3(eval_mod, T_pred: np.ndarray, T_gt: np.ndarray, delta: int):
    _, R, t = eval_mod.umeyama_alignment(
        T_pred[:, :3, 3].astype(np.float64),
        T_gt[:, :3, 3].astype(np.float64),
        with_scale=False,
    )
    T_eval = eval_mod.apply_similarity_to_poses(
        T_pred.astype(np.float64), 1.0, R, t
    ).astype(np.float32)
    ate = eval_mod.compute_ate(T_eval.astype(np.float64), T_gt.astype(np.float64))
    _, rpe = eval_mod.compute_rpe(
        T_eval.astype(np.float64), T_gt.astype(np.float64), delta=delta
    )
    return ate, rpe


def run_prefix(args, device, image_paths, export_mod):
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri

    model, _ = BASE.load_model(args, device)

    images, preprocess_sec = BASE.timed(
        device,
        lambda: load_and_preprocess_images(
            [str(p) for p in image_paths],
            target_size=args.target_size,
            mode=args.preprocess_mode,
        ).to(device),
    )

    def infer():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            return model(images)

    pred, infer_sec = BASE.timed(device, infer)

    def decode():
        extrinsic, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        extrinsic = BASE.tensor_to_np_no_batch(extrinsic).astype(np.float32)
        T_w2c = export_mod.to_homogeneous_4x4(extrinsic)
        if args.align_first_view:
            T_w2c = export_mod.align_w2c_to_first_camera(T_w2c).astype(np.float32)
        return export_mod.invert_se3_np(T_w2c).astype(np.float32)

    T_c2w, decode_sec = BASE.timed(device, decode)
    del model, pred, images
    torch.cuda.empty_cache()
    return T_c2w, {
        "num_images": len(image_paths),
        "preprocess_sec": float(preprocess_sec),
        "infer_sec": float(infer_sec),
        "decode_sec": float(decode_sec),
        "infer_sec_per_image": float(infer_sec / max(len(image_paths), 1)),
        "inference_fps": float(len(image_paths) / max(infer_sec, 1e-12)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="True prefix-only ZipMap stereo scale experiment")
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--left_dir", required=True)
    p.add_argument("--right_dir", required=True)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--gt_pose_file", required=True)
    p.add_argument("--prefix_lengths", default="1,2,3,5,10,15,20,30,40,50")
    p.add_argument("--known_baseline", type=float, default=0.25000006)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    p.add_argument("--affine_invariant", type=BASE.str2bool, default=True)
    p.add_argument("--align_first_view", type=BASE.str2bool, default=True)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    repo = BASE.abs_path(args.zipmap_repo)
    exp_dir = BASE.abs_path(args.experiment_dir)
    out_dir = exp_dir / "pose_eval" / "stereo_prefix_only_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_mod = BASE.import_module_from_path(
        "zipmap_export_prefix_only", repo / "tools" / "export_zipmap_predictions.py"
    )
    eval_mod = BASE.import_module_from_path(
        "zipmap_eval_prefix_only", repo / "tools" / "evaluate_zipmap_pose.py"
    )
    sys.path.insert(0, str(repo))

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA required: {args.device}")

    left_all = BASE.collect_images(BASE.abs_path(args.left_dir), args.recursive)
    right_all = BASE.collect_images(BASE.abs_path(args.right_dir), args.recursive)
    if len(left_all) != len(right_all):
        raise ValueError(f"Left/right count mismatch: {len(left_all)} vs {len(right_all)}")

    left_saved = np.load(exp_dir / "left_only_predictions.npz")
    T_left_full = np.asarray(left_saved["T_c2w_opencv"], dtype=np.float32)
    indices = np.asarray(left_saved["selected_original_indices"], dtype=np.int64)
    n = len(indices)
    prefixes = parse_prefix_lengths(args.prefix_lengths, n)

    T_gt_all = eval_mod.load_gt_trajectory(
        BASE.abs_path(args.gt_pose_file),
        args.gt_convention,
        args.gt_quat_order,
        args.gt_matrix_convention,
    )
    T_gt_full = T_gt_all[indices].astype(np.float32)
    gt_scale, _, _ = eval_mod.umeyama_alignment(
        T_left_full[:, :3, 3].astype(np.float64),
        T_gt_full[:, :3, 3].astype(np.float64),
        with_scale=True,
    )

    rows: list[dict[str, Any]] = []
    for k in prefixes:
        selected = indices[:k]
        stereo_paths = []
        for idx in selected:
            stereo_paths.extend([left_all[int(idx)], right_all[int(idx)]])

        BASE.log(f"[Prefix-only] k={k}, images={len(stereo_paths)}")
        T_stereo, timing = run_prefix(args, device, stereo_paths, export_mod)
        T_l = T_stereo[0::2]
        T_r = T_stereo[1::2]
        baselines = np.linalg.norm(T_r[:, :3, 3] - T_l[:, :3, 3], axis=1)
        median_baseline = float(np.median(baselines))
        scale = float(args.known_baseline / max(median_baseline, 1e-12))

        T_left_scaled = scale_translations(T_left_full, scale)
        ate, rpe = evaluate_se3(eval_mod, T_left_scaled, T_gt_full, args.rpe_delta)

        per_pair_rows = [
            {
                "local_pair_index": i,
                "original_index": int(selected[i]),
                "predicted_baseline": float(baselines[i]),
            }
            for i in range(k)
        ]
        save_csv(out_dir / f"k_{k:03d}_baselines.csv", per_pair_rows)

        row = {
            "num_stereo_pairs": k,
            "num_images": 2 * k,
            "estimated_scale": scale,
            "median_predicted_baseline": median_baseline,
            "relative_scale_error_to_gt_sim3_percent": 100.0 * abs(scale - gt_scale) / max(abs(gt_scale), 1e-12),
            "full_trajectory_ate_rmse": float(ate["rmse"]),
            "full_trajectory_rpe_translation_rmse": float(rpe["translation_rmse"]),
            **timing,
        }
        rows.append(row)
        BASE.log(
            f"  scale={scale:.9f}, ATE={ate['rmse']:.6f}, "
            f"infer={timing['infer_sec']:.4f}s, FPS={timing['inference_fps']:.2f}"
        )

    save_csv(out_dir / "prefix_only_results.csv", rows)
    result = {
        "research_question": "How many stereo pairs are sufficient when each prefix is inferred without future images?",
        "method": {
            "prefix_input": "Only L0,R0,...,L{k-1},R{k-1} are passed to ZipMap for each k",
            "fresh_checkpoint_per_prefix": True,
            "future_images_used": False,
            "scale_aggregation": "median of all baselines predicted within that prefix run",
            "pair_filtering": "none",
            "trajectory_evaluated": "same saved full left-only trajectory",
            "note": "Causal prefix test; not yet state-reusing online inference",
        },
        "gt_sim3_reference_scale": float(gt_scale),
        "prefix_lengths": prefixes,
        "results": rows,
    }
    save_json(out_dir / "summary.json", result)
    save_json(exp_dir / "prefix_only_inference_summary.json", result)
    print(f"[Done] {exp_dir / 'prefix_only_inference_summary.json'}")


if __name__ == "__main__":
    main()
