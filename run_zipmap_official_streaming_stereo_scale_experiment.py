#!/usr/bin/env python3
"""
Experiment 1: use a calibrated stereo baseline to recover one global ZipMap scale.

The first version deliberately uses the simplest procedure:
  - one stereo-interleaved forward: L0, R0, L1, R1, ...
  - all stereo pairs are used equally; no filtering or robust estimator
  - scale = known_baseline / mean(predicted_left_right_baseline)
  - scale only the left-camera translations
  - evaluate with SE(3), which cannot correct scale

A fresh checkpoint is loaded for the left-only control and stereo experiment,
so test-time-training state cannot leak between the two runs.

Example:
TORCH_COMPILE_DISABLE=1 python run_zipmap_official_streaming_stereo_scale_experiment.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --ckpt_path /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_online.pt \
  --left_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --right_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_rcam_front \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
  --output_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_stereo_scale_P000_0_50 \
  --start_index 0 --end_index 50 \
  --known_baseline 0.25000006 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def load_python_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_python_file(
    "zipmap_official_streaming_cli_base",
    Path(__file__).resolve().with_name("run_zipmap_official_streaming_demo_cli_eval.py"),
)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sequence(args, device, image_paths, label, export_mod):
    """Run one independent sequence with a freshly loaded checkpoint."""
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri

    BASE.log(f"[{label}] load fresh model")
    model, model_config = BASE.load_model(args, device)

    images, preprocess_sec = BASE.timed(
        device,
        lambda: load_and_preprocess_images(
            [str(p) for p in image_paths],
            target_size=args.target_size,
            mode=args.preprocess_mode,
        ).to(device),
    )
    BASE.log(f"[{label}] preprocess: shape={tuple(images.shape)}, time={preprocess_sec:.4f}s")

    def infer():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            return model(images)

    predictions, infer_sec = BASE.timed(device, infer)
    BASE.log(f"[{label}] inference: time={infer_sec:.4f}s")

    def decode():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        extrinsic = BASE.tensor_to_np_no_batch(extrinsic).astype(np.float32)
        intrinsic = BASE.tensor_to_np_no_batch(intrinsic).astype(np.float32)
        T_w2c = export_mod.to_homogeneous_4x4(extrinsic)
        if args.align_first_view:
            T_w2c = export_mod.align_w2c_to_first_camera(T_w2c).astype(np.float32)
            extrinsic = T_w2c[:, :3, :4].astype(np.float32)
        T_c2w = export_mod.invert_se3_np(T_w2c).astype(np.float32)
        return extrinsic, intrinsic, T_w2c.astype(np.float32), T_c2w

    (extrinsic, intrinsic, T_w2c, T_c2w), decode_sec = BASE.timed(device, decode)
    BASE.log(f"[{label}] decode: time={decode_sec:.4f}s")

    timing = {
        "num_images": len(image_paths),
        "preprocess_sec": float(preprocess_sec),
        "infer_sec": float(infer_sec),
        "decode_sec": float(decode_sec),
        "total_sec": float(preprocess_sec + infer_sec + decode_sec),
        "infer_sec_per_image": float(infer_sec / max(1, len(image_paths))),
    }
    del model, predictions, images
    torch.cuda.empty_cache()
    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "T_w2c": T_w2c,
        "T_c2w": T_c2w,
        "timing": timing,
        "model_config": model_config,
    }


def stereo_measurements(T_left, T_right, known_baseline):
    rows = []
    for i, (T_l, T_r) in enumerate(zip(T_left, T_right)):
        rel = np.linalg.inv(T_l.astype(np.float64)) @ T_r.astype(np.float64)
        baseline = float(np.linalg.norm(T_r[:3, 3] - T_l[:3, 3]))
        cos_angle = np.clip((np.trace(rel[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        rows.append(
            {
                "pair_local_index": i,
                "predicted_baseline": baseline,
                "known_baseline": float(known_baseline),
                "scale_from_this_pair": float(known_baseline / max(baseline, 1e-12)),
                "relative_rotation_deg": float(np.degrees(np.arccos(cos_angle))),
                "relative_tx": float(rel[0, 3]),
                "relative_ty": float(rel[1, 3]),
                "relative_tz": float(rel[2, 3]),
            }
        )

    b = np.asarray([x["predicted_baseline"] for x in rows], dtype=np.float64)
    r = np.asarray([x["relative_rotation_deg"] for x in rows], dtype=np.float64)
    mean_b = float(np.mean(b))
    summary = {
        "num_pairs": len(rows),
        "known_baseline": float(known_baseline),
        "predicted_baseline_mean": mean_b,
        "predicted_baseline_median": float(np.median(b)),
        "predicted_baseline_std": float(np.std(b)),
        "predicted_baseline_min": float(np.min(b)),
        "predicted_baseline_max": float(np.max(b)),
        "global_scale_method": "known_baseline / mean(all predicted baselines)",
        "global_scale": float(known_baseline / max(mean_b, 1e-12)),
        "relative_rotation_deg_mean": float(np.mean(r)),
        "relative_rotation_deg_median": float(np.median(r)),
    }
    return rows, summary


def scale_translations(T_c2w, scale):
    out = np.asarray(T_c2w, dtype=np.float64).copy()
    p0 = out[0, :3, 3].copy()
    out[:, :3, 3] = p0 + float(scale) * (out[:, :3, 3] - p0)
    return out.astype(np.float32)


def evaluate(eval_mod, T_pred, T_gt, alignment, rpe_delta):
    if alignment == "sim3":
        s, R, t = eval_mod.umeyama_alignment(T_pred[:, :3, 3], T_gt[:, :3, 3], with_scale=True)
    elif alignment == "se3":
        s, R, t = eval_mod.umeyama_alignment(T_pred[:, :3, 3], T_gt[:, :3, 3], with_scale=False)
    else:
        raise ValueError(alignment)
    T_eval = eval_mod.apply_similarity_to_poses(T_pred, s, R, t).astype(np.float32)
    ate = eval_mod.compute_ate(T_eval, T_gt)
    rpe_rows, rpe = eval_mod.compute_rpe(T_eval, T_gt, delta=rpe_delta)
    return T_eval, {
        "alignment": alignment,
        "alignment_scale": float(s),
        "R": R.tolist(),
        "t": t.tolist(),
        "ate": ate,
        "rpe": rpe,
    }, rpe_rows


def save_evaluation(eval_mod, root, name, T_eval, T_gt, names, indices, summary, rpe_rows):
    out = root / "pose_eval" / name
    out.mkdir(parents=True, exist_ok=True)
    eval_mod.write_pose_tum(out / "trajectory_pred_aligned_c2w_opencv.txt", T_eval, names=names)
    eval_mod.write_pose_tum(out / "trajectory_gt_matched_c2w_opencv.txt", T_gt, names=names)
    errors = np.linalg.norm(T_eval[:, :3, 3] - T_gt[:, :3, 3], axis=1)
    save_csv(
        out / "per_frame_errors.csv",
        [
            {
                "local_index": i,
                "original_index": int(indices[i]),
                "image_name": names[i],
                "ate_translation_error": float(errors[i]),
            }
            for i in range(len(errors))
        ],
    )
    save_csv(out / "rpe_errors.csv", rpe_rows)
    BASE.save_json(out / "summary.json", summary)


def plot_baselines(output_dir, rows, global_scale):
    if plt is None:
        return
    x = np.arange(len(rows))
    raw = np.asarray([r["predicted_baseline"] for r in rows])
    known = float(rows[0]["known_baseline"])
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(x, raw)
    plt.xlabel("Stereo pair index")
    plt.ylabel("Predicted baseline [ZipMap unit]")
    plt.title("Raw ZipMap stereo baseline")
    plt.tight_layout()
    plt.savefig(plots / "stereo_baseline_raw.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, raw * global_scale, label="One global scale")
    plt.axhline(known, linestyle="--", label="Known baseline")
    plt.xlabel("Stereo pair index")
    plt.ylabel("Baseline [m]")
    plt.title("Stereo baseline after global scale correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "stereo_baseline_scaled.png", dpi=160)
    plt.close()


def build_parser():
    p = argparse.ArgumentParser(description="ZipMap stereo-baseline scale experiment")
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--left_dir", required=True)
    p.add_argument("--right_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--known_baseline", type=float, default=0.25000006)
    p.add_argument("--run_left_only", type=BASE.str2bool, default=True)
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    p.add_argument("--affine_invariant", type=BASE.str2bool, default=True)
    p.add_argument("--align_first_view", type=BASE.str2bool, default=True)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gt_pose_file", default=None)
    p.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    return p


def main():
    torch.set_float32_matmul_precision("high")
    args = build_parser().parse_args()
    if args.known_baseline <= 0:
        raise ValueError("--known_baseline must be positive")

    repo = BASE.abs_path(args.zipmap_repo)
    output = BASE.abs_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not (repo / "zipmap").is_dir():
        raise FileNotFoundError(f"zipmap package not found under {repo}")

    export_mod = BASE.import_module_from_path(
        "zipmap_export_stereo_scale", repo / "tools" / "export_zipmap_predictions.py"
    )
    eval_mod = BASE.import_module_from_path(
        "zipmap_eval_stereo_scale", repo / "tools" / "evaluate_zipmap_pose.py"
    )
    sys.path.insert(0, str(repo))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required: {args.device}")

    left_all = BASE.collect_images(BASE.abs_path(args.left_dir), args.recursive)
    right_all = BASE.collect_images(BASE.abs_path(args.right_dir), args.recursive)
    if len(left_all) != len(right_all):
        raise ValueError(f"Left/right count mismatch: {len(left_all)} vs {len(right_all)}")
    indices = BASE.make_sample_indices(
        len(left_all), args.start_index, args.end_index, args.stride, args.num_frames
    )
    left = [left_all[i] for i in indices]
    right = [right_all[i] for i in indices]
    stereo_paths = [p for lr in zip(left, right) for p in lr]
    left_names = [p.name for p in left]

    BASE.log(f"[Input] {len(indices)} stereo pairs, original {indices[0]}..{indices[-1]}")
    BASE.log("[Method] one interleaved forward; all pairs equally used; one global scale")

    sequence_rows = []
    for i, original_index in enumerate(indices):
        for camera_id, path, sequence_index in (
            ("left", left[i], 2 * i),
            ("right", right[i], 2 * i + 1),
        ):
            sequence_rows.append(
                {
                    "sequence_index": sequence_index,
                    "pair_local_index": i,
                    "original_index": int(original_index),
                    "camera_id": camera_id,
                    "source_name": path.name,
                    "source_path": str(path),
                }
            )
    save_csv(output / "stereo_sequence.csv", sequence_rows)
    BASE.save_json(output / "stereo_sequence.json", sequence_rows)

    start = time.perf_counter()
    stereo = run_sequence(args, device, stereo_paths, "Stereo", export_mod)
    T_left = stereo["T_c2w"][0::2].copy()
    T_right = stereo["T_c2w"][1::2].copy()
    if len(T_left) != len(indices) or len(T_right) != len(indices):
        raise RuntimeError("Unexpected number of decoded stereo poses")

    baseline_rows, baseline_summary = stereo_measurements(T_left, T_right, args.known_baseline)
    for row, original_index, left_path, right_path in zip(baseline_rows, indices, left, right):
        row.update(
            original_index=int(original_index),
            left_name=left_path.name,
            right_name=right_path.name,
        )
    scale = float(baseline_summary["global_scale"])
    T_left_scaled = scale_translations(T_left, scale)
    save_csv(output / "stereo_baseline_per_pair.csv", baseline_rows)
    BASE.save_json(output / "stereo_baseline_summary.json", baseline_summary)
    plot_baselines(output, baseline_rows, scale)

    np.savez_compressed(
        output / "stereo_predictions.npz",
        extrinsic=stereo["extrinsic"],
        intrinsic=stereo["intrinsic"],
        T_w2c_opencv=stereo["T_w2c"],
        T_c2w_opencv=stereo["T_c2w"],
        T_left_c2w_opencv=T_left,
        T_right_c2w_opencv=T_right,
        T_left_baseline_scaled_c2w_opencv=T_left_scaled,
        selected_original_indices=np.asarray(indices),
        known_baseline=np.asarray(args.known_baseline),
        estimated_global_scale=np.asarray(scale),
    )
    eval_mod.write_pose_tum(output / "poses_stereo_left_raw.txt", T_left, names=left_names)
    eval_mod.write_pose_tum(output / "poses_stereo_right_raw.txt", T_right, names=[p.name for p in right])
    eval_mod.write_pose_tum(output / "poses_stereo_left_baseline_scaled.txt", T_left_scaled, names=left_names)

    left_only = None
    if args.run_left_only:
        left_only = run_sequence(args, device, left, "Left-only control", export_mod)
        np.savez_compressed(
            output / "left_only_predictions.npz",
            extrinsic=left_only["extrinsic"],
            intrinsic=left_only["intrinsic"],
            T_w2c_opencv=left_only["T_w2c"],
            T_c2w_opencv=left_only["T_c2w"],
            selected_original_indices=np.asarray(indices),
        )
        eval_mod.write_pose_tum(output / "poses_left_only_raw.txt", left_only["T_c2w"], names=left_names)

    experiments = {}
    scales = {"stereo_baseline_global_scale": scale}
    if args.gt_pose_file:
        T_gt_all = eval_mod.load_gt_trajectory(
            BASE.abs_path(args.gt_pose_file),
            args.gt_convention,
            args.gt_quat_order,
            args.gt_matrix_convention,
        )
        T_gt = T_gt_all[indices].astype(np.float32)
        specs = [
            ("stereo_raw_se3", T_left, "se3"),
            ("stereo_baseline_scaled_se3", T_left_scaled, "se3"),
            ("stereo_gt_sim3_upper_bound", T_left, "sim3"),
        ]
        if left_only is not None:
            specs += [
                ("left_only_raw_se3", left_only["T_c2w"], "se3"),
                ("left_only_gt_sim3_upper_bound", left_only["T_c2w"], "sim3"),
            ]

        for name, T_pred, alignment in specs:
            T_eval, summary, rpe_rows = evaluate(eval_mod, T_pred, T_gt, alignment, args.rpe_delta)
            experiments[name] = summary
            save_evaluation(
                eval_mod, output, name, T_eval, T_gt, left_names, indices, summary, rpe_rows
            )
            BASE.log(
                f"[Eval] {name}: RMSE={summary['ate']['rmse']:.6f}, "
                f"alignment_scale={summary['alignment_scale']:.6f}"
            )
        scales["stereo_gt_sim3_scale"] = experiments["stereo_gt_sim3_upper_bound"]["alignment_scale"]
        if "left_only_gt_sim3_upper_bound" in experiments:
            scales["left_only_gt_sim3_scale"] = experiments["left_only_gt_sim3_upper_bound"]["alignment_scale"]

    final_summary = {
        "research_question": "Can a known stereo baseline recover one useful global ZipMap scale?",
        "method": {
            "input_order": "L0,R0,L1,R1,...",
            "scale_estimator": "known_baseline / mean(all predicted baselines)",
            "pair_filtering": "none",
            "dynamic_scale": False,
        },
        "selected_original_indices": [int(i) for i in indices],
        "baseline": baseline_summary,
        "scale_comparison": scales,
        "experiments": experiments,
        "timing": {
            "stereo": stereo["timing"],
            "left_only": None if left_only is None else left_only["timing"],
            "total_sec": float(time.perf_counter() - start),
        },
        "model_config": stereo["model_config"],
    }
    BASE.save_json(output / "experiment_summary.json", final_summary)
    BASE.log(f"[Done] stereo global scale={scale:.6f}")
    BASE.log(f"       summary={output / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
