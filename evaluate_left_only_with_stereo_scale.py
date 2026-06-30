#!/usr/bin/env python3
"""
Experiment 2: transfer the stereo-baseline scale to the original left-only ZipMap trajectory.

This is a pure post-processing experiment. It does not rerun ZipMap.
It reads the output of run_zipmap_official_streaming_stereo_scale_experiment.py,
uses the already estimated global stereo scale, applies it to the left-only trajectory,
and evaluates the result with SE(3) alignment.

Example:
python evaluate_left_only_with_stereo_scale.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --experiment_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_stereo_scale_P000_0_50 \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt
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

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def abs_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scale_translations(T_c2w: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(T_c2w, dtype=np.float64).copy()
    p0 = out[0, :3, 3].copy()
    out[:, :3, 3] = p0 + float(scale) * (out[:, :3, 3] - p0)
    return out.astype(np.float32)


def load_left_names(experiment_dir: Path, indices: np.ndarray) -> list[str]:
    sequence_json = experiment_dir / "stereo_sequence.json"
    if sequence_json.exists():
        records = json.loads(sequence_json.read_text(encoding="utf-8"))
        left_records = [r for r in records if str(r.get("camera_id")) == "left"]
        if len(left_records) == len(indices):
            return [str(r.get("source_name", f"{int(indices[i]):06d}")) for i, r in enumerate(left_records)]
    return [f"{int(i):06d}" for i in indices]


def evaluate_se3(eval_mod, T_pred: np.ndarray, T_gt: np.ndarray, rpe_delta: int):
    _, R_align, t_align = eval_mod.umeyama_alignment(
        T_pred[:, :3, 3].astype(np.float64),
        T_gt[:, :3, 3].astype(np.float64),
        with_scale=False,
    )
    T_eval = eval_mod.apply_similarity_to_poses(
        T_pred.astype(np.float64), 1.0, R_align, t_align
    ).astype(np.float32)
    ate = eval_mod.compute_ate(T_eval.astype(np.float64), T_gt.astype(np.float64))
    rpe_rows, rpe = eval_mod.compute_rpe(
        T_eval.astype(np.float64), T_gt.astype(np.float64), delta=rpe_delta
    )
    return T_eval, ate, rpe_rows, rpe, R_align, t_align


def plot_comparison(
    output_dir: Path,
    T_gt: np.ndarray,
    T_raw_eval: np.ndarray,
    T_scaled_eval: np.ndarray,
    raw_errors: np.ndarray,
    scaled_errors: np.ndarray,
) -> None:
    if plt is None:
        return
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot(T_gt[:, 0, 3], T_gt[:, 2, 3], label="GT")
    plt.plot(T_raw_eval[:, 0, 3], T_raw_eval[:, 2, 3], label="Left-only raw + SE3")
    plt.plot(T_scaled_eval[:, 0, 3], T_scaled_eval[:, 2, 3], label="Left-only + stereo scale + SE3")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Left-only trajectory before and after stereo-scale transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "trajectory_comparison_xz.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    x = np.arange(len(raw_errors))
    plt.plot(x, raw_errors, label="Left-only raw + SE3")
    plt.plot(x, scaled_errors, label="Left-only + stereo scale + SE3")
    plt.xlabel("Frame")
    plt.ylabel("ATE translation error [m]")
    plt.title("Per-frame ATE before and after stereo-scale transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "ate_comparison.png", dpi=160)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply stereo baseline scale to a saved left-only ZipMap trajectory")
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--gt_pose_file", required=True)
    p.add_argument(
        "--gt_convention",
        choices=["opencv_c2w", "resplat_tartanair_pose"],
        default="resplat_tartanair_pose",
    )
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo = abs_path(args.zipmap_repo)
    experiment_dir = abs_path(args.experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    eval_mod = import_module_from_path(
        "zipmap_eval_scale_transfer",
        repo / "tools" / "evaluate_zipmap_pose.py",
    )

    summary_path = experiment_dir / "experiment_summary.json"
    left_npz_path = experiment_dir / "left_only_predictions.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not left_npz_path.exists():
        raise FileNotFoundError(f"Missing {left_npz_path}")

    experiment_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stereo_scale = float(experiment_summary["scale_comparison"]["stereo_baseline_global_scale"])

    left_data = np.load(left_npz_path)
    if "T_c2w_opencv" not in left_data:
        raise KeyError(f"{left_npz_path} does not contain T_c2w_opencv")
    T_left_raw = np.asarray(left_data["T_c2w_opencv"], dtype=np.float32)
    indices = np.asarray(
        left_data.get("selected_original_indices", experiment_summary["selected_original_indices"]),
        dtype=np.int64,
    )
    if len(T_left_raw) != len(indices):
        raise ValueError(f"Pose/index mismatch: {len(T_left_raw)} vs {len(indices)}")

    T_left_scaled = scale_translations(T_left_raw, stereo_scale)
    T_gt_all = eval_mod.load_gt_trajectory(
        abs_path(args.gt_pose_file),
        args.gt_convention,
        args.gt_quat_order,
        args.gt_matrix_convention,
    )
    T_gt = T_gt_all[indices].astype(np.float32)
    names = load_left_names(experiment_dir, indices)

    T_raw_eval, raw_ate, raw_rpe_rows, raw_rpe, raw_R, raw_t = evaluate_se3(
        eval_mod, T_left_raw, T_gt, args.rpe_delta
    )
    T_scaled_eval, scaled_ate, scaled_rpe_rows, scaled_rpe, scaled_R, scaled_t = evaluate_se3(
        eval_mod, T_left_scaled, T_gt, args.rpe_delta
    )

    output_dir = experiment_dir / "pose_eval" / "left_only_stereo_baseline_scaled_se3"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_mod.write_pose_tum(
        output_dir / "trajectory_pred_aligned_c2w_opencv.txt",
        T_scaled_eval,
        names=names,
    )
    eval_mod.write_pose_tum(
        output_dir / "trajectory_gt_matched_c2w_opencv.txt",
        T_gt,
        names=names,
    )

    raw_errors = np.linalg.norm(T_raw_eval[:, :3, 3] - T_gt[:, :3, 3], axis=1)
    scaled_errors = np.linalg.norm(T_scaled_eval[:, :3, 3] - T_gt[:, :3, 3], axis=1)
    per_frame_rows = []
    for i in range(len(indices)):
        per_frame_rows.append(
            {
                "local_index": i,
                "original_index": int(indices[i]),
                "image_name": names[i],
                "raw_se3_error": float(raw_errors[i]),
                "stereo_scale_se3_error": float(scaled_errors[i]),
                "error_reduction": float(raw_errors[i] - scaled_errors[i]),
            }
        )
    save_csv(output_dir / "per_frame_errors.csv", per_frame_rows)
    save_csv(output_dir / "rpe_errors.csv", scaled_rpe_rows)
    np.savez_compressed(
        experiment_dir / "left_only_stereo_baseline_scaled_predictions.npz",
        T_c2w_opencv=T_left_scaled,
        T_c2w_opencv_se3_aligned=T_scaled_eval,
        selected_original_indices=indices,
        stereo_baseline_global_scale=np.asarray(stereo_scale),
    )

    raw_rmse = float(raw_ate["rmse"])
    scaled_rmse = float(scaled_ate["rmse"])
    improvement_abs = raw_rmse - scaled_rmse
    improvement_percent = 100.0 * improvement_abs / max(raw_rmse, 1e-12)

    upper = (
        experiment_summary.get("experiments", {})
        .get("left_only_gt_sim3_upper_bound", {})
        .get("ate", {})
        .get("rmse")
    )
    result = {
        "research_question": "Can the stereo-estimated global scale improve the original left-only ZipMap trajectory?",
        "method": {
            "trajectory_source": "left_only_predictions.npz",
            "scale_source": "stereo_baseline_global_scale from experiment_summary.json",
            "scale_value": stereo_scale,
            "pair_filtering": "none",
            "scale_aggregation": "mean of all predicted stereo baselines",
            "dynamic_scale": False,
            "evaluation_alignment": "SE3",
        },
        "left_only_raw_se3": {
            "R": raw_R.tolist(),
            "t": raw_t.tolist(),
            "ate": raw_ate,
            "rpe": raw_rpe,
        },
        "left_only_stereo_baseline_scaled_se3": {
            "R": scaled_R.tolist(),
            "t": scaled_t.tolist(),
            "ate": scaled_ate,
            "rpe": scaled_rpe,
        },
        "improvement": {
            "ate_rmse_absolute_reduction": improvement_abs,
            "ate_rmse_relative_reduction_percent": improvement_percent,
        },
        "left_only_gt_sim3_upper_bound_rmse": None if upper is None else float(upper),
        "remaining_rmse_gap_to_gt_sim3_upper_bound": None
        if upper is None
        else float(scaled_rmse - float(upper)),
        "selected_original_indices": [int(i) for i in indices],
    }
    save_json(output_dir / "summary.json", result)
    save_json(experiment_dir / "scale_transfer_summary.json", result)
    plot_comparison(output_dir, T_gt, T_raw_eval, T_scaled_eval, raw_errors, scaled_errors)

    print("[Done]")
    print(f"  stereo scale: {stereo_scale:.9f}")
    print(f"  left-only raw SE3 RMSE: {raw_rmse:.6f}")
    print(f"  left-only + stereo scale SE3 RMSE: {scaled_rmse:.6f}")
    print(f"  relative RMSE reduction: {improvement_percent:.2f}%")
    if upper is not None:
        print(f"  left-only GT Sim3 upper-bound RMSE: {float(upper):.6f}")
    print(f"  summary: {experiment_dir / 'scale_transfer_summary.json'}")


if __name__ == "__main__":
    main()
