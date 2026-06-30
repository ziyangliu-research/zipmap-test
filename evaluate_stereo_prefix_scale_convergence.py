#!/usr/bin/env python3
"""
Experiment 4: evaluate causal prefix convergence of stereo-baseline scale.

This script does not rerun ZipMap. It uses the saved per-pair stereo baselines
and the saved left-only ZipMap trajectory from Experiment 1.

For each prefix length k = 1..N:
  scale_k = known_baseline / median(predicted_baseline[0:k])

It reports two complementary diagnostics:
  1) full_trajectory_*: apply scale_k to the same full left-only trajectory.
     This isolates only the quality of the scale estimate.
  2) prefix_trajectory_*: apply scale_k and evaluate only frames [0:k].
     This approximates what would be available causally at time k.
     ATE/RPE are reported only when enough poses exist.

No pair filtering, no dynamic scale correction, and no future stereo pair is
used when computing scale_k.

Example:
python evaluate_stereo_prefix_scale_convergence.py \
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


def abs_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()


def import_module(name: str, path: Path):
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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_baseline_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["predicted_baseline"] = float(raw["predicted_baseline"])
            if "original_index" in raw and raw["original_index"] != "":
                row["original_index"] = int(raw["original_index"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No stereo baselines found in {path}")
    return rows


def scale_translations(T: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(T, dtype=np.float64).copy()
    p0 = out[0, :3, 3].copy()
    out[:, :3, 3] = p0 + float(scale) * (out[:, :3, 3] - p0)
    return out.astype(np.float32)


def evaluate_se3(eval_mod, T_pred: np.ndarray, T_gt: np.ndarray, delta: int):
    if len(T_pred) < 3:
        return None
    _, R, t = eval_mod.umeyama_alignment(
        T_pred[:, :3, 3].astype(np.float64),
        T_gt[:, :3, 3].astype(np.float64),
        with_scale=False,
    )
    T_eval = eval_mod.apply_similarity_to_poses(
        T_pred.astype(np.float64), 1.0, R, t
    ).astype(np.float32)
    ate = eval_mod.compute_ate(T_eval.astype(np.float64), T_gt.astype(np.float64))
    rpe_rows, rpe = eval_mod.compute_rpe(
        T_eval.astype(np.float64), T_gt.astype(np.float64), delta=delta
    )
    return {
        "T_eval": T_eval,
        "ate": ate,
        "rpe": rpe,
        "rpe_rows": rpe_rows,
    }


def parse_checkpoints(text: str, n: int) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k < 1:
            raise ValueError(f"Checkpoint must be >= 1, got {k}")
        if k <= n:
            values.append(k)
    if n not in values:
        values.append(n)
    return sorted(set(values))


def make_plots(
    out_dir: Path,
    rows: list[dict[str, Any]],
    final_scale: float,
    gt_scale: float,
) -> None:
    if plt is None:
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    k = np.asarray([r["num_stereo_pairs"] for r in rows], dtype=np.int64)
    scale = np.asarray([r["estimated_scale"] for r in rows], dtype=np.float64)
    full_ate = np.asarray([r["full_trajectory_ate_rmse"] for r in rows], dtype=np.float64)
    prefix_ate = np.asarray([
        np.nan if r.get("prefix_trajectory_ate_rmse") is None else r["prefix_trajectory_ate_rmse"]
        for r in rows
    ], dtype=np.float64)

    plt.figure(figsize=(10, 4))
    plt.plot(k, scale, label="Prefix median scale")
    plt.axhline(final_scale, linestyle="--", label="50-pair median scale")
    plt.axhline(gt_scale, linestyle=":", label="GT Sim3 reference scale")
    plt.xlabel("Number of stereo pairs used")
    plt.ylabel("Estimated global scale")
    plt.title("Causal prefix scale convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "prefix_scale_convergence.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(k, full_ate)
    plt.xlabel("Number of stereo pairs used")
    plt.ylabel("Full-trajectory ATE RMSE [m]")
    plt.title("Scale quality using only the first k stereo pairs")
    plt.tight_layout()
    plt.savefig(plots / "prefix_scale_full_trajectory_ate.png", dpi=160)
    plt.close()

    valid = np.isfinite(prefix_ate)
    if np.any(valid):
        plt.figure(figsize=(10, 4))
        plt.plot(k[valid], prefix_ate[valid])
        plt.xlabel("Number of stereo pairs / available left poses")
        plt.ylabel("Prefix-trajectory ATE RMSE [m]")
        plt.title("Causal prefix trajectory evaluation")
        plt.tight_layout()
        plt.savefig(plots / "causal_prefix_trajectory_ate.png", dpi=160)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate stereo prefix scale convergence")
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--gt_pose_file", required=True)
    p.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    p.add_argument(
        "--report_checkpoints",
        default="1,2,3,5,10,15,20,30,40,50",
        help="Comma-separated prefix lengths highlighted in summary.json",
    )
    args = p.parse_args()

    repo = abs_path(args.zipmap_repo)
    exp_dir = abs_path(args.experiment_dir)
    eval_mod = import_module(
        "zipmap_eval_prefix_scale",
        repo / "tools" / "evaluate_zipmap_pose.py",
    )

    summary = json.loads((exp_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    known_baseline = float(summary["baseline"]["known_baseline"])
    baseline_rows = read_baseline_rows(exp_dir / "stereo_baseline_per_pair.csv")
    baselines = np.asarray([r["predicted_baseline"] for r in baseline_rows], dtype=np.float64)

    left = np.load(exp_dir / "left_only_predictions.npz")
    T_left = np.asarray(left["T_c2w_opencv"], dtype=np.float32)
    indices = np.asarray(left["selected_original_indices"], dtype=np.int64)
    n = min(len(baselines), len(T_left), len(indices))
    baselines = baselines[:n]
    T_left = T_left[:n]
    indices = indices[:n]

    T_gt_all = eval_mod.load_gt_trajectory(
        abs_path(args.gt_pose_file),
        args.gt_convention,
        args.gt_quat_order,
        args.gt_matrix_convention,
    )
    T_gt = T_gt_all[indices].astype(np.float32)

    gt_scale, _, _ = eval_mod.umeyama_alignment(
        T_left[:, :3, 3].astype(np.float64),
        T_gt[:, :3, 3].astype(np.float64),
        with_scale=True,
    )
    final_scale = float(known_baseline / np.median(baselines))
    final_full_eval = evaluate_se3(
        eval_mod,
        scale_translations(T_left, final_scale),
        T_gt,
        args.rpe_delta,
    )
    if final_full_eval is None:
        raise RuntimeError("Full trajectory evaluation unexpectedly failed")
    final_ate_rmse = float(final_full_eval["ate"]["rmse"])

    rows: list[dict[str, Any]] = []
    previous_scale = None
    for k in range(1, n + 1):
        prefix_baselines = baselines[:k]
        median_baseline = float(np.median(prefix_baselines))
        scale_k = float(known_baseline / max(median_baseline, 1e-12))

        full_scaled = scale_translations(T_left, scale_k)
        full_eval = evaluate_se3(eval_mod, full_scaled, T_gt, args.rpe_delta)
        if full_eval is None:
            raise RuntimeError("Full trajectory has fewer than 3 poses")

        prefix_eval = None
        if k >= 3:
            prefix_scaled = scale_translations(T_left[:k], scale_k)
            prefix_eval = evaluate_se3(
                eval_mod,
                prefix_scaled,
                T_gt[:k],
                args.rpe_delta,
            )

        row = {
            "num_stereo_pairs": k,
            "last_original_index": int(indices[k - 1]),
            "prefix_median_predicted_baseline": median_baseline,
            "estimated_scale": scale_k,
            "relative_scale_error_to_50_pair_percent": 100.0 * abs(scale_k - final_scale) / max(abs(final_scale), 1e-12),
            "relative_scale_error_to_gt_sim3_percent": 100.0 * abs(scale_k - gt_scale) / max(abs(gt_scale), 1e-12),
            "relative_scale_change_from_previous_percent": None if previous_scale is None else 100.0 * abs(scale_k - previous_scale) / max(abs(previous_scale), 1e-12),
            "full_trajectory_ate_rmse": float(full_eval["ate"]["rmse"]),
            "full_trajectory_rpe_translation_rmse": float(full_eval["rpe"].get("translation_rmse", np.nan)),
            "full_trajectory_ate_gap_to_50_pair": float(full_eval["ate"]["rmse"] - final_ate_rmse),
            "prefix_trajectory_ate_rmse": None if prefix_eval is None else float(prefix_eval["ate"]["rmse"]),
            "prefix_trajectory_rpe_translation_rmse": None if prefix_eval is None else float(prefix_eval["rpe"].get("translation_rmse", np.nan)),
        }
        rows.append(row)
        previous_scale = scale_k

    out_dir = exp_dir / "pose_eval" / "stereo_prefix_scale_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(out_dir / "prefix_scale_all_k.csv", rows)

    checkpoints = parse_checkpoints(args.report_checkpoints, n)
    checkpoint_rows = [rows[k - 1] for k in checkpoints]
    save_csv(out_dir / "prefix_scale_checkpoints.csv", checkpoint_rows)

    result = {
        "research_question": "How many causally available stereo pairs are needed before the median global scale becomes stable and effective?",
        "method": {
            "scale_at_time_k": "known_baseline / median(first k predicted stereo baselines)",
            "pair_filtering": "none",
            "future_stereo_pairs_used": False,
            "dynamic_scale_correction": False,
            "trajectory_source": "same saved left-only ZipMap trajectory",
            "full_trajectory_evaluation_role": "isolates scale-estimation quality",
            "prefix_trajectory_evaluation_role": "approximates causal accuracy available at time k",
        },
        "num_available_pairs": n,
        "known_baseline": known_baseline,
        "reference": {
            "all_pair_median_scale": final_scale,
            "all_pair_median_full_trajectory_ate_rmse": final_ate_rmse,
            "gt_sim3_reference_scale": float(gt_scale),
            "left_only_gt_sim3_upper_bound_ate_rmse": float(
                summary["experiments"]["left_only_gt_sim3_upper_bound"]["ate"]["rmse"]
            ),
        },
        "reported_checkpoints": checkpoints,
        "checkpoint_results": checkpoint_rows,
        "all_k_csv": str(out_dir / "prefix_scale_all_k.csv"),
    }
    save_json(out_dir / "summary.json", result)
    save_json(exp_dir / "prefix_scale_convergence_summary.json", result)
    make_plots(out_dir, rows, final_scale, float(gt_scale))

    print("[Done]")
    print(f"  all-pair median scale: {final_scale:.9f}")
    print(f"  GT Sim3 reference scale: {float(gt_scale):.9f}")
    print("  checkpoints:")
    for row in checkpoint_rows:
        print(
            f"    k={row['num_stereo_pairs']:2d}: "
            f"scale={row['estimated_scale']:.9f}, "
            f"full_ATE={row['full_trajectory_ate_rmse']:.6f}, "
            f"prefix_ATE={row['prefix_trajectory_ate_rmse']}"
        )
    print(f"  summary: {exp_dir / 'prefix_scale_convergence_summary.json'}")


if __name__ == "__main__":
    main()
