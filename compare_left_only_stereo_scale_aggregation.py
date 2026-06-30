#!/usr/bin/env python3
"""
Experiment 3: compare two basic global stereo-scale aggregation rules.

No ZipMap inference is rerun. No stereo pair is filtered.
The only changed factor is how all predicted stereo baselines are aggregated:
  1) mean baseline:   scale = known_baseline / mean(predicted_baseline)
  2) median baseline: scale = known_baseline / median(predicted_baseline)

Example:
python compare_left_only_stereo_scale_aggregation.py \
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


def read_baselines(path: Path) -> np.ndarray:
    values = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values.append(float(row["predicted_baseline"]))
    if not values:
        raise ValueError(f"No predicted_baseline values found in {path}")
    return np.asarray(values, dtype=np.float64)


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
    return T_eval, ate, rpe


def main() -> None:
    p = argparse.ArgumentParser(description="Compare mean and median stereo baseline aggregation")
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--gt_pose_file", required=True)
    p.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    args = p.parse_args()

    repo = abs_path(args.zipmap_repo)
    exp_dir = abs_path(args.experiment_dir)
    eval_mod = import_module(
        "zipmap_eval_scale_aggregation",
        repo / "tools" / "evaluate_zipmap_pose.py",
    )

    summary = json.loads((exp_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    known_baseline = float(summary["baseline"]["known_baseline"])
    baselines = read_baselines(exp_dir / "stereo_baseline_per_pair.csv")

    mean_baseline = float(np.mean(baselines))
    median_baseline = float(np.median(baselines))
    scale_mean = known_baseline / mean_baseline
    scale_median = known_baseline / median_baseline

    left = np.load(exp_dir / "left_only_predictions.npz")
    T_left = np.asarray(left["T_c2w_opencv"], dtype=np.float32)
    indices = np.asarray(left["selected_original_indices"], dtype=np.int64)

    T_gt_all = eval_mod.load_gt_trajectory(
        abs_path(args.gt_pose_file),
        args.gt_convention,
        args.gt_quat_order,
        args.gt_matrix_convention,
    )
    T_gt = T_gt_all[indices].astype(np.float32)

    methods = {
        "mean_baseline": scale_mean,
        "median_baseline": scale_median,
    }
    results = {}
    out_dir = exp_dir / "pose_eval" / "left_only_scale_aggregation_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, scale in methods.items():
        T_scaled = scale_translations(T_left, scale)
        T_eval, ate, rpe = evaluate_se3(eval_mod, T_scaled, T_gt, args.rpe_delta)
        results[name] = {
            "predicted_baseline_aggregation": name,
            "predicted_baseline_value": mean_baseline if name == "mean_baseline" else median_baseline,
            "global_scale": float(scale),
            "ate": ate,
            "rpe": rpe,
        }
        eval_mod.write_pose_tum(
            out_dir / f"trajectory_{name}_se3_aligned.txt",
            T_eval,
            names=[f"{int(i):06d}" for i in indices],
        )

    upper_rmse = float(
        summary["experiments"]["left_only_gt_sim3_upper_bound"]["ate"]["rmse"]
    )
    best_name = min(results, key=lambda k: results[k]["ate"]["rmse"])
    output = {
        "research_question": "Does median baseline aggregation improve the transferred global scale over mean baseline aggregation?",
        "controlled_conditions": {
            "trajectory": "same saved left-only ZipMap trajectory",
            "stereo_pairs": int(len(baselines)),
            "pair_filtering": "none",
            "dynamic_scale": False,
            "evaluation_alignment": "SE3",
            "only_changed_factor": "mean versus median aggregation of all predicted baselines",
        },
        "known_baseline": known_baseline,
        "predicted_baseline_statistics": {
            "mean": mean_baseline,
            "median": median_baseline,
            "std": float(np.std(baselines)),
            "min": float(np.min(baselines)),
            "max": float(np.max(baselines)),
        },
        "results": results,
        "best_method": best_name,
        "left_only_gt_sim3_upper_bound_rmse": upper_rmse,
        "remaining_gap_to_upper_bound": {
            k: float(v["ate"]["rmse"] - upper_rmse) for k, v in results.items()
        },
    }
    save_json(out_dir / "summary.json", output)
    save_json(exp_dir / "scale_aggregation_comparison_summary.json", output)

    print("[Done]")
    for name, result in results.items():
        print(
            f"  {name}: scale={result['global_scale']:.9f}, "
            f"ATE_RMSE={result['ate']['rmse']:.6f}, "
            f"RPE_t_RMSE={result['rpe']['translation_rmse']:.6f}"
        )
    print(f"  best: {best_name}")
    print(f"  summary: {exp_dir / 'scale_aggregation_comparison_summary.json'}")


if __name__ == "__main__":
    main()
