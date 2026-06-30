#!/usr/bin/env python3
"""
Compare two incremental metric-scale strategies using the same saved left-only
ZipMap raw trajectory and causally estimated stereo prefix scales.

Strategies
----------
1) frozen_init:
   Buffer the first K frames, estimate one scale from the first K stereo pairs,
   then use that fixed scale for every relative translation increment.

2) cumulative_online:
   At frame t, use the scale estimated from stereo pairs [0, ..., t].
   Only the newest relative translation increment is scaled by the current scale.
   Earlier poses are not retroactively rescaled.

The stereo branch is used only for scale. The final trajectory always uses the
saved left-only ZipMap poses for rotation and raw temporal motion.

Prerequisite
------------
run_zipmap_stereo_prefix_only_scale_experiment.py must have been run for every
prefix length 1..N, for example:

  --prefix_lengths $(seq -s, 1 50)

Example
-------
python evaluate_incremental_scale_update_strategies.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --experiment_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_stereo_scale_P000_0_50 \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
  --freeze_after_pairs 10
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
    rpe_rows, rpe = eval_mod.compute_rpe(
        T_eval.astype(np.float64), T_gt.astype(np.float64), delta=delta
    )
    return T_eval, ate, rpe_rows, rpe


def integrate_scaled_relative_poses(
    T_raw_c2w: np.ndarray,
    scale_per_frame: np.ndarray,
) -> np.ndarray:
    """
    Build a metric c2w trajectory by scaling only each newest relative
    translation increment.

    scale_per_frame[t] is used for the increment from t-1 to t. Entry 0 is
    retained only for logging and is not used in a motion increment.
    """
    T_raw = np.asarray(T_raw_c2w, dtype=np.float64)
    scales = np.asarray(scale_per_frame, dtype=np.float64)
    if len(T_raw) != len(scales):
        raise ValueError(f"Pose/scale length mismatch: {len(T_raw)} vs {len(scales)}")

    out = np.repeat(np.eye(4, dtype=np.float64)[None], len(T_raw), axis=0)
    out[0] = T_raw[0]
    for t in range(1, len(T_raw)):
        delta = np.linalg.inv(T_raw[t - 1]) @ T_raw[t]
        delta_scaled = delta.copy()
        delta_scaled[:3, 3] *= float(scales[t])
        out[t] = out[t - 1] @ delta_scaled
    return out.astype(np.float32)


def load_prefix_scales(path: Path, n: int) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_k = {int(r["num_stereo_pairs"]): float(r["estimated_scale"]) for r in data["results"]}
    missing = [k for k in range(1, n + 1) if k not in by_k]
    if missing:
        preview = ",".join(str(k) for k in missing[:10])
        raise RuntimeError(
            "prefix_only_inference_summary.json does not contain every causal prefix "
            f"1..{n}. Missing: {preview}{'...' if len(missing) > 10 else ''}. "
            "Rerun run_zipmap_stereo_prefix_only_scale_experiment.py with "
            f"--prefix_lengths $(seq -s, 1 {n})"
        )
    return np.asarray([by_k[k] for k in range(1, n + 1)], dtype=np.float64)


def plot_results(out_dir: Path, scales: np.ndarray, frozen_scale: float, results: dict[str, Any]) -> None:
    if plt is None:
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(scales) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, scales, label="Cumulative online scale")
    plt.axhline(frozen_scale, linestyle="--", label="Frozen-10 scale")
    plt.xlabel("Number of arrived stereo pairs")
    plt.ylabel("Scale")
    plt.title("Scale used by the two incremental strategies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "scale_sequence.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    for name, item in results.items():
        T = np.asarray(item["T_eval"], dtype=np.float64)
        plt.plot(T[:, 0, 3], T[:, 2, 3], label=name)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Incremental scale update strategy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "trajectory_comparison_xz.png", dpi=160)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Compare frozen and cumulative online scale updates")
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--gt_pose_file", required=True)
    p.add_argument("--freeze_after_pairs", type=int, default=10)
    p.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    args = p.parse_args()

    repo = abs_path(args.zipmap_repo)
    exp_dir = abs_path(args.experiment_dir)
    eval_mod = import_module(
        "zipmap_eval_incremental_scale_strategies",
        repo / "tools" / "evaluate_zipmap_pose.py",
    )

    left = np.load(exp_dir / "left_only_predictions.npz")
    T_raw = np.asarray(left["T_c2w_opencv"], dtype=np.float32)
    indices = np.asarray(left["selected_original_indices"], dtype=np.int64)
    n = len(T_raw)
    if not 1 <= args.freeze_after_pairs <= n:
        raise ValueError(f"freeze_after_pairs must be within [1,{n}]")

    causal_scales = load_prefix_scales(exp_dir / "prefix_only_inference_summary.json", n)
    frozen_scale = float(causal_scales[args.freeze_after_pairs - 1])

    # Strategy A: the first K frames are buffered. After scale initialization,
    # all relative increments, including the buffered prefix, use the same scale.
    frozen_scales = np.full(n, frozen_scale, dtype=np.float64)

    # Strategy B: at frame t, use the scale available after observing stereo
    # pairs 0..t. Only the newest relative increment is affected.
    online_scales = causal_scales.copy()

    T_frozen = integrate_scaled_relative_poses(T_raw, frozen_scales)
    T_online = integrate_scaled_relative_poses(T_raw, online_scales)

    T_gt_all = eval_mod.load_gt_trajectory(
        abs_path(args.gt_pose_file),
        args.gt_convention,
        args.gt_quat_order,
        args.gt_matrix_convention,
    )
    T_gt = T_gt_all[indices].astype(np.float32)

    methods = {
        f"frozen_{args.freeze_after_pairs}": T_frozen,
        "cumulative_online": T_online,
    }
    result_for_plot: dict[str, Any] = {}
    output_methods: dict[str, Any] = {}
    out_dir = exp_dir / "pose_eval" / "incremental_scale_update_strategies"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, T_metric in methods.items():
        T_eval, ate, rpe_rows, rpe = evaluate_se3(eval_mod, T_metric, T_gt, args.rpe_delta)
        result_for_plot[name] = {"T_eval": T_eval.tolist()}
        output_methods[name] = {"ate": ate, "rpe": rpe}
        eval_mod.write_pose_tum(
            out_dir / f"trajectory_{name}_metric_raw.txt",
            T_metric,
            names=[f"{int(i):06d}" for i in indices],
        )
        eval_mod.write_pose_tum(
            out_dir / f"trajectory_{name}_se3_aligned.txt",
            T_eval,
            names=[f"{int(i):06d}" for i in indices],
        )
        save_csv(out_dir / f"rpe_{name}.csv", rpe_rows)

    scale_rows = []
    for t in range(n):
        scale_rows.append(
            {
                "frame_local_index": t,
                "original_index": int(indices[t]),
                "causal_prefix_scale": float(causal_scales[t]),
                "frozen_scale": frozen_scale,
                "scale_used_for_online_increment": None if t == 0 else float(online_scales[t]),
                "scale_used_for_frozen_increment": None if t == 0 else frozen_scale,
            }
        )
    save_csv(out_dir / "scale_sequence.csv", scale_rows)

    best = min(output_methods, key=lambda k: output_methods[k]["ate"]["rmse"])
    summary = {
        "research_question": "Should metric scale be frozen after a 10-pair initialization or updated at every new stereo observation?",
        "controlled_conditions": {
            "raw_pose_trajectory": "same saved left-only ZipMap trajectory",
            "stereo_pose_used_as_final_pose": False,
            "stereo_branch_role": "scale estimation only",
            "scale_aggregation": "same estimator already used by prefix-only experiment",
            "retroactive_map_rescaling": False,
            "metric_pose_update": "scale newest relative translation increment and compose",
        },
        "freeze_after_pairs": int(args.freeze_after_pairs),
        "frozen_scale": frozen_scale,
        "methods": output_methods,
        "best_method_by_ate_rmse": best,
        "note": "This is a pose-only strategy comparison using saved predictions. It does not yet run ReSplat or a persistent-state ZipMap stream.",
    }
    save_json(out_dir / "summary.json", summary)
    save_json(exp_dir / "incremental_scale_update_strategies_summary.json", summary)
    plot_results(out_dir, causal_scales, frozen_scale, result_for_plot)

    print("[Done]")
    print(f"  frozen scale at K={args.freeze_after_pairs}: {frozen_scale:.9f}")
    for name, item in output_methods.items():
        print(
            f"  {name}: ATE_RMSE={item['ate']['rmse']:.6f}, "
            f"RPE_t_RMSE={item['rpe']['translation_rmse']:.6f}, "
            f"RPE_R_deg={item['rpe']['rotation_deg_rmse']:.6f}"
        )
    print(f"  best: {best}")
    print(f"  summary: {exp_dir / 'incremental_scale_update_strategies_summary.json'}")


if __name__ == "__main__":
    main()
