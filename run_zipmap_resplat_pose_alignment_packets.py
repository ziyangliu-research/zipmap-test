#!/usr/bin/env python3
"""Generate ReSplat packets under three ZipMap pose-alignment conditions.

This is an offline diagnostic wrapper around
run_zipmap_resplat_metric_packet_only.py. It changes only the left-camera pose
sequence passed to ReSplat; packet generation, optional refinement, optional
self-render, and packet serialization remain identical.

Modes
-----
predicted_metric:
    Frozen/online stereo metric pose, no GT alignment.
gt_se3:
    Apply one GT-derived global rigid transform (scale fixed to 1).
gt_sim3:
    Apply one GT-derived global similarity transform to the left trajectory.

For gt_sim3, the left-camera trajectory is similarity-aligned while the
physical left-right stereo baseline passed to ReSplat remains the configured
metric baseline. This isolates trajectory-scale error without changing the
local stereo packet's physical scale.
"""
from __future__ import annotations

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


SCRIPT_DIR = Path(__file__).resolve().parent
PACKET = load_module(
    "metric_packet_alignment_base",
    SCRIPT_DIR / "run_zipmap_resplat_metric_packet_only.py",
)
EVAL = load_module(
    "metric_packet_pose_eval",
    SCRIPT_DIR / "tools" / "evaluate_zipmap_pose.py",
)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    PACKET.save_csv(path, rows)


def apply_pose_alignment(
    args,
    T_metric: np.ndarray,
    selected_indices: list[int],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    mode = args.pose_alignment
    if mode == "predicted_metric":
        summary = {
            "mode": mode,
            "uses_gt": False,
            "offline_oracle_diagnostic": False,
            "alignment_scale": 1.0,
            "alignment_rotation": np.eye(3).tolist(),
            "alignment_translation": [0.0, 0.0, 0.0],
            "stereo_baseline_after_alignment": float(args.stereo_baseline),
        }
        return T_metric.copy(), None, summary, []

    if args.gt_pose_file is None:
        raise ValueError(f"--gt_pose_file is required for --pose_alignment {mode}")

    T_gt_all = EVAL.load_gt_trajectory(
        PACKET.abs_path(args.gt_pose_file),
        args.gt_convention,
        args.gt_quat_order,
        args.gt_matrix_convention,
    )
    indices = np.asarray(selected_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(T_gt_all)):
        raise IndexError(
            f"Selected indices exceed GT trajectory length {len(T_gt_all)}"
        )
    T_gt = T_gt_all[indices].astype(np.float64)

    with_scale = mode == "gt_sim3"
    scale, rotation, translation = EVAL.umeyama_alignment(
        T_metric[:, :3, 3].astype(np.float64),
        T_gt[:, :3, 3].astype(np.float64),
        with_scale=with_scale,
    )
    T_aligned = EVAL.apply_similarity_to_poses(
        T_metric.astype(np.float64), scale, rotation, translation
    ).astype(np.float32)

    ate = EVAL.compute_ate(T_aligned.astype(np.float64), T_gt)
    rpe_rows, rpe = EVAL.compute_rpe(
        T_aligned.astype(np.float64), T_gt, delta=args.rpe_delta
    )
    direct_errors = np.linalg.norm(
        T_aligned[:, :3, 3].astype(np.float64) - T_gt[:, :3, 3], axis=1
    )
    per_frame = [
        {
            "frame_local_index": int(local_i),
            "frame_original_index": int(original_i),
            "translation_error": float(direct_errors[local_i]),
        }
        for local_i, original_i in enumerate(selected_indices)
    ]

    summary = {
        "mode": mode,
        "uses_gt": True,
        "offline_oracle_diagnostic": True,
        "alignment_source": "matched left-camera centers over all selected frames",
        "alignment_scale": float(scale),
        "alignment_rotation": rotation.tolist(),
        "alignment_translation": translation.tolist(),
        "ate_after_applied_alignment": ate,
        "rpe_after_applied_alignment": rpe,
        "rpe_delta": int(args.rpe_delta),
        "selected_original_indices": [int(i) for i in selected_indices],
        "stereo_baseline_after_alignment": float(args.stereo_baseline),
        "sim3_baseline_policy": (
            "left trajectory translations receive residual Sim3 scale; fixed physical "
            "stereo baseline is not multiplied by the residual alignment scale"
            if mode == "gt_sim3"
            else "not applicable"
        ),
    }
    summary["rpe_rows"] = rpe_rows
    return T_aligned, T_gt.astype(np.float32), summary, per_frame


def install_packet_metadata_hook(
    args,
    T_metric_before: np.ndarray,
    T_packet_pose: np.ndarray,
    alignment_summary: dict[str, Any],
):
    """Add alignment provenance while preserving fields expected downstream."""

    def add_packet_meta(
        packet,
        runtime_args,
        local_i,
        original_i,
        T_raw,
        _T_metric_argument,
        scales,
        step,
    ):
        packet.update(
            {
                "frame_local_index": int(local_i),
                "frame_original_index": int(original_i),
                "raw_left_pose_c2w_opencv": torch.from_numpy(
                    T_raw[local_i]
                ).float(),
                "metric_left_pose_before_alignment_c2w_opencv": torch.from_numpy(
                    T_metric_before[local_i]
                ).float(),
                "metric_left_pose_c2w_opencv": torch.from_numpy(
                    T_packet_pose[local_i]
                ).float(),
                "left_pose_after_alignment_c2w_opencv": torch.from_numpy(
                    T_packet_pose[local_i]
                ).float(),
                "pose_alignment": str(args.pose_alignment),
                "alignment_scale": float(alignment_summary["alignment_scale"]),
                "alignment_rotation": torch.tensor(
                    alignment_summary["alignment_rotation"], dtype=torch.float32
                ),
                "alignment_translation": torch.tensor(
                    alignment_summary["alignment_translation"], dtype=torch.float32
                ),
                "scale_mode": runtime_args.scale_mode,
                "scale_used_for_increment": float(scales[local_i]),
                "refine_step": int(step),
                "refine_use_target": bool(runtime_args.refine_use_target),
                "fusion_applied": False,
            }
        )

    PACKET.add_packet_meta = add_packet_meta


def build_parser():
    parser = PACKET.parser()
    parser.description = (
        "Generate ReSplat packets using predicted metric, GT-SE3-aligned, or "
        "GT-Sim3-aligned ZipMap poses"
    )
    parser.add_argument(
        "--pose_alignment",
        choices=["predicted_metric", "gt_se3", "gt_sim3"],
        default="predicted_metric",
    )
    parser.add_argument("--gt_pose_file", default=None)
    parser.add_argument(
        "--gt_convention",
        choices=["opencv_c2w", "resplat_tartanair_pose"],
        default="resplat_tartanair_pose",
    )
    parser.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument(
        "--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w"
    )
    parser.add_argument("--rpe_delta", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir = str(PACKET.abs_path(args.work_dir))
    work = PACKET.abs_path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    selected = PACKET.BASE.select_stereo_frames(
        PACKET.abs_path(args.left_dir),
        PACKET.abs_path(args.right_dir),
        args.start_index,
        args.end_index,
        args.stride,
        args.num_frames,
        args.recursive,
    )

    T_raw, T_metric_before, scales, scale_summary = PACKET.prepare_poses(
        args, selected.indices
    )
    T_packet_pose, T_gt, alignment_summary, per_frame_errors = apply_pose_alignment(
        args, T_metric_before, selected.indices
    )

    install_packet_metadata_hook(
        args, T_metric_before, T_packet_pose, alignment_summary
    )

    pose_dir = work / "metric_pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "T_raw_c2w_opencv": T_raw,
        "T_metric_before_alignment_c2w_opencv": T_metric_before,
        "T_packet_pose_c2w_opencv": T_packet_pose,
        "scale_per_frame": scales,
        "selected_original_indices": np.asarray(selected.indices, dtype=np.int64),
    }
    if T_gt is not None:
        arrays["T_gt_matched_c2w_opencv"] = T_gt
    np.savez_compressed(pose_dir / "metric_pose_sequence.npz", **arrays)

    PACKET.write_pose_file(pose_dir / "raw_pose_c2w_opencv.txt", T_raw)
    PACKET.write_pose_file(
        pose_dir / "metric_pose_before_alignment_c2w_opencv.txt", T_metric_before
    )
    PACKET.write_pose_file(
        pose_dir / "packet_pose_after_alignment_c2w_opencv.txt", T_packet_pose
    )
    if T_gt is not None:
        PACKET.write_pose_file(pose_dir / "gt_matched_c2w_opencv.txt", T_gt)

    save_json(pose_dir / "scale_summary.json", scale_summary)
    save_json(pose_dir / "alignment_summary.json", alignment_summary)
    save_csv(pose_dir / "alignment_translation_errors.csv", per_frame_errors)
    rpe_rows = alignment_summary.pop("rpe_rows", [])
    if rpe_rows:
        save_csv(pose_dir / "alignment_rpe_errors.csv", rpe_rows)
        save_json(pose_dir / "alignment_summary.json", alignment_summary)

    runtime = PACKET.BASE.load_resplat_runtime(args)
    manifest = PACKET.generate_packets(
        runtime, selected, T_raw, T_packet_pose, scales, args
    )
    manifest.update(
        {
            "pose_alignment": args.pose_alignment,
            "alignment_summary": str(pose_dir / "alignment_summary.json"),
            "metric_pose_before_alignment": str(
                pose_dir / "metric_pose_before_alignment_c2w_opencv.txt"
            ),
            "packet_pose_after_alignment": str(
                pose_dir / "packet_pose_after_alignment_c2w_opencv.txt"
            ),
        }
    )
    save_json(work / args.packet_out_name / "manifest.json", manifest)

    run_summary = {
        "scale": scale_summary,
        "pose_alignment": args.pose_alignment,
        "alignment": alignment_summary,
        "num_frames": len(selected.indices),
        "packet_counts": manifest["packet_counts"],
        "packet_manifest": str(work / args.packet_out_name / "manifest.json"),
        "fusion_performed": False,
        "self_render_performed": bool(args.self_render_packets),
        "self_render_summary": manifest["self_render_summary"],
        "charts_generated": False,
    }
    save_json(work / "run_summary.json", run_summary)

    print(f"[Done] pose_alignment={args.pose_alignment}")
    print(f"[Done] packets: {work / args.packet_out_name}")
    print(f"[Done] alignment: {pose_dir / 'alignment_summary.json'}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
