#!/usr/bin/env python3
"""Generate per-frame ReSplat packets from an externally estimated metric trajectory.

This is the generic pose-frontend version of
``run_zipmap_resplat_metric_packet_only.py``. It reuses the validated ReSplat
packet-generation code, but it does not estimate, align, or rescale camera poses.
The input trajectory must already be expressed as OpenCV camera-to-world poses in
one common metric coordinate frame.

Expected pairwise ZipMap input (default keys):

    pairwise_pose_results.npz
      T_raw_accumulated_c2w_opencv  [N, 4, 4]
      selected_original_indices     [N]
      scale_per_pair                 [N-1]  (optional metadata)

For the stereo pairwise experiment, ``T_raw_accumulated_c2w_opencv`` already
contains baseline-corrected metric translations. Do not apply scale again here.
No packet fusion, global rendering, or map optimization is performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the validated ReSplat packet generation and self-render implementation.
OLD = load_module(
    "metric_packet_legacy",
    Path(__file__).resolve().with_name("run_zipmap_resplat_metric_packet_only.py"),
)
BASE = OLD.BASE


def load_external_trajectory(args: argparse.Namespace, requested_indices: list[int]):
    pose_path = abs_path(args.pose_npz)
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose NPZ not found: {pose_path}")

    with np.load(pose_path, allow_pickle=False) as data:
        keys = set(data.files)
        if args.pose_key not in keys:
            raise KeyError(
                f"Pose key '{args.pose_key}' not found in {pose_path}. "
                f"Available keys: {sorted(keys)}"
            )
        if args.index_key not in keys:
            raise KeyError(
                f"Index key '{args.index_key}' not found in {pose_path}. "
                f"Available keys: {sorted(keys)}"
            )

        T_metric = np.asarray(data[args.pose_key], dtype=np.float32)
        saved_indices = np.asarray(data[args.index_key], dtype=np.int64).reshape(-1)

        if args.raw_pose_key:
            if args.raw_pose_key not in keys:
                raise KeyError(
                    f"Raw pose key '{args.raw_pose_key}' not found in {pose_path}. "
                    f"Available keys: {sorted(keys)}"
                )
            T_raw = np.asarray(data[args.raw_pose_key], dtype=np.float32)
        else:
            # The pairwise stereo trajectory is already metric. Keep an identical
            # copy only for backward-compatible packet metadata.
            T_raw = T_metric.copy()

        scale_source = "not_available"
        if args.scale_key and args.scale_key in keys:
            scale_values = np.asarray(data[args.scale_key], dtype=np.float64).reshape(-1)
            if len(scale_values) == len(T_metric) - 1:
                scales = np.ones(len(T_metric), dtype=np.float64)
                scales[1:] = scale_values
                scale_source = f"{args.scale_key}: pair-to-frame (first frame=1.0)"
            elif len(scale_values) == len(T_metric):
                scales = scale_values
                scale_source = f"{args.scale_key}: per-frame"
            else:
                raise ValueError(
                    f"Scale key '{args.scale_key}' has length {len(scale_values)}, "
                    f"expected N-1={len(T_metric)-1} or N={len(T_metric)}"
                )
        else:
            scales = np.ones(len(T_metric), dtype=np.float64)

    if T_metric.ndim != 3 or T_metric.shape[1:] != (4, 4):
        raise ValueError(f"Expected metric poses [N,4,4], got {T_metric.shape}")
    if T_raw.shape != T_metric.shape:
        raise ValueError(
            f"Raw and metric pose shapes differ: raw={T_raw.shape}, metric={T_metric.shape}"
        )
    if len(saved_indices) != len(T_metric):
        raise ValueError(
            f"Pose/index length mismatch: poses={len(T_metric)}, indices={len(saved_indices)}"
        )
    if not np.isfinite(T_metric).all():
        raise ValueError("Metric trajectory contains NaN or Inf")

    requested = np.asarray(requested_indices, dtype=np.int64)
    if not np.array_equal(saved_indices, requested):
        raise ValueError(
            "Selected image indices differ from the trajectory NPZ. "
            f"NPZ={saved_indices.tolist()}, requested={requested.tolist()}"
        )

    first_pose_identity_error = float(
        np.linalg.norm(T_metric[0].astype(np.float64) - np.eye(4, dtype=np.float64))
    )
    summary = {
        "mode": "external_metric_trajectory",
        "pose_source_name": args.pose_source_name,
        "pose_npz": str(pose_path),
        "pose_key": args.pose_key,
        "raw_pose_key": args.raw_pose_key,
        "index_key": args.index_key,
        "scale_key": args.scale_key,
        "scale_metadata_source": scale_source,
        "pose_rescaling_applied": False,
        "gt_used": False,
        "coordinate_convention": "OpenCV camera-to-world (c2w/Twc)",
        "metric_scale_expected": True,
        "first_pose_identity_error_frobenius": first_pose_identity_error,
    }
    return T_raw, T_metric, scales, summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="External metric camera trajectory -> per-frame ReSplat packets"
    )
    p.add_argument("--resplat_repo", required=True)
    p.add_argument("--pose_npz", required=True)
    p.add_argument(
        "--pose_key",
        default="T_raw_accumulated_c2w_opencv",
        help="NPZ key containing the final metric OpenCV c2w trajectory.",
    )
    p.add_argument(
        "--raw_pose_key",
        default=None,
        help="Optional diagnostic raw-pose key. If omitted, the metric trajectory is reused.",
    )
    p.add_argument("--index_key", default="selected_original_indices")
    p.add_argument(
        "--scale_key",
        default="scale_per_pair",
        help="Optional scale metadata key. It is recorded only and is never reapplied.",
    )
    p.add_argument("--pose_source_name", default="pairwise_stereo_pose")

    p.add_argument("--left_dir", required=True)
    p.add_argument("--right_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--scene_name", required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--stereo_baseline", type=float, default=0.25000006)
    p.add_argument("--resplat_experiment", required=True)
    p.add_argument("--resplat_checkpoint", default=None)
    p.add_argument("--resplat_override", action="append", default=[])
    p.add_argument("--resplat_out_name", default="resplat_runtime")
    p.add_argument(
        "--resplat_packet_stage",
        choices=["init", "final", "both"],
        default="init",
    )
    p.add_argument("--refine_steps", default=None)
    p.add_argument("--refine_use_target", type=BASE.str2bool, default=False)
    p.add_argument(
        "--resplat_target_camera",
        choices=["left", "right", "both"],
        default="left",
    )
    p.add_argument("--resplat_target_offset", type=int, default=0)
    p.add_argument("--drop_invalid_target_offset", action="store_true")
    p.add_argument("--packet_out_name", default="packets")

    p.add_argument("--self_render_packets", action="store_true")
    p.add_argument("--self_render_out_name", default="packet_self_render")
    p.add_argument(
        "--self_render_link_original", type=BASE.str2bool, default=True
    )
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--empty_cache_every", type=int, default=1)
    return p


def main() -> None:
    args = parser().parse_args()
    args.work_dir = str(abs_path(args.work_dir))
    # OLD.add_packet_meta expects this field. No scale operation is performed.
    args.scale_mode = "external_metric_pose"

    work = abs_path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    selected = BASE.select_stereo_frames(
        abs_path(args.left_dir),
        abs_path(args.right_dir),
        args.start_index,
        args.end_index,
        args.stride,
        args.num_frames,
        args.recursive,
    )
    T_raw, T_metric, scales, pose_summary = load_external_trajectory(
        args, selected.indices
    )

    pose_dir = work / "metric_pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pose_dir / "metric_pose_sequence.npz",
        T_raw_c2w_opencv=T_raw,
        T_metric_c2w_opencv=T_metric,
        scale_per_frame=scales,
        selected_original_indices=np.asarray(selected.indices, dtype=np.int64),
    )
    OLD.write_pose_file(pose_dir / "raw_pose_c2w_opencv.txt", T_raw)
    OLD.write_pose_file(pose_dir / "metric_pose_c2w_opencv.txt", T_metric)
    OLD.save_json(pose_dir / "pose_source_summary.json", pose_summary)

    runtime = BASE.load_resplat_runtime(args)
    manifest = OLD.generate_packets(
        runtime, selected, T_raw, T_metric, scales, args
    )
    manifest["pipeline"] = "External metric pose -> ReSplat packet only"
    manifest["pose_source"] = pose_summary
    OLD.save_json(work / args.packet_out_name / "manifest.json", manifest)

    OLD.save_json(
        work / "run_summary.json",
        {
            "pipeline": "External metric pose -> ReSplat packet only",
            "pose_source": pose_summary,
            "num_frames": len(selected.indices),
            "packet_counts": manifest["packet_counts"],
            "packet_manifest": str(work / args.packet_out_name / "manifest.json"),
            "fusion_performed": False,
            "self_render_performed": bool(args.self_render_packets),
            "self_render_summary": manifest["self_render_summary"],
            "charts_generated": False,
        },
    )
    print(f"[Done] metric poses: {pose_dir / 'metric_pose_sequence.npz'}")
    print(f"[Done] packets: {work / args.packet_out_name}")
    if args.self_render_packets:
        print(
            f"[Done] self-render: "
            f"{work / args.self_render_out_name / 'summary.json'}"
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
