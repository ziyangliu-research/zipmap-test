#!/usr/bin/env python3
"""One-command pipeline: stereo images -> pairwise metric pose -> ReSplat packets.

This lightweight orchestrator runs the validated pose and packet stages in sequence.
The intermediate pose NPZ is kept under work_dir/pairwise_pose for reproducibility,
but the user only needs to run this single script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def run(cmd: list[str], stage: str) -> float:
    print(f"\n[{stage}] {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    sec = time.perf_counter() - t0
    print(f"[{stage}] completed in {sec:.3f}s", flush=True)
    return sec


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stereo pairwise ZipMap -> metric trajectory -> ReSplat packets"
    )
    p.add_argument("--zipmap_repo", required=True)
    p.add_argument("--zipmap_ckpt", required=True)
    p.add_argument("--resplat_repo", required=True)
    p.add_argument("--left_dir", required=True)
    p.add_argument("--right_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--scene_name", required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--stereo_baseline", type=float, default=0.25000006)
    p.add_argument(
        "--scale_aggregation",
        choices=["mean", "median", "first", "second"],
        default="mean",
    )
    p.add_argument("--affine_invariant", default="true")
    p.add_argument("--pose_only_heads", default="true")
    p.add_argument("--ema", default="false")
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    p.add_argument("--gt_pose_file", default=None)
    p.add_argument("--gt_convention", default="resplat_tartanair_pose")
    p.add_argument("--gt_quat_order", default="xyzw")
    p.add_argument("--gt_matrix_convention", default="c2w")
    p.add_argument("--rpe_delta", type=int, default=1)
    p.add_argument("--resplat_experiment", required=True)
    p.add_argument("--resplat_checkpoint", default=None)
    p.add_argument("--resplat_override", action="append", default=[])
    p.add_argument("--resplat_packet_stage", choices=["init", "final", "both"], default="init")
    p.add_argument("--refine_steps", default="0")
    p.add_argument("--refine_use_target", default="false")
    p.add_argument("--resplat_target_camera", choices=["left", "right", "both"], default="left")
    p.add_argument("--resplat_target_offset", type=int, default=0)
    p.add_argument("--packet_out_name", default="packets")
    p.add_argument("--self_render_packets", action="store_true")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    return p


def append_opt(cmd: list[str], name: str, value) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parent
    work = abs_path(args.work_dir)
    pose_dir = work / "pairwise_pose"
    work.mkdir(parents=True, exist_ok=True)

    pose_cmd = [
        sys.executable,
        str(root / "run_zipmap_pairwise_pose_eval.py"),
        "--zipmap_repo", str(abs_path(args.zipmap_repo)),
        "--zipmap_ckpt", str(abs_path(args.zipmap_ckpt)),
        "--left_dir", str(abs_path(args.left_dir)),
        "--right_dir", str(abs_path(args.right_dir)),
        "--output_dir", str(pose_dir),
        "--mode", "stereo_pair",
        "--start_index", str(args.start_index),
        "--stride", str(args.stride),
        "--stereo_baseline", str(args.stereo_baseline),
        "--scale_aggregation", args.scale_aggregation,
        "--affine_invariant", args.affine_invariant,
        "--pose_only_heads", args.pose_only_heads,
        "--ema", args.ema,
        "--target_size", str(args.target_size),
        "--preprocess_mode", args.preprocess_mode,
        "--device", args.device,
    ]
    append_opt(pose_cmd, "--end_index", args.end_index)
    append_opt(pose_cmd, "--max_frames", args.max_frames)
    if args.gt_pose_file:
        pose_cmd += [
            "--gt_pose_file", str(abs_path(args.gt_pose_file)),
            "--gt_convention", args.gt_convention,
            "--gt_quat_order", args.gt_quat_order,
            "--gt_matrix_convention", args.gt_matrix_convention,
            "--rpe_delta", str(args.rpe_delta),
        ]

    pose_sec = run(pose_cmd, "1/2 Pairwise metric pose")

    packet_cmd = [
        sys.executable,
        str(root / "run_pose_resplat_metric_packet_only.py"),
        "--resplat_repo", str(abs_path(args.resplat_repo)),
        "--pose_npz", str(pose_dir / "pairwise_pose_results.npz"),
        "--pose_source_name", "pairwise_stereo_zipmap",
        "--left_dir", str(abs_path(args.left_dir)),
        "--right_dir", str(abs_path(args.right_dir)),
        "--work_dir", str(work),
        "--scene_name", args.scene_name,
        "--start_index", str(args.start_index),
        "--stride", str(args.stride),
        "--stereo_baseline", str(args.stereo_baseline),
        "--resplat_experiment", args.resplat_experiment,
        "--resplat_packet_stage", args.resplat_packet_stage,
        "--refine_steps", args.refine_steps,
        "--refine_use_target", args.refine_use_target,
        "--resplat_target_camera", args.resplat_target_camera,
        "--resplat_target_offset", str(args.resplat_target_offset),
        "--packet_out_name", args.packet_out_name,
        "--device", args.device,
    ]
    append_opt(packet_cmd, "--end_index", args.end_index)
    append_opt(packet_cmd, "--num_frames", args.max_frames)
    append_opt(packet_cmd, "--resplat_checkpoint", args.resplat_checkpoint)
    append_opt(packet_cmd, "--fx", args.fx)
    append_opt(packet_cmd, "--fy", args.fy)
    append_opt(packet_cmd, "--cx", args.cx)
    append_opt(packet_cmd, "--cy", args.cy)
    for override in args.resplat_override:
        packet_cmd += ["--resplat_override", override]
    if args.self_render_packets:
        packet_cmd.append("--self_render_packets")

    packet_sec = run(packet_cmd, "2/2 ReSplat packets")

    summary = {
        "pipeline": "stereo images -> pairwise metric pose -> ReSplat packets",
        "pose_stage_sec": pose_sec,
        "packet_stage_sec": packet_sec,
        "total_sec": pose_sec + packet_sec,
        "pose_output": str(pose_dir / "pairwise_pose_results.npz"),
        "packet_output": str(work / args.packet_out_name),
        "gt_usage": (
            "evaluation only; not used for pose accumulation, metric scale, or packet generation"
            if args.gt_pose_file else "not provided"
        ),
    }
    (work / "combined_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[Done] packets: {work / args.packet_out_name}")
    print(f"[Done] summary: {work / 'combined_pipeline_summary.json'}")


if __name__ == "__main__":
    main()
