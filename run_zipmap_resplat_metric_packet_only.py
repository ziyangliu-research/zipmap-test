#!/usr/bin/env python3
"""
Generate one ReSplat 3DGS packet per stereo time step using metric poses
reconstructed from a saved left-only ZipMap trajectory and stereo scale.

This script intentionally stops after packet generation:
  - no Gaussian fusion
  - no rendering or visualization
  - no map optimization or maintenance

Pose/scale separation
---------------------
- Final temporal pose comes only from the saved left-only ZipMap trajectory.
- Stereo ZipMap predictions are used only to provide causal scale estimates.
- Right-camera poses supplied to ReSplat are synthesized from the fixed stereo rig.

Supported scale modes
---------------------
1) frozen_init (default):
   Use the scale estimated from the first K stereo pairs for every relative
   translation increment. The first K frames are conceptually buffered until
   scale initialization completes.

2) cumulative_online:
   At frame t, use the scale estimated from the first t+1 stereo pairs only for
   the newest relative translation increment. Old poses are not rescaled.

Required cached inputs in --pose_experiment_dir
-----------------------------------------------
- left_only_predictions.npz
- prefix_only_inference_summary.json

Example: Frozen-10 packet generation
------------------------------------
python run_zipmap_resplat_metric_packet_only.py \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --pose_experiment_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_stereo_scale_P000_0_50 \
  --left_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --right_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_rcam_front \
  --work_dir /home/shiyo/Desktop/ZipMap/outputs/zipmap_resplat_packets_frozen10_P000_0_50 \
  --start_index 0 --end_index 50 \
  --scene_name P000 \
  --resplat_experiment tartanair_p000_ft \
  --scale_mode frozen_init \
  --freeze_after_pairs 10 \
  --resplat_packet_stage init \
  --device cuda:0

Example: Cumulative-online packet generation
--------------------------------------------
Use the same command with:
  --scale_mode cumulative_online
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


def abs_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    "zipmap_resplat_packet_only_base",
    Path(__file__).resolve().with_name(
        "run_zipmap_resplat_fusion_api_official_streaming_merged_init_fast.py"
    ),
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


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_causal_scales(summary_path: Path, expected_length: int) -> np.ndarray:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    by_k = {
        int(row["num_stereo_pairs"]): float(row["estimated_scale"])
        for row in data["results"]
    }
    missing = [k for k in range(1, expected_length + 1) if k not in by_k]
    if missing:
        preview = ",".join(str(k) for k in missing[:10])
        raise RuntimeError(
            f"Missing causal prefix scale(s): {preview}"
            f"{'...' if len(missing) > 10 else ''}. "
            "Run run_zipmap_stereo_prefix_only_scale_experiment.py for every prefix 1..N."
        )
    return np.asarray(
        [by_k[k] for k in range(1, expected_length + 1)], dtype=np.float64
    )


def integrate_scaled_relative_poses(
    T_raw_c2w: np.ndarray,
    scale_per_frame: np.ndarray,
) -> np.ndarray:
    """Scale each newest relative translation and compose a metric c2w trajectory."""
    T_raw = np.asarray(T_raw_c2w, dtype=np.float64)
    scales = np.asarray(scale_per_frame, dtype=np.float64)
    if len(T_raw) != len(scales):
        raise ValueError(f"Pose/scale length mismatch: {len(T_raw)} vs {len(scales)}")

    out = np.repeat(np.eye(4, dtype=np.float64)[None], len(T_raw), axis=0)
    out[0] = T_raw[0]
    for t in range(1, len(T_raw)):
        delta = np.linalg.inv(T_raw[t - 1]) @ T_raw[t]
        delta_metric = delta.copy()
        delta_metric[:3, 3] *= float(scales[t])
        out[t] = out[t - 1] @ delta_metric
    return out.astype(np.float32)


def write_pose_matrix_file(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for T in poses:
            f.write(" ".join(f"{float(v):.9f}" for v in T.reshape(-1)) + "\n")


def prepare_metric_trajectory(
    args: argparse.Namespace,
    selected_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pose_dir = abs_path(args.pose_experiment_dir)
    left_path = pose_dir / "left_only_predictions.npz"
    scale_path = pose_dir / "prefix_only_inference_summary.json"
    if not left_path.exists():
        raise FileNotFoundError(left_path)
    if not scale_path.exists():
        raise FileNotFoundError(scale_path)

    left = np.load(left_path)
    T_raw_all = np.asarray(left["T_c2w_opencv"], dtype=np.float32)
    stored_indices = np.asarray(left["selected_original_indices"], dtype=np.int64)
    requested = np.asarray(selected_indices, dtype=np.int64)

    if len(T_raw_all) != len(stored_indices):
        raise ValueError("Saved left-only pose/index length mismatch")
    if not np.array_equal(stored_indices, requested):
        raise ValueError(
            "The selected image indices do not match left_only_predictions.npz. "
            f"saved={stored_indices.tolist()}, requested={requested.tolist()}"
        )

    causal_scales = load_causal_scales(scale_path, len(T_raw_all))
    if args.scale_mode == "frozen_init":
        if not 1 <= args.freeze_after_pairs <= len(T_raw_all):
            raise ValueError(
                f"freeze_after_pairs must be within [1,{len(T_raw_all)}]"
            )
        frozen_scale = float(causal_scales[args.freeze_after_pairs - 1])
        scale_per_frame = np.full(len(T_raw_all), frozen_scale, dtype=np.float64)
        scale_summary = {
            "mode": "frozen_init",
            "freeze_after_pairs": int(args.freeze_after_pairs),
            "frozen_scale": frozen_scale,
            "first_frames_buffered": int(args.freeze_after_pairs),
        }
    else:
        scale_per_frame = causal_scales.copy()
        scale_summary = {
            "mode": "cumulative_online",
            "freeze_after_pairs": None,
            "frozen_scale": None,
            "retroactive_rescaling": False,
        }

    T_metric = integrate_scaled_relative_poses(T_raw_all, scale_per_frame)
    return T_raw_all, T_metric, scale_per_frame, scale_summary


@torch.no_grad()
def generate_packets_save_only(
    runtime: Any,
    selected: Any,
    T_metric: np.ndarray,
    T_raw: np.ndarray,
    scale_per_frame: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = runtime.model
    device = runtime.device
    stage_request = args.resplat_packet_stage
    packet_root = abs_path(args.work_dir) / args.packet_out_name

    stages = ["init", "final"] if stage_request == "both" else [stage_request]
    for stage in stages:
        (packet_root / stage).mkdir(parents=True, exist_ok=True)

    timing_rows: list[dict[str, Any]] = []
    packet_records: list[dict[str, Any]] = []
    stage_counts = {"init": 0, "final": 0}

    BASE.log(
        f"[Packet-only] ReSplat: {len(selected.indices)} stereo pair(s), "
        f"stage={stage_request}, scale_mode={args.scale_mode}"
    )

    for local_i, original_i in enumerate(selected.indices):
        scene_key = f"{args.scene_name}_{local_i:04d}"
        target_local = local_i + args.resplat_target_offset
        if target_local < 0 or target_local >= len(selected.indices):
            if args.drop_invalid_target_offset:
                continue
            target_local = max(0, min(target_local, len(selected.indices) - 1))
        target_original_i = selected.indices[target_local]

        t0 = time.perf_counter()
        batch_cpu = BASE.make_resplat_batch_for_frame(
            runtime=runtime,
            scene_key=scene_key,
            frame_index=original_i,
            left_path=selected.left_paths[local_i],
            right_path=selected.right_paths[local_i],
            T_left_c2w_cv=T_metric[local_i],
            target_camera=args.resplat_target_camera,
            target_offset_frame_index=target_original_i,
            target_left_path=selected.left_paths[target_local],
            target_right_path=selected.right_paths[target_local],
            target_T_left_c2w_cv=T_metric[target_local],
            stereo_baseline=args.stereo_baseline,
        )
        batch = BASE.tensor_to_device(batch_cpu, device)
        batch = model.data_shim(batch)
        cuda_sync(device)
        preprocess_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        enc_out = model.encoder(
            batch["context"],
            0,
            deterministic=False,
            visualization_dump=None,
        )
        cuda_sync(device)
        encoder_sec = time.perf_counter() - t1

        if isinstance(enc_out, dict):
            condition_features = enc_out.get("condition_features", None)
            gaussians_init = enc_out["gaussians"]
        else:
            condition_features = None
            gaussians_init = enc_out

        saved_paths: dict[str, str] = {}
        refine_sec = 0.0

        if stage_request in {"init", "both"}:
            pkt_init = BASE.make_packet_from_gaussians(
                gaussians_init, batch, model.decoder, stage="init"
            )
            pkt_init.update(
                {
                    "frame_local_index": int(local_i),
                    "frame_original_index": int(original_i),
                    "raw_left_pose_c2w_opencv": torch.from_numpy(T_raw[local_i]).float(),
                    "metric_left_pose_c2w_opencv": torch.from_numpy(T_metric[local_i]).float(),
                    "scale_mode": str(args.scale_mode),
                    "scale_used_for_increment": float(scale_per_frame[local_i]),
                    "fusion_applied": False,
                }
            )
            out_path = packet_root / "init" / f"{scene_key}.pt"
            torch.save(pkt_init, out_path)
            saved_paths["init"] = str(out_path)
            stage_counts["init"] += 1
            del pkt_init

        if stage_request in {"final", "both"}:
            gaussians_final = gaussians_init
            if getattr(model.encoder.cfg, "num_refine", 0) > 0:
                if condition_features is None:
                    raise RuntimeError(
                        "encoder.num_refine > 0 but condition_features are unavailable"
                    )
                t_refine = time.perf_counter()
                refine_out = model.encoder.forward_update(
                    batch["context"],
                    None,
                    condition_features,
                    gaussians_init,
                    model.decoder,
                    batch.get("context_remain", None),
                )
                cuda_sync(device)
                refine_sec = time.perf_counter() - t_refine
                if len(refine_out["gaussian"]) == 0:
                    raise RuntimeError("forward_update returned no Gaussian output")
                gaussians_final = refine_out["gaussian"][-1]

            pkt_final = BASE.make_packet_from_gaussians(
                gaussians_final, batch, model.decoder, stage="final"
            )
            pkt_final.update(
                {
                    "frame_local_index": int(local_i),
                    "frame_original_index": int(original_i),
                    "raw_left_pose_c2w_opencv": torch.from_numpy(T_raw[local_i]).float(),
                    "metric_left_pose_c2w_opencv": torch.from_numpy(T_metric[local_i]).float(),
                    "scale_mode": str(args.scale_mode),
                    "scale_used_for_increment": float(scale_per_frame[local_i]),
                    "fusion_applied": False,
                }
            )
            out_path = packet_root / "final" / f"{scene_key}.pt"
            torch.save(pkt_final, out_path)
            saved_paths["final"] = str(out_path)
            stage_counts["final"] += 1
            del pkt_final
            if "refine_out" in locals():
                del refine_out

        total_sec = time.perf_counter() - t0
        timing_rows.append(
            {
                "frame_local_index": int(local_i),
                "frame_original_index": int(original_i),
                "preprocess_and_shim_sec": float(preprocess_sec),
                "encoder_sec": float(encoder_sec),
                "refine_sec": float(refine_sec),
                "total_packet_sec": float(total_sec),
            }
        )
        packet_records.append(
            {
                "scene": scene_key,
                "frame_local_index": int(local_i),
                "frame_original_index": int(original_i),
                "left_image": str(selected.left_paths[local_i]),
                "right_image": str(selected.right_paths[local_i]),
                "scale_used_for_increment": float(scale_per_frame[local_i]),
                "saved_packets": saved_paths,
            }
        )

        del batch, batch_cpu, enc_out, gaussians_init
        if "gaussians_final" in locals():
            del gaussians_final
        if (local_i + 1) % max(1, args.empty_cache_every) == 0:
            torch.cuda.empty_cache()

        if (local_i + 1) % max(1, args.log_every) == 0 or local_i + 1 == len(selected.indices):
            BASE.log(
                f"  generated {local_i + 1}/{len(selected.indices)} packet(s), "
                f"last={total_sec:.3f}s"
            )

    save_csv(packet_root / "packet_timing.csv", timing_rows)
    manifest = {
        "pipeline": "ZipMap metric pose -> ReSplat packet only",
        "scale_mode": args.scale_mode,
        "packet_stage_requested": stage_request,
        "num_init_packets": stage_counts["init"],
        "num_final_packets": stage_counts["final"],
        "selected_original_indices": [int(i) for i in selected.indices],
        "stereo_baseline": float(args.stereo_baseline),
        "fusion_performed": False,
        "visualization_performed": False,
        "packets": packet_records,
    }
    save_json(packet_root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate metric ReSplat packets without fusion or visualization"
    )
    p.add_argument("--resplat_repo", required=True)
    p.add_argument("--pose_experiment_dir", required=True)
    p.add_argument("--left_dir", required=True)
    p.add_argument("--right_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--scene_name", required=True)

    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--recursive", action="store_true")

    p.add_argument(
        "--scale_mode",
        choices=["frozen_init", "cumulative_online"],
        default="frozen_init",
    )
    p.add_argument("--freeze_after_pairs", type=int, default=10)
    p.add_argument("--stereo_baseline", type=float, default=0.25000006)

    p.add_argument("--resplat_experiment", required=True)
    p.add_argument("--resplat_checkpoint", default=None)
    p.add_argument("--resplat_override", action="append", default=[])
    p.add_argument(
        "--resplat_packet_stage",
        choices=["init", "final", "both"],
        default="init",
    )
    p.add_argument(
        "--resplat_target_camera",
        choices=["left", "right", "both"],
        default="left",
    )
    p.add_argument("--resplat_target_offset", type=int, default=0)
    p.add_argument("--drop_invalid_target_offset", action="store_true")
    p.add_argument("--resplat_out_name", default="resplat_runtime")
    p.add_argument("--packet_out_name", default="packets")

    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--empty_cache_every", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir = str(abs_path(args.work_dir))
    work_dir = abs_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    selected = BASE.select_stereo_frames(
        left_dir=abs_path(args.left_dir),
        right_dir=abs_path(args.right_dir),
        start_index=args.start_index,
        end_index=args.end_index,
        stride=args.stride,
        num_frames=args.num_frames,
        recursive=args.recursive,
    )

    T_raw, T_metric, scale_per_frame, scale_summary = prepare_metric_trajectory(
        args, selected.indices
    )

    pose_dir = work_dir / "metric_pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pose_dir / "metric_pose_sequence.npz",
        T_raw_c2w_opencv=T_raw,
        T_metric_c2w_opencv=T_metric,
        scale_per_frame=scale_per_frame,
        selected_original_indices=np.asarray(selected.indices, dtype=np.int64),
    )
    write_pose_matrix_file(pose_dir / "raw_pose_c2w_opencv.txt", T_raw)
    write_pose_matrix_file(pose_dir / "metric_pose_c2w_opencv.txt", T_metric)
    save_json(pose_dir / "scale_summary.json", scale_summary)

    runtime = BASE.load_resplat_runtime(args)
    manifest = generate_packets_save_only(
        runtime=runtime,
        selected=selected,
        T_metric=T_metric,
        T_raw=T_raw,
        scale_per_frame=scale_per_frame,
        args=args,
    )

    save_json(
        work_dir / "run_summary.json",
        {
            "scale": scale_summary,
            "num_frames": len(selected.indices),
            "packet_manifest": str(work_dir / args.packet_out_name / "manifest.json"),
            "fusion_performed": False,
            "visualization_performed": False,
            "packet_counts": {
                "init": manifest["num_init_packets"],
                "final": manifest["num_final_packets"],
            },
        },
    )
    print(f"[Done] packets: {work_dir / args.packet_out_name}")


if __name__ == "__main__":
    main()
