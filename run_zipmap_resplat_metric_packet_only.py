#!/usr/bin/env python3
"""ZipMap metric poses -> per-frame ReSplat packets, with optional self-render.

No packet fusion, global rendering, or map optimization is performed.

Self-render mode renders every generated packet to its own left context view,
computes PSNR/SSIM against the exact preprocessed GT tensor, saves rendered and
processed-GT PNGs, and creates symlinks to the raw left images. No chart is made.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

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


# Reuse only validated ReSplat helpers; importing this file does not run main().
BASE = load_module(
    "metric_packet_base",
    Path(__file__).resolve().with_name(
        "run_gt_resplat_fusion_api_refine_compare_v5_skipfusion_selfrender.py"
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


def parse_refine_steps(spec: Optional[str]) -> Optional[list[int]]:
    if spec is None:
        return None
    values: list[int] = []
    for token in str(spec).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        step = int(token)
        if step < 0:
            raise ValueError("refine steps must be >= 0")
        if step not in values:
            values.append(step)
    if not values:
        raise ValueError("--refine_steps must not be empty")
    return sorted(values)


def stage_name(step: int) -> str:
    return f"refine_{step}"


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_scales(path: Path, n: int) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_k = {
        int(row["num_stereo_pairs"]): float(row["estimated_scale"])
        for row in data["results"]
    }
    missing = [k for k in range(1, n + 1) if k not in by_k]
    if missing:
        raise RuntimeError(f"Missing causal prefix scales: {missing[:10]}")
    return np.asarray([by_k[k] for k in range(1, n + 1)], dtype=np.float64)


def integrate_metric_poses(T_raw: np.ndarray, scales: np.ndarray) -> np.ndarray:
    T_raw = np.asarray(T_raw, dtype=np.float64)
    out = np.repeat(np.eye(4, dtype=np.float64)[None], len(T_raw), axis=0)
    out[0] = T_raw[0]
    for i in range(1, len(T_raw)):
        delta = np.linalg.inv(T_raw[i - 1]) @ T_raw[i]
        delta[:3, 3] *= float(scales[i])
        out[i] = out[i - 1] @ delta
    return out.astype(np.float32)


def prepare_poses(args, indices: list[int]):
    root = abs_path(args.pose_experiment_dir)
    saved = np.load(root / "left_only_predictions.npz")
    T_raw = np.asarray(saved["T_c2w_opencv"], dtype=np.float32)
    saved_indices = np.asarray(saved["selected_original_indices"], dtype=np.int64)
    requested = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(saved_indices, requested):
        raise ValueError(
            "Selected indices differ from left_only_predictions.npz: "
            f"saved={saved_indices.tolist()}, requested={requested.tolist()}"
        )

    causal = load_scales(root / "prefix_only_inference_summary.json", len(T_raw))
    if args.scale_mode == "frozen_init":
        if not 1 <= args.freeze_after_pairs <= len(T_raw):
            raise ValueError("invalid --freeze_after_pairs")
        frozen = float(causal[args.freeze_after_pairs - 1])
        used = np.full(len(T_raw), frozen, dtype=np.float64)
        summary = {
            "mode": "frozen_init",
            "freeze_after_pairs": args.freeze_after_pairs,
            "frozen_scale": frozen,
        }
    else:
        used = causal
        summary = {
            "mode": "cumulative_online",
            "retroactive_rescaling": False,
        }
    return T_raw, integrate_metric_poses(T_raw, used), used, summary


def write_pose_file(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for T in poses:
            f.write(" ".join(f"{float(v):.9f}" for v in T.reshape(-1)) + "\n")


def add_packet_meta(packet, args, local_i, original_i, T_raw, T_metric, scales, step):
    packet.update(
        {
            "frame_local_index": int(local_i),
            "frame_original_index": int(original_i),
            "raw_left_pose_c2w_opencv": torch.from_numpy(T_raw[local_i]).float(),
            "metric_left_pose_c2w_opencv": torch.from_numpy(T_metric[local_i]).float(),
            "scale_mode": args.scale_mode,
            "scale_used_for_increment": float(scales[local_i]),
            "refine_step": int(step),
            "refine_use_target": bool(args.refine_use_target),
            "fusion_applied": False,
        }
    )


def link_or_copy(source: Path, destination: Path) -> str:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    try:
        destination.symlink_to(
            os.path.relpath(source, start=destination.parent.resolve())
        )
        return "symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


@torch.no_grad()
def render_left_packet(
    runtime,
    gaussians,
    batch,
    stage: str,
    scene: str,
    local_i: int,
    original_i: int,
    source_image: Path,
    output_root: Path,
    link_original: bool,
) -> dict[str, Any]:
    from src.evaluation.metrics import compute_psnr, compute_ssim

    context = batch["context"]
    gt = context["image"][:, 0]
    image_shape = tuple(int(v) for v in gt.shape[-2:])

    cuda_sync(runtime.device)
    t0 = time.perf_counter()
    output = runtime.model.decoder.forward(
        gaussians,
        context["extrinsics"][:, 0:1],
        context["intrinsics"][:, 0:1],
        context["near"][:, 0:1],
        context["far"][:, 0:1],
        image_shape,
        depth_mode=None,
    )
    cuda_sync(runtime.device)
    render_sec = time.perf_counter() - t0
    pred = output.color[:, 0]

    psnr = float(compute_psnr(gt, pred).reshape(-1)[0].detach().cpu().item())
    ssim = float(compute_ssim(gt, pred).reshape(-1)[0].detach().cpu().item())

    stage_root = output_root / stage
    rendered = stage_root / "rendered" / f"{scene}.png"
    processed = stage_root / "gt_processed" / f"{scene}.png"
    BASE.save_png_tensor(pred[0], rendered)
    BASE.save_png_tensor(gt[0], processed)

    original_link = None
    link_mode = None
    if link_original:
        suffix = source_image.suffix or ".png"
        original_link = stage_root / "gt_original" / f"{scene}{suffix}"
        link_mode = link_or_copy(source_image, original_link)

    return {
        "stage": stage,
        "scene": scene,
        "frame_local_index": int(local_i),
        "frame_original_index": int(original_i),
        "num_gaussians": int(gaussians.means.shape[1]),
        "psnr": psnr,
        "ssim": ssim,
        "render_sec": float(render_sec),
        "rendered_image": str(rendered),
        "gt_processed_image": str(processed),
        "gt_original_source": str(source_image.resolve()),
        "gt_original_link": None if original_link is None else str(original_link),
        "gt_original_link_mode": link_mode,
        "metric_reference": "preprocessed_left_context_image",
        "global_pose_independent_check": True,
    }


def finalize_self_render(root: Path, rows_by_stage: dict[str, list[dict[str, Any]]]):
    stages = {}
    for stage, rows in rows_by_stage.items():
        psnr = [float(r["psnr"]) for r in rows]
        ssim = [float(r["ssim"]) for r in rows]
        times = [float(r["render_sec"]) for r in rows]
        stage_root = root / stage
        save_csv(stage_root / "metrics.csv", rows)
        summary = {
            "stage": stage,
            "packet_count": len(rows),
            "metrics_mean": {
                "psnr": float(np.mean(psnr)) if psnr else None,
                "ssim": float(np.mean(ssim)) if ssim else None,
            },
            "metrics_median": {
                "psnr": float(np.median(psnr)) if psnr else None,
                "ssim": float(np.median(ssim)) if ssim else None,
            },
            "render_total_sec": float(np.sum(times)) if times else 0.0,
            "render_mean_sec": float(np.mean(times)) if times else None,
            "metrics_csv": str(stage_root / "metrics.csv"),
            "charts_generated": False,
        }
        save_json(stage_root / "summary.json", summary)
        stages[stage] = summary
    top = {
        "purpose": "per-packet left-context self-render validation",
        "stages": stages,
        "charts_generated": False,
    }
    save_json(root / "summary.json", top)
    return top


@torch.no_grad()
def generate_packets(runtime, selected, T_raw, T_metric, scales, args):
    model = runtime.model
    packet_root = abs_path(args.work_dir) / args.packet_out_name
    render_root = abs_path(args.work_dir) / args.self_render_out_name

    requested_steps = parse_refine_steps(args.refine_steps)
    compare_mode = requested_steps is not None
    original_num_refine = int(getattr(model.encoder.cfg, "num_refine", 0))

    if compare_mode:
        steps = requested_steps
        max_step = max(steps)
        stages = [stage_name(s) for s in steps]
        if max_step > 0:
            if original_num_refine <= 0 or not hasattr(model.encoder, "update_module"):
                raise RuntimeError(
                    "Refinement requested, but the loaded encoder has no update module"
                )
            model.encoder.cfg.num_refine = max_step
    else:
        if args.resplat_packet_stage == "init":
            steps, stages = [0], ["init"]
        elif args.resplat_packet_stage == "final":
            steps, stages = [original_num_refine], ["final"]
        else:
            steps, stages = [0, original_num_refine], ["init", "final"]
        max_step = original_num_refine if "final" in stages else 0

    for stage in stages:
        (packet_root / stage).mkdir(parents=True, exist_ok=True)

    counts = {stage: 0 for stage in stages}
    timing_rows = []
    packet_rows = []
    render_rows = {stage: [] for stage in stages}

    try:
        for local_i, original_i in enumerate(selected.indices):
            scene = f"{args.scene_name}_{local_i:04d}"
            target_local = local_i + args.resplat_target_offset
            if target_local < 0 or target_local >= len(selected.indices):
                if args.drop_invalid_target_offset:
                    continue
                target_local = max(0, min(target_local, len(selected.indices) - 1))

            t_all = time.perf_counter()
            batch_cpu = BASE.make_resplat_batch_for_frame(
                runtime=runtime,
                scene_key=scene,
                frame_index=original_i,
                left_path=selected.left_paths[local_i],
                right_path=selected.right_paths[local_i],
                T_left_c2w_cv=T_metric[local_i],
                target_camera=args.resplat_target_camera,
                target_offset_frame_index=selected.indices[target_local],
                target_left_path=selected.left_paths[target_local],
                target_right_path=selected.right_paths[target_local],
                target_T_left_c2w_cv=T_metric[target_local],
                stereo_baseline=args.stereo_baseline,
            )
            batch = BASE.tensor_to_device(batch_cpu, runtime.device)
            batch = model.data_shim(batch)
            cuda_sync(runtime.device)
            prep_sec = time.perf_counter() - t_all

            t_enc = time.perf_counter()
            enc_out = model.encoder(
                batch["context"], 0, deterministic=False, visualization_dump=None
            )
            cuda_sync(runtime.device)
            encoder_sec = time.perf_counter() - t_enc
            if isinstance(enc_out, dict):
                init_g = enc_out["gaussians"]
                condition = enc_out.get("condition_features")
            else:
                init_g = enc_out
                condition = None

            refine_out = None
            refine_sec = 0.0
            if max_step > 0:
                if condition is None:
                    raise RuntimeError("Refinement requested but condition_features missing")
                t_ref = time.perf_counter()
                refine_out = model.encoder.forward_update(
                    batch["context"],
                    batch["target"] if args.refine_use_target else None,
                    condition,
                    init_g,
                    model.decoder,
                    batch.get("context_remain"),
                )
                cuda_sync(runtime.device)
                refine_sec = time.perf_counter() - t_ref
                if len(refine_out["gaussian"]) < max_step:
                    raise RuntimeError("forward_update returned too few Gaussian stages")

            saved = {}
            frame_metrics = {}
            render_sec = 0.0
            for step, stage in zip(steps, stages):
                g = init_g if step == 0 else refine_out["gaussian"][step - 1]
                packet = BASE.make_packet_from_gaussians(g, batch, model.decoder, stage)
                add_packet_meta(
                    packet, args, local_i, original_i, T_raw, T_metric, scales, step
                )
                packet_path = packet_root / stage / f"{scene}.pt"
                torch.save(packet, packet_path)
                saved[stage] = str(packet_path)
                counts[stage] += 1
                del packet

                if args.self_render_packets:
                    row = render_left_packet(
                        runtime,
                        g,
                        batch,
                        stage,
                        scene,
                        local_i,
                        original_i,
                        selected.left_paths[local_i],
                        render_root,
                        args.self_render_link_original,
                    )
                    render_rows[stage].append(row)
                    frame_metrics[stage] = {
                        "psnr": row["psnr"],
                        "ssim": row["ssim"],
                    }
                    render_sec += float(row["render_sec"])

            total_sec = time.perf_counter() - t_all
            timing_rows.append(
                {
                    "frame_local_index": local_i,
                    "frame_original_index": original_i,
                    "preprocess_and_shim_sec": prep_sec,
                    "encoder_sec": encoder_sec,
                    "refine_sec": refine_sec,
                    "self_render_sec": render_sec,
                    "total_sec": total_sec,
                }
            )
            packet_rows.append(
                {
                    "scene": scene,
                    "frame_local_index": local_i,
                    "frame_original_index": original_i,
                    "left_image": str(selected.left_paths[local_i]),
                    "right_image": str(selected.right_paths[local_i]),
                    "saved_packets": saved,
                    "self_render": frame_metrics,
                }
            )

            msg = f"generated {local_i + 1}/{len(selected.indices)}"
            if frame_metrics:
                msg += " " + ", ".join(
                    f"{s}:PSNR={m['psnr']:.3f},SSIM={m['ssim']:.4f}"
                    for s, m in frame_metrics.items()
                )
            BASE.log(msg)
            del batch, batch_cpu, enc_out, init_g
            if refine_out is not None:
                del refine_out
            if (local_i + 1) % max(1, args.empty_cache_every) == 0:
                torch.cuda.empty_cache()
    finally:
        model.encoder.cfg.num_refine = original_num_refine

    save_csv(packet_root / "packet_timing.csv", timing_rows)
    self_summary = None
    if args.self_render_packets:
        self_summary = finalize_self_render(render_root, render_rows)

    manifest = {
        "pipeline": "ZipMap metric pose -> ReSplat packet only",
        "scale_mode": args.scale_mode,
        "refine_steps": requested_steps,
        "refine_use_target": bool(args.refine_use_target),
        "packet_counts": counts,
        "selected_original_indices": [int(i) for i in selected.indices],
        "fusion_performed": False,
        "self_render_performed": bool(args.self_render_packets),
        "self_render_summary": None
        if self_summary is None
        else str(render_root / "summary.json"),
        "charts_generated": False,
        "packets": packet_rows,
    }
    save_json(packet_root / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
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
    T_raw, T_metric, scales, scale_summary = prepare_poses(args, selected.indices)

    pose_dir = work / "metric_pose"
    np.savez_compressed(
        pose_dir / "metric_pose_sequence.npz",
        T_raw_c2w_opencv=T_raw,
        T_metric_c2w_opencv=T_metric,
        scale_per_frame=scales,
        selected_original_indices=np.asarray(selected.indices, dtype=np.int64),
    )
    write_pose_file(pose_dir / "raw_pose_c2w_opencv.txt", T_raw)
    write_pose_file(pose_dir / "metric_pose_c2w_opencv.txt", T_metric)
    save_json(pose_dir / "scale_summary.json", scale_summary)

    runtime = BASE.load_resplat_runtime(args)
    manifest = generate_packets(runtime, selected, T_raw, T_metric, scales, args)
    save_json(
        work / "run_summary.json",
        {
            "scale": scale_summary,
            "num_frames": len(selected.indices),
            "packet_counts": manifest["packet_counts"],
            "packet_manifest": str(work / args.packet_out_name / "manifest.json"),
            "fusion_performed": False,
            "self_render_performed": bool(args.self_render_packets),
            "self_render_summary": manifest["self_render_summary"],
            "charts_generated": False,
        },
    )
    print(f"[Done] packets: {work / args.packet_out_name}")
    if args.self_render_packets:
        print(f"[Done] self-render: {work / args.self_render_out_name / 'summary.json'}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
