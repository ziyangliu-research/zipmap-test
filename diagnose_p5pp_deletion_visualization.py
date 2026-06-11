#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 可视化 P5++ 到底删了哪些高斯
"""
Diagnostic visualization for P5++ footprint free-space pruning.

Purpose
-------
For a current incoming packet t, visualize what the pre-insert pruning is doing:

  M_{t-1} old map
  -> render old map from current view BEFORE deletion
  -> compute P5++ free-space deletion mask
  -> render old map AFTER deleting selected old Gaussians
  -> optionally render temporary map M_{t-1}+G_t without deletion
  -> overlay deleted Gaussians on current GT / old render
  -> save depth-edge overlay and deletion statistics

This script is diagnostic-only. It does not save a fused packet.

Expected use
------------
For two-frame diagnosis around frame 25/26:

python diagnose_p5pp_deletion_visualization.py \
  --p5pp_script /home/shiyo/Desktop/ZipMap/map_maintenance_p5pp_hard_edge_aware.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 25-26 \
  --trajectory_ranges 25-26 \
  --diagnose_indices 26 \
  --depth_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/depth_lcam_front \
  --depth_pattern "{index:06d}_lcam_front_depth.png" \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --output_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/diag_p5pp_25_26_v1like \
  --fs_preset v1_like \
  --device cuda:0

For current v3 balanced:

  --fs_preset balanced

Main outputs per diagnosed packet
---------------------------------
step_0026_gt.png
step_0026_old_before_prune.png
step_0026_old_after_prune.png
step_0026_temp_after_insert_no_prune.png
step_0026_deleted_overlay_on_gt.png
step_0026_deleted_overlay_on_old_render.png
step_0026_depth_edge_overlay.png
step_0026_summary.json
step_0026_deleted_source_hist.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ALL_GAUSSIAN_KEYS = ["means", "covariances", "harmonics", "opacities", "scales", "rotations", "rotations_unnorm"]


def import_module_from_path(path: Path, module_name: str = "p5pp_module"):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def lazy_import_resplat(resplat_repo: Path):
    root = resplat_repo.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.model.types import Gaussians
    from src.model.decoder.gsplat_decoder_splatting_cuda import (
        GSplatDecoderSplattingCUDA,
        GSplatDecoderSplattingCUDACfg,
    )
    return Gaussians, GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg


def make_decoder(resplat_repo: Path, device: torch.device, background_color: Any):
    _, GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg = lazy_import_resplat(resplat_repo)

    class DatasetCfgLike:
        pass

    DatasetCfgLike.background_color = background_color if background_color is not None else [0.0, 0.0, 0.0]
    cfg = GSplatDecoderSplattingCUDACfg(name="gsplat", scale_invariant=False, use_covariances=True)
    return GSplatDecoderSplattingCUDA(cfg, DatasetCfgLike()).to(device).eval()


def normalize_for_render(x: torch.Tensor, key: str) -> torch.Tensor:
    if key in {"means", "scales", "rotations", "rotations_unnorm"}:
        if x.ndim == 2:
            x = x.unsqueeze(0)
    elif key == "covariances":
        if x.ndim == 3:
            x = x.unsqueeze(0)
    elif key == "harmonics":
        if x.ndim == 3:
            x = x.unsqueeze(0)
    elif key == "opacities":
        if x.ndim == 1:
            x = x.unsqueeze(0)
    return x.contiguous()


def make_gaussians(Gaussians_cls, fields: Dict[str, torch.Tensor]):
    kwargs: Dict[str, torch.Tensor] = {}
    for k in ALL_GAUSSIAN_KEYS:
        if k in fields:
            kwargs[k] = normalize_for_render(fields[k], k)
    return Gaussians_cls(**kwargs)


def first_matrix(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().to(device=device, dtype=torch.float32)
    else:
        t = torch.as_tensor(x, device=device, dtype=torch.float32)
    while t.ndim > 2:
        t = t[0]
    return t.float().contiguous()


def first_scalar(x: Any, default: float) -> torch.Tensor:
    if x is None:
        return torch.tensor(default, dtype=torch.float32)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return torch.tensor(default, dtype=torch.float32)
        return x.detach().reshape(-1)[0].float().cpu()
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return torch.tensor(default, dtype=torch.float32)
    return torch.tensor(float(arr[0]), dtype=torch.float32)


def render_fields(
    Gaussians_cls,
    decoder,
    fields: Dict[str, torch.Tensor],
    packet: Dict[str, Any],
    device: torch.device,
    H: int,
    W: int,
) -> torch.Tensor:
    if fields is None or int(fields["means"].shape[0]) == 0:
        return torch.zeros((3, H, W), device=device, dtype=torch.float32)
    gaussians = make_gaussians(Gaussians_cls, fields)
    ext = first_matrix(packet["target_extrinsics"], device)[None, None]
    K = first_matrix(packet["target_intrinsics"], device)[None, None]
    near = first_scalar(packet.get("target_near", None), 0.1).to(device)[None, None]
    far = first_scalar(packet.get("target_far", None), 50.0).to(device)[None, None]
    with torch.no_grad():
        out = decoder.forward(gaussians, ext, K, near, far, (H, W), depth_mode=None)
    return out.color[0, 0].detach().float().clamp(0, 1)


def target_image_tensor(packet: Dict[str, Any], device: torch.device, H: int, W: int) -> torch.Tensor:
    img = packet.get("target_image", None)
    if not isinstance(img, torch.Tensor):
        raise KeyError("Packet has no target_image tensor.")
    img = img.detach().float()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        pass
    elif img.ndim == 3 and img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1)
    else:
        raise RuntimeError(f"Unexpected target_image shape: {tuple(img.shape)}")
    img = img.to(device)
    if img.shape[-2:] != (H, W):
        img = F.interpolate(img[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return img.clamp(0, 1)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().float().cpu().clamp(0, 1)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        arr = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    elif img.ndim == 2:
        arr = (img.numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    else:
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(img.shape)}")
    return Image.fromarray(arr, mode="RGB")


def save_image(path: Path, img: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(img).save(path)


def fields_select(fields: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[mask].contiguous() for k, v in fields.items() if k in ALL_GAUSSIAN_KEYS}


def fields_concat(fields_a: Dict[str, torch.Tensor], fields_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in fields_a.keys():
        if k in fields_b:
            out[k] = torch.cat([fields_a[k], fields_b[k]], dim=0).contiguous()
    return out


def compute_depth_edge_mask(
    depth: torch.Tensor,
    abs_thresh: float,
    rel_thresh: float,
    dilate_px: int,
) -> torch.Tensor:
    """Return boolean [H,W] depth discontinuity mask."""
    if depth is None:
        raise ValueError("No depth available.")
    D = depth.float()
    valid = torch.isfinite(D) & (D > 0)
    D0 = torch.where(valid, D, torch.zeros_like(D))

    # neighbor absolute differences
    dx = torch.zeros_like(D0)
    dy = torch.zeros_like(D0)
    dx[:, :-1] = torch.abs(D0[:, 1:] - D0[:, :-1])
    dy[:-1, :] = torch.abs(D0[1:, :] - D0[:-1, :])
    local = torch.maximum(dx, dy)
    rel = local / torch.clamp(D0, min=1e-6)
    edge = valid & ((local >= float(abs_thresh)) | (rel >= float(rel_thresh)))
    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        edge = F.max_pool2d(edge.float()[None, None], kernel_size=k, stride=1, padding=int(dilate_px))[0, 0] > 0
    return edge


def overlay_mask_on_image(base: Image.Image, mask: torch.Tensor, color=(255, 0, 0), alpha=100) -> Image.Image:
    base = base.convert("RGBA")
    mask_np = mask.detach().cpu().numpy().astype(bool)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    arr[mask_np] = (color[0], color[1], color[2], alpha)
    overlay = Image.fromarray(arr, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def project_for_overlay(p5m, fields: Dict[str, torch.Tensor], view, args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means = fields["means"]
    scale_metric = p5m.compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    u, v, z = p5m.project_points(means, view.Twc_or_Tcw, view.K_pixel, args.extrinsic_type, args.camera_z_sign)
    f = float((view.K_pixel[0, 0] + view.K_pixel[1, 1]) * 0.5)
    r = f * scale_metric / torch.clamp(z, min=1e-6)
    return u, v, z, r


def draw_deleted_overlay(
    base_img: Image.Image,
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    radius: torch.Tensor,
    deleted_mask: torch.Tensor,
    source_ids: torch.Tensor,
    edge_mask: Optional[torch.Tensor],
    max_draw: int,
    radius_scale: float,
    max_circle_radius: float,
) -> Image.Image:
    W, H = base_img.size
    overlay = base_img.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    idx = torch.nonzero(deleted_mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return overlay.convert("RGB")

    # Draw highest visual-impact candidates first: opacity is unavailable here, so use projected radius.
    score = torch.clamp(radius[idx], min=0)
    if idx.numel() > max_draw:
        top = torch.topk(score, k=max_draw, largest=True).indices
        idx = idx[top]

    u_cpu = u[idx].detach().cpu().numpy()
    v_cpu = v[idx].detach().cpu().numpy()
    z_cpu = z[idx].detach().cpu().numpy()
    r_cpu = radius[idx].detach().cpu().numpy()
    src_cpu = source_ids[idx].detach().cpu().numpy()

    edge_np = edge_mask.detach().cpu().numpy().astype(bool) if edge_mask is not None else None

    # Palette by source id.
    palette = [
        (255, 0, 0, 140),       # red
        (255, 128, 0, 140),     # orange
        (255, 0, 255, 140),     # magenta
        (0, 255, 255, 140),     # cyan
        (255, 255, 0, 140),     # yellow
        (0, 255, 0, 140),       # green
        (80, 160, 255, 140),    # blue
    ]

    for uu, vv, zz, rr, sid in zip(u_cpu, v_cpu, z_cpu, r_cpu, src_cpu):
        if not np.isfinite(uu) or not np.isfinite(vv) or not np.isfinite(rr):
            continue
        # Center inside image vs footprint-only.
        center_inside = (0 <= uu < W and 0 <= vv < H and zz > 0)
        # If the center is on a depth-edge band, mark white.
        on_edge = False
        if center_inside and edge_np is not None:
            xi = int(round(uu))
            yi = int(round(vv))
            if 0 <= xi < W and 0 <= yi < H:
                on_edge = bool(edge_np[yi, xi])

        if on_edge:
            color = (255, 255, 255, 180)
        elif center_inside:
            color = palette[int(sid) % len(palette)]
        else:
            color = (255, 0, 255, 160)  # magenta: center outside, footprint can still intersect

        rad = max(1.5, min(float(rr) * float(radius_scale), float(max_circle_radius)))
        x0, y0 = float(uu) - rad, float(vv) - rad
        x1, y1 = float(uu) + rad, float(vv) + rad
        if x1 < 0 or x0 >= W or y1 < 0 or y0 >= H:
            continue
        draw.ellipse([x0, y0, x1, y1], outline=color, width=2)

    return overlay.convert("RGB")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_p5_args(p5m, args, output_dummy: Path):
    # Build a P5++ args object using its own parser, so all expected fields exist.
    argv = [
        "--packet_dir", str(args.packet_dir),
        "--packet_ranges", args.packet_ranges,
        "--trajectory_ranges", args.trajectory_ranges or args.packet_ranges,
        "--depth_dir", str(args.depth_dir),
        "--depth_pattern", args.depth_pattern,
        "--depth_scale", str(args.depth_scale),
        "--output_pt", str(output_dummy),
        "--execution_mode", "incremental",
        "--maintenance_order", "pre_p3_insert_p4",
        "--pre_p3_trajectory_policy", "current",
        "--apply_once",
        "--enable_p3",
        "--p3_mode", "hard",
        "--corridor_radius", "0.50",
        "--corridor_mode", "none",
        "--near_z_thresh", "1.00",
        "--near_mode", "none",
        "--extent_mode", "active",
        "--extent_factor", str(args.extent_factor),
        "--extent_min_count", "1",
        "--extent_min_opacity", "0.02",
        "--extent_min_radius_px", "25",
        "--footprint_px_thresh", "50",
        "--footprint_mode", "active",
        "--footprint_min_count", "1",
        "--footprint_min_opacity", "0.02",
        "--footprint_free_space_mode", "active",
        "--fs_preset", args.fs_preset,
        "--opacity_cap", str(args.opacity_cap),
        "--scale_input", args.scale_input,
        "--scale_metric", args.scale_metric,
        "--chunk_size", str(args.chunk_size),
        "--device", args.device,
    ]
    if hasattr(p5m, "build_argparser"):
        p5_args = p5m.build_argparser().parse_args(argv)
    else:
        raise RuntimeError("P5++ module has no build_argparser().")

    if hasattr(p5m, "apply_fs_preset"):
        p5_args = p5m.apply_fs_preset(p5_args)

    # Manual overrides for diagnosis.
    p5_args.fs_margin_abs = args.fs_margin_abs
    p5_args.fs_margin_rel = args.fs_margin_rel
    return p5_args


def main():
    parser = argparse.ArgumentParser(description="Visual diagnostic for P5++ deleted Gaussians")
    parser.add_argument("--p5pp_script", type=Path, required=True)
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--packet_ranges", type=str, required=True)
    parser.add_argument("--trajectory_ranges", type=str, default=None)
    parser.add_argument("--diagnose_indices", type=str, default=None, help="Sorted packet indices to diagnose. Default: all selected except first.")
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument("--depth_pattern", type=str, default="{index:06d}_lcam_front_depth.png")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--resplat_repo", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fs_preset", choices=["v1_like", "aggressive", "balanced", "conservative"], default="v1_like")
    parser.add_argument("--fs_margin_abs", type=float, default=0.20)
    parser.add_argument("--fs_margin_rel", type=float, default=0.02)
    parser.add_argument("--extent_factor", type=float, default=3.0)
    parser.add_argument("--opacity_cap", type=float, default=0.30)
    parser.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    parser.add_argument("--scale_metric", choices=["max", "mean", "volume"], default="max")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_size", type=int, default=250000)
    parser.add_argument("--max_draw_gaussians", type=int, default=50000)
    parser.add_argument("--draw_radius_scale", type=float, default=0.15, help="Drawn circle radius = projected_radius * scale.")
    parser.add_argument("--draw_max_circle_radius", type=float, default=20.0)
    parser.add_argument("--depth_edge_abs_thresh", type=float, default=0.50)
    parser.add_argument("--depth_edge_rel_thresh", type=float, default=0.15)
    parser.add_argument("--depth_edge_dilate_px", type=int, default=5)
    parser.add_argument("--render_temp_after_insert", action="store_true", default=True)
    parser.add_argument("--no_render_temp_after_insert", action="store_false", dest="render_temp_after_insert")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[1/5] Importing P5++ module: {args.p5pp_script}")
    p5m = import_module_from_path(args.p5pp_script)
    p5_args = build_p5_args(p5m, args, args.output_dir / "_dummy.pt")

    print("[2/5] Loading packets / trajectory views")
    packets = p5m.load_packets(args.packet_dir)
    selected = p5m.parse_ranges(args.packet_ranges, len(packets))
    traj_indices = p5m.parse_ranges(args.trajectory_ranges or args.packet_ranges, len(packets))
    diagnose = p5m.parse_ranges(args.diagnose_indices, len(packets)) if args.diagnose_indices else selected[1:]
    diagnose_set = set(diagnose)
    gaussian_keys = p5m.available_gaussian_keys(packets, selected)
    views = p5m.build_trajectory_views(
        packets,
        traj_indices,
        device,
        p5_args.intrinsics_normalized,
        args.depth_dir,
        args.depth_pattern,
        args.depth_scale,
        stride=1,
    )
    view_by_idx = {v.sorted_index: v for v in views}

    print("[3/5] Initializing renderer")
    bg = packets[selected[0]].data.get("background_color", [0.0, 0.0, 0.0])
    Gaussians_cls, _, _ = lazy_import_resplat(args.resplat_repo)
    decoder = make_decoder(args.resplat_repo, device, bg)

    print("[4/5] Simulating incremental map and saving diagnostics")
    map_fields = None
    map_source = None
    map_uid = None
    uid_cursor = 0
    all_summary_rows: List[Dict[str, Any]] = []

    for step_i, idx in enumerate(selected):
        pref = packets[idx]

        # Before inserting current packet, diagnose deletion from old map using current view.
        if step_i > 0 and idx in diagnose_set:
            if idx not in view_by_idx:
                raise KeyError(f"No trajectory view for packet index {idx}.")
            view = view_by_idx[idx]
            H, W = int(view.H), int(view.W)
            print(f"      diagnose packet={idx:04d}, old_map={int(map_fields['means'].shape[0]):,}")

            # Current GT.
            gt = target_image_tensor(pref.data, device, H, W)
            save_image(args.output_dir / f"step_{idx:04d}_gt.png", gt)

            # Render old map before pruning.
            old_before = render_fields(Gaussians_cls, decoder, map_fields, pref.data, device, H, W)
            save_image(args.output_dir / f"step_{idx:04d}_old_before_prune.png", old_before)

            # Optional temp map after insert, no pruning.
            if args.render_temp_after_insert:
                f_cur, src_cur, uid_cur, _ = p5m.load_packet_fields(pref, gaussian_keys, device, uid_cursor)
                temp_fields = fields_concat(map_fields, f_cur)
                temp_render = render_fields(Gaussians_cls, decoder, temp_fields, pref.data, device, H, W)
                save_image(args.output_dir / f"step_{idx:04d}_temp_after_insert_no_prune.png", temp_render)
                del temp_fields, temp_render, f_cur, src_cur, uid_cur

            # Compute deletion mask.
            centers = torch.stack([p5m.camera_center(view.Twc_or_Tcw, p5_args.extrinsic_type)], dim=0).to(device)
            danger_mask, stats = p5m.compute_p3_masks(map_fields, [view], centers, p5_args)
            deleted_mask = danger_mask

            # Render after deletion from old map.
            keep = ~deleted_mask
            map_after_fields = fields_select(map_fields, keep)
            old_after = render_fields(Gaussians_cls, decoder, map_after_fields, pref.data, device, H, W)
            save_image(args.output_dir / f"step_{idx:04d}_old_after_prune.png", old_after)

            # Depth edge overlay.
            edge_mask = compute_depth_edge_mask(
                view.depth,
                abs_thresh=args.depth_edge_abs_thresh,
                rel_thresh=args.depth_edge_rel_thresh,
                dilate_px=args.depth_edge_dilate_px,
            )
            edge_overlay = overlay_mask_on_image(tensor_to_pil(gt), edge_mask, color=(255, 0, 0), alpha=110)
            edge_overlay.save(args.output_dir / f"step_{idx:04d}_depth_edge_overlay.png")

            # Deleted Gaussian overlay.
            u, v, z, r = project_for_overlay(p5m, map_fields, view, p5_args)
            overlay_gt = draw_deleted_overlay(
                tensor_to_pil(gt),
                u, v, z, r,
                deleted_mask,
                map_source,
                edge_mask=edge_mask,
                max_draw=args.max_draw_gaussians,
                radius_scale=args.draw_radius_scale,
                max_circle_radius=args.draw_max_circle_radius,
            )
            overlay_gt.save(args.output_dir / f"step_{idx:04d}_deleted_overlay_on_gt.png")

            overlay_render = draw_deleted_overlay(
                tensor_to_pil(old_before),
                u, v, z, r,
                deleted_mask,
                map_source,
                edge_mask=edge_mask,
                max_draw=args.max_draw_gaussians,
                radius_scale=args.draw_radius_scale,
                max_circle_radius=args.draw_max_circle_radius,
            )
            overlay_render.save(args.output_dir / f"step_{idx:04d}_deleted_overlay_on_old_render.png")

            # Source histogram and center/footprint stats.
            deleted_idx = torch.nonzero(deleted_mask, as_tuple=False).reshape(-1)
            center_inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            r_clip = torch.clamp(r, min=0, max=float(getattr(p5_args, "fs_radius_clip_px", 120.0)))
            footprint_intersects = (z > 0) & ((u + r_clip) >= 0) & ((u - r_clip) < W) & ((v + r_clip) >= 0) & ((v - r_clip) < H)

            hist_rows: List[Dict[str, Any]] = []
            if deleted_idx.numel() > 0:
                src_deleted = map_source[deleted_idx].detach().cpu().numpy()
                unique, counts = np.unique(src_deleted, return_counts=True)
                for sid, cnt in zip(unique.tolist(), counts.tolist()):
                    hist_rows.append({"source_packet_sorted_index": int(sid), "num_deleted": int(cnt)})
            write_csv(args.output_dir / f"step_{idx:04d}_deleted_source_hist.csv", hist_rows)

            summary = {
                "packet_sorted_index": int(idx),
                "packet_name": pref.path.name,
                "old_map_gaussians": int(map_fields["means"].shape[0]),
                "num_deleted": int(deleted_mask.sum().item()),
                "delete_ratio_old_map": float(deleted_mask.float().mean().item()),
                "num_deleted_center_inside": int((deleted_mask & center_inside).sum().item()),
                "num_deleted_center_outside": int((deleted_mask & (~center_inside)).sum().item()),
                "num_deleted_footprint_intersects": int((deleted_mask & footprint_intersects).sum().item()),
                "num_deleted_center_on_depth_edge": int((deleted_mask & center_inside & edge_mask[torch.clamp(torch.round(v).long(),0,H-1), torch.clamp(torch.round(u).long(),0,W-1)]).sum().item()) if deleted_idx.numel() > 0 else 0,
                "fs_preset": args.fs_preset,
                "depth_edge_abs_thresh": args.depth_edge_abs_thresh,
                "depth_edge_rel_thresh": args.depth_edge_rel_thresh,
                "depth_edge_dilate_px": args.depth_edge_dilate_px,
                **{f"p3_{k}": v for k, v in stats.items() if isinstance(v, (int, float, str, bool)) or v is None},
            }
            with (args.output_dir / f"step_{idx:04d}_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            all_summary_rows.append(summary)

            del map_after_fields, old_before, old_after, danger_mask, deleted_mask, u, v, z, r
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Now insert current packet for the next step.
        f, src, uid, _ = p5m.load_packet_fields(pref, gaussian_keys, device, uid_cursor)
        uid_cursor += int(uid.numel())
        map_fields, map_source, map_uid = p5m.append_map(map_fields, map_source, map_uid, f, src, uid, gaussian_keys)

    print("[5/5] Saving summary CSV")
    write_csv(args.output_dir / "diagnostic_summary.csv", all_summary_rows)
    print(f"saved diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
