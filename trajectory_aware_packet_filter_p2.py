#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory-aware Gaussian packet filtering / map maintenance P2.

Motivation
----------
If naive fusion becomes white/ghosty when the camera moves forward/backward,
packet-local depth consistency may be too weak. A stronger geometric prior is:

    The camera trajectory itself is free space.

Therefore, Gaussians close to the camera path, Gaussians that become too close to
trajectory cameras, or high-opacity large-footprint Gaussians in trajectory views
are dangerous occluders.

This script applies three non-optimization rules before direct concat:

1. Camera-corridor suppression
   Downweight/delete Gaussians whose mean lies within a radius of any trajectory
   camera center.

2. Near-camera projection suppression
   Downweight/delete Gaussians that project inside trajectory views with
   0 < z < near_z_thresh. These are likely to dominate alpha compositing.

3. Projected-footprint suppression
   Downweight/delete Gaussians whose approximate projected radius is too large:
       radius_px ~= f * scale_metric / z

The output is a fused packet .pt usable by render_fused_packet_trajectory.py.

Typical first test
------------------
python trajectory_aware_packet_filter_p2.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --trajectory_ranges 0-79 \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p2_corridor_near_footprint.pt \
  --corridor_radius 0.30 \
  --corridor_mode downweight \
  --corridor_downweight 0.02 \
  --near_z_thresh 0.60 \
  --near_mode downweight \
  --near_downweight 0.02 \
  --near_min_count 1 \
  --footprint_px_thresh 120 \
  --footprint_mode downweight \
  --footprint_downweight 0.05 \
  --footprint_min_count 1 \
  --footprint_min_opacity 0.05 \
  --opacity_cap 0.30 \
  --trajectory_stride 1 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


PRIMARY_GAUSSIAN_KEYS = ["means", "covariances", "harmonics", "opacities"]
OPTIONAL_GAUSSIAN_KEYS = ["scales", "rotations", "rotations_unnorm"]
ALL_GAUSSIAN_KEYS = PRIMARY_GAUSSIAN_KEYS + OPTIONAL_GAUSSIAN_KEYS


@dataclass
class PacketRef:
    sorted_index: int
    path: Path
    data: Dict[str, Any]


@dataclass
class TrajectoryView:
    sorted_index: int
    frame_index: int
    Twc_or_Tcw: torch.Tensor
    K_pixel: torch.Tensor
    H: int
    W: int


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def parse_ranges(spec: Optional[str], n: int) -> List[int]:
    if spec is None or spec.strip() == "":
        return list(range(n))
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a, b = int(a_str), int(b_str)
            if b < a:
                raise ValueError(f"Invalid descending range: {token}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    out = sorted(set(out))
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise IndexError(f"Packet indices out of range 0..{n-1}: {bad[:20]}")
    return out


def ensure_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def normalize_gaussian_tensor(x: Any, key: str, device: torch.device) -> torch.Tensor:
    t = ensure_tensor(x, device=device)
    if key in {"means", "scales", "rotations", "rotations_unnorm"}:
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 2:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    elif key == "covariances":
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    elif key == "harmonics":
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    elif key == "opacities":
        if t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    return t.contiguous()


def first_matrix(x: Any, device: torch.device) -> torch.Tensor:
    t = ensure_tensor(x, device=device)
    while t.ndim > 2:
        t = t[0]
    return t.float()


def first_scalar(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return default
        return float(x.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def first_int(x: Any, default: int) -> int:
    v = first_scalar(x, None)
    return default if v is None else int(round(v))


def infer_image_shape(packet: Dict[str, Any]) -> Tuple[int, int]:
    if "image_shape" in packet:
        s = packet["image_shape"]
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().reshape(-1).tolist()
        if len(s) >= 2:
            return int(s[0]), int(s[1])
    if "target_image" in packet and isinstance(packet["target_image"], torch.Tensor):
        img = packet["target_image"]
        if img.ndim >= 4:
            return int(img.shape[-2]), int(img.shape[-1])
        if img.ndim == 3:
            if img.shape[0] in (1, 3):
                return int(img.shape[1]), int(img.shape[2])
            return int(img.shape[0]), int(img.shape[1])
    return 320, 320


def denormalize_intrinsics(K: torch.Tensor, H: int, W: int, mode: str) -> torch.Tensor:
    K = K.clone().float()
    if mode == "true":
        is_norm = True
    elif mode == "false":
        is_norm = False
    elif mode == "auto":
        is_norm = abs(float(K[0, 0])) <= 10.0 and abs(float(K[1, 1])) <= 10.0
    else:
        raise ValueError("--intrinsics_normalized must be auto/true/false")
    if is_norm:
        K[0, 0] *= float(W)
        K[1, 1] *= float(H)
        K[0, 2] *= float(W)
        K[1, 2] *= float(H)
    return K


def load_packets(packet_dir: Path) -> List[PacketRef]:
    paths = sorted(packet_dir.glob("*.pt"), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .pt packets found in {packet_dir}")
    refs: List[PacketRef] = []
    for i, path in enumerate(paths):
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            raise RuntimeError(f"Packet is not a dict: {path}")
        missing = [k for k in PRIMARY_GAUSSIAN_KEYS if k not in data]
        if missing:
            raise KeyError(f"Packet {path.name} missing keys: {missing}")
        refs.append(PacketRef(i, path, data))
    return refs


def available_gaussian_keys(packets: Sequence[PacketRef], indices: Sequence[int]) -> List[str]:
    keys = [k for k in ALL_GAUSSIAN_KEYS if all(k in packets[i].data for i in indices)]
    missing = [k for k in PRIMARY_GAUSSIAN_KEYS if k not in keys]
    if missing:
        raise KeyError(f"Missing primary Gaussian keys: {missing}")
    return keys


def camera_center(E: torch.Tensor, extrinsic_type: str) -> torch.Tensor:
    if extrinsic_type == "Twc":
        return E[:3, 3]
    if extrinsic_type == "Tcw":
        Twc = torch.linalg.inv(E)
        return Twc[:3, 3]
    raise ValueError("extrinsic_type must be Twc or Tcw")


def build_trajectory_views(
    packets: Sequence[PacketRef],
    trajectory_indices: Sequence[int],
    device: torch.device,
    intrinsics_normalized: str,
    extrinsic_type: str,
    stride: int,
) -> List[TrajectoryView]:
    if stride <= 0:
        stride = 1
    selected = list(trajectory_indices)[::stride]
    views: List[TrajectoryView] = []
    for idx in selected:
        p = packets[idx].data
        if "target_extrinsics" not in p or "target_intrinsics" not in p:
            raise KeyError(f"Trajectory packet {packets[idx].path.name} lacks target_extrinsics/target_intrinsics")
        H, W = infer_image_shape(p)
        E = first_matrix(p["target_extrinsics"], device)
        K = denormalize_intrinsics(first_matrix(p["target_intrinsics"], device), H, W, intrinsics_normalized)
        frame_index = first_int(p.get("target_index", idx), default=idx)
        views.append(TrajectoryView(int(idx), int(frame_index), E, K, H, W))
    return views


def project_points(
    means_world: torch.Tensor,
    E: torch.Tensor,
    K: torch.Tensor,
    extrinsic_type: str,
    camera_z_sign: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if extrinsic_type == "Twc":
        Tcw = torch.linalg.inv(E)
    elif extrinsic_type == "Tcw":
        Tcw = E
    else:
        raise ValueError("extrinsic_type must be Twc/Tcw")
    xyz_cam = means_world @ Tcw[:3, :3].T + Tcw[:3, 3][None, :]
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    if camera_z_sign == "positive":
        zf = z
    elif camera_z_sign == "negative":
        zf = -z
    else:
        raise ValueError("camera_z_sign must be positive/negative")
    eps = 1e-8
    u = K[0, 0] * (x / (zf + eps)) + K[0, 2]
    v = K[1, 1] * (y / (zf + eps)) + K[1, 2]
    return u, v, zf


def scale_to_positive(scales: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return torch.clamp(scales.abs(), min=1e-8)
    if mode == "log":
        return torch.exp(scales)
    if mode == "auto":
        frac_nonpos = float((scales <= 0).float().mean().item())
        if frac_nonpos > 0.05:
            return torch.exp(scales)
        return torch.clamp(scales, min=1e-8)
    raise ValueError("--scale_input must be auto/raw/log")


def compute_scale_metric(scales: torch.Tensor, scale_input: str, scale_metric: str) -> torch.Tensor:
    s = scale_to_positive(scales, scale_input)
    if scale_metric == "max":
        return torch.max(s, dim=-1).values
    if scale_metric == "mean":
        return torch.mean(s, dim=-1)
    if scale_metric == "volume":
        return torch.prod(s, dim=-1)
    raise ValueError("--scale_metric must be max/mean/volume")


def apply_mode(opacities: torch.Tensor, keep_mask: torch.Tensor, mask: torch.Tensor, mode: str, downweight: float) -> None:
    if mode == "none" or not torch.any(mask):
        return
    if mode == "hard":
        keep_mask[mask] = False
    elif mode == "downweight":
        opacities[mask] = opacities[mask] * float(downweight)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def compute_corridor_mask(
    means: torch.Tensor,
    centers: torch.Tensor,
    radius: float,
    chunk_size: int,
) -> torch.Tensor:
    N = int(means.shape[0])
    mask = torch.zeros((N,), device=means.device, dtype=torch.bool)
    if radius <= 0 or centers.numel() == 0:
        return mask
    r2 = float(radius) ** 2
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pts = means[start:end]
        # [chunk, C]
        d2 = torch.cdist(pts, centers) ** 2
        min_d2 = torch.min(d2, dim=1).values
        mask[start:end] = min_d2 <= r2
    return mask


def compute_near_and_footprint_masks(
    means: torch.Tensor,
    opacities: torch.Tensor,
    scale_metric: Optional[torch.Tensor],
    views: Sequence[TrajectoryView],
    extrinsic_type: str,
    camera_z_sign: str,
    near_z_thresh: float,
    near_min_count: int,
    footprint_px_thresh: float,
    footprint_min_count: int,
    footprint_min_opacity: float,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = int(means.shape[0])
    device = means.device
    near_count = torch.zeros((N,), device=device, dtype=torch.int16)
    footprint_count = torch.zeros((N,), device=device, dtype=torch.int16)

    need_near = near_z_thresh > 0
    need_foot = footprint_px_thresh > 0 and scale_metric is not None
    if not need_near and not need_foot:
        z = torch.zeros((N,), device=device, dtype=torch.bool)
        return z, z.clone(), near_count, footprint_count

    for view in views:
        H, W = int(view.H), int(view.W)
        f = float((view.K_pixel[0, 0] + view.K_pixel[1, 1]) * 0.5)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            u, v, z = project_points(means[start:end], view.Twc_or_Tcw, view.K_pixel, extrinsic_type, camera_z_sign)
            ui = torch.round(u).long()
            vi = torch.round(v).long()
            inside = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

            if need_near:
                nm = inside & (z < float(near_z_thresh))
                near_count[start:end][nm] += 1

            if need_foot:
                sm = scale_metric[start:end]
                radius_px = f * sm / torch.clamp(z, min=1e-6)
                fm = inside & (radius_px > float(footprint_px_thresh)) & (opacities[start:end] >= float(footprint_min_opacity))
                footprint_count[start:end][fm] += 1

    near_mask = near_count >= int(near_min_count)
    footprint_mask = footprint_count >= int(footprint_min_count)
    return near_mask, footprint_mask, near_count, footprint_count


def process_packet(
    packet_ref: PacketRef,
    gaussian_keys: Sequence[str],
    views: Sequence[TrajectoryView],
    centers: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
    p = packet_ref.data
    fields = {k: normalize_gaussian_tensor(p[k], k, device) for k in gaussian_keys}
    means = fields["means"]
    opacities = fields["opacities"].clone()
    fields["opacities"] = opacities
    N = int(means.shape[0])
    keep_mask = torch.ones((N,), device=device, dtype=torch.bool)

    if args.opacity_cap > 0:
        opacities.clamp_(max=float(args.opacity_cap))

    corridor_mask = compute_corridor_mask(means, centers, args.corridor_radius, args.chunk_size)
    apply_mode(opacities, keep_mask, corridor_mask, args.corridor_mode, args.corridor_downweight)

    scale_metric = None
    if "scales" in fields and (args.footprint_px_thresh > 0):
        scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)

    near_mask, footprint_mask, near_count, footprint_count = compute_near_and_footprint_masks(
        means=means,
        opacities=opacities,
        scale_metric=scale_metric,
        views=views,
        extrinsic_type=args.extrinsic_type,
        camera_z_sign=args.camera_z_sign,
        near_z_thresh=args.near_z_thresh,
        near_min_count=args.near_min_count,
        footprint_px_thresh=args.footprint_px_thresh,
        footprint_min_count=args.footprint_min_count,
        footprint_min_opacity=args.footprint_min_opacity,
        chunk_size=args.chunk_size,
    )
    apply_mode(opacities, keep_mask, near_mask, args.near_mode, args.near_downweight)
    apply_mode(opacities, keep_mask, footprint_mask, args.footprint_mode, args.footprint_downweight)

    for k in list(fields.keys()):
        fields[k] = fields[k][keep_mask].contiguous()

    source_ids = torch.full((int(fields["means"].shape[0]),), packet_ref.sorted_index, device=device, dtype=torch.long)
    frame_index = first_int(p.get("target_index", packet_ref.sorted_index), default=packet_ref.sorted_index)

    stats = {
        "packet_sorted_index": int(packet_ref.sorted_index),
        "packet_name": packet_ref.path.name,
        "target_frame_index": int(frame_index),
        "num_input": int(N),
        "num_output": int(fields["means"].shape[0]),
        "hard_removed": int(N - fields["means"].shape[0]),
        "num_corridor": int(corridor_mask.sum().item()),
        "num_near": int(near_mask.sum().item()),
        "num_footprint": int(footprint_mask.sum().item()),
        "corridor_ratio": float(corridor_mask.sum().item() / max(N, 1)),
        "near_ratio": float(near_mask.sum().item() / max(N, 1)),
        "footprint_ratio": float(footprint_mask.sum().item() / max(N, 1)),
        "near_count_mean": float(near_count.float().mean().item()),
        "footprint_count_mean": float(footprint_count.float().mean().item()),
        "opacity_mean_after": float(opacities.detach().float().mean().item()),
        "opacity_max_after": float(opacities.detach().float().max().item()),
    }
    return fields, source_ids, stats


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_output_packet(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    packet_fields: List[Dict[str, torch.Tensor]],
    source_ids_list: List[torch.Tensor],
    output_pt: Path,
    meta: Dict[str, Any],
) -> None:
    fused: Dict[str, torch.Tensor] = {}
    for k in gaussian_keys:
        fused[k] = torch.cat([pf[k] for pf in packet_fields], dim=0).detach().cpu().contiguous()
    source_ids = torch.cat(source_ids_list, dim=0).detach().cpu().contiguous()

    base = dict(packets[selected_indices[0]].data)
    for k in ALL_GAUSSIAN_KEYS:
        if k in fused:
            base[k] = fused[k]
        elif k in base:
            del base[k]

    base["source_packet_sorted_index"] = source_ids
    base["fusion_source_packet_names"] = [packets[i].path.name for i in selected_indices]
    base["trajectory_aware_filter_meta"] = meta

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_pt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-aware packet filter P2")
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--packet_ranges", type=str, required=True)
    parser.add_argument("--trajectory_ranges", type=str, default=None)
    parser.add_argument("--trajectory_stride", type=int, default=1)
    parser.add_argument("--output_pt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)

    parser.add_argument("--opacity_cap", type=float, default=0.0)

    parser.add_argument("--corridor_radius", type=float, default=0.0)
    parser.add_argument("--corridor_mode", choices=["none", "downweight", "hard"], default="downweight")
    parser.add_argument("--corridor_downweight", type=float, default=0.02)

    parser.add_argument("--near_z_thresh", type=float, default=0.0)
    parser.add_argument("--near_mode", choices=["none", "downweight", "hard"], default="downweight")
    parser.add_argument("--near_downweight", type=float, default=0.02)
    parser.add_argument("--near_min_count", type=int, default=1)

    parser.add_argument("--footprint_px_thresh", type=float, default=0.0)
    parser.add_argument("--footprint_mode", choices=["none", "downweight", "hard"], default="downweight")
    parser.add_argument("--footprint_downweight", type=float, default=0.05)
    parser.add_argument("--footprint_min_count", type=int, default=1)
    parser.add_argument("--footprint_min_opacity", type=float, default=0.05)

    parser.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    parser.add_argument("--scale_metric", choices=["max", "mean", "volume"], default="max")

    parser.add_argument("--chunk_size", type=int, default=300_000)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    parser.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive")
    parser.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading packets from {args.packet_dir}")
    packets = load_packets(args.packet_dir)
    selected_indices = parse_ranges(args.packet_ranges, len(packets))
    trajectory_indices = parse_ranges(args.trajectory_ranges, len(packets)) if args.trajectory_ranges else selected_indices
    gaussian_keys = available_gaussian_keys(packets, selected_indices)

    print(f"      found packets: {len(packets)}")
    print(f"      selected packets: {selected_indices[:10]}{'...' if len(selected_indices)>10 else ''} ({len(selected_indices)} total)")
    print(f"      trajectory packets: {trajectory_indices[:10]}{'...' if len(trajectory_indices)>10 else ''} ({len(trajectory_indices)} total)")
    print(f"      gaussian keys: {gaussian_keys}")

    print("[2/4] Building trajectory cameras")
    views = build_trajectory_views(
        packets=packets,
        trajectory_indices=trajectory_indices,
        device=device,
        intrinsics_normalized=args.intrinsics_normalized,
        extrinsic_type=args.extrinsic_type,
        stride=args.trajectory_stride,
    )
    centers = torch.stack([camera_center(v.Twc_or_Tcw, args.extrinsic_type) for v in views], dim=0).to(device)
    print(f"      trajectory views used: {len(views)}")
    print(f"      corridor centers: {tuple(centers.shape)}")

    print("[3/4] Filtering selected packets")
    packet_fields: List[Dict[str, torch.Tensor]] = []
    source_ids_list: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []

    for order, idx in enumerate(selected_indices):
        fields, src, stats = process_packet(
            packet_ref=packets[idx],
            gaussian_keys=gaussian_keys,
            views=views,
            centers=centers,
            args=args,
            device=device,
        )
        packet_fields.append(fields)
        source_ids_list.append(src)
        rows.append(stats)
        print(
            f"      [{order+1:03d}/{len(selected_indices):03d}] "
            f"packet={idx:04d}, frame={stats['target_frame_index']:06d}, "
            f"in={stats['num_input']:,}, out={stats['num_output']:,}, "
            f"corridor={stats['num_corridor']:,}, near={stats['num_near']:,}, "
            f"footprint={stats['num_footprint']:,}"
        )

    print("[4/4] Saving fused filtered packet and diagnostics")
    total_in = int(sum(r["num_input"] for r in rows))
    total_out = int(sum(r["num_output"] for r in rows))
    total_corr = int(sum(r["num_corridor"] for r in rows))
    total_near = int(sum(r["num_near"] for r in rows))
    total_foot = int(sum(r["num_footprint"] for r in rows))

    meta = {
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "trajectory_ranges": args.trajectory_ranges if args.trajectory_ranges else args.packet_ranges,
        "trajectory_stride": args.trajectory_stride,
        "opacity_cap": args.opacity_cap,
        "corridor_radius": args.corridor_radius,
        "corridor_mode": args.corridor_mode,
        "corridor_downweight": args.corridor_downweight,
        "near_z_thresh": args.near_z_thresh,
        "near_mode": args.near_mode,
        "near_downweight": args.near_downweight,
        "near_min_count": args.near_min_count,
        "footprint_px_thresh": args.footprint_px_thresh,
        "footprint_mode": args.footprint_mode,
        "footprint_downweight": args.footprint_downweight,
        "footprint_min_count": args.footprint_min_count,
        "footprint_min_opacity": args.footprint_min_opacity,
        "scale_input": args.scale_input,
        "scale_metric": args.scale_metric,
        "chunk_size": args.chunk_size,
        "extrinsic_type": args.extrinsic_type,
        "camera_z_sign": args.camera_z_sign,
        "intrinsics_normalized": args.intrinsics_normalized,
        "gaussian_keys": list(gaussian_keys),
        "num_input_gaussians": total_in,
        "num_output_gaussians": total_out,
        "num_corridor": total_corr,
        "num_near": total_near,
        "num_footprint": total_foot,
        "output_ratio": total_out / max(total_in, 1),
    }

    save_output_packet(
        packets=packets,
        selected_indices=selected_indices,
        gaussian_keys=gaussian_keys,
        packet_fields=packet_fields,
        source_ids_list=source_ids_list,
        output_pt=args.output_pt,
        meta=meta,
    )
    print(f"      saved packet: {args.output_pt}")

    csv_path = output_dir / "trajectory_aware_filter_per_packet.csv"
    write_csv(csv_path, rows)
    print(f"      saved csv: {csv_path}")

    summary_path = output_dir / "trajectory_aware_filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"selected_indices": selected_indices, "trajectory_indices": trajectory_indices, **meta}, f, indent=2, ensure_ascii=False)
    print(f"      saved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
