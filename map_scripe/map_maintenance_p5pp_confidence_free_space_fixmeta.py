#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5++ v2: confidence-aware footprint-depth free-space maintenance + P4 redundancy control.

This script is designed for the current ZipMap -> ReSplat packet workflow.

Why P5
------
P4 and P3 have different roles:

  P4:
    voxel-overlap / coarse-to-fine / top-k redundancy control.
    It can reduce the number of Gaussians substantially, but it does not directly
    solve view-dependent white ghost artifacts.

  P3:
    trajectory/depth/extent-aware safety filtering.
    It targets dangerous Gaussians that cause view-dependent occlusion, but hard
    deletion can remove true surfaces.

P5++ v2 extends P5 with confidence-aware footprint-level depth free-space maintenance. P5 combines modules so that the map is loaded/fused only once for
batch ablations, or maintained sequentially for incremental simulation.

Execution modes
---------------
1) --execution_mode batch

   Offline ablation:
     selected packets -> concat once -> P4 -> P3 -> output fused packet

   This is fastest and useful for comparing:
     naive / P4-only / P3-only / P4->P3

   It is NOT strictly online if --p3_trajectory_policy all is used, because it
   can use future trajectory/depth views.

2) --execution_mode incremental

   Sequential map-maintenance simulation:
     map = empty
     for each selected packet:
         insert packet
         apply P4 to current map
         apply P3 using current/past/all trajectory views
     output final map

   For online-style explanation, use:
     --execution_mode incremental
     --p3_trajectory_policy causal

   This is slower, but the semantics are closer to incremental map maintenance.

Output
------
The output .pt follows the same fused packet schema used by P1/P2/P3/P4 and can
be rendered by render_fused_packet_trajectory.py.

Typical commands
----------------
Batch P4 -> P3:

python map_maintenance_p5_p4_p3.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --trajectory_ranges 0-79 \
  --depth_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/depth_lcam_front \
  --depth_pattern "{index:06d}_lcam_front_depth.png" \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p5_batch_p4topk16_p3soft.pt \
  --execution_mode batch \
  --enable_p4 \
  --p4_mode voxel_topk \
  --voxel_size 0.10 \
  --max_gaussians_per_voxel 16 \
  --voxel_topk_score opacity_over_scale \
  --enable_p3 \
  --p3_mode shrink_downweight \
  --p3_opacity_downweight 0.1 \
  --p3_shrink_factor 0.5 \
  --corridor_radius 0.50 \
  --corridor_mode none \
  --near_z_thresh 1.00 \
  --near_mode none \
  --extent_mode shrink_downweight \
  --extent_factor 3.0 \
  --extent_min_count 1 \
  --extent_min_opacity 0.02 \
  --extent_min_radius_px 25 \
  --footprint_px_thresh 50 \
  --footprint_mode shrink_downweight \
  --footprint_min_opacity 0.02 \
  --opacity_cap 0.30 \
  --device cuda:0

Incremental causal P4 -> P3:

python map_maintenance_p5_p4_p3.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --trajectory_ranges 0-79 \
  --depth_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/depth_lcam_front \
  --depth_pattern "{index:06d}_lcam_front_depth.png" \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p5_incr_p4topk16_p3soft.pt \
  --execution_mode incremental \
  --p3_trajectory_policy causal \
  --enable_p4 \
  --p4_mode voxel_topk \
  --voxel_size 0.10 \
  --max_gaussians_per_voxel 16 \
  --enable_p3 \
  --p3_mode shrink_downweight \
  --p3_opacity_downweight 0.1 \
  --p3_shrink_factor 0.5 \
  --corridor_radius 0.50 \
  --near_z_thresh 1.00 \
  --extent_factor 3.0 \
  --extent_min_opacity 0.02 \
  --extent_min_radius_px 25 \
  --footprint_px_thresh 50 \
  --footprint_min_opacity 0.02 \
  --opacity_cap 0.30 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


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
    depth: Optional[torch.Tensor] = None


def now() -> float:
    return time.perf_counter()


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
        raise IndexError(f"Packet indices out of range 0..{n - 1}: {bad[:20]}")
    if not out:
        raise ValueError(f"No packet selected by range spec: {spec}")
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
            raise RuntimeError(f"Unexpected {key} shape after normalization: {tuple(t.shape)}")
    elif key == "covariances":
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected {key} shape after normalization: {tuple(t.shape)}")
    elif key == "harmonics":
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected {key} shape after normalization: {tuple(t.shape)}")
    elif key == "opacities":
        if t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise RuntimeError(f"Unexpected {key} shape after normalization: {tuple(t.shape)}")
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


def infer_image_shape(packet: Dict[str, Any], depth: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    if depth is not None:
        return int(depth.shape[-2]), int(depth.shape[-1])
    if "image_shape" in packet:
        s = packet["image_shape"]
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().reshape(-1).tolist()
        if len(s) >= 2:
            return int(s[0]), int(s[1])
    img = packet.get("target_image", None)
    if isinstance(img, torch.Tensor):
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


def load_depth(path: Path, depth_scale: float, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Depth map not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ("depth", "target_depth", "pred_depth", "zipmap_depth", "depth_map"):
                if k in obj:
                    obj = obj[k]
                    break
        arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    elif suffix in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        raw = np.array(Image.open(path))
        # TartanAir v2 depth is often float32 packed in RGBA uint8 PNG.
        if raw.ndim == 3 and raw.shape[-1] == 4 and raw.dtype == np.uint8:
            arr = np.ascontiguousarray(raw).view("<f4").reshape(raw.shape[0], raw.shape[1])
        else:
            arr = raw.astype(np.float32) / float(depth_scale)
    elif suffix == ".exr":
        try:
            import imageio.v3 as iio  # type: ignore
            arr = iio.imread(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read EXR depth {path}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported depth format: {path}")
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr.astype(np.float32)).to(device=device)


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
            raise KeyError(f"Packet {path.name} missing primary keys: {missing}")
        refs.append(PacketRef(i, path, data))
    return refs


def available_gaussian_keys(packets: Sequence[PacketRef], selected_indices: Sequence[int]) -> List[str]:
    keys = [
        k for k in ALL_GAUSSIAN_KEYS
        if all(k in packets[i].data and packets[i].data[k] is not None for i in selected_indices)
    ]
    missing = [k for k in PRIMARY_GAUSSIAN_KEYS if k not in keys]
    if missing:
        raise KeyError(f"Missing primary Gaussian keys in selected packets: {missing}")
    return keys


def build_trajectory_views(
    packets: Sequence[PacketRef],
    trajectory_indices: Sequence[int],
    device: torch.device,
    intrinsics_normalized: str,
    depth_dir: Optional[Path],
    depth_pattern: str,
    depth_scale: float,
    stride: int,
) -> List[TrajectoryView]:
    if stride <= 0:
        stride = 1
    views: List[TrajectoryView] = []
    for idx in list(trajectory_indices)[::stride]:
        p = packets[idx].data
        if "target_extrinsics" not in p or "target_intrinsics" not in p:
            raise KeyError(f"Trajectory packet {packets[idx].path.name} lacks target_extrinsics/target_intrinsics")
        frame_index = first_int(p.get("target_index", idx), default=idx)
        depth = None
        if depth_dir is not None:
            depth = load_depth(depth_dir / depth_pattern.format(index=frame_index), depth_scale, device)
        H, W = infer_image_shape(p, depth)
        E = first_matrix(p["target_extrinsics"], device)
        K = denormalize_intrinsics(first_matrix(p["target_intrinsics"], device), H, W, intrinsics_normalized)
        views.append(TrajectoryView(int(idx), int(frame_index), E, K, H, W, depth))
    return views


def load_packet_fields(
    packet_ref: PacketRef,
    gaussian_keys: Sequence[str],
    device: torch.device,
    next_uid_start: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    fields: Dict[str, torch.Tensor] = {}
    n_this: Optional[int] = None
    for k in gaussian_keys:
        t = normalize_gaussian_tensor(packet_ref.data[k], k, device)
        if n_this is None:
            n_this = int(t.shape[0])
        elif int(t.shape[0]) != n_this:
            raise RuntimeError(f"Packet {packet_ref.path.name}: {k} length {t.shape[0]} != {n_this}")
        fields[k] = t
    assert n_this is not None
    source_ids = torch.full((n_this,), int(packet_ref.sorted_index), device=device, dtype=torch.long)
    uids = torch.arange(next_uid_start, next_uid_start + n_this, device=device, dtype=torch.long)
    row = {
        "source_packet_sorted_index": int(packet_ref.sorted_index),
        "source_packet_name": packet_ref.path.name,
        "num_input": int(n_this),
    }
    return fields, source_ids, uids, row


def concat_maps(
    fields_list: Sequence[Dict[str, torch.Tensor]],
    source_list: Sequence[torch.Tensor],
    uid_list: Sequence[torch.Tensor],
    gaussian_keys: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    out_fields = {k: torch.cat([f[k] for f in fields_list], dim=0).contiguous() for k in gaussian_keys}
    out_source = torch.cat(list(source_list), dim=0).contiguous()
    out_uid = torch.cat(list(uid_list), dim=0).contiguous()
    return out_fields, out_source, out_uid


def append_map(
    map_fields: Optional[Dict[str, torch.Tensor]],
    map_source: Optional[torch.Tensor],
    map_uid: Optional[torch.Tensor],
    packet_fields: Dict[str, torch.Tensor],
    packet_source: torch.Tensor,
    packet_uid: torch.Tensor,
    gaussian_keys: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if map_fields is None:
        return packet_fields, packet_source, packet_uid
    fields = {k: torch.cat([map_fields[k], packet_fields[k]], dim=0).contiguous() for k in gaussian_keys}
    source = torch.cat([map_source, packet_source], dim=0).contiguous()
    uid = torch.cat([map_uid, packet_uid], dim=0).contiguous()
    return fields, source, uid


def apply_keep_mask(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    uids: torch.Tensor,
    keep_mask: torch.Tensor,
    gaussian_keys: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    fields = {k: fields[k][keep_mask].contiguous() for k in gaussian_keys}
    return fields, source_ids[keep_mask].contiguous(), uids[keep_mask].contiguous()


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


def compute_voxel_keys(
    means: torch.Tensor,
    voxel_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if voxel_size <= 0:
        raise ValueError("--voxel_size must be positive")
    coords = torch.floor(means / float(voxel_size)).to(torch.long)
    coord_min = coords.min(dim=0).values
    coord_max = coords.max(dim=0).values
    dims = coord_max - coord_min + 1
    shifted = coords - coord_min
    keys = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]
    return coords, keys.contiguous(), coord_min, dims


def compute_group_index(keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    group_id_sorted = torch.repeat_interleave(
        torch.arange(counts.numel(), device=keys.device, dtype=torch.long),
        counts,
    )
    group_id = torch.empty_like(group_id_sorted)
    group_id[order] = group_id_sorted
    return order, unique_keys, counts, group_id


def scatter_group_stats(
    group_id: torch.Tensor,
    num_groups: int,
    opacities: torch.Tensor,
    scale_metric: torch.Tensor,
    source_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = group_id.device
    opacity_sum = torch.zeros((num_groups,), device=device, dtype=torch.float32)
    opacity_sum.scatter_add_(0, group_id, opacities.float())

    scale_min = torch.full((num_groups,), float("inf"), device=device, dtype=torch.float32)
    scale_max = torch.full((num_groups,), -float("inf"), device=device, dtype=torch.float32)
    scale_min.scatter_reduce_(0, group_id, scale_metric.float(), reduce="amin", include_self=True)
    scale_max.scatter_reduce_(0, group_id, scale_metric.float(), reduce="amax", include_self=True)

    src_min = torch.full((num_groups,), int(2**62 - 1), device=device, dtype=torch.long)
    src_max = torch.full((num_groups,), -1, device=device, dtype=torch.long)
    src_min.scatter_reduce_(0, group_id, source_ids.long(), reduce="amin", include_self=True)
    src_max.scatter_reduce_(0, group_id, source_ids.long(), reduce="amax", include_self=True)

    return {
        "opacity_sum": opacity_sum,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "source_min": src_min,
        "source_max": src_max,
    }


def make_score(
    score_name: str,
    opacities: torch.Tensor,
    scale_metric: torch.Tensor,
    source_ids: torch.Tensor,
    newer_bonus: float,
) -> torch.Tensor:
    eps = 1e-8
    if score_name == "small_scale":
        score = -scale_metric.float()
    elif score_name == "opacity_over_scale":
        score = opacities.float().clamp(min=0) / torch.clamp(scale_metric.float(), min=eps)
    elif score_name == "opacity":
        score = opacities.float()
    elif score_name == "newer":
        src = source_ids.float()
        denom = torch.clamp(src.max() - src.min(), min=1.0)
        score = (src - src.min()) / denom
    elif score_name == "newer_small":
        src = source_ids.float()
        denom = torch.clamp(src.max() - src.min(), min=1.0)
        score = -scale_metric.float() + float(newer_bonus) * ((src - src.min()) / denom)
    else:
        raise ValueError(f"Unknown score: {score_name}")
    return score.float()


def grouped_rank_by_score(keys: torch.Tensor, score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    order_score = torch.argsort(-score, stable=True)
    keys_after_score = keys[order_score]
    order_key = torch.argsort(keys_after_score, stable=True)
    order = order_score[order_key]
    keys_sorted = keys[order]
    _, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    local_rank = torch.arange(keys.numel(), device=keys.device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
    return order, counts, starts, local_rank


def apply_p4(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    uids: torch.Tensor,
    args: argparse.Namespace,
    p4_seen_uids: Optional[set[int]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], set[int]]:
    if not args.enable_p4 or args.p4_mode == "none":
        return fields, source_ids, uids, {"p4_enabled": False}, p4_seen_uids or set()

    t0 = now()
    opacities = fields["opacities"]
    if args.opacity_cap > 0:
        opacities = opacities.clone()
        opacities.clamp_(max=float(args.opacity_cap))
        fields["opacities"] = opacities

    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    coords, keys, coord_min, dims = compute_voxel_keys(fields["means"], args.voxel_size)
    order_by_key, unique_keys, counts, group_id = compute_group_index(keys)
    group_stats = scatter_group_stats(group_id, int(unique_keys.numel()), opacities, scale_metric, source_ids)
    keep_mask = torch.ones((opacities.shape[0],), device=opacities.device, dtype=torch.bool)
    modified_mask = torch.zeros_like(keep_mask)

    if args.p4_mode == "coarse_to_fine":
        voxel_min_scale = group_stats["scale_min"][group_id]
        voxel_max_source = group_stats["source_max"][group_id]
        voxel_count = counts[group_id]
        candidate = (
            (source_ids < voxel_max_source)
            & (scale_metric > float(args.scale_ratio) * voxel_min_scale)
            & (voxel_count >= int(args.min_voxel_count))
            & (opacities >= float(args.old_min_opacity))
        )
        if args.apply_once and p4_seen_uids:
            seen_tensor = torch.tensor(list(p4_seen_uids), device=uids.device, dtype=torch.long)
            # torch.isin is acceptable here; seen set remains much smaller than all Gaussians in early iterations.
            candidate = candidate & (~torch.isin(uids, seen_tensor))
        modified_mask |= candidate
        if args.old_mode == "hard":
            keep_mask[candidate] = False
        else:
            fields["opacities"][candidate] = fields["opacities"][candidate] * float(args.old_downweight)

    elif args.p4_mode == "voxel_topk":
        score = make_score(args.voxel_topk_score, opacities, scale_metric, source_ids, args.newer_bonus)
        order, _, _, local_rank = grouped_rank_by_score(keys, score)
        keep_sorted = local_rank < int(args.max_gaussians_per_voxel)
        keep_topk = torch.zeros_like(keep_mask)
        keep_topk[order] = keep_sorted
        modified_mask |= ~keep_topk
        keep_mask &= keep_topk

    elif args.p4_mode == "opacity_budget":
        if args.opacity_budget <= 0:
            raise ValueError("--p4_mode opacity_budget requires --opacity_budget > 0")
        opacity_sum = group_stats["opacity_sum"]
        over = opacity_sum > float(args.opacity_budget)
        if args.opacity_budget_mode == "uniform_downweight":
            group_factor = torch.ones_like(opacity_sum)
            group_factor[over] = float(args.opacity_budget) / torch.clamp(opacity_sum[over], min=1e-8)
            factor = group_factor[group_id]
            changed = factor < 0.999999
            if args.apply_once and p4_seen_uids:
                seen_tensor = torch.tensor(list(p4_seen_uids), device=uids.device, dtype=torch.long)
                changed = changed & (~torch.isin(uids, seen_tensor))
            modified_mask |= changed
            fields["opacities"][changed] = fields["opacities"][changed] * factor[changed]
        else:
            score = make_score(args.opacity_budget_score, opacities, scale_metric, source_ids, args.newer_bonus)
            order, counts2, starts, local_rank = grouped_rank_by_score(keys, score)
            op_sorted = opacities.float().clamp(min=0)[order]
            cum = torch.cumsum(op_sorted, dim=0)
            start_cum = cum[starts] - op_sorted[starts]
            local_cum = cum - torch.repeat_interleave(start_cum, counts2)
            keep_sorted = (local_cum <= float(args.opacity_budget)) | (local_rank == 0)
            keep_budget = torch.zeros_like(keep_mask)
            keep_budget[order] = keep_sorted
            modified_mask |= ~keep_budget
            keep_mask &= keep_budget
    else:
        raise ValueError(f"Unsupported p4_mode: {args.p4_mode}")

    modified_count = int(modified_mask.sum().item())
    removed_count = int((~keep_mask).sum().item())
    if args.apply_once:
        if p4_seen_uids is None:
            p4_seen_uids = set()
        if modified_count > 0:
            p4_seen_uids.update(uids[modified_mask].detach().cpu().tolist())

    if torch.any(~keep_mask):
        fields, source_ids, uids = apply_keep_mask(fields, source_ids, uids, keep_mask, available_keys_from_fields(fields))

    meta = {
        "p4_enabled": True,
        "p4_mode": args.p4_mode,
        "voxel_size": args.voxel_size,
        "num_occupied_voxels": int(unique_keys.numel()),
        "voxel_count_mean": float(counts.float().mean().item()),
        "voxel_count_max": int(counts.max().item()),
        "scale_input": args.scale_input,
        "scale_metric": args.scale_metric,
        "scale_ratio": args.scale_ratio,
        "max_gaussians_per_voxel": args.max_gaussians_per_voxel,
        "num_p4_modified_or_downweighted": modified_count,
        "num_p4_hard_removed": removed_count,
        "p4_time_sec": now() - t0,
    }
    return fields, source_ids, uids, meta, p4_seen_uids or set()


def available_keys_from_fields(fields: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k in ALL_GAUSSIAN_KEYS if k in fields]


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


def camera_center(E: torch.Tensor, extrinsic_type: str) -> torch.Tensor:
    if extrinsic_type == "Twc":
        return E[:3, 3]
    if extrinsic_type == "Tcw":
        return torch.linalg.inv(E)[:3, 3]
    raise ValueError("extrinsic_type must be Twc/Tcw")


def compute_corridor_mask(means: torch.Tensor, centers: torch.Tensor, radius: float, chunk_size: int) -> torch.Tensor:
    N = int(means.shape[0])
    mask = torch.zeros((N,), device=means.device, dtype=torch.bool)
    if radius <= 0 or centers.numel() == 0:
        return mask
    r2 = float(radius) ** 2
    for start in range(0, N, int(chunk_size)):
        end = min(start + int(chunk_size), N)
        d2 = torch.cdist(means[start:end], centers) ** 2
        mask[start:end] = torch.min(d2, dim=1).values <= r2
    return mask


def build_footprint_sample_offsets(pattern: str, device: torch.device) -> torch.Tensor:
    """Normalized offsets multiplied by projected Gaussian radius in pixels."""
    if pattern == "center":
        pts = [(0.0, 0.0)]
    elif pattern == "cross5":
        pts = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    elif pattern == "star9":
        pts = [
            (0.0, 0.0),
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
        ]
    elif pattern == "ring13":
        pts = [
            (0.0, 0.0),
            (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
        ]
    else:
        raise ValueError("--fs_sample_pattern must be center/cross5/star9/ring13")
    return torch.tensor(pts, device=device, dtype=torch.float32)


def compute_p3_masks(
    fields: Dict[str, torch.Tensor],
    views: Sequence[TrajectoryView],
    centers: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Return hard_mask, soft_mask, stats.

    hard_mask:
      Gaussians to remove.

    soft_mask:
      Gaussians to shrink/downweight. In hybrid mode this is weaker than hard_mask.
      In non-hybrid modes it is currently only used for diagnostics.

    P5++ v2 principle:
      - high-confidence footprint free-space violation -> hard delete
      - medium-confidence violation -> shrink/downweight
      - center-depth-consistent Gaussians are protected from hard deletion
    """
    means = fields["means"]
    opacities = fields["opacities"]
    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    N = int(means.shape[0])
    device = means.device

    corridor_mask = compute_corridor_mask(means, centers, args.corridor_radius, args.chunk_size)

    near_count = torch.zeros((N,), device=device, dtype=torch.int16)
    footprint_count = torch.zeros((N,), device=device, dtype=torch.int16)
    extent_count = torch.zeros((N,), device=device, dtype=torch.int16)

    # P5++ v2 confidence-aware free-space stats.
    fs_soft_count = torch.zeros((N,), device=device, dtype=torch.int16)
    fs_hard_count = torch.zeros((N,), device=device, dtype=torch.int16)
    fs_valid_count = torch.zeros((N,), device=device, dtype=torch.int16)
    fs_center_protect_count = torch.zeros((N,), device=device, dtype=torch.int16)

    use_fs = getattr(args, "footprint_free_space_mode", "none") != "none"
    fs_offsets = build_footprint_sample_offsets(args.fs_sample_pattern, device) if use_fs else None

    for view in views:
        H, W = int(view.H), int(view.W)
        f = float((view.K_pixel[0, 0] + view.K_pixel[1, 1]) * 0.5)
        has_depth = view.depth is not None

        for start in range(0, N, int(args.chunk_size)):
            end = min(start + int(args.chunk_size), N)
            u, v, z = project_points(means[start:end], view.Twc_or_Tcw, view.K_pixel, args.extrinsic_type, args.camera_z_sign)
            ui = torch.round(u).long()
            vi = torch.round(v).long()
            inside_center = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

            if args.near_z_thresh > 0:
                nm = inside_center & (z < float(args.near_z_thresh))
                near_count[start:end][nm] += 1

            radius_px = f * scale_metric[start:end] / torch.clamp(z, min=1e-6)

            if args.footprint_px_thresh > 0:
                fm = inside_center & (radius_px > float(args.footprint_px_thresh)) & (opacities[start:end] >= float(args.footprint_min_opacity))
                footprint_count[start:end][fm] += 1

            # Original center-projection extent test.
            if has_depth and args.extent_mode != "none":
                valid_idx = torch.nonzero(inside_center, as_tuple=False).reshape(-1)
                if valid_idx.numel() > 0:
                    d = view.depth[vi[valid_idx], ui[valid_idx]]
                    d_valid = torch.isfinite(d) & (d > 0)
                    if torch.any(d_valid):
                        valid_idx = valid_idx[d_valid]
                        d = d[d_valid]
                        z_valid = z[valid_idx]
                        s_valid = scale_metric[start:end][valid_idx]
                        rpx_valid = radius_px[valid_idx]
                        opa_valid = opacities[start:end][valid_idx]
                        margin = float(args.margin_abs) + float(args.margin_rel) * d
                        front_z = z_valid - float(args.extent_factor) * s_valid
                        em = (
                            (front_z < (d - margin))
                            & (rpx_valid >= float(args.extent_min_radius_px))
                            & (opa_valid >= float(args.extent_min_opacity))
                        )
                        if torch.any(em):
                            extent_count[start + valid_idx[em]] += 1

            # P5++ v2 footprint-level depth free-space maintenance.
            if has_depth and use_fs:
                r_clip = torch.clamp(radius_px, min=0.0, max=float(args.fs_radius_clip_px))
                footprint_intersects = (
                    (z > 0)
                    & (radius_px >= float(args.fs_min_radius_px))
                    & (opacities[start:end] >= float(args.fs_min_opacity))
                    & ((u + r_clip) >= 0)
                    & ((u - r_clip) < W)
                    & ((v + r_clip) >= 0)
                    & ((v - r_clip) < H)
                )
                cand_idx = torch.nonzero(footprint_intersects, as_tuple=False).reshape(-1)
                if cand_idx.numel() > 0:
                    offs = fs_offsets
                    su = u[cand_idx, None] + offs[None, :, 0] * r_clip[cand_idx, None]
                    sv = v[cand_idx, None] + offs[None, :, 1] * r_clip[cand_idx, None]
                    sui = torch.round(su).long()
                    svi = torch.round(sv).long()
                    in_img = (sui >= 0) & (sui < W) & (svi >= 0) & (svi < H)

                    M, S = sui.shape
                    depth_samples = torch.zeros((M, S), device=device, dtype=torch.float32)
                    if torch.any(in_img):
                        depth_samples[in_img] = view.depth[svi[in_img], sui[in_img]]
                    depth_valid = in_img & torch.isfinite(depth_samples) & (depth_samples > 0)
                    valid_n = depth_valid.sum(dim=1)

                    if torch.any(valid_n >= int(args.fs_min_valid_samples)):
                        z_front = z[cand_idx] - float(args.fs_extent_factor) * scale_metric[start:end][cand_idx]

                        soft_margin = float(args.fs_margin_abs) + float(args.fs_margin_rel) * depth_samples
                        hard_margin = float(args.fs_hard_margin_abs) + float(args.fs_margin_rel) * depth_samples

                        soft_violation = depth_valid & (z_front[:, None] < (depth_samples - soft_margin))
                        hard_violation = depth_valid & (z_front[:, None] < (depth_samples - hard_margin))

                        soft_vio_n = soft_violation.sum(dim=1)
                        hard_vio_n = hard_violation.sum(dim=1)
                        soft_ratio = soft_vio_n.float() / torch.clamp(valid_n.float(), min=1.0)
                        hard_ratio = hard_vio_n.float() / torch.clamp(valid_n.float(), min=1.0)

                        soft_fs = (
                            (valid_n >= int(args.fs_min_valid_samples))
                            & (soft_vio_n >= int(args.fs_soft_min_violation_samples))
                            & (soft_ratio >= float(args.fs_soft_violation_ratio))
                        )
                        hard_fs = (
                            (valid_n >= int(args.fs_min_valid_samples))
                            & (hard_vio_n >= int(args.fs_hard_min_violation_samples))
                            & (hard_ratio >= float(args.fs_hard_violation_ratio))
                        )

                        # Protect true visible surfaces: if the Gaussian center is inside the image
                        # and close to the current depth, do not hard delete it.
                        # It may still be softly shrunk/downweighted if the footprint crosses free-space.
                        if args.fs_center_protect:
                            center_valid = torch.zeros((cand_idx.numel(),), device=device, dtype=torch.bool)
                            center_consistent = torch.zeros((cand_idx.numel(),), device=device, dtype=torch.bool)
                            ci = inside_center[cand_idx]
                            if torch.any(ci):
                                d_center = torch.zeros((cand_idx.numel(),), device=device, dtype=torch.float32)
                                d_center[ci] = view.depth[vi[cand_idx[ci]], ui[cand_idx[ci]]]
                                center_valid = ci & torch.isfinite(d_center) & (d_center > 0)
                                center_margin = float(args.fs_center_protect_margin_abs) + float(args.fs_center_protect_margin_rel) * d_center
                                center_consistent = center_valid & (torch.abs(z[cand_idx] - d_center) <= center_margin)
                                hard_fs = hard_fs & (~center_consistent)
                                if torch.any(center_consistent):
                                    fs_center_protect_count[start + cand_idx[center_consistent]] += 1

                        if torch.any(soft_fs):
                            fs_soft_count[start + cand_idx[soft_fs]] += 1
                        if torch.any(hard_fs):
                            fs_hard_count[start + cand_idx[hard_fs]] += 1

                        fs_valid = valid_n >= int(args.fs_min_valid_samples)
                        if torch.any(fs_valid):
                            fs_valid_count[start + cand_idx[fs_valid]] += 1

    near_mask = near_count >= int(args.near_min_count)
    footprint_mask = footprint_count >= int(args.footprint_min_count)
    extent_mask = extent_count >= int(args.extent_min_count)
    fs_soft_mask = fs_soft_count >= int(args.fs_min_count)
    fs_hard_mask = fs_hard_count >= int(args.fs_min_count)

    # Non-FS P3 components remain hard components if enabled.
    hard_mask = torch.zeros((N,), device=device, dtype=torch.bool)
    if args.corridor_mode != "none":
        hard_mask |= corridor_mask
    if args.near_mode != "none":
        hard_mask |= near_mask
    if args.footprint_mode != "none":
        hard_mask |= footprint_mask
    if args.extent_mode != "none":
        hard_mask |= extent_mask
    if use_fs:
        hard_mask |= fs_hard_mask

    soft_mask = torch.zeros((N,), device=device, dtype=torch.bool)
    if use_fs:
        # Soft mask excludes hard mask so that each Gaussian has one action.
        soft_mask |= (fs_soft_mask & (~hard_mask))

    stats = {
        "num_corridor": int(corridor_mask.sum().item()),
        "num_near": int(near_mask.sum().item()),
        "num_footprint": int(footprint_mask.sum().item()),
        "num_extent": int(extent_mask.sum().item()),
        "num_footprint_free_space_soft": int(fs_soft_mask.sum().item()),
        "num_footprint_free_space_hard": int(fs_hard_mask.sum().item()),
        "num_footprint_free_space_center_protected": int((fs_center_protect_count >= int(args.fs_min_count)).sum().item()),
        "num_p3_hard_union": int(hard_mask.sum().item()),
        "num_p3_soft_union": int(soft_mask.sum().item()),
        "near_count_mean": float(near_count.float().mean().item()),
        "footprint_count_mean": float(footprint_count.float().mean().item()),
        "extent_count_mean": float(extent_count.float().mean().item()),
        "footprint_free_space_soft_count_mean": float(fs_soft_count.float().mean().item()),
        "footprint_free_space_hard_count_mean": float(fs_hard_count.float().mean().item()),
        "footprint_free_space_valid_count_mean": float(fs_valid_count.float().mean().item()),
    }
    return hard_mask, soft_mask, stats

def apply_p3(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    uids: torch.Tensor,
    views: Sequence[TrajectoryView],
    args: argparse.Namespace,
    p3_seen_uids: Optional[set[int]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], set[int]]:
    if not args.enable_p3 or args.p3_mode == "none" or len(views) == 0:
        return fields, source_ids, uids, {"p3_enabled": False}, p3_seen_uids or set()

    t0 = now()
    centers = torch.stack([camera_center(v.Twc_or_Tcw, args.extrinsic_type) for v in views], dim=0).to(fields["means"].device)
    hard_mask, soft_mask, stats = compute_p3_masks(fields, views, centers, args)

    action_mask = hard_mask | soft_mask
    if args.apply_once and p3_seen_uids:
        seen_tensor = torch.tensor(list(p3_seen_uids), device=uids.device, dtype=torch.long)
        unseen = ~torch.isin(uids, seen_tensor)
        hard_mask = hard_mask & unseen
        soft_mask = soft_mask & unseen
        action_mask = hard_mask | soft_mask

    keep_mask = torch.ones_like(hard_mask)

    if args.p3_mode == "hard":
        keep_mask[hard_mask | soft_mask] = False
    elif args.p3_mode == "downweight":
        fields["opacities"][hard_mask | soft_mask] = fields["opacities"][hard_mask | soft_mask] * float(args.p3_opacity_downweight)
    elif args.p3_mode == "shrink":
        shrink_danger_gaussians(fields, hard_mask | soft_mask, args.p3_shrink_factor)
    elif args.p3_mode == "shrink_downweight":
        mask = hard_mask | soft_mask
        fields["opacities"][mask] = fields["opacities"][mask] * float(args.p3_opacity_downweight)
        shrink_danger_gaussians(fields, mask, args.p3_shrink_factor)
    elif args.p3_mode == "hybrid":
        # Confidence-aware default:
        #   hard_mask: remove
        #   soft_mask: shrink/downweight
        keep_mask[hard_mask] = False
        if torch.any(soft_mask):
            fields["opacities"][soft_mask] = fields["opacities"][soft_mask] * float(args.fs_soft_opacity_downweight)
            shrink_danger_gaussians(fields, soft_mask, args.fs_soft_shrink_factor)
    else:
        raise ValueError(f"Unsupported p3_mode: {args.p3_mode}")

    hard_count = int(hard_mask.sum().item())
    soft_count = int(soft_mask.sum().item())
    action_count = int(action_mask.sum().item())
    removed_count = int((~keep_mask).sum().item())

    if args.apply_once:
        if p3_seen_uids is None:
            p3_seen_uids = set()
        if action_count > 0:
            p3_seen_uids.update(uids[action_mask].detach().cpu().tolist())

    if torch.any(~keep_mask):
        fields, source_ids, uids = apply_keep_mask(fields, source_ids, uids, keep_mask, available_keys_from_fields(fields))

    stats.update({
        "p3_enabled": True,
        "p3_mode": args.p3_mode,
        "p3_views_used": len(views),
        "num_p3_hard_action": hard_count,
        "num_p3_soft_action": soft_count,
        "num_p3_modified_or_downweighted": action_count,
        "num_p3_hard_removed": removed_count,
        "p3_opacity_downweight": args.p3_opacity_downweight,
        "p3_shrink_factor": args.p3_shrink_factor,
        "fs_soft_opacity_downweight": getattr(args, "fs_soft_opacity_downweight", None),
        "fs_soft_shrink_factor": getattr(args, "fs_soft_shrink_factor", None),
        "p3_time_sec": now() - t0,
    })
    return fields, source_ids, uids, stats, p3_seen_uids or set()

def shrink_danger_gaussians(fields: Dict[str, torch.Tensor], mask: torch.Tensor, factor: float) -> None:
    if factor <= 0:
        raise ValueError("--p3_shrink_factor must be positive")
    if not torch.any(mask):
        return
    f = float(factor)
    if "scales" in fields and fields["scales"] is not None:
        fields["scales"][mask] = fields["scales"][mask] * f
    if "covariances" in fields and fields["covariances"] is not None:
        fields["covariances"][mask] = fields["covariances"][mask] * (f * f)


def select_views_for_step(
    all_views: Sequence[TrajectoryView],
    current_packet_index: int,
    policy: str,
) -> List[TrajectoryView]:
    if policy == "all":
        return list(all_views)
    if policy == "causal":
        return [v for v in all_views if int(v.sorted_index) <= int(current_packet_index)]
    if policy == "current":
        return [v for v in all_views if int(v.sorted_index) == int(current_packet_index)]
    raise ValueError("--p3_trajectory_policy must be all/causal/current")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Rows from different incremental steps may contain different keys:
    # step 0 has no pre-insert P3 stats, later steps do.
    # Use a stable union of all keys instead of keys from rows[0].
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def update_per_packet_rows(
    rows: List[Dict[str, Any]],
    source_ids: torch.Tensor,
    keep_count_total: int,
    fields: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    source_cpu = source_ids.detach().cpu()
    opacities = fields["opacities"].detach().float().cpu()
    scales = fields["scales"].detach().float().cpu() if "scales" in fields else None
    for row in rows:
        sid = int(row["source_packet_sorted_index"])
        m = source_cpu == sid
        n = int(m.sum().item())
        row["num_output"] = n
        row["output_ratio"] = float(n / max(int(row["num_input"]), 1))
        if n > 0:
            row["opacity_mean_after"] = float(opacities[m].mean().item())
            row["opacity_max_after"] = float(opacities[m].max().item())
            if scales is not None:
                row["scale_abs_mean_after"] = float(scales[m].abs().mean().item())
                row["scale_abs_max_after"] = float(scales[m].abs().max().item())
        else:
            row["opacity_mean_after"] = None
            row["opacity_max_after"] = None
            row["scale_abs_mean_after"] = None
            row["scale_abs_max_after"] = None
    return rows


def save_output_packet(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    output_pt: Path,
    meta: Dict[str, Any],
) -> None:
    base = dict(packets[selected_indices[0]].data)
    for k in ALL_GAUSSIAN_KEYS:
        if k in gaussian_keys and k in fields:
            base[k] = fields[k].detach().cpu().contiguous()
        elif k in base:
            del base[k]
    base["source_packet_sorted_index"] = source_ids.detach().cpu().contiguous()
    base["fusion_source_packet_names"] = [packets[i].path.name for i in selected_indices]
    base["p5_map_maintenance_meta"] = meta
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_pt)


def run_batch(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    views: Sequence[TrajectoryView],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.maintenance_order not in {"p4_p3", "p3_p4"}:
        raise ValueError(
            f"Batch mode supports only --maintenance_order p4_p3 or p3_p4, "
            f"got {args.maintenance_order}. Use incremental mode for pre-insert semantics."
        )

    fields_list, source_list, uid_list, per_packet_rows = [], [], [], []
    uid_cursor = 0
    for idx in selected_indices:
        f, src, uid, row = load_packet_fields(packets[idx], gaussian_keys, device, uid_cursor)
        uid_cursor += int(uid.numel())
        fields_list.append(f)
        source_list.append(src)
        uid_list.append(uid)
        per_packet_rows.append(row)

    fields, source_ids, uids = concat_maps(fields_list, source_list, uid_list, gaussian_keys)
    step_rows: List[Dict[str, Any]] = []
    p4_seen: set[int] = set()
    p3_seen: set[int] = set()

    print(f"[Batch] Initial fused map: {int(fields['means'].shape[0]):,} Gaussians")
    if args.maintenance_order == "p4_p3":
        fields, source_ids, uids, p4_meta, p4_seen = apply_p4(fields, source_ids, uids, args, p4_seen)
        print(f"[Batch] After P4: {int(fields['means'].shape[0]):,} Gaussians")
        fields, source_ids, uids, p3_meta, p3_seen = apply_p3(fields, source_ids, uids, list(views), args, p3_seen)
        print(f"[Batch] After P3: {int(fields['means'].shape[0]):,} Gaussians")
    else:
        fields, source_ids, uids, p3_meta, p3_seen = apply_p3(fields, source_ids, uids, list(views), args, p3_seen)
        print(f"[Batch] After P3: {int(fields['means'].shape[0]):,} Gaussians")
        fields, source_ids, uids, p4_meta, p4_seen = apply_p4(fields, source_ids, uids, args, p4_seen)
        print(f"[Batch] After P4: {int(fields['means'].shape[0]):,} Gaussians")

    step_rows.append({
        "step": 0,
        "mode": "batch",
        "maintenance_order": args.maintenance_order,
        "fs_preset": args.fs_preset,
        "footprint_free_space_mode": args.footprint_free_space_mode,
        "fs_margin_abs": args.fs_margin_abs,
        "fs_hard_margin_abs": args.fs_hard_margin_abs,
        "fs_soft_violation_ratio": args.fs_soft_violation_ratio,
        "fs_hard_violation_ratio": args.fs_hard_violation_ratio,
        "fs_soft_min_violation_samples": args.fs_soft_min_violation_samples,
        "fs_hard_min_violation_samples": args.fs_hard_min_violation_samples,
        "fs_soft_opacity_downweight": args.fs_soft_opacity_downweight,
        "fs_soft_shrink_factor": args.fs_soft_shrink_factor,
        "fs_center_protect": args.fs_center_protect,
        "num_gaussians_after": int(fields["means"].shape[0]),
        **{k: v for k, v in p4_meta.items() if isinstance(v, (int, float, str, bool)) or v is None},
        **{k: v for k, v in p3_meta.items() if isinstance(v, (int, float, str, bool)) or v is None},
    })
    per_packet_rows = update_per_packet_rows(per_packet_rows, source_ids, int(fields["means"].shape[0]), fields)
    return fields, source_ids, per_packet_rows, step_rows

def run_incremental(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    all_views: Sequence[TrajectoryView],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]]]:
    map_fields: Optional[Dict[str, torch.Tensor]] = None
    map_source: Optional[torch.Tensor] = None
    map_uid: Optional[torch.Tensor] = None
    per_packet_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    p4_seen: set[int] = set()
    p3_seen: set[int] = set()
    uid_cursor = 0

    for step_i, idx in enumerate(selected_indices):
        f, src, uid, row = load_packet_fields(packets[idx], gaussian_keys, device, uid_cursor)
        uid_cursor += int(uid.numel())
        per_packet_rows.append(row)

        # For the first packet, there is no previous map to clean.
        n_before_pre_p3 = int(map_fields["means"].shape[0]) if map_fields is not None else 0
        pre_p3_meta: Dict[str, Any] = {"p3_enabled": False}
        post_p3_meta: Dict[str, Any] = {"p3_enabled": False}
        p4_meta: Dict[str, Any] = {"p4_enabled": False}

        if args.maintenance_order in {"pre_p3_insert_p4", "pre_p3_insert_p4_post_p3"}:
            # Step A: clean the existing map BEFORE inserting the incoming packet.
            # This directly targets the problem: old Gaussians occlude the incoming/current view.
            if map_fields is not None and map_source is not None and map_uid is not None:
                pre_views = select_views_for_step(all_views, idx, args.pre_p3_trajectory_policy)
                map_fields, map_source, map_uid, pre_p3_meta, p3_seen = apply_p3(
                    map_fields, map_source, map_uid, pre_views, args, p3_seen
                )

            # Step B: insert incoming packet after cleaning old occluders.
            map_fields, map_source, map_uid = append_map(
                map_fields, map_source, map_uid, f, src, uid, gaussian_keys
            )
            assert map_fields is not None and map_source is not None and map_uid is not None
            n_after_insert = int(map_fields["means"].shape[0])

            # Step C: apply P4 redundancy control after insertion.
            map_fields, map_source, map_uid, p4_meta, p4_seen = apply_p4(
                map_fields, map_source, map_uid, args, p4_seen
            )

            # Optional Step D: retention check after insertion.
            # This can target the reverse problem: new Gaussians occlude old/past views.
            if args.maintenance_order == "pre_p3_insert_p4_post_p3" and args.post_p3_trajectory_policy != "none":
                post_views = select_views_for_step(all_views, idx, args.post_p3_trajectory_policy)
                map_fields, map_source, map_uid, post_p3_meta, p3_seen = apply_p3(
                    map_fields, map_source, map_uid, post_views, args, p3_seen
                )

        else:
            # Legacy incremental semantics: insert first, then apply P4/P3 in the requested order.
            map_fields, map_source, map_uid = append_map(
                map_fields, map_source, map_uid, f, src, uid, gaussian_keys
            )
            assert map_fields is not None and map_source is not None and map_uid is not None
            n_after_insert = int(map_fields["means"].shape[0])

            views = select_views_for_step(all_views, idx, args.p3_trajectory_policy)
            if args.maintenance_order == "p4_p3":
                map_fields, map_source, map_uid, p4_meta, p4_seen = apply_p4(
                    map_fields, map_source, map_uid, args, p4_seen
                )
                map_fields, map_source, map_uid, post_p3_meta, p3_seen = apply_p3(
                    map_fields, map_source, map_uid, views, args, p3_seen
                )
            elif args.maintenance_order == "p3_p4":
                map_fields, map_source, map_uid, post_p3_meta, p3_seen = apply_p3(
                    map_fields, map_source, map_uid, views, args, p3_seen
                )
                map_fields, map_source, map_uid, p4_meta, p4_seen = apply_p4(
                    map_fields, map_source, map_uid, args, p4_seen
                )
            else:
                raise ValueError(f"Unknown maintenance_order: {args.maintenance_order}")

        n_after = int(map_fields["means"].shape[0])
        row_step = {
            "step": int(step_i),
            "inserted_packet_sorted_index": int(idx),
            "maintenance_order": args.maintenance_order,
        "fs_preset": args.fs_preset,
        "footprint_free_space_mode": args.footprint_free_space_mode,
        "fs_margin_abs": args.fs_margin_abs,
        "fs_hard_margin_abs": args.fs_hard_margin_abs,
        "fs_soft_violation_ratio": args.fs_soft_violation_ratio,
        "fs_hard_violation_ratio": args.fs_hard_violation_ratio,
        "fs_soft_min_violation_samples": args.fs_soft_min_violation_samples,
        "fs_hard_min_violation_samples": args.fs_hard_min_violation_samples,
        "fs_soft_opacity_downweight": args.fs_soft_opacity_downweight,
        "fs_soft_shrink_factor": args.fs_soft_shrink_factor,
        "fs_center_protect": args.fs_center_protect,
            "num_gaussians_before_pre_p3": int(n_before_pre_p3),
            "num_gaussians_after_insert": int(n_after_insert),
            "num_gaussians_after_maintenance": int(n_after),
        }
        row_step.update({f"pre_p3_{k}": v for k, v in pre_p3_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        row_step.update({f"p4_{k}": v for k, v in p4_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        row_step.update({f"post_p3_{k}": v for k, v in post_p3_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        step_rows.append(row_step)

        print(
            f"[Incremental {step_i+1:03d}/{len(selected_indices):03d}] "
            f"packet={idx:04d}, before_pre_p3={n_before_pre_p3:,}, "
            f"after_insert={n_after_insert:,}, after_maintenance={n_after:,}"
        )
        if args.empty_cache_each_step and device.type == "cuda":
            torch.cuda.empty_cache()

    assert map_fields is not None and map_source is not None
    per_packet_rows = update_per_packet_rows(per_packet_rows, map_source, int(map_fields["means"].shape[0]), map_fields)
    return map_fields, map_source, per_packet_rows, step_rows

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P5++ v2 confidence-aware footprint free-space maintenance")
    p.add_argument("--packet_dir", type=Path, required=True)
    p.add_argument("--packet_ranges", type=str, required=True)
    p.add_argument("--trajectory_ranges", type=str, default=None)
    p.add_argument("--trajectory_stride", type=int, default=1)
    p.add_argument("--depth_dir", type=Path, default=None)
    p.add_argument("--depth_pattern", type=str, default="{index:06d}.png")
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--output_pt", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--execution_mode", choices=["batch", "incremental"], default="batch")
    p.add_argument(
        "--maintenance_order",
        choices=["p4_p3", "p3_p4", "pre_p3_insert_p4", "pre_p3_insert_p4_post_p3"],
        default="p4_p3",
        help=(
            "Order of maintenance. In batch mode, use p4_p3 or p3_p4. "
            "In incremental mode, pre_p3_insert_p4 means: clean previous map using the incoming/current view, "
            "then insert the new packet, then apply P4. pre_p3_insert_p4_post_p3 additionally applies P3 after insertion "
            "using --post_p3_trajectory_policy."
        ),
    )
    p.add_argument(
        "--pre_p3_trajectory_policy",
        choices=["current", "causal", "all"],
        default="current",
        help="Views used for pre-insert P3 in incremental mode.",
    )
    p.add_argument(
        "--post_p3_trajectory_policy",
        choices=["current", "causal", "all", "none"],
        default="none",
        help="Views used for optional post-insert P3 in incremental mode.",
    )
    p.add_argument("--p3_trajectory_policy", choices=["all", "causal", "current"], default="all")
    p.add_argument("--apply_once", action="store_true", help="In incremental mode, avoid applying repeated downweight/shrink to the same Gaussian uid.")
    p.add_argument("--empty_cache_each_step", action="store_true")

    # Shared Gaussian parameters.
    p.add_argument("--opacity_cap", type=float, default=0.0)
    p.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    p.add_argument("--scale_metric", choices=["max", "mean", "volume"], default="max")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # P4.
    p.add_argument("--enable_p4", action="store_true")
    p.add_argument("--p4_mode", choices=["none", "coarse_to_fine", "voxel_topk", "opacity_budget"], default="none")
    p.add_argument("--voxel_size", type=float, default=0.10)
    p.add_argument("--min_voxel_count", type=int, default=2)
    p.add_argument("--scale_ratio", type=float, default=2.0)
    p.add_argument("--old_mode", choices=["downweight", "hard"], default="hard")
    p.add_argument("--old_downweight", type=float, default=0.1)
    p.add_argument("--old_min_opacity", type=float, default=0.0)
    p.add_argument("--max_gaussians_per_voxel", type=int, default=16)
    p.add_argument("--voxel_topk_score", choices=["small_scale", "opacity_over_scale", "opacity", "newer", "newer_small"], default="opacity_over_scale")
    p.add_argument("--newer_bonus", type=float, default=0.25)
    p.add_argument("--opacity_budget", type=float, default=0.0)
    p.add_argument("--opacity_budget_mode", choices=["uniform_downweight", "hard_topscore"], default="uniform_downweight")
    p.add_argument("--opacity_budget_score", choices=["small_scale", "opacity_over_scale", "opacity", "newer", "newer_small"], default="opacity_over_scale")

    # P3.
    p.add_argument("--enable_p3", action="store_true")
    p.add_argument("--p3_mode", choices=["none", "hard", "downweight", "shrink", "shrink_downweight", "hybrid"], default="none")
    p.add_argument("--p3_opacity_downweight", type=float, default=0.1)
    p.add_argument("--p3_shrink_factor", type=float, default=0.5)

    p.add_argument("--corridor_radius", type=float, default=0.0)
    p.add_argument("--corridor_mode", choices=["none", "active"], default="none")
    p.add_argument("--near_z_thresh", type=float, default=0.0)
    p.add_argument("--near_mode", choices=["none", "active"], default="none")
    p.add_argument("--near_min_count", type=int, default=1)
    p.add_argument("--footprint_px_thresh", type=float, default=0.0)
    p.add_argument("--footprint_mode", choices=["none", "active", "shrink_downweight"], default="none")
    p.add_argument("--footprint_min_count", type=int, default=1)
    p.add_argument("--footprint_min_opacity", type=float, default=0.02)
    p.add_argument("--extent_mode", choices=["none", "active", "shrink_downweight"], default="none")
    p.add_argument("--extent_factor", type=float, default=3.0)
    p.add_argument("--extent_min_count", type=int, default=1)
    p.add_argument("--extent_min_opacity", type=float, default=0.02)
    p.add_argument("--extent_min_radius_px", type=float, default=25.0)
    p.add_argument("--margin_abs", type=float, default=0.10)
    p.add_argument("--margin_rel", type=float, default=0.02)

    # P5++ v2 confidence-aware footprint-level depth free-space maintenance.
    # Main user-facing knobs should be fs_preset, fs_margin_abs, fs_hard_violation_ratio, fs_soft_violation_ratio.
    p.add_argument("--footprint_free_space_mode", choices=["none", "active"], default="none")
    p.add_argument("--fs_preset", choices=["balanced", "conservative", "aggressive", "custom"], default="balanced")
    p.add_argument("--fs_sample_pattern", choices=["center", "cross5", "star9", "ring13"], default="star9")
    p.add_argument("--fs_min_radius_px", type=float, default=2.0)
    p.add_argument("--fs_radius_clip_px", type=float, default=120.0)
    p.add_argument("--fs_min_opacity", type=float, default=0.01)
    p.add_argument("--fs_extent_factor", type=float, default=3.0)
    p.add_argument("--fs_margin_abs", type=float, default=None)
    p.add_argument("--fs_hard_margin_abs", type=float, default=None)
    p.add_argument("--fs_margin_rel", type=float, default=0.02)
    p.add_argument("--fs_min_valid_samples", type=int, default=1)
    p.add_argument("--fs_soft_min_violation_samples", type=int, default=None)
    p.add_argument("--fs_hard_min_violation_samples", type=int, default=None)
    p.add_argument("--fs_soft_violation_ratio", type=float, default=None)
    p.add_argument("--fs_hard_violation_ratio", type=float, default=None)
    p.add_argument("--fs_min_count", type=int, default=1)
    p.add_argument("--fs_soft_opacity_downweight", type=float, default=None)
    p.add_argument("--fs_soft_shrink_factor", type=float, default=None)
    p.add_argument("--fs_center_protect", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fs_center_protect_margin_abs", type=float, default=0.25)
    p.add_argument("--fs_center_protect_margin_rel", type=float, default=0.03)
    p.add_argument("--chunk_size", type=int, default=250_000)
    p.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    p.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive")
    p.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto")
    return p



def apply_fs_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Keep CLI simple by using presets.

    balanced:
      medium free-space violation -> shrink/downweight
      high-confidence violation -> hard delete
      center-depth consistent Gaussians protected from hard delete

    conservative:
      fewer hard deletes; more preservation, more residual ghost likely

    aggressive:
      closer to the first P5++ hard-pruning behavior; more ghost removal, more hole risk
    """
    presets = {
        "balanced": {
            "fs_margin_abs": 0.20,
            "fs_hard_margin_abs": 0.35,
            "fs_soft_min_violation_samples": 1,
            "fs_hard_min_violation_samples": 3,
            "fs_soft_violation_ratio": 0.10,
            "fs_hard_violation_ratio": 0.40,
            "fs_soft_opacity_downweight": 0.20,
            "fs_soft_shrink_factor": 0.50,
        },
        "conservative": {
            "fs_margin_abs": 0.25,
            "fs_hard_margin_abs": 0.45,
            "fs_soft_min_violation_samples": 2,
            "fs_hard_min_violation_samples": 4,
            "fs_soft_violation_ratio": 0.20,
            "fs_hard_violation_ratio": 0.60,
            "fs_soft_opacity_downweight": 0.30,
            "fs_soft_shrink_factor": 0.70,
        },
        "aggressive": {
            "fs_margin_abs": 0.15,
            "fs_hard_margin_abs": 0.25,
            "fs_soft_min_violation_samples": 1,
            "fs_hard_min_violation_samples": 2,
            "fs_soft_violation_ratio": 0.05,
            "fs_hard_violation_ratio": 0.25,
            "fs_soft_opacity_downweight": 0.10,
            "fs_soft_shrink_factor": 0.50,
        },
    }
    if args.fs_preset != "custom":
        cfg = presets[args.fs_preset]
        for k, v in cfg.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    # Fill any still-None values for custom or partial overrides.
    defaults = presets["balanced"]
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    if args.fs_hard_margin_abs < args.fs_margin_abs:
        raise ValueError("--fs_hard_margin_abs should be >= --fs_margin_abs")
    if args.fs_hard_violation_ratio < args.fs_soft_violation_ratio:
        raise ValueError("--fs_hard_violation_ratio should be >= --fs_soft_violation_ratio")
    if args.fs_hard_min_violation_samples < args.fs_soft_min_violation_samples:
        raise ValueError("--fs_hard_min_violation_samples should be >= --fs_soft_min_violation_samples")
    return args

def main() -> None:
    args = build_argparser().parse_args()
    args = apply_fs_preset(args)
    t_total = now()
    device = torch.device(args.device)
    out_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading packets from {args.packet_dir}")
    t0 = now()
    packets = load_packets(args.packet_dir)
    selected_indices = parse_ranges(args.packet_ranges, len(packets))
    trajectory_indices = parse_ranges(args.trajectory_ranges, len(packets)) if args.trajectory_ranges else selected_indices
    gaussian_keys = available_gaussian_keys(packets, selected_indices)
    if "scales" not in gaussian_keys:
        raise KeyError("P5 requires scales in all selected packets.")
    load_packet_time = now() - t0
    print(f"      found packets: {len(packets)}")
    print(f"      selected packets: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''} ({len(selected_indices)} total)")
    print(f"      trajectory packets: {trajectory_indices[:10]}{'...' if len(trajectory_indices) > 10 else ''} ({len(trajectory_indices)} total)")
    print(f"      gaussian keys: {gaussian_keys}")

    print("[2/5] Loading trajectory/depth views")
    t1 = now()
    views = build_trajectory_views(
        packets=packets,
        trajectory_indices=trajectory_indices,
        device=device,
        intrinsics_normalized=args.intrinsics_normalized,
        depth_dir=args.depth_dir if args.enable_p3 else None,
        depth_pattern=args.depth_pattern,
        depth_scale=args.depth_scale,
        stride=args.trajectory_stride,
    )
    load_view_time = now() - t1
    print(f"      views loaded: {len(views)}")

    print(f"[3/5] Running P5 maintenance: execution_mode={args.execution_mode}")
    t2 = now()
    if args.execution_mode == "batch":
        fields, source_ids, per_packet_rows, step_rows = run_batch(
            packets, selected_indices, gaussian_keys, views, args, device
        )
        strict_online = False
    else:
        fields, source_ids, per_packet_rows, step_rows = run_incremental(
            packets, selected_indices, gaussian_keys, views, args, device
        )
        strict_online = args.p3_trajectory_policy in {"causal", "current"}
    maintenance_time = now() - t2

    print("[4/5] Saving output packet")
    t3 = now()
    total_in = int(sum(r["num_input"] for r in per_packet_rows))
    total_out = int(fields["means"].shape[0])
    meta = {
        "script": "map_maintenance_p5pp_confidence_free_space.py",
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "trajectory_ranges": args.trajectory_ranges if args.trajectory_ranges else args.packet_ranges,
        "execution_mode": args.execution_mode,
        "maintenance_order": args.maintenance_order,
        "fs_preset": args.fs_preset,
        "footprint_free_space_mode": args.footprint_free_space_mode,
        "fs_margin_abs": args.fs_margin_abs,
        "fs_hard_margin_abs": args.fs_hard_margin_abs,
        "fs_soft_violation_ratio": args.fs_soft_violation_ratio,
        "fs_hard_violation_ratio": args.fs_hard_violation_ratio,
        "fs_soft_min_violation_samples": args.fs_soft_min_violation_samples,
        "fs_hard_min_violation_samples": args.fs_hard_min_violation_samples,
        "fs_soft_opacity_downweight": args.fs_soft_opacity_downweight,
        "fs_soft_shrink_factor": args.fs_soft_shrink_factor,
        "fs_center_protect": args.fs_center_protect,
        "p3_trajectory_policy": args.p3_trajectory_policy,
        "footprint_free_space_mode": args.footprint_free_space_mode,
        "fs_sample_pattern": args.fs_sample_pattern,
        "fs_min_radius_px": args.fs_min_radius_px,
        "fs_radius_clip_px": args.fs_radius_clip_px,
        "fs_margin_abs": args.fs_margin_abs,
        "fs_margin_rel": args.fs_margin_rel,
        "fs_soft_violation_ratio": args.fs_soft_violation_ratio,
        "fs_hard_violation_ratio": args.fs_hard_violation_ratio,
        "pre_p3_trajectory_policy": args.pre_p3_trajectory_policy,
        "post_p3_trajectory_policy": args.post_p3_trajectory_policy,
        "strict_online_semantics": bool(strict_online),
        "enable_p4": bool(args.enable_p4),
        "p4_mode": args.p4_mode,
        "enable_p3": bool(args.enable_p3),
        "p3_mode": args.p3_mode,
        "num_packets_found": len(packets),
        "selected_indices": selected_indices,
        "gaussian_keys": list(gaussian_keys),
        "num_input_gaussians": total_in,
        "num_output_gaussians": total_out,
        "output_ratio": total_out / max(total_in, 1),
        "opacity_mean_after": float(fields["opacities"].detach().float().mean().item()),
        "opacity_max_after": float(fields["opacities"].detach().float().max().item()),
        "load_packet_time_sec": load_packet_time,
        "load_view_time_sec": load_view_time,
        "maintenance_time_sec": maintenance_time,
    }
    save_output_packet(packets, selected_indices, gaussian_keys, fields, source_ids, args.output_pt, meta)
    save_time = now() - t3

    print("[5/5] Saving diagnostics")
    meta["save_time_sec"] = save_time
    meta["total_time_sec"] = now() - t_total
    summary_path = out_dir / "p5pp_confidence_free_space_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    write_csv(out_dir / "p5pp_confidence_free_space_per_packet.csv", per_packet_rows)
    write_csv(out_dir / "p5pp_confidence_free_space_steps.csv", step_rows)

    print(f"      saved packet: {args.output_pt}")
    print(f"      saved summary: {summary_path}")
    print(f"      saved per-packet csv: {out_dir / 'p5pp_confidence_free_space_per_packet.csv'}")
    print(f"      saved steps csv: {out_dir / 'p5pp_confidence_free_space_steps.csv'}")
    print(f"      total: in={total_in:,}, out={total_out:,}, ratio={total_out / max(total_in, 1):.4f}")
    print(f"      total_time_sec={meta['total_time_sec']:.3f}")


if __name__ == "__main__":
    main()
