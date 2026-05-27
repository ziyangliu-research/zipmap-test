#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5++: footprint-depth free-space pruning + P4 redundancy control.

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

P5++ extends P5 with footprint-level depth free-space pruning. P5 combines modules so that the map is loaded/fused only once for
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
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    means = fields["means"]
    opacities = fields["opacities"]
    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    N = int(means.shape[0])
    device = means.device

    corridor_mask = compute_corridor_mask(means, centers, args.corridor_radius, args.chunk_size)

    near_count = torch.zeros((N,), device=device, dtype=torch.int16)
    footprint_count = torch.zeros((N,), device=device, dtype=torch.int16)
    extent_count = torch.zeros((N,), device=device, dtype=torch.int16)
    fs_count = torch.zeros((N,), device=device, dtype=torch.int16)
    fs_valid_count = torch.zeros((N,), device=device, dtype=torch.int16)

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

            # Original P3 extent test: center-projection based.
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

            # P5++ v4: surface-support-aware footprint free-space pruning.
            #
            # Main idea:
            #   A Gaussian at an occlusion boundary can be free-space violating for
            #   background samples while still supporting the true foreground surface.
            #   Such mixed boundary Gaussians should NOT be hard-deleted.
            #
            # Delete only if:
            #   1) footprint-level free-space violation is strong, and
            #   2) current observed-surface support is weak.
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
                    valid_enough = valid_n >= int(args.fs_min_valid_samples)

                    if torch.any(valid_enough):
                        z_c = z[cand_idx]
                        s_c = scale_metric[start:end][cand_idx]

                        # Free-space violation vote.
                        # Positive gap means the Gaussian front lies in front of current observed depth.
                        z_front = z_c - float(args.fs_extent_factor) * s_c
                        margin = float(args.fs_margin_abs) + float(args.fs_margin_rel) * depth_samples
                        depth_gap = (depth_samples - margin) - z_front[:, None]
                        violation = depth_valid & (depth_gap > 0)

                        vio_n = violation.sum(dim=1)
                        vio_ratio = vio_n.float() / torch.clamp(valid_n.float(), min=1.0)
                        positive_gap_sum = torch.where(
                            violation,
                            depth_gap.clamp_min(0.0),
                            torch.zeros_like(depth_gap),
                        ).sum(dim=1)
                        gap_mean = positive_gap_sum / torch.clamp(vio_n.float(), min=1.0)

                        base_hard = (
                            valid_enough
                            & (vio_n >= int(args.fs_min_violation_samples))
                            & (vio_ratio >= float(args.fs_min_violation_ratio))
                            & (gap_mean >= float(args.fs_min_gap))
                        )

                        # Surface-support vote.
                        # A support vote means the Gaussian center/extent is compatible with
                        # the current observed surface at a sampled footprint location.
                        #
                        # This is the key difference from v1/v3:
                        #   mixed boundary Gaussian = violation votes + support votes -> keep.
                        if bool(args.fs_use_surface_support):
                            surface_margin = (
                                float(args.fs_surface_margin_abs)
                                + float(args.fs_surface_margin_rel) * depth_samples
                                + float(args.fs_surface_extent_factor) * s_c[:, None]
                            )
                            support = depth_valid & (torch.abs(z_c[:, None] - depth_samples) <= surface_margin)
                            support_n = support.sum(dim=1)
                            support_ratio = support_n.float() / torch.clamp(valid_n.float(), min=1.0)
                            weak_support = (
                                (support_n <= int(args.fs_max_surface_support_samples))
                                & (support_ratio <= float(args.fs_max_surface_support_ratio))
                            )
                        else:
                            support_n = torch.zeros_like(vio_n)
                            support_ratio = torch.zeros_like(vio_ratio)
                            weak_support = torch.ones_like(base_hard, dtype=torch.bool)

                        fs = base_hard & weak_support

                        if torch.any(fs):
                            fs_count[start + cand_idx[fs]] += 1

                        fs_valid = valid_enough
                        if torch.any(fs_valid):
                            fs_valid_count[start + cand_idx[fs_valid]] += 1

                        # Extra diagnostics.
                        if not hasattr(args, "_fs_diag"):
                            args._fs_diag = {
                                "base_hard_sum": 0,
                                "surface_supported_sum": 0,
                                "weak_support_sum": 0,
                                "final_delete_sum": 0,
                                "mean_support_ratio_sum": 0.0,
                                "mean_violation_ratio_sum": 0.0,
                                "diag_batches": 0,
                            }
                        supported = support_n > 0
                        args._fs_diag["base_hard_sum"] += int(base_hard.sum().item())
                        args._fs_diag["surface_supported_sum"] += int(supported.sum().item())
                        args._fs_diag["weak_support_sum"] += int(weak_support.sum().item())
                        args._fs_diag["final_delete_sum"] += int(fs.sum().item())
                        args._fs_diag["mean_support_ratio_sum"] += float(support_ratio.mean().item())
                        args._fs_diag["mean_violation_ratio_sum"] += float(vio_ratio.mean().item())
                        args._fs_diag["diag_batches"] += 1

    near_mask = near_count >= int(args.near_min_count)
    footprint_mask = footprint_count >= int(args.footprint_min_count)
    extent_mask = extent_count >= int(args.extent_min_count)
    fs_mask = fs_count >= int(args.fs_min_count)

    # Per-component modes: combine only enabled masks.
    danger_mask = torch.zeros((N,), device=device, dtype=torch.bool)
    if args.corridor_mode != "none":
        danger_mask |= corridor_mask
    if args.near_mode != "none":
        danger_mask |= near_mask
    if args.footprint_mode != "none":
        danger_mask |= footprint_mask
    if args.extent_mode != "none":
        danger_mask |= extent_mask
    if getattr(args, "footprint_free_space_mode", "none") != "none":
        danger_mask |= fs_mask

    stats = {
        "num_corridor": int(corridor_mask.sum().item()),
        "num_near": int(near_mask.sum().item()),
        "num_footprint": int(footprint_mask.sum().item()),
        "num_extent": int(extent_mask.sum().item()),
        "num_footprint_free_space": int(fs_mask.sum().item()),
        "num_p3_danger_union": int(danger_mask.sum().item()),
        "near_count_mean": float(near_count.float().mean().item()),
        "footprint_count_mean": float(footprint_count.float().mean().item()),
        "extent_count_mean": float(extent_count.float().mean().item()),
        "footprint_free_space_count_mean": float(fs_count.float().mean().item()),
        "footprint_free_space_valid_count_mean": float(fs_valid_count.float().mean().item()),
    }
    if hasattr(args, "_fs_diag"):
        batches = max(1, int(args._fs_diag.get("diag_batches", 1)))
        stats.update({
            "footprint_free_space_base_hard_candidates": int(args._fs_diag.get("base_hard_sum", 0)),
            "footprint_free_space_surface_supported_candidates": int(args._fs_diag.get("surface_supported_sum", 0)),
            "footprint_free_space_weak_support_candidates": int(args._fs_diag.get("weak_support_sum", 0)),
            "footprint_free_space_final_delete_candidates": int(args._fs_diag.get("final_delete_sum", 0)),
            "footprint_free_space_mean_support_ratio": float(args._fs_diag.get("mean_support_ratio_sum", 0.0)) / batches,
            "footprint_free_space_mean_violation_ratio": float(args._fs_diag.get("mean_violation_ratio_sum", 0.0)) / batches,
        })
        delattr(args, "_fs_diag")
    return danger_mask, stats


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
    danger_mask, stats = compute_p3_masks(fields, views, centers, args)

    if args.apply_once and p3_seen_uids:
        seen_tensor = torch.tensor(list(p3_seen_uids), device=uids.device, dtype=torch.long)
        danger_mask = danger_mask & (~torch.isin(uids, seen_tensor))

    keep_mask = torch.ones_like(danger_mask)
    if args.p3_mode == "hard":
        keep_mask[danger_mask] = False
    elif args.p3_mode == "downweight":
        fields["opacities"][danger_mask] = fields["opacities"][danger_mask] * float(args.p3_opacity_downweight)
    elif args.p3_mode == "shrink":
        shrink_danger_gaussians(fields, danger_mask, args.p3_shrink_factor)
    elif args.p3_mode == "shrink_downweight":
        fields["opacities"][danger_mask] = fields["opacities"][danger_mask] * float(args.p3_opacity_downweight)
        shrink_danger_gaussians(fields, danger_mask, args.p3_shrink_factor)
    else:
        raise ValueError(f"Unsupported p3_mode: {args.p3_mode}")

    modified_count = int(danger_mask.sum().item())
    removed_count = int((~keep_mask).sum().item())

    if args.apply_once:
        if p3_seen_uids is None:
            p3_seen_uids = set()
        if modified_count > 0:
            p3_seen_uids.update(uids[danger_mask].detach().cpu().tolist())

    if torch.any(~keep_mask):
        fields, source_ids, uids = apply_keep_mask(fields, source_ids, uids, keep_mask, available_keys_from_fields(fields))

    stats.update({
        "p3_enabled": True,
        "p3_mode": args.p3_mode,
        "p3_views_used": len(views),
        "num_p3_modified_or_downweighted": modified_count,
        "num_p3_hard_removed": removed_count,
        "p3_opacity_downweight": args.p3_opacity_downweight,
        "p3_shrink_factor": args.p3_shrink_factor,
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


def save_intermediate_output_packet(
    packets: Sequence[PacketRef],
    selected_indices_prefix: Sequence[int],
    gaussian_keys: Sequence[str],
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    output_pt: Path,
    meta: Dict[str, Any],
) -> None:
    """Save one intermediate maintained map using the same packet schema as final output.

    The saved file can be rendered like any fused packet.  It intentionally uses
    the same Gaussian fields as save_output_packet(), so downstream renderers do
    not need to understand P8 internals.
    """
    save_output_packet(
        packets=packets,
        selected_indices=selected_indices_prefix,
        gaussian_keys=gaussian_keys,
        fields=fields,
        source_ids=source_ids,
        output_pt=output_pt,
        meta=meta,
    )


def should_save_intermediate_step(step_i: int, num_steps: int, args: argparse.Namespace) -> bool:
    if not getattr(args, "save_intermediate_maps", False):
        return False
    start = max(0, int(getattr(args, "intermediate_start", 0)))
    every = max(1, int(getattr(args, "intermediate_every", 1)))
    if step_i < start:
        # Always keep the final map if explicitly requested by the schedule?
        return False
    return ((step_i - start) % every == 0) or (step_i == num_steps - 1)


def maybe_save_intermediate_map(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    step_i: int,
    packet_idx: int,
    row_step: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Optionally save the maintained map after one incremental step.

    This is used for custom-pose prefix visualization after P8 maintenance:
      step_0001 = map after packet 0 has been inserted/maintained
      step_0002 = map after packets 0..1 have been inserted/maintained
      ...
    """
    if not should_save_intermediate_step(step_i, len(selected_indices), args):
        return None

    out_dir = Path(args.intermediate_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_gauss = int(fields["means"].shape[0])
    output_pt = out_dir / f"step_{step_i + 1:04d}_packet{int(packet_idx):04d}_gauss{n_gauss:09d}.pt"
    selected_prefix = list(selected_indices[: step_i + 1])

    meta = {
        "script": "map_maintenance_p8_coverage_insert.py",
        "is_intermediate_map": True,
        "intermediate_step": int(step_i),
        "intermediate_step_1based": int(step_i + 1),
        "inserted_packet_sorted_index": int(packet_idx),
        "selected_indices_prefix": [int(x) for x in selected_prefix],
        "num_prefix_packets": int(step_i + 1),
        "num_gaussians": n_gauss,
        "maintenance_order": args.maintenance_order,
        "execution_mode": args.execution_mode,
        "enable_coverage_insert": bool(args.enable_coverage_insert),
        "coverage_mode": args.coverage_mode,
        "enable_p3": bool(args.enable_p3),
        "p3_mode": args.p3_mode,
        "enable_p4": bool(args.enable_p4),
        "p4_mode": args.p4_mode,
        "row_step": {k: v for k, v in row_step.items() if isinstance(v, (int, float, str, bool)) or v is None},
    }
    save_intermediate_output_packet(
        packets=packets,
        selected_indices_prefix=selected_prefix,
        gaussian_keys=gaussian_keys,
        fields=fields,
        source_ids=source_ids,
        output_pt=output_pt,
        meta=meta,
    )
    return {
        "step": int(step_i),
        "step_1based": int(step_i + 1),
        "inserted_packet_sorted_index": int(packet_idx),
        "num_prefix_packets": int(step_i + 1),
        "num_gaussians": n_gauss,
        "path": str(output_pt),
    }



def compute_accumulation_map(
    fields: Dict[str, torch.Tensor],
    view: TrajectoryView,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Approximate 2D accumulation/coverage map from the existing Gaussian map.

    This is intentionally renderer-independent: it projects Gaussian centers,
    samples a small set of points over each projected footprint, and accumulates
    opacity into the corresponding image pixels.

    Returns:
        coverage: [H, W], clamped to [0, 1].
        old_depth: [H, W], approximate nearest Gaussian depth; inf where unknown.
        meta: scalar diagnostics.
    """
    device = fields["means"].device
    H, W = int(view.H), int(view.W)
    flat_n = H * W
    accum_flat = torch.zeros((flat_n,), device=device, dtype=torch.float32)
    depth_flat = torch.full((flat_n,), float("inf"), device=device, dtype=torch.float32)

    means = fields["means"]
    opacities = fields["opacities"].float()
    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    offsets = build_footprint_sample_offsets(args.coverage_sample_pattern, device)
    f_pix = float((view.K_pixel[0, 0] + view.K_pixel[1, 1]) * 0.5)

    num_projected = 0
    num_samples = 0
    num_valid_samples = 0

    for start in range(0, int(means.shape[0]), int(args.chunk_size)):
        end = min(start + int(args.chunk_size), int(means.shape[0]))
        u, v, z = project_points(
            means[start:end],
            view.Twc_or_Tcw,
            view.K_pixel,
            args.extrinsic_type,
            args.camera_z_sign,
        )
        radius_px = f_pix * scale_metric[start:end] / torch.clamp(z, min=1e-6)
        r_clip = torch.clamp(radius_px, min=0.0, max=float(args.coverage_radius_clip_px))
        opa = opacities[start:end]

        cand = (
            (z > 0)
            & (radius_px >= float(args.coverage_min_radius_px))
            & (opa >= float(args.coverage_min_opacity))
            & ((u + r_clip) >= 0)
            & ((u - r_clip) < W)
            & ((v + r_clip) >= 0)
            & ((v - r_clip) < H)
        )
        cand_idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        if cand_idx.numel() == 0:
            continue

        num_projected += int(cand_idx.numel())
        su = u[cand_idx, None] + offsets[None, :, 0] * r_clip[cand_idx, None]
        sv = v[cand_idx, None] + offsets[None, :, 1] * r_clip[cand_idx, None]
        sui = torch.round(su).long()
        svi = torch.round(sv).long()
        in_img = (sui >= 0) & (sui < W) & (svi >= 0) & (svi < H)
        num_samples += int(sui.numel())
        if not torch.any(in_img):
            continue
        num_valid_samples += int(in_img.sum().item())

        flat_idx = (svi[in_img] * W + sui[in_img]).reshape(-1)
        # Approximate accumulation. We use index_add and clamp, not exact alpha compositing.
        # This is enough for deciding whether a region is already covered.
        sample_weights = (
            opa[cand_idx, None].expand_as(sui).float()[in_img].reshape(-1)
            * float(args.coverage_opacity_scale)
            / max(1, int(offsets.shape[0]))
        )
        accum_flat.index_add_(0, flat_idx, sample_weights)

        # Approximate nearest depth for optional alpha_depth mode.
        if getattr(args, "coverage_mode", "alpha") == "alpha_depth":
            sample_z = z[cand_idx, None].expand_as(sui).float()[in_img].reshape(-1)
            # PyTorch 2.x supports scatter_reduce_. If this fails in a very old
            # environment, use --coverage_mode alpha instead.
            depth_flat.scatter_reduce_(0, flat_idx, sample_z, reduce="amin", include_self=True)

    coverage = torch.clamp(accum_flat.reshape(H, W), min=0.0, max=1.0)
    old_depth = depth_flat.reshape(H, W)
    meta = {
        "coverage_num_projected_old_gaussians": int(num_projected),
        "coverage_num_samples": int(num_samples),
        "coverage_num_valid_samples": int(num_valid_samples),
        "coverage_mean": float(coverage.mean().item()),
        "coverage_max": float(coverage.max().item()),
        "coverage_low_ratio": float((coverage < float(args.alpha_thresh)).float().mean().item()),
    }
    return coverage, old_depth, meta


def compute_insert_mask_from_coverage(
    coverage: torch.Tensor,
    old_depth: torch.Tensor,
    view: TrajectoryView,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Pixel mask where incoming packet Gaussians are allowed to be inserted."""
    low_alpha = coverage < float(args.alpha_thresh)
    insert_mask = low_alpha.clone()

    depth_meta: Dict[str, Any] = {
        "coverage_depth_enabled": False,
        "coverage_missing_foreground_ratio": None,
        "coverage_depth_inconsistent_ratio": None,
    }

    if args.coverage_mode == "alpha_depth" and view.depth is not None:
        D = view.depth.float()
        cur_valid = torch.isfinite(D) & (D > 0)
        old_valid = torch.isfinite(old_depth) & (old_depth > 0)

        margin = float(args.coverage_depth_margin_abs) + float(args.coverage_depth_margin_rel) * D
        # Current observation has a closer surface that old map does not explain.
        missing_foreground = cur_valid & old_valid & (D < old_depth - margin)
        # Optional general depth inconsistency; use carefully because it may over-insert.
        depth_inconsistent = cur_valid & old_valid & (torch.abs(D - old_depth) > margin)

        if args.coverage_use_missing_foreground:
            insert_mask |= missing_foreground
        if args.coverage_use_depth_inconsistent:
            insert_mask |= depth_inconsistent

        depth_meta = {
            "coverage_depth_enabled": True,
            "coverage_missing_foreground_ratio": float(missing_foreground.float().mean().item()),
            "coverage_depth_inconsistent_ratio": float(depth_inconsistent.float().mean().item()),
        }

    meta = {
        "insert_mask_ratio": float(insert_mask.float().mean().item()),
        "insert_mask_low_alpha_ratio": float(low_alpha.float().mean().item()),
        **depth_meta,
    }
    return insert_mask, meta


def select_packet_gaussians_by_insert_mask(
    packet_fields: Dict[str, torch.Tensor],
    insert_mask: torch.Tensor,
    view: TrajectoryView,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return keep mask for incoming packet Gaussians using footprint-over-insert-mask voting."""
    device = packet_fields["means"].device
    H, W = int(view.H), int(view.W)
    N = int(packet_fields["means"].shape[0])
    keep = torch.zeros((N,), device=device, dtype=torch.bool)

    means = packet_fields["means"]
    opacities = packet_fields["opacities"].float()
    scale_metric = compute_scale_metric(packet_fields["scales"], args.scale_input, args.scale_metric)
    offsets = build_footprint_sample_offsets(args.coverage_sample_pattern, device)
    f_pix = float((view.K_pixel[0, 0] + view.K_pixel[1, 1]) * 0.5)

    candidate_count = 0
    valid_footprint_count = 0
    max_ratio_seen = 0.0

    for start in range(0, N, int(args.chunk_size)):
        end = min(start + int(args.chunk_size), N)
        u, v, z = project_points(
            means[start:end],
            view.Twc_or_Tcw,
            view.K_pixel,
            args.extrinsic_type,
            args.camera_z_sign,
        )
        radius_px = f_pix * scale_metric[start:end] / torch.clamp(z, min=1e-6)
        r_clip = torch.clamp(radius_px, min=0.0, max=float(args.coverage_radius_clip_px))
        opa = opacities[start:end]

        cand = (
            (z > 0)
            & (radius_px >= float(args.insert_min_radius_px))
            & (opa >= float(args.insert_min_opacity))
            & ((u + r_clip) >= 0)
            & ((u - r_clip) < W)
            & ((v + r_clip) >= 0)
            & ((v - r_clip) < H)
        )
        cand_idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        if cand_idx.numel() == 0:
            continue
        candidate_count += int(cand_idx.numel())

        su = u[cand_idx, None] + offsets[None, :, 0] * r_clip[cand_idx, None]
        sv = v[cand_idx, None] + offsets[None, :, 1] * r_clip[cand_idx, None]
        sui = torch.round(su).long()
        svi = torch.round(sv).long()
        in_img = (sui >= 0) & (sui < W) & (svi >= 0) & (svi < H)
        valid_n = in_img.sum(dim=1)
        valid_enough = valid_n >= int(args.insert_min_valid_samples)
        if not torch.any(valid_enough):
            continue
        valid_footprint_count += int(valid_enough.sum().item())

        hit = torch.zeros_like(in_img, dtype=torch.bool)
        if torch.any(in_img):
            hit[in_img] = insert_mask[svi[in_img], sui[in_img]]
        hit_n = hit.sum(dim=1)
        hit_ratio = hit_n.float() / torch.clamp(valid_n.float(), min=1.0)
        if hit_ratio.numel() > 0:
            max_ratio_seen = max(max_ratio_seen, float(hit_ratio.max().item()))

        local_keep = (
            valid_enough
            & (hit_n >= int(args.insert_min_hit_samples))
            & (hit_ratio >= float(args.insert_ratio_thresh))
        )
        if torch.any(local_keep):
            keep[start + cand_idx[local_keep]] = True

    meta = {
        "coverage_candidate_new_gaussians": int(candidate_count),
        "coverage_valid_footprint_new_gaussians": int(valid_footprint_count),
        "coverage_selected_new_gaussians": int(keep.sum().item()),
        "coverage_selected_ratio": float(keep.float().mean().item()),
        "coverage_max_insert_hit_ratio": float(max_ratio_seen),
    }
    return keep, meta


def apply_coverage_insertion(
    old_fields: Optional[Dict[str, torch.Tensor]],
    packet_fields: Dict[str, torch.Tensor],
    packet_source: torch.Tensor,
    packet_uid: torch.Tensor,
    view: Optional[TrajectoryView],
    gaussian_keys: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Filter incoming packet before insertion based on old-map coverage."""
    n_in = int(packet_fields["means"].shape[0])
    if not getattr(args, "enable_coverage_insert", False):
        return packet_fields, packet_source, packet_uid, {
            "coverage_insert_enabled": False,
            "coverage_input_new_gaussians": n_in,
            "coverage_selected_new_gaussians": n_in,
            "coverage_selected_ratio": 1.0,
        }

    if old_fields is None or view is None:
        return packet_fields, packet_source, packet_uid, {
            "coverage_insert_enabled": True,
            "coverage_reason": "first_packet_or_no_view_insert_all",
            "coverage_input_new_gaussians": n_in,
            "coverage_selected_new_gaussians": n_in,
            "coverage_selected_ratio": 1.0,
        }

    coverage, old_depth, cov_meta = compute_accumulation_map(old_fields, view, args)
    insert_mask, mask_meta = compute_insert_mask_from_coverage(coverage, old_depth, view, args)
    keep, sel_meta = select_packet_gaussians_by_insert_mask(packet_fields, insert_mask, view, args)

    if bool(args.insert_always_keep_at_least_one) and int(keep.sum().item()) == 0 and n_in > 0:
        # Keep the most opaque Gaussian as a safety fallback. Usually disabled.
        idx = torch.argmax(packet_fields["opacities"].float())
        keep[idx] = True

    filtered_fields = {k: packet_fields[k][keep].contiguous() for k in gaussian_keys}
    filtered_source = packet_source[keep].contiguous()
    filtered_uid = packet_uid[keep].contiguous()

    meta = {
        "coverage_insert_enabled": True,
        "coverage_mode": args.coverage_mode,
        "alpha_thresh": float(args.alpha_thresh),
        "insert_ratio_thresh": float(args.insert_ratio_thresh),
        "coverage_input_new_gaussians": n_in,
        **cov_meta,
        **mask_meta,
        **sel_meta,
    }
    return filtered_fields, filtered_source, filtered_uid, meta

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
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    map_fields: Optional[Dict[str, torch.Tensor]] = None
    map_source: Optional[torch.Tensor] = None
    map_uid: Optional[torch.Tensor] = None
    per_packet_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    intermediate_rows: List[Dict[str, Any]] = []
    p4_seen: set[int] = set()
    p3_seen: set[int] = set()
    uid_cursor = 0

    view_by_idx = {int(v.sorted_index): v for v in all_views}

    for step_i, idx in enumerate(selected_indices):
        f, src, uid, row = load_packet_fields(packets[idx], gaussian_keys, device, uid_cursor)
        uid_cursor += int(uid.numel())
        row["num_input_before_coverage_insert"] = int(f["means"].shape[0])

        # For the first packet, there is no previous map to clean.
        n_before_pre_p3 = int(map_fields["means"].shape[0]) if map_fields is not None else 0
        pre_p3_meta: Dict[str, Any] = {"p3_enabled": False}
        post_p3_meta: Dict[str, Any] = {"p3_enabled": False}
        p4_meta: Dict[str, Any] = {"p4_enabled": False}
        coverage_meta: Dict[str, Any] = {"coverage_insert_enabled": False}
        n_after_coverage_select = int(f["means"].shape[0])

        current_view = view_by_idx.get(int(idx), None)

        if args.maintenance_order in {"pre_p3_insert_p4", "pre_p3_insert_p4_post_p3"}:
            # Step A: clean the existing map BEFORE inserting the incoming packet.
            # This directly targets old->new contamination.
            if map_fields is not None and map_source is not None and map_uid is not None:
                pre_views = select_views_for_step(all_views, idx, args.pre_p3_trajectory_policy)
                map_fields, map_source, map_uid, pre_p3_meta, p3_seen = apply_p3(
                    map_fields, map_source, map_uid, pre_views, args, p3_seen
                )

            # Step B: coverage-aware selective insertion.
            # Render/estimate coverage of cleaned old map at the current pose,
            # then insert only incoming Gaussians whose footprint falls in low-coverage regions.
            f, src, uid, coverage_meta = apply_coverage_insertion(
                map_fields, f, src, uid, current_view, gaussian_keys, args
            )
            n_after_coverage_select = int(f["means"].shape[0])
            row.update({k: v for k, v in coverage_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})

            # Step C: insert selected incoming Gaussians after cleaning old occluders.
            map_fields, map_source, map_uid = append_map(
                map_fields, map_source, map_uid, f, src, uid, gaussian_keys
            )
            assert map_fields is not None and map_source is not None and map_uid is not None
            n_after_insert = int(map_fields["means"].shape[0])

            # Step D: apply P4 redundancy control after insertion.
            map_fields, map_source, map_uid, p4_meta, p4_seen = apply_p4(
                map_fields, map_source, map_uid, args, p4_seen
            )

            # Optional Step E: post-insert retention check.
            if args.maintenance_order == "pre_p3_insert_p4_post_p3" and args.post_p3_trajectory_policy != "none":
                post_views = select_views_for_step(all_views, idx, args.post_p3_trajectory_policy)
                map_fields, map_source, map_uid, post_p3_meta, p3_seen = apply_p3(
                    map_fields, map_source, map_uid, post_views, args, p3_seen
                )

        else:
            # Legacy incremental semantics with coverage insertion before append.
            f, src, uid, coverage_meta = apply_coverage_insertion(
                map_fields, f, src, uid, current_view, gaussian_keys, args
            )
            n_after_coverage_select = int(f["means"].shape[0])
            row.update({k: v for k, v in coverage_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})

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

        row["num_after_coverage_insert"] = int(n_after_coverage_select)
        per_packet_rows.append(row)

        n_after = int(map_fields["means"].shape[0])
        row_step = {
            "step": int(step_i),
            "inserted_packet_sorted_index": int(idx),
            "maintenance_order": args.maintenance_order,
            "num_gaussians_before_pre_p3": int(n_before_pre_p3),
            "num_new_gaussians_before_coverage": int(row["num_input_before_coverage_insert"]),
            "num_new_gaussians_after_coverage": int(n_after_coverage_select),
            "num_gaussians_after_insert": int(n_after_insert),
            "num_gaussians_after_maintenance": int(n_after),
        }
        row_step.update({f"pre_p3_{k}": v for k, v in pre_p3_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        row_step.update({f"coverage_{k}": v for k, v in coverage_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        row_step.update({f"p4_{k}": v for k, v in p4_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        row_step.update({f"post_p3_{k}": v for k, v in post_p3_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        step_rows.append(row_step)

        inter_row = maybe_save_intermediate_map(
            packets=packets,
            selected_indices=selected_indices,
            gaussian_keys=gaussian_keys,
            fields=map_fields,
            source_ids=map_source,
            step_i=step_i,
            packet_idx=idx,
            row_step=row_step,
            args=args,
        )
        if inter_row is not None:
            intermediate_rows.append(inter_row)
            print(f"      saved intermediate map: {inter_row['path']}")

        print(
            f"[Incremental {step_i+1:03d}/{len(selected_indices):03d}] "
            f"packet={idx:04d}, before_pre_p3={n_before_pre_p3:,}, "
            f"new={int(row['num_input_before_coverage_insert']):,}->{n_after_coverage_select:,}, "
            f"after_insert={n_after_insert:,}, after_maintenance={n_after:,}"
        )
        if args.empty_cache_each_step and device.type == "cuda":
            torch.cuda.empty_cache()

    assert map_fields is not None and map_source is not None
    per_packet_rows = update_per_packet_rows(per_packet_rows, map_source, int(map_fields["means"].shape[0]), map_fields)
    return map_fields, map_source, per_packet_rows, step_rows, intermediate_rows


def apply_fs_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply compact presets for surface-support-aware free-space pruning.

    Core deletion rule:
      hard delete only if
        free-space violation is strong
        AND current observed-surface support is weak.

    This avoids deleting mixed boundary Gaussians that support a true foreground
    surface while also leaking into neighboring background free space.
    """
    preset = getattr(args, "fs_preset", "balanced")

    if preset == "v1_like":
        # Diagnostic baseline close to earlier aggressive P5++ v1:
        # no surface support protection.
        args.fs_sample_pattern = "star9"
        args.fs_min_valid_samples = 1
        args.fs_min_violation_samples = 1
        args.fs_min_violation_ratio = 0.10
        args.fs_min_gap = 0.0
        args.fs_use_surface_support = False
        args.fs_max_surface_support_samples = 0
        args.fs_max_surface_support_ratio = 1.0

    elif preset == "aggressive":
        # More deletion; still protects clearly supported foreground Gaussians.
        args.fs_sample_pattern = "star9"
        args.fs_min_valid_samples = 2
        args.fs_min_violation_samples = 2
        args.fs_min_violation_ratio = 0.18
        args.fs_min_gap = 0.05
        args.fs_use_surface_support = True
        args.fs_surface_margin_abs = 0.20
        args.fs_surface_margin_rel = 0.03
        args.fs_surface_extent_factor = 1.0
        args.fs_max_surface_support_samples = 1
        args.fs_max_surface_support_ratio = 0.20

    elif preset == "balanced":
        # Main setting: delete high-confidence free-space Gaussians, keep mixed boundary ones.
        args.fs_sample_pattern = "star9"
        args.fs_min_valid_samples = 2
        args.fs_min_violation_samples = 2
        args.fs_min_violation_ratio = 0.25
        args.fs_min_gap = 0.08
        args.fs_use_surface_support = True
        args.fs_surface_margin_abs = 0.25
        args.fs_surface_margin_rel = 0.03
        args.fs_surface_extent_factor = 1.0
        args.fs_max_surface_support_samples = 1
        args.fs_max_surface_support_ratio = 0.15

    elif preset == "conservative":
        # Stronger protection of true foreground / boundary Gaussians.
        args.fs_sample_pattern = "star9"
        args.fs_min_valid_samples = 3
        args.fs_min_violation_samples = 3
        args.fs_min_violation_ratio = 0.35
        args.fs_min_gap = 0.12
        args.fs_use_surface_support = True
        args.fs_surface_margin_abs = 0.30
        args.fs_surface_margin_rel = 0.04
        args.fs_surface_extent_factor = 1.2
        args.fs_max_surface_support_samples = 0
        args.fs_max_surface_support_ratio = 0.05

    else:
        raise ValueError(f"Unknown --fs_preset: {preset}")

    return args


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P8 coverage-aware selective insertion on top of P5++")
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
    p.add_argument("--p3_mode", choices=["none", "hard", "downweight", "shrink", "shrink_downweight"], default="none")
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

    # P5++ v4 surface-support-aware footprint free-space pruning.
    # Usually only --fs_preset and --fs_margin_abs need to be changed.
    p.add_argument("--footprint_free_space_mode", choices=["none", "active"], default="none")
    p.add_argument("--fs_preset", choices=["v1_like", "aggressive", "balanced", "conservative"], default="balanced")
    p.add_argument("--fs_sample_pattern", choices=["center", "cross5", "star9", "ring13"], default="star9")
    p.add_argument("--fs_min_radius_px", type=float, default=2.0)
    p.add_argument("--fs_radius_clip_px", type=float, default=120.0)
    p.add_argument("--fs_min_opacity", type=float, default=0.01)
    p.add_argument("--fs_extent_factor", type=float, default=3.0)
    p.add_argument("--fs_margin_abs", type=float, default=0.20)
    p.add_argument("--fs_margin_rel", type=float, default=0.02)
    p.add_argument("--fs_min_valid_samples", type=int, default=2)
    p.add_argument("--fs_min_violation_samples", type=int, default=2)
    p.add_argument("--fs_min_violation_ratio", type=float, default=0.25)
    p.add_argument("--fs_min_gap", type=float, default=0.08)
    p.add_argument("--fs_min_count", type=int, default=1)

    # Surface-support protection. These are normally set by --fs_preset.
    p.add_argument("--fs_use_surface_support", action="store_true", default=True)
    p.add_argument("--no_fs_use_surface_support", action="store_false", dest="fs_use_surface_support")
    p.add_argument("--fs_surface_margin_abs", type=float, default=0.25)
    p.add_argument("--fs_surface_margin_rel", type=float, default=0.03)
    p.add_argument("--fs_surface_extent_factor", type=float, default=1.0)
    p.add_argument("--fs_max_surface_support_samples", type=int, default=1)
    p.add_argument("--fs_max_surface_support_ratio", type=float, default=0.15)
    # P8: coverage-aware selective insertion.
    p.add_argument("--enable_coverage_insert", action="store_true")
    p.add_argument(
        "--coverage_mode",
        choices=["alpha", "alpha_depth"],
        default="alpha",
        help="alpha uses approximate accumulation only; alpha_depth also uses approximate nearest old-map depth.",
    )
    p.add_argument("--alpha_thresh", type=float, default=0.35)
    p.add_argument("--coverage_sample_pattern", choices=["center", "cross5", "star9", "ring13"], default="star9")
    p.add_argument("--coverage_min_radius_px", type=float, default=1.0)
    p.add_argument("--coverage_radius_clip_px", type=float, default=80.0)
    p.add_argument("--coverage_min_opacity", type=float, default=0.005)
    p.add_argument("--coverage_opacity_scale", type=float, default=1.0)
    p.add_argument("--coverage_depth_margin_abs", type=float, default=0.30)
    p.add_argument("--coverage_depth_margin_rel", type=float, default=0.03)
    p.add_argument("--coverage_use_missing_foreground", action="store_true", default=True)
    p.add_argument("--no_coverage_use_missing_foreground", action="store_false", dest="coverage_use_missing_foreground")
    p.add_argument("--coverage_use_depth_inconsistent", action="store_true", default=False)

    p.add_argument("--insert_min_radius_px", type=float, default=1.0)
    p.add_argument("--insert_min_opacity", type=float, default=0.005)
    p.add_argument("--insert_min_valid_samples", type=int, default=1)
    p.add_argument("--insert_min_hit_samples", type=int, default=1)
    p.add_argument("--insert_ratio_thresh", type=float, default=0.10)
    p.add_argument("--insert_always_keep_at_least_one", action="store_true", default=False)

    # Intermediate maintained-map saving for visualization/debugging.
    # Only meaningful in --execution_mode incremental.
    p.add_argument("--save_intermediate_maps", action="store_true", help="Save the maintained map after incremental steps for later rendering.")
    p.add_argument("--intermediate_out_dir", type=Path, default=None, help="Output directory for intermediate .pt maps. Default: <output_dir>/p8_intermediate_maps")
    p.add_argument("--intermediate_every", type=int, default=1, help="Save every N incremental steps. The final step is always saved once this feature is enabled.")
    p.add_argument("--intermediate_start", type=int, default=0, help="First zero-based incremental step to save.")

    p.add_argument("--chunk_size", type=int, default=250_000)
    p.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    p.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive")
    p.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args = apply_fs_preset(args)
    t_total = now()
    device = torch.device(args.device)
    out_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_intermediate_maps:
        if args.execution_mode != "incremental":
            raise ValueError("--save_intermediate_maps is only supported with --execution_mode incremental")
        if args.intermediate_out_dir is None:
            args.intermediate_out_dir = out_dir / "p8_intermediate_maps"
        args.intermediate_out_dir.mkdir(parents=True, exist_ok=True)

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
    intermediate_rows: List[Dict[str, Any]] = []
    if args.execution_mode == "batch":
        fields, source_ids, per_packet_rows, step_rows = run_batch(
            packets, selected_indices, gaussian_keys, views, args, device
        )
        strict_online = False
    else:
        fields, source_ids, per_packet_rows, step_rows, intermediate_rows = run_incremental(
            packets, selected_indices, gaussian_keys, views, args, device
        )
        strict_online = args.p3_trajectory_policy in {"causal", "current"}
    maintenance_time = now() - t2

    print("[4/5] Saving output packet")
    t3 = now()
    total_in = int(sum(r["num_input"] for r in per_packet_rows))
    total_out = int(fields["means"].shape[0])
    meta = {
        "script": "map_maintenance_p8_coverage_insert.py",
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "trajectory_ranges": args.trajectory_ranges if args.trajectory_ranges else args.packet_ranges,
        "execution_mode": args.execution_mode,
        "maintenance_order": args.maintenance_order,
        "p3_trajectory_policy": args.p3_trajectory_policy,
        "footprint_free_space_mode": args.footprint_free_space_mode,
        "fs_preset": args.fs_preset,
        "fs_sample_pattern": args.fs_sample_pattern,
        "fs_min_radius_px": args.fs_min_radius_px,
        "fs_radius_clip_px": args.fs_radius_clip_px,
        "fs_margin_abs": args.fs_margin_abs,
        "fs_margin_rel": args.fs_margin_rel,
        "fs_min_violation_ratio": args.fs_min_violation_ratio,
        "fs_min_violation_samples": args.fs_min_violation_samples,
        "fs_min_gap": args.fs_min_gap,
        "fs_use_surface_support": args.fs_use_surface_support,
        "fs_surface_margin_abs": args.fs_surface_margin_abs,
        "fs_surface_margin_rel": args.fs_surface_margin_rel,
        "fs_surface_extent_factor": args.fs_surface_extent_factor,
        "fs_max_surface_support_samples": args.fs_max_surface_support_samples,
        "fs_max_surface_support_ratio": args.fs_max_surface_support_ratio,
        "enable_coverage_insert": bool(args.enable_coverage_insert),
        "coverage_mode": args.coverage_mode,
        "alpha_thresh": args.alpha_thresh,
        "insert_ratio_thresh": args.insert_ratio_thresh,
        "coverage_sample_pattern": args.coverage_sample_pattern,
        "coverage_opacity_scale": args.coverage_opacity_scale,
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
        "save_intermediate_maps": bool(args.save_intermediate_maps),
        "intermediate_out_dir": str(args.intermediate_out_dir) if args.intermediate_out_dir is not None else None,
        "num_intermediate_maps_saved": int(len(intermediate_rows)),
        "load_packet_time_sec": load_packet_time,
        "load_view_time_sec": load_view_time,
        "maintenance_time_sec": maintenance_time,
    }
    save_output_packet(packets, selected_indices, gaussian_keys, fields, source_ids, args.output_pt, meta)
    save_time = now() - t3

    print("[5/5] Saving diagnostics")
    meta["save_time_sec"] = save_time
    meta["total_time_sec"] = now() - t_total
    summary_path = out_dir / "p8_coverage_insert_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    write_csv(out_dir / "p8_coverage_insert_per_packet.csv", per_packet_rows)
    write_csv(out_dir / "p8_coverage_insert_steps.csv", step_rows)
    if args.save_intermediate_maps:
        manifest = {
            "script": "map_maintenance_p8_coverage_insert.py",
            "packet_dir": str(args.packet_dir),
            "packet_ranges": args.packet_ranges,
            "trajectory_ranges": args.trajectory_ranges if args.trajectory_ranges else args.packet_ranges,
            "output_pt": str(args.output_pt),
            "intermediate_out_dir": str(args.intermediate_out_dir),
            "num_intermediate_maps_saved": len(intermediate_rows),
            "intermediate_maps": intermediate_rows,
        }
        manifest_path = Path(args.intermediate_out_dir) / "intermediate_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"      saved packet: {args.output_pt}")
    print(f"      saved summary: {summary_path}")
    print(f"      saved per-packet csv: {out_dir / 'p8_coverage_insert_per_packet.csv'}")
    print(f"      saved steps csv: {out_dir / 'p8_coverage_insert_steps.csv'}")
    if args.save_intermediate_maps:
        print(f"      saved intermediate maps: {args.intermediate_out_dir} ({len(intermediate_rows)} files)")
    print(f"      total: in={total_in:,}, out={total_out:,}, ratio={total_out / max(total_in, 1):.4f}")
    print(f"      total_time_sec={meta['total_time_sec']:.3f}")


if __name__ == "__main__":
    main()
