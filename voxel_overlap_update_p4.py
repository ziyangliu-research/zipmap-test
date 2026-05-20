#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4: voxel-overlap-aware Gaussian update / coarse-to-fine replacement.

Purpose
-------
This script is designed for the current ZipMap -> ReSplat packet workflow.

It consumes a directory of ReSplat Gaussian packet .pt files and writes a single
fused/maintained packet .pt that can be rendered by:

    render_fused_packet_trajectory.py

The P4 hypothesis is intentionally narrower than P2/P3:

    Do not use depth.
    Do not use trajectory/free-space constraints.
    Only use voxel overlap, scale, opacity, and packet order.

The main test is whether old large/coarse Gaussians in an overlapped voxel should
be downweighted or removed when later packets contain smaller/finer Gaussians in
the same local region.

Supported modes
---------------
1) stats_only
   Concatenate packets and only write overlap diagnostics.

2) coarse_to_fine
   For each voxel, compute min_scale and max_source_packet_id.
   A Gaussian is considered an old-coarse candidate if:

       source_id < voxel_max_source_id
       scale_metric > scale_ratio * voxel_min_scale
       voxel_count >= min_voxel_count

   Then apply --old_mode hard/downweight.

3) opacity_budget
   Limit opacity accumulation inside each voxel.
   Either uniformly downweight all Gaussians in over-budget voxels, or keep
   high-score Gaussians until a cumulative opacity budget is reached.

4) voxel_topk
   Keep only top-k Gaussians per voxel according to a configurable score.

Typical usage
-------------
Stats only:

python voxel_overlap_update_p4.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p4_stats_naive.pt \
  --mode stats_only \
  --voxel_size 0.05 \
  --device cuda:0

Coarse-to-fine downweight:

python voxel_overlap_update_p4.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p4_coarse_to_fine_v005.pt \
  --mode coarse_to_fine \
  --voxel_size 0.05 \
  --scale_input auto \
  --scale_metric max \
  --scale_ratio 2.0 \
  --old_mode downweight \
  --old_downweight 0.1 \
  --opacity_cap 0.30 \
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
        raise ValueError(f"No packets selected by range spec: {spec}")
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
    keys = [k for k in ALL_GAUSSIAN_KEYS if all(k in packets[i].data and packets[i].data[k] is not None for i in selected_indices)]
    missing = [k for k in PRIMARY_GAUSSIAN_KEYS if k not in keys]
    if missing:
        raise KeyError(f"Missing primary Gaussian keys in selected packets: {missing}")
    return keys


def concat_gaussian_fields(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]]]:
    chunks: Dict[str, List[torch.Tensor]] = {k: [] for k in gaussian_keys}
    source_ids: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []

    for idx in selected_indices:
        p = packets[idx].data
        n_this: Optional[int] = None
        for k in gaussian_keys:
            t = normalize_gaussian_tensor(p[k], k, device)
            if n_this is None:
                n_this = int(t.shape[0])
            elif int(t.shape[0]) != n_this:
                raise RuntimeError(f"Packet {packets[idx].path.name}: {k} length {t.shape[0]} != {n_this}")
            chunks[k].append(t)
        assert n_this is not None
        source_ids.append(torch.full((n_this,), int(idx), device=device, dtype=torch.long))
        rows.append({
            "source_packet_sorted_index": int(idx),
            "source_packet_name": packets[idx].path.name,
            "num_input": int(n_this),
        })

    fields = {k: torch.cat(v, dim=0).contiguous() for k, v in chunks.items()}
    src = torch.cat(source_ids, dim=0).contiguous()
    return fields, src, rows


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
    # Indoor-scale scenes should be safe here. This guard catches accidental mm/unit mistakes.
    prod_dims = int(dims[0].item()) * int(dims[1].item()) * int(dims[2].item())
    if prod_dims >= 9_000_000_000_000_000_000:
        raise RuntimeError(
            f"Voxel linearization would overflow int64. coord_min={coord_min.tolist()}, "
            f"coord_max={coord_max.tolist()}, dims={dims.tolist()}. Increase --voxel_size."
        )
    shifted = coords - coord_min
    keys = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]
    return coords, keys.contiguous(), coord_min, dims


def compute_group_index(keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
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

    scale_sum = torch.zeros((num_groups,), device=device, dtype=torch.float32)
    scale_sum.scatter_add_(0, group_id, scale_metric.float())

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
        "scale_sum": scale_sum,
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
    """Return indices sorted by key asc and score desc inside each key, plus local ranks."""
    # Stable two-pass sort: first by secondary key (-score), then stable sort by primary key.
    order_score = torch.argsort(-score, stable=True)
    keys_after_score = keys[order_score]
    order_key = torch.argsort(keys_after_score, stable=True)
    order = order_score[order_key]

    keys_sorted = keys[order]
    _, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    local_rank = torch.arange(keys.numel(), device=keys.device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
    return order, counts, starts, local_rank


def apply_voxel_topk(
    keys: torch.Tensor,
    opacities: torch.Tensor,
    scale_metric: torch.Tensor,
    source_ids: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    score = make_score(args.voxel_topk_score, opacities, scale_metric, source_ids, args.newer_bonus)
    order, counts, starts, local_rank = grouped_rank_by_score(keys, score)
    keep_sorted = local_rank < int(args.max_gaussians_per_voxel)
    keep = torch.zeros((keys.numel(),), device=keys.device, dtype=torch.bool)
    keep[order] = keep_sorted
    return keep


def apply_opacity_budget(
    keys: torch.Tensor,
    group_id: torch.Tensor,
    group_stats: Dict[str, torch.Tensor],
    opacities: torch.Tensor,
    scale_metric: torch.Tensor,
    source_ids: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = torch.ones((keys.numel(),), device=keys.device, dtype=torch.bool)
    opacity_factor = torch.ones_like(opacities.float())
    budget = float(args.opacity_budget)
    if budget <= 0:
        return keep, opacity_factor

    opacity_sum = group_stats["opacity_sum"]
    over = opacity_sum > budget
    if args.opacity_budget_mode == "uniform_downweight":
        group_factor = torch.ones_like(opacity_sum)
        group_factor[over] = budget / torch.clamp(opacity_sum[over], min=1e-8)
        opacity_factor = group_factor[group_id]
        return keep, opacity_factor

    if args.opacity_budget_mode == "hard_topscore":
        score = make_score(args.opacity_budget_score, opacities, scale_metric, source_ids, args.newer_bonus)
        order, counts, starts, local_rank = grouped_rank_by_score(keys, score)
        op_sorted = opacities.float().clamp(min=0)[order]
        cum = torch.cumsum(op_sorted, dim=0)
        start_cum = cum[starts] - op_sorted[starts]
        local_cum = cum - torch.repeat_interleave(start_cum, counts)
        keep_sorted = (local_cum <= budget) | (local_rank == 0)
        keep = torch.zeros((keys.numel(),), device=keys.device, dtype=torch.bool)
        keep[order] = keep_sorted
        return keep, opacity_factor

    raise ValueError(f"Unknown opacity_budget_mode: {args.opacity_budget_mode}")


def build_voxel_csv_rows(
    coords: torch.Tensor,
    sort_order_by_key: torch.Tensor,
    counts: torch.Tensor,
    group_stats: Dict[str, torch.Tensor],
    max_rows: int,
) -> List[Dict[str, Any]]:
    if max_rows <= 0:
        return []
    starts = torch.cumsum(counts, dim=0) - counts
    rep_indices = sort_order_by_key[starts]
    c = counts.detach().cpu()
    opacity_sum = group_stats["opacity_sum"].detach().cpu()
    scale_min = group_stats["scale_min"].detach().cpu()
    scale_max = group_stats["scale_max"].detach().cpu()
    src_min = group_stats["source_min"].detach().cpu()
    src_max = group_stats["source_max"].detach().cpu()
    rep_coords = coords[rep_indices].detach().cpu()

    # Export the most crowded voxels first. They are the most suspicious for redundancy.
    topk = min(int(max_rows), int(c.numel()))
    _, top_idx = torch.topk(c, k=topk, largest=True)
    rows = []
    for gi_t in top_idx.tolist():
        xyz = rep_coords[gi_t].tolist()
        rows.append({
            "voxel_group_id": int(gi_t),
            "voxel_x": int(xyz[0]),
            "voxel_y": int(xyz[1]),
            "voxel_z": int(xyz[2]),
            "count": int(c[gi_t].item()),
            "opacity_sum": float(opacity_sum[gi_t].item()),
            "scale_min": float(scale_min[gi_t].item()),
            "scale_max": float(scale_max[gi_t].item()),
            "scale_ratio_max_min": float(scale_max[gi_t].item() / max(scale_min[gi_t].item(), 1e-8)),
            "source_min": int(src_min[gi_t].item()),
            "source_max": int(src_max[gi_t].item()),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_output_packet(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    gaussian_keys: Sequence[str],
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    output_pt: Path,
    meta: Dict[str, Any],
) -> None:
    base = dict(packets[selected_indices[0]].data)
    for k in ALL_GAUSSIAN_KEYS:
        if k in gaussian_keys:
            base[k] = fields[k][keep_mask].detach().cpu().contiguous()
        elif k in base:
            del base[k]
    base["source_packet_sorted_index"] = source_ids[keep_mask].detach().cpu().contiguous()
    base["fusion_source_packet_names"] = [packets[i].path.name for i in selected_indices]
    base["voxel_overlap_update_p4_meta"] = meta
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_pt)


def update_per_packet_rows(
    rows: List[Dict[str, Any]],
    source_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    modified_mask: torch.Tensor,
    opacities_before: torch.Tensor,
    opacities_after: torch.Tensor,
    scale_metric: torch.Tensor,
) -> List[Dict[str, Any]]:
    source_cpu = source_ids.detach().cpu()
    keep_cpu = keep_mask.detach().cpu()
    mod_cpu = modified_mask.detach().cpu()
    opa_b = opacities_before.detach().float().cpu()
    opa_a = opacities_after.detach().float().cpu()
    scale_cpu = scale_metric.detach().float().cpu()

    for row in rows:
        sid = int(row["source_packet_sorted_index"])
        m = source_cpu == sid
        n = int(m.sum().item())
        if n == 0:
            row.update({
                "num_output": 0,
                "num_hard_removed": 0,
                "num_modified_or_downweighted": 0,
                "opacity_mean_before": None,
                "opacity_mean_after": None,
                "scale_mean": None,
                "scale_max": None,
            })
            continue
        row["num_output"] = int((m & keep_cpu).sum().item())
        row["num_hard_removed"] = int((m & ~keep_cpu).sum().item())
        row["num_modified_or_downweighted"] = int((m & mod_cpu).sum().item())
        row["opacity_mean_before"] = float(opa_b[m].mean().item())
        row["opacity_mean_after"] = float(opa_a[m].mean().item())
        row["scale_mean"] = float(scale_cpu[m].mean().item())
        row["scale_max"] = float(scale_cpu[m].max().item())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 voxel-overlap-aware Gaussian update")
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--packet_ranges", type=str, required=True)
    parser.add_argument("--output_pt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)

    parser.add_argument("--mode", choices=["stats_only", "coarse_to_fine", "opacity_budget", "voxel_topk"], default="coarse_to_fine")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--min_voxel_count", type=int, default=2)

    parser.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    parser.add_argument("--scale_metric", choices=["max", "mean", "volume"], default="max")
    parser.add_argument("--scale_ratio", type=float, default=2.0)

    parser.add_argument("--old_mode", choices=["downweight", "hard"], default="downweight")
    parser.add_argument("--old_downweight", type=float, default=0.1)
    parser.add_argument("--old_min_opacity", type=float, default=0.0)
    parser.add_argument("--opacity_cap", type=float, default=0.0)

    parser.add_argument("--opacity_budget", type=float, default=0.0)
    parser.add_argument("--opacity_budget_mode", choices=["uniform_downweight", "hard_topscore"], default="uniform_downweight")
    parser.add_argument("--opacity_budget_score", choices=["small_scale", "opacity_over_scale", "opacity", "newer", "newer_small"], default="opacity_over_scale")

    parser.add_argument("--max_gaussians_per_voxel", type=int, default=4)
    parser.add_argument("--voxel_topk_score", choices=["small_scale", "opacity_over_scale", "opacity", "newer", "newer_small"], default="opacity_over_scale")
    parser.add_argument("--newer_bonus", type=float, default=0.25)

    parser.add_argument("--max_voxel_csv_rows", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading packets from {args.packet_dir}")
    packets = load_packets(args.packet_dir)
    selected_indices = parse_ranges(args.packet_ranges, len(packets))
    gaussian_keys = available_gaussian_keys(packets, selected_indices)
    if "scales" not in gaussian_keys:
        raise KeyError("P4 requires 'scales' in all selected packets.")

    print(f"      found packets: {len(packets)}")
    print(f"      selected packets: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''} ({len(selected_indices)} total)")
    print(f"      gaussian keys: {gaussian_keys}")

    print("[2/5] Concatenating Gaussian fields")
    fields, source_ids, per_packet_rows = concat_gaussian_fields(packets, selected_indices, gaussian_keys, device)
    N = int(fields["means"].shape[0])
    opacities_before = fields["opacities"].detach().clone()
    if args.opacity_cap > 0:
        fields["opacities"] = fields["opacities"].clone()
        fields["opacities"].clamp_(max=float(args.opacity_cap))
    opacities = fields["opacities"]

    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    print(f"      total Gaussians: {N:,}")
    print(f"      scale metric: mean={float(scale_metric.mean().item()):.6g}, max={float(scale_metric.max().item()):.6g}")
    print(f"      opacity: mean={float(opacities.float().mean().item()):.6g}, max={float(opacities.float().max().item()):.6g}")

    print("[3/5] Building voxel overlap index")
    coords, keys, coord_min, dims = compute_voxel_keys(fields["means"], args.voxel_size)
    order_by_key, unique_keys, counts, group_id = compute_group_index(keys)
    num_groups = int(unique_keys.numel())
    group_stats = scatter_group_stats(group_id, num_groups, opacities, scale_metric, source_ids)
    voxel_count = counts[group_id]
    print(f"      voxel_size: {args.voxel_size}")
    print(f"      voxel coord_min: {coord_min.detach().cpu().tolist()}, dims: {dims.detach().cpu().tolist()}")
    print(f"      occupied voxels: {num_groups:,}")
    print(f"      voxel count: mean={float(counts.float().mean().item()):.3f}, max={int(counts.max().item())}")

    print("[4/5] Applying P4 rule")
    keep_mask = torch.ones((N,), device=device, dtype=torch.bool)
    modified_mask = torch.zeros((N,), device=device, dtype=torch.bool)

    if args.mode == "stats_only":
        print("      mode=stats_only: no Gaussian is removed/downweighted.")

    elif args.mode == "coarse_to_fine":
        voxel_min_scale = group_stats["scale_min"][group_id]
        voxel_max_source = group_stats["source_max"][group_id]
        candidate = (
            (source_ids < voxel_max_source)
            & (scale_metric > float(args.scale_ratio) * voxel_min_scale)
            & (voxel_count >= int(args.min_voxel_count))
            & (opacities >= float(args.old_min_opacity))
        )
        modified_mask |= candidate
        if args.old_mode == "hard":
            keep_mask[candidate] = False
            print(f"      hard-removing old coarse Gaussians: {int(candidate.sum().item()):,}")
        else:
            opacities[candidate] = opacities[candidate] * float(args.old_downweight)
            print(f"      downweighting old coarse Gaussians: {int(candidate.sum().item()):,} by {args.old_downweight}")

    elif args.mode == "opacity_budget":
        if args.opacity_budget <= 0:
            raise ValueError("--mode opacity_budget requires --opacity_budget > 0")
        keep_budget, opacity_factor = apply_opacity_budget(
            keys=keys,
            group_id=group_id,
            group_stats=group_stats,
            opacities=opacities,
            scale_metric=scale_metric,
            source_ids=source_ids,
            args=args,
        )
        if args.opacity_budget_mode == "uniform_downweight":
            changed = opacity_factor < 0.999999
            opacities[:] = opacities * opacity_factor
            modified_mask |= changed
            print(f"      uniformly downweighted Gaussians in over-budget voxels: {int(changed.sum().item()):,}")
        else:
            keep_mask &= keep_budget
            modified_mask |= ~keep_budget
            print(f"      hard-topscore removed Gaussians by opacity budget: {int((~keep_budget).sum().item()):,}")

    elif args.mode == "voxel_topk":
        if args.max_gaussians_per_voxel <= 0:
            raise ValueError("--max_gaussians_per_voxel must be positive")
        keep_topk = apply_voxel_topk(keys, opacities, scale_metric, source_ids, args)
        keep_mask &= keep_topk
        modified_mask |= ~keep_topk
        print(f"      voxel-topk removed Gaussians: {int((~keep_topk).sum().item()):,}")

    else:
        raise ValueError(args.mode)

    fields["opacities"] = opacities
    total_out = int(keep_mask.sum().item())
    print(f"      output Gaussians: {total_out:,} / {N:,} ({total_out / max(N, 1):.4f})")

    print("[5/5] Saving fused packet and diagnostics")
    # Recompute post-update group opacity for summary only.
    opacity_after = fields["opacities"].detach()
    per_packet_rows = update_per_packet_rows(
        rows=per_packet_rows,
        source_ids=source_ids,
        keep_mask=keep_mask,
        modified_mask=modified_mask,
        opacities_before=opacities_before,
        opacities_after=opacity_after,
        scale_metric=scale_metric,
    )

    over_budget_voxels = None
    if args.opacity_budget > 0:
        over_budget_voxels = int((group_stats["opacity_sum"] > float(args.opacity_budget)).sum().item())

    meta: Dict[str, Any] = {
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "mode": args.mode,
        "voxel_size": args.voxel_size,
        "voxel_coord_min": coord_min.detach().cpu().tolist(),
        "voxel_dims": dims.detach().cpu().tolist(),
        "num_packets_found": len(packets),
        "selected_indices": selected_indices,
        "gaussian_keys": list(gaussian_keys),
        "num_input_gaussians": N,
        "num_output_gaussians": total_out,
        "output_ratio": total_out / max(N, 1),
        "num_occupied_voxels": num_groups,
        "voxel_count_mean": float(counts.float().mean().item()),
        "voxel_count_max": int(counts.max().item()),
        "scale_input": args.scale_input,
        "scale_metric": args.scale_metric,
        "scale_ratio": args.scale_ratio,
        "min_voxel_count": args.min_voxel_count,
        "old_mode": args.old_mode,
        "old_downweight": args.old_downweight,
        "old_min_opacity": args.old_min_opacity,
        "opacity_cap": args.opacity_cap,
        "opacity_budget": args.opacity_budget,
        "opacity_budget_mode": args.opacity_budget_mode,
        "opacity_budget_score": args.opacity_budget_score,
        "over_budget_voxels_before": over_budget_voxels,
        "max_gaussians_per_voxel": args.max_gaussians_per_voxel,
        "voxel_topk_score": args.voxel_topk_score,
        "newer_bonus": args.newer_bonus,
        "num_modified_or_downweighted": int(modified_mask.sum().item()),
        "num_hard_removed": int((~keep_mask).sum().item()),
        "opacity_mean_before": float(opacities_before.float().mean().item()),
        "opacity_mean_after": float(opacity_after.float().mean().item()),
        "scale_mean": float(scale_metric.float().mean().item()),
        "scale_max": float(scale_metric.float().max().item()),
    }

    save_output_packet(
        packets=packets,
        selected_indices=selected_indices,
        gaussian_keys=gaussian_keys,
        fields=fields,
        source_ids=source_ids,
        keep_mask=keep_mask,
        output_pt=args.output_pt,
        meta=meta,
    )
    print(f"      saved packet: {args.output_pt}")

    summary_path = output_dir / "voxel_overlap_update_p4_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"      saved summary: {summary_path}")

    per_packet_csv = output_dir / "voxel_overlap_update_p4_per_packet.csv"
    write_csv(per_packet_csv, per_packet_rows)
    print(f"      saved per-packet csv: {per_packet_csv}")

    voxel_rows = build_voxel_csv_rows(
        coords=coords,
        sort_order_by_key=order_by_key,
        counts=counts,
        group_stats=group_stats,
        max_rows=args.max_voxel_csv_rows,
    )
    voxel_csv = output_dir / "voxel_overlap_update_p4_top_voxels.csv"
    write_csv(voxel_csv, voxel_rows)
    if voxel_rows:
        print(f"      saved top-voxel csv: {voxel_csv}")


if __name__ == "__main__":
    main()
