#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P6: render-error-guided Gaussian cleanup for packet-level incremental 3DGS map maintenance.

Motivation
----------
P3/P4 are geometric heuristics:
  P3: trajectory/depth/extent dangerous-Gaussian filtering.
  P4: voxel-level redundancy control.

P6 directly tests a stronger hypothesis:
  after inserting a new packet, render the updated map from the current target view;
  if some image regions are wrong, identify old Gaussians whose projected footprint
  overlaps high-error pixels, accumulate a Gaussian-level bad score, and then
  delete/downweight/shrink those high-score Gaussians.

This is NOT true rasterizer-internal per-Gaussian contribution yet.
It is a practical approximation:
  RGB error mask + projected footprint sampling + opacity/scale weighting.

Recommended first experiment
----------------------------
Use only packets 27-29 first:

python render_error_guided_cleanup_p6.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 27-29 \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p6_seq_27_29_rgb_hard_p4topk64.pt \
  --execution_mode incremental \
  --cleanup_scope old_only \
  --p6_mode hard \
  --rgb_error_thresh 0.18 \
  --bad_mask_percentile 85 \
  --bad_mask_dilate_px 3 \
  --score_percentile 90 \
  --max_prune_per_step 200000 \
  --min_projected_radius_px 2 \
  --sample_radius_clip_px 80 \
  --enable_p4 \
  --voxel_size 0.05 \
  --max_gaussians_per_voxel 64 \
  --voxel_topk_score opacity_over_scale \
  --opacity_cap 0.30 \
  --device cuda:0

Then render with render_fused_packet_trajectory.py.

Full 0-79 sequence uses the same command with --packet_ranges 0-79.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PRIMARY_GAUSSIAN_KEYS = ["means", "covariances", "harmonics", "opacities"]
OPTIONAL_GAUSSIAN_KEYS = ["scales", "rotations", "rotations_unnorm"]
ALL_GAUSSIAN_KEYS = PRIMARY_GAUSSIAN_KEYS + OPTIONAL_GAUSSIAN_KEYS


@dataclass
class PacketRef:
    sorted_index: int
    path: Path
    data: Dict[str, Any]


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
        raise IndexError(f"indices out of range 0..{n-1}: {bad[:20]}")
    if not out:
        raise ValueError(f"No index selected by {spec}")
    return out


def ensure_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def first_matrix(x: Any, device: torch.device) -> torch.Tensor:
    t = ensure_tensor(x, device)
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


def first_int(x: Any, default: int) -> int:
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return default
        return int(round(float(x.detach().cpu().reshape(-1)[0].item())))
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return default
    return int(round(float(arr[0])))


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


def normalize_for_render(x: torch.Tensor, key: str) -> torch.Tensor:
    """Return field with explicit batch dimension: [1, N, ...]."""
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


def infer_image_shape(packet: Dict[str, Any]) -> Tuple[int, int]:
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


def target_image_tensor(packet: Dict[str, Any], device: torch.device, H: int, W: int) -> torch.Tensor:
    img = packet.get("target_image", None)
    if not isinstance(img, torch.Tensor):
        raise KeyError("Packet has no target_image tensor; P6 RGB error requires target_image.")
    img = img.detach().float()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        # [C,H,W]
        pass
    elif img.ndim == 3 and img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1)
    else:
        raise RuntimeError(f"Unexpected target_image shape: {tuple(img.shape)}")
    img = img.to(device)
    if img.shape[-2:] != (H, W):
        img = F.interpolate(img[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return img.clamp(0, 1).contiguous()


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


def available_gaussian_keys(packets: Sequence[PacketRef], selected: Sequence[int]) -> List[str]:
    keys = [
        k for k in ALL_GAUSSIAN_KEYS
        if all(k in packets[i].data and packets[i].data[k] is not None for i in selected)
    ]
    missing = [k for k in PRIMARY_GAUSSIAN_KEYS if k not in keys]
    if missing:
        raise KeyError(f"Missing primary keys: {missing}")
    if "scales" not in keys:
        raise KeyError("P6 requires scales for projected radius scoring.")
    return keys


def load_packet_fields(
    pref: PacketRef,
    gaussian_keys: Sequence[str],
    device: torch.device,
    uid_start: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    fields: Dict[str, torch.Tensor] = {}
    n: Optional[int] = None
    for k in gaussian_keys:
        t = normalize_gaussian_tensor(pref.data[k], k, device)
        if n is None:
            n = int(t.shape[0])
        elif int(t.shape[0]) != n:
            raise RuntimeError(f"{pref.path.name}: {k} length mismatch {t.shape[0]} != {n}")
        fields[k] = t
    assert n is not None
    source_ids = torch.full((n,), int(pref.sorted_index), device=device, dtype=torch.long)
    uids = torch.arange(uid_start, uid_start + n, device=device, dtype=torch.long)
    row = {
        "source_packet_sorted_index": int(pref.sorted_index),
        "source_packet_name": pref.path.name,
        "num_input": int(n),
    }
    return fields, source_ids, uids, row


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
    out = {k: torch.cat([map_fields[k], packet_fields[k]], dim=0).contiguous() for k in gaussian_keys}
    return out, torch.cat([map_source, packet_source], dim=0).contiguous(), torch.cat([map_uid, packet_uid], dim=0).contiguous()


def apply_keep_mask(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    uids: torch.Tensor,
    keep: torch.Tensor,
    gaussian_keys: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    fields = {k: fields[k][keep].contiguous() for k in gaussian_keys}
    return fields, source_ids[keep].contiguous(), uids[keep].contiguous()


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


def render_map_tensor(
    decoder,
    fields: Dict[str, torch.Tensor],
    packet: Dict[str, Any],
    device: torch.device,
    gaussian_keys: Sequence[str],
    H: int,
    W: int,
) -> torch.Tensor:
    Gaussians, _, _ = lazy_import_resplat(Path("."))  # already imported via sys.path
    g_kwargs = {}
    for k in ALL_GAUSSIAN_KEYS:
        if k in fields:
            g_kwargs[k] = normalize_for_render(fields[k], k)
    gaussians = Gaussians(**g_kwargs)

    ext = first_matrix(packet["target_extrinsics"], device)[None, None]
    K = first_matrix(packet["target_intrinsics"], device)[None, None]
    near = first_scalar(packet.get("target_near", None), 0.1).to(device)[None, None]
    far = first_scalar(packet.get("target_far", None), 50.0).to(device)[None, None]
    with torch.no_grad():
        out = decoder.forward(gaussians, ext, K, near, far, (H, W), depth_mode=None)
    return out.color[0, 0].detach().float().clamp(0, 1).contiguous()


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


def project_points(
    xyz_world: torch.Tensor,
    E: torch.Tensor,
    K_pixel: torch.Tensor,
    extrinsic_type: str,
    camera_z_sign: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if extrinsic_type == "Twc":
        Tcw = torch.linalg.inv(E)
    elif extrinsic_type == "Tcw":
        Tcw = E
    else:
        raise ValueError("--extrinsic_type must be Twc/Tcw")
    xyz_cam = xyz_world @ Tcw[:3, :3].T + Tcw[:3, 3][None, :]
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    if camera_z_sign == "positive":
        zf = z
    elif camera_z_sign == "negative":
        zf = -z
    else:
        raise ValueError("--camera_z_sign must be positive/negative")
    eps = 1e-8
    u = K_pixel[0, 0] * (x / (zf + eps)) + K_pixel[0, 2]
    v = K_pixel[1, 1] * (y / (zf + eps)) + K_pixel[1, 2]
    return u, v, zf


def make_bad_mask(
    render_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    rgb_thresh: float,
    percentile: float,
    dilate_px: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # [C,H,W]
    err = torch.mean(torch.abs(render_rgb - target_rgb), dim=0)
    if percentile > 0:
        q = torch.quantile(err.reshape(-1), float(percentile) / 100.0)
        thresh = max(float(rgb_thresh), float(q.item()))
    else:
        thresh = float(rgb_thresh)
    bad = err >= thresh
    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        bad_f = F.max_pool2d(bad.float()[None, None], kernel_size=k, stride=1, padding=int(dilate_px))[0, 0] > 0
        bad = bad_f
    meta = {
        "rgb_error_mean": float(err.mean().item()),
        "rgb_error_max": float(err.max().item()),
        "bad_mask_threshold": float(thresh),
        "bad_pixel_count": int(bad.sum().item()),
        "bad_pixel_ratio": float(bad.float().mean().item()),
    }
    return err.contiguous(), bad.float().contiguous(), meta


def sample_map_bilinear(img2d: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # img2d: [H,W], u/v: [M]
    H, W = img2d.shape
    if W <= 1 or H <= 1:
        raise ValueError("Invalid image shape")
    gx = (u / float(W - 1)) * 2.0 - 1.0
    gy = (v / float(H - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    val = F.grid_sample(
        img2d[None, None].float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return val.view(-1)


def build_offsets(device: torch.device, pattern: str) -> torch.Tensor:
    if pattern == "center":
        arr = [(0.0, 0.0)]
    elif pattern == "cross5":
        arr = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    elif pattern == "star9":
        arr = [
            (0.0, 0.0),
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
        ]
    elif pattern == "ring13":
        arr = [
            (0.0, 0.0),
            (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
        ]
    else:
        raise ValueError("--sample_pattern must be center/cross5/star9/ring13")
    return torch.tensor(arr, device=device, dtype=torch.float32)


def compute_p6_scores(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    current_packet_index: int,
    packet: Dict[str, Any],
    err_map: torch.Tensor,
    bad_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    device = fields["means"].device
    H, W = err_map.shape
    N = int(fields["means"].shape[0])
    scores = torch.zeros((N,), device=device, dtype=torch.float32)
    candidates = torch.zeros((N,), device=device, dtype=torch.bool)

    E = first_matrix(packet["target_extrinsics"], device)
    K_raw = first_matrix(packet["target_intrinsics"], device)
    K = denormalize_intrinsics(K_raw, H, W, args.intrinsics_normalized)

    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    opacities = fields["opacities"].float().clamp(min=0)
    f = float((K[0, 0] + K[1, 1]) * 0.5)
    offsets = build_offsets(device, args.sample_pattern)
    num_samples = int(offsets.shape[0])

    for start in range(0, N, int(args.chunk_size)):
        end = min(start + int(args.chunk_size), N)
        means = fields["means"][start:end]
        u, v, z = project_points(means, E, K, args.extrinsic_type, args.camera_z_sign)
        inside_center = (z > 0) & (u >= -args.sample_radius_clip_px) & (u < W + args.sample_radius_clip_px) & (v >= -args.sample_radius_clip_px) & (v < H + args.sample_radius_clip_px)

        sm = scale_metric[start:end]
        opa = opacities[start:end]
        radius = f * sm / torch.clamp(z, min=1e-6)
        radius_for_sample = torch.clamp(radius, min=0.0, max=float(args.sample_radius_clip_px))

        # [chunk, samples]
        us = u[:, None] + offsets[None, :, 0] * radius_for_sample[:, None]
        vs = v[:, None] + offsets[None, :, 1] * radius_for_sample[:, None]
        flat_u = us.reshape(-1)
        flat_v = vs.reshape(-1)
        sampled_err = sample_map_bilinear(err_map, flat_u, flat_v).view(-1, num_samples)
        sampled_bad = sample_map_bilinear(bad_mask, flat_u, flat_v).view(-1, num_samples)

        bad_weighted_err = sampled_err * sampled_bad
        bad_sum = bad_weighted_err.sum(dim=1)
        bad_count = (sampled_bad > 0.25).sum(dim=1)

        # Large-footprint Gaussians should accumulate stronger score.
        area_weight = torch.clamp((radius / max(float(args.radius_norm_px), 1e-6)) ** 2, min=0.25, max=float(args.max_area_weight))
        score = bad_sum * opa * area_weight

        if args.cleanup_scope == "old_only":
            scope = source_ids[start:end] < int(current_packet_index)
        elif args.cleanup_scope == "current_only":
            scope = source_ids[start:end] == int(current_packet_index)
        elif args.cleanup_scope == "all":
            scope = torch.ones_like(inside_center)
        else:
            raise ValueError("--cleanup_scope must be old_only/current_only/all")

        cand = (
            scope
            & inside_center
            & (opa >= float(args.min_opacity))
            & (radius >= float(args.min_projected_radius_px))
            & (bad_count >= int(args.min_bad_samples))
            & (bad_sum >= float(args.min_bad_error_sum))
        )

        scores[start:end] = score
        candidates[start:end] = cand

    meta = {
        "num_p6_candidates": int(candidates.sum().item()),
        "score_mean_all": float(scores.mean().item()),
        "score_max_all": float(scores.max().item()),
        "score_mean_candidates": float(scores[candidates].mean().item()) if torch.any(candidates) else 0.0,
        "score_max_candidates": float(scores[candidates].max().item()) if torch.any(candidates) else 0.0,
    }
    return scores, candidates, meta


def select_p6_gaussians(scores: torch.Tensor, candidates: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, Any]]:
    selected = torch.zeros_like(candidates)
    if not torch.any(candidates):
        return selected, {"p6_score_threshold": None, "num_p6_selected": 0}
    cand_scores = scores[candidates]
    if args.score_percentile > 0:
        q = torch.quantile(cand_scores, float(args.score_percentile) / 100.0)
        thresh = max(float(args.bad_score_thresh), float(q.item()))
    else:
        thresh = float(args.bad_score_thresh)
    selected = candidates & (scores >= thresh)

    if args.max_prune_per_step > 0 and int(selected.sum().item()) > int(args.max_prune_per_step):
        idx = torch.nonzero(selected, as_tuple=False).reshape(-1)
        top = torch.topk(scores[idx], k=int(args.max_prune_per_step), largest=True).indices
        keep_idx = idx[top]
        new_sel = torch.zeros_like(selected)
        new_sel[keep_idx] = True
        selected = new_sel

    return selected, {
        "p6_score_threshold": float(thresh),
        "num_p6_selected": int(selected.sum().item()),
        "selected_score_mean": float(scores[selected].mean().item()) if torch.any(selected) else 0.0,
        "selected_score_max": float(scores[selected].max().item()) if torch.any(selected) else 0.0,
    }


def shrink_gaussians(fields: Dict[str, torch.Tensor], mask: torch.Tensor, factor: float) -> None:
    if factor <= 0:
        raise ValueError("--shrink_factor must be positive")
    if not torch.any(mask):
        return
    f = float(factor)
    if "scales" in fields:
        fields["scales"][mask] = fields["scales"][mask] * f
    if "covariances" in fields:
        fields["covariances"][mask] = fields["covariances"][mask] * (f * f)


def apply_p6_cleanup(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    uids: torch.Tensor,
    current_packet_index: int,
    current_packet: Dict[str, Any],
    decoder,
    gaussian_keys: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    t0 = now()
    H, W = infer_image_shape(current_packet)
    render_rgb = render_map_tensor(decoder, fields, current_packet, fields["means"].device, gaussian_keys, H, W)
    target_rgb = target_image_tensor(current_packet, fields["means"].device, H, W)

    err_map, bad_mask, mask_meta = make_bad_mask(
        render_rgb,
        target_rgb,
        args.rgb_error_thresh,
        args.bad_mask_percentile,
        args.bad_mask_dilate_px,
    )
    scores, candidates, score_meta = compute_p6_scores(
        fields,
        source_ids,
        current_packet_index,
        current_packet,
        err_map,
        bad_mask,
        args,
    )
    selected, sel_meta = select_p6_gaussians(scores, candidates, args)

    keep = torch.ones((fields["means"].shape[0],), device=fields["means"].device, dtype=torch.bool)
    if args.p6_mode == "hard":
        keep[selected] = False
    elif args.p6_mode == "downweight":
        fields["opacities"][selected] = fields["opacities"][selected] * float(args.opacity_downweight)
    elif args.p6_mode == "shrink":
        shrink_gaussians(fields, selected, args.shrink_factor)
    elif args.p6_mode == "shrink_downweight":
        fields["opacities"][selected] = fields["opacities"][selected] * float(args.opacity_downweight)
        shrink_gaussians(fields, selected, args.shrink_factor)
    elif args.p6_mode == "none":
        pass
    else:
        raise ValueError("--p6_mode must be none/hard/downweight/shrink/shrink_downweight")

    removed = int((~keep).sum().item())
    if removed > 0:
        fields, source_ids, uids = apply_keep_mask(fields, source_ids, uids, keep, gaussian_keys)

    meta = {
        "p6_enabled": True,
        "p6_mode": args.p6_mode,
        "cleanup_scope": args.cleanup_scope,
        "num_gaussians_before_p6": int(keep.numel()),
        "num_p6_hard_removed": removed,
        "num_gaussians_after_p6": int(fields["means"].shape[0]),
        "p6_time_sec": now() - t0,
        **mask_meta,
        **score_meta,
        **sel_meta,
    }
    return fields, source_ids, uids, meta


def compute_voxel_keys(means: torch.Tensor, voxel_size: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.floor(means / float(voxel_size)).to(torch.long)
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    dims = cmax - cmin + 1
    shifted = coords - cmin
    keys = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]
    return keys.contiguous(), cmin, dims


def make_score(
    score_name: str,
    opacities: torch.Tensor,
    scale_metric: torch.Tensor,
    source_ids: torch.Tensor,
    newer_bonus: float,
) -> torch.Tensor:
    eps = 1e-8
    if score_name == "small_scale":
        return -scale_metric.float()
    if score_name == "opacity_over_scale":
        return opacities.float().clamp(min=0) / torch.clamp(scale_metric.float(), min=eps)
    if score_name == "opacity":
        return opacities.float()
    if score_name == "newer":
        src = source_ids.float()
        denom = torch.clamp(src.max() - src.min(), min=1.0)
        return (src - src.min()) / denom
    if score_name == "newer_small":
        src = source_ids.float()
        denom = torch.clamp(src.max() - src.min(), min=1.0)
        return -scale_metric.float() + float(newer_bonus) * ((src - src.min()) / denom)
    raise ValueError(f"Unknown score: {score_name}")


def grouped_rank_by_score(keys: torch.Tensor, score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order_score = torch.argsort(-score, stable=True)
    keys_after_score = keys[order_score]
    order_key = torch.argsort(keys_after_score, stable=True)
    order = order_score[order_key]
    keys_sorted = keys[order]
    _, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    local_rank = torch.arange(keys.numel(), device=keys.device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
    return order, counts, local_rank


def apply_p4_topk(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    uids: torch.Tensor,
    gaussian_keys: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not args.enable_p4:
        return fields, source_ids, uids, {"p4_enabled": False}
    t0 = now()
    if args.opacity_cap > 0:
        fields["opacities"] = torch.clamp(fields["opacities"], max=float(args.opacity_cap)).contiguous()
    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    keys, cmin, dims = compute_voxel_keys(fields["means"], args.voxel_size)
    score = make_score(args.voxel_topk_score, fields["opacities"], scale_metric, source_ids, args.newer_bonus)
    order, counts, local_rank = grouped_rank_by_score(keys, score)
    keep_sorted = local_rank < int(args.max_gaussians_per_voxel)
    keep = torch.zeros((fields["means"].shape[0],), device=fields["means"].device, dtype=torch.bool)
    keep[order] = keep_sorted
    removed = int((~keep).sum().item())
    if removed > 0:
        fields, source_ids, uids = apply_keep_mask(fields, source_ids, uids, keep, gaussian_keys)
    meta = {
        "p4_enabled": True,
        "p4_mode": "voxel_topk",
        "voxel_size": args.voxel_size,
        "max_gaussians_per_voxel": args.max_gaussians_per_voxel,
        "voxel_topk_score": args.voxel_topk_score,
        "num_occupied_voxels": int(counts.numel()),
        "voxel_count_mean": float(counts.float().mean().item()),
        "voxel_count_max": int(counts.max().item()),
        "num_p4_removed": removed,
        "num_gaussians_after_p4": int(fields["means"].shape[0]),
        "p4_time_sec": now() - t0,
    }
    return fields, source_ids, uids, meta


def update_per_packet_rows(rows: List[Dict[str, Any]], source_ids: torch.Tensor, fields: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    src = source_ids.detach().cpu()
    op = fields["opacities"].detach().float().cpu()
    for row in rows:
        sid = int(row["source_packet_sorted_index"])
        m = src == sid
        n = int(m.sum().item())
        row["num_output"] = n
        row["output_ratio"] = n / max(int(row["num_input"]), 1)
        row["opacity_mean_after"] = float(op[m].mean().item()) if n > 0 else None
        row["opacity_max_after"] = float(op[m].max().item()) if n > 0 else None
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_output_packet(
    base_packet: Dict[str, Any],
    selected_packet_names: List[str],
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    output_pt: Path,
    gaussian_keys: Sequence[str],
    meta: Dict[str, Any],
) -> None:
    out = dict(base_packet)
    for k in ALL_GAUSSIAN_KEYS:
        if k in gaussian_keys and k in fields:
            out[k] = fields[k].detach().cpu().contiguous()
        elif k in out:
            del out[k]
    out["source_packet_sorted_index"] = source_ids.detach().cpu().contiguous()
    out["fusion_source_packet_names"] = selected_packet_names
    out["p6_render_error_cleanup_meta"] = meta
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_pt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P6 render-error-guided Gaussian cleanup")
    p.add_argument("--packet_dir", type=Path, required=True)
    p.add_argument("--packet_ranges", type=str, required=True)
    p.add_argument("--resplat_repo", type=Path, required=True)
    p.add_argument("--output_pt", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--execution_mode", choices=["incremental"], default="incremental")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk_size", type=int, default=200_000)

    # P6 cleanup.
    p.add_argument("--p6_mode", choices=["none", "hard", "downweight", "shrink", "shrink_downweight"], default="hard")
    p.add_argument("--cleanup_scope", choices=["old_only", "current_only", "all"], default="old_only")
    p.add_argument("--rgb_error_thresh", type=float, default=0.18)
    p.add_argument("--bad_mask_percentile", type=float, default=85.0)
    p.add_argument("--bad_mask_dilate_px", type=int, default=3)
    p.add_argument("--sample_pattern", choices=["center", "cross5", "star9", "ring13"], default="star9")
    p.add_argument("--sample_radius_clip_px", type=float, default=80.0)
    p.add_argument("--min_projected_radius_px", type=float, default=2.0)
    p.add_argument("--min_opacity", type=float, default=0.01)
    p.add_argument("--min_bad_samples", type=int, default=1)
    p.add_argument("--min_bad_error_sum", type=float, default=0.02)
    p.add_argument("--radius_norm_px", type=float, default=20.0)
    p.add_argument("--max_area_weight", type=float, default=20.0)
    p.add_argument("--bad_score_thresh", type=float, default=0.02)
    p.add_argument("--score_percentile", type=float, default=90.0)
    p.add_argument("--max_prune_per_step", type=int, default=200_000)
    p.add_argument("--opacity_downweight", type=float, default=0.1)
    p.add_argument("--shrink_factor", type=float, default=0.5)

    # Geometry/projection.
    p.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    p.add_argument("--scale_metric", choices=["max", "mean", "volume"], default="max")
    p.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    p.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive")
    p.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto")

    # Gentle P4.
    p.add_argument("--enable_p4", action="store_true")
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--max_gaussians_per_voxel", type=int, default=64)
    p.add_argument("--voxel_topk_score", choices=["small_scale", "opacity_over_scale", "opacity", "newer", "newer_small"], default="opacity_over_scale")
    p.add_argument("--newer_bonus", type=float, default=0.25)
    p.add_argument("--opacity_cap", type=float, default=0.30)

    p.add_argument("--empty_cache_each_step", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    t_total = now()
    device = torch.device(args.device)
    out_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading packets from {args.packet_dir}")
    packets = load_packets(args.packet_dir)
    selected = parse_ranges(args.packet_ranges, len(packets))
    gaussian_keys = available_gaussian_keys(packets, selected)
    print(f"      found packets: {len(packets)}")
    print(f"      selected: {selected[:10]}{'...' if len(selected) > 10 else ''} ({len(selected)} total)")
    print(f"      gaussian keys: {gaussian_keys}")

    print("[2/5] Initializing ReSplat renderer")
    bg = packets[selected[0]].data.get("background_color", [0.0, 0.0, 0.0])
    decoder = make_decoder(args.resplat_repo, device, bg)

    print("[3/5] Incremental insert -> render-guided cleanup -> P4")
    map_fields: Optional[Dict[str, torch.Tensor]] = None
    map_source: Optional[torch.Tensor] = None
    map_uid: Optional[torch.Tensor] = None
    uid_cursor = 0
    step_rows: List[Dict[str, Any]] = []
    per_packet_rows: List[Dict[str, Any]] = []

    for step_i, idx in enumerate(selected):
        pref = packets[idx]
        f, src, uid, row = load_packet_fields(pref, gaussian_keys, device, uid_cursor)
        uid_cursor += int(uid.numel())
        per_packet_rows.append(row)

        map_fields, map_source, map_uid = append_map(map_fields, map_source, map_uid, f, src, uid, gaussian_keys)
        assert map_fields is not None and map_source is not None and map_uid is not None
        n_after_insert = int(map_fields["means"].shape[0])

        p6_meta = {"p6_enabled": False}
        if step_i > 0 and args.p6_mode != "none":
            map_fields, map_source, map_uid, p6_meta = apply_p6_cleanup(
                map_fields, map_source, map_uid,
                current_packet_index=idx,
                current_packet=pref.data,
                decoder=decoder,
                gaussian_keys=gaussian_keys,
                args=args,
            )

        p4_meta = {"p4_enabled": False}
        if args.enable_p4:
            map_fields, map_source, map_uid, p4_meta = apply_p4_topk(
                map_fields, map_source, map_uid, gaussian_keys, args
            )

        n_after = int(map_fields["means"].shape[0])
        step_row = {
            "step": int(step_i),
            "packet_sorted_index": int(idx),
            "packet_name": pref.path.name,
            "num_gaussians_after_insert": int(n_after_insert),
            "num_gaussians_after_maintenance": int(n_after),
        }
        step_row.update({f"p6_{k}": v for k, v in p6_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        step_row.update({f"p4_{k}": v for k, v in p4_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        step_rows.append(step_row)

        print(
            f"      [{step_i+1:03d}/{len(selected):03d}] packet={idx:04d}, "
            f"after_insert={n_after_insert:,}, after={n_after:,}, "
            f"bad_pixels={p6_meta.get('bad_pixel_count', None)}, "
            f"p6_selected={p6_meta.get('num_p6_selected', None)}, "
            f"p6_removed={p6_meta.get('num_p6_hard_removed', None)}, "
            f"p4_removed={p4_meta.get('num_p4_removed', None)}"
        )
        if args.empty_cache_each_step and device.type == "cuda":
            torch.cuda.empty_cache()

    assert map_fields is not None and map_source is not None
    print("[4/5] Saving output packet")
    per_packet_rows = update_per_packet_rows(per_packet_rows, map_source, map_fields)
    total_in = int(sum(r["num_input"] for r in per_packet_rows))
    total_out = int(map_fields["means"].shape[0])
    meta = {
        "script": "render_error_guided_cleanup_p6.py",
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "selected_indices": selected,
        "resplat_repo": str(args.resplat_repo),
        "num_input_gaussians": total_in,
        "num_output_gaussians": total_out,
        "output_ratio": total_out / max(total_in, 1),
        "p6_mode": args.p6_mode,
        "cleanup_scope": args.cleanup_scope,
        "rgb_error_thresh": args.rgb_error_thresh,
        "bad_mask_percentile": args.bad_mask_percentile,
        "score_percentile": args.score_percentile,
        "max_prune_per_step": args.max_prune_per_step,
        "enable_p4": bool(args.enable_p4),
        "voxel_size": args.voxel_size,
        "max_gaussians_per_voxel": args.max_gaussians_per_voxel,
        "total_time_sec": now() - t_total,
    }
    save_output_packet(
        base_packet=packets[selected[0]].data,
        selected_packet_names=[packets[i].path.name for i in selected],
        fields=map_fields,
        source_ids=map_source,
        output_pt=args.output_pt,
        gaussian_keys=gaussian_keys,
        meta=meta,
    )

    print("[5/5] Saving diagnostics")
    summary_path = out_dir / "p6_render_error_cleanup_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    write_csv(out_dir / "p6_render_error_cleanup_steps.csv", step_rows)
    write_csv(out_dir / "p6_render_error_cleanup_per_packet.csv", per_packet_rows)

    print(f"      saved packet: {args.output_pt}")
    print(f"      saved summary: {summary_path}")
    print(f"      saved steps csv: {out_dir / 'p6_render_error_cleanup_steps.csv'}")
    print(f"      saved per-packet csv: {out_dir / 'p6_render_error_cleanup_per_packet.csv'}")
    print(f"      total: in={total_in:,}, out={total_out:,}, ratio={total_out / max(total_in, 1):.4f}")
    print(f"      total_time_sec={meta['total_time_sec']:.3f}")


if __name__ == "__main__":
    main()
