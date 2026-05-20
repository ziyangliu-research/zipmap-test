#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P7-A: render-supervised local opacity optimization for incremental 3DGS map maintenance.

Core idea
---------
For each incoming local Gaussian packet G_t:

  1. Existing map M_{t-1} is already available.
  2. Insert G_t to obtain temporary map M'_t.
  3. Render M'_t from the current target view.
  4. Optimize only OLD Gaussian opacities using RGB reconstruction loss.
  5. Optionally apply gentle voxel top-k compression P4.

This is a first optimization-based map-maintenance baseline.
It is deliberately conservative:
  - Poses are fixed.
  - Gaussian means are fixed.
  - Covariances/scales/rotations are fixed.
  - SH/color is fixed.
  - Current packet Gaussians are frozen by default.
  - Only old opacity values are optimized.

Recommended first run
---------------------
Use 25-35 first, no P4:

python render_supervised_opacity_update_p7.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 25-35 \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/p7_seq_25_35_opacity_rgb_noP4.pt \
  --optimize_scope visible_old \
  --num_steps 20 \
  --lr_opacity 0.05 \
  --loss_mode l1 \
  --loss_mask_mode none \
  --opacity_cap 0.30 \
  --visible_margin_px 64 \
  --min_projected_radius_px 1.0 \
  --device cuda:0

Then render with render_fused_packet_trajectory.py.

If this improves 28/29, test with gentle P4 by adding:
  --enable_p4 --voxel_size 0.05 --max_gaussians_per_voxel 64
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


def tnow() -> float:
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
        raise KeyError("Packet has no target_image tensor; P7 RGB loss requires target_image.")
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
        raise KeyError("P7 visible-scope selection requires scales.")
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
        fields[k] = t.detach().contiguous()
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


def make_gaussians(Gaussians_cls, fields: Dict[str, torch.Tensor], gaussian_keys: Sequence[str], opacities_override: Optional[torch.Tensor] = None):
    kwargs: Dict[str, torch.Tensor] = {}
    for k in ALL_GAUSSIAN_KEYS:
        if k not in fields:
            continue
        val = fields[k]
        if k == "opacities" and opacities_override is not None:
            val = opacities_override
        kwargs[k] = normalize_for_render(val, k)
    return Gaussians_cls(**kwargs)


def render_rgb(
    Gaussians_cls,
    decoder,
    fields: Dict[str, torch.Tensor],
    packet: Dict[str, Any],
    device: torch.device,
    gaussian_keys: Sequence[str],
    H: int,
    W: int,
    opacities_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gaussians = make_gaussians(Gaussians_cls, fields, gaussian_keys, opacities_override=opacities_override)
    ext = first_matrix(packet["target_extrinsics"], device)[None, None]
    K = first_matrix(packet["target_intrinsics"], device)[None, None]
    near = first_scalar(packet.get("target_near", None), 0.1).to(device)[None, None]
    far = first_scalar(packet.get("target_far", None), 50.0).to(device)[None, None]
    out = decoder.forward(gaussians, ext, K, near, far, (H, W), depth_mode=None)
    return out.color[0, 0].float().clamp(0, 1)


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


def build_optimization_mask(
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    current_packet_index: int,
    packet: Dict[str, Any],
    H: int,
    W: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = fields["means"].device
    N = int(fields["means"].shape[0])

    if args.optimize_scope == "old_all":
        mask = source_ids < int(current_packet_index)
        return mask, {"num_opt_candidates": int(mask.sum().item()), "opt_scope": args.optimize_scope}

    if args.optimize_scope == "all":
        mask = torch.ones((N,), device=device, dtype=torch.bool)
        return mask, {"num_opt_candidates": int(mask.sum().item()), "opt_scope": args.optimize_scope}

    if args.optimize_scope not in {"visible_old", "visible_all"}:
        raise ValueError("--optimize_scope must be visible_old/old_all/visible_all/all")

    E = first_matrix(packet["target_extrinsics"], device)
    K_raw = first_matrix(packet["target_intrinsics"], device)
    K = denormalize_intrinsics(K_raw, H, W, args.intrinsics_normalized)
    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    f = float((K[0, 0] + K[1, 1]) * 0.5)

    visible = torch.zeros((N,), device=device, dtype=torch.bool)
    radius_all = torch.zeros((N,), device=device, dtype=torch.float32)
    for start in range(0, N, int(args.projection_chunk_size)):
        end = min(start + int(args.projection_chunk_size), N)
        u, v, z = project_points(
            fields["means"][start:end],
            E,
            K,
            args.extrinsic_type,
            args.camera_z_sign,
        )
        radius = f * scale_metric[start:end] / torch.clamp(z, min=1e-6)
        inside = (
            (z > 0)
            & (u >= -float(args.visible_margin_px))
            & (u < W + float(args.visible_margin_px))
            & (v >= -float(args.visible_margin_px))
            & (v < H + float(args.visible_margin_px))
            & (radius >= float(args.min_projected_radius_px))
            & (fields["opacities"][start:end] >= float(args.min_opacity_for_opt))
        )
        visible[start:end] = inside
        radius_all[start:end] = radius

    if args.optimize_scope == "visible_old":
        mask = visible & (source_ids < int(current_packet_index))
    else:
        mask = visible

    if args.max_opt_gaussians > 0 and int(mask.sum().item()) > int(args.max_opt_gaussians):
        # Keep the largest projected-radius/opacities score candidates.
        score = radius_all * torch.clamp(fields["opacities"].float(), min=0)
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        top = torch.topk(score[idx], k=int(args.max_opt_gaussians), largest=True).indices
        keep_idx = idx[top]
        reduced = torch.zeros_like(mask)
        reduced[keep_idx] = True
        mask = reduced

    return mask, {
        "num_opt_candidates": int(mask.sum().item()),
        "opt_scope": args.optimize_scope,
        "visible_margin_px": float(args.visible_margin_px),
        "min_projected_radius_px": float(args.min_projected_radius_px),
    }


def make_loss_mask(rendered: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> Optional[torch.Tensor]:
    if args.loss_mask_mode == "none":
        return None
    with torch.no_grad():
        err = torch.mean(torch.abs(rendered.detach() - target), dim=0, keepdim=True)
        if args.loss_mask_mode == "error_threshold":
            mask = err >= float(args.loss_mask_thresh)
        elif args.loss_mask_mode == "error_percentile":
            q = torch.quantile(err.reshape(-1), float(args.loss_mask_percentile) / 100.0)
            mask = err >= q
        else:
            raise ValueError("--loss_mask_mode must be none/error_threshold/error_percentile")
        if args.loss_mask_dilate_px > 0:
            k = int(args.loss_mask_dilate_px) * 2 + 1
            mask = F.max_pool2d(mask.float()[None], kernel_size=k, stride=1, padding=int(args.loss_mask_dilate_px))[0] > 0
    return mask.float()


def compute_rgb_loss(rendered: torch.Tensor, target: torch.Tensor, args: argparse.Namespace, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if args.loss_mode == "l1":
        per = torch.abs(rendered - target)
    elif args.loss_mode == "l2":
        per = (rendered - target) ** 2
    elif args.loss_mode == "charbonnier":
        per = torch.sqrt((rendered - target) ** 2 + 1e-6)
    else:
        raise ValueError("--loss_mode must be l1/l2/charbonnier")

    if mask is not None:
        denom = torch.clamp(mask.sum() * rendered.shape[0], min=1.0)
        return (per * mask).sum() / denom
    return per.mean()


def opacities_to_logits(opacities: torch.Tensor, cap: float, eps: float = 1e-5) -> torch.Tensor:
    cap = float(cap)
    x = torch.clamp(opacities / cap, min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def logits_to_opacities(logits: torch.Tensor, cap: float) -> torch.Tensor:
    return float(cap) * torch.sigmoid(logits)


def optimize_old_opacity_for_current_view(
    Gaussians_cls,
    decoder,
    fields: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    current_packet_index: int,
    current_packet: Dict[str, Any],
    gaussian_keys: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    device = fields["means"].device
    H, W = infer_image_shape(current_packet)
    target = target_image_tensor(current_packet, device, H, W)

    opt_mask, mask_meta = build_optimization_mask(fields, source_ids, current_packet_index, current_packet, H, W, args)
    nopt = int(opt_mask.sum().item())
    if nopt == 0 or args.num_steps <= 0:
        return fields, {
            "p7_enabled": True,
            "num_opt_gaussians": nopt,
            "num_steps": 0,
            "loss_initial": None,
            "loss_final": None,
            **mask_meta,
        }

    # Baseline render and optional loss mask.
    with torch.no_grad():
        rendered0 = render_rgb(Gaussians_cls, decoder, fields, current_packet, device, gaussian_keys, H, W)
    loss_mask = make_loss_mask(rendered0, target, args)

    base_opacities = fields["opacities"].detach().clone()
    opt_init = base_opacities[opt_mask].detach().clone()
    logits = torch.nn.Parameter(opacities_to_logits(torch.clamp(opt_init, min=0.0, max=float(args.opacity_cap)), args.opacity_cap))
    optimizer = torch.optim.Adam([logits], lr=float(args.lr_opacity))

    loss_initial_val: Optional[float] = None
    loss_final_val: Optional[float] = None
    op_initial_mean = float(opt_init.mean().item())
    op_initial_max = float(opt_init.max().item())

    for it in range(int(args.num_steps)):
        optimizer.zero_grad(set_to_none=True)
        opt_values = logits_to_opacities(logits, args.opacity_cap)
        op_override = base_opacities.clone()
        op_override[opt_mask] = opt_values

        rendered = render_rgb(
            Gaussians_cls,
            decoder,
            fields,
            current_packet,
            device,
            gaussian_keys,
            H,
            W,
            opacities_override=op_override,
        )
        loss_rgb = compute_rgb_loss(rendered, target, args, loss_mask)
        loss = loss_rgb

        if args.opacity_l1 > 0:
            loss = loss + float(args.opacity_l1) * opt_values.mean()
        if args.opacity_l2_delta > 0:
            loss = loss + float(args.opacity_l2_delta) * ((opt_values - opt_init) ** 2).mean()

        if it == 0:
            loss_initial_val = float(loss_rgb.detach().item())
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([logits], float(args.grad_clip))
        optimizer.step()
        loss_final_val = float(loss_rgb.detach().item())

        if args.verbose_opt and ((it + 1) % max(1, int(args.print_every)) == 0 or it == 0):
            print(
                f"        opt iter {it+1:03d}/{args.num_steps}: "
                f"rgb_loss={float(loss_rgb.detach().item()):.6f}, "
                f"opacity_mean={float(opt_values.detach().mean().item()):.6f}"
            )

    with torch.no_grad():
        opt_final = logits_to_opacities(logits, args.opacity_cap).detach()
        fields["opacities"] = fields["opacities"].detach().clone()
        fields["opacities"][opt_mask] = opt_final
        fields["opacities"] = torch.clamp(fields["opacities"], min=0.0, max=float(args.opacity_cap)).contiguous()

    meta = {
        "p7_enabled": True,
        "num_opt_gaussians": nopt,
        "num_steps": int(args.num_steps),
        "lr_opacity": float(args.lr_opacity),
        "loss_mode": args.loss_mode,
        "loss_mask_mode": args.loss_mask_mode,
        "loss_initial": loss_initial_val,
        "loss_final": loss_final_val,
        "loss_delta": None if loss_initial_val is None or loss_final_val is None else float(loss_final_val - loss_initial_val),
        "opacity_mean_before": op_initial_mean,
        "opacity_max_before": op_initial_max,
        "opacity_mean_after": float(opt_final.mean().item()),
        "opacity_max_after": float(opt_final.max().item()),
        "opacity_changed_mean_abs": float(torch.abs(opt_final - opt_init).mean().item()),
        **mask_meta,
    }
    return fields, meta


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
    t0 = tnow()
    if args.opacity_cap > 0:
        fields["opacities"] = torch.clamp(fields["opacities"], max=float(args.opacity_cap)).contiguous()
    scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
    keys, _, _ = compute_voxel_keys(fields["means"], args.voxel_size)
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
        "p4_time_sec": tnow() - t0,
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
    out["p7_render_supervised_opacity_meta"] = meta
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_pt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P7-A render-supervised local opacity optimization")
    p.add_argument("--packet_dir", type=Path, required=True)
    p.add_argument("--packet_ranges", type=str, required=True)
    p.add_argument("--resplat_repo", type=Path, required=True)
    p.add_argument("--output_pt", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--empty_cache_each_step", action="store_true")

    # Optimization.
    p.add_argument("--optimize_scope", choices=["visible_old", "old_all", "visible_all", "all"], default="visible_old")
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--lr_opacity", type=float, default=0.05)
    p.add_argument("--opacity_cap", type=float, default=0.30)
    p.add_argument("--opacity_l1", type=float, default=0.0)
    p.add_argument("--opacity_l2_delta", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--max_opt_gaussians", type=int, default=0, help="0 means no limit.")
    p.add_argument("--verbose_opt", action="store_true")
    p.add_argument("--print_every", type=int, default=5)

    # Loss.
    p.add_argument("--loss_mode", choices=["l1", "l2", "charbonnier"], default="l1")
    p.add_argument("--loss_mask_mode", choices=["none", "error_threshold", "error_percentile"], default="none")
    p.add_argument("--loss_mask_thresh", type=float, default=0.18)
    p.add_argument("--loss_mask_percentile", type=float, default=85.0)
    p.add_argument("--loss_mask_dilate_px", type=int, default=3)

    # Visibility selection / projection.
    p.add_argument("--visible_margin_px", type=float, default=64.0)
    p.add_argument("--min_projected_radius_px", type=float, default=1.0)
    p.add_argument("--min_opacity_for_opt", type=float, default=0.005)
    p.add_argument("--projection_chunk_size", type=int, default=500_000)
    p.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    p.add_argument("--scale_metric", choices=["max", "mean", "volume"], default="max")
    p.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    p.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive")
    p.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto")

    # Optional P4.
    p.add_argument("--enable_p4", action="store_true")
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--max_gaussians_per_voxel", type=int, default=64)
    p.add_argument("--voxel_topk_score", choices=["small_scale", "opacity_over_scale", "opacity", "newer", "newer_small"], default="opacity_over_scale")
    p.add_argument("--newer_bonus", type=float, default=0.25)

    return p


def main() -> None:
    args = build_parser().parse_args()
    t_total = tnow()
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
    Gaussians_cls, _, _ = lazy_import_resplat(args.resplat_repo)
    decoder = make_decoder(args.resplat_repo, device, bg)

    print("[3/5] Incremental insert -> opacity optimization -> optional P4")
    map_fields: Optional[Dict[str, torch.Tensor]] = None
    map_source: Optional[torch.Tensor] = None
    map_uid: Optional[torch.Tensor] = None
    uid_cursor = 0
    step_rows: List[Dict[str, Any]] = []
    per_packet_rows: List[Dict[str, Any]] = []

    for step_i, idx in enumerate(selected):
        step_t0 = tnow()
        pref = packets[idx]
        f, src, uid, row = load_packet_fields(pref, gaussian_keys, device, uid_cursor)
        uid_cursor += int(uid.numel())
        per_packet_rows.append(row)

        map_fields, map_source, map_uid = append_map(map_fields, map_source, map_uid, f, src, uid, gaussian_keys)
        assert map_fields is not None and map_source is not None and map_uid is not None
        n_after_insert = int(map_fields["means"].shape[0])

        p7_meta = {"p7_enabled": False}
        if step_i > 0 and args.num_steps > 0:
            opt_t0 = tnow()
            map_fields, p7_meta = optimize_old_opacity_for_current_view(
                Gaussians_cls,
                decoder,
                map_fields,
                map_source,
                current_packet_index=idx,
                current_packet=pref.data,
                gaussian_keys=gaussian_keys,
                args=args,
            )
            p7_meta["p7_time_sec"] = tnow() - opt_t0

        p4_meta = {"p4_enabled": False}
        if args.enable_p4:
            map_fields, map_source, map_uid, p4_meta = apply_p4_topk(
                map_fields,
                map_source,
                map_uid,
                gaussian_keys,
                args,
            )

        n_after = int(map_fields["means"].shape[0])
        step_row = {
            "step": int(step_i),
            "packet_sorted_index": int(idx),
            "packet_name": pref.path.name,
            "num_gaussians_after_insert": int(n_after_insert),
            "num_gaussians_after_maintenance": int(n_after),
            "step_time_sec": tnow() - step_t0,
        }
        step_row.update({f"p7_{k}": v for k, v in p7_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        step_row.update({f"p4_{k}": v for k, v in p4_meta.items() if isinstance(v, (int, float, str, bool)) or v is None})
        step_rows.append(step_row)

        print(
            f"      [{step_i+1:03d}/{len(selected):03d}] packet={idx:04d}, "
            f"after_insert={n_after_insert:,}, after={n_after:,}, "
            f"opt_n={p7_meta.get('num_opt_gaussians', None)}, "
            f"loss={p7_meta.get('loss_initial', None)}->{p7_meta.get('loss_final', None)}, "
            f"op_mean={p7_meta.get('opacity_mean_before', None)}->{p7_meta.get('opacity_mean_after', None)}, "
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
        "script": "render_supervised_opacity_update_p7.py",
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "selected_indices": selected,
        "resplat_repo": str(args.resplat_repo),
        "num_input_gaussians": total_in,
        "num_output_gaussians": total_out,
        "output_ratio": total_out / max(total_in, 1),
        "optimize_scope": args.optimize_scope,
        "num_steps": args.num_steps,
        "lr_opacity": args.lr_opacity,
        "loss_mode": args.loss_mode,
        "loss_mask_mode": args.loss_mask_mode,
        "opacity_cap": args.opacity_cap,
        "enable_p4": bool(args.enable_p4),
        "voxel_size": args.voxel_size if args.enable_p4 else None,
        "max_gaussians_per_voxel": args.max_gaussians_per_voxel if args.enable_p4 else None,
        "total_time_sec": tnow() - t_total,
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
    summary_path = out_dir / "p7_opacity_update_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    write_csv(out_dir / "p7_opacity_update_steps.csv", step_rows)
    write_csv(out_dir / "p7_opacity_update_per_packet.csv", per_packet_rows)

    print(f"      saved packet: {args.output_pt}")
    print(f"      saved summary: {summary_path}")
    print(f"      saved steps csv: {out_dir / 'p7_opacity_update_steps.csv'}")
    print(f"      saved per-packet csv: {out_dir / 'p7_opacity_update_per_packet.csv'}")
    print(f"      total: in={total_in:,}, out={total_out:,}, ratio={total_out / max(total_in, 1):.4f}")
    print(f"      total_time_sec={meta['total_time_sec']:.3f}")


if __name__ == "__main__":
    main()
