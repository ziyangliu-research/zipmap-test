#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-guided lightweight Gaussian packet filtering / map maintenance P1.

Purpose
-------
This is a non-optimization baseline for feed-forward 3DGS packet fusion.

For each incoming packet, before direct concat, the script applies simple
depth-guided and opacity/scale-aware maintenance rules:

1. Self-depth consistency
   Project each Gaussian center into the packet target camera and compare its
   camera depth z_g with the observed target depth D(u).

   free-space candidate:
       z_g < D(u) - (margin_abs + margin_rel * D(u))

   behind-surface candidate:
       z_g > D(u) + (margin_abs + margin_rel * D(u))

2. Opacity cap
   Clamp opacity to reduce alpha accumulation.

3. Danger suppression
   Downweight high-opacity + large-scale Gaussians using a simple score:
       score = opacity * scale_metric
   where scale_metric is max(scale) or volume(scale).

The filtered packets are then concatenated into one fused packet .pt that can be
rendered by your existing fused trajectory renderer.

Typical usage
-------------
python depth_guided_packet_filter_p1.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --depth_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/depth_lcam_front \
  --depth_pattern "{index:06d}_lcam_front_depth.png" \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/depth_filter_p1_opcap03.pt \
  --free_space_mode downweight \
  --free_space_downweight 0.05 \
  --behind_mode none \
  --opacity_cap 0.30 \
  --danger_top_frac 0.01 \
  --danger_downweight 0.2 \
  --margin_abs 0.10 \
  --margin_rel 0.02 \
  --device cuda:0

Recommended first ablation
--------------------------
A. opacity cap only:
   --free_space_mode none --behind_mode none --opacity_cap 0.30 --danger_top_frac 0

B. self-depth free-space only:
   --free_space_mode downweight --free_space_downweight 0.05 --opacity_cap 0 --danger_top_frac 0

C. opacity cap + danger suppression:
   --free_space_mode none --opacity_cap 0.30 --danger_top_frac 0.01 --danger_downweight 0.2

D. combined:
   --free_space_mode downweight --opacity_cap 0.30 --danger_top_frac 0.01
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
from PIL import Image


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
        raise IndexError(f"Packet indices out of range 0..{n-1}: {bad[:20]}")
    return out


def ensure_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def normalize_gaussian_tensor(x: Any, key: str, device: torch.device) -> torch.Tensor:
    """Return unbatched Gaussian tensor with first dim = G."""
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


def infer_image_shape(packet: Dict[str, Any], depth: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    if depth is not None:
        return int(depth.shape[-2]), int(depth.shape[-1])
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
    raise RuntimeError("Cannot infer image shape. Provide depth maps or packet image_shape.")


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
        # TartanAir v2 RGBA uint8 float32-packed depth.
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


def packet_depth(packet: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    for k in ("target_depth", "depth", "pred_depth", "zipmap_depth", "depth_map"):
        if k in packet:
            d = ensure_tensor(packet[k], device=device)
            while d.ndim > 2:
                d = d[0]
            return d.float()
    return None


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
        raise ValueError("extrinsic_type must be Twc or Tcw")

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


def scale_to_positive(scales: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return torch.clamp(scales.abs(), min=1e-8)
    if mode == "log":
        return torch.exp(scales)
    if mode == "auto":
        # If many values are <= 0, likely log-space; otherwise use positive raw scale.
        frac_nonpos = float((scales <= 0).float().mean().item())
        if frac_nonpos > 0.05:
            return torch.exp(scales)
        return torch.clamp(scales, min=1e-8)
    raise ValueError("--scale_input must be auto/raw/log")


def compute_scale_metric(scales: torch.Tensor, scale_input: str, scale_metric: str) -> torch.Tensor:
    s = scale_to_positive(scales, scale_input)
    if scale_metric == "max":
        return torch.max(s, dim=-1).values
    if scale_metric == "volume":
        return torch.prod(s, dim=-1)
    if scale_metric == "mean":
        return torch.mean(s, dim=-1)
    raise ValueError("--scale_metric must be max/volume/mean")


def apply_mode(
    opacities: torch.Tensor,
    keep_mask: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    downweight: float,
) -> None:
    if not torch.any(mask):
        return
    if mode == "none":
        return
    if mode == "hard":
        keep_mask[mask] = False
    elif mode == "downweight":
        opacities[mask] = opacities[mask] * float(downweight)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def process_one_packet(
    packet_ref: PacketRef,
    gaussian_keys: Sequence[str],
    depth_dir: Optional[Path],
    depth_pattern: str,
    depth_scale: float,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
    packet = packet_ref.data
    fields = {k: normalize_gaussian_tensor(packet[k], k, device) for k in gaussian_keys}
    means = fields["means"]
    opacities = fields["opacities"].clone()
    fields["opacities"] = opacities
    N = int(means.shape[0])
    keep_mask = torch.ones((N,), device=device, dtype=torch.bool)

    # Load target depth.
    depth = packet_depth(packet, device)
    frame_index = first_int(packet.get("target_index", packet_ref.sorted_index), default=packet_ref.sorted_index)
    if depth is None:
        if depth_dir is None:
            raise ValueError("No depth inside packet and --depth_dir is missing.")
        depth_path = depth_dir / depth_pattern.format(index=int(frame_index))
        depth = load_depth(depth_path, depth_scale, device)

    H, W = infer_image_shape(packet, depth)
    K = denormalize_intrinsics(first_matrix(packet["target_intrinsics"], device), H, W, args.intrinsics_normalized)
    E = first_matrix(packet["target_extrinsics"], device)

    # Project centers to own target depth.
    u, v, z = project_points(means, E, K, args.extrinsic_type, args.camera_z_sign)
    ui = torch.round(u).long()
    vi = torch.round(v).long()

    valid = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    near = first_scalar(packet.get("target_near", None), None)
    far = first_scalar(packet.get("target_far", None), None)
    if near is not None:
        valid &= z >= float(near)
    if far is not None:
        valid &= z <= float(far)

    depth_valid_full = torch.zeros((N,), device=device, dtype=torch.bool)
    free_space_mask = torch.zeros((N,), device=device, dtype=torch.bool)
    behind_mask = torch.zeros((N,), device=device, dtype=torch.bool)

    if torch.any(valid):
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        d = depth[vi[valid_idx], ui[valid_idx]]
        d_valid = torch.isfinite(d) & (d > 0)
        if torch.any(d_valid):
            valid_idx = valid_idx[d_valid]
            d = d[d_valid]
            depth_valid_full[valid_idx] = True
            z_valid = z[valid_idx]
            margin = float(args.margin_abs) + float(args.margin_rel) * d
            free_space_mask[valid_idx] = z_valid < (d - margin)
            behind_mask[valid_idx] = z_valid > (d + margin)

    # Rule 1: self-depth consistency.
    apply_mode(opacities, keep_mask, free_space_mask, args.free_space_mode, args.free_space_downweight)
    apply_mode(opacities, keep_mask, behind_mask, args.behind_mode, args.behind_downweight)

    # Rule 2: global/per-packet opacity cap.
    if args.opacity_cap > 0:
        opacities.clamp_(max=float(args.opacity_cap))

    # Rule 3: high-opacity + large-scale suppression.
    danger_mask = torch.zeros((N,), device=device, dtype=torch.bool)
    if args.danger_top_frac > 0:
        if "scales" not in fields:
            raise KeyError("--danger_top_frac requires scales in packet.")
        scale_metric = compute_scale_metric(fields["scales"], args.scale_input, args.scale_metric)
        # Use post-depth-rule opacity. This is intentional.
        score = opacities.float().clamp(min=0) * scale_metric.float().clamp(min=0)
        if args.danger_valid_only:
            candidate = depth_valid_full & keep_mask
        else:
            candidate = keep_mask
        if torch.any(candidate):
            candidate_idx = torch.nonzero(candidate, as_tuple=False).reshape(-1)
            k = max(1, int(math.ceil(float(args.danger_top_frac) * int(candidate_idx.numel()))))
            k = min(k, int(candidate_idx.numel()))
            _, top_local = torch.topk(score[candidate_idx], k=k, largest=True)
            danger_idx = candidate_idx[top_local]
            if args.danger_min_opacity > 0:
                danger_idx = danger_idx[opacities[danger_idx] >= float(args.danger_min_opacity)]
            danger_mask[danger_idx] = True
            apply_mode(opacities, keep_mask, danger_mask, args.danger_mode, args.danger_downweight)

    # Rule 4: optional global topk after filtering per packet.
    topk_mask = torch.ones((N,), device=device, dtype=torch.bool)
    if args.per_packet_topk > 0 and args.per_packet_topk < int(keep_mask.sum().item()):
        kept_idx = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
        _, top_local = torch.topk(opacities[kept_idx], k=int(args.per_packet_topk), largest=True)
        selected = kept_idx[top_local]
        topk_mask = torch.zeros((N,), device=device, dtype=torch.bool)
        topk_mask[selected] = True
        keep_mask &= topk_mask

    # Apply hard filters to all fields.
    for k in list(fields.keys()):
        fields[k] = fields[k][keep_mask].contiguous()

    source_ids = torch.full((int(fields["means"].shape[0]),), packet_ref.sorted_index, device=device, dtype=torch.long)

    stats = {
        "packet_sorted_index": int(packet_ref.sorted_index),
        "packet_name": packet_ref.path.name,
        "target_frame_index": int(frame_index),
        "num_input": int(N),
        "num_project_valid": int(valid.sum().item()),
        "num_depth_valid": int(depth_valid_full.sum().item()),
        "num_free_space": int(free_space_mask.sum().item()),
        "num_behind": int(behind_mask.sum().item()),
        "num_danger": int(danger_mask.sum().item()),
        "num_output": int(fields["means"].shape[0]),
        "hard_removed": int(N - fields["means"].shape[0]),
        "free_space_ratio": float(free_space_mask.sum().item() / max(N, 1)),
        "behind_ratio": float(behind_mask.sum().item() / max(N, 1)),
        "danger_ratio": float(danger_mask.sum().item() / max(N, 1)),
        "output_ratio": float(fields["means"].shape[0] / max(N, 1)),
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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


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
    base["depth_guided_filter_meta"] = meta

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_pt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-guided lightweight packet filter P1")
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--packet_ranges", type=str, required=True)
    parser.add_argument("--depth_dir", type=Path, default=None)
    parser.add_argument("--depth_pattern", type=str, default="{index:06d}.png")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--output_pt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)

    parser.add_argument("--margin_abs", type=float, default=0.10)
    parser.add_argument("--margin_rel", type=float, default=0.02)

    parser.add_argument("--free_space_mode", choices=["none", "downweight", "hard"], default="downweight")
    parser.add_argument("--free_space_downweight", type=float, default=0.05)
    parser.add_argument("--behind_mode", choices=["none", "downweight", "hard"], default="none")
    parser.add_argument("--behind_downweight", type=float, default=0.5)

    parser.add_argument("--opacity_cap", type=float, default=0.0, help="<=0 disables opacity cap")

    parser.add_argument("--danger_mode", choices=["none", "downweight", "hard"], default="downweight")
    parser.add_argument("--danger_top_frac", type=float, default=0.0, help="Fraction of high opacity*scale Gaussians to suppress per packet. 0 disables.")
    parser.add_argument("--danger_downweight", type=float, default=0.2)
    parser.add_argument("--danger_min_opacity", type=float, default=0.0)
    parser.add_argument("--danger_valid_only", action="store_true", help="Only select danger Gaussians among depth-valid projections.")
    parser.add_argument("--scale_input", choices=["auto", "raw", "log"], default="auto")
    parser.add_argument("--scale_metric", choices=["max", "volume", "mean"], default="max")

    parser.add_argument("--per_packet_topk", type=int, default=-1, help="Optional top-k opacity retention per packet after filtering.")
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
    gaussian_keys = available_gaussian_keys(packets, selected_indices)
    print(f"      found packets: {len(packets)}")
    print(f"      selected packets: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''} ({len(selected_indices)} total)")
    print(f"      gaussian keys: {gaussian_keys}")

    print("[2/4] Filtering packets")
    packet_fields: List[Dict[str, torch.Tensor]] = []
    source_ids_list: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []

    for order, idx in enumerate(selected_indices):
        fields, source_ids, stats = process_one_packet(
            packet_ref=packets[idx],
            gaussian_keys=gaussian_keys,
            depth_dir=args.depth_dir,
            depth_pattern=args.depth_pattern,
            depth_scale=args.depth_scale,
            device=device,
            args=args,
        )
        packet_fields.append(fields)
        source_ids_list.append(source_ids)
        rows.append(stats)
        print(
            f"      [{order+1:03d}/{len(selected_indices):03d}] "
            f"packet={idx:04d}, frame={stats['target_frame_index']:06d}, "
            f"in={stats['num_input']:,}, out={stats['num_output']:,}, "
            f"free={stats['num_free_space']:,}, behind={stats['num_behind']:,}, "
            f"danger={stats['num_danger']:,}"
        )

    print("[3/4] Saving fused filtered packet")
    total_in = int(sum(r["num_input"] for r in rows))
    total_out = int(sum(r["num_output"] for r in rows))
    total_free = int(sum(r["num_free_space"] for r in rows))
    total_behind = int(sum(r["num_behind"] for r in rows))
    total_danger = int(sum(r["num_danger"] for r in rows))

    meta = {
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "depth_dir": str(args.depth_dir) if args.depth_dir is not None else None,
        "depth_pattern": args.depth_pattern,
        "depth_scale": args.depth_scale,
        "margin_abs": args.margin_abs,
        "margin_rel": args.margin_rel,
        "free_space_mode": args.free_space_mode,
        "free_space_downweight": args.free_space_downweight,
        "behind_mode": args.behind_mode,
        "behind_downweight": args.behind_downweight,
        "opacity_cap": args.opacity_cap,
        "danger_mode": args.danger_mode,
        "danger_top_frac": args.danger_top_frac,
        "danger_downweight": args.danger_downweight,
        "danger_min_opacity": args.danger_min_opacity,
        "danger_valid_only": bool(args.danger_valid_only),
        "scale_input": args.scale_input,
        "scale_metric": args.scale_metric,
        "per_packet_topk": args.per_packet_topk,
        "extrinsic_type": args.extrinsic_type,
        "camera_z_sign": args.camera_z_sign,
        "intrinsics_normalized": args.intrinsics_normalized,
        "gaussian_keys": list(gaussian_keys),
        "num_packets": len(selected_indices),
        "num_input_gaussians": total_in,
        "num_output_gaussians": total_out,
        "num_free_space": total_free,
        "num_behind": total_behind,
        "num_danger": total_danger,
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
    print(f"      saved: {args.output_pt}")

    print("[4/4] Saving diagnostics")
    per_packet_csv = output_dir / "depth_guided_filter_per_packet.csv"
    write_csv(per_packet_csv, rows)

    summary_path = output_dir / "depth_guided_filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"selected_indices": selected_indices, **meta}, f, indent=2, ensure_ascii=False)

    print(f"      saved per-packet csv: {per_packet_csv}")
    print(f"      saved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
