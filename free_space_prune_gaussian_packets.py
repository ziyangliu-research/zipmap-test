#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 free-space consistency pruning/downweighting for fused feed-forward 3DGS packets.

This script is designed for the current ZipMap -> ReSplat packet workflow.

A Gaussian is considered a free-space violator in a validation view if its center
projects to pixel u and lies in front of the observed surface depth:

    z_g < D(u) - (margin_abs + margin_rel * D(u))

Typical usage
-------------
python free_space_prune_gaussian_packets_p0.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P001_0_80/gaussian_packets_api/final \
  --packet_ranges 0-79 \
  --val_packet_ranges 30-79 \
  --depth_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P001/depth_lcam_front \
  --depth_pattern "{index:06d}.png" \
  --output_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P001_0_80/free_space_pruned_0_79.pt \
  --mode hard \
  --margin_abs 0.10 \
  --margin_rel 0.02 \
  --min_observed 2 \
  --min_violations 2 \
  --violation_ratio 0.5 \
  --export_bad_ply \
  --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
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


@dataclass
class ValidationView:
    name: str
    sorted_index: int
    frame_index: int
    Twc_or_Tcw: torch.Tensor
    K_pixel: torch.Tensor
    depth: torch.Tensor
    near: Optional[float]
    far: Optional[float]


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


def first_camera_tensor(x: Any, device: torch.device) -> torch.Tensor:
    t = ensure_tensor(x, device=device)
    while t.ndim > 2:
        t = t[0]
    return t


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


def first_int(x: Any, default: int = -1) -> int:
    v = first_scalar(x, None)
    return default if v is None else int(round(v))


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


def available_gaussian_keys(packets: Sequence[PacketRef], selected_indices: Sequence[int]) -> List[str]:
    keys = [k for k in ALL_GAUSSIAN_KEYS if all(k in packets[i].data for i in selected_indices)]
    missing = [k for k in PRIMARY_GAUSSIAN_KEYS if k not in keys]
    if missing:
        raise KeyError(f"Missing primary Gaussian keys: {missing}")
    return keys


def concat_gaussian_fields(
    packets: Sequence[PacketRef],
    selected_indices: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    keys = available_gaussian_keys(packets, selected_indices)
    chunks: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
    source_ids: List[torch.Tensor] = []

    for idx in selected_indices:
        p = packets[idx].data
        n_this: Optional[int] = None
        for k in keys:
            t = normalize_gaussian_tensor(p[k], k, device)
            if n_this is None:
                n_this = int(t.shape[0])
            elif int(t.shape[0]) != n_this:
                raise RuntimeError(f"Packet {packets[idx].path.name}: {k} length {t.shape[0]} != {n_this}")
            chunks[k].append(t)
        assert n_this is not None
        source_ids.append(torch.full((n_this,), int(idx), device=device, dtype=torch.long))

    fields = {k: torch.cat(v, dim=0).contiguous() for k, v in chunks.items()}
    src = torch.cat(source_ids, dim=0).contiguous()
    return fields, src, keys


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


def denormalize_intrinsics(K: torch.Tensor, H: int, W: int, mode: str = "auto") -> torch.Tensor:
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


def load_packet_depth_if_available(packet: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    for k in ("target_depth", "depth", "pred_depth", "zipmap_depth", "depth_map"):
        if k in packet:
            d = ensure_tensor(packet[k], device=device)
            while d.ndim > 2:
                d = d[0]
            return d.float()
    return None


def load_validation_views(
    packets: Sequence[PacketRef],
    val_indices: Sequence[int],
    depth_dir: Optional[Path],
    depth_pattern: str,
    depth_scale: float,
    device: torch.device,
    intrinsics_normalized: str,
) -> List[ValidationView]:
    views: List[ValidationView] = []
    for idx in val_indices:
        packet = packets[idx].data
        if "target_extrinsics" not in packet or "target_intrinsics" not in packet:
            raise KeyError(f"Validation packet {packets[idx].path.name} lacks target_extrinsics/target_intrinsics")
        frame_index = first_int(packet.get("target_index", idx), default=idx)

        depth = load_packet_depth_if_available(packet, device)
        if depth is None:
            if depth_dir is None:
                raise ValueError("No depth inside packet and --depth_dir is missing.")
            depth = load_depth(depth_dir / depth_pattern.format(index=int(frame_index)), depth_scale, device)

        H, W = infer_image_shape(packet, depth)
        K = denormalize_intrinsics(first_camera_tensor(packet["target_intrinsics"], device), H, W, intrinsics_normalized)
        E = first_camera_tensor(packet["target_extrinsics"], device)
        near = first_scalar(packet.get("target_near", None), None)
        far = first_scalar(packet.get("target_far", None), None)
        views.append(ValidationView(
            name=f"packet{idx:04d}_frame{frame_index:06d}",
            sorted_index=int(idx),
            frame_index=int(frame_index),
            Twc_or_Tcw=E,
            K_pixel=K,
            depth=depth,
            near=near,
            far=far,
        ))
    return views


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


def evaluate_violations(
    means: torch.Tensor,
    views: Sequence[ValidationView],
    extrinsic_type: str,
    camera_z_sign: str,
    margin_abs: float,
    margin_rel: float,
    chunk_size: int,
    max_val_views: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    device = means.device
    N = int(means.shape[0])
    violation_count = torch.zeros((N,), device=device, dtype=torch.int16)
    observed_count = torch.zeros((N,), device=device, dtype=torch.int16)
    per_view: List[Dict[str, Any]] = []
    eval_views = list(views)[:max_val_views] if max_val_views > 0 else list(views)

    for view_i, view in enumerate(eval_views):
        H, W = int(view.depth.shape[-2]), int(view.depth.shape[-1])
        obs_total = 0
        viol_total = 0
        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)
            u, v, z = project_points(means[start:end], view.Twc_or_Tcw, view.K_pixel, extrinsic_type, camera_z_sign)
            ui = torch.round(u).long()
            vi = torch.round(v).long()
            valid = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if view.near is not None:
                valid &= z >= float(view.near)
            if view.far is not None:
                valid &= z <= float(view.far)
            if not torch.any(valid):
                continue
            local = torch.nonzero(valid, as_tuple=False).reshape(-1)
            d = view.depth[vi[local], ui[local]]
            depth_valid = torch.isfinite(d) & (d > 0)
            if not torch.any(depth_valid):
                continue
            local = local[depth_valid]
            d = d[depth_valid]
            z_valid = z[local]
            margin = float(margin_abs) + float(margin_rel) * d
            viol = z_valid < (d - margin)
            global_idx = start + local
            observed_count[global_idx] += 1
            violation_count[global_idx[viol]] += 1
            obs_total += int(local.numel())
            viol_total += int(viol.sum().item())
        row = {
            "view_order": view_i,
            "validation_packet_sorted_index": view.sorted_index,
            "frame_index": view.frame_index,
            "name": view.name,
            "observed_gaussian_count": obs_total,
            "violation_count": viol_total,
            "violation_ratio_among_observed": float(viol_total / max(obs_total, 1)),
        }
        per_view.append(row)
        print(f"[view {view_i+1:03d}/{len(eval_views):03d}] {view.name}: "
              f"observed={obs_total:,}, violations={viol_total:,}, "
              f"ratio={viol_total / max(obs_total, 1):.4f}")
    return violation_count, observed_count, per_view


def make_bad_mask(
    violation_count: torch.Tensor,
    observed_count: torch.Tensor,
    min_observed: int,
    min_violations: int,
    violation_ratio: float,
) -> torch.Tensor:
    ratio = violation_count.float() / torch.clamp(observed_count.float(), min=1.0)
    return (
        (observed_count >= int(min_observed))
        & (violation_count >= int(min_violations))
        & (ratio >= float(violation_ratio))
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_point_ply(path: Path, xyz: torch.Tensor, source_ids: Optional[torch.Tensor] = None, max_points: int = 300_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = xyz.detach().cpu()
    src = source_ids.detach().cpu() if source_ids is not None else None
    if pts.shape[0] > max_points:
        perm = torch.randperm(pts.shape[0])[:max_points]
        pts = pts[perm]
        src = src[perm] if src is not None else None
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(pts.shape[0]):
            p = pts[i]
            if src is None:
                r, g, b = 255, 0, 0
            else:
                sid = int(src[i].item())
                r = (53 * sid + 255) % 256
                g = (97 * sid + 64) % 256
                b = (193 * sid + 32) % 256
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])} {r} {g} {b}\n")


def make_output_packet(
    packets: Sequence[PacketRef],
    map_indices: Sequence[int],
    fields: Dict[str, torch.Tensor],
    gaussian_keys: Sequence[str],
    source_ids: torch.Tensor,
    bad_mask: torch.Tensor,
    violation_count: torch.Tensor,
    observed_count: torch.Tensor,
    mode: str,
    downweight: float,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    base = dict(packets[map_indices[0]].data)
    keep_mask = ~bad_mask

    if mode == "hard":
        for k in gaussian_keys:
            base[k] = fields[k][keep_mask].detach().cpu()
        out_source = source_ids[keep_mask].detach().cpu()
    elif mode == "downweight":
        for k in gaussian_keys:
            val = fields[k].clone()
            if k == "opacities":
                val[bad_mask] = val[bad_mask] * float(downweight)
            base[k] = val.detach().cpu()
        out_source = source_ids.detach().cpu()
    elif mode == "mark_only":
        for k in gaussian_keys:
            base[k] = fields[k].detach().cpu()
        out_source = source_ids.detach().cpu()
    else:
        raise ValueError(mode)

    for k in ALL_GAUSSIAN_KEYS:
        if k not in gaussian_keys and k in base:
            del base[k]

    base["source_packet_sorted_index"] = out_source
    base["free_space_bad_mask"] = bad_mask.detach().cpu()
    base["free_space_violation_count"] = violation_count.detach().cpu()
    base["free_space_observed_count"] = observed_count.detach().cpu()
    base["fusion_source_packet_names"] = [packets[i].path.name for i in map_indices]
    base["free_space_pruning_meta"] = meta
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 free-space pruning/downweighting for fused 3DGS packets")
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--packet_ranges", type=str, required=True)
    parser.add_argument("--val_packet_ranges", type=str, required=True)
    parser.add_argument("--depth_dir", type=Path, default=None)
    parser.add_argument("--depth_pattern", type=str, default="{index:06d}.png")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--output_pt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["hard", "downweight", "mark_only"], default="hard")
    parser.add_argument("--downweight", type=float, default=0.1)
    parser.add_argument("--margin_abs", type=float, default=0.10)
    parser.add_argument("--margin_rel", type=float, default=0.02)
    parser.add_argument("--min_observed", type=int, default=2)
    parser.add_argument("--min_violations", type=int, default=2)
    parser.add_argument("--violation_ratio", type=float, default=0.5)
    parser.add_argument("--max_val_views", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    parser.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive")
    parser.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--export_bad_ply", action="store_true")
    parser.add_argument("--export_keep_ply", action="store_true")
    parser.add_argument("--max_ply_points", type=int, default=300_000)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading packets: {args.packet_dir}")
    packets = load_packets(args.packet_dir)
    map_indices = parse_ranges(args.packet_ranges, len(packets))
    val_indices = parse_ranges(args.val_packet_ranges, len(packets))
    print(f"      found packets: {len(packets)}")
    print(f"      map packets: {map_indices[:10]}{'...' if len(map_indices) > 10 else ''} ({len(map_indices)} total)")
    print(f"      val packets: {val_indices[:10]}{'...' if len(val_indices) > 10 else ''} ({len(val_indices)} total)")

    print("[2/5] Concatenating Gaussian fields")
    fields, source_ids, gaussian_keys = concat_gaussian_fields(packets, map_indices, device)
    N = int(fields["means"].shape[0])
    print(f"      available Gaussian keys: {gaussian_keys}")
    print(f"      total Gaussians: {N:,}")

    print("[3/5] Loading validation views/depth")
    views = load_validation_views(
        packets, val_indices, args.depth_dir, args.depth_pattern, args.depth_scale,
        device, args.intrinsics_normalized
    )
    print(f"      validation views loaded: {len(views)}")

    print("[4/5] Evaluating free-space violations")
    violation_count, observed_count, per_view_rows = evaluate_violations(
        fields["means"], views, args.extrinsic_type, args.camera_z_sign,
        args.margin_abs, args.margin_rel, args.chunk_size, args.max_val_views
    )

    bad_mask = make_bad_mask(
        violation_count, observed_count,
        args.min_observed, args.min_violations, args.violation_ratio
    )
    keep_mask = ~bad_mask
    num_bad = int(bad_mask.sum().item())
    num_keep = int(keep_mask.sum().item())
    prune_ratio = num_bad / max(N, 1)
    print(f"      bad/free-space violators: {num_bad:,} / {N:,} ({prune_ratio:.2%})")

    print("[5/5] Saving output packet and diagnostics")
    per_source_rows: List[Dict[str, Any]] = []
    for sid in torch.unique(source_ids).detach().cpu().tolist():
        sid_int = int(sid)
        sm = source_ids == sid_int
        sn = int(sm.sum().item())
        sb = int((bad_mask & sm).sum().item())
        per_source_rows.append({
            "source_packet_sorted_index": sid_int,
            "source_packet_name": packets[sid_int].path.name,
            "num_gaussians": sn,
            "num_bad": sb,
            "bad_ratio": float(sb / max(sn, 1)),
            "mean_observed_count": float(observed_count[sm].float().mean().item()) if sn > 0 else 0.0,
            "mean_violation_count": float(violation_count[sm].float().mean().item()) if sn > 0 else 0.0,
        })

    write_csv(out_dir / "free_space_pruning_per_view.csv", per_view_rows)
    write_csv(out_dir / "free_space_pruning_per_source.csv", per_source_rows)

    meta = {
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "val_packet_ranges": args.val_packet_ranges,
        "depth_dir": str(args.depth_dir) if args.depth_dir is not None else None,
        "depth_pattern": args.depth_pattern,
        "depth_scale": args.depth_scale,
        "mode": args.mode,
        "downweight": args.downweight,
        "margin_abs": args.margin_abs,
        "margin_rel": args.margin_rel,
        "min_observed": args.min_observed,
        "min_violations": args.min_violations,
        "violation_ratio": args.violation_ratio,
        "extrinsic_type": args.extrinsic_type,
        "camera_z_sign": args.camera_z_sign,
        "intrinsics_normalized": args.intrinsics_normalized,
        "gaussian_keys": list(gaussian_keys),
        "num_input_gaussians": N,
        "num_bad": num_bad,
        "num_keep": num_keep,
        "prune_ratio": prune_ratio,
        "num_output_gaussians": num_keep if args.mode == "hard" else N,
    }

    output_packet = make_output_packet(
        packets, map_indices, fields, gaussian_keys, source_ids,
        bad_mask, violation_count, observed_count,
        args.mode, args.downweight, meta
    )
    torch.save(output_packet, args.output_pt)
    print(f"      saved packet: {args.output_pt}")

    summary_path = out_dir / "free_space_pruning_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"num_packets_found": len(packets), "map_indices": map_indices, "val_indices": val_indices, **meta}, f, indent=2, ensure_ascii=False)
    print(f"      saved summary: {summary_path}")

    if args.export_bad_ply and num_bad > 0:
        bad_ply = out_dir / "bad_free_space_gaussians.ply"
        save_point_ply(bad_ply, fields["means"][bad_mask], source_ids[bad_mask], args.max_ply_points)
        print(f"      saved bad PLY: {bad_ply}")
    if args.export_keep_ply and num_keep > 0:
        keep_ply = out_dir / "kept_gaussians.ply"
        save_point_ply(keep_ply, fields["means"][keep_mask], source_ids[keep_mask], args.max_ply_points)
        print(f"      saved keep PLY: {keep_ply}")

    print("Done.")


if __name__ == "__main__":
    main()
