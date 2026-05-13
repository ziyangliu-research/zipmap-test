#!/usr/bin/env python3
"""
Free-space consistency pruning for fused feed-forward 3DGS packets.

Purpose
-------
Given a set of Gaussian packet .pt files and a set of validation views with depth,
remove or down-weight Gaussians whose centers lie in the observed free space of
validation views.

A Gaussian g is marked as a free-space violator in view v if:
    z_g(v) < D_v(u_g) - margin
where u_g is the projected pixel of g in view v, z_g(v) is its camera-depth, and
D_v is the observed depth map. This catches false-positive Gaussians that are in
front of the observed surface and therefore would act as occluders in later views.

This is a P0 diagnostic / baseline script. It does not do gradient optimization.

Typical usage
-------------
# Use packets 0-80 as the map, and validation views from packet targets 30-80.
# Depth maps are loaded by target index from depth_dir.
python free_space_prune_gaussian_packets.py \
  --packet_dir outputs/test/P000/gaussian_packets \
  --packet_ranges 0-80 \
  --val_packet_ranges 30-80 \
  --depth_dir /path/to/depth_lcam_front \
  --depth_pattern "{index:06d}.png" \
  --depth_scale 1000.0 \
  --output_pt outputs/pruned/free_space_pruned_0_80.pt \
  --mode hard \
  --margin_abs 0.10 \
  --margin_rel 0.02 \
  --min_violations 2 \
  --violation_ratio 0.5 \
  --device cuda:0

# If you only want to reduce opacity instead of deleting Gaussians:
python free_space_prune_gaussian_packets.py ... --mode downweight --downweight 0.1

Assumptions
-----------
- Packet files contain at least: means, covariances, harmonics, opacities.
- Validation packets contain target_extrinsics and target_intrinsics.
- Default extrinsics are Twc camera-to-world matrices, as in the current project.
- Intrinsics are automatically treated as normalized if fx/fy/cx/cy are small.

Depth formats supported
-----------------------
- .npy: metric depth by default
- .pt/.pth: torch tensor
- .npy: metric depth by default
- .pt/.pth: torch tensor
- .png/.tif/.tiff: integer depth divided by --depth_scale
- TartanAir V2 depth PNG: 4-channel uint8 RGBA storing one little-endian float32 per pixel, decoded automatically
- .exr: supported only if imageio/OpenCV installation can read it
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


GAUSSIAN_KEYS = ["means", "covariances", "harmonics", "opacities"]
CAMERA_KEYS = ["target_extrinsics", "target_intrinsics", "target_near", "target_far", "target_index"]


@dataclass
class PacketRef:
    sorted_index: int
    path: Path
    data: Dict


@dataclass
class ValidationView:
    name: str
    frame_index: int
    Twc_or_Tcw: torch.Tensor  # [4, 4]
    K: torch.Tensor           # [3, 3]
    depth: torch.Tensor       # [H, W]
    near: Optional[float]
    far: Optional[float]


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def parse_ranges(spec: Optional[str], n: int) -> List[int]:
    """Parse strings like '0-10,15,20-25'. Inclusive ranges."""
    if spec is None or spec.strip() == "":
        return list(range(n))
    indices: List[int] = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a_str, b_str = token.split('-', 1)
            a, b = int(a_str), int(b_str)
            if b < a:
                raise ValueError(f"Invalid descending range: {token}")
            indices.extend(range(a, b + 1))
        else:
            indices.append(int(token))
    indices = sorted(set(indices))
    bad = [i for i in indices if i < 0 or i >= n]
    if bad:
        raise IndexError(f"Packet indices out of range 0..{n-1}: {bad[:10]}")
    return indices


def as_tensor(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def squeeze_first_view(x: torch.Tensor) -> torch.Tensor:
    """Select the first view if tensor has extra leading view/batch dimensions."""
    # Typical shapes:
    # extrinsics: [4,4], [1,4,4], [V,4,4], [B,V,4,4]
    # intrinsics: [3,3], [1,3,3], [V,3,3], [B,V,3,3]
    while x.ndim > 2:
        x = x[0]
    return x


def extract_first_scalar(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return default
        return float(x.reshape(-1)[0].detach().cpu().item())
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[0])
    return float(x)


def extract_first_int(x, default: int = -1) -> int:
    val = extract_first_scalar(x, None)
    if val is None:
        return default
    return int(round(val))


def infer_image_shape(packet: Dict, depth: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    if depth is not None:
        return int(depth.shape[-2]), int(depth.shape[-1])
    if "image_shape" in packet:
        shape = packet["image_shape"]
        if isinstance(shape, torch.Tensor):
            shape = shape.detach().cpu().reshape(-1).tolist()
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    if "target_image" in packet:
        img = packet["target_image"]
        if isinstance(img, torch.Tensor):
            # Common: [V, C, H, W], [C, H, W], [H, W, C]
            if img.ndim >= 4:
                return int(img.shape[-2]), int(img.shape[-1])
            if img.ndim == 3:
                if img.shape[0] in (1, 3):
                    return int(img.shape[1]), int(img.shape[2])
                return int(img.shape[0]), int(img.shape[1])
    raise ValueError("Cannot infer image shape from packet/depth. Provide depth maps or image_shape in packet.")


def maybe_denormalize_intrinsics(K: torch.Tensor, H: int, W: int, force_normalized: str = "auto") -> torch.Tensor:
    """Convert normalized intrinsics to pixel intrinsics if needed.

    MVSplat-style saved intrinsics are often normalized:
      fx_norm = fx / W, fy_norm = fy / H, cx_norm = cx / W, cy_norm = cy / H.
    If force_normalized == 'auto', treat K as normalized when fx and fy are <= 10.
    """
    K = K.clone()
    if force_normalized not in {"auto", "true", "false"}:
        raise ValueError("force_normalized must be auto/true/false")
    if force_normalized == "true":
        normalized = True
    elif force_normalized == "false":
        normalized = False
    else:
        normalized = bool(abs(float(K[0, 0])) <= 10.0 and abs(float(K[1, 1])) <= 10.0)
    if normalized:
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= W
        K[1, 2] *= H
    return K


def load_depth(path: Path, depth_scale: float, device: torch.device) -> torch.Tensor:
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"Depth map not found: {path}")
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".pt", ".pth"}:
        t = torch.load(path, map_location="cpu")
        if isinstance(t, dict):
            # Try common keys.
            for k in ["depth", "pred_depth", "target_depth", "depth_map"]:
                if k in t:
                    t = t[k]
                    break
        arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    elif suffix in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        arr_raw = np.array(Image.open(path))

        # TartanAir V2 stores depth as H x W x 4 uint8 PNG. The four channels are a
        # lossless byte representation of one little-endian float32 depth value.
        # Official decoding is effectively: cv2.imread(..., IMREAD_UNCHANGED).view("<f4").
        if arr_raw.ndim == 3 and arr_raw.shape[-1] == 4 and arr_raw.dtype == np.uint8:
            arr = np.ascontiguousarray(arr_raw).view("<f4").reshape(arr_raw.shape[0], arr_raw.shape[1])
        else:
            # Standard integer depth PNG/TIFF path, e.g. uint16 millimeters.
            arr = arr_raw.astype(np.float32) / float(depth_scale)
    elif suffix == ".exr":
        try:
            import imageio.v3 as iio  # type: ignore
            arr = iio.imread(path).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to read EXR depth {path}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported depth format: {path}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def depth_path_from_index(depth_dir: Path, pattern: str, index: int) -> Path:
    # Supports both format names: {index} and {index:06d}
    return depth_dir / pattern.format(index=index)


def load_packets(packet_dir: Path, device: torch.device) -> List[PacketRef]:
    paths = sorted(packet_dir.glob("*.pt"), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .pt packets found in {packet_dir}")
    refs: List[PacketRef] = []
    for i, p in enumerate(paths):
        data = torch.load(p, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"Packet is not a dict: {p}")
        missing = [k for k in GAUSSIAN_KEYS if k not in data]
        if missing:
            raise KeyError(f"Packet {p.name} is missing Gaussian keys: {missing}")
        refs.append(PacketRef(i, p, data))
    return refs


def concat_gaussians(packets: Sequence[PacketRef], selected_indices: Sequence[int], device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    source_ids: List[torch.Tensor] = []
    for k in GAUSSIAN_KEYS:
        chunks = []
        for idx in selected_indices:
            t = as_tensor(packets[idx].data[k], device=device)
            # Typical opacities may be [N] or [N,1]. Keep original dimensions.
            chunks.append(t)
            if k == "means":
                source_ids.append(torch.full((t.shape[0],), idx, device=device, dtype=torch.long))
        out[k] = torch.cat(chunks, dim=0)
    src = torch.cat(source_ids, dim=0)
    n = out["means"].shape[0]
    for k, v in out.items():
        if v.shape[0] != n:
            raise ValueError(f"Concatenated key {k} has inconsistent first dimension {v.shape[0]} != {n}")
    return out, src


def load_validation_views(
    packets: Sequence[PacketRef],
    val_indices: Sequence[int],
    depth_dir: Optional[Path],
    depth_pattern: str,
    depth_scale: float,
    device: torch.device,
    intrinsics_normalized: str,
    extrinsic_type: str,
) -> List[ValidationView]:
    views: List[ValidationView] = []
    for idx in val_indices:
        data = packets[idx].data
        if "target_extrinsics" not in data or "target_intrinsics" not in data:
            raise KeyError(f"Validation packet {packets[idx].path.name} lacks target_extrinsics/target_intrinsics")
        frame_idx = extract_first_int(data.get("target_index", idx), default=idx)

        # Prefer depth stored inside packet if available.
        depth = None
        for dk in ["target_depth", "depth", "pred_depth", "zipmap_depth"]:
            if dk in data:
                depth = as_tensor(data[dk], device=device)
                while depth.ndim > 2:
                    depth = depth[0]
                break
        if depth is None:
            if depth_dir is None:
                raise ValueError(
                    "No depth found in packets and --depth_dir was not provided. "
                    "For P0 validation, provide GT/stereo/ZipMap depth maps."
                )
            dpath = depth_path_from_index(depth_dir, depth_pattern, frame_idx)
            depth = load_depth(dpath, depth_scale=depth_scale, device=device)

        H, W = infer_image_shape(data, depth)
        K = squeeze_first_view(as_tensor(data["target_intrinsics"], device=device))
        K = maybe_denormalize_intrinsics(K, H=H, W=W, force_normalized=intrinsics_normalized)
        E = squeeze_first_view(as_tensor(data["target_extrinsics"], device=device))
        near = extract_first_scalar(data.get("target_near", None), None)
        far = extract_first_scalar(data.get("target_far", None), None)
        views.append(
            ValidationView(
                name=f"packet{idx:04d}_frame{frame_idx}",
                frame_index=frame_idx,
                Twc_or_Tcw=E,
                K=K,
                depth=depth,
                near=near,
                far=far,
            )
        )
    return views


def project_points(
    means_world: torch.Tensor,
    E: torch.Tensor,
    K: torch.Tensor,
    extrinsic_type: str,
    camera_z_sign: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space means into one camera.

    Returns u, v, z tensors of shape [N].
    Default assumes E is Twc and OpenCV-style positive camera z.
    """
    if extrinsic_type == "Twc":
        Tcw = torch.linalg.inv(E)
    elif extrinsic_type == "Tcw":
        Tcw = E
    else:
        raise ValueError("extrinsic_type must be Twc or Tcw")

    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    xyz_cam = means_world @ R.T + t[None, :]
    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    z = xyz_cam[:, 2]
    if camera_z_sign == "negative":
        # Some graphics conventions use negative z forward. Convert to positive forward depth.
        z_proj = -z
        x_proj = x
        y_proj = y
    elif camera_z_sign == "positive":
        z_proj = z
        x_proj = x
        y_proj = y
    else:
        raise ValueError("camera_z_sign must be positive or negative")

    eps = 1e-8
    u = K[0, 0] * (x_proj / (z_proj + eps)) + K[0, 2]
    v = K[1, 1] * (y_proj / (z_proj + eps)) + K[1, 2]
    return u, v, z_proj


def evaluate_free_space_violations(
    means: torch.Tensor,
    views: Sequence[ValidationView],
    extrinsic_type: str,
    camera_z_sign: str,
    margin_abs: float,
    margin_rel: float,
    max_views: Optional[int] = None,
    chunk_size: int = 1_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """Return violation_count and observed_count per Gaussian."""
    device = means.device
    N = means.shape[0]
    violation_count = torch.zeros((N,), device=device, dtype=torch.int16)
    observed_count = torch.zeros((N,), device=device, dtype=torch.int16)
    per_view_stats: List[Dict[str, float]] = []
    eval_views = list(views)
    if max_views is not None and max_views > 0:
        eval_views = eval_views[:max_views]

    for vi, view in enumerate(eval_views):
        depth = view.depth
        H, W = int(depth.shape[-2]), int(depth.shape[-1])
        view_observed = 0
        view_violations = 0
        # Process in chunks to avoid allocating too many intermediates.
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            m = means[start:end]
            u, v, z = project_points(m, view.Twc_or_Tcw, view.K, extrinsic_type, camera_z_sign)
            ui = torch.round(u).long()
            vi_pix = torch.round(v).long()
            valid = (z > 0) & (ui >= 0) & (ui < W) & (vi_pix >= 0) & (vi_pix < H)
            if view.near is not None:
                valid &= z >= float(view.near)
            if view.far is not None:
                valid &= z <= float(view.far)
            if not torch.any(valid):
                continue
            idx_local = torch.nonzero(valid, as_tuple=False).reshape(-1)
            d = depth[vi_pix[idx_local], ui[idx_local]]
            depth_valid = torch.isfinite(d) & (d > 0)
            if not torch.any(depth_valid):
                continue
            idx_local = idx_local[depth_valid]
            d = d[depth_valid]
            z_valid = z[idx_local]
            margin = float(margin_abs) + float(margin_rel) * d
            viol = z_valid < (d - margin)
            global_idx = start + idx_local
            observed_count[global_idx] += 1
            violation_count[global_idx[viol]] += 1
            view_observed += int(idx_local.numel())
            view_violations += int(viol.sum().item())
        per_view_stats.append(
            {
                "view_order": vi,
                "name": view.name,
                "frame_index": int(view.frame_index),
                "observed_gaussian_count": int(view_observed),
                "violation_count": int(view_violations),
                "violation_ratio_among_observed": float(view_violations / max(view_observed, 1)),
            }
        )
        print(
            f"[View {vi+1:03d}/{len(eval_views):03d}] {view.name}: "
            f"observed={view_observed}, violations={view_violations}, "
            f"ratio={view_violations / max(view_observed, 1):.4f}"
        )
    return violation_count, observed_count, per_view_stats


def make_prune_mask(
    violation_count: torch.Tensor,
    observed_count: torch.Tensor,
    min_violations: int,
    violation_ratio: float,
    min_observed: int,
) -> torch.Tensor:
    ratio = violation_count.float() / torch.clamp(observed_count.float(), min=1.0)
    bad = (
        (observed_count >= int(min_observed))
        & (violation_count >= int(min_violations))
        & (ratio >= float(violation_ratio))
    )
    return bad


def save_ply_points(path: Path, points: torch.Tensor, source_ids: Optional[torch.Tensor] = None, max_points: int = 300_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.detach().cpu()
    if pts.shape[0] > max_points:
        perm = torch.randperm(pts.shape[0])[:max_points]
        pts = pts[perm]
        src = source_ids.detach().cpu()[perm] if source_ids is not None else None
    else:
        src = source_ids.detach().cpu() if source_ids is not None else None
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            if src is None:
                r, g, b = 255, 0, 0
            else:
                # Deterministic pseudo-color by source packet id.
                sid = int(src[i].item())
                r = (53 * sid + 255) % 256
                g = (97 * sid + 64) % 256
                b = (193 * sid + 32) % 256
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])} {r} {g} {b}\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Free-space pruning for fused 3DGS packets")
    parser.add_argument("--packet_dir", type=Path, required=True, help="Directory containing Gaussian packet .pt files")
    parser.add_argument("--packet_ranges", type=str, required=True, help="Inclusive map packet indices, e.g. '0-80' or '0-20,30-80'")
    parser.add_argument("--val_packet_ranges", type=str, required=True, help="Inclusive validation packet indices, e.g. '30-80'")
    parser.add_argument("--depth_dir", type=Path, default=None, help="Directory containing validation depth maps. If omitted, script tries packet depth keys.")
    parser.add_argument("--depth_pattern", type=str, default="{index:06d}.png", help="Depth filename pattern using target frame index")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="For uint PNG/TIFF depths, metric_depth = value / depth_scale")
    parser.add_argument("--output_pt", type=Path, required=True, help="Output pruned/downweighted packet .pt")
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional diagnostics dir. Default: output_pt parent")
    parser.add_argument("--mode", choices=["hard", "downweight", "mark_only"], default="hard", help="hard=remove bad Gaussians; downweight=scale opacity; mark_only=save mask only")
    parser.add_argument("--downweight", type=float, default=0.1, help="Opacity multiplier for bad Gaussians in downweight mode")
    parser.add_argument("--margin_abs", type=float, default=0.10, help="Absolute depth margin in meters")
    parser.add_argument("--margin_rel", type=float, default=0.02, help="Relative depth margin multiplied by observed depth")
    parser.add_argument("--min_violations", type=int, default=2, help="Minimum number of free-space violations to mark bad")
    parser.add_argument("--violation_ratio", type=float, default=0.5, help="Minimum violation_count / observed_count to mark bad")
    parser.add_argument("--min_observed", type=int, default=1, help="Minimum number of validation views that observe the Gaussian")
    parser.add_argument("--max_val_views", type=int, default=0, help="If >0, only use first N validation views")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="Projection chunk size")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc", help="Convention of target_extrinsics saved in packets")
    parser.add_argument("--camera_z_sign", choices=["positive", "negative"], default="positive", help="Camera forward z convention")
    parser.add_argument("--intrinsics_normalized", choices=["auto", "true", "false"], default="auto", help="Whether saved intrinsics are normalized by image size")
    parser.add_argument("--export_bad_ply", action="store_true", help="Export bad Gaussian centers as PLY")
    parser.add_argument("--export_keep_ply", action="store_true", help="Export kept Gaussian centers as PLY")
    parser.add_argument("--max_ply_points", type=int, default=300_000)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.output_dir if args.output_dir is not None else args.output_pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading packets from: {args.packet_dir}")
    packets = load_packets(args.packet_dir, device=device)
    map_indices = parse_ranges(args.packet_ranges, len(packets))
    val_indices = parse_ranges(args.val_packet_ranges, len(packets))
    print(f"Found {len(packets)} packets")
    print(f"Map packet indices: {map_indices[:10]}{'...' if len(map_indices) > 10 else ''} ({len(map_indices)} total)")
    print(f"Validation packet indices: {val_indices[:10]}{'...' if len(val_indices) > 10 else ''} ({len(val_indices)} total)")

    gaussians, source_ids = concat_gaussians(packets, map_indices, device=device)
    N = int(gaussians["means"].shape[0])
    print(f"Concatenated Gaussians: {N:,}")

    views = load_validation_views(
        packets=packets,
        val_indices=val_indices,
        depth_dir=args.depth_dir,
        depth_pattern=args.depth_pattern,
        depth_scale=args.depth_scale,
        device=device,
        intrinsics_normalized=args.intrinsics_normalized,
        extrinsic_type=args.extrinsic_type,
    )
    print(f"Loaded validation views: {len(views)}")

    violation_count, observed_count, per_view_stats = evaluate_free_space_violations(
        means=gaussians["means"],
        views=views,
        extrinsic_type=args.extrinsic_type,
        camera_z_sign=args.camera_z_sign,
        margin_abs=args.margin_abs,
        margin_rel=args.margin_rel,
        max_views=args.max_val_views if args.max_val_views > 0 else None,
        chunk_size=args.chunk_size,
    )

    bad_mask = make_prune_mask(
        violation_count=violation_count,
        observed_count=observed_count,
        min_violations=args.min_violations,
        violation_ratio=args.violation_ratio,
        min_observed=args.min_observed,
    )
    keep_mask = ~bad_mask
    num_bad = int(bad_mask.sum().item())
    num_keep = int(keep_mask.sum().item())
    prune_ratio = num_bad / max(N, 1)
    print(f"Bad/free-space violator Gaussians: {num_bad:,} / {N:,} ({prune_ratio:.2%})")

    # Per-source packet diagnostics.
    per_source_rows: List[Dict] = []
    unique_sources = torch.unique(source_ids).detach().cpu().tolist()
    for sid in unique_sources:
        sm = source_ids == int(sid)
        sn = int(sm.sum().item())
        sb = int((bad_mask & sm).sum().item())
        so = float(observed_count[sm].float().mean().item()) if sn > 0 else 0.0
        sv = float(violation_count[sm].float().mean().item()) if sn > 0 else 0.0
        per_source_rows.append(
            {
                "source_packet_sorted_index": int(sid),
                "source_packet_name": packets[int(sid)].path.name,
                "num_gaussians": sn,
                "num_bad": sb,
                "bad_ratio": sb / max(sn, 1),
                "mean_observed_count": so,
                "mean_violation_count": sv,
            }
        )
    write_csv(out_dir / "free_space_pruning_per_source.csv", per_source_rows)
    write_csv(out_dir / "free_space_pruning_per_view.csv", per_view_stats)

    # Save output packet.
    base = dict(packets[map_indices[0]].data)
    # Store merged/pruned Gaussian tensors on CPU.
    if args.mode == "hard":
        for k in GAUSSIAN_KEYS:
            base[k] = gaussians[k][keep_mask].detach().cpu()
    elif args.mode == "downweight":
        for k in GAUSSIAN_KEYS:
            val = gaussians[k].clone()
            if k == "opacities":
                val[bad_mask] = val[bad_mask] * float(args.downweight)
            base[k] = val.detach().cpu()
    elif args.mode == "mark_only":
        for k in GAUSSIAN_KEYS:
            base[k] = gaussians[k].detach().cpu()
    else:
        raise ValueError(args.mode)

    base["free_space_bad_mask"] = bad_mask.detach().cpu()
    base["free_space_violation_count"] = violation_count.detach().cpu()
    base["free_space_observed_count"] = observed_count.detach().cpu()
    base["source_packet_sorted_index"] = source_ids.detach().cpu()
    base["fusion_source_packet_names"] = [packets[i].path.name for i in map_indices]
    base["free_space_pruning_meta"] = {
        "packet_dir": str(args.packet_dir),
        "packet_ranges": args.packet_ranges,
        "val_packet_ranges": args.val_packet_ranges,
        "depth_dir": str(args.depth_dir) if args.depth_dir is not None else None,
        "depth_pattern": args.depth_pattern,
        "depth_scale": args.depth_scale,
        "mode": args.mode,
        "margin_abs": args.margin_abs,
        "margin_rel": args.margin_rel,
        "min_violations": args.min_violations,
        "violation_ratio": args.violation_ratio,
        "min_observed": args.min_observed,
        "extrinsic_type": args.extrinsic_type,
        "camera_z_sign": args.camera_z_sign,
        "intrinsics_normalized": args.intrinsics_normalized,
        "num_input_gaussians": N,
        "num_bad": num_bad,
        "num_output_gaussians": int(base["means"].shape[0]),
        "prune_ratio": prune_ratio,
    }
    torch.save(base, args.output_pt)
    print(f"Saved output packet: {args.output_pt}")

    summary = {
        "num_packets_found": len(packets),
        "map_indices": map_indices,
        "val_indices": val_indices,
        "num_input_gaussians": N,
        "num_bad": num_bad,
        "num_keep": num_keep,
        "prune_ratio": prune_ratio,
        "num_output_gaussians": int(base["means"].shape[0]),
        "args": vars(args) | {"packet_dir": str(args.packet_dir), "depth_dir": str(args.depth_dir) if args.depth_dir is not None else None, "output_pt": str(args.output_pt), "output_dir": str(out_dir)},
    }
    with open(out_dir / "free_space_pruning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {out_dir / 'free_space_pruning_summary.json'}")

    if args.export_bad_ply and num_bad > 0:
        save_ply_points(out_dir / "bad_free_space_gaussians.ply", gaussians["means"][bad_mask], source_ids[bad_mask], max_points=args.max_ply_points)
        print(f"Saved bad PLY: {out_dir / 'bad_free_space_gaussians.ply'}")
    if args.export_keep_ply and num_keep > 0:
        save_ply_points(out_dir / "kept_gaussians.ply", gaussians["means"][keep_mask], source_ids[keep_mask], max_points=args.max_ply_points)
        print(f"Saved kept PLY: {out_dir / 'kept_gaussians.ply'}")

    print("Done.")


if __name__ == "__main__":
    main()
