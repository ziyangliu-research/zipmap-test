#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render one fused/pruned Gaussian packet along the target-camera trajectory stored
in original packet files, with optional camera-trajectory overlay.

This is a backward-compatible extension of render_fused_packet_trajectory.py.
V5 overlay changes: adds custom_pose_intermediate_maps mode for rendering P8 intermediate maintained-map .pt files. Overlay remains center-dot-only.
New feature:
  --overlay_trajectory
      Draw camera centers and viewing directions projected into each rendered view.

Typical usage:
python render_fused_packet_trajectory_with_camera_overlay.py \
  --fused_packet_pt /path/to/fused.pt \
  --trajectory_packet_dir /path/to/gaussian_packets_api/final \
  --trajectory_ranges 0-79 \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --output_dir /path/to/render_overlay \
  --device cuda:0 \
  --filename_simple \
  --save_gt \
  --save_diff \
  --overlay_trajectory \
  --overlay_every 2 \
  --overlay_arrow_every 5 \
  --overlay_arrow_len 0.25 \
  --extrinsic_type Twc
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

GAUSSIAN_KEYS = ["means", "covariances", "harmonics", "opacities", "scales", "rotations", "rotations_unnorm"]


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def parse_ranges(spec: Optional[str], n: int) -> List[int]:
    if spec is None or spec.strip() == "":
        return list(range(n))
    out: List[int] = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a_str, b_str = token.split('-', 1)
            a, b = int(a_str), int(b_str)
            if b < a:
                raise ValueError(f"Invalid descending range: {token}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    out = sorted(set(out))
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise IndexError(f"Trajectory packet indices out of range 0..{n-1}: {bad[:20]}")
    return out


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def normalize_gaussian_field(x: Any, key: str) -> torch.Tensor:
    """Return field with explicit batch dim: [1, N, ...]."""
    t = to_tensor(x)
    if key in {"means", "scales", "rotations", "rotations_unnorm"}:
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    elif key == "covariances":
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.ndim != 4:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    elif key == "harmonics":
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.ndim != 4:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    elif key == "opacities":
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim == 3 and t.shape[-1] == 1:
            t = t[..., 0]
        if t.ndim != 2:
            raise RuntimeError(f"Unexpected {key} shape: {tuple(t.shape)}")
    return t.contiguous()


def load_fused_gaussians(packet_pt: Path, device: str, resplat_repo: Path):
    data = torch.load(packet_pt, map_location="cpu")
    if not isinstance(data, dict):
        raise RuntimeError(f"fused_packet_pt is not a dict: {packet_pt}")
    missing = [k for k in GAUSSIAN_KEYS if k not in data]
    if missing:
        raise KeyError(f"Fused packet missing Gaussian keys: {missing}")

    fields = {k: normalize_gaussian_field(data[k], k).to(device) for k in GAUSSIAN_KEYS}
    Gaussians, GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg = lazy_import_resplat(resplat_repo)

    class DatasetCfgLike:
        background_color = data.get("background_color", [0.0, 0.0, 0.0])

    decoder_cfg = GSplatDecoderSplattingCUDACfg(name="gsplat", scale_invariant=False, use_covariances=True)
    decoder = GSplatDecoderSplattingCUDA(decoder_cfg, DatasetCfgLike()).to(device).eval()
    gaussians = Gaussians(
        means=fields["means"],
        covariances=fields["covariances"],
        harmonics=fields["harmonics"],
        opacities=fields["opacities"],
        scales=fields["scales"],
        rotations=fields["rotations"],
        rotations_unnorm=fields["rotations_unnorm"],
    )
    return data, decoder, gaussians, int(fields["means"].shape[1])


def first_camera_matrix(x: Any, shape: Tuple[int, int]) -> torch.Tensor:
    t = to_tensor(x)
    while t.ndim > 2:
        t = t[0]
    if tuple(t.shape) != shape:
        raise RuntimeError(f"Expected camera matrix shape {shape}, got {tuple(t.shape)}")
    return t


def first_scalar_tensor(x: Any, default: float) -> torch.Tensor:
    if x is None:
        return torch.tensor(default, dtype=torch.float32)
    t = to_tensor(x)
    if t.numel() == 0:
        return torch.tensor(default, dtype=torch.float32)
    return t.reshape(-1)[0].float()


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
    return (320, 320)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    arr = (x.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def make_abs_diff(render_img: Image.Image, gt_img: Optional[Image.Image], gain: float = 4.0) -> Optional[Image.Image]:
    if gt_img is None:
        return None
    r = np.asarray(render_img).astype(np.float32) / 255.0
    g = np.asarray(gt_img.resize(render_img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    d = np.clip(np.abs(r - g) * gain, 0.0, 1.0)
    return Image.fromarray((d * 255.0).astype(np.uint8))


def compute_psnr(render_img: Image.Image, gt_img: Optional[Image.Image]) -> Optional[float]:
    if gt_img is None:
        return None
    r = np.asarray(render_img).astype(np.float32) / 255.0
    g = np.asarray(gt_img.resize(render_img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    mse = float(np.mean((r - g) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * math.log10(mse))


def compute_ssim_simple(render_img: Image.Image, gt_img: Optional[Image.Image]) -> Optional[float]:
    if gt_img is None:
        return None
    x = np.asarray(render_img.resize(gt_img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    y = np.asarray(gt_img).astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x.mean(axis=2)
    if y.ndim == 3:
        y = y.mean(axis=2)
    ux, uy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - ux) * (y - uy)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)))


def load_trajectory_packets(packet_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    paths = sorted(packet_dir.glob("*.pt"), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .pt trajectory packets found in {packet_dir}")
    out = []
    for p in paths:
        d = torch.load(p, map_location="cpu")
        if not isinstance(d, dict):
            raise RuntimeError(f"Trajectory packet is not a dict: {p}")
        for k in ["target_extrinsics", "target_intrinsics"]:
            if k not in d:
                raise KeyError(f"Trajectory packet {p.name} missing {k}")
        out.append((p, d))
    return out


def render_one(decoder, gaussians, packet: Dict[str, Any], device: str, image_shape_override: Optional[Tuple[int, int]] = None) -> Image.Image:
    H, W = image_shape_override or infer_image_shape(packet)
    ext = first_camera_matrix(packet["target_extrinsics"], (4, 4))[None, None].to(device)
    K = first_camera_matrix(packet["target_intrinsics"], (3, 3))[None, None].to(device)
    near = first_scalar_tensor(packet.get("target_near", None), 0.1)[None, None].to(device)
    far = first_scalar_tensor(packet.get("target_far", None), 50.0)[None, None].to(device)
    with torch.no_grad():
        out = decoder.forward(gaussians, ext, K, near, far, (H, W), depth_mode=None)
    return tensor_to_pil(out.color[0, 0])


def get_gt(packet: Dict[str, Any]) -> Optional[Image.Image]:
    img = packet.get("target_image", None)
    if not isinstance(img, torch.Tensor):
        return None
    if img.ndim == 4:
        img = img[0]
    return tensor_to_pil(img)


def read_pose_json(path: Path) -> torch.Tensor:
    """Read a 4x4 custom pose JSON. Compatible with keys Twc/extrinsics/pose or raw 4x4 list."""
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("Twc", "extrinsics", "pose"):
            if key in data:
                data = data[key]
                break
    T = torch.tensor(data, dtype=torch.float32)
    if T.numel() == 16:
        T = T.reshape(4, 4)
    if tuple(T.shape) != (4, 4):
        raise ValueError(f"Pose JSON must contain a 4x4 matrix, got {tuple(T.shape)} from {path}")
    return T


def build_norm_intrinsics_from_args(args: argparse.Namespace, image_shape: Tuple[int, int]) -> torch.Tensor:
    """Build normalized intrinsics for ReSplat gsplat decoder."""
    H, W = image_shape
    if args.probe_intrinsics_norm is not None and args.probe_intrinsics_pixel is not None:
        raise ValueError("Use either --probe_intrinsics_norm or --probe_intrinsics_pixel, not both.")
    if args.probe_intrinsics_norm is not None:
        fx, fy, cx, cy = [float(x) for x in args.probe_intrinsics_norm]
    elif args.probe_intrinsics_pixel is not None:
        fxp, fyp, cxp, cyp = [float(x) for x in args.probe_intrinsics_pixel]
        fx, fy, cx, cy = fxp / W, fyp / H, cxp / W, cyp / H
    else:
        # Conservative fallback for TartanAir-like normalized intrinsics.
        fx, fy, cx, cy = 0.5, 0.5, 0.5, 0.5
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def build_custom_pose_packet(args: argparse.Namespace) -> Dict[str, Any]:
    """Create a packet-like dict for rendering one fixed custom pose."""
    if args.probe_pose_json is None:
        raise ValueError("--render_mode custom_pose requires --probe_pose_json")
    if args.probe_image_shape is None and args.image_shape is None:
        raise ValueError("--render_mode custom_pose requires --probe_image_shape H W or --image_shape H W")
    image_shape = tuple(args.probe_image_shape if args.probe_image_shape is not None else args.image_shape)
    image_shape = (int(image_shape[0]), int(image_shape[1]))
    T = read_pose_json(args.probe_pose_json)
    K = build_norm_intrinsics_from_args(args, image_shape)
    return {
        "target_extrinsics": T,
        "target_intrinsics": K,
        "target_near": torch.tensor([float(args.probe_near)], dtype=torch.float32),
        "target_far": torch.tensor([float(args.probe_far)], dtype=torch.float32),
        "target_image": None,
        "target_index": torch.tensor([-1], dtype=torch.int64),
        "image_shape": image_shape,
        "scene": "custom_pose",
    }


# -------------------------
# Camera trajectory overlay
# -------------------------

def get_pose_matrix(packet: Dict[str, Any]) -> np.ndarray:
    T = first_camera_matrix(packet["target_extrinsics"], (4, 4)).detach().cpu().numpy().astype(np.float64)
    return T


def twc_from_extrinsic(T: np.ndarray, extrinsic_type: str) -> np.ndarray:
    if extrinsic_type.lower() == "twc":
        return T
    if extrinsic_type.lower() == "tcw":
        return np.linalg.inv(T)
    raise ValueError(f"Unknown extrinsic_type: {extrinsic_type}")


def tcw_from_extrinsic(T: np.ndarray, extrinsic_type: str) -> np.ndarray:
    if extrinsic_type.lower() == "tcw":
        return T
    if extrinsic_type.lower() == "twc":
        return np.linalg.inv(T)
    raise ValueError(f"Unknown extrinsic_type: {extrinsic_type}")


def camera_center_and_forward(packet: Dict[str, Any], extrinsic_type: str, camera_forward_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    T_raw = get_pose_matrix(packet)
    Twc = twc_from_extrinsic(T_raw, extrinsic_type)
    center = Twc[:3, 3].astype(np.float64)
    Rwc = Twc[:3, :3].astype(np.float64)

    axis = camera_forward_axis.lower()
    if axis == "z":
        fwd_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif axis == "-z":
        fwd_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif axis == "x":
        fwd_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif axis == "-x":
        fwd_cam = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError("camera_forward_axis must be z, -z, x, or -x")

    forward = Rwc @ fwd_cam
    n = np.linalg.norm(forward)
    if n > 1e-12:
        forward = forward / n
    return center, forward


def get_pixel_intrinsics_from_packet(render_packet: Dict[str, Any], image_size: Tuple[int, int]) -> np.ndarray:
    """Return pixel-space intrinsics for overlay projection.

    ReSplat packets usually store normalized intrinsics because the decoder
    internally multiplies by image width/height. For overlay drawing we need
    pixel-space intrinsics. If K looks normalized, convert it here.
    """
    H, W = image_size
    K = first_camera_matrix(render_packet["target_intrinsics"], (3, 3)).detach().cpu().numpy().astype(np.float64).copy()
    # Normalized ReSplat-style intrinsics usually have fx/fy/cx/cy in roughly [0, 2].
    # Pixel intrinsics for 320/540/960 images are much larger.
    if abs(K[0, 0]) < 10.0 and abs(K[1, 1]) < 10.0 and abs(K[0, 2]) <= 2.0 and abs(K[1, 2]) <= 2.0:
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= W
        K[1, 2] *= H
    return K


def project_world_points(points_w: np.ndarray, render_packet: Dict[str, Any], extrinsic_type: str, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    T_raw = get_pose_matrix(render_packet)
    Tcw = tcw_from_extrinsic(T_raw, extrinsic_type)
    K = get_pixel_intrinsics_from_packet(render_packet, image_size)

    pts_h = np.concatenate([points_w.astype(np.float64), np.ones((points_w.shape[0], 1), dtype=np.float64)], axis=1)
    pc = (Tcw @ pts_h.T).T[:, :3]
    z = pc[:, 2]
    valid_z = z > 1e-6
    uv = np.full((points_w.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid_z):
        x = pc[valid_z, 0] / z[valid_z]
        y = pc[valid_z, 1] / z[valid_z]
        uv[valid_z, 0] = K[0, 0] * x + K[0, 2]
        uv[valid_z, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid_z


def in_loose_image(u: float, v: float, W: int, H: int, margin: float) -> bool:
    return -margin <= u < W + margin and -margin <= v < H + margin


def rgba(xs: Sequence[int], alpha_default: int) -> Tuple[int, int, int, int]:
    if len(xs) >= 4:
        return int(xs[0]), int(xs[1]), int(xs[2]), int(xs[3])
    return int(xs[0]), int(xs[1]), int(xs[2]), int(alpha_default)


def draw_arrow(draw: ImageDraw.ImageDraw, p0: Tuple[float, float], p1: Tuple[float, float], color: Tuple[int, int, int, int], width: int) -> None:
    draw.line([p0, p1], fill=color, width=width)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 3.0:
        return
    ux, uy = dx / length, dy / length
    head_len = min(10.0, max(5.0, 0.25 * length))
    angle = math.radians(28.0)
    ca, sa = math.cos(angle), math.sin(angle)
    for sign in (1.0, -1.0):
        rx = ca * ux - sign * sa * uy
        ry = sign * sa * ux + ca * uy
        q = (p1[0] - head_len * rx, p1[1] - head_len * ry)
        draw.line([p1, q], fill=color, width=width)


def clamp_point_to_image(u: float, v: float, W: int, H: int, pad: int = 8) -> Tuple[float, float, bool]:
    """Clamp point to image rectangle. Return (u,v,was_clamped)."""
    cu = min(max(float(u), float(pad)), float(W - 1 - pad))
    cv = min(max(float(v), float(pad)), float(H - 1 - pad))
    return cu, cv, (abs(cu - float(u)) > 1e-3 or abs(cv - float(v)) > 1e-3)


def draw_camera_triangle(
    draw: ImageDraw.ImageDraw,
    center_uv: Tuple[float, float],
    dir_uv: Optional[Tuple[float, float]],
    size: int,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int] = (255, 255, 255, 230),
    outline_width: int = 3,
    fill_enabled: bool = False,
    flip_direction: bool = False,
) -> None:
    """Draw an oriented isosceles camera marker.

    Convention:
      - The triangle tip points to the projected camera forward direction.
      - The base is behind the camera center.
      - By default it is drawn as an outline marker, because solid red triangles
        are visually too dominant on rendered images.

    If the arrow/triangle direction looks opposite in your coordinate convention,
    run with --overlay_flip_triangle.
    """
    cx, cy = center_uv
    if dir_uv is None:
        ux, uy = 0.0, -1.0
    else:
        dx, dy = dir_uv[0] - cx, dir_uv[1] - cy
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            ux, uy = 0.0, -1.0
        else:
            ux, uy = dx / norm, dy / norm

    if flip_direction:
        ux, uy = -ux, -uy

    px, py = -uy, ux

    tip = (cx + size * ux, cy + size * uy)
    base_center = (cx - 0.50 * size * ux, cy - 0.50 * size * uy)
    half_base = 0.45 * size
    left = (base_center[0] + half_base * px, base_center[1] + half_base * py)
    right = (base_center[0] - half_base * px, base_center[1] - half_base * py)

    pts = [tip, left, right]
    if fill_enabled:
        draw.polygon(pts, fill=fill)
    draw.line([tip, left, right, tip], fill=outline, width=max(1, int(outline_width)), joint="curve")

    rr = max(2, size // 7)
    draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=outline)

def overlay_trajectory_on_image(
    image: Image.Image,
    render_packet: Dict[str, Any],
    trajectory: Sequence[Tuple[Path, Dict[str, Any]]],
    overlay_indices: Sequence[int],
    current_packet_index: int,
    args: argparse.Namespace,
) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    W, H = base.size
    margin = float(args.overlay_margin_px)
    image_size = (H, W)

    centers: List[np.ndarray] = []
    forwards: List[np.ndarray] = []
    for idx in overlay_indices:
        _, pkt = trajectory[idx]
        c, f = camera_center_and_forward(pkt, args.extrinsic_type, args.camera_forward_axis)
        centers.append(c)
        forwards.append(f)
    if not centers:
        return base.convert("RGB")

    centers_np = np.stack(centers, axis=0)
    uv, valid_z = project_world_points(centers_np, render_packet, args.extrinsic_type, image_size)

    traj_color = rgba(args.overlay_traj_color, args.overlay_alpha)
    point_color = rgba(args.overlay_point_color, args.overlay_alpha)
    arrow_color = rgba(args.overlay_arrow_color, args.overlay_alpha)
    current_color = rgba(args.overlay_current_color, 255)
    triangle_color = rgba(args.overlay_triangle_color, 255)

    # Optional polyline. For custom prefix visualization, marker-only is often clearer.
    if not args.overlay_no_line:
        last_pt = None
        last_valid = False
        for i, idx in enumerate(overlay_indices):
            u, v = uv[i]
            valid = bool(valid_z[i]) and np.isfinite(u) and np.isfinite(v) and in_loose_image(float(u), float(v), W, H, margin)
            if valid and args.overlay_clamp_to_image:
                u, v, _ = clamp_point_to_image(float(u), float(v), W, H, pad=int(args.overlay_edge_pad))
            if last_valid and valid and last_pt is not None:
                draw.line([last_pt, (float(u), float(v))], fill=traj_color, width=int(args.overlay_line_width))
            if valid:
                last_pt = (float(u), float(v))
                last_valid = True
            else:
                last_pt = None
                last_valid = False

    every = max(1, int(args.overlay_every))
    drawn = 0
    behind = 0
    outside = 0

    # Draw large camera triangles. This is intentionally more visible than dots.
    for i, idx in enumerate(overlay_indices):
        if i % every != 0:
            continue
        u, v = uv[i]
        if not bool(valid_z[i]):
            behind += 1
            continue
        if not (np.isfinite(u) and np.isfinite(v)):
            continue

        visible = in_loose_image(float(u), float(v), W, H, margin)
        if not visible:
            outside += 1
            if not args.overlay_clamp_to_image:
                continue

        # Project a short forward endpoint to orient the triangle.
        c = centers_np[i]
        p1 = c + float(args.overlay_arrow_len) * forwards[i]
        uv2, valid2 = project_world_points(np.stack([c, p1], axis=0), render_packet, args.extrinsic_type, image_size)
        dir_uv: Optional[Tuple[float, float]] = None
        if valid2[0] and valid2[1] and np.all(np.isfinite(uv2)):
            dir_uv = (float(uv2[1, 0]), float(uv2[1, 1]))

        uu, vv = float(u), float(v)
        was_clamped = False
        if args.overlay_clamp_to_image:
            uu, vv, was_clamped = clamp_point_to_image(uu, vv, W, H, pad=int(args.overlay_edge_pad))
            if dir_uv is not None:
                du, dv, _ = clamp_point_to_image(dir_uv[0], dir_uv[1], W, H, pad=int(args.overlay_edge_pad))
                dir_uv = (du, dv)

        # Center-dot-only marker. Direction/frustum overlay is intentionally disabled.
        fill = rgba(args.overlay_dot_color, args.overlay_alpha)
        outline = rgba(args.overlay_dot_outline_color, 240)
        if was_clamped:
            # If clamped to border, use yellow-orange so user knows it is outside the view.
            fill = (255, 200, 0, int(args.overlay_alpha))
        r = int(args.overlay_dot_radius)
        draw.ellipse(
            [uu - r, vv - r, uu + r, vv + r],
            fill=fill,
            outline=outline,
            width=max(1, int(args.overlay_dot_outline_width)),
        )
        drawn += 1

    # Highlight current camera with a larger marker.
    if current_packet_index in overlay_indices:
        j = list(overlay_indices).index(current_packet_index)
        u, v = uv[j]
        if bool(valid_z[j]) and np.isfinite(u) and np.isfinite(v):
            if in_loose_image(float(u), float(v), W, H, margin) or args.overlay_clamp_to_image:
                uu, vv = float(u), float(v)
                if args.overlay_clamp_to_image:
                    uu, vv, _ = clamp_point_to_image(uu, vv, W, H, pad=int(args.overlay_edge_pad))
                # Current camera center marker. Direction/frustum overlay is intentionally disabled.
                r = int(args.overlay_current_dot_radius)
                fill = rgba(args.overlay_current_dot_color, 255)
                outline = rgba(args.overlay_current_dot_outline_color, 255)
                draw.ellipse(
                    [uu - r, vv - r, uu + r, vv + r],
                    fill=fill,
                    outline=outline,
                    width=max(2, int(args.overlay_dot_outline_width) + 1),
                )

    if args.overlay_legend:
        text = f"trajectory {overlay_indices[0]}-{overlay_indices[-1]} | current {current_packet_index} | drawn {drawn}"
        if behind or outside:
            text += f" | behind {behind}, outside {outside}"
        pad = 6
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([8, 8, 8 + tw + 2 * pad, 8 + th + 2 * pad], fill=(0, 0, 0, 160))
        draw.text((8 + pad, 8 + pad), text, fill=(255, 255, 255, 245))

    return base.convert("RGB")




def create_decoder_from_background(background_color: Any, device: str, resplat_repo: Path):
    """Create ReSplat gsplat decoder without requiring a fused packet file."""
    _, GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg = lazy_import_resplat(resplat_repo)

    class DatasetCfgLike:
        pass

    if isinstance(background_color, torch.Tensor):
        bg = background_color.detach().cpu()
    elif background_color is None:
        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        bg = torch.tensor(background_color, dtype=torch.float32)
    DatasetCfgLike.background_color = bg

    decoder_cfg = GSplatDecoderSplattingCUDACfg(name="gsplat", scale_invariant=False, use_covariances=True)
    return GSplatDecoderSplattingCUDA(decoder_cfg, DatasetCfgLike()).to(device).eval()


def concat_packets_to_gaussians_for_render(packet_list: Sequence[Dict[str, Any]], device: str, resplat_repo: Path):
    """Concatenate raw packet dicts into one ReSplat Gaussians object.

    This is intentionally simple and mirrors the prefix-fusion behavior in
    run_zipmap_resplat_fusion_api.py: packet fields are concatenated without
    pruning/maintenance. Use this for diagnostic prefix rendering only.
    """
    if not packet_list:
        raise ValueError("packet_list is empty")
    Gaussians, _, _ = lazy_import_resplat(resplat_repo)

    def norm_no_batch(pkt: Dict[str, Any], key: str) -> torch.Tensor:
        if key not in pkt or pkt[key] is None:
            raise KeyError(f"Packet missing Gaussian key {key}")
        t = to_tensor(pkt[key])
        # Raw packet fields are usually [N,...]. If a batch dim exists, remove first batch.
        if key in {"means", "scales", "rotations", "rotations_unnorm"}:
            if t.ndim == 3 and t.shape[0] == 1:
                t = t[0]
            if t.ndim != 2:
                raise RuntimeError(f"Unexpected {key} shape in packet: {tuple(t.shape)}")
        elif key == "covariances":
            if t.ndim == 4 and t.shape[0] == 1:
                t = t[0]
            if t.ndim != 3:
                raise RuntimeError(f"Unexpected {key} shape in packet: {tuple(t.shape)}")
        elif key == "harmonics":
            if t.ndim == 4 and t.shape[0] == 1:
                t = t[0]
            if t.ndim != 3:
                raise RuntimeError(f"Unexpected {key} shape in packet: {tuple(t.shape)}")
        elif key == "opacities":
            if t.ndim == 2 and t.shape[0] == 1:
                t = t[0]
            if t.ndim == 2 and t.shape[-1] == 1:
                t = t[:, 0]
            if t.ndim != 1:
                raise RuntimeError(f"Unexpected {key} shape in packet: {tuple(t.shape)}")
        return t.contiguous()

    fields = {}
    for key in GAUSSIAN_KEYS:
        fields[key] = torch.cat([norm_no_batch(pkt, key) for pkt in packet_list], dim=0).to(device, non_blocking=True).unsqueeze(0).contiguous()

    gaussians = Gaussians(
        means=fields["means"],
        covariances=fields["covariances"],
        harmonics=fields["harmonics"],
        opacities=fields["opacities"],
        scales=fields["scales"],
        rotations=fields["rotations"],
        rotations_unnorm=fields["rotations_unnorm"],
    )
    return gaussians, int(fields["means"].shape[1])

def main() -> None:
    parser = argparse.ArgumentParser(description="Render a fused/pruned Gaussian packet along original packet target trajectory, optionally with trajectory overlay.")
    parser.add_argument("--fused_packet_pt", type=Path, default=None, help="Required for --render_mode trajectory/custom_pose. Not used for custom_pose_prefix/custom_pose_intermediate_maps.")
    parser.add_argument("--intermediate_map_dir", type=Path, default=None, help="Directory containing P8 intermediate maintained-map .pt files for --render_mode custom_pose_intermediate_maps.")
    parser.add_argument("--intermediate_map_glob", type=str, default="step_*.pt", help="Glob pattern inside --intermediate_map_dir for intermediate maps.")
    parser.add_argument("--trajectory_packet_dir", type=Path, required=True)
    parser.add_argument("--trajectory_ranges", type=str, default="")
    parser.add_argument("--resplat_repo", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--filename_simple", action="store_true", help="Save rendered images as 001.png, 002.png, ...")
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_diff", action="store_true")

    # Render camera mode.
    parser.add_argument("--render_mode", choices=["trajectory", "custom_pose", "custom_pose_prefix", "custom_pose_intermediate_maps"], default="trajectory",
                        help="trajectory: render one fused map along packet target views; custom_pose: render one fused map at one fixed custom camera; custom_pose_prefix: read raw packets and render each prefix-concat map; custom_pose_intermediate_maps: read pre-saved maintained-map .pt files and render each at one fixed custom camera.")
    parser.add_argument("--probe_pose_json", type=Path, default=None, help="4x4 custom pose JSON for --render_mode custom_pose.")
    parser.add_argument("--probe_image_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--probe_intrinsics_norm", type=float, nargs=4, default=None, metavar=("fx", "fy", "cx", "cy"))
    parser.add_argument("--probe_intrinsics_pixel", type=float, nargs=4, default=None, metavar=("fx", "fy", "cx", "cy"))
    parser.add_argument("--probe_near", type=float, default=0.1)
    parser.add_argument("--probe_far", type=float, default=50.0)

    # Overlay options.
    parser.add_argument("--overlay_trajectory", action="store_true", help="Draw camera-center trajectory dots on each rendered image. Direction/frustum markers are disabled in this V4 script.")
    parser.add_argument("--overlay_ranges", type=str, default=None, help="Trajectory indices to draw. Default: same as --trajectory_ranges.")
    parser.add_argument("--overlay_every", type=int, default=1, help="Draw one camera center every N trajectory poses.")
    parser.add_argument("--overlay_arrow_every", type=int, default=5, help="Draw one camera direction arrow every N trajectory poses.")
    parser.add_argument("--overlay_arrow_len", type=float, default=0.25, help="Arrow length in world units.")
    parser.add_argument("--overlay_point_radius", type=int, default=3)
    parser.add_argument("--overlay_current_radius", type=int, default=6)
    parser.add_argument("--overlay_line_width", type=int, default=2)
    parser.add_argument("--overlay_arrow_width", type=int, default=2)
    parser.add_argument("--overlay_alpha", type=int, default=220)
    parser.add_argument("--overlay_margin_px", type=float, default=80.0)
    parser.add_argument("--overlay_legend", action="store_true")
    parser.add_argument("--overlay_replace_rendered", action="store_true", help="Also overwrite rendered/ images with overlay images.")
    parser.add_argument("--extrinsic_type", choices=["Twc", "Tcw"], default="Twc")
    parser.add_argument("--camera_forward_axis", choices=["z", "-z", "x", "-x"], default="z")
    parser.add_argument("--overlay_traj_color", type=int, nargs="+", default=[255, 255, 0, 220])
    parser.add_argument("--overlay_point_color", type=int, nargs="+", default=[0, 255, 255, 220])
    parser.add_argument("--overlay_arrow_color", type=int, nargs="+", default=[255, 128, 0, 220])
    parser.add_argument("--overlay_current_color", type=int, nargs="+", default=[255, 0, 0, 255])
    parser.add_argument("--overlay_triangle_color", type=int, nargs="+", default=[255, 0, 0, 60], help="Fill color for camera triangle markers; used only when --overlay_triangle_fill is set.")
    parser.add_argument("--overlay_triangle_outline_color", type=int, nargs="+", default=[255, 80, 80, 230], help="Outline color for camera triangle markers.")
    parser.add_argument("--overlay_current_outline_color", type=int, nargs="+", default=[255, 255, 0, 255], help="Outline color for current camera marker.")
    parser.add_argument("--overlay_triangle_fill", action="store_true", help="Fill camera triangles. Default: outline only.")
    parser.add_argument("--overlay_triangle_outline_width", type=int, default=3)
    parser.add_argument("--overlay_flip_triangle", action="store_true", help="Flip the 2D triangle direction if camera forward appears reversed.")
    parser.add_argument("--overlay_triangle_size", type=int, default=18, help="Triangle marker size for trajectory cameras.")
    parser.add_argument("--overlay_current_triangle_size", type=int, default=28, help="Triangle marker size for the current camera.")
    # Effective V4 marker options: center dots only.
    parser.add_argument("--overlay_dot_radius", type=int, default=5, help="Radius of historical camera center dots.")
    parser.add_argument("--overlay_current_dot_radius", type=int, default=9, help="Radius of the current camera center dot.")
    parser.add_argument("--overlay_dot_color", type=int, nargs="+", default=[255, 80, 80, 180], help="Fill color for historical camera center dots.")
    parser.add_argument("--overlay_dot_outline_color", type=int, nargs="+", default=[255, 255, 255, 230], help="Outline color for historical camera center dots.")
    parser.add_argument("--overlay_current_dot_color", type=int, nargs="+", default=[255, 255, 0, 230], help="Fill color for the current camera center dot.")
    parser.add_argument("--overlay_current_dot_outline_color", type=int, nargs="+", default=[0, 0, 0, 255], help="Outline color for the current camera center dot.")
    parser.add_argument("--overlay_dot_outline_width", type=int, default=2)

    parser.add_argument("--overlay_no_line", action="store_true", help="Do not draw trajectory lines; draw camera triangles only.")
    parser.add_argument("--overlay_clamp_to_image", action="store_true", default=False, help="Clamp out-of-frame projected camera markers to image border so they remain visible. Default: off; invisible cameras are skipped.")
    parser.add_argument("--no_overlay_clamp_to_image", action="store_false", dest="overlay_clamp_to_image")
    parser.add_argument("--overlay_edge_pad", type=int, default=14, help="Padding used when clamping markers to the image border.")
    args = parser.parse_args()

    out_dir = args.output_dir
    render_dir = out_dir / "rendered"
    overlay_dir = out_dir / "rendered_overlay"
    gt_dir = out_dir / "gt"
    diff_dir = out_dir / "diff"
    ensure_dir(render_dir)
    if args.overlay_trajectory:
        ensure_dir(overlay_dir)
    if args.save_gt:
        ensure_dir(gt_dir)
    if args.save_diff:
        ensure_dir(diff_dir)

    print(f"[1/3] Preparing renderer")
    decoder = None
    gaussians = None
    num_gauss = None
    if args.render_mode in {"trajectory", "custom_pose"}:
        if args.fused_packet_pt is None:
            raise ValueError(f"--fused_packet_pt is required for --render_mode {args.render_mode}")
        print(f"      loading fused packet: {args.fused_packet_pt}")
        _, decoder, gaussians, num_gauss = load_fused_gaussians(args.fused_packet_pt, args.device, args.resplat_repo)
        print(f"      num Gaussians: {num_gauss:,}")

    print(f"[2/3] Loading trajectory packets: {args.trajectory_packet_dir}")
    trajectory = load_trajectory_packets(args.trajectory_packet_dir)
    if args.render_mode == "custom_pose_prefix":
        # Use first packet background color to initialize decoder. Gaussians are built per prefix.
        decoder = create_decoder_from_background(trajectory[0][1].get("background_color", None), args.device, args.resplat_repo)

    intermediate_maps = []
    if args.render_mode == "custom_pose_intermediate_maps":
        if args.intermediate_map_dir is None:
            raise ValueError("--intermediate_map_dir is required for --render_mode custom_pose_intermediate_maps")
        intermediate_maps = sorted(Path(args.intermediate_map_dir).glob(args.intermediate_map_glob), key=natural_sort_key)
        if not intermediate_maps:
            raise FileNotFoundError(f"No intermediate maps found: {args.intermediate_map_dir}/{args.intermediate_map_glob}")
        print(f"      intermediate maps: {len(intermediate_maps)} from {args.intermediate_map_dir}")

    selected = parse_ranges(args.trajectory_ranges, len(trajectory))
    custom_packet = None
    if args.render_mode == "trajectory":
        render_jobs = []
        for packet_i in selected:
            path, packet = trajectory[packet_i]
            frame_idx = first_int(packet.get("target_index", packet_i), packet_i)
            render_jobs.append((packet_i, path, packet, frame_idx, None))
        overlay_indices = parse_ranges(args.overlay_ranges, len(trajectory)) if args.overlay_ranges else selected
        print(f"      selected trajectory views: {len(selected)}")
    elif args.render_mode == "custom_pose":
        custom_packet = build_custom_pose_packet(args)
        render_jobs = [(-1, args.probe_pose_json, custom_packet, -1, None)]
        # For custom pose, trajectory_ranges controls the overlay trajectory unless --overlay_ranges is given.
        overlay_spec = args.overlay_ranges if args.overlay_ranges is not None else args.trajectory_ranges
        overlay_indices = parse_ranges(overlay_spec, len(trajectory))
        print("      selected custom pose views: 1")
        print(f"      custom pose: {args.probe_pose_json}")
    elif args.render_mode == "custom_pose_prefix":
        custom_packet = build_custom_pose_packet(args)
        render_jobs = []
        # Each output frame renders one prefix map at the same custom pose.
        # prefix_indices are selected[0:i+1], so image k shows fusion progress up to packet selected[k].
        for out_i, packet_i in enumerate(selected):
            path, _ = trajectory[packet_i]
            frame_idx = first_int(trajectory[packet_i][1].get("target_index", packet_i), packet_i)
            prefix_indices = selected[: out_i + 1]
            render_jobs.append((packet_i, path, custom_packet, frame_idx, prefix_indices))
        overlay_indices = selected
        print(f"      selected prefix steps: {len(selected)}")
        print(f"      custom pose: {args.probe_pose_json}")
    elif args.render_mode == "custom_pose_intermediate_maps":
        custom_packet = build_custom_pose_packet(args)
        render_jobs = []
        n_jobs = min(len(intermediate_maps), len(selected))
        if len(intermediate_maps) != len(selected):
            print(f"      WARNING: intermediate maps ({len(intermediate_maps)}) != selected trajectory indices ({len(selected)}); using first {n_jobs}")
        for out_i in range(n_jobs):
            packet_i = selected[out_i]
            map_path = intermediate_maps[out_i]
            frame_idx = first_int(trajectory[packet_i][1].get("target_index", packet_i), packet_i)
            prefix_indices = selected[: out_i + 1]
            render_jobs.append((packet_i, map_path, custom_packet, frame_idx, prefix_indices))
        overlay_indices = selected[:n_jobs]
        print(f"      selected maintained-map steps: {len(render_jobs)}")
        print(f"      custom pose: {args.probe_pose_json}")
    else:
        raise ValueError(f"Unsupported render_mode={args.render_mode}")

    if args.overlay_trajectory:
        print(f"      overlay poses: {len(overlay_indices)}")
        print(f"      extrinsic_type={args.extrinsic_type}, forward_axis={args.camera_forward_axis}")

    print("[3/3] Rendering")
    rows = []
    shape_override = tuple(args.image_shape) if (args.image_shape is not None and args.render_mode == "trajectory") else None
    for out_i, (packet_i, path, packet, frame_idx, prefix_indices) in enumerate(render_jobs):
        if args.filename_simple:
            filename = f"{out_i + 1:03d}.png"
        else:
            if args.render_mode == "custom_pose":
                filename = f"{out_i + 1:03d}_custom_{Path(path).stem}.png"
            elif args.render_mode == "custom_pose_prefix":
                filename = f"{out_i + 1:03d}_prefix_to_packet{packet_i:04d}_frame{frame_idx:06d}.png"
            elif args.render_mode == "custom_pose_intermediate_maps":
                filename = f"{out_i + 1:03d}_maintained_to_packet{packet_i:04d}_frame{frame_idx:06d}.png"
            else:
                filename = f"{out_i + 1:03d}_packet{packet_i:04d}_frame{frame_idx:06d}.png"

        step_num_gauss = num_gauss
        if args.render_mode == "custom_pose_prefix":
            assert prefix_indices is not None
            prefix_packets = [trajectory[i][1] for i in prefix_indices]
            gaussians_step, step_num_gauss = concat_packets_to_gaussians_for_render(
                prefix_packets, args.device, args.resplat_repo
            )
            render_img = render_one(decoder, gaussians_step, packet, args.device, shape_override)
            # Free per-prefix map immediately.
            del gaussians_step
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        elif args.render_mode == "custom_pose_intermediate_maps":
            # Here path is one maintained-map .pt file saved by P8.
            _, decoder_step, gaussians_step, step_num_gauss = load_fused_gaussians(Path(path), args.device, args.resplat_repo)
            render_img = render_one(decoder_step, gaussians_step, packet, args.device, shape_override)
            del decoder_step, gaussians_step
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        else:
            render_img = render_one(decoder, gaussians, packet, args.device, shape_override)
        render_img.save(render_dir / filename)

        overlay_path = None
        if args.overlay_trajectory:
            # In custom_pose_prefix/custom_pose_intermediate_maps, image k should show only the camera path up to current packet.
            # In trajectory/custom_pose mode, draw the user-specified overlay_indices.
            overlay_now = prefix_indices if (args.render_mode in {"custom_pose_prefix", "custom_pose_intermediate_maps"} and prefix_indices is not None) else overlay_indices
            overlay_img = overlay_trajectory_on_image(render_img, packet, trajectory, overlay_now, packet_i, args)
            overlay_path = overlay_dir / filename
            overlay_img.save(overlay_path)
            if args.overlay_replace_rendered:
                overlay_img.save(render_dir / filename)

        gt_img = get_gt(packet)
        if args.save_gt and gt_img is not None:
            gt_img.save(gt_dir / filename)
        if args.save_diff and gt_img is not None:
            diff = make_abs_diff(render_img, gt_img, gain=4.0)
            if diff is not None:
                diff.save(diff_dir / filename)

        psnr = compute_psnr(render_img, gt_img)
        ssim = compute_ssim_simple(render_img, gt_img)
        rows.append({
            "order": out_i,
            "trajectory_packet_sorted_index": packet_i,
            "packet_name": path.name,
            "target_frame_index": frame_idx,
            "prefix_indices": prefix_indices,
            "num_gaussians": step_num_gauss,
            "rendered": str(render_dir / filename),
            "rendered_overlay": str(overlay_path) if overlay_path is not None else None,
            "gt": str(gt_dir / filename) if args.save_gt and gt_img is not None else None,
            "psnr": psnr,
            "ssim": ssim,
        })
        print(f"      [{out_i + 1:03d}/{len(selected):03d}] packet={packet_i:04d}, frame={frame_idx:06d}, psnr={psnr}, ssim={ssim}")

    metrics = [r for r in rows if r["psnr"] is not None]
    summary = {
        "fused_packet_pt": str(args.fused_packet_pt),
        "trajectory_packet_dir": str(args.trajectory_packet_dir),
        "trajectory_ranges": args.trajectory_ranges,
        "render_mode": args.render_mode,
        "probe_pose_json": None if args.probe_pose_json is None else str(args.probe_pose_json),
        "probe_image_shape": args.probe_image_shape,
        "probe_intrinsics_norm": args.probe_intrinsics_norm,
        "num_gaussians": num_gauss,
        "num_views": len(rows),
        "custom_pose_prefix_final_num_gaussians": rows[-1]["num_gaussians"] if rows and args.render_mode in {"custom_pose_prefix", "custom_pose_intermediate_maps"} else None,
        "overlay_trajectory": bool(args.overlay_trajectory),
        "overlay_ranges": args.overlay_ranges if args.overlay_ranges else args.trajectory_ranges,
        "extrinsic_type": args.extrinsic_type,
        "camera_forward_axis": args.camera_forward_axis,
        "mean_psnr": float(np.mean([r["psnr"] for r in metrics])) if metrics else None,
        "mean_ssim": float(np.mean([r["ssim"] for r in metrics])) if metrics else None,
        "per_view": rows,
    }
    with (out_dir / "trajectory_render_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {out_dir / 'trajectory_render_summary.json'}")
    if args.overlay_trajectory:
        print(f"Saved overlay images: {overlay_dir}")


if __name__ == "__main__":
    main()
