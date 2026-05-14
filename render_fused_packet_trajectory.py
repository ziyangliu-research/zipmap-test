#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render one fused/pruned Gaussian packet along the target-camera trajectory stored in original packet files.

Use case
--------
1) Build a fused/pruned packet with free_space_prune_gaussian_packets_p0.py:
   - mark_only output can be used as the naive fused baseline.
   - downweight/hard output can be used as the maintained-map result.

2) Render both along the same trajectory_packet_dir for before/after comparison.

Example
-------
python render_fused_packet_trajectory.py \
  --fused_packet_pt /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/free_space_downweight_0_79.pt \
  --trajectory_packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/gaussian_packets_api/final \
  --trajectory_ranges 0-79 \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --output_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/render_downweight_0_79 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a fused/pruned Gaussian packet along original packet target trajectory.")
    parser.add_argument("--fused_packet_pt", type=Path, required=True)
    parser.add_argument("--trajectory_packet_dir", type=Path, required=True)
    parser.add_argument("--trajectory_ranges", type=str, default="")
    parser.add_argument("--resplat_repo", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--filename_simple", action="store_true", help="Save rendered images as 001.png, 002.png, ...")
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_diff", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    render_dir = out_dir / "rendered"
    gt_dir = out_dir / "gt"
    diff_dir = out_dir / "diff"
    ensure_dir(render_dir)
    if args.save_gt:
        ensure_dir(gt_dir)
    if args.save_diff:
        ensure_dir(diff_dir)

    print(f"[1/3] Loading fused packet: {args.fused_packet_pt}")
    _, decoder, gaussians, num_gauss = load_fused_gaussians(args.fused_packet_pt, args.device, args.resplat_repo)
    print(f"      num Gaussians: {num_gauss:,}")

    print(f"[2/3] Loading trajectory packets: {args.trajectory_packet_dir}")
    trajectory = load_trajectory_packets(args.trajectory_packet_dir)
    selected = parse_ranges(args.trajectory_ranges, len(trajectory))
    print(f"      selected trajectory views: {len(selected)}")

    print("[3/3] Rendering")
    rows = []
    shape_override = tuple(args.image_shape) if args.image_shape is not None else None
    for out_i, packet_i in enumerate(selected):
        path, packet = trajectory[packet_i]
        frame_idx = first_int(packet.get("target_index", packet_i), packet_i)
        if args.filename_simple:
            filename = f"{out_i + 1:03d}.png"
        else:
            filename = f"{out_i + 1:03d}_packet{packet_i:04d}_frame{frame_idx:06d}.png"

        render_img = render_one(decoder, gaussians, packet, args.device, shape_override)
        render_img.save(render_dir / filename)

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
            "rendered": str(render_dir / filename),
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
        "num_gaussians": num_gauss,
        "num_views": len(rows),
        "mean_psnr": float(np.mean([r["psnr"] for r in metrics])) if metrics else None,
        "mean_ssim": float(np.mean([r["ssim"] for r in metrics])) if metrics else None,
        "per_view": rows,
    }
    with (out_dir / "trajectory_render_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {out_dir / 'trajectory_render_summary.json'}")


if __name__ == "__main__":
    main()
