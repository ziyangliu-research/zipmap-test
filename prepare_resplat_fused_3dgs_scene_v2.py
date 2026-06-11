#!/usr/bin/env python3
"""
ReSplat/ZipMap 输出的 gaussian packets
→ initial_fused_3dgs.ply
→ 3DGS 可读取的 transforms_train.json / transforms_test.json
→ packet-camera scene
也就是一个转换接口，把之前系统输出的高斯packets转换成原版3dgs可以读取的ply格式。

命令：
python prepare_resplat_fused_3dgs_scene_v2.py \
  --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_30/gaussian_packets_api/final \
  --packet_ranges 0-29 \
  --dataset_start_index 0 \
  --image_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --image_pattern "{index:06d}_lcam_front.png" \
  --camera_source packet \
  --packet_camera_convention twc_cv \
  --packet_camera_view_index -1 \
  --fx 320 \
  --fy 320 \
  --cx 320 \
  --cy 320 \
  --width 640 \
  --height 640 \
  --output_scene /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_30/3dgs_scene_packetcam \
  --sh_degree 3 \
  --test_same_as_train \
  --copy_images
"""
"""
Prepare an original graphdeco-inria/gaussian-splatting compatible scene
from ReSplat/MVSplat-style Gaussian packet .pt files.

Output:
  <output_scene>/
    images/*.png
    transforms_train.json
    transforms_test.json
    points3d.ply                  # lightweight point-cloud PLY for original Scene init
    initial_fused_3dgs.ply         # full 3DGS Gaussian PLY used by train_fused_3dgs_baseline.py
    prepare_summary.json

Typical use:
  python prepare_resplat_fused_3dgs_scene.py \
    --packet_dir /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_20/gaussian_packets_api/final \
    --packet_ranges 0-19 \
    --image_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
    --image_pattern "{index:06d}_lcam_front.png" \
    --camera_source packet \
    --packet_camera_convention twc_cv \
    --fx 320 --fy 320 --cx 320 --cy 320 --width 640 --height 640 \
    --output_scene /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_20/3dgs_scene \
    --sh_degree 3

Notes:
  - The NeRF/Blender JSON transform_matrix is written in OpenGL camera convention.
    If your input poses are CV camera-to-world (x right, y down, z forward), the script
    writes Twc_cv @ diag(1,-1,-1,1), matching original 3DGS' Blender reader.
  - The full Gaussian PLY stores original 3DGS internal parameterization:
      opacity = inverse_sigmoid(alpha), scale = log(positive_scale), rot = wxyz quaternion.
  - Packet field names vary across MVSplat/ReSplat forks. Run --inspect_only first if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from plyfile import PlyData, PlyElement
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs plyfile. Install it in the 3DGS env: pip install plyfile") from exc

try:
    from scipy.spatial.transform import Rotation as SciRot
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs scipy. Install it in the 3DGS env: pip install scipy") from exc

C0 = 0.28209479177387814
EPS = 1e-6


# ----------------------------- generic utils -----------------------------

def parse_int_ranges(spec: str) -> List[int]:
    out: List[int] = []
    if spec is None or spec.strip() == "":
        return out
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    # preserve order, remove duplicates
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def sorted_packet_files(packet_dir: Path) -> List[Path]:
    files = sorted([p for p in packet_dir.iterdir() if p.suffix.lower() in {'.pt', '.pth'}])
    if not files:
        raise FileNotFoundError(f"No .pt/.pth packets found under {packet_dir}")
    return files


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def walk_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix.rstrip('.'), obj
    elif isinstance(obj, np.ndarray):
        yield prefix.rstrip('.'), torch.from_numpy(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_tensors(v, prefix + str(k) + '.')
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            yield from walk_tensors(v, prefix + str(k) + '.')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_tensors(v, prefix + str(i) + '.')


def get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split('.'):
        if part == '':
            continue
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def tensor_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_last(x: np.ndarray, last_shape: Tuple[int, ...], name: str) -> np.ndarray:
    x = np.asarray(x)
    if len(last_shape) == 1:
        d = last_shape[0]
        if x.shape[-1] != d:
            raise ValueError(f"{name}: expected last dim {d}, got shape {x.shape}")
        return x.reshape(-1, d)
    if tuple(x.shape[-len(last_shape):]) != last_shape:
        raise ValueError(f"{name}: expected trailing shape {last_shape}, got shape {x.shape}")
    return x.reshape(-1, *last_shape)


def inverse_sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def rgb_to_sh_np(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb - 0.5) / C0


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), EPS)
    return q


def rotmat_to_quat_wxyz(Rm: np.ndarray) -> np.ndarray:
    # scipy returns xyzw
    q_xyzw = SciRot.from_matrix(Rm).as_quat().astype(np.float32)
    return normalize_quat_wxyz(np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1))


def cov_to_scale_rot(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov, dtype=np.float32)
    if cov.ndim == 2 and cov.shape[1] == 9:
        cov = cov.reshape(-1, 3, 3)
    elif cov.ndim == 2 and cov.shape[1] == 6:
        c = np.zeros((cov.shape[0], 3, 3), dtype=np.float32)
        # convention: xx, xy, xz, yy, yz, zz
        c[:, 0, 0] = cov[:, 0]
        c[:, 0, 1] = c[:, 1, 0] = cov[:, 1]
        c[:, 0, 2] = c[:, 2, 0] = cov[:, 2]
        c[:, 1, 1] = cov[:, 3]
        c[:, 1, 2] = c[:, 2, 1] = cov[:, 4]
        c[:, 2, 2] = cov[:, 5]
        cov = c
    elif cov.ndim >= 3 and cov.shape[-2:] == (3, 3):
        cov = cov.reshape(-1, 3, 3)
    else:
        raise ValueError(f"Cannot parse covariance shape {cov.shape}")

    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-10, None)
    scales = np.sqrt(eigvals).astype(np.float32)
    Rm = eigvecs.astype(np.float32)
    # enforce right-handed rotation
    det = np.linalg.det(Rm)
    bad = det < 0
    if np.any(bad):
        Rm[bad, :, 0] *= -1.0
    rots = rotmat_to_quat_wxyz(Rm)
    return scales, rots


# ------------------------------ field finding ------------------------------

@dataclass
class GaussianArrays:
    xyz: np.ndarray
    opacity_internal: np.ndarray
    scale_internal: np.ndarray
    rot_wxyz: np.ndarray
    f_dc: np.ndarray
    f_rest: np.ndarray


def print_packet_structure(packet: Any, max_items: int = 200) -> None:
    print("\n[inspect] Tensor-like fields:")
    rows = []
    for path, t in walk_tensors(packet):
        rows.append((path, tuple(t.shape), str(t.dtype)))
    rows = sorted(rows, key=lambda r: r[0])
    for i, (path, shape, dtype) in enumerate(rows[:max_items]):
        print(f"  {i:03d}  {path:80s} shape={shape} dtype={dtype}")
    if len(rows) > max_items:
        print(f"  ... {len(rows) - max_items} more")


def choose_tensor(packet: Any, explicit: Optional[str], candidates: Sequence[str], shape_test, label: str) -> Tuple[str, torch.Tensor]:
    if explicit:
        val = get_by_path(packet, explicit)
        if not isinstance(val, torch.Tensor):
            val = torch.as_tensor(val)
        return explicit, val
    scored: List[Tuple[int, str, torch.Tensor]] = []
    for path, t in walk_tensors(packet):
        p = path.lower()
        if any(bad in p for bad in ['intrinsic', 'extrinsic', 'camera', 'pose', 'image', 'rgb_image', 'depth']):
            # allow rgb/color candidates below, but avoid camera tensors for xyz/scale/cov/rot/opacity
            if label not in {'color', 'sh'}:
                continue
        try:
            if not shape_test(t):
                continue
        except Exception:
            continue
        score = 0
        for c in candidates:
            if c in p:
                score += 10 + len(c)
        # prefer shorter/natural paths when tied
        score -= path.count('.')
        if score > 0:
            scored.append((score, path, t))
    if not scored:
        raise KeyError(f"Could not infer {label} tensor. Use --{label}_key or run --inspect_only.")
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1], scored[0][2]


def extract_gaussians_from_packet(packet: Any, args: argparse.Namespace) -> GaussianArrays:
    xyz_path, xyz_t = choose_tensor(
        packet, args.xyz_key,
        ['means', 'mean', 'xyz', 'position', 'positions', 'gaussian_means'],
        lambda t: t.ndim >= 2 and int(t.shape[-1]) == 3,
        'xyz'
    )
    xyz = flatten_last(tensor_to_numpy(xyz_t), (3,), xyz_path).astype(np.float32)
    N = xyz.shape[0]

    # opacity
    op_path, op_t = choose_tensor(
        packet, args.opacity_key,
        ['opacities', 'opacity', 'alpha', 'alphas'],
        lambda t: t.numel() >= N and (int(t.shape[-1]) in [1] or t.ndim >= 1),
        'opacity'
    )
    opacity = tensor_to_numpy(op_t).reshape(-1, 1).astype(np.float32)
    if opacity.shape[0] != N:
        # common case: (..., 1) but includes batch/view dims matching xyz before flatten
        opacity = opacity[:N]
    if args.opacity_mode == 'activated' or (args.opacity_mode == 'auto' and opacity.min() >= -EPS and opacity.max() <= 1.0 + EPS):
        opacity_internal = inverse_sigmoid_np(opacity)
    else:
        opacity_internal = opacity

    # scale + rotation, or covariance fallback
    scale_internal = None
    rot_wxyz = None

    if args.scale_key or args.rotation_key:
        sc_path, sc_t = choose_tensor(
            packet, args.scale_key,
            ['scales', 'scale', 'scaling'],
            lambda t: t.ndim >= 1 and int(t.shape[-1]) in [1, 3],
            'scale'
        )
        scales = tensor_to_numpy(sc_t).reshape(-1, int(sc_t.shape[-1])).astype(np.float32)
        if scales.shape[0] != N:
            scales = scales[:N]
        if scales.shape[1] == 1:
            scales = np.repeat(scales, 3, axis=1)
        if args.scale_mode == 'activated' or (args.scale_mode == 'auto' and np.all(scales > 0)):
            scale_internal = np.log(np.clip(scales, 1e-8, None))
        else:
            scale_internal = scales

        rot_path, rot_t = choose_tensor(
            packet, args.rotation_key,
            ['rotations', 'rotation', 'quaternion', 'quaternions', 'quat', 'rot'],
            lambda t: t.ndim >= 2 and int(t.shape[-1]) == 4,
            'rotation'
        )
        rots = tensor_to_numpy(rot_t).reshape(-1, 4).astype(np.float32)
        if rots.shape[0] != N:
            rots = rots[:N]
        if args.rotation_format == 'xyzw':
            rot_wxyz = np.concatenate([rots[:, 3:4], rots[:, 0:3]], axis=1)
        else:
            rot_wxyz = rots
        rot_wxyz = normalize_quat_wxyz(rot_wxyz)
    else:
        try:
            cov_path, cov_t = choose_tensor(
                packet, args.covariance_key,
                ['covariances', 'covariance', 'cov3d', 'cov'],
                lambda t: (t.ndim >= 3 and tuple(t.shape[-2:]) == (3, 3)) or (t.ndim >= 2 and int(t.shape[-1]) in [6, 9]),
                'covariance'
            )
            cov = tensor_to_numpy(cov_t)
            scales, rot_wxyz = cov_to_scale_rot(cov)
            if scales.shape[0] != N:
                scales = scales[:N]
                rot_wxyz = rot_wxyz[:N]
            scale_internal = np.log(np.clip(scales, 1e-8, None))
            sc_path = cov_path + ' -> eig(scale)'
            rot_path = cov_path + ' -> eig(rotation)'
        except Exception as exc:
            if args.allow_identity_covariance:
                print(f"[warn] covariance/scale/rotation not found. Using isotropic scale={args.default_scale}: {exc}")
                scale_internal = np.full((N, 3), math.log(args.default_scale), dtype=np.float32)
                rot_wxyz = np.zeros((N, 4), dtype=np.float32)
                rot_wxyz[:, 0] = 1.0
                sc_path = 'default'
                rot_path = 'identity'
            else:
                raise

    # color / SH
    f_dc = None
    f_rest = np.zeros((N, 3 * ((args.sh_degree + 1) ** 2 - 1)), dtype=np.float32)
    if args.feature_dc_key:
        fdc = tensor_to_numpy(get_by_path(packet, args.feature_dc_key)).astype(np.float32)
        f_dc = fdc.reshape(-1, 3)[:N]
        color_path = args.feature_dc_key
    else:
        # Try harmonics / SH first if present, otherwise RGB colors.
        try:
            sh_path, sh_t = choose_tensor(
                packet, args.sh_key,
                ['harmonics', 'spherical_harmonics', 'sh', 'features'],
                lambda t: t.ndim >= 3 and (int(t.shape[-1]) == 3 or int(t.shape[-2]) == 3),
                'sh'
            )
            sh = tensor_to_numpy(sh_t).astype(np.float32)
            # accepted layouts: (..., K, 3) or (..., 3, K)
            if sh.shape[-1] == 3:
                sh = sh.reshape(-1, sh.shape[-2], 3)  # N,K,3
                if sh.shape[0] != N:
                    sh = sh[:N]
                f_dc = sh[:, 0, :]
                rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(sh.shape[0], -1)
            elif sh.shape[-2] == 3:
                sh = sh.reshape(-1, 3, sh.shape[-1])  # N,3,K
                if sh.shape[0] != N:
                    sh = sh[:N]
                f_dc = sh[:, :, 0]
                rest = sh[:, :, 1:].reshape(sh.shape[0], -1)
            else:
                raise ValueError(f"Unsupported SH shape {sh.shape}")
            if rest.shape[1] > 0:
                f_rest[:, :min(rest.shape[1], f_rest.shape[1])] = rest[:, :min(rest.shape[1], f_rest.shape[1])]
            color_path = sh_path
        except Exception:
            col_path, col_t = choose_tensor(
                packet, args.color_key,
                ['colors', 'color', 'rgb', 'rgbs'],
                lambda t: t.ndim >= 2 and int(t.shape[-1]) == 3,
                'color'
            )
            rgb = flatten_last(tensor_to_numpy(col_t), (3,), col_path).astype(np.float32)
            if rgb.shape[0] != N:
                rgb = rgb[:N]
            if rgb.max() > 2.0:
                rgb = rgb / 255.0
            if args.color_mode == 'rgb':
                f_dc = rgb_to_sh_np(rgb)
            elif args.color_mode == 'sh':
                f_dc = rgb
            else:
                # auto: colors/rgb path -> RGB; features/harmonics path handled above as SH
                f_dc = rgb_to_sh_np(rgb)
            color_path = col_path

    assert f_dc is not None
    f_dc = f_dc.astype(np.float32)
    if f_dc.shape[0] != N:
        f_dc = f_dc[:N]

    print(f"[packet fields] xyz={xyz_path}, opacity={op_path}, scale={sc_path}, rot={rot_path}, color/sh={color_path}, N={N}")
    return GaussianArrays(
        xyz=xyz.astype(np.float32),
        opacity_internal=opacity_internal.astype(np.float32),
        scale_internal=scale_internal.astype(np.float32),
        rot_wxyz=rot_wxyz.astype(np.float32),
        f_dc=f_dc.astype(np.float32),
        f_rest=f_rest.astype(np.float32),
    )


def concat_gaussians(parts: List[GaussianArrays]) -> GaussianArrays:
    return GaussianArrays(
        xyz=np.concatenate([p.xyz for p in parts], axis=0),
        opacity_internal=np.concatenate([p.opacity_internal for p in parts], axis=0),
        scale_internal=np.concatenate([p.scale_internal for p in parts], axis=0),
        rot_wxyz=np.concatenate([p.rot_wxyz for p in parts], axis=0),
        f_dc=np.concatenate([p.f_dc for p in parts], axis=0),
        f_rest=np.concatenate([p.f_rest for p in parts], axis=0),
    )


# ------------------------------- PLY output -------------------------------

def write_full_3dgs_ply(path: Path, g: GaussianArrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    N = g.xyz.shape[0]
    normals = np.zeros_like(g.xyz, dtype=np.float32)

    attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attrs += [f'f_dc_{i}' for i in range(3)]
    attrs += [f'f_rest_{i}' for i in range(g.f_rest.shape[1])]
    attrs += ['opacity']
    attrs += [f'scale_{i}' for i in range(3)]
    attrs += [f'rot_{i}' for i in range(4)]
    dtype = [(a, 'f4') for a in attrs]

    data = np.concatenate([
        g.xyz.astype(np.float32),
        normals,
        g.f_dc.astype(np.float32),
        g.f_rest.astype(np.float32),
        g.opacity_internal.astype(np.float32),
        g.scale_internal.astype(np.float32),
        g.rot_wxyz.astype(np.float32),
    ], axis=1)
    out = np.empty(N, dtype=dtype)
    out[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(out, 'vertex')], text=False).write(str(path))
    print(f"[write] full 3DGS Gaussian PLY: {path} ({N:,} gaussians)")


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return np.clip(f_dc * C0 + 0.5, 0.0, 1.0)


def write_pointcloud_ply(path: Path, xyz: np.ndarray, f_dc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (sh_dc_to_rgb(f_dc) * 255.0).clip(0, 255).astype(np.uint8)
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    out = np.empty(xyz.shape[0], dtype=dtype)
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out['nx'], out['ny'], out['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    out['red'], out['green'], out['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(out, 'vertex')], text=False).write(str(path))
    print(f"[write] point-cloud PLY for Scene init: {path}")


# ----------------------------- camera export -----------------------------

def read_pose_file(path: Path, pose_format: str) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    lines = path.read_text().splitlines()
    seq_idx = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if pose_format == 'tartanair_twc':
            if len(parts) < 7:
                continue
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[:7])
            idx = seq_idx
            seq_idx += 1
        elif pose_format == 'tum_twc':
            if len(parts) < 8:
                continue
            idx = seq_idx
            seq_idx += 1
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:8])
        else:
            raise ValueError(f"Unknown pose_format: {pose_format}")
        R_wc = SciRot.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        Twc = np.eye(4, dtype=np.float32)
        Twc[:3, :3] = R_wc
        Twc[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        poses[idx] = Twc
    return poses


def cv_twc_to_nerf_transform(Twc_cv: np.ndarray, rotation_mode: str) -> np.ndarray:
    Twc_cv = Twc_cv.copy().astype(np.float32)
    if rotation_mode == 'w2c':
        # Input file stored camera orientation as world-to-camera; invert rotation only while keeping t as camera center.
        Twc_cv[:3, :3] = Twc_cv[:3, :3].T
    elif rotation_mode != 'c2w':
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return Twc_cv @ cv_to_gl


def matrix_to_nerf_transform(M: np.ndarray, convention: str) -> np.ndarray:
    """Convert a packet camera matrix to the OpenGL/NeRF c2w transform expected by transforms_*.json.

    Supported conventions:
      - twc_cv:     camera-to-world, OpenCV camera axes: x right, y down, z forward
      - tcw_cv:     world-to-camera, OpenCV camera axes
      - twc_opengl: camera-to-world, OpenGL/NeRF axes: x right, y up, z backward
      - tcw_opengl: world-to-camera, OpenGL/NeRF axes
    """
    M = np.asarray(M, dtype=np.float32)
    if M.shape != (4, 4):
        raise ValueError(f"Expected 4x4 camera matrix, got {M.shape}")
    if convention == 'twc_cv':
        return M @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    if convention == 'tcw_cv':
        return np.linalg.inv(M) @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    if convention in {'twc_opengl', 'twc_nerf'}:
        return M
    if convention in {'tcw_opengl', 'tcw_nerf'}:
        return np.linalg.inv(M)
    raise ValueError(f"Unknown packet_camera_convention: {convention}")


def choose_camera_matrix(packet: Any, explicit: Optional[str]) -> Tuple[str, torch.Tensor]:
    """Find a candidate 4x4 camera matrix tensor inside a packet."""
    if explicit:
        val = get_by_path(packet, explicit)
        if not isinstance(val, torch.Tensor):
            val = torch.as_tensor(val)
        return explicit, val

    candidates = [
        'target.extrinsics', 'target.extrinsic', 'target.pose', 'target.c2w', 'target.twc',
        'extrinsics', 'extrinsic', 'camera_to_world', 'c2w', 'cam2world', 'world_from_cam', 'twc',
        'w2c', 'world_to_camera', 'camera_from_world', 'pose', 'poses', 'camera_pose', 'camera_poses',
    ]
    scored: List[Tuple[int, str, torch.Tensor]] = []
    for path, t in walk_tensors(packet):
        if t.ndim < 2 or tuple(t.shape[-2:]) != (4, 4):
            continue
        p = path.lower()
        # Exclude unrelated transforms if obvious.
        if any(bad in p for bad in ['augmentation', 'crop', 'resize', 'normalize']):
            continue
        score = 0
        for c in candidates:
            if c in p:
                score += 10 + len(c)
        if 'target' in p:
            score += 8
        if 'context' in p:
            score -= 4
        if 'extrinsic' in p:
            score += 6
        if 'pose' in p:
            score += 4
        score -= path.count('.')
        if score > 0:
            scored.append((score, path, t))
    if not scored:
        raise KeyError("Could not infer packet camera matrix. Use --packet_camera_key or run --inspect_only.")
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1], scored[0][2]


def select_matrix_from_tensor(t: torch.Tensor, view_index: int) -> np.ndarray:
    arr = tensor_to_numpy(t).astype(np.float32)
    if tuple(arr.shape[-2:]) != (4, 4):
        raise ValueError(f"Camera tensor must end with 4x4, got {arr.shape}")
    mats = arr.reshape(-1, 4, 4)
    if mats.shape[0] == 0:
        raise ValueError("Empty camera matrix tensor")
    idx = view_index
    if idx < 0:
        idx = mats.shape[0] + idx
    if idx < 0 or idx >= mats.shape[0]:
        raise IndexError(f"packet_camera_view_index {view_index} out of range for {mats.shape[0]} matrices")
    return mats[idx]


def packet_camera_to_frame(packet: Any, args: argparse.Namespace, image_file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cam_path, cam_t = choose_camera_matrix(packet, args.packet_camera_key)
    M = select_matrix_from_tensor(cam_t, args.packet_camera_view_index)
    T_nerf = matrix_to_nerf_transform(M, args.packet_camera_convention)
    frame = {
        'file_path': image_file_path,
        'transform_matrix': T_nerf.tolist(),
    }
    info = {
        'camera_key': cam_path,
        'camera_tensor_shape': list(cam_t.shape),
        'camera_view_index': args.packet_camera_view_index,
        'camera_convention': args.packet_camera_convention,
        'camera_center_raw': M[:3, 3].astype(float).tolist(),
    }
    return frame, info


def copy_images_and_write_transforms(args: argparse.Namespace, frame_indices: List[int], output_scene: Path, packet_files: Optional[List[Path]] = None) -> Dict[str, Any]:
    image_out = output_scene / 'images'
    image_out.mkdir(parents=True, exist_ok=True)

    if args.camera_source == 'pose_file':
        if not args.pose_file:
            raise ValueError('--pose_file is required when --camera_source pose_file')
        poses = read_pose_file(Path(args.pose_file), args.pose_format)
    else:
        poses = None
        if packet_files is None:
            raise ValueError('packet_files must be provided when --camera_source packet')
        if len(packet_files) != len(frame_indices):
            raise ValueError('packet_files and frame_indices length mismatch')

    fovx = 2.0 * math.atan(args.width / (2.0 * args.fx))
    frames = []
    missing = []
    camera_infos = []

    for local_i, idx in enumerate(frame_indices):
        src = Path(args.image_dir) / args.image_pattern.format(index=idx, local_index=local_i)
        if not src.exists():
            missing.append(str(src))
            continue
        with Image.open(src) as im:
            w, h = im.size
        if args.strict_image_size and (w != args.width or h != args.height):
            raise ValueError(f"Image size mismatch for {src}: got {w}x{h}, expected {args.width}x{args.height}")

        name = f"{idx:06d}.png"
        dst = image_out / name
        if args.copy_images:
            shutil.copyfile(src, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(os.path.abspath(src), dst)

        image_file_path = f"images/{idx:06d}"

        if args.camera_source == 'pose_file':
            assert poses is not None
            if idx not in poses:
                raise KeyError(f"Pose index {idx} not found in {args.pose_file}")
            T_nerf = cv_twc_to_nerf_transform(poses[idx], args.pose_rotation_mode)
            frames.append({
                'file_path': image_file_path,
                'transform_matrix': T_nerf.tolist(),
            })
            camera_infos.append({
                'frame_index': idx,
                'camera_source': 'pose_file',
                'camera_key': args.pose_file,
                'camera_convention': f'{args.pose_format}/{args.pose_rotation_mode}',
                'camera_center_raw': poses[idx][:3, 3].astype(float).tolist(),
            })
        elif args.camera_source == 'packet':
            packet = safe_torch_load(packet_files[local_i])
            frame, info = packet_camera_to_frame(packet, args, image_file_path)
            frames.append(frame)
            info.update({
                'frame_index': idx,
                'packet_file': str(packet_files[local_i]),
                'camera_source': 'packet',
            })
            camera_infos.append(info)
        else:
            raise ValueError(f"Unknown camera_source: {args.camera_source}")

    if missing:
        raise FileNotFoundError("Missing images:\n" + "\n".join(missing[:20]) + ("\n..." if len(missing) > 20 else ""))

    if args.test_same_as_train:
        train_frames = frames
        test_frames = frames
    elif args.test_every > 0:
        test_frames = [f for i, f in enumerate(frames) if i % args.test_every == 0]
        train_frames = [f for i, f in enumerate(frames) if i % args.test_every != 0]
        if not train_frames:
            train_frames = frames
    else:
        train_frames = frames
        test_frames = []

    meta_train = {'camera_angle_x': fovx, 'frames': train_frames}
    meta_test = {'camera_angle_x': fovx, 'frames': test_frames}
    (output_scene / 'transforms_train.json').write_text(json.dumps(meta_train, indent=2))
    (output_scene / 'transforms_test.json').write_text(json.dumps(meta_test, indent=2))
    print(f"[write] transforms_train.json: {len(train_frames)} frames")
    print(f"[write] transforms_test.json:  {len(test_frames)} frames")
    if camera_infos:
        first = camera_infos[0]
        print(f"[camera] source={args.camera_source}, first_key={first.get('camera_key')}, first_center={first.get('camera_center_raw')}")
    return {
        'num_train_frames': len(train_frames),
        'num_test_frames': len(test_frames),
        'camera_angle_x': fovx,
        'camera_source': args.camera_source,
        'camera_infos': camera_infos,
    }


# ----------------------------------- main -----------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--packet_dir', required=True)
    p.add_argument('--packet_ranges', required=True, help='e.g. 0-19 or 0-9,20-29. These index sorted packet files.')
    p.add_argument('--dataset_start_index', type=int, default=0, help='image/pose index offset: frame_index = dataset_start_index + packet_index')
    p.add_argument('--output_scene', required=True)

    # image/camera
    p.add_argument('--image_dir', required=True)
    p.add_argument('--image_pattern', default='{index:06d}.png')
    p.add_argument('--pose_file', default=None)
    p.add_argument('--pose_format', choices=['tartanair_twc', 'tum_twc'], default='tartanair_twc')
    p.add_argument('--pose_rotation_mode', choices=['c2w', 'w2c'], default='c2w')
    p.add_argument('--fx', type=float, required=True)
    p.add_argument('--fy', type=float, default=None)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--copy_images', action='store_true', help='copy images instead of symlinking')
    p.add_argument('--strict_image_size', action='store_true')
    p.add_argument('--test_every', type=int, default=0, help='0 = no held-out test; N = every Nth selected frame as test')
    p.add_argument('--test_same_as_train', action='store_true')
    p.add_argument('--camera_source', choices=['pose_file', 'packet'], default='pose_file',
                   help='pose_file = use external pose file; packet = use camera matrix stored in each packet, recommended for fused-packet baseline')
    p.add_argument('--packet_camera_key', default=None,
                   help='Explicit tensor path for packet camera matrix, e.g. target.extrinsics or cameras.extrinsics')
    p.add_argument('--packet_camera_view_index', type=int, default=-1,
                   help='If the packet camera tensor contains multiple 4x4 matrices, select this flattened index. -1 usually means target/last view.')
    p.add_argument('--packet_camera_convention', choices=['twc_cv', 'tcw_cv', 'twc_opengl', 'tcw_opengl', 'twc_nerf', 'tcw_nerf'], default='twc_cv',
                   help='Convention of packet camera matrix. Most MVSplat/ReSplat-style tensors use twc_cv/camera-to-world.')

    # Gaussian conversion
    p.add_argument('--sh_degree', type=int, default=3)
    p.add_argument('--opacity_mode', choices=['auto', 'activated', 'internal'], default='auto')
    p.add_argument('--scale_mode', choices=['auto', 'activated', 'internal'], default='auto')
    p.add_argument('--rotation_format', choices=['wxyz', 'xyzw'], default='wxyz')
    p.add_argument('--color_mode', choices=['auto', 'rgb', 'sh'], default='auto')
    p.add_argument('--allow_identity_covariance', action='store_true')
    p.add_argument('--default_scale', type=float, default=0.01)

    # explicit packet field paths
    p.add_argument('--xyz_key', default=None)
    p.add_argument('--opacity_key', default=None)
    p.add_argument('--scale_key', default=None)
    p.add_argument('--rotation_key', default=None)
    p.add_argument('--covariance_key', default=None)
    p.add_argument('--color_key', default=None)
    p.add_argument('--sh_key', default=None)
    p.add_argument('--feature_dc_key', default=None)

    p.add_argument('--inspect_only', action='store_true')
    p.add_argument('--max_inspect_items', type=int, default=200)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.fy is None:
        args.fy = args.fx
    if args.cx is None:
        args.cx = args.width / 2.0
    if args.cy is None:
        args.cy = args.height / 2.0

    packet_dir = Path(args.packet_dir)
    output_scene = Path(args.output_scene)
    output_scene.mkdir(parents=True, exist_ok=True)

    packet_files_all = sorted_packet_files(packet_dir)
    packet_indices = parse_int_ranges(args.packet_ranges)
    if not packet_indices:
        raise ValueError('--packet_ranges produced no packet indices')
    for pi in packet_indices:
        if pi < 0 or pi >= len(packet_files_all):
            raise IndexError(f"packet index {pi} out of range [0,{len(packet_files_all)-1}]")

    first_packet = safe_torch_load(packet_files_all[packet_indices[0]])
    if args.inspect_only:
        print(f"[inspect] packet file: {packet_files_all[packet_indices[0]]}")
        print_packet_structure(first_packet, args.max_inspect_items)
        return

    parts: List[GaussianArrays] = []
    selected_packet_paths = []
    for pi in packet_indices:
        path = packet_files_all[pi]
        selected_packet_paths.append(str(path))
        print(f"[load] packet[{pi}] {path.name}")
        packet = safe_torch_load(path)
        parts.append(extract_gaussians_from_packet(packet, args))

    fused = concat_gaussians(parts)
    print(f"[fused] total gaussians: {fused.xyz.shape[0]:,}")

    initial_ply = output_scene / 'initial_fused_3dgs.ply'
    points_ply = output_scene / 'points3d.ply'
    write_full_3dgs_ply(initial_ply, fused)
    write_pointcloud_ply(points_ply, fused.xyz, fused.f_dc)

    frame_indices = [args.dataset_start_index + pi for pi in packet_indices]
    cam_summary = copy_images_and_write_transforms(args, frame_indices, output_scene, [packet_files_all[pi] for pi in packet_indices])

    summary = {
        'packet_dir': str(packet_dir),
        'packet_ranges': args.packet_ranges,
        'packet_indices': packet_indices,
        'frame_indices': frame_indices,
        'selected_packet_paths': selected_packet_paths,
        'num_gaussians': int(fused.xyz.shape[0]),
        'initial_ply': str(initial_ply),
        'points3d_ply': str(points_ply),
        'output_scene': str(output_scene),
        'sh_degree': args.sh_degree,
        'camera': {
            'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy,
            'width': args.width, 'height': args.height,
        },
        **cam_summary,
    }
    (output_scene / 'prepare_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"[done] scene prepared at {output_scene}")
    print(f"[next] put train_fused_3dgs_baseline.py in the original 3DGS repo and run with --source_path {output_scene} --initial_ply {initial_ply}")


if __name__ == '__main__':
    main()
