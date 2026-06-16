#!/usr/bin/env python3
"""
Prepare an original graphdeco-inria/gaussian-splatting compatible scene from
ReSplat/MVSplat-style Gaussian packet .pt files, with an optional strict
train/test split.

Main strict-split use case:
  - Use only train-frame packets to write initial_fused_3dgs.ply.
  - Write transforms_train.json from train cameras.
  - Write transforms_test.json from held-out test cameras.
  - Test frames do not contribute Gaussian packets to the initial map.

This script is intentionally conservative: for strict held-out evaluation, use
--camera_source pose_file so test cameras can be written without generating or
loading test-frame packets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from plyfile import PlyData, PlyElement
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs plyfile. Install it: pip install plyfile") from exc

try:
    from scipy.spatial.transform import Rotation as SciRot
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs scipy. Install it: pip install scipy") from exc

C0 = 0.28209479177387814
EPS = 1e-6


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

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
    q_xyzw = SciRot.from_matrix(Rm).as_quat().astype(np.float32)
    return normalize_quat_wxyz(np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1))


def cov_to_scale_rot(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov, dtype=np.float32)
    if cov.ndim == 2 and cov.shape[1] == 9:
        cov = cov.reshape(-1, 3, 3)
    elif cov.ndim == 2 and cov.shape[1] == 6:
        c = np.zeros((cov.shape[0], 3, 3), dtype=np.float32)
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
    det = np.linalg.det(Rm)
    bad = det < 0
    if np.any(bad):
        Rm[bad, :, 0] *= -1.0
    rots = rotmat_to_quat_wxyz(Rm)
    return scales, rots


# -----------------------------------------------------------------------------
# Gaussian extraction
# -----------------------------------------------------------------------------

@dataclass
class GaussianArrays:
    xyz: np.ndarray
    opacity_internal: np.ndarray
    scale_internal: np.ndarray
    rot_wxyz: np.ndarray
    f_dc: np.ndarray
    f_rest: np.ndarray


def print_packet_structure(packet: Any, max_items: int = 200) -> None:
    rows = []
    for path, t in walk_tensors(packet):
        rows.append((path, tuple(t.shape), str(t.dtype)))
    rows = sorted(rows, key=lambda r: r[0])
    print("\n[inspect] Tensor-like fields:")
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

    op_path, op_t = choose_tensor(
        packet, args.opacity_key,
        ['opacities', 'opacity', 'alpha', 'alphas'],
        lambda t: t.numel() >= N and (int(t.shape[-1]) in [1] or t.ndim >= 1),
        'opacity'
    )
    opacity = tensor_to_numpy(op_t).reshape(-1, 1).astype(np.float32)
    if opacity.shape[0] != N:
        opacity = opacity[:N]
    if args.opacity_mode == 'activated' or (args.opacity_mode == 'auto' and opacity.min() >= -EPS and opacity.max() <= 1.0 + EPS):
        opacity_internal = inverse_sigmoid_np(opacity)
    else:
        opacity_internal = opacity

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

    f_dc = None
    f_rest = np.zeros((N, 3 * ((args.sh_degree + 1) ** 2 - 1)), dtype=np.float32)
    if args.feature_dc_key:
        fdc = tensor_to_numpy(get_by_path(packet, args.feature_dc_key)).astype(np.float32)
        f_dc = fdc.reshape(-1, 3)[:N]
        color_path = args.feature_dc_key
    else:
        try:
            sh_path, sh_t = choose_tensor(
                packet, args.sh_key,
                ['harmonics', 'spherical_harmonics', 'sh', 'features'],
                lambda t: t.ndim >= 3 and (int(t.shape[-1]) == 3 or int(t.shape[-2]) == 3),
                'sh'
            )
            sh = tensor_to_numpy(sh_t).astype(np.float32)
            if sh.shape[-1] == 3:
                sh = sh.reshape(-1, sh.shape[-2], 3)
                if sh.shape[0] != N:
                    sh = sh[:N]
                f_dc = sh[:, 0, :]
                rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(sh.shape[0], -1)
            elif sh.shape[-2] == 3:
                sh = sh.reshape(-1, 3, sh.shape[-1])
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
            f_dc = rgb_to_sh_np(rgb) if args.color_mode in {'auto', 'rgb'} else rgb
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
    if not parts:
        raise ValueError("No Gaussian parts to concatenate")
    return GaussianArrays(
        xyz=np.concatenate([p.xyz for p in parts], axis=0),
        opacity_internal=np.concatenate([p.opacity_internal for p in parts], axis=0),
        scale_internal=np.concatenate([p.scale_internal for p in parts], axis=0),
        rot_wxyz=np.concatenate([p.rot_wxyz for p in parts], axis=0),
        f_dc=np.concatenate([p.f_dc for p in parts], axis=0),
        f_rest=np.concatenate([p.f_rest for p in parts], axis=0),
    )


# -----------------------------------------------------------------------------
# PLY output
# -----------------------------------------------------------------------------

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
        g.xyz.astype(np.float32), normals,
        g.f_dc.astype(np.float32), g.f_rest.astype(np.float32),
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
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    out = np.empty(xyz.shape[0], dtype=dtype)
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out['nx'], out['ny'], out['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    out['red'], out['green'], out['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(out, 'vertex')], text=False).write(str(path))
    print(f"[write] point-cloud PLY for Scene init: {path}")


# -----------------------------------------------------------------------------
# Pose / camera utilities
# -----------------------------------------------------------------------------

def tartan_from_cv_matrix_np(dtype=np.float64) -> np.ndarray:
    """T_tartanCam_from_cvCam. For TartanAir pose7: Twc_cv = Twc_tartan @ T_tartanCam_from_cvCam."""
    T = np.eye(4, dtype=dtype)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)
    return T


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {q}")
    q = q / n
    if q[-1] < 0:
        q = -q
    return q


def quat_xyzw_to_rotmat(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(np.asarray(q, dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def make_T_c2w_from_pose7(values: Sequence[float], quat_order: str = 'xyzw') -> np.ndarray:
    if len(values) != 7:
        raise ValueError(f"pose7 must have 7 values, got {len(values)}")
    tx, ty, tz = values[:3]
    if quat_order == 'xyzw':
        q = values[3:7]
    elif quat_order == 'wxyz':
        qw, qx, qy, qz = values[3:7]
        q = [qx, qy, qz, qw]
    else:
        raise ValueError(f"Unknown quat_order: {quat_order}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(q)
    T[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float64)
    return T


def invert_se3_np(T: np.ndarray) -> np.ndarray:
    single = T.ndim == 2
    if single:
        T = T[None]
    R = T[:, :3, :3]
    t = T[:, :3, 3:4]
    Rt = np.transpose(R, (0, 2, 1))
    out = np.tile(np.eye(4, dtype=T.dtype), (T.shape[0], 1, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3:4] = -np.matmul(Rt, t)
    return out[0] if single else out


def read_numeric_pose_file(path: Path) -> Tuple[Optional[List[str]], np.ndarray, str]:
    rows: List[List[float]] = []
    stamps: List[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.replace(',', ' ').split()
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(nums)
        stamps.append(parts[0])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    ncols = len(rows[0])
    if any(len(r) != ncols for r in rows):
        raise ValueError(f"Inconsistent column counts in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    if ncols == 8:
        return stamps, arr[:, 1:8], 'tum_pose7'
    if ncols == 7:
        return None, arr, 'pose7'
    if ncols == 13:
        return stamps, arr[:, 1:13].reshape(-1, 3, 4), 'matrix34'
    if ncols == 12:
        return None, arr.reshape(-1, 3, 4), 'matrix34'
    if ncols == 17:
        return stamps, arr[:, 1:17].reshape(-1, 4, 4), 'matrix44'
    if ncols == 16:
        return None, arr.reshape(-1, 4, 4), 'matrix44'
    raise ValueError(f"Unsupported pose format in {path}: {ncols} numeric columns")


def load_gt_trajectory(pose_file: Path, convention: str, quat_order: str, matrix_convention: str) -> np.ndarray:
    _, data, fmt = read_numeric_pose_file(pose_file)
    if fmt in {'pose7', 'tum_pose7'}:
        T_pose = np.stack([make_T_c2w_from_pose7(row, quat_order=quat_order) for row in data], axis=0)
        if convention == 'opencv_c2w':
            return T_pose.astype(np.float32)
        if convention == 'resplat_tartanair_pose':
            return np.matmul(T_pose, tartan_from_cv_matrix_np()[None]).astype(np.float32)
        raise ValueError(f"Unsupported convention for pose7: {convention}")
    if fmt == 'matrix34':
        T = np.tile(np.eye(4, dtype=np.float64), (data.shape[0], 1, 1))
        T[:, :3, :4] = data
    else:
        T = data.astype(np.float64)
    if matrix_convention == 'w2c':
        T = invert_se3_np(T)
    elif matrix_convention != 'c2w':
        raise ValueError(f"Unknown matrix_convention: {matrix_convention}")
    if convention == 'opencv_c2w':
        return T.astype(np.float32)
    if convention == 'resplat_tartanair_pose':
        return np.matmul(T, tartan_from_cv_matrix_np()[None]).astype(np.float32)
    raise ValueError(f"Unsupported convention: {convention}")


def cv_twc_to_nerf_transform(Twc_cv: np.ndarray) -> np.ndarray:
    return Twc_cv.astype(np.float32) @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def matrix_to_nerf_transform(M: np.ndarray, convention: str) -> np.ndarray:
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
        raise KeyError("Could not infer packet camera matrix. Use --packet_camera_key or --camera_source pose_file.")
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1], scored[0][2]


def select_matrix_from_tensor(t: torch.Tensor, view_index: int) -> np.ndarray:
    arr = tensor_to_numpy(t).astype(np.float32)
    if tuple(arr.shape[-2:]) != (4, 4):
        raise ValueError(f"Camera tensor must end with 4x4, got {arr.shape}")
    mats = arr.reshape(-1, 4, 4)
    idx = view_index
    if idx < 0:
        idx = mats.shape[0] + idx
    if idx < 0 or idx >= mats.shape[0]:
        raise IndexError(f"packet_camera_view_index {view_index} out of range for {mats.shape[0]} matrices")
    return mats[idx]


# -----------------------------------------------------------------------------
# Split and transforms
# -----------------------------------------------------------------------------

def is_test_frame(local_pos: int, packet_index: int, frame_index: int, args: argparse.Namespace) -> bool:
    if not args.internal_split:
        return False
    if args.split_index_mode == 'local_index':
        v = local_pos
    elif args.split_index_mode == 'packet_index':
        v = packet_index
    elif args.split_index_mode == 'frame_index':
        v = frame_index
    else:
        raise ValueError(f"Unknown split_index_mode: {args.split_index_mode}")
    return (v - args.split_offset) % args.split_every == 0


def make_split(packet_indices_all: List[int], args: argparse.Namespace) -> Tuple[List[int], List[int], List[int], List[int]]:
    train_packet_indices: List[int] = []
    test_packet_indices: List[int] = []
    train_frame_indices: List[int] = []
    test_frame_indices: List[int] = []
    for local_pos, pi in enumerate(packet_indices_all):
        fi = args.dataset_start_index + pi
        if is_test_frame(local_pos, pi, fi, args):
            test_packet_indices.append(pi)
            test_frame_indices.append(fi)
        else:
            train_packet_indices.append(pi)
            train_frame_indices.append(fi)
    if not args.internal_split:
        train_packet_indices = list(packet_indices_all)
        train_frame_indices = [args.dataset_start_index + pi for pi in packet_indices_all]
        if args.test_same_as_train:
            test_packet_indices = list(packet_indices_all)
            test_frame_indices = list(train_frame_indices)
        elif args.test_every > 0:
            test_packet_indices = [pi for j, pi in enumerate(packet_indices_all) if j % args.test_every == 0]
            test_frame_indices = [args.dataset_start_index + pi for pi in test_packet_indices]
            train_packet_indices = [pi for j, pi in enumerate(packet_indices_all) if j % args.test_every != 0]
            train_frame_indices = [args.dataset_start_index + pi for pi in train_packet_indices]
            if not train_packet_indices:
                train_packet_indices = list(packet_indices_all)
                train_frame_indices = [args.dataset_start_index + pi for pi in packet_indices_all]
        else:
            test_packet_indices = []
            test_frame_indices = []
    if not train_packet_indices:
        raise ValueError("No train packets after split")
    return train_packet_indices, test_packet_indices, train_frame_indices, test_frame_indices


def copy_or_link_image(src: Path, dst: Path, copy_images: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copyfile(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)


def build_frame_record(
    frame_index: int,
    local_packet_index: Optional[int],
    image_file_path: str,
    args: argparse.Namespace,
    poses_cv: Optional[np.ndarray],
    packet_files_all: Optional[List[Path]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if args.camera_source == 'pose_file':
        if poses_cv is None:
            raise ValueError("poses_cv is required for camera_source=pose_file")
        if frame_index < 0 or frame_index >= poses_cv.shape[0]:
            raise IndexError(f"frame_index {frame_index} out of range for pose_file length {poses_cv.shape[0]}")
        T_nerf = cv_twc_to_nerf_transform(poses_cv[frame_index])
        info = {
            'frame_index': frame_index,
            'camera_source': 'pose_file',
            'camera_key': str(args.pose_file),
            'camera_convention': args.gt_convention,
            'camera_center_raw': poses_cv[frame_index][:3, 3].astype(float).tolist(),
        }
        return {'file_path': image_file_path, 'transform_matrix': T_nerf.tolist()}, info

    if args.camera_source == 'packet':
        if packet_files_all is None or local_packet_index is None:
            raise ValueError("packet camera source requires packet file index")
        packet = safe_torch_load(packet_files_all[local_packet_index])
        cam_path, cam_t = choose_camera_matrix(packet, args.packet_camera_key)
        M = select_matrix_from_tensor(cam_t, args.packet_camera_view_index)
        T_nerf = matrix_to_nerf_transform(M, args.packet_camera_convention)
        info = {
            'frame_index': frame_index,
            'packet_index': local_packet_index,
            'packet_file': str(packet_files_all[local_packet_index]),
            'camera_source': 'packet',
            'camera_key': cam_path,
            'camera_tensor_shape': list(cam_t.shape),
            'camera_view_index': args.packet_camera_view_index,
            'camera_convention': args.packet_camera_convention,
            'camera_center_raw': M[:3, 3].astype(float).tolist(),
        }
        return {'file_path': image_file_path, 'transform_matrix': T_nerf.tolist()}, info

    raise ValueError(f"Unknown camera_source: {args.camera_source}")


def write_transforms_and_images(
    args: argparse.Namespace,
    output_scene: Path,
    train_packet_indices: List[int],
    test_packet_indices: List[int],
    train_frame_indices: List[int],
    test_frame_indices: List[int],
    packet_files_all: List[Path],
) -> Dict[str, Any]:
    image_out = output_scene / 'images'
    image_out.mkdir(parents=True, exist_ok=True)

    if args.camera_source == 'pose_file':
        if not args.pose_file:
            raise ValueError('--pose_file is required when --camera_source pose_file')
        poses_cv = load_gt_trajectory(Path(args.pose_file), args.gt_convention, args.gt_quat_order, args.gt_matrix_convention)
    else:
        poses_cv = None
        if args.internal_split and test_frame_indices and args.strict_no_test_packets and args.camera_source == 'packet':
            raise ValueError("Strict held-out split should not use --camera_source packet for test frames. Use --camera_source pose_file.")

    fovx = 2.0 * math.atan(args.width / (2.0 * args.fx))
    camera_infos: List[Dict[str, Any]] = []

    def build_frames(split_name: str, packet_indices: List[int], frame_indices: List[int]) -> List[Dict[str, Any]]:
        frames = []
        for pi, fi in zip(packet_indices, frame_indices):
            src = Path(args.image_dir) / args.image_pattern.format(index=fi, local_index=pi)
            with Image.open(src) as im:
                w, h = im.size
            if args.strict_image_size and (w != args.width or h != args.height):
                raise ValueError(f"Image size mismatch for {src}: got {w}x{h}, expected {args.width}x{args.height}")
            dst = image_out / f"{fi:06d}.png"
            copy_or_link_image(src, dst, args.copy_images)
            image_file_path = f"images/{fi:06d}"
            frame, info = build_frame_record(fi, pi, image_file_path, args, poses_cv, packet_files_all)
            info['split'] = split_name
            camera_infos.append(info)
            frames.append(frame)
        return frames

    train_frames = build_frames('train', train_packet_indices, train_frame_indices)
    test_frames = build_frames('test', test_packet_indices, test_frame_indices)

    meta_train = {'camera_angle_x': fovx, 'frames': train_frames}
    meta_test = {'camera_angle_x': fovx, 'frames': test_frames}
    (output_scene / 'transforms_train.json').write_text(json.dumps(meta_train, indent=2), encoding='utf-8')
    (output_scene / 'transforms_test.json').write_text(json.dumps(meta_test, indent=2), encoding='utf-8')
    print(f"[write] transforms_train.json: {len(train_frames)} frames")
    print(f"[write] transforms_test.json:  {len(test_frames)} frames")

    return {
        'num_train_frames': len(train_frames),
        'num_test_frames': len(test_frames),
        'camera_angle_x': fovx,
        'camera_source': args.camera_source,
        'camera_infos': camera_infos,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare fused ReSplat 3DGS scene with optional strict held-out split')
    p.add_argument('--packet_dir', required=True)
    p.add_argument('--packet_ranges', required=True, help='e.g. 0-49. These index sorted packet files.')
    p.add_argument('--dataset_start_index', type=int, default=0, help='frame_index = dataset_start_index + packet_index')
    p.add_argument('--output_scene', required=True)

    # Split control.
    p.add_argument('--internal_split', action='store_true', help='Enable interleaved train/test split inside this prepare script.')
    p.add_argument('--split_every', type=int, default=5, help='Every Nth frame is test when --internal_split is enabled.')
    p.add_argument('--split_offset', type=int, default=4, help='Test if (index - offset) % split_every == 0.')
    p.add_argument('--split_index_mode', choices=['local_index', 'packet_index', 'frame_index'], default='local_index')
    p.add_argument('--strict_no_test_packets', action='store_true', default=True,
                   help='For strict evaluation, test packets are not fused. This is enabled by default.')
    p.add_argument('--allow_test_packets_for_fusion', dest='strict_no_test_packets', action='store_false',
                   help='Fuse all packets even under internal split. This is the diagnostic all-packet protocol, not strict held-out.')

    # Legacy split options when --internal_split is disabled.
    p.add_argument('--test_every', type=int, default=0)
    p.add_argument('--test_same_as_train', action='store_true')

    # Image/camera.
    p.add_argument('--image_dir', required=True)
    p.add_argument('--image_pattern', default='{index:06d}.png')
    p.add_argument('--fx', type=float, required=True)
    p.add_argument('--fy', type=float, default=None)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--copy_images', action='store_true')
    p.add_argument('--strict_image_size', action='store_true')

    p.add_argument('--camera_source', choices=['pose_file', 'packet'], default='pose_file',
                   help='For strict held-out evaluation, use pose_file. packet requires test packets to exist for test cameras.')
    p.add_argument('--pose_file', default=None, help='GT pose file. Required for --camera_source pose_file.')
    p.add_argument('--gt_convention', choices=['opencv_c2w', 'resplat_tartanair_pose'], default='resplat_tartanair_pose')
    p.add_argument('--gt_quat_order', choices=['xyzw', 'wxyz'], default='xyzw')
    p.add_argument('--gt_matrix_convention', choices=['c2w', 'w2c'], default='c2w')
    p.add_argument('--packet_camera_key', default=None)
    p.add_argument('--packet_camera_view_index', type=int, default=-1)
    p.add_argument('--packet_camera_convention', choices=['twc_cv', 'tcw_cv', 'twc_opengl', 'tcw_opengl', 'twc_nerf', 'tcw_nerf'], default='twc_cv')

    # Gaussian conversion.
    p.add_argument('--sh_degree', type=int, default=3)
    p.add_argument('--opacity_mode', choices=['auto', 'activated', 'internal'], default='auto')
    p.add_argument('--scale_mode', choices=['auto', 'activated', 'internal'], default='auto')
    p.add_argument('--rotation_format', choices=['wxyz', 'xyzw'], default='wxyz')
    p.add_argument('--color_mode', choices=['auto', 'rgb', 'sh'], default='auto')
    p.add_argument('--allow_identity_covariance', action='store_true')
    p.add_argument('--default_scale', type=float, default=0.01)
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
    if args.split_every <= 0:
        raise ValueError('--split_every must be positive')
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
    packet_indices_all = parse_int_ranges(args.packet_ranges)
    if not packet_indices_all:
        raise ValueError('--packet_ranges produced no packet indices')
    for pi in packet_indices_all:
        if pi < 0 or pi >= len(packet_files_all):
            raise IndexError(f"packet index {pi} out of range [0,{len(packet_files_all)-1}]")

    first_packet = safe_torch_load(packet_files_all[packet_indices_all[0]])
    if args.inspect_only:
        print(f"[inspect] packet file: {packet_files_all[packet_indices_all[0]]}")
        print_packet_structure(first_packet, args.max_inspect_items)
        return

    train_packet_indices, test_packet_indices, train_frame_indices, test_frame_indices = make_split(packet_indices_all, args)
    if args.internal_split and args.strict_no_test_packets:
        fuse_packet_indices = train_packet_indices
        protocol = 'strict_heldout_frames'
    else:
        fuse_packet_indices = packet_indices_all
        protocol = 'all_packets_fused'

    print(f"[split] protocol={protocol}")
    print(f"[split] train packets/frames: {len(train_packet_indices)}")
    print(f"[split] test packets/frames:  {len(test_packet_indices)}")
    print(f"[fuse] packets used for initial_fused_3dgs.ply: {len(fuse_packet_indices)}")

    parts: List[GaussianArrays] = []
    selected_packet_paths = []
    for pi in fuse_packet_indices:
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

    cam_summary = write_transforms_and_images(
        args,
        output_scene,
        train_packet_indices,
        test_packet_indices,
        train_frame_indices,
        test_frame_indices,
        packet_files_all,
    )

    summary = {
        'script_version': 'prepare_resplat_fused_3dgs_scene_v3_strict_split',
        'protocol': protocol,
        'packet_dir': str(packet_dir),
        'packet_ranges': args.packet_ranges,
        'packet_indices_all': packet_indices_all,
        'train_packet_indices': train_packet_indices,
        'test_packet_indices': test_packet_indices,
        'train_frame_indices': train_frame_indices,
        'test_frame_indices': test_frame_indices,
        'fused_packet_indices': fuse_packet_indices,
        'selected_packet_paths_for_fusion': selected_packet_paths,
        'num_gaussians': int(fused.xyz.shape[0]),
        'initial_ply': str(initial_ply),
        'points3d_ply': str(points_ply),
        'output_scene': str(output_scene),
        'sh_degree': args.sh_degree,
        'split': {
            'internal_split': bool(args.internal_split),
            'split_every': args.split_every,
            'split_offset': args.split_offset,
            'split_index_mode': args.split_index_mode,
            'strict_no_test_packets': bool(args.strict_no_test_packets),
        },
        'camera': {
            'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy,
            'width': args.width, 'height': args.height,
        },
        **cam_summary,
    }
    (output_scene / 'prepare_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"[done] scene prepared at {output_scene}")
    print(f"[next] train with --initial_ply {initial_ply}")


if __name__ == '__main__':
    main()
