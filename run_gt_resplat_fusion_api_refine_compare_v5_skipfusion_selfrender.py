#!/usr/bin/env python3
# GT-pose only ReSplat packet fusion script.
# TartanAir stereo + GT pose -> ReSplat Gaussian packets -> fusion/render
# This intentionally removes ZipMap inference/alignment logic from
# run_zipmap_resplat_fusion_api_official_streaming_merged_init_fast.py.

from __future__ import annotations

SCRIPT_VERSION = "2026-06-11-gt-resplat-refine-compare-v5-skipfusion-selfrender"


import argparse
import contextlib
import csv
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as tf

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Generic helpers
# =============================================================================


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_refine_steps_spec(spec: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated refinement counts. 0 means raw init Gaussians."""
    if spec is None:
        return None
    parts = [p.strip() for p in str(spec).replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--refine_steps must not be empty when specified.")
    out: list[int] = []
    seen: set[int] = set()
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid refine step {p!r}; expected integer.") from exc
        if v < 0:
            raise argparse.ArgumentTypeError(f"Invalid refine step {v}; expected >= 0.")
        if v not in seen:
            out.append(v)
            seen.add(v)
    return sorted(out)


def refine_stage_name(step: int) -> str:
    return f"refine_{int(step)}"


def abs_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def log(msg: str) -> None:
    print(msg, flush=True)


@contextlib.contextmanager
def temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def tensor_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: tensor_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tensor_to_device(v, device) for v in obj)
    return obj


def save_png_tensor(image: torch.Tensor, path: Path) -> None:
    """Save [3,H,W] image tensor in [0,1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    arr = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def collect_images(root: Path, recursive: bool = False) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image files found under: {root}")
    return paths


def make_sample_indices(
    n: int,
    start_index: int,
    end_index: Optional[int],
    stride: int,
    max_count: Optional[int],
) -> list[int]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    start = max(0, int(start_index))
    end = n if end_index is None else min(n, int(end_index))
    if start >= end:
        raise ValueError(f"Invalid range: start_index={start_index}, end_index={end_index}, length={n}")
    indices = list(range(start, end, stride))
    if max_count is not None:
        if max_count <= 0:
            raise ValueError(f"num_frames must be positive, got {max_count}")
        indices = indices[:max_count]
    if not indices:
        raise ValueError("Sampling produced zero frames.")
    return indices


def build_pixel_K(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = float(fx)
    K[1, 1] = float(fy)
    K[0, 2] = float(cx)
    K[1, 2] = float(cy)
    return K


def process_image_and_K(
    image_path: Path,
    K_pixel: torch.Tensor,
    image_shape: tuple[int, int],
    normalize_intrinsics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the TartanAir loader preprocessing used by ReSplat."""
    to_tensor = tf.ToTensor()
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        K = K_pixel.clone().float()
        target_h, target_w = image_shape

        scale = max(target_w / orig_w, target_h / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))
        im = im.resize((resized_w, resized_h), Image.BILINEAR)

        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale

        left = int(round((resized_w - target_w) / 2.0))
        top = int(round((resized_h - target_h) / 2.0))
        im = im.crop((left, top, left + target_w, top + target_h))
        K[0, 2] -= left
        K[1, 2] -= top

        if normalize_intrinsics:
            K[0, 0] /= target_w
            K[1, 1] /= target_h
            K[0, 2] /= target_w
            K[1, 2] /= target_h
        return to_tensor(im), K


# =============================================================================
# Pose utilities. These are copied into this script to avoid ZipMap dependency.
# =============================================================================


def tartan_from_cv_matrix_np(dtype=np.float64) -> np.ndarray:
    """T_tartanCam_from_cvCam. For TartanAir pose7: Twc_cv = Twc_tartan @ T_tartanCam_from_cvCam."""
    T = np.eye(4, dtype=dtype)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )
    return T


def tartan_from_cv_matrix_torch(device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    return T


def fixed_tartanair_stereo_rig_cv(
    baseline: float = 0.25000006,
    device: torch.device | None = None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Return the same constant relative transform used in the previous API script.

    It is used as:
        T_right_c2w_cv = T_left_c2w_cv @ T_rel

    This preserves compatibility with your existing ReSplat packet generation path.
    """
    T_tartan_from_cv = tartan_from_cv_matrix_torch(device=device, dtype=dtype)
    T_cv_from_tartan = torch.linalg.inv(T_tartan_from_cv)

    T_rel_tartan = torch.eye(4, dtype=dtype, device=device)
    T_rel_tartan[:3, 3] = torch.tensor([0.0, float(baseline), 0.0], dtype=dtype, device=device)
    return T_cv_from_tartan @ T_rel_tartan @ T_tartan_from_cv


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
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_T_c2w_from_pose7(values: Sequence[float], quat_order: str = "xyzw") -> np.ndarray:
    if len(values) != 7:
        raise ValueError(f"pose7 must have 7 values, got {len(values)}")
    tx, ty, tz = values[:3]
    if quat_order == "xyzw":
        q = values[3:7]
    elif quat_order == "wxyz":
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


def rotation_angle_deg(R: np.ndarray) -> float:
    val = (float(np.trace(R)) - 1.0) / 2.0
    val = min(1.0, max(-1.0, val))
    return math.degrees(math.acos(val))


def read_numeric_pose_file(path: Path) -> tuple[Optional[list[str]], np.ndarray, str]:
    rows: list[list[float]] = []
    stamps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
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
        return stamps, arr[:, 1:8], "tum_pose7"
    if ncols == 7:
        return None, arr, "pose7"
    if ncols == 13:
        return stamps, arr[:, 1:13].reshape(-1, 3, 4), "matrix34"
    if ncols == 12:
        return None, arr.reshape(-1, 3, 4), "matrix34"
    if ncols == 17:
        return stamps, arr[:, 1:17].reshape(-1, 4, 4), "matrix44"
    if ncols == 16:
        return None, arr.reshape(-1, 4, 4), "matrix44"
    raise ValueError(
        f"Unsupported pose format in {path}: {ncols} numeric columns. "
        "Expected 7/8 for pose7/TUM, 12/13 for 3x4, or 16/17 for 4x4."
    )


def load_gt_trajectory(
    pose_file: Path,
    convention: str = "resplat_tartanair_pose",
    quat_order: str = "xyzw",
    matrix_convention: str = "c2w",
) -> np.ndarray:
    _, data, fmt = read_numeric_pose_file(pose_file)
    if fmt in {"pose7", "tum_pose7"}:
        T_pose = np.stack([make_T_c2w_from_pose7(row, quat_order=quat_order) for row in data], axis=0)
        if convention == "opencv_c2w":
            return T_pose.astype(np.float32)
        if convention == "resplat_tartanair_pose":
            return np.matmul(T_pose, tartan_from_cv_matrix_np()[None]).astype(np.float32)
        raise ValueError(f"Unsupported convention for pose7: {convention}")

    if fmt == "matrix34":
        T = np.tile(np.eye(4, dtype=np.float64), (data.shape[0], 1, 1))
        T[:, :3, :4] = data
    else:
        T = data.astype(np.float64)

    if matrix_convention == "w2c":
        T = invert_se3_np(T)
    elif matrix_convention != "c2w":
        raise ValueError(f"Unknown matrix_convention: {matrix_convention}")

    if convention == "opencv_c2w":
        return T.astype(np.float32)
    if convention == "resplat_tartanair_pose":
        return np.matmul(T, tartan_from_cv_matrix_np()[None]).astype(np.float32)
    raise ValueError(f"Unsupported convention: {convention}")


# =============================================================================
# Frame and GT pose stage
# =============================================================================


@dataclass
class SelectedStereoFrames:
    indices: list[int]
    left_paths: list[Path]
    right_paths: list[Path]
    frame_records: list[dict[str, Any]]
    stereo_pairs: list[dict[str, Any]]


@dataclass
class GTPoseResult:
    out_dir: Path
    selected: SelectedStereoFrames
    left_T_c2w_cv: np.ndarray
    right_T_c2w_cv: Optional[np.ndarray]
    right_pose_source_resolved: str
    diagnostics: dict[str, Any]


def select_stereo_frames(
    left_dir: Path,
    right_dir: Path,
    start_index: int,
    end_index: Optional[int],
    stride: int,
    num_frames: Optional[int],
    recursive: bool,
) -> SelectedStereoFrames:
    left_all = collect_images(left_dir, recursive=recursive)
    right_all = collect_images(right_dir, recursive=recursive)
    if len(left_all) != len(right_all):
        raise ValueError(f"Left/right image count mismatch: left={len(left_all)}, right={len(right_all)}")

    indices = make_sample_indices(
        len(left_all), start_index=start_index, end_index=end_index, stride=stride, max_count=num_frames
    )
    left_paths = [left_all[i] for i in indices]
    right_paths = [right_all[i] for i in indices]

    frame_records: list[dict[str, Any]] = []
    stereo_pairs: list[dict[str, Any]] = []
    for local_i, original_i in enumerate(indices):
        frame_records.append(
            {
                "local_index": local_i,
                "original_index": int(original_i),
                "gt_index": int(original_i),
                "left_name": left_all[original_i].name,
                "right_name": right_all[original_i].name,
            }
        )
        stereo_pairs.append(
            {
                "local_index": local_i,
                "original_index": int(original_i),
                "left": str(left_all[original_i]),
                "right": str(right_all[original_i]),
            }
        )
    return SelectedStereoFrames(indices, left_paths, right_paths, frame_records, stereo_pairs)


def resolve_right_poses(
    args: argparse.Namespace,
    left_T: np.ndarray,
    selected: SelectedStereoFrames,
) -> tuple[Optional[np.ndarray], str, dict[str, Any]]:
    right_pose_source = args.right_pose_source
    if right_pose_source == "auto":
        right_pose_source = "gt_per_frame" if args.right_gt_pose_file is not None else "fixed_baseline"

    diagnostics: dict[str, Any] = {"right_pose_source": right_pose_source}
    right_T_all: Optional[np.ndarray] = None
    right_T_selected: Optional[np.ndarray] = None

    if args.right_gt_pose_file is not None:
        right_T_all = load_gt_trajectory(
            abs_path(args.right_gt_pose_file),
            convention=args.gt_convention,
            quat_order=args.gt_quat_order,
            matrix_convention=args.gt_matrix_convention,
        )
        right_T_selected = right_T_all[selected.indices]
        diagnostics["right_gt_pose_file"] = str(abs_path(args.right_gt_pose_file))

    if right_pose_source == "gt_per_frame":
        if right_T_selected is None:
            raise ValueError("--right_pose_source gt_per_frame requires --right_gt_pose_file")
        return right_T_selected.astype(np.float32), right_pose_source, diagnostics

    if right_pose_source == "gt_first_relative":
        if right_T_selected is None:
            raise ValueError("--right_pose_source gt_first_relative requires --right_gt_pose_file")
        T_rel0 = invert_se3_np(left_T[0]) @ right_T_selected[0]
        right_T = np.matmul(left_T, T_rel0[None]).astype(np.float32)
        diagnostics["T_rel0_left_inv_right"] = T_rel0.tolist()
        return right_T, right_pose_source, diagnostics

    if right_pose_source == "fixed_baseline":
        return None, right_pose_source, diagnostics

    raise ValueError(f"Unknown right_pose_source: {right_pose_source}")


def write_stereo_diagnostics(
    out_dir: Path,
    selected: SelectedStereoFrames,
    left_T: np.ndarray,
    right_T_gt: Optional[np.ndarray],
    stereo_baseline: float,
) -> dict[str, Any]:
    diag_dir = out_dir / "stereo_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    T_fixed = fixed_tartanair_stereo_rig_cv(baseline=stereo_baseline).cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []

    if right_T_gt is not None:
        rels = np.matmul(invert_se3_np(left_T.astype(np.float64)), right_T_gt.astype(np.float64))
        rel0 = rels[0]
        for local_i, original_i in enumerate(selected.indices):
            rel = rels[local_i]
            rel_vs_first = invert_se3_np(rel0) @ rel
            rel_vs_fixed = invert_se3_np(T_fixed) @ rel
            rows.append(
                {
                    "local_index": local_i,
                    "original_index": int(original_i),
                    "rel_tx": float(rel[0, 3]),
                    "rel_ty": float(rel[1, 3]),
                    "rel_tz": float(rel[2, 3]),
                    "rel_norm": float(np.linalg.norm(rel[:3, 3])),
                    "rot_vs_first_deg": float(rotation_angle_deg(rel_vs_first[:3, :3])),
                    "trans_vs_first_norm": float(np.linalg.norm(rel_vs_first[:3, 3])),
                    "rot_vs_fixed_deg": float(rotation_angle_deg(rel_vs_fixed[:3, :3])),
                    "trans_vs_fixed_norm": float(np.linalg.norm(rel_vs_fixed[:3, 3])),
                }
            )
        with (diag_dir / "left_right_relative_pose.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        rel_norms = np.asarray([r["rel_norm"] for r in rows], dtype=np.float64)
        trans_vs_first = np.asarray([r["trans_vs_first_norm"] for r in rows], dtype=np.float64)
        rot_vs_first = np.asarray([r["rot_vs_first_deg"] for r in rows], dtype=np.float64)
        trans_vs_fixed = np.asarray([r["trans_vs_fixed_norm"] for r in rows], dtype=np.float64)
        rot_vs_fixed = np.asarray([r["rot_vs_fixed_deg"] for r in rows], dtype=np.float64)
        summary = {
            "has_right_gt": True,
            "fixed_baseline": float(stereo_baseline),
            "fixed_T_rel_used_by_old_script": T_fixed.tolist(),
            "rel_norm_mean": float(rel_norms.mean()),
            "rel_norm_std": float(rel_norms.std()),
            "rel_norm_min": float(rel_norms.min()),
            "rel_norm_max": float(rel_norms.max()),
            "trans_vs_first_mean": float(trans_vs_first.mean()),
            "trans_vs_first_max": float(trans_vs_first.max()),
            "rot_vs_first_deg_mean": float(rot_vs_first.mean()),
            "rot_vs_first_deg_max": float(rot_vs_first.max()),
            "trans_vs_fixed_mean": float(trans_vs_fixed.mean()),
            "trans_vs_fixed_max": float(trans_vs_fixed.max()),
            "rot_vs_fixed_deg_mean": float(rot_vs_fixed.mean()),
            "rot_vs_fixed_deg_max": float(rot_vs_fixed.max()),
            "csv": str(diag_dir / "left_right_relative_pose.csv"),
        }
    else:
        summary = {
            "has_right_gt": False,
            "fixed_baseline": float(stereo_baseline),
            "fixed_T_rel_used_by_old_script": T_fixed.tolist(),
            "note": "No --right_gt_pose_file was provided. Only the fixed baseline transform can be reported.",
        }

    save_json(diag_dir / "summary.json", summary)
    return summary


def load_gt_pose_stage(args: argparse.Namespace, selected: SelectedStereoFrames) -> GTPoseResult:
    out_dir = abs_path(args.work_dir) / args.gt_out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    left_T_all = load_gt_trajectory(
        abs_path(args.gt_pose_file),
        convention=args.gt_convention,
        quat_order=args.gt_quat_order,
        matrix_convention=args.gt_matrix_convention,
    )
    left_T = left_T_all[selected.indices].astype(np.float32)
    right_T, right_pose_source, diagnostics = resolve_right_poses(args, left_T, selected)

    save_json(out_dir / "frame_records.json", selected.frame_records)
    save_json(out_dir / "stereo_pairs.json", selected.stereo_pairs)
    np.savez_compressed(
        out_dir / "gt_poses_selected.npz",
        left_T_c2w_opencv=left_T,
        right_T_c2w_opencv=np.asarray([]) if right_T is None else right_T,
        selected_original_indices=np.asarray(selected.indices, dtype=np.int64),
    )

    stereo_diag = write_stereo_diagnostics(
        out_dir=out_dir,
        selected=selected,
        left_T=left_T,
        right_T_gt=right_T if args.right_gt_pose_file is not None else None,
        stereo_baseline=args.stereo_baseline,
    )
    diagnostics.update(stereo_diag)
    save_json(out_dir / "summary.json", diagnostics)

    log(f"[1/3] GT pose: loaded {len(selected.indices)} left poses from {abs_path(args.gt_pose_file)}")
    log(f"      right_pose_source={right_pose_source}")
    if args.right_gt_pose_file is not None:
        log(
            "      stereo GT rel: "
            f"norm_mean={stereo_diag.get('rel_norm_mean', 0.0):.6f}, "
            f"trans_vs_first_max={stereo_diag.get('trans_vs_first_max', 0.0):.6e}, "
            f"trans_vs_fixed_max={stereo_diag.get('trans_vs_fixed_max', 0.0):.6e}"
        )

    return GTPoseResult(
        out_dir=out_dir,
        selected=selected,
        left_T_c2w_cv=left_T,
        right_T_c2w_cv=right_T,
        right_pose_source_resolved=right_pose_source,
        diagnostics=diagnostics,
    )


# =============================================================================
# ReSplat runtime and packet generation
# =============================================================================


@dataclass
class ResplatRuntime:
    model: Any
    cfg: Any
    device: torch.device
    image_shape: tuple[int, int]
    near: float
    far: float
    normalize_intrinsics: bool
    K_pixel: torch.Tensor


def resolve_repo_relative(path_like: Any, repo: Path) -> Optional[str]:
    if path_like is None:
        return None
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return str(p)
    return str((repo / p).resolve())


def compose_resplat_cfg(args: argparse.Namespace):
    resplat_repo = abs_path(args.resplat_repo)
    sys.path.insert(0, str(resplat_repo))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        f"+experiment={args.resplat_experiment}",
        "mode=test",
        "wandb.mode=disabled",
        f"output_dir={str(abs_path(args.work_dir) / args.resplat_out_name)}",
    ]
    overrides.extend(args.resplat_override or [])
    if args.resplat_checkpoint is not None:
        overrides.append(f"checkpointing.pretrained_model={str(abs_path(args.resplat_checkpoint))}")

    with initialize_config_dir(config_dir=str(resplat_repo / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    return cfg_dict


def load_resplat_runtime(args: argparse.Namespace) -> ResplatRuntime:
    resplat_repo = abs_path(args.resplat_repo)
    sys.path.insert(0, str(resplat_repo))

    with temporary_cwd(resplat_repo):
        cfg_dict = compose_resplat_cfg(args)
        from src.config import load_typed_root_config
        from src.global_cfg import set_cfg
        from src.loss import get_losses
        from src.misc.step_tracker import StepTracker
        from src.model.decoder import get_decoder
        from src.model.encoder import get_encoder
        from src.model.model_wrapper import ModelWrapper

        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
        model = ModelWrapper(
            cfg.optimizer,
            cfg.test,
            cfg.train,
            encoder,
            encoder_visualizer,
            get_decoder(cfg.model.decoder, cfg.dataset),
            get_losses(cfg.loss),
            StepTracker(),
            eval_data_cfg=None,
        )

        strict_load = not cfg.checkpointing.no_strict_load
        pretrained_model_path = resolve_repo_relative(cfg.checkpointing.pretrained_model, resplat_repo)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=strict_load)
            log(f"[2/3] ReSplat: loaded pretrained model {pretrained_model_path}")
        else:
            log("[2/3] ReSplat: no checkpointing.pretrained_model configured")

        pretrained_depth_path = resolve_repo_relative(cfg.checkpointing.pretrained_depth, resplat_repo)
        if pretrained_depth_path is not None:
            depth_ckpt = torch.load(pretrained_depth_path, map_location="cpu")
            if isinstance(depth_ckpt, dict) and "state_dict" in depth_ckpt:
                depth_ckpt = depth_ckpt["state_dict"]
            if isinstance(depth_ckpt, dict) and "model" in depth_ckpt:
                depth_ckpt = depth_ckpt["model"]
            model.encoder.depth_predictor.load_state_dict(depth_ckpt, strict=strict_load)
            log(f"      loaded pretrained depth {pretrained_depth_path}")

        update_path = resolve_repo_relative(cfg.checkpointing.resume_update_module, resplat_repo)
        if update_path is not None:
            update_ckpt = torch.load(update_path, map_location="cpu")
            if isinstance(update_ckpt, dict) and "state_dict" in update_ckpt:
                update_ckpt = update_ckpt["state_dict"]
            if isinstance(update_ckpt, dict) and "model" in update_ckpt:
                update_ckpt = update_ckpt["model"]
            model_state = model.state_dict()
            filtered = {
                k: v for k, v in update_ckpt.items()
                if "encoder.update" in k and k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
            }
            model.load_state_dict(filtered, strict=False)
            log(f"      loaded update-module weights {update_path} ({len(filtered)} tensors)")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"ReSplat expects CUDA. Requested device={args.device}, cuda_available={torch.cuda.is_available()}")
    model.eval().to(device)

    dataset_cfg = cfg.dataset
    image_shape = tuple(int(x) for x in dataset_cfg.image_shape)
    near = float(dataset_cfg.near)
    far = float(dataset_cfg.far)
    normalize_intrinsics = bool(dataset_cfg.normalize_intrinsics)

    seq0 = dataset_cfg.sequences[0] if getattr(dataset_cfg, "sequences", None) else dataset_cfg
    fx = float(getattr(seq0, "fx", getattr(dataset_cfg, "fx", 0.0)))
    fy = float(getattr(seq0, "fy", getattr(dataset_cfg, "fy", 0.0)))
    cx = float(getattr(seq0, "cx", getattr(dataset_cfg, "cx", 0.0)))
    cy = float(getattr(seq0, "cy", getattr(dataset_cfg, "cy", 0.0)))
    if args.fx is not None:
        fx = args.fx
    if args.fy is not None:
        fy = args.fy
    if args.cx is not None:
        cx = args.cx
    if args.cy is not None:
        cy = args.cy
    if not all(v > 0 for v in (fx, fy, cx, cy)):
        raise ValueError(f"Invalid ReSplat intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    log(f"      image_shape={image_shape}, near={near}, far={far}, K=({fx},{fy},{cx},{cy})")
    return ResplatRuntime(
        model=model,
        cfg=cfg,
        device=device,
        image_shape=image_shape,
        near=near,
        far=far,
        normalize_intrinsics=normalize_intrinsics,
        K_pixel=build_pixel_K(fx, fy, cx, cy),
    )


def make_resplat_batch_for_frame(
    runtime: ResplatRuntime,
    scene_key: str,
    frame_index: int,
    left_path: Path,
    right_path: Path,
    T_left_c2w_cv: np.ndarray,
    target_camera: str,
    target_offset_frame_index: Optional[int] = None,
    target_left_path: Optional[Path] = None,
    target_right_path: Optional[Path] = None,
    target_T_left_c2w_cv: Optional[np.ndarray] = None,
    T_right_c2w_cv: Optional[np.ndarray] = None,
    target_T_right_c2w_cv: Optional[np.ndarray] = None,
    stereo_baseline: float = 0.25000006,
) -> dict[str, Any]:
    T_left = torch.tensor(T_left_c2w_cv, dtype=torch.float32)
    if T_right_c2w_cv is None:
        T_rel = fixed_tartanair_stereo_rig_cv(baseline=stereo_baseline, device=None, dtype=torch.float32)
        T_right = T_left @ T_rel
    else:
        T_right = torch.tensor(T_right_c2w_cv, dtype=torch.float32)

    img_l, K_l = process_image_and_K(left_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)
    img_r, K_r = process_image_and_K(right_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)

    context = {
        "extrinsics": torch.stack([T_left, T_right], dim=0).unsqueeze(0),
        "intrinsics": torch.stack([K_l, K_r], dim=0).unsqueeze(0),
        "image": torch.stack([img_l, img_r], dim=0).unsqueeze(0),
        "near": torch.full((1, 2), runtime.near, dtype=torch.float32),
        "far": torch.full((1, 2), runtime.far, dtype=torch.float32),
        "index": torch.tensor([[frame_index, frame_index]], dtype=torch.long),
        "camera_id": torch.tensor([[0, 1]], dtype=torch.long),
    }

    if target_offset_frame_index is None:
        target_offset_frame_index = frame_index
    if target_left_path is None:
        target_left_path = left_path
    if target_right_path is None:
        target_right_path = right_path
    if target_T_left_c2w_cv is None:
        target_T_left = T_left
    else:
        target_T_left = torch.tensor(target_T_left_c2w_cv, dtype=torch.float32)
    if target_T_right_c2w_cv is None:
        if T_right_c2w_cv is not None and target_T_left_c2w_cv is None:
            target_T_right = T_right
        else:
            T_rel = fixed_tartanair_stereo_rig_cv(baseline=stereo_baseline, device=None, dtype=torch.float32)
            target_T_right = target_T_left @ T_rel
    else:
        target_T_right = torch.tensor(target_T_right_c2w_cv, dtype=torch.float32)

    target_images: list[torch.Tensor] = []
    target_intrinsics: list[torch.Tensor] = []
    target_extrinsics: list[torch.Tensor] = []
    target_indices: list[int] = []
    target_camera_ids: list[int] = []

    if target_camera in {"left", "both"}:
        img_tl, K_tl = process_image_and_K(target_left_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)
        target_images.append(img_tl)
        target_intrinsics.append(K_tl)
        target_extrinsics.append(target_T_left)
        target_indices.append(target_offset_frame_index)
        target_camera_ids.append(0)
    if target_camera in {"right", "both"}:
        img_tr, K_tr = process_image_and_K(target_right_path, runtime.K_pixel, runtime.image_shape, runtime.normalize_intrinsics)
        target_images.append(img_tr)
        target_intrinsics.append(K_tr)
        target_extrinsics.append(target_T_right)
        target_indices.append(target_offset_frame_index)
        target_camera_ids.append(1)
    if not target_images:
        raise ValueError("target_camera must be left, right, or both")

    target = {
        "extrinsics": torch.stack(target_extrinsics, dim=0).unsqueeze(0),
        "intrinsics": torch.stack(target_intrinsics, dim=0).unsqueeze(0),
        "image": torch.stack(target_images, dim=0).unsqueeze(0),
        "near": torch.full((1, len(target_images)), runtime.near, dtype=torch.float32),
        "far": torch.full((1, len(target_images)), runtime.far, dtype=torch.float32),
        "index": torch.tensor([target_indices], dtype=torch.long),
        "camera_id": torch.tensor([target_camera_ids], dtype=torch.long),
    }

    return {
        "context": context,
        "target": target,
        "scene": [scene_key],
        "scene_name": [scene_key.rsplit("_", 1)[0] if scene_key.rsplit("_", 1)[-1].isdigit() else scene_key],
    }


def make_packet_from_gaussians(
    gaussians: Any,
    batch: dict[str, Any],
    decoder: Any,
    stage: str,
) -> dict[str, Any]:
    scene_value = batch["scene"]
    scene = scene_value[0] if isinstance(scene_value, (list, tuple)) else scene_value
    background_color = getattr(decoder, "background_color", None)
    if background_color is None:
        background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    elif torch.is_tensor(background_color):
        background_color = background_color.detach().cpu()
    else:
        background_color = torch.tensor(background_color, dtype=torch.float32)

    return {
        "scene": str(scene),
        "packet_stage": stage,
        "context_index": batch["context"]["index"][0].detach().cpu(),
        "target_index": batch["target"]["index"][0].detach().cpu(),
        "target_camera_id": batch["target"]["camera_id"][0].detach().cpu(),
        "context_extrinsics": batch["context"]["extrinsics"][0].detach().cpu(),
        "context_intrinsics": batch["context"]["intrinsics"][0].detach().cpu(),
        "context_near": batch["context"]["near"][0].detach().cpu(),
        "context_far": batch["context"]["far"][0].detach().cpu(),
        "target_extrinsics": batch["target"]["extrinsics"][0].detach().cpu(),
        "target_intrinsics": batch["target"]["intrinsics"][0].detach().cpu(),
        "target_near": batch["target"]["near"][0].detach().cpu(),
        "target_far": batch["target"]["far"][0].detach().cpu(),
        "target_image": batch["target"]["image"][0].detach().cpu(),
        "image_shape": tuple(int(x) for x in batch["target"]["image"].shape[-2:]),
        "background_color": background_color,
        "means": gaussians.means[0].detach().cpu(),
        "covariances": gaussians.covariances[0].detach().cpu() if gaussians.covariances is not None else None,
        "harmonics": gaussians.harmonics[0].detach().cpu(),
        "opacities": gaussians.opacities[0].detach().cpu(),
        "scales": gaussians.scales[0].detach().cpu() if gaussians.scales is not None else None,
        "rotations": gaussians.rotations[0].detach().cpu() if gaussians.rotations is not None else None,
        "rotations_unnorm": gaussians.rotations_unnorm[0].detach().cpu() if gaussians.rotations_unnorm is not None else None,
    }


@torch.no_grad()
def infer_resplat_packets(
    runtime: ResplatRuntime,
    gt_pose: GTPoseResult,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    model = runtime.model
    device = runtime.device
    selected = gt_pose.selected
    left_T_all = gt_pose.left_T_c2w_cv
    right_T_all = gt_pose.right_T_c2w_cv

    refine_steps = parse_refine_steps_spec(args.refine_steps)
    use_refine_compare_mode = refine_steps is not None

    if use_refine_compare_mode:
        assert refine_steps is not None
        max_refine = max(refine_steps)
        packet_groups: dict[str, list[dict[str, Any]]] = {refine_stage_name(s): [] for s in refine_steps}
        requested_stage_labels = [refine_stage_name(s) for s in refine_steps]

        original_cfg_num_refine = int(getattr(model.encoder.cfg, "num_refine", 0))
        if max_refine > 0:
            # The update/refine module is recurrent. The number of iterations is read
            # from cfg.num_refine inside forward_update(), so changing this runtime
            # field lets us test 1/2/4/... update iterations without rebuilding the
            # network. This is valid only if the model was instantiated with a
            # refinement module, i.e. the original config had num_refine > 0.
            if original_cfg_num_refine <= 0 or not hasattr(model.encoder, "update_module"):
                raise RuntimeError(
                    "--refine_steps requests refinement, but the loaded ReSplat encoder "
                    "was built without an update/refine module. Use a config whose "
                    "model.encoder.num_refine > 0."
                )
            model.encoder.cfg.num_refine = int(max_refine)
            log(
                f"[RefineCompare] overriding encoder.cfg.num_refine "
                f"{original_cfg_num_refine} -> {max_refine} for this run"
            )
    else:
        stage_request = args.resplat_packet_stage
        packet_groups = {"init": [], "final": []}
        requested_stage_labels = ["init", "final"] if stage_request == "both" else [stage_request]

    packet_out_root = abs_path(args.work_dir) / args.packet_out_name
    if args.save_packets:
        for stage_label in requested_stage_labels:
            (packet_out_root / stage_label).mkdir(parents=True, exist_ok=True)

    log(f"[3/3] ReSplat: generating packets from {len(selected.indices)} GT-posed stereo pairs")
    for local_i, original_i in enumerate(selected.indices):
        scene_key = f"{args.scene_name}_{local_i:04d}"
        target_local = local_i + args.resplat_target_offset
        if target_local < 0 or target_local >= len(selected.indices):
            if args.drop_invalid_target_offset:
                continue
            target_local = max(0, min(target_local, len(selected.indices) - 1))
        target_original_i = selected.indices[target_local]

        batch_cpu = make_resplat_batch_for_frame(
            runtime=runtime,
            scene_key=scene_key,
            frame_index=original_i,
            left_path=selected.left_paths[local_i],
            right_path=selected.right_paths[local_i],
            T_left_c2w_cv=left_T_all[local_i],
            T_right_c2w_cv=None if right_T_all is None else right_T_all[local_i],
            target_camera=args.resplat_target_camera,
            target_offset_frame_index=target_original_i,
            target_left_path=selected.left_paths[target_local],
            target_right_path=selected.right_paths[target_local],
            target_T_left_c2w_cv=left_T_all[target_local],
            target_T_right_c2w_cv=None if right_T_all is None else right_T_all[target_local],
            stereo_baseline=args.stereo_baseline,
        )
        batch = tensor_to_device(batch_cpu, device)
        batch = model.data_shim(batch)

        context = batch["context"]
        enc_out = model.encoder(
            context,
            0,
            deterministic=False,
            visualization_dump=None,
        )
        condition_features = None
        if isinstance(enc_out, dict):
            condition_features = enc_out.get("condition_features", None)
            gaussians_init = enc_out["gaussians"]
        else:
            gaussians_init = enc_out

        if use_refine_compare_mode:
            assert refine_steps is not None
            max_refine = max(refine_steps)
            refine_out = None
            if max_refine > 0:
                if condition_features is None:
                    raise RuntimeError(
                        "Refinement was requested but encoder output did not contain condition_features. "
                        "Check that model.encoder.num_refine was positive when the encoder was constructed."
                    )
                refine_out = model.encoder.forward_update(
                    batch["context"],
                    batch["target"] if args.refine_use_target else None,
                    condition_features,
                    gaussians_init,
                    model.decoder,
                    batch.get("context_remain", None),
                )
                if len(refine_out["gaussian"]) < max_refine:
                    raise RuntimeError(
                        f"forward_update returned {len(refine_out['gaussian'])} gaussian outputs, "
                        f"but --refine_steps requested {max_refine}."
                    )

            for step in refine_steps:
                stage_label = refine_stage_name(step)
                if step == 0:
                    g = gaussians_init
                else:
                    assert refine_out is not None
                    g = refine_out["gaussian"][step - 1]
                pkt = make_packet_from_gaussians(g, batch, model.decoder, stage=stage_label)
                pkt["refine_step"] = int(step)
                packet_groups[stage_label].append(pkt)
                if args.save_packets:
                    torch.save(pkt, packet_out_root / stage_label / f"{scene_key}.pt")
        else:
            stage_request = args.resplat_packet_stage
            if stage_request in {"init", "both"}:
                pkt = make_packet_from_gaussians(gaussians_init, batch, model.decoder, stage="init")
                packet_groups["init"].append(pkt)
                if args.save_packets:
                    torch.save(pkt, packet_out_root / "init" / f"{scene_key}.pt")

            if stage_request in {"final", "both"}:
                gaussians_final = gaussians_init
                if getattr(model.encoder.cfg, "num_refine", 0) > 0:
                    if condition_features is None:
                        raise RuntimeError(
                            "encoder.num_refine > 0 but encoder output did not contain condition_features. "
                            "Use --resplat_packet_stage init to skip refinement, or check ReSplat config."
                        )
                    refine_out = model.encoder.forward_update(
                        batch["context"],
                        None,
                        condition_features,
                        gaussians_init,
                        model.decoder,
                        batch.get("context_remain", None),
                    )
                    if len(refine_out["gaussian"]) == 0:
                        raise RuntimeError("forward_update returned no gaussian outputs.")
                    gaussians_final = refine_out["gaussian"][-1]

                pkt = make_packet_from_gaussians(gaussians_final, batch, model.decoder, stage="final")
                packet_groups["final"].append(pkt)
                if args.save_packets:
                    torch.save(pkt, packet_out_root / "final" / f"{scene_key}.pt")

        if (local_i + 1) % max(1, args.log_every) == 0 or local_i + 1 == len(selected.indices):
            log(f"      generated {local_i + 1}/{len(selected.indices)} packet(s)")

        del batch, batch_cpu, enc_out
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    manifest = {
        "script_version": SCRIPT_VERSION,
        "pose_source": "gt",
        "right_pose_source": gt_pose.right_pose_source_resolved,
        "refine_steps": None if refine_steps is None else [int(s) for s in refine_steps],
        "refine_compare_mode": bool(use_refine_compare_mode),
        "refine_use_target": bool(args.refine_use_target),
        "packet_stage_requested": args.resplat_packet_stage,
        "packet_stage_for_fusion": args.packet_stage_for_fusion,
        "packet_counts": {k: len(v) for k, v in packet_groups.items()},
        "selected_original_indices": [int(i) for i in selected.indices],
        "target_camera": args.resplat_target_camera,
        "target_offset": args.resplat_target_offset,
    }
    save_json(packet_out_root / "manifest.json", manifest)
    return packet_groups


# =============================================================================
# Fusion and rendering
# =============================================================================


@dataclass
class ProbeView:
    label: str
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    near: torch.Tensor
    far: torch.Tensor
    image_shape: tuple[int, int]
    gt_image: Optional[torch.Tensor]
    meta: dict[str, Any]


def packet_sort_key(packet: dict[str, Any]) -> int:
    return int(packet["context_index"].reshape(-1)[0].item())


def concat_packets_to_gaussians(packet_list: list[dict[str, Any]], device: torch.device):
    from src.model.types import Gaussians

    def cat_optional(key: str):
        vals = [p.get(key, None) for p in packet_list]
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            raise ValueError(f"Mixed None/non-None field for {key}")
        return torch.cat(vals, dim=0).to(device, non_blocking=True).unsqueeze(0).contiguous()

    return Gaussians(
        means=cat_optional("means"),
        covariances=cat_optional("covariances"),
        harmonics=cat_optional("harmonics"),
        opacities=cat_optional("opacities"),
        scales=cat_optional("scales"),
        rotations=cat_optional("rotations"),
        rotations_unnorm=cat_optional("rotations_unnorm"),
    )


def probe_from_packet_target(packet: dict[str, Any], target_view_i: int, label_prefix: str) -> ProbeView:
    idx = int(packet["target_index"][target_view_i].item())
    cam_id = int(packet["target_camera_id"][target_view_i].item())
    cam = "left" if cam_id == 0 else "right"
    return ProbeView(
        label=f"{label_prefix}_{idx:06d}_{cam}",
        extrinsics=packet["target_extrinsics"][target_view_i].clone(),
        intrinsics=packet["target_intrinsics"][target_view_i].clone(),
        near=packet["target_near"][target_view_i].reshape(1).clone(),
        far=packet["target_far"][target_view_i].reshape(1).clone(),
        image_shape=tuple(int(x) for x in packet["image_shape"]),
        gt_image=packet["target_image"][target_view_i].clone(),
        meta={
            "source": "packet_target",
            "packet_scene": packet["scene"],
            "target_index": idx,
            "camera_id": cam_id,
            "camera": cam,
        },
    )


def build_packet_trajectory_probes(packets: list[dict[str, Any]]) -> list[ProbeView]:
    probes: list[ProbeView] = []
    for packet_i, packet in enumerate(packets):
        n_views = int(packet["target_image"].shape[0])
        for view_i in range(n_views):
            probe = probe_from_packet_target(packet, view_i, label_prefix=f"traj_{packet_i:04d}")
            probe.meta["trajectory_packet_i"] = packet_i
            probe.meta["trajectory_view_i"] = view_i
            probes.append(probe)
    return probes


@torch.no_grad()
def render_gaussians_to_probes(
    runtime: ResplatRuntime,
    gaussians,
    probes: list[ProbeView],
    images_dir: Path,
    gt_dir: Path,
    should_compute_lpips: bool,
    render_chunk_size: int = 1,
    filename_prefix: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], list[str]]:
    device = runtime.device
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    render_chunk_size = max(1, int(render_chunk_size))
    image_shapes = {p.image_shape for p in probes}
    if len(image_shapes) != 1:
        raise ValueError(f"All probes must share image_shape, got {sorted(image_shapes)}")
    H, W = next(iter(image_shapes))

    rendered_names: list[str] = []
    gt_names: list[str] = []
    per_view: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for chunk_start in range(0, len(probes), render_chunk_size):
        chunk = probes[chunk_start : chunk_start + render_chunk_size]
        extr = torch.stack([p.extrinsics for p in chunk], dim=0).unsqueeze(0).to(device)
        intr = torch.stack([p.intrinsics for p in chunk], dim=0).unsqueeze(0).to(device)
        near = torch.stack([p.near.reshape(()) for p in chunk], dim=0).unsqueeze(0).to(device)
        far = torch.stack([p.far.reshape(()) for p in chunk], dim=0).unsqueeze(0).to(device)

        output = runtime.model.decoder.forward(gaussians, extr, intr, near, far, (H, W), depth_mode=None)
        rendered = output.color[0].detach().cpu()

        chunk_gt_local_indices = [i for i, p in enumerate(chunk) if p.gt_image is not None]
        chunk_metrics: dict[str, Any] = {}
        if chunk_gt_local_indices:
            pred = rendered[chunk_gt_local_indices].to(device)
            gt = torch.stack([chunk[i].gt_image for i in chunk_gt_local_indices], dim=0).to(device)
            chunk_metrics["psnr"] = compute_psnr(gt, pred)
            chunk_metrics["ssim"] = compute_ssim(gt, pred)
            if should_compute_lpips:
                chunk_metrics["lpips"] = compute_lpips(gt, pred)

        for local_i, (probe, img) in enumerate(zip(chunk, rendered)):
            global_i = chunk_start + local_i
            filename = f"{filename_prefix}{global_i + 1:03d}.png"
            save_png_tensor(img, images_dir / filename)
            rendered_names.append(filename)
            row: dict[str, Any] = {
                "probe_label": probe.label,
                "gt_available": probe.gt_image is not None,
                "meta": probe.meta,
                "rendered_image": filename,
            }
            if probe.gt_image is not None:
                save_png_tensor(probe.gt_image, gt_dir / filename)
                gt_names.append(filename)
                metric_i = chunk_gt_local_indices.index(local_i)
                psnr = float(chunk_metrics["psnr"][metric_i].detach().cpu().item())
                ssim = float(chunk_metrics["ssim"][metric_i].detach().cpu().item())
                row["psnr"] = psnr
                row["ssim"] = ssim
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                if should_compute_lpips:
                    lpips = float(chunk_metrics["lpips"][metric_i].detach().cpu().item())
                    row["lpips"] = lpips
                    lpips_values.append(lpips)
            per_view.append(row)

        del output, rendered, extr, intr, near, far
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    metrics_mean: dict[str, Any] = {}
    if psnr_values:
        metrics_mean["psnr"] = float(sum(psnr_values) / len(psnr_values))
        metrics_mean["ssim"] = float(sum(ssim_values) / len(ssim_values))
        if should_compute_lpips and lpips_values:
            metrics_mean["lpips"] = float(sum(lpips_values) / len(lpips_values))
    return per_view, metrics_mean, rendered_names, gt_names


@torch.no_grad()
def run_packet_self_render(
    runtime: ResplatRuntime,
    packets: list[dict[str, Any]],
    args: argparse.Namespace,
    out_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if not args.self_render_packets:
        return None
    out_dir = abs_path(args.work_dir) / (out_name or args.self_render_out_name)
    images_dir = out_dir / "rendered"
    gt_dir = out_dir / "gt"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, packet in enumerate(packets):
        gaussians = concat_packets_to_gaussians([packet], runtime.device)
        probes = [probe_from_packet_target(packet, v, label_prefix=f"packet_{i:04d}") for v in range(packet["target_image"].shape[0])]
        per_view, metrics_mean, rendered_names, gt_names = render_gaussians_to_probes(
            runtime=runtime,
            gaussians=gaussians,
            probes=probes,
            images_dir=images_dir,
            gt_dir=gt_dir,
            should_compute_lpips=False,
            render_chunk_size=1,
            filename_prefix=f"packet_{i:04d}_",
        )
        row = {
            "packet_i": i,
            "scene": packet["scene"],
            "num_gaussians": int(packet["means"].shape[0]),
            "metrics_mean": metrics_mean,
            "rendered_images": rendered_names,
            "gt_images": gt_names,
            "metrics_per_view": per_view,
        }
        rows.append(row)
        if metrics_mean:
            log(f"      self-render {i + 1}/{len(packets)}: PSNR={metrics_mean['psnr']:.4f}, SSIM={metrics_mean['ssim']:.4f}")
        del gaussians
        if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
            torch.cuda.empty_cache()

    psnr = [float(r["metrics_mean"]["psnr"]) for r in rows if r.get("metrics_mean") and "psnr" in r["metrics_mean"]]
    ssim = [float(r["metrics_mean"]["ssim"]) for r in rows if r.get("metrics_mean") and "ssim" in r["metrics_mean"]]
    summary = {
        "output_dir": str(out_dir),
        "packet_count": len(packets),
        "metrics_mean": {
            "psnr": float(sum(psnr) / len(psnr)) if psnr else None,
            "ssim": float(sum(ssim) / len(ssim)) if ssim else None,
        },
        "packets": rows,
    }
    save_json(out_dir / "summary.json", summary)
    return summary


@torch.no_grad()
def run_fusion_api(
    runtime: ResplatRuntime,
    packet_groups: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    stage_override: Optional[str] = None,
    fusion_out_name_override: Optional[str] = None,
    self_render_out_name_override: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    stage = stage_override or args.packet_stage_for_fusion
    packets = packet_groups.get(stage, [])
    if not packets:
        raise RuntimeError(f"No packets available for fusion stage={stage!r}.")
    packets = sorted(packets, key=packet_sort_key)

    if args.fusion_packet_ranges is not None:
        selected_indices = parse_packet_ranges(args.fusion_packet_ranges, len(packets))
        packets = [packets[i] for i in selected_indices]
    elif args.fusion_max_packets is not None:
        packets = packets[: args.fusion_max_packets]

    out_dir = abs_path(args.work_dir) / (fusion_out_name_override or args.fusion_out_name)
    self_render_summary = run_packet_self_render(runtime, packets, args, out_name=self_render_out_name_override)
    self_render_summary_path = None if self_render_summary is None else str(
        abs_path(args.work_dir) / (self_render_out_name_override or args.self_render_out_name) / "summary.json"
    )
    self_render_metrics_mean = None if self_render_summary is None else self_render_summary.get("metrics_mean")

    # Important: --skip_fusion now skips only the final concat+trajectory render.
    # It still allows --self_render_packets to validate each local ReSplat packet.
    if args.skip_fusion:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "output_dir": str(out_dir),
            "packet_stage_for_fusion": stage,
            "fusion_probe_mode": args.fusion_probe_mode,
            "fusion_skipped": True,
            "semantics": "fusion_skipped_local_packet_self_render_only",
            "selected_packet_count": len(packets),
            "num_gaussians": None,
            "num_gaussians_if_fused": int(sum(int(p["means"].shape[0]) for p in packets)),
            "packet_scenes": [p["scene"] for p in packets],
            "context_indices": [p["context_index"].tolist() for p in packets],
            "target_indices": [p["target_index"].tolist() for p in packets],
            "rendered_images": [],
            "gt_images": [],
            "metrics_mean": {},
            "metrics_per_view": [],
            "self_render_summary": self_render_summary_path,
            "self_render_metrics_mean": self_render_metrics_mean,
            "selection": {
                "fusion_packet_ranges": args.fusion_packet_ranges,
                "fusion_max_packets": args.fusion_max_packets,
                "trajectory_render_chunk_size": args.trajectory_render_chunk_size,
            },
        }
        save_json(out_dir / "summary.json", summary)
        log(f"      fusion skipped for stage={stage}; summary: {out_dir / 'summary.json'}")
        if self_render_metrics_mean and self_render_metrics_mean.get("psnr") is not None:
            log(
                f"      self-render mean: PSNR={self_render_metrics_mean['psnr']:.4f}, "
                f"SSIM={self_render_metrics_mean['ssim']:.4f}"
            )
        return summary

    images_dir = out_dir / "images"
    gt_dir = out_dir / "gt"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    if args.fusion_probe_mode not in {"packet_last_only", "packet_trajectory"}:
        raise ValueError(
            "This GT-only script keeps only the final-map trajectory-render path. "
            "Use --fusion_probe_mode packet_trajectory."
        )

    fused = concat_packets_to_gaussians(packets, runtime.device)
    probes = build_packet_trajectory_probes(packets)
    log(
        f"      Fusion: final trajectory render with {len(packets)} packet(s), "
        f"{int(fused.means.shape[1])} Gaussians, {len(probes)} probe view(s), "
        f"render_chunk_size={args.trajectory_render_chunk_size}"
    )
    per_view, metrics_mean, rendered_names, gt_names = render_gaussians_to_probes(
        runtime=runtime,
        gaussians=fused,
        probes=probes,
        images_dir=images_dir,
        gt_dir=gt_dir,
        should_compute_lpips=args.fusion_compute_lpips,
        render_chunk_size=args.trajectory_render_chunk_size,
    )
    if metrics_mean:
        msg = f"      final trajectory: PSNR={metrics_mean['psnr']:.4f}, SSIM={metrics_mean['ssim']:.4f}"
        if args.fusion_compute_lpips and "lpips" in metrics_mean:
            msg += f", LPIPS={metrics_mean['lpips']:.4f}"
        log(msg)

    summary = {
        "output_dir": str(out_dir),
        "packet_stage_for_fusion": stage,
        "fusion_probe_mode": args.fusion_probe_mode,
        "fusion_skipped": False,
        "semantics": "final_fused_map_rendered_along_selected_packet_target_trajectory",
        "selected_packet_count": len(packets),
        "num_gaussians": int(fused.means.shape[1]),
        "packet_scenes": [p["scene"] for p in packets],
        "context_indices": [p["context_index"].tolist() for p in packets],
        "target_indices": [p["target_index"].tolist() for p in packets],
        "rendered_images": rendered_names,
        "gt_images": gt_names,
        "metrics_mean": metrics_mean,
        "metrics_per_view": per_view,
        "self_render_summary": self_render_summary_path,
        "self_render_metrics_mean": self_render_metrics_mean,
        "selection": {
            "fusion_packet_ranges": args.fusion_packet_ranges,
            "fusion_max_packets": args.fusion_max_packets,
            "trajectory_render_chunk_size": args.trajectory_render_chunk_size,
        },
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def run_refine_comparison_fusions(
    runtime: ResplatRuntime,
    packet_groups: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    refine_steps = parse_refine_steps_spec(args.refine_steps)
    if refine_steps is None:
        return run_fusion_api(runtime, packet_groups, args)

    stage_summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    base_fusion_out = args.fusion_out_name
    base_self_out = args.self_render_out_name

    for step in refine_steps:
        stage = refine_stage_name(step)
        log(f"[FusionCompare] stage={stage}")
        summary = run_fusion_api(
            runtime,
            packet_groups,
            args,
            stage_override=stage,
            fusion_out_name_override=f"{base_fusion_out}_{stage}",
            self_render_out_name_override=f"{base_self_out}_{stage}",
        )
        stage_summaries[stage] = summary
        metrics = {} if summary is None else (summary.get("metrics_mean") or {})
        self_metrics = {} if summary is None else (summary.get("self_render_metrics_mean") or {})
        self_summary = None if summary is None else summary.get("self_render_summary")
        rows.append(
            {
                "stage": stage,
                "refine_step": int(step),
                "fusion_skipped": None if summary is None else bool(summary.get("fusion_skipped", False)),
                "fusion_psnr": metrics.get("psnr"),
                "fusion_ssim": metrics.get("ssim"),
                "fusion_lpips": metrics.get("lpips"),
                "self_render_psnr": self_metrics.get("psnr"),
                "self_render_ssim": self_metrics.get("ssim"),
                "num_gaussians": None if summary is None else summary.get("num_gaussians"),
                "num_gaussians_if_fused": None if summary is None else summary.get("num_gaussians_if_fused"),
                "fusion_summary": None if summary is None else str(abs_path(args.work_dir) / f"{base_fusion_out}_{stage}" / "summary.json"),
                "self_render_summary": self_summary,
            }
        )

    comparison = {
        "script_version": SCRIPT_VERSION,
        "refine_steps": [int(s) for s in refine_steps],
        "rows": rows,
        "stage_summaries": {
            k: None if v is None else str(abs_path(args.work_dir) / f"{base_fusion_out}_{k}" / "summary.json")
            for k, v in stage_summaries.items()
        },
    }
    out_path = abs_path(args.work_dir) / "refine_comparison_summary.json"
    save_json(out_path, comparison)
    return comparison


def parse_packet_ranges(spec: str, num_packets: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            indices = range(int(a), int(b) + 1)
        else:
            indices = [int(part)]
        for i in indices:
            if i < 0 or i >= num_packets:
                raise IndexError(f"Packet index {i} out of range 0..{num_packets-1}")
            if i not in seen:
                selected.append(i)
                seen.add(i)
    if not selected:
        raise ValueError(f"No packet selected by range spec: {spec}")
    return selected


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GT-pose-only TartanAir stereo -> ReSplat packets -> fusion pipeline")

    # Repositories and input.
    ap.add_argument("--resplat_repo", required=True)
    ap.add_argument("--left_dir", required=True)
    ap.add_argument("--right_dir", required=True)
    ap.add_argument("--gt_pose_file", required=True, help="Left-camera GT pose file. TartanAir pose_lcam_front.txt by default.")
    ap.add_argument("--right_gt_pose_file", default=None, help="Optional right-camera GT pose file, e.g. pose_rcam_front.txt.")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=None, help="Exclusive end index. Example: 0 50 selects 0..49.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--scene_name", default="P000")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--script_version", action="store_true", help="Print script version and exit.")

    # GT pose convention.
    ap.add_argument("--gt_out_name", default="gt_pose_inputs")
    ap.add_argument("--gt_convention", choices=["opencv_c2w", "resplat_tartanair_pose"], default="resplat_tartanair_pose")
    ap.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    ap.add_argument("--gt_matrix_convention", choices=["c2w", "w2c"], default="c2w")
    ap.add_argument(
        "--right_pose_source",
        choices=["auto", "fixed_baseline", "gt_first_relative", "gt_per_frame"],
        default="auto",
        help=(
            "How to provide the right-camera pose to ReSplat. "
            "auto: use gt_per_frame if --right_gt_pose_file is given, otherwise fixed_baseline. "
            "fixed_baseline: reproduce old behavior with a constant baseline transform. "
            "gt_first_relative: compute one constant left->right relative transform from the first selected GT frame. "
            "gt_per_frame: use the right GT pose for every frame."
        ),
    )
    ap.add_argument("--stereo_baseline", type=float, default=0.25000006)

    # ReSplat config/model.
    ap.add_argument("--resplat_experiment", default="tartanair_p000_ft")
    ap.add_argument("--resplat_override", action="append", default=[], help="Hydra override for ReSplat config/model.")
    ap.add_argument("--resplat_checkpoint", default=None, help="Override checkpointing.pretrained_model.")
    ap.add_argument("--resplat_out_name", default="resplat_runtime")
    ap.add_argument("--resplat_packet_stage", choices=["init", "final", "both"], default="init")
    ap.add_argument("--packet_stage_for_fusion", default="init", help="Stage to fuse in legacy mode. With --refine_steps, this is ignored and every requested refine_N stage is evaluated.")
    ap.add_argument("--refine_steps", default=None, help="Comma-separated refinement counts to save/evaluate, e.g. 0,1,2,4. 0 means raw init Gaussians.")
    ap.add_argument("--refine_use_target", type=str2bool, default=False, help="Pass target views to forward_update. Default false matches the API packet-generation path used before.")
    ap.add_argument("--resplat_target_camera", choices=["left", "right", "both"], default="left")
    ap.add_argument("--resplat_target_offset", type=int, default=0)
    ap.add_argument("--drop_invalid_target_offset", action="store_true")
    ap.add_argument("--save_packets", type=str2bool, default=True)
    ap.add_argument("--packet_out_name", default="gaussian_packets_api")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)

    # Fusion/rendering.
    ap.add_argument("--skip_fusion", action="store_true", help="Skip final packet concat and trajectory render. --self_render_packets still runs, so this is useful for local packet-quality tests on long sequences.")
    ap.add_argument("--fusion_out_name", default="fusion_eval_api")
    ap.add_argument(
        "--fusion_probe_mode",
        choices=["packet_last_only", "packet_trajectory"],
        default="packet_trajectory",
        help="Fuse selected packets once and render the final map along packet target trajectory.",
    )
    ap.add_argument("--fusion_packet_ranges", default=None)
    ap.add_argument("--fusion_max_packets", type=int, default=None)
    ap.add_argument("--trajectory_render_chunk_size", type=int, default=1)
    ap.add_argument("--fusion_compute_lpips", action="store_true")

    # Optional local packet self-render diagnostic.
    ap.add_argument("--self_render_packets", action="store_true", help="Render each individual packet back to its own target view.")
    ap.add_argument("--self_render_out_name", default="packet_self_render")

    return ap


def main() -> None:
    if "--script_version" in sys.argv:
        print(SCRIPT_VERSION)
        return
    args = build_argparser().parse_args()
    work_dir = abs_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    refine_steps = parse_refine_steps_spec(args.refine_steps)
    if refine_steps is None and args.packet_stage_for_fusion == "final" and args.resplat_packet_stage == "init":
        raise ValueError("--packet_stage_for_fusion final requires --resplat_packet_stage final or both")

    t0 = time.time()
    selected = select_stereo_frames(
        left_dir=abs_path(args.left_dir),
        right_dir=abs_path(args.right_dir),
        start_index=args.start_index,
        end_index=args.end_index,
        stride=args.stride,
        num_frames=args.num_frames,
        recursive=args.recursive,
    )
    log(
        f"Selected {len(selected.indices)} stereo pair(s): "
        f"original_index first={selected.indices[0]}, last={selected.indices[-1]} "
        f"(local_index first=0, last={len(selected.indices) - 1})"
    )

    gt_pose = load_gt_pose_stage(args, selected)
    runtime = load_resplat_runtime(args)
    packet_groups = infer_resplat_packets(runtime, gt_pose, args)
    fusion_summary = run_refine_comparison_fusions(runtime, packet_groups, args)

    run_summary = {
        "script_version": SCRIPT_VERSION,
        "work_dir": str(work_dir),
        "pose_source": "gt",
        "right_pose_source": gt_pose.right_pose_source_resolved,
        "selected_original_indices": [int(i) for i in selected.indices],
        "gt_pose_inputs": str(gt_pose.out_dir),
        "stereo_diagnostics": gt_pose.diagnostics,
        "refine_steps": None if refine_steps is None else [int(s) for s in refine_steps],
        "packet_counts": {k: len(v) for k, v in packet_groups.items()},
        "fusion_summary": None if fusion_summary is None else (str(work_dir / args.fusion_out_name / "summary.json") if refine_steps is None else str(work_dir / "refine_comparison_summary.json")),
        "fusion_skipped": bool(args.skip_fusion),
        "elapsed_sec": time.time() - t0,
    }
    save_json(work_dir / "run_summary.json", run_summary)
    log("Done.")
    log(f"  run summary: {work_dir / 'run_summary.json'}")
    log(f"  GT pose inputs: {gt_pose.out_dir}")
    if args.save_packets:
        log(f"  packets: {work_dir / args.packet_out_name}")
    if fusion_summary is not None:
        log(f"  fusion: {work_dir / args.fusion_out_name}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
