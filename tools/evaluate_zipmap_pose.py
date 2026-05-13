#!/usr/bin/env python3
"""
Evaluate ZipMap pose outputs before feeding them into ReSplat.

Main use cases
--------------
1) Compare ZipMap camera trajectory with GT trajectory using Sim(3)-aligned ATE/RPE.
2) Diagnose stereo left-right pose consistency when ZipMap was run with stereo interleave.

Expected ZipMap directory
-------------------------
This script expects the output directory created by export_zipmap_predictions_v2.py:

  output_dir/
    predictions.npz              # contains T_c2w_opencv, T_w2c_opencv
    meta.json                    # contains sampled image_paths/image_names/stereo metadata and selected_gt_indices
    frame_records.json           # optional; emitted by export_zipmap_predictions_v3.py
    stereo_pairs.json            # optional

Pose convention
---------------
ZipMap exported T_c2w_opencv is OpenCV camera-to-world:
  camera axes: x right, y down, z forward.

If your GT file is the same TartanAir-style pose consumed by the provided ReSplat loader,
use:
  --gt_convention resplat_tartanair_pose

This applies the same camera-axis conversion as your loader:
  Twc_cv = Twc_pose @ T_tartanCam_from_cvCam

Outputs
-------
output_eval_dir/
  summary.json
  matched_indices.txt
  trajectory_pred_aligned_c2w_opencv.txt
  trajectory_gt_matched_c2w_opencv.txt
  per_frame_errors.csv
  rpe_errors.csv
  plots/trajectory_xy.png
  plots/ate_per_frame.png
  plots/rpe_translation.png
  stereo_baseline.csv                 # if stereo metadata is available
  plots/stereo_baseline.png            # if stereo metadata is available
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Basic SE(3) / quaternion utilities
# -----------------------------------------------------------------------------


def tartan_from_cv_matrix(dtype=np.float64) -> np.ndarray:
    """Same matrix as the user's ReSplat loader: Twc_cv = Twc_pose @ T_tartanCam_from_cvCam."""
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


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {q}")
    q = q / n
    if q[-1] < 0:  # stable sign for xyzw
        q = -q
    return q


def quat_xyzw_to_rotmat(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(np.array(q, dtype=np.float64))
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


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion(np.array([qx, qy, qz, qw], dtype=np.float64))


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
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# Loading trajectories
# -----------------------------------------------------------------------------


def read_numeric_pose_file(path: Path) -> Tuple[Optional[List[str]], np.ndarray, str]:
    """
    Read a loose numeric pose file.

    Supported row formats:
      timestamp tx ty tz qx qy qz qw   -> TUM pose file
      tx ty tz qx qy qz qw             -> pose7 file
      idx + flattened 3x4              -> matrix34 with leading index
      flattened 3x4                    -> matrix34
      idx + flattened 4x4              -> matrix44 with leading index
      flattened 4x4                    -> matrix44
    """
    rows: List[List[float]] = []
    stamps: List[str] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
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
    gt_pose_file: Path,
    gt_convention: str,
    quat_order: str,
    gt_matrix_convention: str,
) -> np.ndarray:
    _, data, fmt = read_numeric_pose_file(gt_pose_file)

    if fmt in {"pose7", "tum_pose7"}:
        T_pose = np.stack([make_T_c2w_from_pose7(row, quat_order=quat_order) for row in data], axis=0)
        if gt_convention == "opencv_c2w":
            return T_pose
        if gt_convention == "resplat_tartanair_pose":
            return np.matmul(T_pose, tartan_from_cv_matrix()[None])
        raise ValueError(f"Unsupported gt_convention for pose7: {gt_convention}")

    if fmt == "matrix34":
        T = np.tile(np.eye(4, dtype=np.float64), (data.shape[0], 1, 1))
        T[:, :3, :4] = data
    else:
        T = data.astype(np.float64)

    if gt_matrix_convention == "w2c":
        T = invert_se3(T)
    elif gt_matrix_convention != "c2w":
        raise ValueError(f"Unknown gt_matrix_convention: {gt_matrix_convention}")

    if gt_convention == "opencv_c2w":
        return T
    if gt_convention == "resplat_tartanair_pose":
        return np.matmul(T, tartan_from_cv_matrix()[None])
    raise ValueError(f"Unsupported gt_convention: {gt_convention}")


def load_zipmap_pred(zipmap_dir: Path) -> Tuple[np.ndarray, Dict]:
    npz_path = zipmap_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing predictions.npz: {npz_path}")
    data = np.load(npz_path)
    if "T_c2w_opencv" in data:
        T = np.asarray(data["T_c2w_opencv"], dtype=np.float64)
    elif "T_w2c_opencv" in data:
        T = invert_se3(np.asarray(data["T_w2c_opencv"], dtype=np.float64))
    else:
        raise KeyError("predictions.npz must contain T_c2w_opencv or T_w2c_opencv")
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"Expected T_c2w_opencv shape [S,4,4], got {T.shape}")

    meta_path = zipmap_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return T, meta


# -----------------------------------------------------------------------------
# Matching and alignment
# -----------------------------------------------------------------------------


def parse_last_integer(text: str) -> Optional[int]:
    nums = re.findall(r"\d+", Path(text).stem)
    if not nums:
        return None
    return int(nums[-1])


def get_image_names_for_matching(meta: Dict, pred_len: int) -> List[str]:
    names = meta.get("image_names") or []
    if len(names) == pred_len:
        return [str(x) for x in names]
    paths = meta.get("image_paths") or []
    if len(paths) == pred_len:
        return [Path(str(x)).name for x in paths]
    return [f"{i:06d}" for i in range(pred_len)]


def filter_prediction_by_camera_ids(
    T_pred: np.ndarray,
    meta: Dict,
    camera_ids_csv: Optional[str],
) -> Tuple[np.ndarray, Dict, List[int]]:
    """Filter prediction poses using meta.frame_records camera_id entries."""
    if camera_ids_csv is None or camera_ids_csv.strip().lower() in {"", "all", "none"}:
        return T_pred, meta, list(range(T_pred.shape[0]))

    wanted = {x.strip() for x in camera_ids_csv.split(",") if x.strip()}
    records = meta.get("frame_records") or []
    if len(records) != T_pred.shape[0]:
        raise ValueError(
            "--eval_camera_ids requires meta.frame_records with one record per prediction. "
            f"Got {len(records)} records for {T_pred.shape[0]} predictions."
        )
    keep = [i for i, r in enumerate(records) if str(r.get("camera_id")) in wanted]
    if not keep:
        raise ValueError(f"No prediction frames matched --eval_camera_ids={camera_ids_csv}")

    filtered_meta = dict(meta)
    filtered_records = [records[i] for i in keep]
    filtered_meta["frame_records"] = filtered_records
    filtered_meta["image_names"] = [str(r.get("source_name", f"{i:06d}")) for i, r in zip(keep, filtered_records)]
    filtered_meta["image_paths"] = [str(r.get("source_path", "")) for r in filtered_records]
    filtered_meta["selected_original_indices"] = [int(r.get("original_index", r.get("gt_index", i))) for i, r in zip(keep, filtered_records)]
    filtered_meta["selected_gt_indices"] = [int(r.get("gt_index", r.get("original_index", i))) for i, r in zip(keep, filtered_records)]
    return T_pred[keep], filtered_meta, keep


def match_gt_to_pred(
    T_pred: np.ndarray,
    T_gt_all: np.ndarray,
    meta: Dict,
    mode: str,
) -> Tuple[np.ndarray, List[int], List[str]]:
    pred_len = T_pred.shape[0]
    if mode == "auto":
        if meta.get("selected_gt_indices") is not None or meta.get("frame_records") is not None:
            mode = "meta_index"
        else:
            mode = "sequential" if T_gt_all.shape[0] == pred_len else "filename_index"

    if mode == "sequential":
        if T_gt_all.shape[0] < pred_len:
            raise ValueError(f"GT has fewer poses ({T_gt_all.shape[0]}) than predictions ({pred_len}).")
        indices = list(range(pred_len))
        names = get_image_names_for_matching(meta, pred_len)
        return T_gt_all[:pred_len], indices, names

    if mode == "meta_index":
        indices_raw = meta.get("selected_gt_indices")
        if indices_raw is None:
            records = meta.get("frame_records") or []
            if len(records) == pred_len:
                indices_raw = [r.get("gt_index", r.get("original_index")) for r in records]
        if indices_raw is None or len(indices_raw) != pred_len:
            raise ValueError(
                "meta_index matching requires meta.selected_gt_indices or frame_records with one gt_index/original_index per prediction. "
                "Re-export with export_zipmap_predictions_v3.py."
            )
        indices = [int(x) for x in indices_raw]
        for idx in indices:
            if idx < 0 or idx >= T_gt_all.shape[0]:
                raise IndexError(f"GT index {idx} is out of range for GT length {T_gt_all.shape[0]}")
        names = get_image_names_for_matching(meta, pred_len)
        return T_gt_all[indices], indices, names

    if mode == "filename_index":
        names = get_image_names_for_matching(meta, pred_len)
        indices: List[int] = []
        for name in names:
            idx = parse_last_integer(name)
            if idx is None:
                raise ValueError(f"Could not parse frame index from image name: {name}")
            if idx < 0 or idx >= T_gt_all.shape[0]:
                raise IndexError(f"Parsed index {idx} from {name}, but GT length is {T_gt_all.shape[0]}")
            indices.append(idx)
        return T_gt_all[indices], indices, names

    raise ValueError(f"Unknown match mode: {mode}")

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find s,R,t such that dst ~= s * R @ src + t. src/dst shape [N,3]."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must both be [N,3], got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("At least 3 poses are recommended for Sim(3) alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(X * X) / n
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def apply_similarity_to_poses(T_pred: np.ndarray, scale: float, R_align: np.ndarray, t_align: np.ndarray) -> np.ndarray:
    out = T_pred.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", R_align, T_pred[:, :3, :3])
    out[:, :3, 3] = scale * (T_pred[:, :3, 3] @ R_align.T) + t_align
    return out


# -----------------------------------------------------------------------------
# Metrics and exports
# -----------------------------------------------------------------------------


def compute_ate(T_pred_aligned: np.ndarray, T_gt: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(T_pred_aligned[:, :3, 3] - T_gt[:, :3, 3], axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "min": float(np.min(err)),
        "max": float(np.max(err)),
        "std": float(np.std(err)),
    }


def compute_rpe(T_pred_aligned: np.ndarray, T_gt: np.ndarray, delta: int) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    if delta <= 0:
        raise ValueError("delta must be positive")
    rows: List[Dict[str, float]] = []
    for i in range(0, T_pred_aligned.shape[0] - delta):
        j = i + delta
        d_pred = invert_se3(T_pred_aligned[i]) @ T_pred_aligned[j]
        d_gt = invert_se3(T_gt[i]) @ T_gt[j]
        err_T = invert_se3(d_gt) @ d_pred
        rows.append(
            {
                "i": i,
                "j": j,
                "translation_error": float(np.linalg.norm(err_T[:3, 3])),
                "rotation_error_deg": float(rotation_angle_deg(err_T[:3, :3])),
                "gt_motion": float(np.linalg.norm(d_gt[:3, 3])),
                "pred_motion": float(np.linalg.norm(d_pred[:3, 3])),
            }
        )
    if not rows:
        return rows, {}
    trans = np.array([r["translation_error"] for r in rows], dtype=np.float64)
    rot = np.array([r["rotation_error_deg"] for r in rows], dtype=np.float64)
    return rows, {
        "delta": delta,
        "translation_rmse": float(np.sqrt(np.mean(trans ** 2))),
        "translation_mean": float(np.mean(trans)),
        "translation_median": float(np.median(trans)),
        "rotation_deg_mean": float(np.mean(rot)),
        "rotation_deg_median": float(np.median(rot)),
        "rotation_deg_rmse": float(np.sqrt(np.mean(rot ** 2))),
    }


def write_pose_tum(path: Path, T_c2w: np.ndarray, names: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, T in enumerate(T_c2w):
            stamp = Path(names[i]).stem if names and i < len(names) else f"{i:.6f}"
            q = rotmat_to_quat_xyzw(T[:3, :3])
            t = T[:3, 3]
            f.write(
                f"{stamp} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plots(out_dir: Path, T_pred_aligned: np.ndarray, T_gt: np.ndarray, ate_errors: np.ndarray, rpe_rows: List[Dict[str, float]]) -> None:
    if plt is None:
        print("[Plot] matplotlib not available; skipping plots.")
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot(T_gt[:, 0, 3], T_gt[:, 2, 3], label="GT")
    plt.plot(T_pred_aligned[:, 0, 3], T_pred_aligned[:, 2, 3], label="ZipMap aligned")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Trajectory top-down view (x-z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "trajectory_xz.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(ate_errors)), ate_errors)
    plt.xlabel("Matched frame")
    plt.ylabel("ATE translation error")
    plt.title("Per-frame ATE after alignment")
    plt.tight_layout()
    plt.savefig(plot_dir / "ate_per_frame.png", dpi=160)
    plt.close()

    if rpe_rows:
        xs = [r["i"] for r in rpe_rows]
        trans = [r["translation_error"] for r in rpe_rows]
        rot = [r["rotation_error_deg"] for r in rpe_rows]

        plt.figure(figsize=(9, 4))
        plt.plot(xs, trans)
        plt.xlabel("Start frame")
        plt.ylabel("RPE translation error")
        plt.title("Relative pose translation error")
        plt.tight_layout()
        plt.savefig(plot_dir / "rpe_translation.png", dpi=160)
        plt.close()

        plt.figure(figsize=(9, 4))
        plt.plot(xs, rot)
        plt.xlabel("Start frame")
        plt.ylabel("RPE rotation error [deg]")
        plt.title("Relative pose rotation error")
        plt.tight_layout()
        plt.savefig(plot_dir / "rpe_rotation.png", dpi=160)
        plt.close()


def evaluate_stereo_baseline(
    T_pred: np.ndarray,
    meta: Dict,
    out_dir: Path,
    known_baseline: Optional[float],
) -> Optional[Dict[str, object]]:
    mode = meta.get("stereo_pair_mode")
    num_pairs = meta.get("stereo_num_pairs")
    if mode not in {"interleave", "concat_left_then_right"} or not num_pairs:
        return None

    n = int(num_pairs)
    rows: List[Dict[str, float]] = []
    for pair_idx in range(n):
        if mode == "interleave":
            li, ri = 2 * pair_idx, 2 * pair_idx + 1
        else:
            li, ri = pair_idx, pair_idx + n
        if ri >= T_pred.shape[0]:
            break
        T_L = T_pred[li]
        T_R = T_pred[ri]
        rel = invert_se3(T_L) @ T_R
        baseline = float(np.linalg.norm(T_R[:3, 3] - T_L[:3, 3]))
        rel_rot_deg = float(rotation_angle_deg(rel[:3, :3]))
        rows.append(
            {
                "pair_index": pair_idx,
                "left_frame": li,
                "right_frame": ri,
                "baseline_est": baseline,
                "relative_rotation_deg": rel_rot_deg,
                "rel_tx": float(rel[0, 3]),
                "rel_ty": float(rel[1, 3]),
                "rel_tz": float(rel[2, 3]),
            }
        )

    if not rows:
        return None

    baselines = np.array([r["baseline_est"] for r in rows], dtype=np.float64)
    rots = np.array([r["relative_rotation_deg"] for r in rows], dtype=np.float64)
    summary: Dict[str, object] = {
        "stereo_pair_mode": mode,
        "num_pairs_evaluated": len(rows),
        "baseline_mean": float(np.mean(baselines)),
        "baseline_median": float(np.median(baselines)),
        "baseline_std": float(np.std(baselines)),
        "baseline_min": float(np.min(baselines)),
        "baseline_max": float(np.max(baselines)),
        "baseline_coeff_var": float(np.std(baselines) / max(abs(np.mean(baselines)), 1e-12)),
        "relative_rotation_deg_mean": float(np.mean(rots)),
        "relative_rotation_deg_median": float(np.median(rots)),
    }
    if known_baseline is not None and known_baseline > 0:
        summary["known_baseline"] = float(known_baseline)
        summary["scale_from_median_baseline"] = float(known_baseline / max(float(np.median(baselines)), 1e-12))
        summary["scale_from_mean_baseline"] = float(known_baseline / max(float(np.mean(baselines)), 1e-12))

    save_csv(out_dir / "stereo_baseline.csv", rows)

    if plt is not None:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 4))
        plt.plot([r["pair_index"] for r in rows], baselines)
        if known_baseline is not None and known_baseline > 0:
            plt.axhline(float(known_baseline), linestyle="--", label="Known baseline")
            plt.legend()
        plt.xlabel("Stereo pair index")
        plt.ylabel("Estimated baseline")
        plt.title("ZipMap left-right baseline stability")
        plt.tight_layout()
        plt.savefig(plot_dir / "stereo_baseline.png", dpi=160)
        plt.close()

    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate ZipMap pose outputs against GT and/or stereo rig consistency.")
    p.add_argument("--zipmap_dir", required=True, type=str, help="Output directory from export_zipmap_predictions_v2.py")
    p.add_argument("--output_dir", default=None, type=str, help="Evaluation output directory. Default: zipmap_dir/pose_eval")

    p.add_argument("--gt_pose_file", default=None, type=str, help="Optional GT pose file for ATE/RPE evaluation")
    p.add_argument(
        "--gt_convention",
        choices=["opencv_c2w", "resplat_tartanair_pose"],
        default="resplat_tartanair_pose",
        help="GT pose convention. Use resplat_tartanair_pose for the pose file consumed by your current ReSplat loader.",
    )
    p.add_argument("--gt_quat_order", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument(
        "--gt_matrix_convention",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Only used if GT file rows are 3x4/4x4 matrices.",
    )
    p.add_argument(
        "--match",
        choices=["auto", "sequential", "filename_index", "meta_index"],
        default="auto",
        help="How to select GT rows corresponding to ZipMap sampled images. meta_index uses selected_gt_indices saved by export_zipmap_predictions_v3.py.",
    )
    p.add_argument(
        "--eval_camera_ids",
        default="all",
        type=str,
        help="Optional camera_id filter using frame_records.json, e.g. 'left' for stereo interleave outputs. Default: all.",
    )
    p.add_argument(
        "--alignment",
        choices=["sim3", "se3", "none"],
        default="sim3",
        help="Trajectory alignment. sim3 is recommended for affine-invariant ZipMap checkpoints.",
    )
    p.add_argument("--rpe_delta", type=int, default=1, help="Frame delta for RPE. Default: 1")

    p.add_argument("--known_stereo_baseline", type=float, default=None, help="Optional physical stereo baseline for scale diagnosis")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    zipmap_dir = Path(args.zipmap_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else zipmap_dir / "pose_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    T_pred, meta = load_zipmap_pred(zipmap_dir)
    T_pred, meta, kept_pred_indices = filter_prediction_by_camera_ids(T_pred, meta, args.eval_camera_ids)
    print(f"[Load] ZipMap poses: {T_pred.shape[0]} frames from {zipmap_dir}")
    if args.eval_camera_ids.strip().lower() not in {"", "all", "none"}:
        print(f"[Filter] eval_camera_ids={args.eval_camera_ids}; kept prediction indices: {kept_pred_indices[:10]}{'...' if len(kept_pred_indices) > 10 else ''}")

    summary: Dict[str, object] = {
        "zipmap_dir": str(zipmap_dir),
        "num_pred_poses": int(T_pred.shape[0]),
        "eval_camera_ids": args.eval_camera_ids,
        "kept_pred_indices": kept_pred_indices,
        "alignment": args.alignment,
        "rpe_delta": int(args.rpe_delta),
    }

    stereo_summary = evaluate_stereo_baseline(T_pred, meta, out_dir, args.known_stereo_baseline)
    if stereo_summary is not None:
        summary["stereo_baseline"] = stereo_summary
        print(
            "[Stereo] baseline median/mean/std = "
            f"{stereo_summary['baseline_median']:.6f} / "
            f"{stereo_summary['baseline_mean']:.6f} / "
            f"{stereo_summary['baseline_std']:.6f}"
        )

    if args.gt_pose_file is not None:
        gt_path = Path(args.gt_pose_file).expanduser().resolve()
        T_gt_all = load_gt_trajectory(gt_path, args.gt_convention, args.gt_quat_order, args.gt_matrix_convention)
        T_gt, gt_indices, names = match_gt_to_pred(T_pred, T_gt_all, meta, args.match)
        print(f"[GT] Matched {len(gt_indices)} GT poses from {gt_path}")

        pred_centers = T_pred[:, :3, 3]
        gt_centers = T_gt[:, :3, 3]
        if args.alignment == "sim3":
            scale, R_align, t_align = umeyama_alignment(pred_centers, gt_centers, with_scale=True)
        elif args.alignment == "se3":
            scale, R_align, t_align = umeyama_alignment(pred_centers, gt_centers, with_scale=False)
        else:
            scale, R_align, t_align = 1.0, np.eye(3), np.zeros(3)

        T_pred_aligned = apply_similarity_to_poses(T_pred, scale, R_align, t_align)
        ate_vec = np.linalg.norm(T_pred_aligned[:, :3, 3] - T_gt[:, :3, 3], axis=1)
        ate = compute_ate(T_pred_aligned, T_gt)
        rpe_rows, rpe_summary = compute_rpe(T_pred_aligned, T_gt, delta=args.rpe_delta)

        frame_rows: List[Dict[str, object]] = []
        for i, (name, gt_idx, err) in enumerate(zip(names, gt_indices, ate_vec)):
            frame_rows.append(
                {
                    "matched_frame": i,
                    "image_name": name,
                    "gt_index": gt_idx,
                    "ate_translation_error": float(err),
                    "pred_x": float(T_pred_aligned[i, 0, 3]),
                    "pred_y": float(T_pred_aligned[i, 1, 3]),
                    "pred_z": float(T_pred_aligned[i, 2, 3]),
                    "gt_x": float(T_gt[i, 0, 3]),
                    "gt_y": float(T_gt[i, 1, 3]),
                    "gt_z": float(T_gt[i, 2, 3]),
                }
            )

        (out_dir / "matched_indices.txt").write_text(
            "\n".join(f"{i} {name} {idx}" for i, (name, idx) in enumerate(zip(names, gt_indices))) + "\n",
            encoding="utf-8",
        )
        save_csv(out_dir / "per_frame_errors.csv", frame_rows)
        save_csv(out_dir / "rpe_errors.csv", rpe_rows)
        write_pose_tum(out_dir / "trajectory_pred_aligned_c2w_opencv.txt", T_pred_aligned, names=names)
        write_pose_tum(out_dir / "trajectory_gt_matched_c2w_opencv.txt", T_gt, names=names)
        make_plots(out_dir, T_pred_aligned, T_gt, ate_vec, rpe_rows)

        summary["gt_pose_file"] = str(gt_path)
        summary["gt_convention"] = args.gt_convention
        summary["gt_quat_order"] = args.gt_quat_order
        summary["match"] = args.match
        summary["num_matched"] = int(len(gt_indices))
        summary["sim_or_se3_alignment"] = {
            "scale": float(scale),
            "R": R_align.tolist(),
            "t": t_align.tolist(),
        }
        summary["ate"] = ate
        summary["rpe"] = rpe_summary

        print(
            "[ATE] RMSE/mean/median = "
            f"{ate['rmse']:.6f} / {ate['mean']:.6f} / {ate['median']:.6f}"
        )
        if rpe_summary:
            print(
                "[RPE] trans_mean / rot_mean_deg = "
                f"{rpe_summary['translation_mean']:.6f} / {rpe_summary['rotation_deg_mean']:.6f}"
            )
        print(f"[Align] scale = {scale:.9f}")
    else:
        print("[GT] No --gt_pose_file provided. Skipping ATE/RPE; only stereo diagnostics are computed if available.")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Done] Evaluation written to: {out_dir}")


if __name__ == "__main__":
    main()
