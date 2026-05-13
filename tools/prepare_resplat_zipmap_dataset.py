#!/usr/bin/env python3
"""
Prepare a ReSplat-style posed image folder from ZipMap outputs.

Typical use after pose evaluation:

  python tools/prepare_resplat_zipmap_dataset.py \
    --zipmap_dir outputs/zipmap/left_only_sparse_0_100 \
    --output_dir outputs/resplat_inputs/zipmap_left_only_sparse_0_100 \
    --pose_source aligned \
    --copy_mode symlink

Inputs expected from export_zipmap_predictions.py:
  zipmap_dir/
    predictions.npz                    # contains T_c2w_opencv, T_w2c_opencv
    frame_records.json
    images_zipmap_input/000000.png ... # optional, recommended for first ReSplat tests
    intrinsics.txt

If --pose_source aligned, this script reads:
  zipmap_dir/pose_eval/trajectory_pred_aligned_c2w_opencv.txt

Outputs:
  output_dir/
    images/000000.png ...
    pose.txt                           # tx ty tz qx qy qz qw for your ReSplat loader
    pose_resplat_tartanair_loader.txt  # same as pose.txt
    pose_c2w_opencv_tum.txt            # timestamp tx ty tz qx qy qz qw, diagnostic
    intrinsics.txt                     # filtered/copied per-frame intrinsics
    frame_records.json                 # filtered records mapped to output indices
    meta.json

Convention:
  ZipMap/OpenCV c2w is converted to the pose7 format expected by the provided loader:

      Twc = Twc_pose @ T_tartanCam_from_cvCam

  Therefore this script writes:

      Twc_pose = Twc_cv @ inverse(T_tartanCam_from_cvCam)

  so that the loader reconstructs Twc_cv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y", "t"):
        return True
    if v.lower() in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def tartan_from_cv_matrix(dtype=np.float64) -> np.ndarray:
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
        raise ValueError(f"Near-zero quaternion: {q}")
    q = q / n
    if q[-1] < 0:
        q = -q
    return q


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


def pose7_to_T(values: Sequence[float]) -> np.ndarray:
    if len(values) != 7:
        raise ValueError(f"Expected pose7, got {len(values)} values")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(values[3:7])
    T[:3, 3] = np.asarray(values[:3], dtype=np.float64)
    return T


def read_tum_or_pose7(path: Path) -> np.ndarray:
    """Read timestamp tx ty tz qx qy qz qw or tx ty tz qx qy qz qw."""
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        # First column may be non-numeric image name. In that case, use last 7 numeric columns.
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        if len(nums) < 7:
            continue
        pose7 = nums[-7:]
        poses.append(pose7_to_T(pose7))
    if not poses:
        raise ValueError(f"No poses loaded from {path}")
    return np.stack(poses, axis=0)


def load_raw_T_c2w(zipmap_dir: Path) -> np.ndarray:
    npz_path = zipmap_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    if "T_c2w_opencv" in data:
        T = np.asarray(data["T_c2w_opencv"], dtype=np.float64)
    elif "extrinsic" in data:
        # extrinsic is OpenCV w2c 3x4; invert to c2w.
        E = np.asarray(data["extrinsic"], dtype=np.float64)
        if E.ndim != 3 or E.shape[1:] != (3, 4):
            raise ValueError(f"Unexpected extrinsic shape: {E.shape}")
        T_w2c = np.tile(np.eye(4, dtype=np.float64), (E.shape[0], 1, 1))
        T_w2c[:, :3, :4] = E
        T = invert_se3(T_w2c)
    else:
        raise KeyError("predictions.npz must contain T_c2w_opencv or extrinsic")
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"Expected T_c2w [N,4,4], got {T.shape}")
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


def compute_resplat_loader_pose_from_cv(T_c2w_cv: np.ndarray) -> np.ndarray:
    T_tartan_from_cv = tartan_from_cv_matrix(dtype=T_c2w_cv.dtype)
    T_cv_from_tartan = np.linalg.inv(T_tartan_from_cv)
    return np.matmul(T_c2w_cv, T_cv_from_tartan[None])


def write_pose7(path: Path, T_c2w: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for T in T_c2w:
            q = rotmat_to_quat_xyzw(T[:3, :3])
            t = T[:3, 3]
            f.write(
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def write_tum(path: Path, T_c2w: np.ndarray, names: Optional[List[str]] = None) -> None:
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


def load_frame_records(zipmap_dir: Path, n_pred: int) -> List[dict]:
    path = zipmap_dir / "frame_records.json"
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
        if len(records) != n_pred:
            raise ValueError(f"frame_records length {len(records)} != pose count {n_pred}")
        return records
    # Fallback for older exports.
    return [
        {
            "pred_index": i,
            "camera_id": "unknown",
            "source_name": f"{i:06d}.png",
            "gt_index": i,
            "original_index": i,
        }
        for i in range(n_pred)
    ]


def parse_camera_ids(arg: str) -> Optional[set]:
    if arg.lower() in ("all", "*"):
        return None
    return {x.strip() for x in arg.split(",") if x.strip()}


def get_kept_indices(records: List[dict], camera_ids: Optional[set]) -> List[int]:
    if camera_ids is None:
        return list(range(len(records)))
    kept = [i for i, r in enumerate(records) if str(r.get("camera_id", "unknown")) in camera_ids]
    if not kept:
        raise ValueError(f"No frames kept for camera_ids={camera_ids}")
    return kept


def source_image_path(zipmap_dir: Path, rec: dict, out_index: int, image_source: str) -> Path:
    if image_source == "preprocessed":
        p = zipmap_dir / "images_zipmap_input" / f"{int(rec.get('pred_index', out_index)):06d}.png"
        if p.exists():
            return p
        # fallback if records were filtered earlier
        p2 = zipmap_dir / "images_zipmap_input" / f"{out_index:06d}.png"
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Could not find preprocessed image for pred_index={rec.get('pred_index')}")
    if image_source == "original":
        p = Path(str(rec.get("source_path", ""))).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Original source_path not found: {p}")
        return p
    raise ValueError(f"Unknown image_source: {image_source}")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "none":
        return
    else:
        raise ValueError(f"Unknown copy_mode: {mode}")


def load_intrinsics(zipmap_dir: Path, n_pred: int) -> Optional[np.ndarray]:
    path = zipmap_dir / "intrinsics.txt"
    if path.exists():
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            nums = [float(x) for x in s.split()]
            if len(nums) == 10:
                nums = nums[1:]
            if len(nums) == 9:
                rows.append(nums)
        if rows:
            K = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 3)
            if len(K) != n_pred:
                print(f"[Warn] intrinsics count {len(K)} != pose count {n_pred}; still writing filtered available rows.")
            return K
    npz_path = zipmap_dir / "predictions.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if "intrinsic" in data:
            K = np.asarray(data["intrinsic"], dtype=np.float64)
            if K.ndim == 3 and K.shape[1:] == (3, 3):
                return K
    return None


def write_intrinsics(path: Path, K: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, k in enumerate(K):
            flat = k.reshape(-1)
            f.write(str(i) + " " + " ".join(f"{v:.9f}" for v in flat) + "\n")


def load_poses(args, zipmap_dir: Path) -> Tuple[np.ndarray, str]:
    source = args.pose_source
    if source == "auto":
        aligned = zipmap_dir / "pose_eval" / "trajectory_pred_aligned_c2w_opencv.txt"
        source = "aligned" if aligned.exists() else "raw"
    if source == "raw":
        return load_raw_T_c2w(zipmap_dir), "raw_predictions_npz"
    if source == "aligned":
        p = Path(args.aligned_pose_file).expanduser().resolve() if args.aligned_pose_file else zipmap_dir / "pose_eval" / "trajectory_pred_aligned_c2w_opencv.txt"
        if not p.exists():
            raise FileNotFoundError(f"Aligned pose file not found: {p}")
        return read_tum_or_pose7(p), str(p)
    if source == "file":
        if not args.pose_file:
            raise ValueError("--pose_source file requires --pose_file")
        p = Path(args.pose_file).expanduser().resolve()
        return read_tum_or_pose7(p), str(p)
    raise ValueError(f"Unknown pose_source: {args.pose_source}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zipmap_dir", required=True, help="Output directory from export_zipmap_predictions_v3.py")
    ap.add_argument("--output_dir", required=True, help="Directory to create for ReSplat input")
    ap.add_argument("--pose_source", choices=["auto", "raw", "aligned", "file"], default="auto")
    ap.add_argument("--aligned_pose_file", default=None)
    ap.add_argument("--pose_file", default=None)
    ap.add_argument("--camera_ids", default="all", help="all, left, right, unknown, or comma-separated IDs")
    ap.add_argument("--image_source", choices=["preprocessed", "original"], default="preprocessed")
    ap.add_argument("--copy_mode", choices=["symlink", "copy", "hardlink", "none"], default="symlink")
    ap.add_argument("--overwrite", type=str2bool, default=True)
    args = ap.parse_args()

    zipmap_dir = Path(args.zipmap_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not zipmap_dir.is_dir():
        raise FileNotFoundError(zipmap_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    T_all, pose_source_resolved = load_poses(args, zipmap_dir)
    records_all = load_frame_records(zipmap_dir, n_pred=len(load_raw_T_c2w(zipmap_dir)))

    camera_ids = parse_camera_ids(args.camera_ids)
    kept_pred_indices = get_kept_indices(records_all, camera_ids)

    # If pose_source is aligned, evaluate_zipmap_pose_v2.py already filtered prediction poses
    # before writing trajectory_pred_aligned_c2w_opencv.txt. Therefore len(T_all) may equal
    # len(kept_pred_indices), not the full ZipMap prediction count.
    if args.pose_source in ("aligned", "auto") and len(T_all) == len(kept_pred_indices):
        T_kept = T_all
    elif len(T_all) == len(records_all):
        T_kept = T_all[kept_pred_indices]
    else:
        raise ValueError(
            f"Cannot map poses to records: pose_count={len(T_all)}, full_records={len(records_all)}, kept={len(kept_pred_indices)}. "
            "If using aligned poses, use the same --camera_ids as evaluation; for stereo left-only evaluation, pass --camera_ids left."
        )

    records_kept = [dict(records_all[i]) for i in kept_pred_indices]
    for out_i, rec in enumerate(records_kept):
        rec["output_index"] = out_i
        rec["old_pred_index"] = int(rec.get("pred_index", kept_pred_indices[out_i]))
        rec["pred_index"] = out_i

    # Images
    image_out_dir = out_dir / "images"
    source_names: List[str] = []
    for out_i, rec in enumerate(records_kept):
        src = source_image_path(zipmap_dir, rec | {"pred_index": rec.get("old_pred_index", out_i)}, out_i, args.image_source)
        ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".png"
        dst = image_out_dir / f"{out_i:06d}{ext}"
        link_or_copy(src, dst, args.copy_mode)
        source_names.append(src.name)

    # Poses
    T_resplat_pose = compute_resplat_loader_pose_from_cv(T_kept)
    write_pose7(out_dir / "pose.txt", T_resplat_pose)
    write_pose7(out_dir / "pose_resplat_tartanair_loader.txt", T_resplat_pose)
    write_tum(out_dir / "pose_resplat_tartanair_loader_tum.txt", T_resplat_pose, names=source_names)
    write_tum(out_dir / "pose_c2w_opencv_tum.txt", T_kept, names=source_names)

    # Intrinsics
    K_all = load_intrinsics(zipmap_dir, n_pred=len(records_all))
    if K_all is not None:
        if len(K_all) == len(records_all):
            K_kept = K_all[kept_pred_indices]
        elif len(K_all) == len(kept_pred_indices):
            K_kept = K_all
        else:
            K_kept = None
            print(f"[Warn] Not writing intrinsics because count {len(K_all)} cannot map to kept {len(kept_pred_indices)}.")
        if K_kept is not None:
            write_intrinsics(out_dir / "intrinsics.txt", K_kept)

    (out_dir / "frame_records.json").write_text(json.dumps(records_kept, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "source_images.txt").write_text(
        "\n".join(f"{i:06d} {name}" for i, name in enumerate(source_names)) + "\n",
        encoding="utf-8",
    )

    meta = {
        "zipmap_dir": str(zipmap_dir),
        "pose_source_requested": args.pose_source,
        "pose_source_resolved": pose_source_resolved,
        "camera_ids": args.camera_ids,
        "image_source": args.image_source,
        "copy_mode": args.copy_mode,
        "num_frames": int(len(T_kept)),
        "kept_old_pred_indices": [int(i) for i in kept_pred_indices],
        "note": "pose.txt is tx ty tz qx qy qz qw designed for the provided ReSplat _build_Twc_from_pose().",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Done] Wrote ReSplat input dataset: {out_dir}")
    print(f"  frames: {len(T_kept)}")
    print(f"  images: {image_out_dir}")
    print(f"  pose:   {out_dir / 'pose.txt'}")
    print(f"  source: {pose_source_resolved}")


if __name__ == "__main__":
    main()
