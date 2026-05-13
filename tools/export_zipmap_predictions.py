#!/usr/bin/env python3
"""
Offline ZipMap exporter for pose/depth/point outputs.

Purpose
-------
This script removes the Gradio/UI layer from demo_gradio_zipmap.py and exports
ZipMap predictions in files that are easier to connect to ReSplat/MVSplat-style
posed-image loaders.

Main outputs
------------
output_dir/
  predictions.npz                         # compact ZipMap outputs
  meta.json                               # image list, sampling, stereo metadata, original GT indices, shapes, options
  frame_records.json                      # one record per ZipMap input frame, including original_index / gt_index
  selected_gt_indices.txt                 # pred_index gt_index camera_id source_name
  poses_w2c_opencv.txt                    # OpenCV camera-from-world, 3x4 flattened
  poses_c2w_opencv_tum.txt                # timestamp tx ty tz qx qy qz qw, OpenCV c2w
  poses_resplat_tartanair_loader.txt      # tx ty tz qx qy qz qw, compatible with your _build_Twc_from_pose()
  intrinsics.txt                          # one 3x3 intrinsic per frame, flattened
  images_zipmap_input/000000.png ...      # preprocessed images matching exported intrinsics
  stereo_pairs.json                       # optional, if stereo_left/right dirs are used
  depth/000000.npy ...                    # optional per-frame depth
  points/world_points_from_depth.npy      # optional dense point map from depth
  points/world_points.npy                 # optional dense point map from local point head

Important convention
--------------------
ZipMap's pose_encoding_to_extri_intri returns extrinsics as OpenCV w2c/Tcw.
For ReSplat code that reads tx ty tz qx qy qz qw and then applies:

    Twc = Twc_pose @ T_tartanCam_from_cvCam

this script writes poses_resplat_tartanair_loader.txt as:

    Twc_pose = Twc_cv @ inv(T_tartanCam_from_cvCam)

so that the loader reconstructs Twc_cv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# Same model config as the official demo_gradio_zipmap.py.
MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": {
            "bias": True,
            "head_dim": 1024,
            "inter_multi": 2,
            "base_lr": 0.01,
            "muon_update_steps": 5,
            "use_gate_fn": True,
        },
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def find_repo_root(repo_root_arg: Optional[str]) -> Path:
    if repo_root_arg is not None:
        root = Path(repo_root_arg).expanduser().resolve()
    else:
        # If this script is placed in tools/, repo root is normally parent of tools/.
        here = Path(__file__).resolve()
        candidates = [Path.cwd().resolve(), here.parent.resolve(), here.parent.parent.resolve()]
        root = next((c for c in candidates if (c / "zipmap").is_dir()), Path.cwd().resolve())
    if not (root / "zipmap").is_dir():
        raise FileNotFoundError(
            f"Could not find ZipMap package under repo root: {root}. "
            "Run from the ZipMap repository or pass --repo_root /path/to/ZipMap."
        )
    return root


def collect_image_paths(image_dir: str, recursive: bool = False) -> List[Path]:
    root = Path(image_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"image_dir does not exist or is not a directory: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    image_paths = sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {root}")
    return image_paths


def make_sample_indices(
    n: int,
    start_index: int = 0,
    end_index: Optional[int] = None,
    stride: int = 1,
    max_count: Optional[int] = None,
) -> List[int]:
    """Return indices into the original lexically sorted image list."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    start = max(0, start_index)
    end = n if end_index is None else min(n, end_index)
    if start >= end:
        raise ValueError(f"Invalid range: start_index={start_index}, end_index={end_index}, length={n}")
    indices = list(range(start, end, stride))
    if max_count is not None:
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")
        indices = indices[:max_count]
    if len(indices) == 0:
        raise ValueError("Sampling produced zero images.")
    return indices


def apply_sequence_sampling(
    image_paths: List[Path],
    start_index: int = 0,
    end_index: Optional[int] = None,
    stride: int = 1,
    max_images: Optional[int] = None,
) -> Tuple[List[Path], List[int]]:
    """Apply deterministic frame sampling after lexical sorting and preserve original sorted indices."""
    indices = make_sample_indices(
        len(image_paths),
        start_index=start_index,
        end_index=end_index,
        stride=stride,
        max_count=max_images,
    )
    return [image_paths[i] for i in indices], indices


def collect_stereo_image_paths(args) -> Tuple[List[Path], List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Collect stereo images as pair-indexed paths, then flatten according to stereo_pair_mode.

    Returns:
      flattened: image paths fed to ZipMap
      pair_meta: sampled stereo-pair records with original pair indices
      frame_records: one record per ZipMap input frame; gt_index is the original sorted row index
                     to use when the GT file has one row per left/right time step.
    """
    left_paths = collect_image_paths(args.stereo_left_dir, recursive=args.recursive)
    right_paths = collect_image_paths(args.stereo_right_dir, recursive=args.recursive)
    if len(left_paths) != len(right_paths):
        raise ValueError(f"Stereo directories have different counts: left={len(left_paths)}, right={len(right_paths)}")

    pair_indices = make_sample_indices(
        len(left_paths),
        start_index=args.start_index,
        end_index=args.end_index,
        stride=args.stride,
        max_count=args.max_pairs,
    )

    flattened: List[Path] = []
    pair_meta: List[Dict[str, object]] = []
    frame_records: List[Dict[str, object]] = []

    def append_frame(path: Path, camera_id: str, original_pair_index: int, sampled_pair_index: int) -> None:
        pred_index = len(flattened)
        flattened.append(path)
        frame_records.append(
            {
                "pred_index": pred_index,
                "source_path": str(path),
                "source_name": path.name,
                "camera_id": camera_id,
                "sampled_pair_index": sampled_pair_index,
                "original_pair_index": original_pair_index,
                "original_index": original_pair_index,
                "gt_index": original_pair_index,
            }
        )

    for sampled_pair_index, original_pair_index in enumerate(pair_indices):
        left = left_paths[original_pair_index]
        right = right_paths[original_pair_index]
        pair_meta.append(
            {
                "sampled_pair_index": sampled_pair_index,
                "original_pair_index": original_pair_index,
                "left": str(left),
                "right": str(right),
            }
        )
        if args.stereo_pair_mode == "left_only":
            append_frame(left, "left", original_pair_index, sampled_pair_index)
        elif args.stereo_pair_mode == "right_only":
            append_frame(right, "right", original_pair_index, sampled_pair_index)
        elif args.stereo_pair_mode == "interleave":
            append_frame(left, "left", original_pair_index, sampled_pair_index)
            append_frame(right, "right", original_pair_index, sampled_pair_index)
        elif args.stereo_pair_mode == "concat_left_then_right":
            pass
        else:
            raise ValueError(f"Unknown stereo_pair_mode: {args.stereo_pair_mode}")

    if args.stereo_pair_mode == "concat_left_then_right":
        for sampled_pair_index, original_pair_index in enumerate(pair_indices):
            append_frame(left_paths[original_pair_index], "left", original_pair_index, sampled_pair_index)
        for sampled_pair_index, original_pair_index in enumerate(pair_indices):
            append_frame(right_paths[original_pair_index], "right", original_pair_index, sampled_pair_index)

    return flattened, pair_meta, frame_records

def save_tensor_images_as_png(images_chw: torch.Tensor, out_dir: Path, names: List[str]) -> None:
    """Save preprocessed image tensor [S,3,H,W] in [0,1] as PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = images_chw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    for i, img in enumerate(images):
        arr = np.transpose(img, (1, 2, 0))
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
        stem = f"{i:06d}"
        Image.fromarray(arr).save(out_dir / f"{stem}.png")
    with open(out_dir / "source_names.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(names):
            f.write(f"{i:06d}.png {name}\n")


def to_homogeneous_4x4(T_3x4: np.ndarray) -> np.ndarray:
    if T_3x4.ndim != 3 or T_3x4.shape[1:] != (3, 4):
        raise ValueError(f"Expected [S,3,4], got {T_3x4.shape}")
    S = T_3x4.shape[0]
    T = np.tile(np.eye(4, dtype=T_3x4.dtype), (S, 1, 1))
    T[:, :3, :4] = T_3x4
    return T


def invert_se3_np(T: np.ndarray) -> np.ndarray:
    """Invert batched SE(3), shape [S,4,4]."""
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"Expected [S,4,4], got {T.shape}")
    R = T[:, :3, :3]
    t = T[:, :3, 3:4]
    R_inv = np.transpose(R, (0, 2, 1))
    t_inv = -np.matmul(R_inv, t)
    out = np.tile(np.eye(4, dtype=T.dtype), (T.shape[0], 1, 1))
    out[:, :3, :3] = R_inv
    out[:, :3, 3:4] = t_inv
    return out


def align_w2c_to_first_camera(T_w2c: np.ndarray) -> np.ndarray:
    """
    Make the first camera the world frame.
    T'_w2c_i = T_w2c_i @ T_c2w_0, so the first camera becomes identity.
    """
    T_c2w_0 = invert_se3_np(T_w2c[0:1])[0]
    return np.matmul(T_w2c, T_c2w_0[None])


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion in x,y,z,w order.
    Accepts [3,3]. Returns [4].
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    # Keep a stable sign convention.
    if q[3] < 0:
        q = -q
    return q.astype(np.float32)


def write_tum_pose_file(path: Path, T_c2w: np.ndarray, timestamps: Optional[List[str]] = None) -> None:
    """Write timestamp tx ty tz qx qy qz qw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, T in enumerate(T_c2w):
            stamp = timestamps[i] if timestamps is not None else f"{i:.6f}"
            t = T[:3, 3]
            q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])
            f.write(
                f"{stamp} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def write_pose7_file(path: Path, T_c2w: np.ndarray) -> None:
    """Write tx ty tz qx qy qz qw, one frame per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for T in T_c2w:
            t = T[:3, 3]
            q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])
            f.write(
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def write_matrix_file(path: Path, T: np.ndarray) -> None:
    """Write flattened 3x4 or 4x4 matrices with frame index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, mat in enumerate(T):
            flat = mat.reshape(-1)
            f.write(str(i) + " " + " ".join(f"{v:.9f}" for v in flat) + "\n")


def tartan_from_cv_matrix(dtype=np.float32) -> np.ndarray:
    """
    Matrix used by the user's current ReSplat loader:
        Twc_cv = Twc_pose @ T_tartanCam_from_cvCam
    """
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


def compute_resplat_loader_pose_from_cv(T_c2w_cv: np.ndarray) -> np.ndarray:
    """
    Given desired OpenCV Twc, compute Twc_pose to write into pose txt so that
    the existing loader returns Twc_cv:
        Twc_cv = Twc_pose @ T_tartan_from_cv
        Twc_pose = Twc_cv @ inv(T_tartan_from_cv)
    """
    T_tartan_from_cv = tartan_from_cv_matrix(dtype=T_c2w_cv.dtype)
    T_cv_from_tartan = np.linalg.inv(T_tartan_from_cv)
    return np.matmul(T_c2w_cv, T_cv_from_tartan[None])


def save_intrinsics(path: Path, intrinsics: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, K in enumerate(intrinsics):
            f.write(str(i) + " " + " ".join(f"{v:.9f}" for v in K.reshape(-1)) + "\n")


def save_depth_frames(depth: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    for i in range(depth.shape[0]):
        np.save(out_dir / f"{i:06d}.npy", depth[i].astype(np.float32))


def sanitize_predictions_for_npz(predictions: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Keep only array-like values that can be safely stored in npz."""
    out: Dict[str, np.ndarray] = {}
    for k, v in predictions.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().float().numpy()
            if arr.ndim >= 1 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)
            out[k] = arr
        elif isinstance(v, np.ndarray):
            out[k] = v
        # Skip lists such as pose_enc_list/state_list; they are large and not needed for the next stage.
    return out


def load_zipmap_model(args, device: str):
    from zipmap.models.ZipMap import ZipMap

    config = dict(MODEL_CONFIG)
    config["other_config"] = dict(MODEL_CONFIG["other_config"])
    config["other_config"]["affine_invariant"] = args.affine_invariant

    model = ZipMap(**config)
    try:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")

    if args.ema and isinstance(checkpoint, dict) and "ema" in checkpoint:
        model_state_dict = checkpoint["ema"]
        print("[ZipMap] Using EMA weights from checkpoint.")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint

    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        print(f"[ZipMap] Missing keys: {missing}")
    if unexpected:
        print(f"[ZipMap] Unexpected keys: {unexpected}")
    model.eval().to(device)
    return model, config


def run_zipmap(args) -> None:
    repo_root = find_repo_root(args.repo_root)
    sys.path.insert(0, str(repo_root))

    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
    from zipmap.utils.geometry import (
        unproject_depth_map_to_point_map,
        closed_form_inverse_se3,
        homogenize_points,
    )

    stereo_pair_meta: Optional[List[Dict[str, object]]] = None
    frame_records: Optional[List[Dict[str, object]]] = None
    if args.stereo_left_dir or args.stereo_right_dir:
        if not (args.stereo_left_dir and args.stereo_right_dir):
            raise ValueError("Both --stereo_left_dir and --stereo_right_dir must be provided for stereo mode.")
        image_paths, stereo_pair_meta, frame_records = collect_stereo_image_paths(args)
        input_root_for_log = f"stereo: left={Path(args.stereo_left_dir).resolve()}, right={Path(args.stereo_right_dir).resolve()}"
    else:
        if args.image_dir is None:
            raise ValueError("Provide --image_dir for mono/multi-view mode, or --stereo_left_dir and --stereo_right_dir for stereo mode.")
        all_image_paths = collect_image_paths(args.image_dir, recursive=args.recursive)
        image_paths, original_indices = apply_sequence_sampling(
            all_image_paths,
            start_index=args.start_index,
            end_index=args.end_index,
            stride=args.stride,
            max_images=args.max_images,
        )
        frame_records = [
            {
                "pred_index": pred_index,
                "source_path": str(path),
                "source_name": path.name,
                "camera_id": "mono",
                "original_index": int(original_index),
                "gt_index": int(original_index),
            }
            for pred_index, (path, original_index) in enumerate(zip(image_paths, original_indices))
        ]
        input_root_for_log = str(Path(args.image_dir).resolve())

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device != "cuda":
        raise RuntimeError("ZipMap inference is expected to run on CUDA. Remove --cpu only if you intentionally patched ZipMap for CPU.")

    print(f"[Input] {len(image_paths)} images from {input_root_for_log}")
    print(f"[Output] {out_dir}")
    print(f"[Device] {device}")

    model, config = load_zipmap_model(args, device)

    image_names = [str(p) for p in image_paths]
    images = load_and_preprocess_images(
        image_names,
        target_size=args.target_size,
        mode=args.preprocess_mode,
    ).to(device)
    print(f"[Preprocess] images: {tuple(images.shape)}")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions_raw = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions_raw["pose_enc"], images.shape[-2:])
    predictions_raw["extrinsic"] = extrinsic
    predictions_raw["intrinsic"] = intrinsic

    pred = sanitize_predictions_for_npz(predictions_raw)

    # ZipMap gives OpenCV w2c/Tcw as [S,3,4].
    T_w2c_cv = to_homogeneous_4x4(pred["extrinsic"].astype(np.float32))
    if args.align_first_view:
        print("[Pose] Aligning output frame to the first camera.")
        T_w2c_cv = align_w2c_to_first_camera(T_w2c_cv)
        pred["extrinsic"] = T_w2c_cv[:, :3, :4].astype(np.float32)
    else:
        print("[Pose] Keeping ZipMap predicted world frame.")

    T_c2w_cv = invert_se3_np(T_w2c_cv).astype(np.float32)
    intrinsics = pred["intrinsic"].astype(np.float32)

    # Recompute point maps after optional first-view alignment.
    if "depth" in pred:
        print("[Point] Computing world_points_from_depth...")
        pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
            pred["depth"], pred["extrinsic"], intrinsics, if_c2w=False
        ).astype(np.float32)

    if "local_points" in pred:
        print("[Point] Transforming local_points to world_points...")
        cam_to_world = closed_form_inverse_se3(pred["extrinsic"]).astype(np.float32)
        local_points = pred["local_points"].astype(np.float32)
        world_points = np.einsum("sij,shwj->shwi", cam_to_world, homogenize_points(local_points))
        pred["world_points"] = world_points[..., :3].astype(np.float32)
        if "local_points_conf" in pred:
            pred["world_points_conf"] = pred["local_points_conf"].astype(np.float32)
        elif "depth_conf" in pred:
            pred["world_points_conf"] = pred["depth_conf"].astype(np.float32)

    # Store canonical matrices explicitly.
    pred["T_w2c_opencv"] = T_w2c_cv.astype(np.float32)
    pred["T_c2w_opencv"] = T_c2w_cv.astype(np.float32)
    pred["intrinsic"] = intrinsics

    # Save npz.
    np.savez_compressed(out_dir / "predictions.npz", **pred)

    # Save image lists and preprocessed images matching ZipMap intrinsics.
    source_names = [p.name for p in image_paths]
    if args.save_preprocessed_images:
        save_tensor_images_as_png(images, out_dir / "images_zipmap_input", source_names)
    if stereo_pair_meta is not None:
        with open(out_dir / "stereo_pairs.json", "w", encoding="utf-8") as f:
            json.dump(stereo_pair_meta, f, indent=2, ensure_ascii=False)
    if frame_records is not None:
        with open(out_dir / "frame_records.json", "w", encoding="utf-8") as f:
            json.dump(frame_records, f, indent=2, ensure_ascii=False)
        with open(out_dir / "selected_gt_indices.txt", "w", encoding="utf-8") as f:
            for r in frame_records:
                f.write(f"{r['pred_index']} {r['gt_index']} {r['camera_id']} {r['source_name']}\n")

    # Pose/matrix exports.
    timestamps = [p.stem for p in image_paths] if args.use_filename_as_timestamp else None
    write_tum_pose_file(out_dir / "poses_c2w_opencv_tum.txt", T_c2w_cv, timestamps=timestamps)
    write_matrix_file(out_dir / "poses_w2c_opencv.txt", T_w2c_cv[:, :3, :4])
    write_matrix_file(out_dir / "poses_c2w_opencv_matrix.txt", T_c2w_cv)

    T_resplat_loader_pose = compute_resplat_loader_pose_from_cv(T_c2w_cv).astype(np.float32)
    write_pose7_file(out_dir / "poses_resplat_tartanair_loader.txt", T_resplat_loader_pose)
    write_tum_pose_file(out_dir / "poses_resplat_tartanair_loader_tum.txt", T_resplat_loader_pose, timestamps=timestamps)

    save_intrinsics(out_dir / "intrinsics.txt", intrinsics)

    # Optional heavier exports.
    if args.save_depth_npy and "depth" in pred:
        save_depth_frames(pred["depth"], out_dir / "depth")
    if args.save_points_npy:
        points_dir = out_dir / "points"
        points_dir.mkdir(parents=True, exist_ok=True)
        if "world_points_from_depth" in pred:
            np.save(points_dir / "world_points_from_depth.npy", pred["world_points_from_depth"].astype(np.float32))
        if "world_points" in pred:
            np.save(points_dir / "world_points.npy", pred["world_points"].astype(np.float32))
        if "world_points_conf" in pred:
            np.save(points_dir / "world_points_conf.npy", pred["world_points_conf"].astype(np.float32))

    # Metadata.
    meta = {
        "input_root": input_root_for_log,
        "image_dir": None if args.image_dir is None else str(Path(args.image_dir).expanduser().resolve()),
        "stereo_left_dir": None if args.stereo_left_dir is None else str(Path(args.stereo_left_dir).expanduser().resolve()),
        "stereo_right_dir": None if args.stereo_right_dir is None else str(Path(args.stereo_right_dir).expanduser().resolve()),
        "stereo_pair_mode": args.stereo_pair_mode if stereo_pair_meta is not None else None,
        "stereo_num_pairs": len(stereo_pair_meta) if stereo_pair_meta is not None else None,
        "sampling": {
            "start_index": args.start_index,
            "end_index": args.end_index,
            "stride": args.stride,
            "max_images": args.max_images,
            "max_pairs": args.max_pairs,
        },
        "image_paths": [str(p) for p in image_paths],
        "image_names": source_names,
        "frame_records": frame_records,
        "selected_original_indices": [int(r["original_index"]) for r in frame_records] if frame_records is not None else None,
        "selected_gt_indices": [int(r["gt_index"]) for r in frame_records] if frame_records is not None else None,
        "num_images": len(image_paths),
        "preprocessed_shape_SCHW": list(images.shape),
        "target_size": args.target_size,
        "preprocess_mode": args.preprocess_mode,
        "align_first_view": args.align_first_view,
        "affine_invariant": args.affine_invariant,
        "ckpt_path": str(Path(args.ckpt_path).expanduser().resolve()),
        "zipmap_model_config": config,
        "pose_conventions": {
            "T_w2c_opencv": "OpenCV camera-from-world, x right, y down, z forward; ZipMap extrinsic convention.",
            "T_c2w_opencv": "Inverse of T_w2c_opencv.",
            "poses_c2w_opencv_tum.txt": "timestamp tx ty tz qx qy qz qw; OpenCV c2w.",
            "poses_resplat_tartanair_loader.txt": "tx ty tz qx qy qz qw; designed for the provided _build_Twc_from_pose() that multiplies by T_tartanCam_from_cvCam.",
        },
        "array_shapes": {k: list(v.shape) for k, v in pred.items() if isinstance(v, np.ndarray)},
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[Done] Exported ZipMap predictions.")
    print(f"  predictions: {out_dir / 'predictions.npz'}")
    print(f"  ReSplat-loader pose txt: {out_dir / 'poses_resplat_tartanair_loader.txt'}")
    print(f"  OpenCV c2w TUM pose txt: {out_dir / 'poses_c2w_opencv_tum.txt'}")
    print(f"  intrinsics: {out_dir / 'intrinsics.txt'}")
    if args.save_preprocessed_images:
        print(f"  preprocessed images: {out_dir / 'images_zipmap_input'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ZipMap predictions without Gradio UI.")
    parser.add_argument("--image_dir", default=None, type=str, help="Directory containing input images for mono/multi-view mode.")
    parser.add_argument("--stereo_left_dir", default=None, type=str, help="Left image directory for stereo mode.")
    parser.add_argument("--stereo_right_dir", default=None, type=str, help="Right image directory for stereo mode.")
    parser.add_argument(
        "--stereo_pair_mode",
        choices=["left_only", "right_only", "interleave", "concat_left_then_right"],
        default="interleave",
        help="How to flatten stereo pairs before feeding ZipMap. Default: interleave = L0,R0,L1,R1,...",
    )
    parser.add_argument("--ckpt_path", required=True, type=str, help="Path to ZipMap checkpoint, e.g. checkpoints/checkpoint_aff_inv.pt.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--repo_root", default=None, type=str, help="Path to ZipMap repository root. Optional if running from repo root.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search image directories for images.")
    parser.add_argument("--start_index", type=int, default=0, help="Start frame or stereo-pair index after sorting. Default: 0.")
    parser.add_argument("--end_index", type=int, default=None, help="Exclusive end frame or stereo-pair index after sorting. Default: all.")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame or stereo pair. Default: 1.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of mono/multi-view images after sampling.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Maximum number of stereo pairs after sampling.")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights if available in checkpoint.")
    parser.add_argument("--align_first_view", type=str2bool, default=True, help="Align output world frame to the first camera. Default: true.")
    parser.add_argument("--affine_invariant", type=str2bool, default=True, help="Use affine invariant mode. Default follows demo: true.")
    parser.add_argument("--target_size", type=int, default=518, help="ZipMap preprocessing target size. Default: 518.")
    parser.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop", help="Use the same load_fn mode. Default: crop.")
    parser.add_argument("--save_preprocessed_images", type=str2bool, default=True, help="Save ZipMap input images as PNG. Default: true.")
    parser.add_argument("--save_depth_npy", type=str2bool, default=True, help="Save per-frame depth .npy files. Default: true.")
    parser.add_argument("--save_points_npy", type=str2bool, default=True, help="Save dense point maps .npy files. Default: true.")
    parser.add_argument("--use_filename_as_timestamp", type=str2bool, default=True, help="Use image stem as TUM timestamp. Default: true.")
    parser.add_argument("--cpu", action="store_true", help="For debugging only; normal ZipMap inference expects CUDA.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_zipmap(args)


if __name__ == "__main__":
    main()
