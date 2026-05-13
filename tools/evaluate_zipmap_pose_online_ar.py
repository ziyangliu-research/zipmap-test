#!/usr/bin/env python3
"""
Offline pose evaluation for ZipMap official online/AR model.

This script reproduces the author's streaming-demo model path without Gradio:
  - zipmap.models.ZipMap_AR.ZipMap
  - checkpoint_online.pt
  - enable_camera=False, enable_camera_mlp=True
  - ttt_config.window_size=1 by default

Two inference modes are provided:
  1) --inference_mode full
     Exact demo-style call: load selected images and call model(images) once.
     This may OOM for long sequences.

  2) --inference_mode stateful
     Process the sequence in chunks and pass Aggregator TTT state_list from one
     chunk to the next. This is intended for long sequences and lower memory.
     It avoids external overlap Sim(3) stitching. GT is used only for final
     trajectory evaluation.

Example:
  python tools/evaluate_zipmap_pose_online_ar.py \
    --zipmap_repo /home/shiyo/Desktop/ZipMap \
    --ckpt_path /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_online.pt \
    --image_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
    --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
    --gt_convention resplat_tartanair_pose \
    --output /home/shiyo/Desktop/ZipMap/outputs/pose_eval_online_ar/P000_stateful_w1_c20 \
    --inference_mode stateful \
    --stream_chunk_size 20 \
    --online_window_size 1 \
    --camera_only true \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot


def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return x.lower() in {"1", "true", "yes", "y", "on"}


def sort_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def collect_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(image_dir.glob(pat))
    paths = sorted(paths, key=sort_key)
    if not paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    return paths


def quat_to_rotmat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return SciRot.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float64)


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    return SciRot.from_matrix(R).as_quat().astype(np.float64)


def T_tartan_cam_from_cv_cam_np() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return T


def build_Twc_from_tartanair_pose(row: list[float]) -> np.ndarray:
    tx, ty, tz, qx, qy, qz, qw = row[:7]
    Twc_pose = np.eye(4, dtype=np.float64)
    Twc_pose[:3, :3] = quat_to_rotmat_xyzw(qx, qy, qz, qw)
    Twc_pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    # Convert camera axes from Tartan/AirSim camera frame to OpenCV camera frame.
    return Twc_pose @ T_tartan_cam_from_cv_cam_np()


def load_gt_poses(path: Path, convention: str) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.replace(",", " ").split()]
            if len(vals) < 7:
                continue
            if convention == "resplat_tartanair_pose":
                Twc = build_Twc_from_tartanair_pose(vals)
            elif convention == "opencv_c2w":
                tx, ty, tz, qx, qy, qz, qw = vals[:7]
                Twc = np.eye(4, dtype=np.float64)
                Twc[:3, :3] = quat_to_rotmat_xyzw(qx, qy, qz, qw)
                Twc[:3, 3] = [tx, ty, tz]
            else:
                raise ValueError(f"Unsupported gt_convention: {convention}")
            poses.append(Twc)
    if not poses:
        raise RuntimeError(f"No valid GT poses loaded from: {path}")
    return poses


def to_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    if T.shape[-2:] == (4, 4):
        return T.astype(np.float64)
    if T.shape[-2:] != (3, 4):
        raise ValueError(f"Expected [...,3,4] or [...,4,4], got {T.shape}")
    out = np.tile(np.eye(4, dtype=np.float64), T.shape[:-2] + (1, 1))
    out[..., :3, :4] = T
    return out


def invert_se3_np(T: np.ndarray) -> np.ndarray:
    T = to_4x4(T)
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    Rt = np.swapaxes(R, -1, -2)
    out = np.tile(np.eye(4, dtype=np.float64), T.shape[:-2] + (1, 1))
    out[..., :3, :3] = Rt
    out[..., :3, 3:4] = -Rt @ t
    return out


def umeyama_sim3(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Find s, R, t such that gt ~= s * R @ pred + t."""
    X = np.asarray(pred_xyz, dtype=np.float64)
    Y = np.asarray(gt_xyz, dtype=np.float64)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Invalid point shapes: pred={X.shape}, gt={Y.shape}")

    n = X.shape[0]
    mux = X.mean(axis=0)
    muy = Y.mean(axis=0)
    Xc = X - mux
    Yc = Y - muy
    var_x = np.mean(np.sum(Xc * Xc, axis=1))
    if n < 3 or var_x < 1e-12:
        return 1.0, np.eye(3, dtype=np.float64), muy - mux

    cov = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_x)
    t = muy - s * (R @ mux)
    return s, R, t


def apply_sim3_to_c2w(Twc_pred: np.ndarray, s: float, R_align: np.ndarray, t_align: np.ndarray) -> np.ndarray:
    out = np.array(Twc_pred, dtype=np.float64, copy=True)
    out[:, :3, :3] = R_align[None] @ out[:, :3, :3]
    out[:, :3, 3] = (s * (R_align[None] @ out[:, :3, 3:4]).squeeze(-1)) + t_align[None]
    return out


def rot_angle_deg(R: np.ndarray) -> float:
    val = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(val))


def se3_error_ate(T_pred: np.ndarray, T_gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(T_pred[:, :3, 3] - T_gt[:, :3, 3], axis=1)


def compute_rpe_rows(T_pred: np.ndarray, T_gt: np.ndarray, original_indices: list[int], delta: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(T_pred) <= delta:
        return rows
    for i in range(0, len(T_pred) - delta):
        j = i + delta
        Trel_p = invert_se3_np(T_pred[i]) @ T_pred[j]
        Trel_g = invert_se3_np(T_gt[i]) @ T_gt[j]
        Terr = invert_se3_np(Trel_g) @ Trel_p
        rows.append(
            {
                "original_index_i": int(original_indices[i]),
                "original_index_j": int(original_indices[j]),
                "rpe_trans": float(np.linalg.norm(Terr[:3, 3])),
                "rpe_rot_deg": float(rot_angle_deg(Terr[:3, :3])),
            }
        )
    return rows


def build_online_ar_model_config(camera_only: bool, affine_invariant: bool, online_window_size: int | str) -> dict[str, Any]:
    return {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "enable_camera": False,
        "enable_camera_mlp": True,
        "enable_local_point": not camera_only,
        "enable_depth": not camera_only,
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
            "window_size": online_window_size,
        },
        "other_config": {
            "use_gradient_checkpointing_local_point": False,
            "use_gradient_checkpointing_depth": False,
            "affine_invariant": affine_invariant,
        },
    }


def load_zipmap_ar_model(
    zipmap_repo: Path,
    ckpt_path: Path,
    camera_only: bool,
    affine_invariant: bool,
    online_window_size: int | str,
    use_ema: bool,
    device: torch.device,
):
    sys.path.insert(0, str(zipmap_repo))
    sys.path.insert(0, str(zipmap_repo / "zipmap"))
    from zipmap.models.ZipMap_AR import ZipMap  # type: ignore

    cfg = build_online_ar_model_config(
        camera_only=camera_only,
        affine_invariant=affine_invariant,
        online_window_size=online_window_size,
    )
    model = ZipMap(**cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if use_ema and isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Model] ZipMap_AR loaded from {ckpt_path}")
    print(f"[Model] camera_only={camera_only} affine_invariant={affine_invariant} online_window_size={online_window_size}")
    print(f"[Model] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[Model] missing sample:", missing[:20])
    if unexpected:
        print("[Model] unexpected sample:", unexpected[:20])
    model.eval().to(device)
    return model


def detach_state_list(state_list: list[Any]) -> list[Any]:
    """Move cached TTT state to detached tensors to prevent memory growth."""
    out: list[Any] = []
    for state in state_list:
        if isinstance(state, dict):
            out.append({k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in state.items()})
        elif isinstance(state, torch.Tensor):
            out.append(state.detach())
        else:
            out.append(state)
    return out


@torch.no_grad()
def forward_ar_pose_with_state(
    model,
    images: torch.Tensor,
    previous_state_list: list[Any] | None,
    online_window_size: int | None,
    store_state: bool = True,
):
    """Run ZipMap_AR pose head while optionally seeding aggregator with previous TTT state."""
    if len(images.shape) == 4:
        images = images.unsqueeze(0)

    info: dict[str, Any] = {"store_state": store_state}
    if online_window_size is not None:
        info["window_size"] = online_window_size
    if previous_state_list is not None:
        info["state_list"] = previous_state_list

    aggregated_tokens_list, patch_start_idx, state_list = model.aggregator(
        images,
        target_query_conditions=None,
        info=info,
    )
    input_view_num = images.shape[1]
    input_img_aggregated_tokens_list = [tokens[:, :input_view_num, :] for tokens in aggregated_tokens_list]

    predictions: dict[str, Any] = {}
    with torch.amp.autocast(device_type="cuda", enabled=False):
        if getattr(model, "camera_mlp_head", None) is not None:
            camera_tokens = input_img_aggregated_tokens_list[-1][:, :, 0]
            pose_enc_mlp_list = [model.camera_mlp_head(camera_tokens)]
            predictions["pose_enc"] = pose_enc_mlp_list[-1]
            predictions["pose_enc_mlp_list"] = pose_enc_mlp_list
        elif getattr(model, "camera_head", None) is not None:
            pose_enc_list = model.camera_head(input_img_aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list
        else:
            raise RuntimeError("Neither camera_mlp_head nor camera_head is enabled.")

    predictions["state_list"] = state_list
    return predictions


@torch.no_grad()
def run_online_ar_full(
    model,
    image_paths: list[Path],
    device: torch.device,
    target_size: int,
    preprocess_mode: str,
    online_window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from zipmap.utils.load_fn import load_and_preprocess_images  # type: ignore
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

    images = load_and_preprocess_images([str(p) for p in image_paths], target_size=target_size, mode=preprocess_mode).to(device)
    print(f"[Full] images={tuple(images.shape)}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
        pred = model(images, window_size=online_window_size)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    Tcw = to_4x4(extrinsic.detach().cpu().float().numpy().squeeze(0))
    Twc = invert_se3_np(Tcw)
    K = intrinsic.detach().cpu().float().numpy().squeeze(0)

    del pred, extrinsic, intrinsic, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return Twc, K


@torch.no_grad()
def run_online_ar_stateful(
    model,
    image_paths: list[Path],
    device: torch.device,
    target_size: int,
    preprocess_mode: str,
    online_window_size: int,
    stream_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from zipmap.utils.load_fn import load_and_preprocess_images  # type: ignore
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

    if stream_chunk_size <= 0:
        raise ValueError("--stream_chunk_size must be > 0")

    all_Twc: list[np.ndarray] = []
    all_K: list[np.ndarray] = []
    state_list: list[Any] | None = None
    n = len(image_paths)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    for start in range(0, n, stream_chunk_size):
        end = min(start + stream_chunk_size, n)
        paths = image_paths[start:end]
        print(f"[Stateful] chunk [{start}, {end}) images={len(paths)} prev_state={'yes' if state_list is not None else 'no'}", flush=True)
        images = load_and_preprocess_images([str(p) for p in paths], target_size=target_size, mode=preprocess_mode).to(device)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            pred = forward_ar_pose_with_state(
                model=model,
                images=images,
                previous_state_list=state_list,
                online_window_size=online_window_size,
                store_state=True,
            )
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        Tcw = to_4x4(extrinsic.detach().cpu().float().numpy().squeeze(0))
        Twc = invert_se3_np(Tcw)
        K = intrinsic.detach().cpu().float().numpy().squeeze(0)
        all_Twc.append(Twc)
        all_K.append(K)
        state_list = detach_state_list(pred["state_list"])

        del pred, extrinsic, intrinsic, images
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return np.concatenate(all_Twc, axis=0), np.concatenate(all_K, axis=0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tum_like(path: Path, original_indices: list[int], Twc: np.ndarray) -> None:
    lines = []
    for idx, T in zip(original_indices, Twc):
        qx, qy, qz, qw = rotmat_to_quat_xyzw(T[:3, :3])
        tx, ty, tz = T[:3, 3]
        lines.append(f"{idx} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipmap_repo", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--gt_pose_file", type=Path, required=True)
    parser.add_argument("--gt_convention", choices=["resplat_tartanair_pose", "opencv_c2w"], default="resplat_tartanair_pose")
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--rpe_delta", type=int, default=1)

    parser.add_argument("--inference_mode", choices=["full", "stateful"], default="stateful")
    parser.add_argument("--stream_chunk_size", type=int, default=20)
    parser.add_argument("--online_window_size", type=int, default=1)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--camera_only", type=str2bool, default=True)
    parser.add_argument("--affine_invariant", type=str2bool, default=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.online_window_size <= 0:
        raise ValueError("--online_window_size must be > 0")

    args.output.mkdir(parents=True, exist_ok=True)

    image_paths_all = collect_images(args.image_dir)
    end = len(image_paths_all) if args.end_index < 0 else min(args.end_index, len(image_paths_all))
    selected_original_indices = list(range(args.start_index, end, args.stride))
    if args.max_frames > 0:
        selected_original_indices = selected_original_indices[: args.max_frames]
    if len(selected_original_indices) < 2:
        raise RuntimeError(f"Need at least 2 selected frames, got {len(selected_original_indices)}")

    selected_paths = [image_paths_all[i] for i in selected_original_indices]
    gt_all = load_gt_poses(args.gt_pose_file, args.gt_convention)
    max_idx = max(selected_original_indices)
    if max_idx >= len(gt_all):
        raise IndexError(f"Selected original index {max_idx} exceeds GT pose count {len(gt_all)}")
    gt_selected = np.stack([gt_all[i] for i in selected_original_indices], axis=0)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model = load_zipmap_ar_model(
        zipmap_repo=args.zipmap_repo,
        ckpt_path=args.ckpt_path,
        camera_only=args.camera_only,
        affine_invariant=args.affine_invariant,
        online_window_size=args.online_window_size,
        use_ema=args.ema,
        device=device,
    )

    print(f"[Data] original images={len(image_paths_all)} selected={len(selected_paths)}")
    print(f"[Eval] inference_mode={args.inference_mode}; no external overlap Sim(3) chaining is used.")

    if args.inference_mode == "full":
        Twc_pred, K_pred = run_online_ar_full(
            model=model,
            image_paths=selected_paths,
            device=device,
            target_size=args.target_size,
            preprocess_mode=args.preprocess_mode,
            online_window_size=args.online_window_size,
        )
    else:
        Twc_pred, K_pred = run_online_ar_stateful(
            model=model,
            image_paths=selected_paths,
            device=device,
            target_size=args.target_size,
            preprocess_mode=args.preprocess_mode,
            online_window_size=args.online_window_size,
            stream_chunk_size=args.stream_chunk_size,
        )

    if Twc_pred.shape[0] != len(selected_paths):
        raise RuntimeError(f"Predicted pose count {Twc_pred.shape[0]} != selected frame count {len(selected_paths)}")

    # One and only one GT-based alignment for the complete predicted trajectory.
    s_global, R_global, t_global = umeyama_sim3(Twc_pred[:, :3, 3], gt_selected[:, :3, 3])
    Twc_aligned = apply_sim3_to_c2w(Twc_pred, s_global, R_global, t_global)
    ate = se3_error_ate(Twc_aligned, gt_selected)

    per_frame_rows = []
    for pos, orig_idx, img_path, err in zip(range(len(selected_paths)), selected_original_indices, selected_paths, ate):
        per_frame_rows.append(
            {
                "selected_position": int(pos),
                "original_index": int(orig_idx),
                "image_name": img_path.name,
                "ate": float(err),
            }
        )

    rpe_rows = compute_rpe_rows(Twc_aligned, gt_selected, selected_original_indices, delta=args.rpe_delta)
    rpe_trans = np.array([r["rpe_trans"] for r in rpe_rows], dtype=np.float64) if rpe_rows else np.array([])
    rpe_rot = np.array([r["rpe_rot_deg"] for r in rpe_rows], dtype=np.float64) if rpe_rows else np.array([])

    write_csv(args.output / "per_frame_errors.csv", per_frame_rows)
    write_csv(args.output / "rpe_errors.csv", rpe_rows)
    (args.output / "selected_original_indices.txt").write_text("\n".join(map(str, selected_original_indices)) + "\n", encoding="utf-8")

    np.savez_compressed(
        args.output / "online_ar_trajectory.npz",
        original_indices=np.array(selected_original_indices, dtype=np.int64),
        Twc_pred_raw=Twc_pred,
        Twc_pred_aligned=Twc_aligned,
        Twc_gt=gt_selected,
        K_pred=K_pred,
        global_sim3_scale=np.array(s_global),
        global_sim3_R=R_global,
        global_sim3_t=t_global,
    )
    write_tum_like(args.output / "trajectory_pred_raw.txt", selected_original_indices, Twc_pred)
    write_tum_like(args.output / "trajectory_pred_aligned.txt", selected_original_indices, Twc_aligned)
    write_tum_like(args.output / "trajectory_gt.txt", selected_original_indices, gt_selected)

    summary = {
        "image_dir": str(args.image_dir),
        "gt_pose_file": str(args.gt_pose_file),
        "num_original_images": len(image_paths_all),
        "num_selected_frames": len(selected_paths),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "model": "ZipMap_AR",
        "ckpt_path": str(args.ckpt_path),
        "camera_head": "CameraHead_MLP",
        "camera_only": args.camera_only,
        "affine_invariant": args.affine_invariant,
        "inference_mode": args.inference_mode,
        "online_window_size": args.online_window_size,
        "stream_chunk_size": args.stream_chunk_size if args.inference_mode == "stateful" else None,
        "external_chaining": "none",
        "final_gt_alignment": "single_global_sim3",
        "global_sim3_scale": float(s_global),
        "ate_rmse": float(np.sqrt(np.mean(ate ** 2))) if ate.size else None,
        "ate_mean": float(np.mean(ate)) if ate.size else None,
        "ate_median": float(np.median(ate)) if ate.size else None,
        "ate_p90": float(np.percentile(ate, 90)) if ate.size else None,
        "ate_max": float(np.max(ate)) if ate.size else None,
        "rpe_trans_mean": float(np.mean(rpe_trans)) if rpe_trans.size else None,
        "rpe_trans_median": float(np.median(rpe_trans)) if rpe_trans.size else None,
        "rpe_rot_deg_mean": float(np.mean(rpe_rot)) if rpe_rot.size else None,
        "rpe_rot_deg_median": float(np.median(rpe_rot)) if rpe_rot.size else None,
    }

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))
    print(f"[Done] Results written to: {args.output}")


if __name__ == "__main__":
    main()
