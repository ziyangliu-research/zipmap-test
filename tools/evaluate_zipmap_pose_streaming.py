#!/usr/bin/env python3
"""
Windowed ZipMap pose evaluation for long sequences.

Runs ZipMap on overlapping windows to avoid OOM, evaluates only camera poses,
and aggregates Sim(3)-aligned pose errors against a GT pose file whose rows
correspond to the original sorted image order.

This is a windowed streaming evaluator, not a true recurrent state-cache
implementation. Each window is forwarded independently.
"""

# 每个窗口单独 GT Sim(3), 通过滑动窗口在zipmap计算位姿。

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
    # Match your ReSplat/TartanAir loader: Twc = Twc_pose @ T_tartanCam_from_cvCam
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


def compute_rpe_rows(T_pred: np.ndarray, T_gt: np.ndarray, original_indices: list[int], window_id: int, delta: int = 1) -> list[dict[str, Any]]:
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
                "window_id": window_id,
                "original_index_i": int(original_indices[i]),
                "original_index_j": int(original_indices[j]),
                "rpe_trans": float(np.linalg.norm(Terr[:3, 3])),
                "rpe_rot_deg": float(rot_angle_deg(Terr[:3, :3])),
            }
        )
    return rows


def build_model_config(camera_only: bool, affine_invariant: bool) -> dict[str, Any]:
    return {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "enable_camera": True,
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
        },
        "other_config": {
            "use_gradient_checkpointing_local_point": False,
            "use_gradient_checkpointing_depth": False,
            "affine_invariant": affine_invariant,
        },
    }


def load_zipmap_model(zipmap_repo: Path, ckpt_path: Path, camera_only: bool, affine_invariant: bool, use_ema: bool, device: torch.device):
    sys.path.insert(0, str(zipmap_repo))
    sys.path.insert(0, str(zipmap_repo / "zipmap"))
    from zipmap.models.ZipMap import ZipMap  # type: ignore

    model = ZipMap(**build_model_config(camera_only=camera_only, affine_invariant=affine_invariant))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if use_ema and isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Model] missing={len(missing)} unexpected={len(unexpected)} camera_only={camera_only}")
    if missing:
        print("[Model] missing sample:", missing[:10])
    if unexpected:
        print("[Model] unexpected sample:", unexpected[:10])
    model.eval().to(device)
    return model


@torch.no_grad()
def run_zipmap_window(model, image_paths: list[Path], device: torch.device, target_size: int = 518, preprocess_mode: str = "crop") -> tuple[np.ndarray, np.ndarray]:
    from zipmap.utils.load_fn import load_and_preprocess_images  # type: ignore
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

    images = load_and_preprocess_images([str(p) for p in image_paths], target_size=target_size, mode=preprocess_mode).to(device)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
        pred = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    Tcw = to_4x4(extrinsic.detach().cpu().float().numpy().squeeze(0))
    Twc = invert_se3_np(Tcw)
    K = intrinsic.detach().cpu().float().numpy().squeeze(0)

    del pred, extrinsic, intrinsic, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return Twc, K


def make_windows(n: int, window_size: int, window_overlap: int) -> list[tuple[int, int]]:
    if window_size <= 1:
        raise ValueError("--window_size must be > 1")
    if window_overlap < 0 or window_overlap >= window_size:
        raise ValueError("--window_overlap must satisfy 0 <= overlap < window_size")
    step = window_size - window_overlap
    windows = []
    s = 0
    while s < n:
        e = min(s + window_size, n)
        if e - s >= 2:
            windows.append((s, e))
        if e == n:
            break
        s += step
    return windows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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

    parser.add_argument("--window_size", type=int, default=80)
    parser.add_argument("--window_overlap", type=int, default=20)
    parser.add_argument("--rpe_delta", type=int, default=1)

    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--camera_only", type=str2bool, default=True)
    parser.add_argument("--affine_invariant", type=str2bool, default=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--eval_policy",
        choices=["new_only", "all"],
        default="new_only",
        help="new_only counts each selected frame once; all counts overlapped frames multiple times.",
    )
    parser.add_argument("--save_window_poses", type=str2bool, default=False)
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

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
    gt_selected = [gt_all[i] for i in selected_original_indices]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model = load_zipmap_model(
        zipmap_repo=args.zipmap_repo,
        ckpt_path=args.ckpt_path,
        camera_only=args.camera_only,
        affine_invariant=args.affine_invariant,
        use_ema=args.ema,
        device=device,
    )

    windows = make_windows(len(selected_paths), args.window_size, args.window_overlap)
    print(f"[Data] original images={len(image_paths_all)} selected={len(selected_paths)}")
    print(f"[Window] count={len(windows)} size={args.window_size} overlap={args.window_overlap} eval_policy={args.eval_policy}")

    per_frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    rpe_rows: list[dict[str, Any]] = []
    counted_selected_positions: set[int] = set()

    for wid, (s, e) in enumerate(windows):
        print(f"\n[Window {wid:03d}] selected [{s}, {e}) original [{selected_original_indices[s]}, {selected_original_indices[e-1]}]", flush=True)
        paths_w = selected_paths[s:e]
        orig_w = selected_original_indices[s:e]
        gt_w = np.stack(gt_selected[s:e], axis=0)

        Twc_pred_raw, _ = run_zipmap_window(
            model=model,
            image_paths=paths_w,
            device=device,
            target_size=args.target_size,
            preprocess_mode=args.preprocess_mode,
        )

        scale, R_align, t_align = umeyama_sim3(Twc_pred_raw[:, :3, 3], gt_w[:, :3, 3])
        Twc_pred_aligned = apply_sim3_to_c2w(Twc_pred_raw, scale, R_align, t_align)
        ate = se3_error_ate(Twc_pred_aligned, gt_w)

        if args.eval_policy == "all":
            eval_local_positions = list(range(e - s))
        else:
            eval_local_positions = []
            for local_pos, global_pos in enumerate(range(s, e)):
                if global_pos not in counted_selected_positions:
                    eval_local_positions.append(local_pos)
                    counted_selected_positions.add(global_pos)

        for lp in eval_local_positions:
            per_frame_rows.append(
                {
                    "window_id": wid,
                    "local_index": lp,
                    "selected_position": s + lp,
                    "original_index": orig_w[lp],
                    "image_name": paths_w[lp].name,
                    "ate": float(ate[lp]),
                    "sim3_scale": float(scale),
                }
            )

        if len(eval_local_positions) >= args.rpe_delta + 1:
            locs = eval_local_positions
            rpe_rows.extend(
                compute_rpe_rows(
                    Twc_pred_aligned[locs],
                    gt_w[locs],
                    [orig_w[i] for i in locs],
                    window_id=wid,
                    delta=args.rpe_delta,
                )
            )

        window_rows.append(
            {
                "window_id": wid,
                "selected_start": s,
                "selected_end_exclusive": e,
                "original_start": orig_w[0],
                "original_end": orig_w[-1],
                "num_window_frames": e - s,
                "num_eval_frames": len(eval_local_positions),
                "ate_rmse_window_all": float(np.sqrt(np.mean(ate ** 2))),
                "ate_mean_window_all": float(np.mean(ate)),
                "ate_median_window_all": float(np.median(ate)),
                "sim3_scale": float(scale),
            }
        )

        if args.save_window_poses:
            wdir = args.output / "window_poses"
            wdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                wdir / f"window_{wid:03d}.npz",
                original_indices=np.array(orig_w, dtype=np.int64),
                Twc_pred_raw=Twc_pred_raw,
                Twc_pred_aligned=Twc_pred_aligned,
                Twc_gt=gt_w,
                sim3_scale=np.array(scale),
                sim3_R=R_align,
                sim3_t=t_align,
            )

    write_csv(args.output / "per_frame_errors.csv", per_frame_rows)
    write_csv(args.output / "window_summaries.csv", window_rows)
    write_csv(args.output / "rpe_errors.csv", rpe_rows)

    ate_all = np.array([r["ate"] for r in per_frame_rows], dtype=np.float64)
    rpe_trans = np.array([r["rpe_trans"] for r in rpe_rows], dtype=np.float64) if rpe_rows else np.array([])
    rpe_rot = np.array([r["rpe_rot_deg"] for r in rpe_rows], dtype=np.float64) if rpe_rows else np.array([])

    summary = {
        "image_dir": str(args.image_dir),
        "gt_pose_file": str(args.gt_pose_file),
        "num_original_images": len(image_paths_all),
        "num_selected_frames": len(selected_paths),
        "num_windows": len(windows),
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "stride": args.stride,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_frames": args.max_frames,
        "eval_policy": args.eval_policy,
        "camera_only": args.camera_only,
        "affine_invariant": args.affine_invariant,
        "ate_rmse": float(np.sqrt(np.mean(ate_all ** 2))) if ate_all.size else None,
        "ate_mean": float(np.mean(ate_all)) if ate_all.size else None,
        "ate_median": float(np.median(ate_all)) if ate_all.size else None,
        "ate_p90": float(np.percentile(ate_all, 90)) if ate_all.size else None,
        "rpe_trans_mean": float(np.mean(rpe_trans)) if rpe_trans.size else None,
        "rpe_trans_median": float(np.median(rpe_trans)) if rpe_trans.size else None,
        "rpe_rot_deg_mean": float(np.mean(rpe_rot)) if rpe_rot.size else None,
        "rpe_rot_deg_median": float(np.median(rpe_rot)) if rpe_rot.size else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "selected_original_indices.txt").write_text("\n".join(map(str, selected_original_indices)) + "\n", encoding="utf-8")

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))
    print(f"[Done] Results written to: {args.output}")


if __name__ == "__main__":
    main()
