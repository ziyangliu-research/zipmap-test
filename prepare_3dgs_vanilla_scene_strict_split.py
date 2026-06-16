#!/usr/bin/env python3
"""
Prepare a vanilla original-3DGS compatible scene for fair baseline experiments.

This script writes only:
  <output_scene>/images/*
  <output_scene>/transforms_train.json
  <output_scene>/transforms_test.json
  <output_scene>/vanilla_scene_summary.json

It intentionally does NOT write ReSplat-derived points3d.ply or initial_fused_3dgs.ply.
If points3d.ply is absent, the original GraphDECO Blender/NeRF reader will use its
own scene initialization path (typically random point cloud generation for synthetic scenes).

Typical left-only command:
  python prepare_3dgs_vanilla_scene_strict_split.py \
    --frame_ranges 0-49 \
    --dataset_start_index 0 \
    --image_dir /path/to/image_lcam_front \
    --image_pattern "{index:06d}_lcam_front.png" \
    --pose_file /path/to/pose_lcam_front.txt \
    --pose_convention resplat_tartanair_pose \
    --fx 320 --fy 320 --cx 320 --cy 320 --width 640 --height 640 \
    --output_scene /path/to/3dgs_scene_vanilla_left_strict_split \
    --internal_split --split_every 5 --split_offset 4 \
    --copy_images

Typical stereo command:
  python prepare_3dgs_vanilla_scene_strict_split.py \
    --frame_ranges 0-49 \
    --dataset_start_index 0 \
    --image_dir /path/to/image_lcam_front \
    --image_pattern "{index:06d}_lcam_front.png" \
    --pose_file /path/to/pose_lcam_front.txt \
    --right_image_dir /path/to/image_rcam_front \
    --right_image_pattern "{index:06d}_rcam_front.png" \
    --right_pose_file /path/to/pose_rcam_front.txt \
    --stereo \
    --pose_convention resplat_tartanair_pose \
    --fx 320 --fy 320 --cx 320 --cy 320 --width 640 --height 640 \
    --output_scene /path/to/3dgs_scene_vanilla_stereo_strict_split \
    --internal_split --split_every 5 --split_offset 4 \
    --copy_images
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.spatial.transform import Rotation as SciRot
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy. Install it in the 3DGS env: pip install scipy") from exc


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


def read_pose_file(path: Path, pose_format: str, quat_order: str, matrix_convention: str) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    seq_idx = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if pose_format == 'tartanair_twc':
            if len(parts) < 7:
                continue
            idx = seq_idx
            seq_idx += 1
            vals = list(map(float, parts[:7]))
        elif pose_format == 'tum_twc':
            if len(parts) < 8:
                continue
            idx = seq_idx
            seq_idx += 1
            vals = list(map(float, parts[1:8]))
        else:
            raise ValueError(f"Unknown pose_format: {pose_format}")

        tx, ty, tz = vals[:3]
        q = vals[3:7]
        if quat_order == 'xyzw':
            q_xyzw = q
        elif quat_order == 'wxyz':
            qw, qx, qy, qz = q
            q_xyzw = [qx, qy, qz, qw]
        else:
            raise ValueError(f"Unknown quat_order: {quat_order}")

        R = SciRot.from_quat(q_xyzw).as_matrix().astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

        if matrix_convention == 'c2w':
            Twc = T
        elif matrix_convention == 'w2c':
            Twc = np.linalg.inv(T).astype(np.float32)
        else:
            raise ValueError(f"Unknown matrix_convention: {matrix_convention}")
        poses[idx] = Twc
    return poses


def tartanair_pose_twc_to_opencv_twc(Twc_tartan: np.ndarray) -> np.ndarray:
    """Convert TartanAir camera pose convention to OpenCV camera-to-world.

    This follows the convention used in the ZipMap/ReSplat TartanAir integration:
      x_cv right, y_cv down, z_cv forward
    with columns expressed in the TartanAir camera coordinate basis.
    """
    R_tartan_from_cv = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    T_tartan_from_cv = np.eye(4, dtype=np.float32)
    T_tartan_from_cv[:3, :3] = R_tartan_from_cv
    return (Twc_tartan.astype(np.float32) @ T_tartan_from_cv).astype(np.float32)


def pose_to_nerf_transform(Twc_raw: np.ndarray, pose_convention: str) -> np.ndarray:
    if pose_convention == 'resplat_tartanair_pose':
        Twc_cv = tartanair_pose_twc_to_opencv_twc(Twc_raw)
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        return (Twc_cv @ cv_to_gl).astype(np.float32)
    if pose_convention == 'opencv_c2w':
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        return (Twc_raw.astype(np.float32) @ cv_to_gl).astype(np.float32)
    if pose_convention in {'opengl_c2w', 'nerf_c2w'}:
        return Twc_raw.astype(np.float32)
    raise ValueError(f"Unknown pose_convention: {pose_convention}")


def is_test_frame(local_i: int, frame_index: int, args: argparse.Namespace) -> bool:
    if not args.internal_split:
        return False
    key = local_i if args.split_index_mode == 'local_index' else frame_index
    return (key % args.split_every) == args.split_offset


def copy_or_link(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copyfile(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def make_frame_entry(file_path_no_ext: str, T_nerf: np.ndarray, cam_name: str, frame_index: int, local_index: int) -> dict:
    return {
        'file_path': file_path_no_ext,
        'transform_matrix': T_nerf.astype(float).tolist(),
        'camera_name': cam_name,
        'frame_index': int(frame_index),
        'local_index': int(local_index),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare vanilla original-3DGS scene with strict train/test split.')
    p.add_argument('--frame_ranges', required=True, help='Frame/local indices to include, e.g. 0-49 or 0-3,5-8.')
    p.add_argument('--dataset_start_index', type=int, default=0)
    p.add_argument('--output_scene', required=True)

    p.add_argument('--image_dir', required=True, help='Left image directory.')
    p.add_argument('--image_pattern', default='{index:06d}.png')
    p.add_argument('--pose_file', required=True, help='Left pose file.')

    p.add_argument('--stereo', action='store_true', help='Also write right-camera views into train/test transforms.')
    p.add_argument('--right_image_dir', default=None)
    p.add_argument('--right_image_pattern', default='{index:06d}_rcam_front.png')
    p.add_argument('--right_pose_file', default=None)

    p.add_argument('--pose_format', choices=['tartanair_twc', 'tum_twc'], default='tartanair_twc')
    p.add_argument('--pose_convention', choices=['resplat_tartanair_pose', 'opencv_c2w', 'opengl_c2w', 'nerf_c2w'], default='resplat_tartanair_pose')
    p.add_argument('--gt_quat_order', choices=['xyzw', 'wxyz'], default='xyzw')
    p.add_argument('--gt_matrix_convention', choices=['c2w', 'w2c'], default='c2w')

    p.add_argument('--fx', type=float, required=True)
    p.add_argument('--fy', type=float, default=None)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--strict_image_size', action='store_true')
    p.add_argument('--copy_images', action='store_true')

    p.add_argument('--internal_split', action='store_true', help='Enable interleaved train/test split.')
    p.add_argument('--split_every', type=int, default=5)
    p.add_argument('--split_offset', type=int, default=4)
    p.add_argument('--split_index_mode', choices=['local_index', 'frame_index'], default='local_index')

    p.add_argument('--force_clean_geometry', action='store_true', help='Delete existing points3d.ply / initial_fused_3dgs.ply in output_scene if present.')
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.fy is None:
        args.fy = args.fx
    if args.cx is None:
        args.cx = args.width / 2.0
    if args.cy is None:
        args.cy = args.height / 2.0
    if args.split_every <= 0:
        raise ValueError('--split_every must be positive')
    if args.split_offset < 0 or args.split_offset >= args.split_every:
        raise ValueError('--split_offset must satisfy 0 <= split_offset < split_every')

    if args.stereo:
        if not args.right_image_dir or not args.right_pose_file:
            raise ValueError('--stereo requires --right_image_dir and --right_pose_file')

    output_scene = Path(args.output_scene)
    output_scene.mkdir(parents=True, exist_ok=True)

    dangerous = [output_scene / 'points3d.ply', output_scene / 'initial_fused_3dgs.ply']
    existing = [p for p in dangerous if p.exists()]
    if existing and not args.force_clean_geometry:
        raise FileExistsError(
            'Vanilla scene must not contain ReSplat-derived geometry files. Existing files:\n'
            + '\n'.join(str(p) for p in existing)
            + '\nUse a new --output_scene or pass --force_clean_geometry to delete them.'
        )
    for p in existing:
        p.unlink()
        print(f'[clean] removed {p}')

    frame_locals = parse_int_ranges(args.frame_ranges)
    if not frame_locals:
        raise ValueError('--frame_ranges produced no frames')
    frame_indices = [args.dataset_start_index + i for i in frame_locals]

    left_poses = read_pose_file(Path(args.pose_file), args.pose_format, args.gt_quat_order, args.gt_matrix_convention)
    right_poses = None
    if args.stereo:
        right_poses = read_pose_file(Path(args.right_pose_file), args.pose_format, args.gt_quat_order, args.gt_matrix_convention)

    image_out = output_scene / 'images'
    image_out.mkdir(parents=True, exist_ok=True)

    fovx = 2.0 * math.atan(args.width / (2.0 * args.fx))
    fovy = 2.0 * math.atan(args.height / (2.0 * args.fy))
    train_frames: List[dict] = []
    test_frames: List[dict] = []
    split_records = []

    for local_i, frame_index in zip(frame_locals, frame_indices):
        split = 'test' if is_test_frame(local_i, frame_index, args) else 'train'
        target_list = test_frames if split == 'test' else train_frames

        # Left camera.
        if frame_index not in left_poses:
            raise KeyError(f'Left pose index {frame_index} not found in {args.pose_file}')
        left_src = Path(args.image_dir) / args.image_pattern.format(index=frame_index, local_index=local_i)
        if not left_src.exists():
            raise FileNotFoundError(f'Missing left image: {left_src}')
        with Image.open(left_src) as im:
            w, h = im.size
        if args.strict_image_size and (w != args.width or h != args.height):
            raise ValueError(f'Left image size mismatch for {left_src}: got {w}x{h}, expected {args.width}x{args.height}')

        if args.stereo:
            left_name = f'{frame_index:06d}_l.png'
            left_file_path = f'images/{frame_index:06d}_l'
        else:
            left_name = f'{frame_index:06d}.png'
            left_file_path = f'images/{frame_index:06d}'
        copy_or_link(left_src, image_out / left_name, args.copy_images)
        target_list.append(make_frame_entry(left_file_path, pose_to_nerf_transform(left_poses[frame_index], args.pose_convention), 'left', frame_index, local_i))

        record = {
            'local_index': int(local_i),
            'frame_index': int(frame_index),
            'split': split,
            'left_image': str(left_src),
        }

        # Right camera.
        if args.stereo:
            assert right_poses is not None
            if frame_index not in right_poses:
                raise KeyError(f'Right pose index {frame_index} not found in {args.right_pose_file}')
            right_src = Path(args.right_image_dir) / args.right_image_pattern.format(index=frame_index, local_index=local_i)
            if not right_src.exists():
                raise FileNotFoundError(f'Missing right image: {right_src}')
            with Image.open(right_src) as im:
                rw, rh = im.size
            if args.strict_image_size and (rw != args.width or rh != args.height):
                raise ValueError(f'Right image size mismatch for {right_src}: got {rw}x{rh}, expected {args.width}x{args.height}')
            right_name = f'{frame_index:06d}_r.png'
            right_file_path = f'images/{frame_index:06d}_r'
            copy_or_link(right_src, image_out / right_name, args.copy_images)
            target_list.append(make_frame_entry(right_file_path, pose_to_nerf_transform(right_poses[frame_index], args.pose_convention), 'right', frame_index, local_i))
            record['right_image'] = str(right_src)

        split_records.append(record)

    meta_base = {
        'camera_angle_x': fovx,
        'camera_angle_y': fovy,
        'fl_x': args.fx,
        'fl_y': args.fy,
        'cx': args.cx,
        'cy': args.cy,
        'w': args.width,
        'h': args.height,
    }
    (output_scene / 'transforms_train.json').write_text(json.dumps({**meta_base, 'frames': train_frames}, indent=2))
    (output_scene / 'transforms_test.json').write_text(json.dumps({**meta_base, 'frames': test_frames}, indent=2))

    train_ts = sorted({r['frame_index'] for r in split_records if r['split'] == 'train'})
    test_ts = sorted({r['frame_index'] for r in split_records if r['split'] == 'test'})
    summary = {
        'output_scene': str(output_scene),
        'mode': 'stereo' if args.stereo else 'left_only',
        'frame_ranges': args.frame_ranges,
        'dataset_start_index': args.dataset_start_index,
        'num_timestamps_total': len(frame_indices),
        'num_train_timestamps': len(train_ts),
        'num_test_timestamps': len(test_ts),
        'num_train_camera_views': len(train_frames),
        'num_test_camera_views': len(test_frames),
        'train_frame_indices': train_ts,
        'test_frame_indices': test_ts,
        'split': {
            'internal_split': args.internal_split,
            'split_every': args.split_every,
            'split_offset': args.split_offset,
            'split_index_mode': args.split_index_mode,
        },
        'camera': {
            'fx': args.fx,
            'fy': args.fy,
            'cx': args.cx,
            'cy': args.cy,
            'width': args.width,
            'height': args.height,
            'camera_angle_x': fovx,
            'camera_angle_y': fovy,
        },
        'pose': {
            'pose_file': args.pose_file,
            'right_pose_file': args.right_pose_file if args.stereo else None,
            'pose_format': args.pose_format,
            'pose_convention': args.pose_convention,
            'gt_quat_order': args.gt_quat_order,
            'gt_matrix_convention': args.gt_matrix_convention,
        },
        'geometry_files_written': [],
        'note': 'No points3d.ply or initial_fused_3dgs.ply is written by this script. Do not reuse a ReSplat-prepared scene for vanilla baseline.',
        'records': split_records,
    }
    (output_scene / 'vanilla_scene_summary.json').write_text(json.dumps(summary, indent=2))

    print(f'[write] {output_scene / "transforms_train.json"}: {len(train_frames)} camera views')
    print(f'[write] {output_scene / "transforms_test.json"}:  {len(test_frames)} camera views')
    print(f'[split] train timestamps={len(train_ts)}, test timestamps={len(test_ts)}')
    print(f'[done] vanilla 3DGS scene prepared at {output_scene}')
    print('[note] No ReSplat-derived geometry file was written.')


if __name__ == '__main__':
    main()
