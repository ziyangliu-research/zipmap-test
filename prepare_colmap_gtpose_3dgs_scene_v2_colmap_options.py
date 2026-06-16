#!/usr/bin/env python3
"""
Prepare a strict-split 3DGS scene initialized by COLMAP triangulated sparse points
using known/GT camera poses.

Purpose
-------
This is the third baseline for fair comparison:
  - GT pose + Original 3DGS-Random          (already prepared by vanilla script)
  - GT pose + ReSplat-init 3DGS             (already prepared by ReSplat init script)
  - GT pose + COLMAP/SfM sparse points + Original 3DGS  (this script)

Important strict-held-out rule
------------------------------
Test frames are written to transforms_test.json for evaluation, but they are NOT
copied into the COLMAP workspace and do NOT participate in feature extraction,
matching, or triangulation. Only train frames are used to create points3d.ply.

What this script creates
------------------------
<output_scene>/
  images/                       # train + test images for 3DGS render/eval
  transforms_train.json          # GT poses for train views
  transforms_test.json           # GT poses for held-out test views
  points3d.ply                   # produced from COLMAP triangulated train-only sparse points
  colmap_gtpose_summary.json
  colmap_workspace/
    images/                      # train images only
    database.db
    sparse_gt/                   # known-pose empty sparse model for point_triangulator
      cameras.txt
      images.txt
      points3D.txt               # empty
    sparse_triangulated/         # COLMAP point_triangulator output
    sparse_triangulated_txt/     # TXT conversion for parsing points3D.txt

Typical command for TartanAir P000 0-49 left-only:
  python prepare_colmap_gtpose_3dgs_scene_v2_colmap_options.py \
    --frame_ranges 0-49 \
    --dataset_start_index 0 \
    --image_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
    --image_pattern "{index:06d}_lcam_front.png" \
    --pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/pose_lcam_front.txt \
    --pose_convention resplat_tartanair_pose \
    --fx 320 --fy 320 --cx 320 --cy 320 --width 640 --height 640 \
    --output_scene /home/shiyo/Desktop/ZipMap/outputs/gt_resplat_P000_0_50/3dgs_scene_colmap_gtpose_left_strict_split \
    --internal_split --split_every 5 --split_offset 4 \
    --copy_images \
    --run_colmap \
    --colmap_bin colmap \
    --sift_use_gpu 1 \
    --matcher exhaustive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.spatial.transform import Rotation as SciRot
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy. Install it in the 3DGS env: pip install scipy") from exc


# -----------------------------------------------------------------------------
# Range / pose utilities
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


def read_pose_file(path: Path, pose_format: str, quat_order: str, matrix_convention: str) -> Dict[int, np.ndarray]:
    """Read poses and return camera-to-world matrices Twc indexed by line index.

    tartanair_twc: each non-comment line is tx ty tz qx qy qz qw by default.
    tum_twc: timestamp tx ty tz qx qy qz qw.
    """
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

        R = SciRot.from_quat(q_xyzw).as_matrix().astype(np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

        if matrix_convention == 'c2w':
            Twc = T
        elif matrix_convention == 'w2c':
            Twc = np.linalg.inv(T)
        else:
            raise ValueError(f"Unknown matrix_convention: {matrix_convention}")
        poses[idx] = Twc
    return poses


def tartanair_pose_twc_to_opencv_twc(Twc_tartan: np.ndarray) -> np.ndarray:
    """Convert TartanAir pose convention to OpenCV camera-to-world.

    Same convention as the previous ReSplat/vanilla scene preparation scripts.
    Output camera coordinates: x right, y down, z forward.
    """
    R_tartan_from_cv = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    T_tartan_from_cv = np.eye(4, dtype=np.float64)
    T_tartan_from_cv[:3, :3] = R_tartan_from_cv
    return Twc_tartan.astype(np.float64) @ T_tartan_from_cv


def raw_twc_to_opencv_twc(Twc_raw: np.ndarray, pose_convention: str) -> np.ndarray:
    if pose_convention == 'resplat_tartanair_pose':
        return tartanair_pose_twc_to_opencv_twc(Twc_raw)
    if pose_convention == 'opencv_c2w':
        return Twc_raw.astype(np.float64)
    if pose_convention in {'opengl_c2w', 'nerf_c2w'}:
        # OpenGL camera: x right, y up, z backward -> OpenCV: x right, y down, z forward.
        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        return Twc_raw.astype(np.float64) @ gl_to_cv
    raise ValueError(f"Unknown pose_convention: {pose_convention}")


def raw_twc_to_nerf_twc(Twc_raw: np.ndarray, pose_convention: str) -> np.ndarray:
    """Return OpenGL/NeRF-style camera-to-world transform for transforms_*.json."""
    if pose_convention == 'resplat_tartanair_pose':
        Twc_cv = tartanair_pose_twc_to_opencv_twc(Twc_raw)
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        return Twc_cv @ cv_to_gl
    if pose_convention == 'opencv_c2w':
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        return Twc_raw.astype(np.float64) @ cv_to_gl
    if pose_convention in {'opengl_c2w', 'nerf_c2w'}:
        return Twc_raw.astype(np.float64)
    raise ValueError(f"Unknown pose_convention: {pose_convention}")


def opencv_twc_to_colmap_qvec_tvec(Twc_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """COLMAP images.txt expects world-to-camera Rcw/tcw with qvec in wxyz."""
    Tcw = np.linalg.inv(Twc_cv)
    Rcw = Tcw[:3, :3]
    tcw = Tcw[:3, 3]
    q_xyzw = SciRot.from_matrix(Rcw).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    q_wxyz = q_wxyz / max(np.linalg.norm(q_wxyz), 1e-12)
    return q_wxyz, tcw.astype(np.float64)


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


# -----------------------------------------------------------------------------
# COLMAP DB/model utilities
# -----------------------------------------------------------------------------

CAMERA_MODEL_ID_TO_NAME = {
    0: 'SIMPLE_PINHOLE',
    1: 'PINHOLE',
    2: 'SIMPLE_RADIAL',
    3: 'RADIAL',
    4: 'OPENCV',
    5: 'OPENCV_FISHEYE',
    6: 'FULL_OPENCV',
    7: 'FOV',
    8: 'SIMPLE_RADIAL_FISHEYE',
    9: 'RADIAL_FISHEYE',
    10: 'THIN_PRISM_FISHEYE',
}


def read_colmap_database_images_and_cameras(database_path: Path) -> Tuple[Dict[str, dict], Dict[int, dict]]:
    if not database_path.exists():
        raise FileNotFoundError(f"COLMAP database does not exist: {database_path}")
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        images = {}
        for image_id, name, camera_id in cur.execute('SELECT image_id, name, camera_id FROM images'):
            images[str(name)] = {
                'image_id': int(image_id),
                'name': str(name),
                'camera_id': int(camera_id),
            }
        cameras = {}
        for camera_id, model, width, height, params, prior_focal_length in cur.execute(
            'SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras'
        ):
            model_id = int(model)
            params_arr = np.frombuffer(params, dtype=np.float64).copy()
            cameras[int(camera_id)] = {
                'camera_id': int(camera_id),
                'model_id': model_id,
                'model_name': CAMERA_MODEL_ID_TO_NAME.get(model_id, str(model_id)),
                'width': int(width),
                'height': int(height),
                'params': params_arr.tolist(),
                'prior_focal_length': int(prior_focal_length),
            }
    finally:
        conn.close()
    if not images:
        raise RuntimeError(f"No images found in COLMAP database: {database_path}")
    if not cameras:
        raise RuntimeError(f"No cameras found in COLMAP database: {database_path}")
    return images, cameras


def write_known_pose_sparse_model(
    sparse_gt_dir: Path,
    db_images: Dict[str, dict],
    db_cameras: Dict[int, dict],
    train_records: List[dict],
) -> None:
    sparse_gt_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt
    cam_lines = [
        '# Camera list with one line of data per camera:',
        '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
        '# Number of cameras: {}'.format(len(db_cameras)),
    ]
    for camera_id in sorted(db_cameras):
        cam = db_cameras[camera_id]
        params = ' '.join(f'{float(x):.17g}' for x in cam['params'])
        cam_lines.append(f"{camera_id} {cam['model_name']} {cam['width']} {cam['height']} {params}")
    (sparse_gt_dir / 'cameras.txt').write_text('\n'.join(cam_lines) + '\n')

    # images.txt: two lines per registered image. Second line is empty 2D points.
    img_lines = [
        '# Image list with two lines of data per image:',
        '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME',
        '#   POINTS2D[] as (X, Y, POINT3D_ID)',
        '# Number of images: {}'.format(len(train_records)),
    ]
    missing = []
    for rec in train_records:
        image_name = rec['colmap_image_name']
        if image_name not in db_images:
            missing.append(image_name)
            continue
        db_img = db_images[image_name]
        image_id = int(db_img['image_id'])
        camera_id = int(db_img['camera_id'])
        q = rec['colmap_qvec_wxyz']
        t = rec['colmap_tvec']
        img_lines.append(
            f"{image_id} "
            f"{q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} "
            f"{camera_id} {image_name}"
        )
        img_lines.append('')
    if missing:
        raise KeyError('Train image names missing from COLMAP database:\n' + '\n'.join(missing[:20]))
    (sparse_gt_dir / 'images.txt').write_text('\n'.join(img_lines) + '\n')

    # Empty points3D.txt is the expected known-pose triangulation input.
    (sparse_gt_dir / 'points3D.txt').write_text(
        '# 3D point list with one line of data per point:\n'
        '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
        '# Number of points: 0, mean track length: 0\n'
    )


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print('[cmd] ' + ' '.join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def colmap_help_text(colmap_bin: str, command: str) -> str:
    """Return COLMAP help text for one command. Some COLMAP builds/options differ by version.

    We use this to avoid passing options such as --SiftExtraction.use_gpu when
    the installed COLMAP build does not expose them.
    """
    try:
        proc = subprocess.run(
            [colmap_bin, command, '-h'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.stdout or ''
    except Exception as exc:
        print(f'[warn] failed to query COLMAP help for {command}: {exc}')
        return ''


def colmap_supports_option(colmap_bin: str, command: str, option_name: str) -> bool:
    return option_name in colmap_help_text(colmap_bin, command)


def append_colmap_option_if_supported(cmd: List[str], colmap_bin: str, command: str, option_name: str, value: object) -> None:
    """Append --option value only if the current COLMAP command supports it."""
    if colmap_supports_option(colmap_bin, command, option_name):
        cmd.extend([option_name, str(value)])
    else:
        print(f'[warn] COLMAP command `{command}` does not support {option_name}; skip it.')


def parse_points3d_txt(points_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xyz = []
    rgb = []
    if not points_path.exists():
        raise FileNotFoundError(f"Missing COLMAP points3D.txt: {points_path}")
    for raw in points_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        # POINT3D_ID X Y Z R G B ERROR ...
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(float(parts[4])), int(float(parts[5])), int(float(parts[6]))])
    if not xyz:
        raise RuntimeError(f"No triangulated points found in {points_path}")
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb size mismatch: {xyz.shape} vs {rgb.shape}")
    with path.open('w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {xyz.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(xyz, rgb):
            f.write(
                f'{float(p[0]):.8f} {float(p[1]):.8f} {float(p[2]):.8f} '
                f'0.0 0.0 0.0 {int(c[0])} {int(c[1])} {int(c[2])}\n'
            )


# -----------------------------------------------------------------------------
# Main preparation
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare GT-pose COLMAP triangulated point initialization for 3DGS.')
    p.add_argument('--frame_ranges', required=True, help='Frame/local indices to include, e.g. 0-49 or 0-3,5-8.')
    p.add_argument('--dataset_start_index', type=int, default=0)
    p.add_argument('--output_scene', required=True, help='Final 3DGS scene folder.')
    p.add_argument('--colmap_workspace', default=None, help='COLMAP workspace. Default: <output_scene>/colmap_workspace')

    p.add_argument('--image_dir', required=True, help='Input image directory. V2 is left-only; COLMAP GPU/matcher options are auto-detected from command help.')
    p.add_argument('--image_pattern', default='{index:06d}.png')
    p.add_argument('--pose_file', required=True, help='GT pose file.')

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
    p.add_argument('--camera_model', default='PINHOLE', choices=['PINHOLE', 'SIMPLE_PINHOLE', 'OPENCV'])
    p.add_argument('--camera_params', default=None, help='Override COLMAP camera params string. Default from fx/fy/cx/cy.')

    p.add_argument('--strict_image_size', action='store_true')
    p.add_argument('--copy_images', action='store_true', help='Copy images instead of symlinking.')
    p.add_argument('--overwrite', action='store_true', help='Delete existing output_scene/colmap_workspace before preparing.')

    p.add_argument('--internal_split', action='store_true', help='Enable interleaved train/test split.')
    p.add_argument('--split_every', type=int, default=5)
    p.add_argument('--split_offset', type=int, default=4)
    p.add_argument('--split_index_mode', choices=['local_index', 'frame_index'], default='local_index')

    p.add_argument('--run_colmap', action='store_true', help='Run feature extraction, matching, triangulation, and PLY conversion.')
    p.add_argument('--colmap_bin', default='colmap')
    p.add_argument('--matcher', choices=['exhaustive', 'sequential'], default='exhaustive')
    p.add_argument('--sift_use_gpu', type=int, default=1, choices=[0, 1], help='Requested SIFT GPU flag. V2 only passes it if the installed COLMAP command exposes --SiftExtraction.use_gpu / --SiftMatching.use_gpu.')
    p.add_argument('--guided_matching', type=int, default=1, choices=[0, 1])
    p.add_argument('--sequential_overlap', type=int, default=10)
    p.add_argument('--skip_if_database_exists', action='store_true', help='Do not rerun feature extraction if database.db already exists.')
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

    output_scene = Path(args.output_scene)
    colmap_ws = Path(args.colmap_workspace) if args.colmap_workspace else output_scene / 'colmap_workspace'

    if args.overwrite:
        if output_scene.exists():
            print(f'[clean] remove output_scene: {output_scene}')
            shutil.rmtree(output_scene)
        if colmap_ws.exists() and colmap_ws != output_scene / 'colmap_workspace':
            print(f'[clean] remove colmap_workspace: {colmap_ws}')
            shutil.rmtree(colmap_ws)
    else:
        # Avoid silently mixing old COLMAP sparse points with new split/settings.
        if (output_scene / 'points3d.ply').exists():
            raise FileExistsError(f'{output_scene / "points3d.ply"} already exists. Use --overwrite or a new output_scene.')
        if (colmap_ws / 'database.db').exists() and not args.skip_if_database_exists:
            raise FileExistsError(f'{colmap_ws / "database.db"} already exists. Use --overwrite or --skip_if_database_exists.')

    output_scene.mkdir(parents=True, exist_ok=True)
    (output_scene / 'images').mkdir(parents=True, exist_ok=True)
    (colmap_ws / 'images').mkdir(parents=True, exist_ok=True)

    frame_locals = parse_int_ranges(args.frame_ranges)
    if not frame_locals:
        raise ValueError('--frame_ranges produced no frames')
    frame_indices = [args.dataset_start_index + i for i in frame_locals]

    poses = read_pose_file(Path(args.pose_file), args.pose_format, args.gt_quat_order, args.gt_matrix_convention)

    fovx = 2.0 * math.atan(args.width / (2.0 * args.fx))
    fovy = 2.0 * math.atan(args.height / (2.0 * args.fy))
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

    train_frames = []
    test_frames = []
    records = []
    train_records = []

    for local_i, frame_index in zip(frame_locals, frame_indices):
        if frame_index not in poses:
            raise KeyError(f'Pose index {frame_index} not found in {args.pose_file}')
        src = Path(args.image_dir) / args.image_pattern.format(index=frame_index, local_index=local_i)
        if not src.exists():
            raise FileNotFoundError(f'Missing image: {src}')
        with Image.open(src) as im:
            w, h = im.size
        if args.strict_image_size and (w != args.width or h != args.height):
            raise ValueError(f'Image size mismatch for {src}: got {w}x{h}, expected {args.width}x{args.height}')

        split = 'test' if is_test_frame(local_i, frame_index, args) else 'train'
        scene_name = f'{frame_index:06d}.png'
        scene_file_no_ext = f'images/{frame_index:06d}'
        copy_or_link(src, output_scene / 'images' / scene_name, args.copy_images)

        T_nerf = raw_twc_to_nerf_twc(poses[frame_index], args.pose_convention)
        frame_entry = {
            'file_path': scene_file_no_ext,
            'transform_matrix': T_nerf.astype(float).tolist(),
            'camera_name': 'left',
            'frame_index': int(frame_index),
            'local_index': int(local_i),
        }
        if split == 'train':
            train_frames.append(frame_entry)
        else:
            test_frames.append(frame_entry)

        rec = {
            'local_index': int(local_i),
            'frame_index': int(frame_index),
            'split': split,
            'source_image': str(src),
            'scene_image_name': scene_name,
        }

        # COLMAP train-only workspace.
        if split == 'train':
            colmap_name = f'{frame_index:06d}.png'
            copy_or_link(src, colmap_ws / 'images' / colmap_name, args.copy_images)
            Twc_cv = raw_twc_to_opencv_twc(poses[frame_index], args.pose_convention)
            qvec, tvec = opencv_twc_to_colmap_qvec_tvec(Twc_cv)
            train_records.append({
                **rec,
                'colmap_image_name': colmap_name,
                'colmap_qvec_wxyz': qvec.tolist(),
                'colmap_tvec': tvec.tolist(),
            })
            rec['colmap_used_for_triangulation'] = True
            rec['colmap_image_name'] = colmap_name
        else:
            rec['colmap_used_for_triangulation'] = False
        records.append(rec)

    (output_scene / 'transforms_train.json').write_text(json.dumps({**meta_base, 'frames': train_frames}, indent=2))
    (output_scene / 'transforms_test.json').write_text(json.dumps({**meta_base, 'frames': test_frames}, indent=2))

    train_ts = [r['frame_index'] for r in records if r['split'] == 'train']
    test_ts = [r['frame_index'] for r in records if r['split'] == 'test']

    # Initial summary before COLMAP.
    summary = {
        'output_scene': str(output_scene),
        'colmap_workspace': str(colmap_ws),
        'mode': 'left_only_gt_pose_colmap_triangulation',
        'frame_ranges': args.frame_ranges,
        'dataset_start_index': args.dataset_start_index,
        'num_timestamps_total': len(frame_indices),
        'num_train_timestamps': len(train_ts),
        'num_test_timestamps': len(test_ts),
        'train_frame_indices': train_ts,
        'test_frame_indices': test_ts,
        'strict_heldout_rule': 'Test frames are written to transforms_test.json only; they are not copied into colmap_workspace/images and do not participate in feature extraction, matching, or triangulation.',
        'split': {
            'internal_split': args.internal_split,
            'split_every': args.split_every,
            'split_offset': args.split_offset,
            'split_index_mode': args.split_index_mode,
        },
        'camera': {
            'model': args.camera_model,
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
            'pose_format': args.pose_format,
            'pose_convention': args.pose_convention,
            'gt_quat_order': args.gt_quat_order,
            'gt_matrix_convention': args.gt_matrix_convention,
        },
        'colmap': {
            'run_colmap': bool(args.run_colmap),
            'matcher': args.matcher,
            'sift_use_gpu': args.sift_use_gpu,
            'guided_matching': args.guided_matching,
            'database_path': str(colmap_ws / 'database.db'),
            'sparse_gt_path': str(colmap_ws / 'sparse_gt'),
            'sparse_triangulated_path': str(colmap_ws / 'sparse_triangulated'),
            'sparse_triangulated_txt_path': str(colmap_ws / 'sparse_triangulated_txt'),
            'points3d_ply': str(output_scene / 'points3d.ply'),
        },
        'records': records,
    }

    print(f'[write] {output_scene / "transforms_train.json"}: {len(train_frames)} train views')
    print(f'[write] {output_scene / "transforms_test.json"}:  {len(test_frames)} test views')
    print(f'[split] train timestamps={len(train_ts)}, test timestamps={len(test_ts)}')
    print(f'[colmap] train-only images={len(train_records)} under {colmap_ws / "images"}')

    if args.run_colmap:
        db_path = colmap_ws / 'database.db'
        if not (args.skip_if_database_exists and db_path.exists()):
            if db_path.exists():
                db_path.unlink()
            if args.camera_params is not None:
                camera_params = args.camera_params
            elif args.camera_model == 'PINHOLE':
                camera_params = f'{args.fx},{args.fy},{args.cx},{args.cy}'
            elif args.camera_model == 'SIMPLE_PINHOLE':
                # Use fx as f.
                camera_params = f'{args.fx},{args.cx},{args.cy}'
            elif args.camera_model == 'OPENCV':
                # fx,fy,cx,cy,k1,k2,p1,p2. Default no distortion.
                camera_params = f'{args.fx},{args.fy},{args.cx},{args.cy},0,0,0,0'
            else:
                raise ValueError(args.camera_model)

            feature_cmd = [
                args.colmap_bin, 'feature_extractor',
                '--database_path', str(db_path),
                '--image_path', str(colmap_ws / 'images'),
                '--ImageReader.single_camera', '1',
                '--ImageReader.camera_model', args.camera_model,
                '--ImageReader.camera_params', camera_params,
            ]
            append_colmap_option_if_supported(
                feature_cmd, args.colmap_bin, 'feature_extractor',
                '--SiftExtraction.use_gpu', args.sift_use_gpu,
            )
            run_cmd(feature_cmd)
        else:
            print(f'[skip] reuse existing database: {db_path}')

        # After feature extraction, DB image IDs are known. Write GT sparse model with matching IDs.
        db_images, db_cameras = read_colmap_database_images_and_cameras(db_path)
        write_known_pose_sparse_model(colmap_ws / 'sparse_gt', db_images, db_cameras, train_records)
        print(f'[write] known-pose COLMAP sparse model: {colmap_ws / "sparse_gt"}')

        if args.matcher == 'exhaustive':
            match_cmd = [
                args.colmap_bin, 'exhaustive_matcher',
                '--database_path', str(db_path),
            ]
            append_colmap_option_if_supported(
                match_cmd, args.colmap_bin, 'exhaustive_matcher',
                '--SiftMatching.use_gpu', args.sift_use_gpu,
            )
            append_colmap_option_if_supported(
                match_cmd, args.colmap_bin, 'exhaustive_matcher',
                '--FeatureMatching.guided_matching', args.guided_matching,
            )
            run_cmd(match_cmd)
        else:
            match_cmd = [
                args.colmap_bin, 'sequential_matcher',
                '--database_path', str(db_path),
            ]
            append_colmap_option_if_supported(
                match_cmd, args.colmap_bin, 'sequential_matcher',
                '--SiftMatching.use_gpu', args.sift_use_gpu,
            )
            append_colmap_option_if_supported(
                match_cmd, args.colmap_bin, 'sequential_matcher',
                '--FeatureMatching.guided_matching', args.guided_matching,
            )
            append_colmap_option_if_supported(
                match_cmd, args.colmap_bin, 'sequential_matcher',
                '--SequentialMatching.overlap', args.sequential_overlap,
            )
            run_cmd(match_cmd)

        sparse_triangulated = colmap_ws / 'sparse_triangulated'
        sparse_triangulated_txt = colmap_ws / 'sparse_triangulated_txt'
        if sparse_triangulated.exists():
            shutil.rmtree(sparse_triangulated)
        if sparse_triangulated_txt.exists():
            shutil.rmtree(sparse_triangulated_txt)
        sparse_triangulated.mkdir(parents=True, exist_ok=True)
        sparse_triangulated_txt.mkdir(parents=True, exist_ok=True)

        triangulate_cmd = [
            args.colmap_bin, 'point_triangulator',
            '--database_path', str(db_path),
            '--image_path', str(colmap_ws / 'images'),
            '--input_path', str(colmap_ws / 'sparse_gt'),
            '--output_path', str(sparse_triangulated),
        ]
        for opt_name, opt_val in [
            ('--Mapper.ba_refine_focal_length', 0),
            ('--Mapper.ba_refine_principal_point', 0),
            ('--Mapper.ba_refine_extra_params', 0),
        ]:
            append_colmap_option_if_supported(
                triangulate_cmd, args.colmap_bin, 'point_triangulator', opt_name, opt_val
            )
        run_cmd(triangulate_cmd)

        run_cmd([
            args.colmap_bin, 'model_converter',
            '--input_path', str(sparse_triangulated),
            '--output_path', str(sparse_triangulated_txt),
            '--output_type', 'TXT',
        ])

        xyz, rgb = parse_points3d_txt(sparse_triangulated_txt / 'points3D.txt')
        ply_path = output_scene / 'points3d.ply'
        write_points_ply(ply_path, xyz, rgb)
        summary['colmap']['num_triangulated_points'] = int(xyz.shape[0])
        summary['colmap']['points3d_ply_written'] = str(ply_path)
        print(f'[write] {ply_path}: {xyz.shape[0]:,} points')
    else:
        print('[note] --run_colmap not set. Scene/known split prepared, but points3d.ply not generated yet.')

    (output_scene / 'colmap_gtpose_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[write] {output_scene / "colmap_gtpose_summary.json"}')
    print('[done]')
    print('[next] Train with train_3dgs_vanilla_metrics_v4_fix_eval_logging.py and pass --allow_existing_geometry (do NOT pass --reset_random_points3d).')


if __name__ == '__main__':
    main()
