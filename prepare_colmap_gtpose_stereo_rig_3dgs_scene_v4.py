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
  (V3 adds optional --stereo_colmap with right image/pose only for COLMAP triangulation.)
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




def write_rig_config_json(
    path: Path,
    left_prefix: str,
    right_prefix: str,
    q_right_from_left_wxyz: np.ndarray,
    t_right_from_left: np.ndarray,
    camera_model: str,
    camera_params: List[float],
) -> None:
    """Write COLMAP rig_config.json for a calibrated stereo rig.

    The left camera is the reference sensor. The right sensor pose is camera_from_rig,
    i.e. right_cam_from_left_cam, because the rig coordinate is the left camera coordinate.
    """
    cfg = [
        {
            "cameras": [
                {
                    "image_prefix": left_prefix,
                    "ref_sensor": True,
                    "camera_model_name": camera_model,
                    "camera_params": [float(x) for x in camera_params],
                },
                {
                    "image_prefix": right_prefix,
                    "cam_from_rig_rotation": [float(x) for x in q_right_from_left_wxyz],
                    "cam_from_rig_translation": [float(x) for x in t_right_from_left],
                    "camera_model_name": camera_model,
                    "camera_params": [float(x) for x in camera_params],
                },
            ]
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def write_known_pose_stereo_rig_sparse_model(
    sparse_gt_dir: Path,
    db_images: Dict[str, dict],
    db_cameras: Dict[int, dict],
    train_records: List[dict],
    q_right_from_left_wxyz: np.ndarray,
    t_right_from_left: np.ndarray,
) -> None:
    """Write COLMAP text model with rigs.txt/frames.txt/cameras.txt/images.txt/points3D.txt.

    The model uses the left camera as rig reference. For each frame, RIG_FROM_WORLD is
    left_cam_from_world. The right image pose is generated from the fixed rig extrinsic,
    not from per-frame right pose, so the text model is strictly rig-consistent.
    """
    sparse_gt_dir.mkdir(parents=True, exist_ok=True)

    # Identify DB names and camera IDs.
    left_names = [r['left_colmap_name'] for r in train_records]
    right_names = [r['right_colmap_name'] for r in train_records]
    missing = [n for n in left_names + right_names if n not in db_images]
    if missing:
        raise KeyError('Image names missing from COLMAP database:\n' + '\n'.join(missing[:30]))

    left_camera_ids = sorted({int(db_images[n]['camera_id']) for n in left_names})
    right_camera_ids = sorted({int(db_images[n]['camera_id']) for n in right_names})
    if len(left_camera_ids) != 1 or len(right_camera_ids) != 1:
        raise RuntimeError(f'Expected one camera per folder. left={left_camera_ids}, right={right_camera_ids}')
    left_cam_id = left_camera_ids[0]
    right_cam_id = right_camera_ids[0]
    if left_cam_id == right_cam_id:
        raise RuntimeError('Left and right folders unexpectedly share the same CAMERA_ID. Use --ImageReader.single_camera_per_folder 1.')

    # cameras.txt
    cam_ids_needed = sorted({left_cam_id, right_cam_id})
    cam_lines = [
        '# Camera list with one line of data per camera:',
        '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
        '# Number of cameras: {}'.format(len(cam_ids_needed)),
    ]
    for camera_id in cam_ids_needed:
        cam = db_cameras[camera_id]
        params = ' '.join(f'{float(x):.17g}' for x in cam['params'])
        cam_lines.append(f"{camera_id} {cam['model_name']} {cam['width']} {cam['height']} {params}")
    (sparse_gt_dir / 'cameras.txt').write_text('\n'.join(cam_lines) + '\n')

    # rigs.txt. Format excludes the ref sensor from SENSORS[] in the common exported text format.
    q = q_right_from_left_wxyz
    t = t_right_from_left
    rig_lines = [
        '# Rig calib list with one line of data per calib:',
        '#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])',
        '# Number of rigs: 1',
        (
            f"1 2 CAMERA {left_cam_id} "
            f"CAMERA {right_cam_id} 1 "
            f"{q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g}"
        ),
    ]
    (sparse_gt_dir / 'rigs.txt').write_text('\n'.join(rig_lines) + '\n')

    # images.txt and frames.txt.
    img_lines = [
        '# Image list with two lines of data per image:',
        '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME',
        '#   POINTS2D[] as (X, Y, POINT3D_ID)',
        '# Number of images: {}'.format(len(train_records) * 2),
    ]
    frame_lines = [
        '# Frame list with one line of data per frame:',
        '#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)',
        '# Number of frames: {}'.format(len(train_records)),
    ]

    # Fixed right_cam_from_left matrix.
    R_rl = SciRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T_right_from_left = np.eye(4, dtype=np.float64)
    T_right_from_left[:3, :3] = R_rl
    T_right_from_left[:3, 3] = t

    for frame_id, rec in enumerate(train_records, start=1):
        left_name = rec['left_colmap_name']
        right_name = rec['right_colmap_name']
        left_img_id = int(db_images[left_name]['image_id'])
        right_img_id = int(db_images[right_name]['image_id'])

        Twc_left = np.asarray(rec['Twc_left_cv'], dtype=np.float64)
        T_left_from_world = np.linalg.inv(Twc_left)
        T_right_from_world = T_right_from_left @ T_left_from_world

        # Left image pose = rig_from_world.
        R_lw = T_left_from_world[:3, :3]
        t_lw = T_left_from_world[:3, 3]
        q_l_xyzw = SciRot.from_matrix(R_lw).as_quat()
        q_l = np.array([q_l_xyzw[3], q_l_xyzw[0], q_l_xyzw[1], q_l_xyzw[2]], dtype=np.float64)
        q_l = q_l / max(np.linalg.norm(q_l), 1e-12)

        # Right image pose derived from fixed stereo rig.
        R_rw = T_right_from_world[:3, :3]
        t_rw = T_right_from_world[:3, 3]
        q_r_xyzw = SciRot.from_matrix(R_rw).as_quat()
        q_r = np.array([q_r_xyzw[3], q_r_xyzw[0], q_r_xyzw[1], q_r_xyzw[2]], dtype=np.float64)
        q_r = q_r / max(np.linalg.norm(q_r), 1e-12)

        img_lines.append(
            f"{left_img_id} "
            f"{q_l[0]:.17g} {q_l[1]:.17g} {q_l[2]:.17g} {q_l[3]:.17g} "
            f"{t_lw[0]:.17g} {t_lw[1]:.17g} {t_lw[2]:.17g} "
            f"{left_cam_id} {left_name}"
        )
        img_lines.append('')
        img_lines.append(
            f"{right_img_id} "
            f"{q_r[0]:.17g} {q_r[1]:.17g} {q_r[2]:.17g} {q_r[3]:.17g} "
            f"{t_rw[0]:.17g} {t_rw[1]:.17g} {t_rw[2]:.17g} "
            f"{right_cam_id} {right_name}"
        )
        img_lines.append('')

        frame_lines.append(
            f"{frame_id} 1 "
            f"{q_l[0]:.17g} {q_l[1]:.17g} {q_l[2]:.17g} {q_l[3]:.17g} "
            f"{t_lw[0]:.17g} {t_lw[1]:.17g} {t_lw[2]:.17g} "
            f"2 CAMERA {left_cam_id} {left_img_id} CAMERA {right_cam_id} {right_img_id}"
        )

    (sparse_gt_dir / 'images.txt').write_text('\n'.join(img_lines) + '\n')
    (sparse_gt_dir / 'frames.txt').write_text('\n'.join(frame_lines) + '\n')
    (sparse_gt_dir / 'points3D.txt').write_text(
        '# 3D point list with one line of data per point:\n'
        '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
        '# Number of points: 0, mean track length: 0\n'
    )

    return None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare original COLMAP stereo-rig GT-pose triangulated point initialization for 3DGS. COLMAP uses rig_configurator and rigs/frames text model; 3DGS optimization remains left-view only.')
    p.add_argument('--frame_ranges', required=True)
    p.add_argument('--dataset_start_index', type=int, default=0)
    p.add_argument('--output_scene', required=True)
    p.add_argument('--colmap_workspace', default=None)

    p.add_argument('--left_image_dir', '--image_dir', dest='left_image_dir', required=True)
    p.add_argument('--left_image_pattern', '--image_pattern', dest='left_image_pattern', default='{index:06d}_lcam_front.png')
    p.add_argument('--left_pose_file', '--pose_file', dest='left_pose_file', required=True)
    p.add_argument('--right_image_dir', required=True)
    p.add_argument('--right_image_pattern', default='{index:06d}_rcam_front.png')
    p.add_argument('--right_pose_file', required=True)

    p.add_argument('--pose_format', choices=['tartanair_twc', 'tum_twc'], default='tartanair_twc')
    p.add_argument('--pose_convention', choices=['resplat_tartanair_pose', 'opencv_c2w', 'opengl_c2w', 'nerf_c2w'], default='resplat_tartanair_pose')
    p.add_argument('--gt_quat_order', choices=['xyzw', 'wxyz'], default='xyzw')
    p.add_argument('--gt_matrix_convention', choices=['c2w', 'w2c'], default='c2w')
    p.add_argument('--right_pose_format', choices=['tartanair_twc', 'tum_twc'], default=None)
    p.add_argument('--right_pose_convention', choices=['resplat_tartanair_pose', 'opencv_c2w', 'opengl_c2w', 'nerf_c2w'], default=None)
    p.add_argument('--right_gt_quat_order', choices=['xyzw', 'wxyz'], default=None)
    p.add_argument('--right_gt_matrix_convention', choices=['c2w', 'w2c'], default=None)

    p.add_argument('--fx', type=float, required=True)
    p.add_argument('--fy', type=float, default=None)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--camera_model', default='PINHOLE', choices=['PINHOLE', 'SIMPLE_PINHOLE', 'OPENCV'])
    p.add_argument('--camera_params', default=None)

    p.add_argument('--strict_image_size', action='store_true')
    p.add_argument('--copy_images', action='store_true')
    p.add_argument('--overwrite', action='store_true')

    p.add_argument('--internal_split', action='store_true')
    p.add_argument('--split_every', type=int, default=5)
    p.add_argument('--split_offset', type=int, default=4)
    p.add_argument('--split_index_mode', choices=['local_index', 'frame_index'], default='local_index')

    p.add_argument('--run_colmap', action='store_true')
    p.add_argument('--colmap_bin', default='colmap')
    p.add_argument('--matcher', choices=['exhaustive', 'sequential'], default='exhaustive')
    p.add_argument('--sift_use_gpu', type=int, default=1, choices=[0, 1])
    p.add_argument('--guided_matching', type=int, default=1, choices=[0, 1])
    p.add_argument('--sequential_overlap', type=int, default=10)
    p.add_argument('--triangulate_two_view_tracks', type=int, default=1, choices=[0, 1], help='Pass --Mapper.tri_ignore_two_view_tracks 0 if supported, to keep stereo two-view tracks.')
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.fy is None: args.fy = args.fx
    if args.cx is None: args.cx = args.width / 2.0
    if args.cy is None: args.cy = args.height / 2.0
    if args.right_pose_format is None: args.right_pose_format = args.pose_format
    if args.right_pose_convention is None: args.right_pose_convention = args.pose_convention
    if args.right_gt_quat_order is None: args.right_gt_quat_order = args.gt_quat_order
    if args.right_gt_matrix_convention is None: args.right_gt_matrix_convention = args.gt_matrix_convention
    if args.split_every <= 0:
        raise ValueError('--split_every must be positive')
    if args.split_offset < 0 or args.split_offset >= args.split_every:
        raise ValueError('--split_offset must satisfy 0 <= split_offset < split_every')

    output_scene = Path(args.output_scene)
    colmap_ws = Path(args.colmap_workspace) if args.colmap_workspace else output_scene / 'colmap_workspace'
    images_root = colmap_ws / 'images'
    left_colmap_dir = images_root / 'rig1' / 'left'
    right_colmap_dir = images_root / 'rig1' / 'right'

    if args.overwrite:
        if output_scene.exists():
            print(f'[clean] remove output_scene: {output_scene}')
            shutil.rmtree(output_scene)
        if colmap_ws.exists() and colmap_ws != output_scene / 'colmap_workspace':
            print(f'[clean] remove colmap_workspace: {colmap_ws}')
            shutil.rmtree(colmap_ws)
    else:
        if (output_scene / 'points3d.ply').exists():
            raise FileExistsError(f'{output_scene / "points3d.ply"} already exists. Use --overwrite or new output_scene.')
        if (colmap_ws / 'database.db').exists():
            raise FileExistsError(f'{colmap_ws / "database.db"} already exists. Use --overwrite or new output_scene.')

    (output_scene / 'images').mkdir(parents=True, exist_ok=True)
    left_colmap_dir.mkdir(parents=True, exist_ok=True)
    right_colmap_dir.mkdir(parents=True, exist_ok=True)

    frame_locals = parse_int_ranges(args.frame_ranges)
    frame_indices = [args.dataset_start_index + i for i in frame_locals]
    if not frame_indices:
        raise ValueError('--frame_ranges produced no frames')

    left_poses_raw = read_pose_file(Path(args.left_pose_file), args.pose_format, args.gt_quat_order, args.gt_matrix_convention)
    right_poses_raw = read_pose_file(Path(args.right_pose_file), args.right_pose_format, args.right_gt_quat_order, args.right_gt_matrix_convention)

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

    train_frames, test_frames, records, train_records = [], [], [], []
    rel_mats = []

    for local_i, frame_index in zip(frame_locals, frame_indices):
        if frame_index not in left_poses_raw:
            raise KeyError(f'Left pose index {frame_index} not found')
        if frame_index not in right_poses_raw:
            raise KeyError(f'Right pose index {frame_index} not found')

        left_src = Path(args.left_image_dir) / args.left_image_pattern.format(index=frame_index, local_index=local_i)
        right_src = Path(args.right_image_dir) / args.right_image_pattern.format(index=frame_index, local_index=local_i)
        if not left_src.exists(): raise FileNotFoundError(f'Missing left image: {left_src}')
        if not right_src.exists(): raise FileNotFoundError(f'Missing right image: {right_src}')

        if args.strict_image_size:
            with Image.open(left_src) as im:
                if im.size != (args.width, args.height):
                    raise ValueError(f'Left image size mismatch: {left_src}: {im.size}')
            with Image.open(right_src) as im:
                if im.size != (args.width, args.height):
                    raise ValueError(f'Right image size mismatch: {right_src}: {im.size}')

        split = 'test' if is_test_frame(local_i, frame_index, args) else 'train'
        scene_name = f'{frame_index:06d}.png'
        copy_or_link(left_src, output_scene / 'images' / scene_name, args.copy_images)
        T_nerf = raw_twc_to_nerf_twc(left_poses_raw[frame_index], args.pose_convention)
        entry = {
            'file_path': f'images/{frame_index:06d}',
            'transform_matrix': T_nerf.astype(float).tolist(),
            'camera_name': 'left',
            'frame_index': int(frame_index),
            'local_index': int(local_i),
        }
        if split == 'train': train_frames.append(entry)
        else: test_frames.append(entry)

        rec = {'local_index': int(local_i), 'frame_index': int(frame_index), 'split': split}
        if split == 'train':
            name = f'{frame_index:06d}.png'
            left_name = f'rig1/left/{name}'
            right_name = f'rig1/right/{name}'
            copy_or_link(left_src, left_colmap_dir / name, args.copy_images)
            copy_or_link(right_src, right_colmap_dir / name, args.copy_images)
            Twc_left_cv = raw_twc_to_opencv_twc(left_poses_raw[frame_index], args.pose_convention)
            Twc_right_cv = raw_twc_to_opencv_twc(right_poses_raw[frame_index], args.right_pose_convention)
            T_right_from_left = np.linalg.inv(Twc_right_cv) @ Twc_left_cv
            rel_mats.append(T_right_from_left)
            train_records.append({
                'local_index': int(local_i),
                'frame_index': int(frame_index),
                'left_colmap_name': left_name,
                'right_colmap_name': right_name,
                'Twc_left_cv': Twc_left_cv.tolist(),
                'T_right_from_left_raw': T_right_from_left.tolist(),
            })
            rec['colmap_used_for_triangulation'] = True
            rec['left_colmap_name'] = left_name
            rec['right_colmap_name'] = right_name
        else:
            rec['colmap_used_for_triangulation'] = False
        records.append(rec)

    if not train_records:
        raise RuntimeError('No train records. Check split settings.')

    # Use the first train pair to define the fixed stereo rig. Report deviations for sanity.
    T_right_from_left = np.asarray(rel_mats[0], dtype=np.float64)
    q_rl_xyzw = SciRot.from_matrix(T_right_from_left[:3, :3]).as_quat()
    q_rl_wxyz = np.array([q_rl_xyzw[3], q_rl_xyzw[0], q_rl_xyzw[1], q_rl_xyzw[2]], dtype=np.float64)
    q_rl_wxyz = q_rl_wxyz / max(np.linalg.norm(q_rl_wxyz), 1e-12)
    t_rl = T_right_from_left[:3, 3].astype(np.float64)

    trans_devs = [float(np.linalg.norm(np.asarray(M)[:3, 3] - t_rl)) for M in rel_mats]
    rot_devs_deg = []
    R0 = T_right_from_left[:3, :3]
    for M in rel_mats:
        dR = np.asarray(M)[:3, :3] @ R0.T
        rot_devs_deg.append(float(SciRot.from_matrix(dR).magnitude() * 180.0 / math.pi))

    (output_scene / 'transforms_train.json').write_text(json.dumps({**meta_base, 'frames': train_frames}, indent=2))
    (output_scene / 'transforms_test.json').write_text(json.dumps({**meta_base, 'frames': test_frames}, indent=2))

    if args.camera_params is not None:
        camera_params_str = args.camera_params
        camera_params_list = [float(x) for x in args.camera_params.split(',')]
    elif args.camera_model == 'PINHOLE':
        camera_params_str = f'{args.fx},{args.fy},{args.cx},{args.cy}'
        camera_params_list = [args.fx, args.fy, args.cx, args.cy]
    elif args.camera_model == 'SIMPLE_PINHOLE':
        camera_params_str = f'{args.fx},{args.cx},{args.cy}'
        camera_params_list = [args.fx, args.cx, args.cy]
    elif args.camera_model == 'OPENCV':
        camera_params_str = f'{args.fx},{args.fy},{args.cx},{args.cy},0,0,0,0'
        camera_params_list = [args.fx, args.fy, args.cx, args.cy, 0, 0, 0, 0]
    else:
        raise ValueError(args.camera_model)

    train_ts = [r['frame_index'] for r in records if r['split'] == 'train']
    test_ts = [r['frame_index'] for r in records if r['split'] == 'test']

    summary = {
        'output_scene': str(output_scene),
        'colmap_workspace': str(colmap_ws),
        'mode': 'original_colmap_stereo_rig_gt_pose_triangulation_left_3dgs_supervision',
        'num_timestamps_total': len(frame_indices),
        'num_train_timestamps': len(train_ts),
        'num_test_timestamps': len(test_ts),
        'train_frame_indices': train_ts,
        'test_frame_indices': test_ts,
        'strict_heldout_rule': 'Only train timestamps are copied into COLMAP rig workspace. Test timestamps are only written to transforms_test.json for evaluation.',
        'camera': {'model': args.camera_model, 'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy, 'width': args.width, 'height': args.height},
        'rig': {
            'reference_sensor': 'left',
            'right_cam_from_left_qwxyz': q_rl_wxyz.tolist(),
            'right_cam_from_left_t': t_rl.tolist(),
            'translation_deviation_max': max(trans_devs) if trans_devs else 0.0,
            'rotation_deviation_deg_max': max(rot_devs_deg) if rot_devs_deg else 0.0,
            'translation_deviation_mean': float(np.mean(trans_devs)) if trans_devs else 0.0,
            'rotation_deviation_deg_mean': float(np.mean(rot_devs_deg)) if rot_devs_deg else 0.0,
        },
        'colmap': {
            'run_colmap': bool(args.run_colmap),
            'database_path': str(colmap_ws / 'database.db'),
            'rig_config_path': str(colmap_ws / 'rig_config.json'),
            'sparse_gt_path': str(colmap_ws / 'sparse_gt'),
            'sparse_triangulated_path': str(colmap_ws / 'sparse_triangulated'),
            'sparse_triangulated_txt_path': str(colmap_ws / 'sparse_triangulated_txt'),
            'points3d_ply': str(output_scene / 'points3d.ply'),
            'matcher': args.matcher,
            'sift_use_gpu': args.sift_use_gpu,
            'guided_matching': args.guided_matching,
        },
        'records': records,
    }

    print(f'[write] transforms_train.json: {len(train_frames)} left train views')
    print(f'[write] transforms_test.json:  {len(test_frames)} left test views')
    print(f'[colmap-rig] train stereo frames={len(train_records)}, COLMAP images={len(train_records)*2}')
    print(f'[rig] right_from_left t={t_rl.tolist()}, max trans dev={summary["rig"]["translation_deviation_max"]:.6g}, max rot dev deg={summary["rig"]["rotation_deviation_deg_max"]:.6g}')

    if args.run_colmap:
        db_path = colmap_ws / 'database.db'
        if db_path.exists(): db_path.unlink()
        feature_cmd = [
            args.colmap_bin, 'feature_extractor',
            '--database_path', str(db_path),
            '--image_path', str(images_root),
            '--ImageReader.single_camera_per_folder', '1',
            '--ImageReader.camera_model', args.camera_model,
            '--ImageReader.camera_params', camera_params_str,
        ]
        append_colmap_option_if_supported(feature_cmd, args.colmap_bin, 'feature_extractor', '--SiftExtraction.use_gpu', args.sift_use_gpu)
        run_cmd(feature_cmd)

        # Official COLMAP stereo rig workflow: configure rig after feature extraction and before matching.
        rig_config_path = colmap_ws / 'rig_config.json'
        write_rig_config_json(
            rig_config_path,
            left_prefix='rig1/left/',
            right_prefix='rig1/right/',
            q_right_from_left_wxyz=q_rl_wxyz,
            t_right_from_left=t_rl,
            camera_model=args.camera_model,
            camera_params=camera_params_list,
        )
        run_cmd([args.colmap_bin, 'rig_configurator', '--database_path', str(db_path), '--rig_config_path', str(rig_config_path)])

        db_images, db_cameras = read_colmap_database_images_and_cameras(db_path)
        write_known_pose_stereo_rig_sparse_model(
            colmap_ws / 'sparse_gt',
            db_images,
            db_cameras,
            train_records,
            q_rl_wxyz,
            t_rl,
        )
        print(f'[write] rig-aware known-pose sparse model: {colmap_ws / "sparse_gt"}')

        if args.matcher == 'exhaustive':
            match_cmd = [args.colmap_bin, 'exhaustive_matcher', '--database_path', str(db_path)]
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'exhaustive_matcher', '--SiftMatching.use_gpu', args.sift_use_gpu)
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'exhaustive_matcher', '--FeatureMatching.guided_matching', args.guided_matching)
            # If available, use rig verification rather than treating rig as independent images.
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'exhaustive_matcher', '--FeatureMatching.rig_verification', 1)
            run_cmd(match_cmd)
        else:
            match_cmd = [args.colmap_bin, 'sequential_matcher', '--database_path', str(db_path)]
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'sequential_matcher', '--SiftMatching.use_gpu', args.sift_use_gpu)
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'sequential_matcher', '--FeatureMatching.guided_matching', args.guided_matching)
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'sequential_matcher', '--FeatureMatching.rig_verification', 1)
            append_colmap_option_if_supported(match_cmd, args.colmap_bin, 'sequential_matcher', '--SequentialMatching.overlap', args.sequential_overlap)
            run_cmd(match_cmd)

        sparse_triangulated = colmap_ws / 'sparse_triangulated'
        sparse_triangulated_txt = colmap_ws / 'sparse_triangulated_txt'
        if sparse_triangulated.exists(): shutil.rmtree(sparse_triangulated)
        if sparse_triangulated_txt.exists(): shutil.rmtree(sparse_triangulated_txt)
        sparse_triangulated.mkdir(parents=True, exist_ok=True)
        sparse_triangulated_txt.mkdir(parents=True, exist_ok=True)

        tri_cmd = [
            args.colmap_bin, 'point_triangulator',
            '--database_path', str(db_path),
            '--image_path', str(images_root),
            '--input_path', str(colmap_ws / 'sparse_gt'),
            '--output_path', str(sparse_triangulated),
        ]
        for opt_name, opt_val in [
            ('--Mapper.ba_refine_focal_length', 0),
            ('--Mapper.ba_refine_principal_point', 0),
            ('--Mapper.ba_refine_extra_params', 0),
            ('--Mapper.ba_refine_sensor_from_rig', 0),
        ]:
            append_colmap_option_if_supported(tri_cmd, args.colmap_bin, 'point_triangulator', opt_name, opt_val)
        if args.triangulate_two_view_tracks:
            append_colmap_option_if_supported(tri_cmd, args.colmap_bin, 'point_triangulator', '--Mapper.tri_ignore_two_view_tracks', 0)
        run_cmd(tri_cmd)

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
        print('[note] --run_colmap not set; scene and rig workspace prepared, but points3d.ply not generated.')

    (output_scene / 'colmap_gtpose_stereo_rig_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[write] {output_scene / "colmap_gtpose_stereo_rig_summary.json"}')
    print('[done]')
    print('[next] Train 3DGS with --allow_existing_geometry and do NOT pass --reset_random_points3d.')


if __name__ == '__main__':
    main()
