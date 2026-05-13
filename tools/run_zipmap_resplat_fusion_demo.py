#!/usr/bin/env python3
"""
Run a minimal ZipMap -> ReSplat -> Gaussian packet fusion demo.

Intended demo
-------------
1. Take N left images from a stereo sequence and run ZipMap only on the left images.
2. Evaluate ZipMap pose against GT and write Sim(3)-aligned poses.
3. Convert the aligned ZipMap poses into the pose.txt format expected by the current
   ReSplat/TartanAir-style loader.
4. Build a small stereo input folder for ReSplat:

      resplat_dataset_root/
        image_lcam_front/000000.png ...
        image_rcam_front/000000.png ...
        pose.txt

   The right image folder is built by taking the same sampled original indices from
   the original right-image directory. Right camera poses should be synthesized in
   the ReSplat loader from left poses + fixed stereo rig.
5. Run ReSplat test to save Gaussian packets.
6. Run fuse_gaussian_packets.py on the generated packets.

This script is a pipeline wrapper. It does not modify ZipMap, ReSplat, or the fusion
script internals. Fill in the paths and Hydra overrides for your local environment.

Example
-------
python tools/run_zipmap_resplat_fusion_demo.py \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --zipmap_ckpt /home/shiyo/Desktop/ZipMap/checkpoints/checkpoint_aff_inv.pt \
  --left_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/image_lcam_front/House/Data_easy/P000/image_lcam_front \
  --right_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/image_lcam_front/House/Data_easy/P000/image_rcam_front \
  --gt_pose_file /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/image_lcam_front/House/Data_easy/P000/pose_lcam_front.txt \
  --work_dir /home/shiyo/Desktop/ZipMap/outputs/demo_zipmap_resplat_10 \
  --num_frames 10 \
  --stride 1 \
  --resplat_experiment tartanair_p000_ft \
  --resplat_output_dir outputs/test/resplat_zipmap_demo_10 \
  --resplat_override dataset.sequences.0.root=/home/shiyo/Desktop/ZipMap/outputs/demo_zipmap_resplat_10/resplat_dataset \
  --resplat_override dataset.test_len=10

Notes
-----
- Use --dry_run first. It prints commands without running them.
- If your Hydra dataset root override has a different key, change --resplat_override.
- The demo uses --pose_source aligned, so GT pose is required for the pose-evaluation
  stage. For no-GT deployment, change the pipeline later to use raw pose.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def abs_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def collect_images(root: Path, recursive: bool = False) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No image files found under: {root}")
    return paths


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    dry_run: bool = False,
    title: Optional[str] = None,
) -> None:
    if title:
        print(f"\n===== {title} =====", flush=True)
    print("[CMD] " + " ".join(str(x) for x in cmd), flush=True)
    if cwd is not None:
        print(f"[CWD] {cwd}", flush=True)
    if dry_run:
        return
    subprocess.run(list(map(str, cmd)), cwd=None if cwd is None else str(cwd), env=env, check=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown copy_mode: {mode}")


def copy_or_link_tree_files(src_dir: Path, dst_dir: Path, mode: str) -> List[Path]:
    files = collect_images(src_dir, recursive=False)
    out_files = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        dst = dst_dir / f"{i:06d}{src.suffix.lower()}"
        link_or_copy(src, dst, mode)
        out_files.append(dst)
    return out_files


def load_frame_records(zipmap_dir: Path) -> List[dict]:
    p = zipmap_dir / "frame_records.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing frame_records.json: {p}")
    records = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid frame_records.json: {p}")
    return records


def build_resplat_stereo_dataset(
    *,
    prepared_left_dir: Path,
    right_dir: Path,
    out_root: Path,
    left_camera_dirname: str,
    right_camera_dirname: str,
    copy_mode: str,
    recursive: bool,
) -> None:
    """Create a ReSplat stereo folder from prepared left images/pose and original right images."""
    records = json.loads((prepared_left_dir / "frame_records.json").read_text(encoding="utf-8"))
    right_images_all = collect_images(right_dir, recursive=recursive)

    out_root.mkdir(parents=True, exist_ok=True)
    left_out = out_root / left_camera_dirname
    right_out = out_root / right_camera_dirname
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)

    # Left images come from prepare_resplat_zipmap_dataset.py output.
    prepared_images = collect_images(prepared_left_dir / "images", recursive=False)
    if len(prepared_images) != len(records):
        raise ValueError(
            f"Prepared left image count {len(prepared_images)} != frame_records count {len(records)}"
        )

    manifest = []
    for out_i, (rec, left_src) in enumerate(zip(records, prepared_images)):
        gt_index = int(rec.get("gt_index", rec.get("original_index", out_i)))
        if gt_index < 0 or gt_index >= len(right_images_all):
            raise IndexError(
                f"Right image index {gt_index} out of range. right_dir has {len(right_images_all)} images."
            )
        right_src = right_images_all[gt_index]

        left_dst = left_out / f"{out_i:06d}{left_src.suffix.lower()}"
        right_dst = right_out / f"{out_i:06d}{right_src.suffix.lower()}"
        link_or_copy(left_src, left_dst, copy_mode)
        link_or_copy(right_src, right_dst, copy_mode)
        manifest.append(
            {
                "output_index": out_i,
                "gt_index": gt_index,
                "left_source": str(left_src),
                "right_source": str(right_src),
                "left_output": str(left_dst),
                "right_output": str(right_dst),
            }
        )

    # pose.txt is left camera pose only. Your ReSplat loader should synthesize right pose
    # from fixed stereo rig, not read a right pose file.
    for name in [
        "pose.txt",
        "pose_resplat_tartanair_loader.txt",
        "pose_resplat_tartanair_loader_tum.txt",
        "pose_c2w_opencv_tum.txt",
        "intrinsics.txt",
        "frame_records.json",
        "source_images.txt",
        "meta.json",
    ]:
        src = prepared_left_dir / name
        if src.exists():
            dst = out_root / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if src.is_dir():
                continue
            if copy_mode == "symlink":
                dst.symlink_to(src.resolve())
            elif copy_mode == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)

    (out_root / "stereo_demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Done] ReSplat stereo dataset root: {out_root}")
    print(f"  left images : {left_out}")
    print(f"  right images: {right_out}")
    print(f"  pose file   : {out_root / 'pose.txt'}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Minimal ZipMap -> ReSplat -> fusion demo wrapper")

    # Repositories and helper scripts.
    ap.add_argument("--zipmap_repo", required=True, help="Path to ZipMap repository root")
    ap.add_argument("--resplat_repo", required=True, help="Path to ReSplat repository root")
    ap.add_argument("--zipmap_export_script", default="tools/export_zipmap_predictions.py")
    ap.add_argument("--zipmap_eval_script", default="tools/evaluate_zipmap_pose.py")
    ap.add_argument("--prepare_script", default="tools/prepare_resplat_zipmap_dataset.py")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use for all subprocesses")

    # Input data.
    ap.add_argument("--zipmap_ckpt", required=True)
    ap.add_argument("--left_dir", required=True, help="Original left image directory")
    ap.add_argument("--right_dir", required=True, help="Original right image directory")
    ap.add_argument("--gt_pose_file", required=True, help="GT left pose file, one row per original frame")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_frames", type=int, default=10, help="Number of left frames / stereo pairs for the demo")

    # Working outputs.
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--zipmap_out_name", default="zipmap_left_only")
    ap.add_argument("--prepared_left_name", default="prepared_left_aligned")
    ap.add_argument("--resplat_dataset_name", default="resplat_dataset")
    ap.add_argument("--left_camera_dirname", default="image_lcam_front")
    ap.add_argument("--right_camera_dirname", default="image_rcam_front")
    ap.add_argument("--copy_mode", choices=["symlink", "copy", "hardlink"], default="symlink")

    # Stage switches.
    ap.add_argument("--skip_zipmap", action="store_true")
    ap.add_argument("--skip_pose_eval", action="store_true")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_resplat", action="store_true")
    ap.add_argument("--skip_fusion", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cleanup_intermediate", type=str2bool, default=False)

    # ReSplat command.
    ap.add_argument("--cuda_visible_devices", default="0")
    ap.add_argument("--hydra_full_error", default="1")
    ap.add_argument("--resplat_experiment", default="tartanair_p000_ft")
    ap.add_argument("--resplat_output_dir", default="outputs/test/resplat_zipmap_demo")
    ap.add_argument("--resplat_num_workers", type=int, default=4)
    ap.add_argument("--resplat_test_len", type=int, default=None)
    ap.add_argument(
        "--resplat_override",
        action="append",
        default=[],
        help=(
            "Extra Hydra override. Use multiple times. Example: "
            "--resplat_override dataset.sequences.0.root=/abs/path/to/resplat_dataset"
        ),
    )

    # Fusion command.
    ap.add_argument("--fusion_output_dir", default=None)
    ap.add_argument("--fusion_packet_dir", default=None, help="Override packet dir. Default: <resplat_output_dir>/gaussian_packets/final")
    ap.add_argument("--fusion_probe_mode", default="packet_last_only")
    ap.add_argument("--fusion_device", default="cuda:0")
    ap.add_argument("--fusion_packet_ranges", default=None, help="Default is 0-(num_frames-1)")
    ap.add_argument("--fusion_extra_arg", action="append", default=[], help="Extra raw arg for fuse_gaussian_packets.py")

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    zipmap_repo = abs_path(args.zipmap_repo)
    resplat_repo = abs_path(args.resplat_repo)
    work_dir = abs_path(args.work_dir)
    left_dir = abs_path(args.left_dir)
    right_dir = abs_path(args.right_dir)
    gt_pose_file = abs_path(args.gt_pose_file)
    zipmap_ckpt = abs_path(args.zipmap_ckpt)

    zipmap_dir = work_dir / args.zipmap_out_name
    prepared_left_dir = work_dir / args.prepared_left_name
    resplat_dataset_root = work_dir / args.resplat_dataset_name
    fusion_output_dir = abs_path(args.fusion_output_dir) if args.fusion_output_dir else work_dir / "fusion_eval"

    work_dir.mkdir(parents=True, exist_ok=True)

    zipmap_export_script = abs_path(zipmap_repo / args.zipmap_export_script)
    zipmap_eval_script = abs_path(zipmap_repo / args.zipmap_eval_script)
    prepare_script = abs_path(zipmap_repo / args.prepare_script)

    # ------------------------------------------------------------------
    # 1) ZipMap left-only pose prediction.
    # ------------------------------------------------------------------
    if not args.skip_zipmap:
        export_cmd = [
            args.python,
            str(zipmap_export_script),
            "--stereo_left_dir",
            str(left_dir),
            "--stereo_right_dir",
            str(right_dir),
            "--stereo_pair_mode",
            "left_only",
            "--ckpt_path",
            str(zipmap_ckpt),
            "--output",
            str(zipmap_dir),
            "--repo_root",
            str(zipmap_repo),
            "--start_index",
            str(args.start_index),
            "--stride",
            str(args.stride),
            "--max_pairs",
            str(args.num_frames),
            "--align_first_view",
            "true",
            "--save_preprocessed_images",
            "false",
            "--save_depth_npy",
            "false",
            "--save_points_npy",
            "false",
        ]
        if args.end_index is not None:
            export_cmd += ["--end_index", str(args.end_index)]
        if args.recursive:
            export_cmd += ["--recursive"]
        run_command(export_cmd, cwd=zipmap_repo, dry_run=args.dry_run, title="1. ZipMap export")

    # ------------------------------------------------------------------
    # 2) Pose evaluation / Sim(3) alignment.
    # ------------------------------------------------------------------
    if not args.skip_pose_eval:
        eval_cmd = [
            args.python,
            str(zipmap_eval_script),
            "--zipmap_dir",
            str(zipmap_dir),
            "--gt_pose_file",
            str(gt_pose_file),
            "--gt_convention",
            "resplat_tartanair_pose",
            "--match",
            "meta_index",
            "--alignment",
            "sim3",
            "--rpe_delta",
            "1",
        ]
        run_command(eval_cmd, cwd=zipmap_repo, dry_run=args.dry_run, title="2. ZipMap pose eval / alignment")

    # ------------------------------------------------------------------
    # 3) Prepare aligned left poses/images, then make stereo ReSplat root.
    # ------------------------------------------------------------------
    if not args.skip_prepare:
        prep_cmd = [
            args.python,
            str(prepare_script),
            "--zipmap_dir",
            str(zipmap_dir),
            "--output_dir",
            str(prepared_left_dir),
            "--pose_source",
            "aligned",
            "--camera_ids",
            "all",
            "--image_source",
            "original",
            "--copy_mode",
            args.copy_mode,
            "--overwrite",
            "true",
        ]
        run_command(prep_cmd, cwd=zipmap_repo, dry_run=args.dry_run, title="3a. Prepare ReSplat left input")
        if not args.dry_run:
            build_resplat_stereo_dataset(
                prepared_left_dir=prepared_left_dir,
                right_dir=right_dir,
                out_root=resplat_dataset_root,
                left_camera_dirname=args.left_camera_dirname,
                right_camera_dirname=args.right_camera_dirname,
                copy_mode=args.copy_mode,
                recursive=args.recursive,
            )
        else:
            print(f"[Dry-run] would build ReSplat stereo dataset at {resplat_dataset_root}")

    # ------------------------------------------------------------------
    # 4) Run ReSplat test and save Gaussian packets.
    # ------------------------------------------------------------------
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["HYDRA_FULL_ERROR"] = str(args.hydra_full_error)

    resplat_test_len = args.resplat_test_len if args.resplat_test_len is not None else args.num_frames
    if not args.skip_resplat:
        resplat_cmd = [
            args.python,
            "-m",
            "src.main",
            f"+experiment={args.resplat_experiment}",
            "mode=test",
            f"data_loader.test.num_workers={args.resplat_num_workers}",
            "test.compute_scores=true",
            "test.save_image=true",
            "test.save_gaussian_packet=true",
            "test.save_gaussian_packet_stage=both",
            "test.save_gaussian=false",
            "wandb.mode=disabled",
            f"output_dir={args.resplat_output_dir}",
            f"dataset.test_len={resplat_test_len}",
        ]
        resplat_cmd.extend(args.resplat_override)
        run_command(resplat_cmd, cwd=resplat_repo, env=env, dry_run=args.dry_run, title="4. ReSplat test / packet generation")

    # ------------------------------------------------------------------
    # 5) Fuse generated packets.
    # ------------------------------------------------------------------
    if not args.skip_fusion:
        resplat_output_abs = Path(args.resplat_output_dir)
        if not resplat_output_abs.is_absolute():
            resplat_output_abs = resplat_repo / resplat_output_abs
        packet_dir = abs_path(args.fusion_packet_dir) if args.fusion_packet_dir else resplat_output_abs / "gaussian_packets" / "final"
        packet_ranges = args.fusion_packet_ranges if args.fusion_packet_ranges is not None else f"0-{args.num_frames - 1}"

        fusion_cmd = [
            args.python,
            "-m",
            "src.scripts.fuse_gaussian_packets",
            "--packet_dir",
            str(packet_dir),
            "--output_dir",
            str(fusion_output_dir),
            "--probe_mode",
            args.fusion_probe_mode,
            "--device",
            args.fusion_device,
            "--packet_ranges",
            packet_ranges,
        ]
        fusion_cmd.extend(args.fusion_extra_arg)
        run_command(fusion_cmd, cwd=resplat_repo, env=env, dry_run=args.dry_run, title="5. Gaussian packet fusion")

    if args.cleanup_intermediate and not args.dry_run:
        # Conservative cleanup: keep prepared ReSplat dataset, ReSplat output, and fusion output.
        # Remove only heavy ZipMap prediction dir.
        if zipmap_dir.exists():
            shutil.rmtree(zipmap_dir)
            print(f"[Cleanup] removed {zipmap_dir}")

    print("\n===== Demo wrapper finished =====")
    print(f"ZipMap dir             : {zipmap_dir}")
    print(f"Prepared left dir      : {prepared_left_dir}")
    print(f"ReSplat dataset root   : {resplat_dataset_root}")
    print(f"ReSplat output_dir     : {args.resplat_output_dir}")
    print(f"Fusion output_dir      : {fusion_output_dir}")
    print("Fill your ReSplat dataset YAML/root override so it reads the ReSplat dataset root above.")


if __name__ == "__main__":
    main()
