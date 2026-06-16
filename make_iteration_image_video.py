#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a video from the same rendered image across iteration_* folders."
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to eval_renders directory that contains iteration_* folders.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Image filename, e.g. 0000_000004 or 0000_000004.png",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Output video fps.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/shiyo/Desktop/ZipMap/outputs/gt_resplat_P000_0_50",
        help="Output directory.",
    )

    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="Output mp4 name. Default: <filename>_iteration_progress.mp4",
    )

    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help="Optional iteration list, e.g. 0,500,1000,2000,5000,10000. If omitted, auto-scan iteration_* folders.",
    )

    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing images instead of raising an error.",
    )

    return parser.parse_args()


def get_iteration_number(path: Path):
    m = re.match(r"iteration_(\d+)$", path.name)
    if m is None:
        return None
    return int(m.group(1))


def main():
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = args.filename
    if not filename.endswith(".png"):
        filename += ".png"

    stem = Path(filename).stem

    if args.out_name is None:
        out_name = f"{stem}_iteration_progress.mp4"
    else:
        out_name = args.out_name
        if not out_name.endswith(".mp4"):
            out_name += ".mp4"

    out_path = out_dir / out_name

    # Collect iteration folders
    if args.iterations is not None:
        iter_nums = [int(x.strip()) for x in args.iterations.split(",") if x.strip()]
        iter_folders = [(i, eval_dir / f"iteration_{i}") for i in iter_nums]
    else:
        iter_folders = []
        for p in eval_dir.iterdir():
            if not p.is_dir():
                continue
            it = get_iteration_number(p)
            if it is not None:
                iter_folders.append((it, p))
        iter_folders.sort(key=lambda x: x[0])

    if len(iter_folders) == 0:
        raise RuntimeError(f"No iteration_* folders found in: {eval_dir}")

    image_paths = []
    missing = []

    for it, folder in iter_folders:
        img = folder / "test" / "rendered" / filename
        if img.exists():
            image_paths.append((it, img))
        else:
            missing.append((it, img))

    if missing and not args.skip_missing:
        print("Missing images:")
        for it, img in missing:
            print(f"  iteration_{it}: {img}")
        raise RuntimeError("Some images are missing. Use --skip_missing to ignore them.")

    if len(image_paths) == 0:
        raise RuntimeError("No valid images found.")

    list_path = out_dir / f"_{stem}_ffmpeg_concat_list.txt"

    with open(list_path, "w") as f:
        for it, img in image_paths:
            f.write(f"file '{img.resolve()}'\n")

    print("Using frames:")
    for it, img in image_paths:
        print(f"  iteration_{it}: {img}")

    cmd = [
        "ffmpeg",
        "-y",
        "-r", str(args.fps),
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"\nSaved video to: {out_path}")


if __name__ == "__main__":
    main()