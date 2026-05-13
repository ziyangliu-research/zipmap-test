#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export fused feed-forward 3DGS packets (.pt) to a standard 3DGS PLY file.

Intended workflow:
    ReSplat/MVSplat packets (.pt)
      -> this script exports fused_map.ply
      -> Niantic SPZ Converter / other tools convert PLY to SPZ/SPZ4
      -> online 3DGS viewer displays the result

Input packet directory is expected to contain files like:
    gaussian_packets_api/final/*.pt
or any directory of packet .pt files.

Required packet fields:
    means, opacities, scales, rotations, harmonics

Expected tensor layouts, both supported:
    means      [G, 3] or [1, G, 3]
    scales     [G, 3] or [1, G, 3]
    rotations  [G, 4] or [1, G, 4]
    opacities  [G]    or [1, G]
    harmonics  [G, 3, C] or [1, G, 3, C]

Notes:
- Standard 3DGS PLY viewers usually expect opacity as logit and scales as log-scale.
  This script defaults to auto conversion:
      alpha in [0, 1] -> logit(alpha)
      positive scales -> log(scales)
- For quick SPZ debugging, start with --max_sh_degree 0 and --topk 200000.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


REQUIRED_FIELDS = ["means", "opacities", "scales", "rotations", "harmonics"]


def parse_packet_ranges(text: Optional[str], n_files: int) -> List[int]:
    """Parse inclusive ranges like '0-4,8,10-12'."""
    if text is None or str(text).strip() == "":
        return list(range(n_files))
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
            if b < a:
                raise ValueError(f"Invalid packet range: {part}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    dedup = []
    seen = set()
    for i in out:
        if i < 0 or i >= n_files:
            raise IndexError(f"Packet index out of range: {i}, valid=[0,{n_files-1}]")
        if i not in seen:
            dedup.append(i)
            seen.add(i)
    return dedup


def tensor_no_batch(x: torch.Tensor, field_name: str) -> torch.Tensor:
    """Convert supported packet tensor layout to unbatched layout."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if field_name in {"means", "scales", "rotations", "rotations_unnorm"}:
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    elif field_name == "opacities":
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    elif field_name == "harmonics":
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    return x.detach().cpu().float().contiguous()


def load_packet(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Packet is not a dict: {path}")
    missing = [k for k in REQUIRED_FIELDS if k not in obj]
    if missing:
        raise RuntimeError(f"Packet missing fields {missing}: {path}")
    out = {k: tensor_no_batch(obj[k], k) for k in REQUIRED_FIELDS}
    return out


def load_opacity_only(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if "opacities" not in obj:
        raise RuntimeError(f"Packet missing opacities: {path}")
    return tensor_no_batch(obj["opacities"], "opacities")


def infer_coeff_count(packet_path: Path, max_sh_degree: int) -> int:
    obj = torch.load(packet_path, map_location="cpu")
    if "harmonics" not in obj:
        raise RuntimeError(f"Packet missing harmonics: {packet_path}")
    h = tensor_no_batch(obj["harmonics"], "harmonics")
    available = int(h.shape[-1])
    requested = int((max_sh_degree + 1) ** 2)
    return min(available, requested)


def compute_global_opacity_cutoff(packet_paths: Sequence[Path], opacity_threshold: float, topk: int) -> Optional[float]:
    """Return kth opacity cutoff for global topk after threshold, or None."""
    if topk is None or int(topk) <= 0:
        return None
    chunks = []
    total = 0
    for path in packet_paths:
        opa = load_opacity_only(path)
        if opacity_threshold > 0:
            opa = opa[opa >= float(opacity_threshold)]
        if opa.numel() > 0:
            chunks.append(opa)
            total += int(opa.numel())
    if total == 0:
        raise RuntimeError("No Gaussians left after opacity threshold; cannot apply topk.")
    k = min(int(topk), total)
    all_opa = torch.cat(chunks, dim=0)
    # kth largest value as threshold. This may keep slightly more than k when ties exist.
    cutoff = torch.topk(all_opa, k=k, largest=True).values[-1].item()
    return float(cutoff)


def build_mask(opacities: torch.Tensor, opacity_threshold: float, topk_cutoff: Optional[float]) -> torch.Tensor:
    keep = torch.ones_like(opacities, dtype=torch.bool)
    if opacity_threshold > 0.0:
        keep &= opacities >= float(opacity_threshold)
    if topk_cutoff is not None:
        keep &= opacities >= float(topk_cutoff)
    return keep


def count_selected(packet_paths: Sequence[Path], opacity_threshold: float, topk_cutoff: Optional[float]) -> Tuple[int, List[int]]:
    counts: List[int] = []
    total = 0
    for path in packet_paths:
        opa = load_opacity_only(path)
        keep = build_mask(opa, opacity_threshold, topk_cutoff)
        c = int(keep.sum().item())
        counts.append(c)
        total += c
    return total, counts


def logit_np(alpha: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.clip(alpha, eps, 1.0 - eps)
    return np.log(a / (1.0 - a)).astype(np.float32)


def convert_opacity(opacity: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "raw":
        return opacity.astype(np.float32)
    if mode == "alpha":
        return opacity.astype(np.float32)
    if mode == "logit":
        return logit_np(opacity)
    if mode == "auto":
        if opacity.size == 0:
            return opacity.astype(np.float32)
        if np.nanmin(opacity) >= -1e-6 and np.nanmax(opacity) <= 1.0 + 1e-6:
            return logit_np(opacity)
        return opacity.astype(np.float32)
    raise ValueError(f"Unknown opacity output mode: {mode}")


def convert_scales(scales: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "raw":
        return scales.astype(np.float32)
    if mode == "log":
        return np.log(np.clip(scales, 1e-12, None)).astype(np.float32)
    if mode == "auto":
        if scales.size == 0:
            return scales.astype(np.float32)
        if np.nanmin(scales) > 0.0:
            return np.log(np.clip(scales, 1e-12, None)).astype(np.float32)
        return scales.astype(np.float32)
    raise ValueError(f"Unknown scale output mode: {mode}")


def normalize_quaternion(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (q / n).astype(np.float32)


def convert_rotation_order(rot: np.ndarray, order: str) -> np.ndarray:
    order = order.lower()
    if order == "as_saved" or order == "wxyz":
        out = rot
    elif order == "xyzw_to_wxyz":
        # input [x, y, z, w] -> output [w, x, y, z]
        out = rot[:, [3, 0, 1, 2]]
    elif order == "wxyz_to_xyzw":
        # rarely useful for standard 3DGS PLY, but kept for debugging.
        out = rot[:, [1, 2, 3, 0]]
    else:
        raise ValueError(f"Unknown rotation order: {order}")
    return out.astype(np.float32)


def ply_property_names(coeff_count: int) -> List[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    f_rest_count = max(0, (coeff_count - 1) * 3)
    names += [f"f_rest_{i}" for i in range(f_rest_count)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def write_ply_header(f, vertex_count: int, property_names: Sequence[str], binary: bool = True):
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = [
        "ply",
        f"format {fmt}",
        f"element vertex {int(vertex_count)}",
    ]
    for name in property_names:
        header.append(f"property float {name}")
    header.append("end_header")
    f.write(("\n".join(header) + "\n").encode("ascii"))


def packet_rows(
    packet: Dict[str, torch.Tensor],
    keep_indices: torch.Tensor,
    coeff_count: int,
    opacity_output: str,
    scale_output: str,
    rotation_order: str,
    normalize_rotations: bool,
    row_chunk_size: int,
) -> Iterable[np.ndarray]:
    indices = keep_indices.cpu().long()
    n = int(indices.numel())
    if n == 0:
        return
    for start in range(0, n, int(row_chunk_size)):
        idx = indices[start : start + int(row_chunk_size)]
        means = packet["means"][idx].numpy().astype(np.float32)
        harmonics = packet["harmonics"][idx, :, :coeff_count].numpy().astype(np.float32)  # [M, 3, C]
        opacity = packet["opacities"][idx].numpy().astype(np.float32)[:, None]
        scales = packet["scales"][idx].numpy().astype(np.float32)
        rotations = packet["rotations"][idx].numpy().astype(np.float32)

        normals = np.zeros_like(means, dtype=np.float32)
        f_dc = harmonics[:, :, 0].astype(np.float32)  # [M, RGB]
        if coeff_count > 1:
            # Channel-major order: R coeffs, G coeffs, B coeffs.
            f_rest = harmonics[:, :, 1:].reshape(harmonics.shape[0], -1).astype(np.float32)
        else:
            f_rest = np.empty((means.shape[0], 0), dtype=np.float32)

        opacity_out = convert_opacity(opacity, opacity_output)
        scales_out = convert_scales(scales, scale_output)
        rotations_out = convert_rotation_order(rotations, rotation_order)
        if normalize_rotations:
            rotations_out = normalize_quaternion(rotations_out)

        row = np.concatenate(
            [means, normals, f_dc, f_rest, opacity_out, scales_out, rotations_out],
            axis=1,
        ).astype(np.float32, copy=False)
        yield row


def write_ply_binary(
    output_path: Path,
    packet_paths: Sequence[Path],
    vertex_count: int,
    coeff_count: int,
    opacity_threshold: float,
    topk_cutoff: Optional[float],
    opacity_output: str,
    scale_output: str,
    rotation_order: str,
    normalize_rotations: bool,
    row_chunk_size: int,
):
    property_names = ply_property_names(coeff_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        write_ply_header(f, vertex_count, property_names, binary=True)
        written = 0
        for p_i, path in enumerate(packet_paths):
            packet = load_packet(path)
            keep = build_mask(packet["opacities"], opacity_threshold, topk_cutoff)
            keep_indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
            for row in packet_rows(
                packet=packet,
                keep_indices=keep_indices,
                coeff_count=coeff_count,
                opacity_output=opacity_output,
                scale_output=scale_output,
                rotation_order=rotation_order,
                normalize_rotations=normalize_rotations,
                row_chunk_size=row_chunk_size,
            ):
                row.tofile(f)
                written += int(row.shape[0])
            print(f"[write] packet {p_i+1}/{len(packet_paths)}: {path.name}, kept={int(keep.sum().item())}, written_total={written}")
    if written != vertex_count:
        raise RuntimeError(f"Written vertex count mismatch: header={vertex_count}, written={written}")


def main():
    parser = argparse.ArgumentParser(description="Export fused 3DGS packets to standard binary PLY.")
    parser.add_argument("--packet_dir", type=str, required=True, help="Directory containing packet .pt files.")
    parser.add_argument("--output", type=str, required=True, help="Output .ply path.")
    parser.add_argument("--packet_ranges", type=str, default=None, help="Inclusive ranges by sorted packet index, e.g. '0-4,8-10'. Default: all.")
    parser.add_argument("--max_packets", type=int, default=-1, help="Use first N selected packets after packet_ranges. <=0 means no limit.")
    parser.add_argument("--opacity_threshold", type=float, default=0.0, help="Drop Gaussians with opacity below this value before topk.")
    parser.add_argument("--topk", type=int, default=-1, help="Global top-k by opacity after threshold. <=0 keeps all.")
    parser.add_argument("--max_sh_degree", type=int, default=0, choices=[0, 1, 2, 3, 4], help="SH degree to export. 0 exports only DC color and is much smaller.")
    parser.add_argument("--opacity_output", type=str, default="auto", choices=["auto", "raw", "alpha", "logit"], help="How to write PLY opacity. auto: alpha in [0,1] -> logit.")
    parser.add_argument("--scale_output", type=str, default="auto", choices=["auto", "raw", "log"], help="How to write PLY scales. auto: positive scales -> log.")
    parser.add_argument("--rotation_order", type=str, default="as_saved", choices=["as_saved", "wxyz", "xyzw_to_wxyz", "wxyz_to_xyzw"], help="Quaternion order conversion before writing rot_0..rot_3. Standard 3DGS PLY usually uses wxyz.")
    parser.add_argument("--no_normalize_rotations", action="store_true", help="Do not normalize quaternions before writing.")
    parser.add_argument("--row_chunk_size", type=int, default=200000, help="Rows written per internal chunk.")
    parser.add_argument("--summary", type=str, default=None, help="Optional JSON summary path. Default: output path + .summary.json")
    args = parser.parse_args()

    packet_dir = Path(args.packet_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not packet_dir.exists():
        raise FileNotFoundError(f"packet_dir not found: {packet_dir}")
    all_files = sorted(packet_dir.glob("*.pt"))
    if not all_files:
        raise RuntimeError(f"No .pt files found in: {packet_dir}")

    selected_idx = parse_packet_ranges(args.packet_ranges, len(all_files))
    if args.max_packets is not None and int(args.max_packets) > 0:
        selected_idx = selected_idx[: int(args.max_packets)]
    packet_paths = [all_files[i] for i in selected_idx]
    if not packet_paths:
        raise RuntimeError("No packets selected.")

    coeff_count = infer_coeff_count(packet_paths[0], int(args.max_sh_degree))
    property_names = ply_property_names(coeff_count)

    print(f"[select] packet_dir={packet_dir}")
    print(f"[select] total_files={len(all_files)}, selected_packets={len(packet_paths)}")
    print(f"[select] first={packet_paths[0].name}, last={packet_paths[-1].name}")
    print(f"[format] max_sh_degree={args.max_sh_degree}, coeff_count={coeff_count}, properties={len(property_names)}")
    print(f"[filter] opacity_threshold={args.opacity_threshold}, topk={args.topk}")
    print(f"[convert] opacity_output={args.opacity_output}, scale_output={args.scale_output}, rotation_order={args.rotation_order}, normalize_rotations={not args.no_normalize_rotations}")

    topk_cutoff = compute_global_opacity_cutoff(packet_paths, float(args.opacity_threshold), int(args.topk))
    if topk_cutoff is not None:
        print(f"[filter] global topk opacity cutoff={topk_cutoff:.8f}")

    vertex_count, per_packet_counts = count_selected(packet_paths, float(args.opacity_threshold), topk_cutoff)
    if vertex_count <= 0:
        raise RuntimeError("No Gaussians left after filtering.")
    print(f"[count] selected_gaussians={vertex_count}")
    print(f"[output] {output_path}")

    write_ply_binary(
        output_path=output_path,
        packet_paths=packet_paths,
        vertex_count=vertex_count,
        coeff_count=coeff_count,
        opacity_threshold=float(args.opacity_threshold),
        topk_cutoff=topk_cutoff,
        opacity_output=args.opacity_output,
        scale_output=args.scale_output,
        rotation_order=args.rotation_order,
        normalize_rotations=not args.no_normalize_rotations,
        row_chunk_size=int(args.row_chunk_size),
    )

    summary_path = Path(args.summary).expanduser().resolve() if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")
    summary = {
        "packet_dir": str(packet_dir),
        "output": str(output_path),
        "total_packet_files": len(all_files),
        "selected_packet_indices": selected_idx,
        "selected_packet_files": [p.name for p in packet_paths],
        "selected_gaussians": int(vertex_count),
        "per_packet_selected_counts": per_packet_counts,
        "opacity_threshold": float(args.opacity_threshold),
        "topk": int(args.topk),
        "topk_cutoff": topk_cutoff,
        "max_sh_degree": int(args.max_sh_degree),
        "coeff_count": int(coeff_count),
        "property_names": property_names,
        "opacity_output": args.opacity_output,
        "scale_output": args.scale_output,
        "rotation_order": args.rotation_order,
        "normalize_rotations": not args.no_normalize_rotations,
        "notes": [
            "PLY is binary_little_endian with standard 3DGS fields.",
            "Default opacity_output=auto converts alpha [0,1] to logit for standard 3DGS PLY viewers.",
            "Default scale_output=auto converts positive scales to log-scale for standard 3DGS PLY viewers.",
            "If online viewer shows over/under-sized splats, retry with --scale_output raw or --scale_output log.",
            "If opacity looks wrong, retry with --opacity_output raw/alpha/logit.",
            "If orientation looks wrong, test --rotation_order xyzw_to_wxyz.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote PLY: {output_path}")
    print(f"[done] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
