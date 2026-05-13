import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
import torchvision.transforms as tf
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
    from src.misc.image_io import prep_image, save_image, save_video
    from src.model.types import Gaussians

from gsplat.rendering import rasterization


REQUIRED_PACKET_FIELDS = {
    "scene",
    "means",
    "covariances",
    "harmonics",
    "opacities",
    "scales",
    "rotations",
    "rotations_unnorm",
    "target_extrinsics",
    "target_intrinsics",
    "target_near",
    "target_far",
    "target_image",
    "image_shape",
    "background_color",
    "context_index",
    "target_index",
    "target_camera_id",
}


@dataclass
class LoadedPacket:
    path: Path
    data: dict[str, Any]
    context_sort_index: int
    scene_key: str
    base_scene: str


@dataclass
class ProbeView:
    label: str
    extrinsics: torch.Tensor         # [4, 4]
    intrinsics: torch.Tensor         # [3, 3]
    near: torch.Tensor               # scalar tensor
    far: torch.Tensor                # scalar tensor
    image_shape: tuple[int, int]     # (H, W)
    background_color: torch.Tensor   # [3]
    gt_image: Optional[torch.Tensor] # [3, H, W] or None
    meta: dict[str, Any]


@dataclass
class RenderJob:
    probe: ProbeView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline additive prefix fusion for Gaussian packets with flexible probe rendering.\n"
            "Supports packet-based probes, raw TartanAir dataset probes, and custom pose probes.\n"
            "GT is optional: if GT exists, metrics are computed; otherwise only rendered images/videos are saved."
        )
    )
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--max_packets",
        type=int,
        default=None,
        help="Use the first N packets after sorting. Ignored if --packet_ranges is set.",
    )

    parser.add_argument(
        "--packet_ranges",
        type=str,
        default=None,
        help=(
            "Comma-separated packet index ranges after filename sorting, e.g. "
            "'0-20,300-500' or '0-20,35,300-500'. Inclusive ranges."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Also compute LPIPS when GT exists. Slower than PSNR/SSIM.",
    )

    # Probe source selection.
    parser.add_argument(
        "--probe_mode",
        choices=[
            "packet_last_only",
            "packet_all_prefix_targets",
            "packet_fixed_target",
            "tartanair_frame",
            "custom_pose",
            "custom_pose_sequence",
        ],
        default="packet_fixed_target",
    )

    # Packet-based fixed probe.
    parser.add_argument(
        "--fixed_target_index",
        type=int,
        default=None,
        help="Global target frame index to probe (searches across packet target_index fields).",
    )
    parser.add_argument(
        "--fixed_target_camera_id",
        type=int,
        default=0,
        help="Camera id for fixed target probe. 0=left, 1=right.",
    )

    # Dataset-backed TartanAir probe.
    parser.add_argument(
        "--dataset_cfg",
        type=Path,
        default=None,
        help="Path to a tartanair dataset yaml, e.g. config/dataset/tartanair_p000_eval.yaml.",
    )
    parser.add_argument(
        "--sequence_index",
        type=int,
        default=0,
        help="Which sequence entry to use from dataset_cfg. Default 0.",
    )
    parser.add_argument(
        "--probe_frame_index",
        type=int,
        default=None,
        help="Frame index in the raw dataset to use as probe when probe_mode=tartanair_frame.",
    )
    parser.add_argument(
        "--probe_camera",
        choices=["left", "right"],
        default="left",
        help="Probe camera for tartanair_frame.",
    )

    # Custom pose probe.
    parser.add_argument(
        "--probe_pose_json",
        type=Path,
        default=None,
        help="Path to a JSON file containing one 4x4 Twc pose for probe_mode=custom_pose.",
    )
    parser.add_argument(
        "--probe_poses_json",
        type=Path,
        default=None,
        help="Path to a JSON file containing a list of 4x4 Twc poses for probe_mode=custom_pose_sequence.",
    )
    parser.add_argument(
        "--gt_image_path",
        type=Path,
        default=None,
        help="Optional GT image path for custom_pose / custom_pose_sequence.",
    )
    parser.add_argument(
        "--custom_background_color",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB background color for custom probes in [0,1]. Defaults to packet/dataset cfg background.",
    )

    # Video / export controls.
    parser.add_argument(
        "--save_progress_video",
        action="store_true",
        help="Save a video showing how renderings change as packet prefix grows.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=6,
        help="FPS for saved progress video.",
    )

    parser.add_argument(
        "--probe_image_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override probe render image shape as H W, e.g. 320 640.",
    )

    parser.add_argument(
        "--probe_intrinsics_norm",
        type=float,
        nargs=4,
        default=None,
        metavar=("FX", "FY", "CX", "CY"),
        help=(
            "Override normalized probe intrinsics as fx fy cx cy. "
            "Example for 640x320 wide view: 0.25 0.5 0.5 0.5."
        ),
    )

    return parser.parse_args()

def canonicalize_scene_name(scene: str) -> str:
    parts = scene.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scene


def resolve_device(device_str: str) -> torch.device:
    try:
        device = torch.device(device_str)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device '{device_str}': {exc}") from exc

    if device.type != "cuda":
        raise RuntimeError(
            "This script uses gsplat rasterization, which is expected to run on CUDA. "
            f"Requested device='{device_str}'. Please use a CUDA device such as 'cuda:0'."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requested a CUDA device, but torch.cuda.is_available() is False."
        )

    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device '{device_str}', but only {torch.cuda.device_count()} CUDA "
            "device(s) are available."
        )

    return device


def as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def packet_id(packet: LoadedPacket) -> str:
    return packet.path.name


def validate_packet(packet: dict[str, Any], path: Path) -> None:
    missing = sorted(REQUIRED_PACKET_FIELDS - set(packet.keys()))
    if missing:
        raise KeyError(f"Packet '{path}' is missing required fields: {missing}")

    if not isinstance(packet["scene"], str):
        raise TypeError(f"Packet '{path}' field 'scene' must be a string.")

    tensor_fields = (
        "means",
        "harmonics",
        "opacities",
        "scales",
        "rotations",
        "rotations_unnorm",
    )
    for field in tensor_fields:
        if not isinstance(packet[field], torch.Tensor):
            raise TypeError(f"Packet '{path}' field '{field}' must be a torch.Tensor.")

    means = packet["means"]
    covariances = packet["covariances"]
    harmonics = packet["harmonics"]
    opacities = packet["opacities"]
    scales = packet["scales"]
    rotations = packet["rotations"]
    rotations_unnorm = packet["rotations_unnorm"]

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Packet '{path}' has invalid means shape {tuple(means.shape)}.")
    if covariances is not None:
        if not isinstance(covariances, torch.Tensor):
            raise TypeError(f"Packet '{path}' field 'covariances' must be a tensor or None.")
        if covariances.ndim != 3 or covariances.shape[-2:] != (3, 3):
            raise ValueError(
                f"Packet '{path}' has invalid covariances shape {tuple(covariances.shape)}."
            )
    if harmonics.ndim != 3 or harmonics.shape[1] != 3:
        raise ValueError(
            f"Packet '{path}' has invalid harmonics shape {tuple(harmonics.shape)}."
        )
    if opacities.ndim != 1:
        raise ValueError(
            f"Packet '{path}' has invalid opacities shape {tuple(opacities.shape)}."
        )
    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError(f"Packet '{path}' has invalid scales shape {tuple(scales.shape)}.")
    if rotations.ndim != 2 or rotations.shape[-1] != 4:
        raise ValueError(f"Packet '{path}' has invalid rotations shape {tuple(rotations.shape)}.")
    if rotations_unnorm.ndim != 2 or rotations_unnorm.shape[-1] != 4:
        raise ValueError(
            f"Packet '{path}' has invalid rotations_unnorm shape {tuple(rotations_unnorm.shape)}."
        )

    num_gaussians = means.shape[0]
    for field_name, field_value in (
        ("harmonics", harmonics),
        ("opacities", opacities),
        ("scales", scales),
        ("rotations", rotations),
        ("rotations_unnorm", rotations_unnorm),
    ):
        if field_value.shape[0] != num_gaussians:
            raise ValueError(
                f"Packet '{path}' gaussian field lengths do not match: "
                f"means has {num_gaussians}, {field_name} has {field_value.shape[0]}."
            )
    if covariances is not None and covariances.shape[0] != num_gaussians:
        raise ValueError(
            f"Packet '{path}' gaussian field lengths do not match: "
            f"means has {num_gaussians}, covariances has {covariances.shape[0]}."
        )

    context_index = packet["context_index"]
    target_index = packet["target_index"]
    target_camera_id = packet["target_camera_id"]
    if not isinstance(context_index, torch.Tensor) or context_index.numel() == 0:
        raise ValueError(
            f"Packet '{path}' has invalid context_index; expected a non-empty tensor."
        )
    if not isinstance(target_index, torch.Tensor):
        raise ValueError(
            f"Packet '{path}' has invalid target_index; expected a tensor."
        )
    if not isinstance(target_camera_id, torch.Tensor):
        raise ValueError(
            f"Packet '{path}' has invalid target_camera_id; expected a tensor."
        )

    target_extrinsics = packet["target_extrinsics"]
    target_intrinsics = packet["target_intrinsics"]
    target_near = packet["target_near"]
    target_far = packet["target_far"]
    target_image = packet["target_image"]

    if target_extrinsics.ndim != 3 or target_extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            f"Packet '{path}' has invalid target_extrinsics shape {tuple(target_extrinsics.shape)}."
        )
    if target_intrinsics.ndim != 3 or target_intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"Packet '{path}' has invalid target_intrinsics shape {tuple(target_intrinsics.shape)}."
        )
    if target_image.ndim != 4 or target_image.shape[1] != 3:
        raise ValueError(
            f"Packet '{path}' has invalid target_image shape {tuple(target_image.shape)}."
        )
    if target_near.ndim != 1 or target_far.ndim != 1:
        raise ValueError(
            f"Packet '{path}' target_near/target_far must be 1D tensors; got "
            f"{tuple(target_near.shape)} and {tuple(target_far.shape)}."
        )

    num_target_views = target_extrinsics.shape[0]
    if target_intrinsics.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_image.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_near.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_far.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_index.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_camera_id.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")

    image_shape = packet["image_shape"]
    if len(image_shape) != 2:
        raise ValueError(
            f"Packet '{path}' has invalid image_shape={image_shape}; expected (H, W)."
        )

    background_color = packet["background_color"]
    if not isinstance(background_color, torch.Tensor):
        raise TypeError(
            f"Packet '{path}' field 'background_color' must be a torch.Tensor."
        )
    if background_color.numel() != 3:
        raise ValueError(
            f"Packet '{path}' has invalid background_color shape {tuple(background_color.shape)}."
        )

def parse_packet_ranges(spec: str, num_packets: int) -> list[int]:
    """
    Parse a range spec over sorted packet indices.
    Example:
      "0-20,300-500" -> [0,1,...,20,300,...,500]
      "0-20,35,100-120" also works.

    Ranges are inclusive.
    Duplicate indices are removed while preserving order.
    """
    if spec is None or spec.strip() == "":
        raise ValueError("Empty --packet_ranges spec.")

    selected: list[int] = []
    seen: set[int] = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError(f"Invalid packet range item: '{part}'")
            start = int(pieces[0])
            end = int(pieces[1])
            if start > end:
                raise ValueError(f"Invalid packet range '{part}': start > end")
            indices = range(start, end + 1)
        else:
            idx = int(part)
            indices = [idx]

        for idx in indices:
            if idx < 0 or idx >= num_packets:
                raise IndexError(
                    f"Packet index {idx} out of range. "
                    f"Available sorted packet indices: 0..{num_packets - 1}"
                )
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

    if not selected:
        raise ValueError(f"No packet indices selected from --packet_ranges='{spec}'")

    return selected


def select_packet_paths(
    packet_dir: Path,
    max_packets: int | None,
    packet_ranges: str | None,
) -> list[Path]:
    if not packet_dir.exists():
        raise FileNotFoundError(f"packet_dir does not exist: {packet_dir}")
    if not packet_dir.is_dir():
        raise NotADirectoryError(f"packet_dir is not a directory: {packet_dir}")

    packet_paths_all = sorted(packet_dir.glob("*.pt"))
    if not packet_paths_all:
        raise FileNotFoundError(f"No .pt packets found under: {packet_dir}")

    if packet_ranges is not None:
        selected_indices = parse_packet_ranges(packet_ranges, len(packet_paths_all))
        packet_paths = [packet_paths_all[i] for i in selected_indices]
        print(
            f"Selected {len(packet_paths)} packet(s) by ranges '{packet_ranges}' "
            f"from {len(packet_paths_all)} available packet(s).",
            flush=True,
        )
        print(
            "Selected sorted packet indices: "
            + ",".join(str(i) for i in selected_indices[:30])
            + ("..." if len(selected_indices) > 30 else ""),
            flush=True,
        )
        return packet_paths

    if max_packets is None:
        raise ValueError("Either --max_packets or --packet_ranges must be specified.")

    if max_packets <= 0:
        raise ValueError(f"--max_packets must be > 0, got {max_packets}")

    packet_paths = packet_paths_all[:max_packets]
    print(
        f"Selected first {len(packet_paths)} packet(s) "
        f"from {len(packet_paths_all)} available packet(s).",
        flush=True,
    )
    return packet_paths

def load_packets(
    packet_dir: Path,
    max_packets: int | None = None,
    packet_ranges: str | None = None,
) -> list[LoadedPacket]:
    packet_paths = select_packet_paths(
        packet_dir=packet_dir,
        max_packets=max_packets,
        packet_ranges=packet_ranges,
    )

    print(f"Loading {len(packet_paths)} selected packet(s) from {packet_dir}", flush=True)

    loaded_packets: list[LoadedPacket] = []
    for path in packet_paths:
        packet = torch.load(path, map_location="cpu")
        if not isinstance(packet, dict):
            raise TypeError(f"Packet '{path}' must contain a dict, got {type(packet)}.")
        validate_packet(packet, path)
        scene_key = packet["scene"]
        loaded_packets.append(
            LoadedPacket(
                path=path,
                data=packet,
                context_sort_index=int(packet["context_index"][0].item()),
                scene_key=scene_key,
                base_scene=canonicalize_scene_name(scene_key),
            )
        )

    base_scene_names = sorted({packet.base_scene for packet in loaded_packets})
    if len(base_scene_names) != 1:
        raise RuntimeError(
            "This baseline only supports fusing packets from one base sequence at a time. "
            f"Found multiple base sequences in packet_dir: {base_scene_names}"
        )

    loaded_packets.sort(key=lambda item: item.context_sort_index)
    return loaded_packets


def validate_packet_collection(packets: list[LoadedPacket]) -> tuple[int, int]:
    image_shapes = {
        tuple(int(x) for x in packet.data["image_shape"]) for packet in packets
    }
    if len(image_shapes) != 1:
        raise ValueError(
            "All packets must share the same image_shape, got "
            f"{sorted(image_shapes)}."
        )

    harmonic_dims = {int(packet.data["harmonics"].shape[-1]) for packet in packets}
    if len(harmonic_dims) != 1:
        raise ValueError(
            "All packets must share the same SH dimensionality, got "
            f"{sorted(harmonic_dims)}."
        )

    scale_dims = {tuple(packet.data["scales"].shape[1:]) for packet in packets}
    rotation_dims = {tuple(packet.data["rotations_unnorm"].shape[1:]) for packet in packets}
    if scale_dims != {(3,)}:
        raise ValueError(f"All packets must have scales with trailing shape (3,), got {sorted(scale_dims)}.")
    if rotation_dims != {(4,)}:
        raise ValueError(
            f"All packets must have rotations_unnorm with trailing shape (4,), got {sorted(rotation_dims)}."
        )

    return next(iter(image_shapes))


def _cat_optional_tensor(
    packets: list[LoadedPacket],
    key: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    values = [packet.data.get(key, None) for packet in packets]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            f"Some packets have '{key}' but others have None. This mixed state is unsupported."
        )
    return torch.cat(values, dim=0).to(device)


def concat_fused_gaussians(packets: list[LoadedPacket], device: torch.device) -> Gaussians:
    means = torch.cat([packet.data["means"] for packet in packets], dim=0).to(device)
    covariances = _cat_optional_tensor(packets, "covariances", device)
    harmonics = torch.cat([packet.data["harmonics"] for packet in packets], dim=0).to(device)
    opacities = torch.cat([packet.data["opacities"] for packet in packets], dim=0).to(device)
    scales = torch.cat([packet.data["scales"] for packet in packets], dim=0).to(device)
    rotations = torch.cat([packet.data["rotations"] for packet in packets], dim=0).to(device)
    rotations_unnorm = torch.cat([packet.data["rotations_unnorm"] for packet in packets], dim=0).to(device)

    return Gaussians(
        means=means.unsqueeze(0).contiguous(),
        covariances=None if covariances is None else covariances.unsqueeze(0).contiguous(),
        harmonics=harmonics.unsqueeze(0).contiguous(),
        opacities=opacities.unsqueeze(0).contiguous(),
        scales=scales.unsqueeze(0).contiguous(),
        rotations=rotations.unsqueeze(0).contiguous(),
        rotations_unnorm=rotations_unnorm.unsqueeze(0).contiguous(),
    )


# -----------------------------------------------------------------------------
# Generic probe helpers
# -----------------------------------------------------------------------------
def save_png_tensor(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 3:
        raise ValueError(
            f"Expected image tensor with shape [C, H, W], got {tuple(image.shape)}"
        )
    save_image(image.detach().cpu().float(), path)


def save_progress_video_safe(
    frames: list[torch.Tensor],
    path: Path,
    fps: int = 10,
) -> tuple[bool, str]:
    frames_dir = path.parent / f"{path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        save_image(frame.detach().cpu().float(), frames_dir / f"frame_{i:04d}.png")

    return False, f"Saved PNG frame sequence to {frames_dir}"


def plot_curve(x_values: list[int], y_values: list[float], title: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Number of fused packets")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@torch.no_grad()
def render_fused_views(
    fused: Gaussians,
    probes: list[ProbeView],
    device: torch.device,
) -> torch.Tensor:
    """Render fused Resplat/GSplat Gaussians for one or more probe views.

    This mirrors src/model/decoder/gsplat_decoder_splatting_cuda.py:
    - extrinsics are Twc and are inverted into view matrices.
    - intrinsics in packets/config are normalized and are scaled by image width/height.
    - harmonics are converted from [B, G, 3, D] to [B, G, D, 3].
    """
    image_shapes = {probe.image_shape for probe in probes}
    if len(image_shapes) != 1:
        raise ValueError(f"All probes must share the same image_shape, got {sorted(image_shapes)}.")
    image_shape = next(iter(image_shapes))
    height, width = image_shape

    extrinsics = torch.stack([probe.extrinsics for probe in probes], dim=0).to(device=device, dtype=torch.float32)
    intrinsics = torch.stack([probe.intrinsics for probe in probes], dim=0).to(device=device, dtype=torch.float32)
    near = torch.stack([probe.near.reshape(()) for probe in probes], dim=0).to(device=device, dtype=torch.float32)
    far = torch.stack([probe.far.reshape(()) for probe in probes], dim=0).to(device=device, dtype=torch.float32)

    # Resplat's GSplat decoder expects batch dimension first: [B, V, ...].
    extrinsics_b = extrinsics.unsqueeze(0).contiguous()  # [1, V, 4, 4]
    intrinsics_b = intrinsics.unsqueeze(0).clone().contiguous()  # [1, V, 3, 3]
    intrinsics_b[:, :, 0] *= width
    intrinsics_b[:, :, 1] *= height
    viewmats = extrinsics_b.inverse().contiguous()

    colors = fused.harmonics.permute(0, 1, 3, 2).contiguous()  # [1, G, D_sh, 3]
    sh_degree = int(math.sqrt(colors.shape[-2])) - 1
    if (sh_degree + 1) ** 2 != colors.shape[-2]:
        raise ValueError(
            f"Invalid SH dimension {colors.shape[-2]}; expected a square number like 1, 4, 9, 16."
        )

    if fused.scales is None:
        raise ValueError("Resplat/GSplat rendering requires fused.scales, but it is None.")
    if fused.rotations_unnorm is None:
        raise ValueError("Resplat/GSplat rendering requires fused.rotations_unnorm, but it is None.")

    # The Resplat decoder uses scalar near/far from the first view. For mixed probes,
    # using min(near) and max(far) is safer because it avoids accidentally clipping views.
    near_plane = float(near.min().item())
    far_plane = float(far.max().item())

    render_colors, render_alphas, meta = rasterization(
        means=fused.means.contiguous(),
        quats=fused.rotations_unnorm.contiguous(),
        scales=fused.scales.contiguous(),
        opacities=fused.opacities.contiguous(),
        colors=colors,
        sh_degree=sh_degree,
        viewmats=viewmats,
        Ks=intrinsics_b,
        width=int(width),
        height=int(height),
        near_plane=near_plane,
        far_plane=far_plane,
        eps2d=0.1,
        rasterize_mode="antialiased",
        packed=True,
        absgrad=False,
        sparse_grad=False,
        render_mode="RGB+ED",
        covars=None if fused.covariances is None else fused.covariances.contiguous(),
    )

    # render_colors: [B, V, H, W, 4] for RGB+ED. Return [V, 3, H, W].
    return render_colors[0, ..., :3].permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def compute_metrics_for_available_gt(
    rendered: torch.Tensor,
    probes: list[ProbeView],
    compute_lpips_flag: bool,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    gt_indices = [i for i, probe in enumerate(probes) if probe.gt_image is not None]
    if not gt_indices:
        return {}, gt_indices

    gt_images = torch.stack([probes[i].gt_image for i in gt_indices], dim=0).to(device=rendered.device, dtype=torch.float32)
    pred_images = rendered[gt_indices]

    metrics: dict[str, torch.Tensor] = {
        "psnr": compute_psnr(gt_images, pred_images),
        "ssim": compute_ssim(gt_images, pred_images),
    }
    if compute_lpips_flag:
        metrics["lpips"] = compute_lpips(gt_images, pred_images)
    return metrics, gt_indices


def metric_mean_as_float(metric_values: torch.Tensor) -> float:
    return float(metric_values.detach().cpu().mean().item())


# -----------------------------------------------------------------------------
# Packet-based probe modes
# -----------------------------------------------------------------------------
def build_packet_last_only_probes(prefix_packets: list[LoadedPacket]) -> list[ProbeView]:
    packet = prefix_packets[-1]
    probes: list[ProbeView] = []
    for i in range(packet.data["target_image"].shape[0]):
        target_index = int(packet.data["target_index"][i].item())
        camera_id = int(packet.data["target_camera_id"][i].item())
        label = f"packet_last_target_{target_index:04d}_{'left' if camera_id == 0 else 'right'}"
        probes.append(
            ProbeView(
                label=label,
                extrinsics=packet.data["target_extrinsics"][i].clone(),
                intrinsics=packet.data["target_intrinsics"][i].clone(),
                near=packet.data["target_near"][i].reshape(1).clone(),
                far=packet.data["target_far"][i].reshape(1).clone(),
                image_shape=tuple(int(x) for x in packet.data["image_shape"]),
                background_color=packet.data["background_color"].reshape(-1).clone(),
                gt_image=packet.data["target_image"][i].clone(),
                meta={
                    "source": "packet_last_only",
                    "packet_file": packet.path.name,
                    "target_index": target_index,
                    "camera_id": camera_id,
                },
            )
        )
    return probes


def build_packet_all_prefix_target_probes(prefix_packets: list[LoadedPacket]) -> list[ProbeView]:
    probes: list[ProbeView] = []
    for packet in prefix_packets:
        for i in range(packet.data["target_image"].shape[0]):
            target_index = int(packet.data["target_index"][i].item())
            camera_id = int(packet.data["target_camera_id"][i].item())
            label = f"{packet.path.stem}_target_{target_index:06d}_{'left' if camera_id == 0 else 'right'}"
            probes.append(
                ProbeView(
                    label=label,
                    extrinsics=packet.data["target_extrinsics"][i].clone(),
                    intrinsics=packet.data["target_intrinsics"][i].clone(),
                    near=packet.data["target_near"][i].reshape(1).clone(),
                    far=packet.data["target_far"][i].reshape(1).clone(),
                    image_shape=tuple(int(x) for x in packet.data["image_shape"]),
                    background_color=packet.data["background_color"].reshape(-1).clone(),
                    gt_image=packet.data["target_image"][i].clone(),
                    meta={
                        "source": "packet_all_prefix_targets",
                        "packet_file": packet.path.name,
                        "target_index": target_index,
                        "camera_id": camera_id,
                    },
                )
            )
    return probes


def resolve_packet_fixed_target_probe(
    packets: list[LoadedPacket],
    fixed_target_index: int,
    fixed_target_camera_id: int,
) -> ProbeView:
    matches: list[ProbeView] = []
    for packet in packets:
        for i in range(packet.data["target_image"].shape[0]):
            target_index = int(packet.data["target_index"][i].item())
            camera_id = int(packet.data["target_camera_id"][i].item())
            if target_index == fixed_target_index and camera_id == fixed_target_camera_id:
                matches.append(
                    ProbeView(
                        label=f"fixed_target_{target_index:06d}_{'left' if camera_id == 0 else 'right'}",
                        extrinsics=packet.data["target_extrinsics"][i].clone(),
                        intrinsics=packet.data["target_intrinsics"][i].clone(),
                        near=packet.data["target_near"][i].reshape(1).clone(),
                        far=packet.data["target_far"][i].reshape(1).clone(),
                        image_shape=tuple(int(x) for x in packet.data["image_shape"]),
                        background_color=packet.data["background_color"].reshape(-1).clone(),
                        gt_image=packet.data["target_image"][i].clone(),
                        meta={
                            "source": "packet_fixed_target",
                            "packet_file": packet.path.name,
                            "target_index": target_index,
                            "camera_id": camera_id,
                            "packet_scene_key": packet.scene_key,
                        },
                    )
                )

    if not matches:
        raise ValueError(
            f"No packet target view found with target_index={fixed_target_index}, "
            f"camera_id={fixed_target_camera_id} in packet_dir={packets[0].path.parent}."
        )
    if len(matches) > 1:
        print(
            f"Warning: found {len(matches)} matches for fixed target index {fixed_target_index} "
            f"and camera_id={fixed_target_camera_id}. Using the first match from "
            f"{matches[0].meta['packet_file']}."
        )
    return matches[0]


# -----------------------------------------------------------------------------
# TartanAir raw-dataset-backed probe mode
# -----------------------------------------------------------------------------
def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> torch.Tensor:
    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    q = q / q.norm()
    x, y, z, w = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.tensor(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=torch.float32,
    )


def tartanair_build_twc_from_pose(
    tx: float,
    ty: float,
    tz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> torch.Tensor:
    # Keep exactly the same camera-axis conversion used in src/dataset/dataset_tartanair.py
    twc_pose = torch.eye(4, dtype=torch.float32)
    twc_pose[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
    twc_pose[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)

    t_tartan_cam_from_cv_cam = torch.eye(4, dtype=torch.float32)
    t_tartan_cam_from_cv_cam[:3, :3] = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # cv z (forward) -> tartan x
            [1.0, 0.0, 0.0],  # cv x (right)   -> tartan y
            [0.0, 1.0, 0.0],  # cv y (down)    -> tartan z
        ],
        dtype=torch.float32,
    )
    return twc_pose @ t_tartan_cam_from_cv_cam


def build_pose_from_values(values: list[float], pose_format: str, pose_matrix_type: str) -> torch.Tensor:
    if pose_format == "tx_ty_tz_qx_qy_qz_qw":
        tx, ty, tz, qx, qy, qz, qw = values
    elif pose_format == "tx_ty_tz_qw_qx_qy_qz":
        tx, ty, tz, qw, qx, qy, qz = values
    else:
        raise ValueError(f"Unsupported pose_format: {pose_format}")

    twc = tartanair_build_twc_from_pose(tx, ty, tz, qx, qy, qz, qw)
    if pose_matrix_type == "Twc":
        return twc
    if pose_matrix_type == "Tcw":
        return torch.linalg.inv(twc)
    raise ValueError(f"Unsupported pose_matrix_type: {pose_matrix_type}")


def parse_pose_matrix_json(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "Twc" in data:
            data = data["Twc"]
        elif "extrinsics" in data:
            data = data["extrinsics"]
        elif "pose" in data:
            data = data["pose"]
        else:
            raise ValueError(
                f"Unsupported pose json dict keys in {path}. Expected one of Twc/extrinsics/pose."
            )
    tensor = torch.tensor(data, dtype=torch.float32)
    if tensor.numel() == 16:
        tensor = tensor.view(4, 4)
    if tensor.shape != (4, 4):
        raise ValueError(f"Pose file {path} must contain a 4x4 matrix, got shape {tuple(tensor.shape)}.")
    return tensor


def parse_pose_sequence_json(path: Path) -> list[torch.Tensor]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "poses" in data:
        data = data["poses"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of poses in {path}.")
    poses = []
    for i, item in enumerate(data):
        tensor = torch.tensor(item, dtype=torch.float32)
        if tensor.numel() == 16:
            tensor = tensor.view(4, 4)
        if tensor.shape != (4, 4):
            raise ValueError(f"Pose #{i} in {path} must be a 4x4 matrix, got {tuple(tensor.shape)}.")
        poses.append(tensor)
    return poses


def build_k(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    k = torch.eye(3, dtype=torch.float32)
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 2] = cx
    k[1, 2] = cy
    return k


def sorted_image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(folder.glob(pattern))
    def sort_key(path: Path):
        stem = path.stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)
    return sorted(paths, key=sort_key)


def load_pose_file(path: Path, pose_format: str, pose_matrix_type: str) -> list[torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Pose file not found: {path}")
    poses = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 7:
                continue
            values = list(map(float, parts[:7]))
            poses.append(build_pose_from_values(values, pose_format, pose_matrix_type))
    if not poses:
        raise RuntimeError(f"No valid poses found in file: {path}")
    return poses


def process_image_and_k(
    image_path: Path,
    k: torch.Tensor,
    image_shape: tuple[int, int],
    normalize_intrinsics: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    to_tensor = tf.ToTensor()
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        k = k.clone()
        target_h, target_w = image_shape
        scale = max(target_w / orig_w, target_h / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))
        im = im.resize((resized_w, resized_h), Image.BILINEAR)
        k[0, 0] *= scale
        k[1, 1] *= scale
        k[0, 2] *= scale
        k[1, 2] *= scale
        left = int(round((resized_w - target_w) / 2.0))
        top = int(round((resized_h - target_h) / 2.0))
        right = left + target_w
        bottom = top + target_h
        im = im.crop((left, top, right, bottom))
        k[0, 2] -= left
        k[1, 2] -= top
        if normalize_intrinsics:
            k[0, 0] /= target_w
            k[1, 1] /= target_h
            k[0, 2] /= target_w
            k[1, 2] /= target_h
        image_tensor = to_tensor(im)
    return image_tensor, k


def load_tartanair_dataset_cfg(path: Path, sequence_index: int) -> dict[str, Any]:
    if path is None:
        raise ValueError("--dataset_cfg is required for tartanair_frame/custom_pose/custom_pose_sequence.")
    if not path.exists():
        raise FileNotFoundError(f"dataset_cfg does not exist: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("name") != "tartanair":
        raise ValueError(f"dataset_cfg must be a tartanair config, got name={cfg.get('name')}")
    sequences = cfg.get("sequences", [])
    if not sequences:
        raise ValueError(f"No sequences found in dataset_cfg={path}")
    if sequence_index < 0 or sequence_index >= len(sequences):
        raise IndexError(f"sequence_index={sequence_index} out of range for {len(sequences)} sequences")
    seq = sequences[sequence_index]
    out = dict(cfg)
    out["sequence"] = seq
    return out


def tartanair_frame_probe_from_cfg(
    dataset_cfg: dict[str, Any],
    probe_frame_index: int,
    probe_camera: str,
    background_color_override: Optional[list[float]] = None,
) -> ProbeView:
    seq = dataset_cfg["sequence"]
    scene_root = Path(seq["root"])
    left_dir = scene_root / dataset_cfg.get("left_camera_dirname", "image_lcam_front")
    right_dir = scene_root / dataset_cfg.get("right_camera_dirname", "image_rcam_front")
    left_pose_file = scene_root / dataset_cfg.get("left_pose_filename", "pose_lcam_front.txt")
    right_pose_file = scene_root / dataset_cfg.get("right_pose_filename", "pose_rcam_front.txt")

    left_imgs = sorted_image_paths(left_dir)
    right_imgs = sorted_image_paths(right_dir)
    left_poses = load_pose_file(
        left_pose_file,
        dataset_cfg.get("pose_format", "tx_ty_tz_qx_qy_qz_qw"),
        dataset_cfg.get("pose_matrix_type", "Twc"),
    )
    right_poses = load_pose_file(
        right_pose_file,
        dataset_cfg.get("pose_format", "tx_ty_tz_qx_qy_qz_qw"),
        dataset_cfg.get("pose_matrix_type", "Twc"),
    )

    n = min(len(left_imgs), len(right_imgs), len(left_poses), len(right_poses))
    if probe_frame_index < 0 or probe_frame_index >= n:
        raise IndexError(
            f"probe_frame_index={probe_frame_index} out of range for scene={seq['scene']} "
            f"with {n} aligned stereo frames."
        )

    fx = float(seq["fx"])
    fy = float(seq["fy"])
    cx = float(seq["cx"])
    cy = float(seq["cy"])
    k_base = build_k(fx, fy, cx, cy)
    image_shape = tuple(int(x) for x in dataset_cfg["image_shape"])
    normalize_intrinsics = bool(dataset_cfg.get("normalize_intrinsics", True))
    near = torch.tensor([float(dataset_cfg.get("near", 0.1))], dtype=torch.float32)
    far = torch.tensor([float(dataset_cfg.get("far", 50.0))], dtype=torch.float32)

    if probe_camera == "left":
        image_path = left_imgs[probe_frame_index]
        extrinsics = left_poses[probe_frame_index]
        camera_id = 0
    elif probe_camera == "right":
        image_path = right_imgs[probe_frame_index]
        extrinsics = right_poses[probe_frame_index]
        camera_id = 1
    else:
        raise ValueError(f"Unsupported probe_camera={probe_camera}")

    gt_image, intrinsics = process_image_and_k(image_path, k_base, image_shape, normalize_intrinsics)

    background_color = (
        torch.tensor(background_color_override, dtype=torch.float32)
        if background_color_override is not None
        else torch.tensor(dataset_cfg.get("background_color", [0.0, 0.0, 0.0]), dtype=torch.float32)
    )

    return ProbeView(
        label=f"tartanair_frame_{probe_frame_index:06d}_{probe_camera}",
        extrinsics=extrinsics.clone(),
        intrinsics=intrinsics.clone(),
        near=near.clone(),
        far=far.clone(),
        image_shape=image_shape,
        background_color=background_color.reshape(-1).clone(),
        gt_image=gt_image.clone(),
        meta={
            "source": "tartanair_frame",
            "scene": seq["scene"],
            "scene_root": str(scene_root),
            "frame_index": probe_frame_index,
            "camera": probe_camera,
            "camera_id": camera_id,
            "gt_image_path": str(image_path),
        },
    )


def custom_pose_probe_from_cfg(
    dataset_cfg: dict[str, Any],
    pose: torch.Tensor,
    label: str,
    gt_image_path: Optional[Path],
    background_color_override: Optional[list[float]] = None,
    image_shape_override: Optional[list[int]] = None,
    intrinsics_norm_override: Optional[list[float]] = None,
) -> ProbeView:
    seq = dataset_cfg["sequence"]
    fx = float(seq["fx"])
    fy = float(seq["fy"])
    cx = float(seq["cx"])
    cy = float(seq["cy"])
    image_shape = (
        tuple(int(x) for x in image_shape_override)
        if image_shape_override is not None
        else tuple(int(x) for x in dataset_cfg["image_shape"])
    )

    normalize_intrinsics = bool(dataset_cfg.get("normalize_intrinsics", True))
    near = torch.tensor([float(dataset_cfg.get("near", 0.1))], dtype=torch.float32)
    far = torch.tensor([float(dataset_cfg.get("far", 50.0))], dtype=torch.float32)

    if intrinsics_norm_override is not None:
        fx_n, fy_n, cx_n, cy_n = [float(x) for x in intrinsics_norm_override]
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[0, 0] = fx_n
        intrinsics[1, 1] = fy_n
        intrinsics[0, 2] = cx_n
        intrinsics[1, 2] = cy_n
    else:
        dummy_width = int(image_shape[1])
        dummy_height = int(image_shape[0])
        intrinsics = build_k(fx, fy, cx, cy)
        if normalize_intrinsics:
            intrinsics[0, 0] /= dummy_width
            intrinsics[1, 1] /= dummy_height
            intrinsics[0, 2] /= dummy_width
            intrinsics[1, 2] /= dummy_height

    if gt_image_path is not None:
        gt_image, _ = process_image_and_k(
            gt_image_path,
            build_k(fx, fy, cx, cy),
            image_shape,
            normalize_intrinsics,
        )
    else:
        gt_image = None

    background_color = (
        torch.tensor(background_color_override, dtype=torch.float32)
        if background_color_override is not None
        else torch.tensor(dataset_cfg.get("background_color", [0.0, 0.0, 0.0]), dtype=torch.float32)
    )

    return ProbeView(
        label=label,
        extrinsics=pose.clone(),
        intrinsics=intrinsics.clone(),
        near=near.clone(),
        far=far.clone(),
        image_shape=image_shape,
        background_color=background_color.reshape(-1).clone(),
        gt_image=gt_image,
        meta={
            "source": "custom_pose",
            "scene": seq["scene"],
            "gt_image_path": None if gt_image_path is None else str(gt_image_path),
        },
    )


def build_probes(
    args: argparse.Namespace,
    packets: list[LoadedPacket],
) -> tuple[Optional[ProbeView], Optional[list[ProbeView]]]:
    """
    Returns:
      fixed_probe_for_all_prefixes, or
      dynamic_probes_per_prefix_mode handled elsewhere
    """
    if args.probe_mode == "packet_last_only":
        return None, None
    if args.probe_mode == "packet_all_prefix_targets":
        return None, None
    if args.probe_mode == "packet_fixed_target":
        if args.fixed_target_index is None:
            raise ValueError("--fixed_target_index is required for probe_mode=packet_fixed_target")
        probe = resolve_packet_fixed_target_probe(
            packets,
            fixed_target_index=args.fixed_target_index,
            fixed_target_camera_id=args.fixed_target_camera_id,
        )
        return probe, [probe]
    if args.probe_mode == "tartanair_frame":
        if args.probe_frame_index is None:
            raise ValueError("--probe_frame_index is required for probe_mode=tartanair_frame")
        dataset_cfg = load_tartanair_dataset_cfg(args.dataset_cfg, args.sequence_index)
        probe = tartanair_frame_probe_from_cfg(
            dataset_cfg=dataset_cfg,
            probe_frame_index=args.probe_frame_index,
            probe_camera=args.probe_camera,
            background_color_override=args.custom_background_color,
        )
        return probe, [probe]
    if args.probe_mode == "custom_pose":
        if args.probe_pose_json is None:
            raise ValueError("--probe_pose_json is required for probe_mode=custom_pose")
        dataset_cfg = load_tartanair_dataset_cfg(args.dataset_cfg, args.sequence_index)
        pose = parse_pose_matrix_json(args.probe_pose_json)
        probe = custom_pose_probe_from_cfg(
            dataset_cfg=dataset_cfg,
            pose=pose,
            label=args.probe_pose_json.stem,
            gt_image_path=args.gt_image_path,
            background_color_override=args.custom_background_color,
            image_shape_override=args.probe_image_shape,
            intrinsics_norm_override=args.probe_intrinsics_norm,
        )
        return probe, [probe]
    if args.probe_mode == "custom_pose_sequence":
        if args.probe_poses_json is None:
            raise ValueError("--probe_poses_json is required for probe_mode=custom_pose_sequence")
        dataset_cfg = load_tartanair_dataset_cfg(args.dataset_cfg, args.sequence_index)
        poses = parse_pose_sequence_json(args.probe_poses_json)
        probes = [
            custom_pose_probe_from_cfg(
                dataset_cfg=dataset_cfg,
                pose=pose,
                label=f"{args.probe_poses_json.stem}_{i:04d}",
                gt_image_path=args.gt_image_path,
                background_color_override=args.custom_background_color,
            )
            for i, pose in enumerate(poses)
        ]
        return None, probes
    raise ValueError(f"Unsupported probe_mode={args.probe_mode}")


def probes_for_prefix(
    args: argparse.Namespace,
    prefix_packets: list[LoadedPacket],
    fixed_probe: Optional[ProbeView],
    fixed_probe_sequence: Optional[list[ProbeView]],
) -> list[ProbeView]:
    if args.probe_mode == "packet_last_only":
        return build_packet_last_only_probes(prefix_packets)
    if args.probe_mode == "packet_all_prefix_targets":
        return build_packet_all_prefix_target_probes(prefix_packets)
    if args.probe_mode in {"packet_fixed_target", "tartanair_frame", "custom_pose"}:
        assert fixed_probe is not None
        return [fixed_probe]
    if args.probe_mode == "custom_pose_sequence":
        assert fixed_probe_sequence is not None
        return fixed_probe_sequence
    raise ValueError(f"Unsupported probe_mode={args.probe_mode}")


def main() -> None:
    args = parse_args()
    if args.max_packets is not None and args.max_packets <= 0:
        raise ValueError(f"--max_packets must be > 0, got {args.max_packets}")

    if args.max_packets is None and args.packet_ranges is None:
        raise ValueError("Either --max_packets or --packet_ranges must be specified.")

    device = resolve_device(args.device)
    packets = load_packets(
        args.packet_dir,
        max_packets=args.max_packets,
        packet_ranges=args.packet_ranges,
    )
    packet_image_shape = validate_packet_collection(packets)

    max_packets = len(packets)

    fixed_probe, fixed_probe_sequence = build_probes(args, packets)
    if fixed_probe is not None:
        print(f"Using fixed probe: {fixed_probe.label} | meta={fixed_probe.meta}")
    elif fixed_probe_sequence is not None and args.probe_mode == "custom_pose_sequence":
        print(f"Using fixed probe sequence: {len(fixed_probe_sequence)} poses from {args.probe_poses_json}")

    images_dir = args.output_dir / "images"
    gt_dir = args.output_dir / "gt"
    plots_dir = args.output_dir / "plots"
    videos_dir = args.output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    summary_steps: list[dict[str, Any]] = []
    curve_packet_counts: list[int] = []
    curve_num_gaussians: list[float] = []
    curve_psnr: list[float] = []
    curve_ssim: list[float] = []
    curve_lpips: list[float] = []

    progress_frames: dict[str, list[torch.Tensor]] = {}
    warned_probe_shape_mismatch = False

    for k in range(1, max_packets + 1):
        prefix_packets = packets[:k]
        fused = concat_fused_gaussians(prefix_packets, device)
        probes = probes_for_prefix(args, prefix_packets, fixed_probe, fixed_probe_sequence)

        image_shapes = {probe.image_shape for probe in probes}
        if len(image_shapes) != 1:
            raise ValueError(f"All probes must share the same image shape, got {sorted(image_shapes)}")
        probe_image_shape = next(iter(image_shapes))
        if tuple(probe_image_shape) != tuple(packet_image_shape) and not warned_probe_shape_mismatch:
            print(
                f"Warning: packet image_shape={packet_image_shape} differs from probe image_shape={probe_image_shape}. "
                "This is allowed for custom visualization, but make sure your probe intrinsics/FOV are intentional.",
                flush=True,
            )
            warned_probe_shape_mismatch = True

        rendered = render_fused_views(fused=fused, probes=probes, device=device)
        metrics, gt_indices = compute_metrics_for_available_gt(
            rendered=rendered,
            probes=probes,
            compute_lpips_flag=args.compute_lpips,
        )

        rendered_filenames: list[str] = []
        gt_filenames: list[str] = []
        per_view_metrics: list[dict[str, Any]] = []

        for render_idx, (probe, rendered_image) in enumerate(zip(probes, rendered)):
            # filename = f"k_{k:03d}_{probe.label}.png"
            filename = f"{probe.label}.png"
            save_png_tensor(rendered_image, images_dir / filename)
            rendered_filenames.append(filename)

            if probe.gt_image is not None:
                save_png_tensor(probe.gt_image, gt_dir / filename)
                gt_filenames.append(filename)

            if args.save_progress_video:
                progress_frames.setdefault(probe.label, []).append(rendered_image.detach().cpu().float())

            row = {
                "probe_label": probe.label,
                "gt_available": probe.gt_image is not None,
                "meta": probe.meta,
                "rendered_image": filename,
            }

            if probe.gt_image is not None:
                local_gt_idx = gt_indices.index(render_idx)
                row["psnr"] = float(metrics["psnr"][local_gt_idx].detach().cpu().item())
                row["ssim"] = float(metrics["ssim"][local_gt_idx].detach().cpu().item())
                if args.compute_lpips:
                    row["lpips"] = float(metrics["lpips"][local_gt_idx].detach().cpu().item())
            per_view_metrics.append(row)

        curve_packet_counts.append(k)
        curve_num_gaussians.append(float(fused.means.shape[1]))

        step_summary: dict[str, Any] = {
            "num_packets": k,
            "num_gaussians": int(fused.means.shape[1]),
            "base_scene": prefix_packets[0].base_scene,
            "probe_mode": args.probe_mode,
            "packet_filenames": [packet_id(packet) for packet in prefix_packets],
            "context_indices": [as_list(packet.data["context_index"]) for packet in prefix_packets],
            "target_indices": [as_list(packet.data["target_index"]) for packet in prefix_packets],
            "rendered_images": rendered_filenames,
            "gt_images": gt_filenames,
            "metrics_per_view": per_view_metrics,
        }

        if metrics:
            mean_psnr = metric_mean_as_float(metrics["psnr"])
            mean_ssim = metric_mean_as_float(metrics["ssim"])
            curve_psnr.append(mean_psnr)
            curve_ssim.append(mean_ssim)
            step_summary["metrics_mean"] = {"psnr": mean_psnr, "ssim": mean_ssim}
            log_message = (
                f"[{k}/{max_packets}] fused {k} packet(s) -> {int(fused.means.shape[1])} gaussians, "
                f"rendered {len(rendered_filenames)} probe view(s), "
                f"GT-views={len(gt_indices)}, PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}"
            )
            if args.compute_lpips:
                mean_lpips = metric_mean_as_float(metrics["lpips"])
                curve_lpips.append(mean_lpips)
                step_summary["metrics_mean"]["lpips"] = mean_lpips
                log_message += f", LPIPS={mean_lpips:.4f}"
            # print(log_message)
        else:
            log_message = (
                f"[{k}/{max_packets}] fused {k} packet(s) -> {int(fused.means.shape[1])} gaussians, "
                f"rendered {len(rendered_filenames)} probe view(s), no GT available."
            )
            # print(log_message)

        summary_steps.append(step_summary)

    plot_curve(
        curve_packet_counts,
        curve_num_gaussians,
        "Number of Gaussians vs Number of Fused Packets",
        "Number of Gaussians",
        plots_dir / "num_gaussians_vs_packets.png",
    )

    if curve_psnr:
        plot_curve(
            curve_packet_counts,
            curve_psnr,
            "PSNR vs Number of Fused Packets",
            "PSNR",
            plots_dir / "psnr_vs_packets.png",
        )
    if curve_ssim:
        plot_curve(
            curve_packet_counts,
            curve_ssim,
            "SSIM vs Number of Fused Packets",
            "SSIM",
            plots_dir / "ssim_vs_packets.png",
        )
    if args.compute_lpips and curve_lpips:
        plot_curve(
            curve_packet_counts,
            curve_lpips,
            "LPIPS vs Number of Fused Packets",
            "LPIPS",
            plots_dir / "lpips_vs_packets.png",
        )

    video_save_messages: list[str] = []
    if args.save_progress_video:
        for probe_label, frames in progress_frames.items():
            if frames:
                ok, message = save_progress_video_safe(
                    frames, videos_dir / f"{probe_label}.mp4"
                )
                video_save_messages.append(message)

    summary: dict[str, Any] = {
        "packet_dir": str(args.packet_dir),
        "output_dir": str(args.output_dir),
        "probe_mode": args.probe_mode,
        "packet_ranges": args.packet_ranges,
        "selected_packet_count": len(packets),
        "compute_lpips": bool(args.compute_lpips),
        "device": str(device),
        "available_packets": len(packets),
        "processed_packets": max_packets,
        "base_scene": packets[0].base_scene,
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "video_save_messages": video_save_messages,
        "curves": {
            "num_packets": curve_packet_counts,
            "num_gaussians": curve_num_gaussians,
        },
        "steps": summary_steps,
    }
    if curve_psnr:
        summary["curves"]["psnr"] = curve_psnr
    if curve_ssim:
        summary["curves"]["ssim"] = curve_ssim
    if args.compute_lpips and curve_lpips:
        summary["curves"]["lpips"] = curve_lpips
    if fixed_probe is not None:
        summary["fixed_probe"] = {
            "label": fixed_probe.label,
            "meta": fixed_probe.meta,
            "gt_available": fixed_probe.gt_image is not None,
        }
    if fixed_probe_sequence is not None and args.probe_mode == "custom_pose_sequence":
        summary["fixed_probe_sequence"] = {
            "count": len(fixed_probe_sequence),
            "labels": [probe.label for probe in fixed_probe_sequence],
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plots to {plots_dir}")
    if args.save_progress_video:
        # for message in video_save_messages:
            # print(message)
        print(f"Saved video-related outputs under {videos_dir}")


if __name__ == "__main__":
    main()
