#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio viewer for fused Gaussian packets.

Update highlights (v4):
- Based on the user-modified v3 layout.
- Default render map is now the final fusion of all loaded packets.
- The main slider now controls trajectory camera index, not prefix length.
- Added probe_offset_view: start from an existing packet target camera, then fine-tune local translation, rotation, and distance scale.
- Added custom render image shape and normalized intrinsics controls, equivalent to --probe_image_shape and --probe_intrinsics_norm.
- Robustly supports packet tensors saved either with or without an explicit batch dimension.

This viewer reads an existing work_dir produced by run_zipmap_resplat_fusion_api.py
and provides:
    1) GLB preview of Gaussian centers (all packets)
    2) Final-fused trajectory render viewer
    3) Packet-target render mode
    4) Orbit/free-view render mode
    5) Probe-offset render mode based on an existing trajectory camera
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch is required.") from e

try:
    import trimesh
except Exception:
    trimesh = None


def lazy_import_resplat(resplat_repo: Path):
    import sys
    resplat_src = resplat_repo / "src"
    if str(resplat_src.parent) not in sys.path:
        sys.path.insert(0, str(resplat_src.parent))
    from src.model.types import Gaussians
    from src.model.decoder.gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg
    return Gaussians, GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_batched_tensor(x: Any, field_name: str) -> torch.Tensor:
    """Return packet tensor with explicit batch dim.

    New API packets usually store Gaussian fields without batch dim, e.g.
      means: [G, 3], harmonics: [G, 3, SH]
    Older/debug packets may store them as
      means: [1, G, 3], harmonics: [1, G, 3, SH].
    The viewer internally uses batched tensors [1, ...].
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if field_name in {"means", "scales", "rotations", "rotations_unnorm"}:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    elif field_name == "covariances":
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    elif field_name == "harmonics":
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    elif field_name == "opacities":
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise RuntimeError(f"Unexpected {field_name} shape: {tuple(x.shape)}")
    return x


def tensor_first_view(x: Any) -> torch.Tensor:
    """Return first view/image/camera tensor robustly for [V,...] or already [...] shapes."""
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x[0] if x.ndim >= 4 else x



def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 3:
        x = x.permute(1, 2, 0).numpy()
    elif x.ndim == 2:
        x = x.numpy()
    arr = (x * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(arr)


def make_abs_diff(render_img: Optional[Image.Image], gt_img: Optional[Image.Image], gain: float = 4.0) -> Optional[Image.Image]:
    if render_img is None or gt_img is None:
        return None
    r = np.asarray(render_img).astype(np.float32) / 255.0
    gt_img = gt_img.resize(render_img.size, Image.BILINEAR)
    g = np.asarray(gt_img).astype(np.float32) / 255.0
    diff = np.clip(np.abs(r - g) * gain, 0.0, 1.0)
    diff = (diff * 255.0).astype(np.uint8)
    return Image.fromarray(diff)


def compute_psnr(render_img: Optional[Image.Image], gt_img: Optional[Image.Image]) -> Optional[float]:
    if render_img is None or gt_img is None:
        return None
    r = np.asarray(render_img).astype(np.float32) / 255.0
    g = np.asarray(gt_img.resize(render_img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    mse = float(np.mean((r - g) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def compute_ssim_simple(render_img: Optional[Image.Image], gt_img: Optional[Image.Image]) -> Optional[float]:
    if render_img is None or gt_img is None:
        return None
    x = np.asarray(render_img.resize(gt_img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    y = np.asarray(gt_img).astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x.mean(axis=2)
    if y.ndim == 3:
        y = y.mean(axis=2)
    ux, uy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - ux) * (y - uy)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(ssim)


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (sh0 * c0 + 0.5).clamp(0.0, 1.0)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def look_at_Twc(camera_pos: np.ndarray, target: np.ndarray, world_up: np.array = np.array([0, 1, 0], dtype=np.float32), roll_deg: float = 0.0) -> np.ndarray:
    z_cam_world = normalize(target - camera_pos)
    if np.linalg.norm(z_cam_world) < 1e-8:
        z_cam_world = np.array([0, 0, 1], dtype=np.float32)
    x_cam_world = normalize(np.cross(z_cam_world, world_up))
    if np.linalg.norm(x_cam_world) < 1e-8:
        alt_up = np.array([0, 0, 1], dtype=np.float32)
        x_cam_world = normalize(np.cross(z_cam_world, alt_up))
    y_cam_world = normalize(np.cross(z_cam_world, x_cam_world))
    y_cam_world = -y_cam_world
    if abs(roll_deg) > 1e-6:
        theta = math.radians(roll_deg)
        c, s = math.cos(theta), math.sin(theta)
        xr = c * x_cam_world + s * y_cam_world
        yr = -s * x_cam_world + c * y_cam_world
        x_cam_world, y_cam_world = xr, yr
    R = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=1).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = camera_pos.astype(np.float32)
    return T


def orbit_camera_Twc(pivot: np.ndarray, yaw_deg: float, pitch_deg: float, radius: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    offset = np.array([
        radius * cp * sy,
        radius * sp,
        radius * cp * cy,
    ], dtype=np.float32)
    cam_pos = pivot + offset
    return look_at_Twc(cam_pos, pivot, roll_deg=roll_deg)



def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def local_camera_delta_Twc(
    base_Twc: np.ndarray,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    distance_scale: float,
    pivot: np.ndarray,
) -> np.ndarray:
    """Create a custom camera pose by perturbing an existing camera pose.

    Translation is in the base camera's local axes.
    Rotation is camera-local: yaw around local y, pitch around local x, roll around local z.
    distance_scale moves the camera radially relative to the pivot before applying local translation.
    """
    base = np.asarray(base_Twc, dtype=np.float32).copy()
    if base.shape != (4, 4):
        raise RuntimeError(f"base_Twc must be 4x4, got {base.shape}")
    scale = max(float(distance_scale), 1e-4)
    R_base = base[:3, :3]
    c_base = base[:3, 3]
    c_scaled = pivot + (c_base - pivot) * scale
    local_t = np.array([tx, ty, tz], dtype=np.float32)
    c_new = c_scaled + R_base @ local_t
    Ry = rot_y(math.radians(yaw_deg))
    Rx = rot_x(math.radians(pitch_deg))
    Rz = rot_z(math.radians(roll_deg))
    R_delta = Ry @ Rx @ Rz
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = (R_base @ R_delta).astype(np.float32)
    out[:3, 3] = c_new.astype(np.float32)
    return out


def get_packet_Twc(packet: Dict[str, Any]) -> np.ndarray:
    ext = packet["target_extrinsics"]
    if torch.is_tensor(ext):
        ext_cpu = ext.detach().cpu()
        ext_np = ext_cpu[0].numpy() if ext_cpu.ndim == 3 else ext_cpu.numpy()
    else:
        ext_np = np.asarray(ext, dtype=np.float32)
        if ext_np.ndim == 3:
            ext_np = ext_np[0]
    return np.asarray(ext_np, dtype=np.float32)


def get_packet_intrinsics(packet: Dict[str, Any]) -> torch.Tensor:
    K = packet["target_intrinsics"]
    if not torch.is_tensor(K):
        K = torch.tensor(K, dtype=torch.float32)
    K = K.detach().float().cpu()
    if K.ndim == 3:
        K = K[0]
    return K


def normalized_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def resolve_render_camera_settings(
    probe_packet: Dict[str, Any],
    viewer_default_hw: Tuple[int, int],
    use_custom_render_settings: bool,
    render_h: float,
    render_w: float,
    fx_norm: float,
    fy_norm: float,
    cx_norm: float,
    cy_norm: float,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if use_custom_render_settings:
        H = max(1, int(render_h))
        W = max(1, int(render_w))
        K = normalized_intrinsics_matrix(fx_norm, fy_norm, cx_norm, cy_norm)
        return K, (H, W)
    return get_packet_intrinsics(probe_packet), viewer_default_hw


def get_packet_gt_image(packet: Dict[str, Any]) -> Optional[Image.Image]:
    tgt = packet.get("target_image", None)
    if torch.is_tensor(tgt):
        if tgt.ndim == 4:
            tgt = tgt[0]
        return tensor_to_pil(tgt)
    return None


@dataclass
class PacketRecord:
    path: Path
    original_index: int
    local_index: Optional[int]
    target_index: Optional[int]
    target_camera_id: Optional[int]
    packet: Dict[str, Any]


@dataclass
class ViewerState:
    work_dir: Path
    packet_stage: str
    packet_dir: Path
    viewer_out_dir: Path
    packets: List[PacketRecord]
    probe_choices: List[str]
    glb_path: Optional[Path]
    centroid_all: np.ndarray
    selected_probe_default: str
    image_shape_hw: Tuple[int, int]
    device: str
    resplat_repo: Path
    fused_cache_key: Optional[Tuple[float, int]] = None
    fused_cache: Optional[Dict[str, torch.Tensor]] = None


def load_single_packet(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Packet is not a dict: {path}")
    return obj


def get_packet_original_index(packet: Dict[str, Any], fallback_name: str) -> int:
    meta = packet.get("packet_meta", None)
    if isinstance(meta, dict) and "original_index" in meta and meta["original_index"] is not None:
        return int(meta["original_index"])
    if "target_index" in packet and packet["target_index"] is not None:
        try:
            t = packet["target_index"]
            if torch.is_tensor(t):
                return int(t.reshape(-1)[0].item())
            return int(t)
        except Exception:
            pass
    stem = Path(fallback_name).stem
    parts = stem.split("_")
    for x in reversed(parts):
        if x.isdigit():
            return int(x)
    return 0


def get_probe_label(packet: Dict[str, Any], idx: int, default_orig: int) -> str:
    target_cam = packet.get("target_camera_id", None)
    if torch.is_tensor(target_cam):
        target_cam = int(target_cam.reshape(-1)[0].item())
    cam_name = "cam?" if target_cam is None else ("left" if int(target_cam) == 0 else "right")
    target_idx = packet.get("target_index", None)
    if torch.is_tensor(target_idx):
        target_idx = int(target_idx.reshape(-1)[0].item())
    if target_idx is None:
        target_idx = default_orig
    return f"{idx:03d} | frame={int(target_idx):06d} | {cam_name}"


def load_packets(work_dir: Path, packet_stage: str = "final", map_location: str = "cpu") -> Tuple[List[PacketRecord], Tuple[int, int]]:
    packet_dir = work_dir / "gaussian_packets_api" / packet_stage
    if not packet_dir.exists():
        raise RuntimeError(f"Packet directory not found: {packet_dir}")
    paths = sorted(packet_dir.glob("*.pt"))
    if len(paths) == 0:
        raise RuntimeError(f"No packet files found in: {packet_dir}")
    packets = []
    img_hw = None
    for path in paths:
        packet = load_single_packet(path, map_location=map_location)
        orig = get_packet_original_index(packet, path.name)
        local_index = None
        if isinstance(packet.get("packet_meta", None), dict):
            local_index = packet["packet_meta"].get("local_index", None)
        target_idx = packet.get("target_index", None)
        if torch.is_tensor(target_idx):
            target_idx = int(target_idx.reshape(-1)[0].item()) if target_idx.numel() > 0 else None
        cam_id = packet.get("target_camera_id", None)
        if torch.is_tensor(cam_id):
            cam_id = int(cam_id.reshape(-1)[0].item()) if cam_id.numel() > 0 else None
        if img_hw is None:
            target_img = packet.get("target_image", None)
            if torch.is_tensor(target_img):
                if target_img.ndim == 3:
                    _, h, w = target_img.shape
                    img_hw = (int(h), int(w))
                elif target_img.ndim == 4:
                    _, _, h, w = target_img.shape
                    img_hw = (int(h), int(w))
        packets.append(PacketRecord(path=path, original_index=int(orig), local_index=None if local_index is None else int(local_index), target_index=None if target_idx is None else int(target_idx), target_camera_id=None if cam_id is None else int(cam_id), packet=packet))
    if img_hw is None:
        img_hw = (320, 320)
    return packets, img_hw


def packet_to_gaussian_fields(packet: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    needed = ["means", "covariances", "harmonics", "opacities", "scales", "rotations", "rotations_unnorm"]
    out = {}
    for k in needed:
        if k not in packet:
            raise RuntimeError(f"Packet missing field: {k}")
        out[k] = ensure_batched_tensor(packet[k], k)
    return out


def concat_packets_as_gaussians(packet_dicts: List[Dict[str, Any]], opacity_threshold: float = 0.0, topk: int = -1, device: str = "cpu"):
    if len(packet_dicts) == 0:
        raise RuntimeError("No packets selected for fusion.")
    means, covars, harms, opas, scales, rots, rots_u = [], [], [], [], [], [], []
    for packet in packet_dicts:
        g = packet_to_gaussian_fields(packet)
        means.append(g["means"])
        covars.append(g["covariances"])
        harms.append(g["harmonics"])
        opas.append(g["opacities"])
        scales.append(g["scales"])
        rots.append(g["rotations"])
        rots_u.append(g["rotations_unnorm"])
    means = torch.cat(means, dim=1)
    covars = torch.cat(covars, dim=1)
    harms = torch.cat(harms, dim=1)
    opas = torch.cat(opas, dim=1)
    scales = torch.cat(scales, dim=1)
    rots = torch.cat(rots, dim=1)
    rots_u = torch.cat(rots_u, dim=1)
    keep = torch.ones_like(opas, dtype=torch.bool)
    if opacity_threshold > 0.0:
        keep = keep & (opas >= opacity_threshold)
    if topk is not None and int(topk) > 0 and int(topk) < opas.shape[1]:
        _, idx = torch.topk(opas[0], k=int(topk), largest=True)
        keep2 = torch.zeros_like(keep)
        keep2[:, idx] = True
        keep = keep & keep2
    means = means[:, keep[0]]
    covars = covars[:, keep[0]]
    harms = harms[:, keep[0]]
    opas = opas[:, keep[0]]
    scales = scales[:, keep[0]]
    rots = rots[:, keep[0]]
    rots_u = rots_u[:, keep[0]]
    return {
        "means": means.to(device),
        "covariances": covars.to(device),
        "harmonics": harms.to(device),
        "opacities": opas.to(device),
        "scales": scales.to(device),
        "rotations": rots.to(device),
        "rotations_unnorm": rots_u.to(device),
    }


def fused_centroid_from_packets(packet_dicts: List[Dict[str, Any]]) -> np.ndarray:
    pts, ws = [], []
    for p in packet_dicts:
        g = packet_to_gaussian_fields(p)
        pts.append(g["means"][0].detach().cpu())
        ws.append(g["opacities"][0].detach().cpu().reshape(-1, 1))
    pts = torch.cat(pts, dim=0).numpy()
    ws = torch.cat(ws, dim=0).numpy()
    s = float(ws.sum())
    if s < 1e-8:
        return pts.mean(axis=0).astype(np.float32)
    c = (pts * ws).sum(axis=0) / s
    return c.astype(np.float32)


def selected_probe_center(packet: Dict[str, Any]) -> np.ndarray:
    t = packet.get("target_extrinsics", None)
    if t is None:
        return np.zeros(3, dtype=np.float32)
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    if t.ndim == 2:
        return t[:3, 3].astype(np.float32)
    if t.ndim == 3:
        return t[0, :3, 3].astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def export_glb_preview(work_dir: Path, packets: List[PacketRecord], max_preview_points: int = 120000) -> Optional[Path]:
    if trimesh is None:
        return None
    out_path = work_dir / "viewer_outputs" / "gaussian_centers_preview.glb"
    ensure_dir(out_path.parent)
    pts_all, color_all, opa_all = [], [], []
    for rec in packets:
        g = packet_to_gaussian_fields(rec.packet)
        means = g["means"][0].detach().cpu()
        opa = g["opacities"][0].detach().cpu()
        sh0 = g["harmonics"][0, :, :, 0].detach().cpu()
        rgb = sh0_to_rgb(sh0)
        pts_all.append(means)
        color_all.append(rgb)
        opa_all.append(opa)
    pts = torch.cat(pts_all, dim=0)
    colors = torch.cat(color_all, dim=0)
    opacities = torch.cat(opa_all, dim=0)
    n = pts.shape[0]
    if n > max_preview_points:
        _, idx = torch.topk(opacities, k=max_preview_points, largest=True)
        pts = pts[idx]
        colors = colors[idx]
        opacities = opacities[idx]
    verts = pts.numpy().astype(np.float32)
    rgba = torch.cat([colors, opacities[:, None].clamp(0.0, 1.0)], dim=1).numpy()
    rgba = (rgba * 255.0).clip(0, 255).astype(np.uint8)
    cloud = trimesh.points.PointCloud(vertices=verts, colors=rgba)
    scene = trimesh.Scene()
    scene.add_geometry(cloud)
    scene.export(str(out_path))
    return out_path


def build_decoder_and_gaussians(viewer_state: ViewerState, gauss_fields: Dict[str, torch.Tensor]):
    Gaussians, GSplatDecoderSplattingCUDA, GSplatDecoderSplattingCUDACfg = lazy_import_resplat(viewer_state.resplat_repo)
    class DatasetCfgLike:
        background_color = [0.0, 0.0, 0.0]
    decoder_cfg = GSplatDecoderSplattingCUDACfg(name="gsplat", scale_invariant=False, use_covariances=True)
    decoder = GSplatDecoderSplattingCUDA(decoder_cfg, DatasetCfgLike()).to(viewer_state.device)
    g = Gaussians(means=gauss_fields["means"], covariances=gauss_fields["covariances"], harmonics=gauss_fields["harmonics"], opacities=gauss_fields["opacities"], scales=gauss_fields["scales"], rotations=gauss_fields["rotations"], rotations_unnorm=gauss_fields["rotations_unnorm"])
    return decoder, g


def render_with_pose(viewer_state: ViewerState, gauss_fields: Dict[str, torch.Tensor], Twc: np.ndarray, intrinsics: Optional[torch.Tensor] = None, image_hw: Optional[Tuple[int, int]] = None) -> Image.Image:
    decoder, g = build_decoder_and_gaussians(viewer_state, gauss_fields)
    if image_hw is None:
        image_hw = viewer_state.image_shape_hw
    H, W = image_hw
    if intrinsics is None:
        K = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
    else:
        K = intrinsics.detach().float().cpu()
        if K.ndim == 3:
            K = K[0]
    ext = torch.tensor(Twc, dtype=torch.float32)[None, None].to(viewer_state.device)
    K = K[None, None].to(viewer_state.device)
    near = torch.tensor([[0.1]], dtype=torch.float32, device=viewer_state.device)
    far = torch.tensor([[50.0]], dtype=torch.float32, device=viewer_state.device)
    with torch.no_grad():
        out = decoder.forward(g, ext, K, near, far, (H, W))
    img = out.color[0, 0]
    return tensor_to_pil(img)


def load_run(work_dir_str: str, packet_stage: str, device: str, resplat_repo_str: str):
    try:
        work_dir = Path(work_dir_str).expanduser().resolve()
        resplat_repo = Path(resplat_repo_str).expanduser().resolve()
        packets, img_hw = load_packets(work_dir, packet_stage=packet_stage, map_location="cpu")
        ensure_dir(work_dir / "viewer_outputs")
        glb_path = export_glb_preview(work_dir, packets)
        probe_choices = [get_probe_label(rec.packet, i, rec.original_index) for i, rec in enumerate(packets)]
        centroid = fused_centroid_from_packets([r.packet for r in packets])
        default_probe = probe_choices[0]
        state = ViewerState(work_dir=work_dir, packet_stage=packet_stage, packet_dir=work_dir / "gaussian_packets_api" / packet_stage, viewer_out_dir=work_dir / "viewer_outputs", packets=packets, probe_choices=probe_choices, glb_path=glb_path, centroid_all=centroid, selected_probe_default=default_probe, image_shape_hw=img_hw, device=device, resplat_repo=resplat_repo)
        msg = f"Loaded {len(packets)} packets from {state.packet_dir}"
        return (state, gr.update(maximum=len(packets), value=min(1, len(packets))), gr.update(choices=probe_choices, value=default_probe), str(glb_path) if glb_path is not None else None, msg)
    except Exception as e:
        return None, gr.update(), gr.update(), None, f"[Load failed] {e}"


def parse_probe_index(label: str) -> int:
    try:
        prefix = label.split("|")[0].strip()
        return int(prefix)
    except Exception:
        return 0


def preset_values(name: str):
    if name == "top_down":
        return 0.0, 85.0, 4.0, 0.0
    if name == "oblique_top":
        return 35.0, 55.0, 4.5, 0.0
    if name == "front":
        return 0.0, 0.0, 3.5, 0.0
    if name == "left":
        return -90.0, 0.0, 3.5, 0.0
    if name == "right":
        return 90.0, 0.0, 3.5, 0.0
    return 0.0, 30.0, 4.0, 0.0


def choose_pivot(viewer_state: ViewerState, selected_probe_label: str, pivot_mode: str, custom_x: float, custom_y: float, custom_z: float) -> np.ndarray:
    if pivot_mode == "fused_centroid":
        return viewer_state.centroid_all
    if pivot_mode == "selected_probe_center":
        probe_idx = parse_probe_index(selected_probe_label)
        probe_idx = max(0, min(probe_idx, len(viewer_state.packets) - 1))
        return selected_probe_center(viewer_state.packets[probe_idx].packet)
    return np.array([custom_x, custom_y, custom_z], dtype=np.float32)


def get_final_fused_fields(viewer_state: ViewerState, opacity_threshold: float, topk: int) -> Dict[str, torch.Tensor]:
    key = (round(float(opacity_threshold), 6), int(topk))
    if viewer_state.fused_cache is not None and viewer_state.fused_cache_key == key:
        return viewer_state.fused_cache
    selected_packet_dicts = [rec.packet for rec in viewer_state.packets]
    viewer_state.fused_cache = concat_packets_as_gaussians(
        selected_packet_dicts,
        opacity_threshold=float(opacity_threshold),
        topk=int(topk),
        device=viewer_state.device,
    )
    viewer_state.fused_cache_key = key
    return viewer_state.fused_cache


def save_pose_json(path: Path, Twc: np.ndarray):
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump({"Twc": np.asarray(Twc, dtype=np.float32).tolist()}, f, indent=2)


def render_action(
    viewer_state: ViewerState,
    trajectory_index: int,
    selected_probe_label: str,
    camera_mode: str,
    pivot_mode: str,
    yaw_deg: float,
    pitch_deg: float,
    radius: float,
    roll_deg: float,
    custom_x: float,
    custom_y: float,
    custom_z: float,
    offset_tx: float,
    offset_ty: float,
    offset_tz: float,
    offset_yaw_deg: float,
    offset_pitch_deg: float,
    offset_roll_deg: float,
    offset_distance_scale: float,
    use_custom_render_settings: bool,
    render_h: float,
    render_w: float,
    fx_norm: float,
    fy_norm: float,
    cx_norm: float,
    cy_norm: float,
    opacity_threshold: float,
    topk: int,
):
    if viewer_state is None:
        return None, None, None, "Please load a run first.", "", ""
    try:
        n = len(viewer_state.packets)
        traj_i = max(1, min(int(trajectory_index), n))
        probe_idx = traj_i - 1
        probe_packet = viewer_state.packets[probe_idx].packet
        selected_probe_label = get_probe_label(probe_packet, probe_idx, viewer_state.packets[probe_idx].original_index)

        gauss_fields = get_final_fused_fields(
            viewer_state,
            opacity_threshold=float(opacity_threshold),
            topk=int(topk),
        )

        K, image_hw = resolve_render_camera_settings(
            probe_packet,
            viewer_state.image_shape_hw,
            bool(use_custom_render_settings),
            render_h,
            render_w,
            fx_norm,
            fy_norm,
            cx_norm,
            cy_norm,
        )

        gt_img = None
        base_Twc = get_packet_Twc(probe_packet)

        if camera_mode == "packet_target":
            Twc = base_Twc
            render_img = render_with_pose(viewer_state, gauss_fields, Twc, intrinsics=K, image_hw=image_hw)
            gt_img = get_packet_gt_image(probe_packet)
            pose_name = f"render_final_traj{traj_i:03d}_packet_target_pose.json"
        elif camera_mode == "probe_offset_view":
            pivot = choose_pivot(viewer_state, selected_probe_label, pivot_mode, custom_x, custom_y, custom_z)
            Twc = local_camera_delta_Twc(
                base_Twc=base_Twc,
                tx=float(offset_tx),
                ty=float(offset_ty),
                tz=float(offset_tz),
                yaw_deg=float(offset_yaw_deg),
                pitch_deg=float(offset_pitch_deg),
                roll_deg=float(offset_roll_deg),
                distance_scale=float(offset_distance_scale),
                pivot=pivot,
            )
            render_img = render_with_pose(viewer_state, gauss_fields, Twc, intrinsics=K, image_hw=image_hw)
            pose_name = f"render_final_traj{traj_i:03d}_probe_offset_pose.json"
        else:
            pivot = choose_pivot(viewer_state, selected_probe_label, pivot_mode, custom_x, custom_y, custom_z)
            Twc = orbit_camera_Twc(pivot=pivot, yaw_deg=float(yaw_deg), pitch_deg=float(pitch_deg), radius=float(radius), roll_deg=float(roll_deg))
            render_img = render_with_pose(viewer_state, gauss_fields, Twc, intrinsics=K, image_hw=image_hw)
            pose_name = f"render_final_traj{traj_i:03d}_orbit_pose.json"

        pose_path = viewer_state.viewer_out_dir / pose_name
        save_pose_json(pose_path, Twc)
        diff_img = make_abs_diff(render_img, gt_img, gain=4.0)
        psnr = compute_psnr(render_img, gt_img)
        ssim = compute_ssim_simple(render_img, gt_img)
        num_gauss = int(gauss_fields["means"].shape[1])
        pose_text = json.dumps({"Twc": np.asarray(Twc).round(6).tolist()}, indent=2)
        stats = {
            "fusion_semantics": "final_all_loaded_packets",
            "trajectory_index_1_based": int(traj_i),
            "probe_idx_0_based": int(probe_idx),
            "num_loaded_packets": n,
            "num_fused_gaussians": num_gauss,
            "opacity_threshold": float(opacity_threshold),
            "topk": int(topk),
            "camera_mode": camera_mode,
            "pivot_mode": pivot_mode,
            "probe_label": selected_probe_label,
            "custom_render_settings": bool(use_custom_render_settings),
            "render_image_hw": [int(image_hw[0]), int(image_hw[1])],
            "intrinsics_norm": np.asarray(K.detach().cpu()).round(6).tolist(),
            "psnr": None if psnr is None else round(float(psnr), 4),
            "ssim": None if ssim is None else round(float(ssim), 4),
            "saved_pose_json": str(pose_path),
        }
        stats_text = json.dumps(stats, indent=2)
        return render_img, gt_img, diff_img, "Render complete.", pose_text, stats_text
    except Exception as e:
        return None, None, None, f"[Render failed] {e}", "", ""


def trajectory_index_to_probe_label(viewer_state: ViewerState, trajectory_index: int):
    if viewer_state is None:
        return gr.update()
    n = len(viewer_state.packets)
    idx = max(1, min(int(trajectory_index), n)) - 1
    return gr.update(value=viewer_state.probe_choices[idx])

def apply_preset(name: str):
    return preset_values(name)


def build_app(default_work_dir: str, default_packet_stage: str, default_device: str, default_resplat_repo: str, default_probe_image_shape: Tuple[int, int], default_probe_intrinsics_norm: Tuple[float, float, float, float]):
    theme = gr.themes.Ocean()
    with gr.Blocks(theme=theme, title="Fusion Viewer") as demo:
        viewer_state = gr.State(None)
        gr.Markdown("""
# Fused 3DGS Viewer

This viewer has two parts:
- **GLB Preview**: fast geometry preview of Gaussian centers from **all loaded packets**
- **Rendered View**: actual CUDA render of the final fused Gaussian map

### Orbit parameter meanings
- **yaw [deg]**: horizontal rotation around the pivot (turn left/right around the scene)
- **pitch [deg]**: vertical elevation angle; larger positive values move toward top-down view
- **radius**: distance from the pivot to the camera
- **roll [deg]**: rotate the camera around its own forward axis (like tilting your head)

**Tip:** for room overview, start with:
- camera mode = `probe_offset_view` to start from an existing camera and fine-tune it
- camera mode = `orbit_free_view` for global free-view inspection
- enable custom render settings for `--probe_image_shape` / `--probe_intrinsics_norm` equivalent behavior
""")
        with gr.Row():
            with gr.Column(scale=2):
                work_dir = gr.Textbox(label="work_dir", value=default_work_dir)
                packet_stage = gr.Dropdown(label="packet_stage", choices=["final", "init"], value=default_packet_stage)
                device = gr.Textbox(label="device", value=default_device)
                resplat_repo = gr.Textbox(label="resplat_repo", value=default_resplat_repo)
                load_btn = gr.Button("Load run", variant="primary")
                load_log = gr.Markdown("")
            with gr.Column(scale=3):
                glb_preview = gr.Model3D(label="GLB preview (all loaded packets)", height=420, clear_color=[1, 1, 1, 1])
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Fusion / Camera Controls")
                trajectory_index = gr.Slider(label="trajectory camera index (final fused map)", minimum=1, maximum=1, value=1, step=1)
                selected_probe = gr.Dropdown(label="current/base packet target", choices=[], value=None)
                camera_mode = gr.Radio(label="camera mode", choices=["packet_target", "probe_offset_view", "orbit_free_view"], value="packet_target")
                with gr.Group():
                    gr.Markdown("### Orbit preset and pivot")
                    preset = gr.Dropdown(label="preset", choices=["none", "top_down", "oblique_top", "front", "left", "right"], value="oblique_top")
                    apply_preset_btn = gr.Button("Apply preset")
                    pivot_mode = gr.Radio(label="pivot mode", choices=["fused_centroid", "selected_probe_center", "custom_xyz"], value="fused_centroid")
                    with gr.Row():
                        custom_x = gr.Number(label="custom_x", value=0.0)
                        custom_y = gr.Number(label="custom_y", value=0.0)
                        custom_z = gr.Number(label="custom_z", value=0.0)
                with gr.Group():
                    gr.Markdown("### Orbit camera parameters")
                    yaw_deg = gr.Slider(label="yaw [deg]", minimum=-180, maximum=180, value=35.0, step=1.0)
                    pitch_deg = gr.Slider(label="pitch [deg]", minimum=-89, maximum=89, value=55.0, step=1.0)
                    radius = gr.Slider(label="radius", minimum=0.1, maximum=20.0, value=4.5, step=0.1)
                    roll_deg = gr.Slider(label="roll [deg]", minimum=-180, maximum=180, value=0.0, step=1.0)
                with gr.Group():
                    gr.Markdown("### Probe-offset camera parameters")
                    gr.Markdown("Start from the selected trajectory camera, then apply local translation/rotation. distance_scale moves the base camera closer/farther relative to the pivot.")
                    with gr.Row():
                        offset_tx = gr.Number(label="local tx", value=0.0)
                        offset_ty = gr.Number(label="local ty", value=0.0)
                        offset_tz = gr.Number(label="local tz", value=0.0)
                    with gr.Row():
                        offset_yaw_deg = gr.Slider(label="offset yaw [deg]", minimum=-180, maximum=180, value=0.0, step=1.0)
                        offset_pitch_deg = gr.Slider(label="offset pitch [deg]", minimum=-89, maximum=89, value=0.0, step=1.0)
                        offset_roll_deg = gr.Slider(label="offset roll [deg]", minimum=-180, maximum=180, value=0.0, step=1.0)
                    offset_distance_scale = gr.Slider(label="distance scale", minimum=0.1, maximum=5.0, value=1.0, step=0.05)
                with gr.Group():
                    gr.Markdown("### Custom render settings")
                    use_custom_render_settings = gr.Checkbox(label="use custom image shape / intrinsics", value=False)
                    with gr.Row():
                        render_h = gr.Number(label="render H", value=default_probe_image_shape[0], precision=0)
                        render_w = gr.Number(label="render W", value=default_probe_image_shape[1], precision=0)
                    with gr.Row():
                        fx_norm = gr.Number(label="fx_norm", value=default_probe_intrinsics_norm[0])
                        fy_norm = gr.Number(label="fy_norm", value=default_probe_intrinsics_norm[1])
                        cx_norm = gr.Number(label="cx_norm", value=default_probe_intrinsics_norm[2])
                        cy_norm = gr.Number(label="cy_norm", value=default_probe_intrinsics_norm[3])
                with gr.Group():
                    gr.Markdown("### Gaussian filtering")
                    opacity_threshold = gr.Slider(label="opacity threshold", minimum=0.0, maximum=1.0, value=0.0, step=0.005)
                    topk = gr.Number(label="topk (<=0 means keep all)", value=-1, precision=0)
                with gr.Row():
                    render_btn = gr.Button("Render", variant="primary")
                    gr.Markdown("Orbit sliders also trigger render automatically.")
            with gr.Column(scale=3):
                gr.Markdown("## Render Output")
                with gr.Row():
                    render_img = gr.Image(label="rendered image", type="pil", height=360)
                    gt_img = gr.Image(label="GT image (available in packet_target mode)", type="pil", height=360)
                    diff_img = gr.Image(label="abs diff ×4", type="pil", height=360)
                render_log = gr.Markdown("")
                with gr.Row():
                    pose_json = gr.Code(label="current Twc pose JSON", language="json")
                    stats_json = gr.Code(label="render stats", language="json")
        load_btn.click(load_run, inputs=[work_dir, packet_stage, device, resplat_repo], outputs=[viewer_state, trajectory_index, selected_probe, glb_preview, load_log])
        apply_preset_btn.click(apply_preset, inputs=[preset], outputs=[yaw_deg, pitch_deg, radius, roll_deg])
        trajectory_index.change(trajectory_index_to_probe_label, inputs=[viewer_state, trajectory_index], outputs=[selected_probe])
        render_inputs = [viewer_state, trajectory_index, selected_probe, camera_mode, pivot_mode, yaw_deg, pitch_deg, radius, roll_deg, custom_x, custom_y, custom_z, offset_tx, offset_ty, offset_tz, offset_yaw_deg, offset_pitch_deg, offset_roll_deg, offset_distance_scale, use_custom_render_settings, render_h, render_w, fx_norm, fy_norm, cx_norm, cy_norm, opacity_threshold, topk]
        render_outputs = [render_img, gt_img, diff_img, render_log, pose_json, stats_json]
        render_btn.click(render_action, inputs=render_inputs, outputs=render_outputs)
        for ctrl in [trajectory_index, selected_probe, camera_mode, pivot_mode, yaw_deg, pitch_deg, radius, roll_deg, custom_x, custom_y, custom_z, offset_tx, offset_ty, offset_tz, offset_yaw_deg, offset_pitch_deg, offset_roll_deg, offset_distance_scale, use_custom_render_settings, render_h, render_w, fx_norm, fy_norm, cx_norm, cy_norm, opacity_threshold, topk]:
            ctrl.change(render_action, inputs=render_inputs, outputs=render_outputs)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/home/shiyo/Desktop/ZipMap/outputs/api_demo_p000_0_20")
    parser.add_argument("--packet_stage", type=str, default="final")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resplat_repo", type=str, default="/home/shiyo/Desktop/Resplat")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--probe_image_shape", type=int, nargs=2, default=[540, 960], metavar=("H", "W"))
    parser.add_argument("--probe_intrinsics_norm", type=float, nargs=4, default=[0.35, 0.55, 0.5, 0.5], metavar=("FX", "FY", "CX", "CY"))
    args = parser.parse_args()
    demo = build_app(
        default_work_dir=args.work_dir,
        default_packet_stage=args.packet_stage,
        default_device=args.device,
        default_resplat_repo=args.resplat_repo,
        default_probe_image_shape=(int(args.probe_image_shape[0]), int(args.probe_image_shape[1])),
        default_probe_intrinsics_norm=tuple(float(x) for x in args.probe_intrinsics_norm),
    )
    demo.queue(max_size=20).launch(server_name=args.server_name, server_port=args.server_port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
