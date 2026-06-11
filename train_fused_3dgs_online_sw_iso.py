#!/usr/bin/env python3
"""
online sliding-window single-view stochastic refinement
每次从当前 window 随机采样 1 个 view
render/update 当前已插入的 all Gaussians
支持:
  active_mode all_inserted
  trainable_mode all
  lambda_iso = 0 / 1
  
虽然脚本名字带 iso，但 lambda_iso=0 时就是当前主线
"""
"""
Online-style sliding-window refinement with optional isotropic regularization for a fused ReSplat/MVSplat Gaussian map
using the original GraphDECO 3DGS backend.

Intended location:
  /home/shiyo/Desktop/ZipMap/train_fused_3dgs_online_sliding_window_replay.py

This script assumes that a packet-camera 3DGS scene has already been prepared,
e.g. by prepare_resplat_fused_3dgs_scene_v2.py with --camera_source packet.
It uses:
  - source_path: transforms_train.json / images / points3d.ply
  - initial_ply: full fused 3DGS PLY for packets 0..N-1

Online/incremental semantics:
  - Future packets are NOT present in the active global map.
  - Start by inserting the first window_size packets.
  - At every stride, insert new stride packets.
  - Render using all currently inserted Gaussians.
  - Render using all currently inserted Gaussians.
  - Active/update set is controlled by --active_mode:
      window       : update only Gaussians from packets in the current window.
      all_inserted : update all currently inserted Gaussians.
  - For the isotropic-prior experiment, use --active_mode all_inserted and --trainable_mode all.

First target setting:
  P000 0-30, window_size=10, stride=5
  W0: insert 0-9, optimize 0-9
  W1: insert 10-14, optimize 5-14
  W2: insert 15-19, optimize 10-19
  W3: insert 20-24, optimize 15-24
  W4: insert 25-29, optimize 20-29

Limitations:
  - This script assumes equal Gaussians per packet, inferred from full initial_ply
    unless --gaussians_per_packet is specified.
  - It is fixed-count only. Densification/pruning inside windows is intentionally
    not supported in this first online sliding-window prototype.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint
import random
from typing import Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Pre-parse original 3DGS repo path before importing graphdeco modules.
# -----------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gs_repo", type=str, default="/home/shiyo/Desktop/gaussian-splatting")
_pre_args, _remaining = _pre.parse_known_args(sys.argv[1:])
if _pre_args.gs_repo and _pre_args.gs_repo not in sys.path:
    sys.path.insert(0, _pre_args.gs_repo)

import torch
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


TRAINABLE_GROUPS = {
    "all": {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"},
    "opacity_color": {"f_dc", "f_rest", "opacity"},
    "opacity_color_scale": {"f_dc", "f_rest", "opacity", "scaling"},
}

PARAM_ATTR_BY_GROUP = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "opacity": "_opacity",
    "scaling": "_scaling",
    "rotation": "_rotation",
}


def parse_int_ranges(spec: str) -> List[int]:
    out: List[int] = []
    if spec is None or spec.strip() == "":
        return out
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
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


def init_wandb(args):
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed; skip W&B logging.")
        return None
    run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.model_path))
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def log_wandb_eval(wandb_run, entry: Dict, step: int, prefix: str = "eval") -> None:
    if wandb_run is None:
        return
    log_data = {}
    for k in [
        "optimization_time_sec", "optimization_time_per_iter_sec", "wall_time_sec",
        "eval_time_sec", "eval_time_sec_accum", "num_gaussians",
        "inserted_packets", "active_packets", "active_gaussians", "global_iteration",
        "window_id", "window_start", "window_end_exclusive",
    ]:
        if k in entry:
            log_data[f"{prefix}/{k}"] = entry[k]
    for split, metrics in entry.items():
        if not isinstance(metrics, dict):
            continue
        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)):
                log_data[f"{prefix}/{split}_{mk}"] = mv
    wandb_run.log(log_data, step=step)


def prepare_output_and_logger(args):
    if not args.model_path:
        raise ValueError("--model_path is required")
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def save_image_tensor(path: str, image: torch.Tensor) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (torch.clamp(image.detach(), 0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def sorted_cameras(scene: Scene, split: str, deduplicate: bool = True) -> List:
    """Return cameras in chronological order.

    Original 3DGS merges transforms_test.json into train cameras when dataset.eval=False.
    If transforms_test.json is the same as transforms_train.json, this creates duplicate
    image names like [000000, 000000, 000001, 000001, ...].  Sliding-window camera
    slicing by packet id requires one camera per packet, so we deduplicate by image_name
    as an additional safety guard.
    """
    cams = scene.getTrainCameras() if split == "train" else scene.getTestCameras()
    cams = sorted(cams, key=lambda c: c.image_name)
    if not deduplicate:
        return cams
    unique = []
    seen = set()
    for cam in cams:
        name = str(cam.image_name)
        if name in seen:
            continue
        unique.append(cam)
        seen.add(name)
    return unique


@torch.no_grad()
def evaluate_camera_list(
    cams: Sequence,
    gaussians: GaussianModel,
    pipe,
    background,
    dataset,
    max_views: int = 0,
    rendered_dir: Optional[str] = None,
    gt_dir: Optional[str] = None,
    save_gt: bool = True,
    overwrite_gt: bool = False,
) -> Dict[str, float]:
    cams = list(cams)
    if max_views and max_views > 0:
        cams = cams[:max_views]
    if len(cams) == 0:
        return {"num_views": 0}

    render_args_extra = (1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
    l1_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    for i, viewpoint in enumerate(cams):
        image = torch.clamp(
            render(viewpoint, gaussians, pipe, background, *render_args_extra)["render"],
            0.0,
            1.0,
        )
        gt = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        l1_sum += l1_loss(image, gt).mean().double().item()
        psnr_sum += psnr(image, gt).mean().double().item()
        ssim_sum += ssim(image, gt).mean().double().item()

        filename = f"{i:04d}_{viewpoint.image_name}.png"
        if rendered_dir is not None:
            save_image_tensor(os.path.join(rendered_dir, filename), image)
        if save_gt and gt_dir is not None:
            gt_path = os.path.join(gt_dir, filename)
            if overwrite_gt or (not os.path.exists(gt_path)):
                save_image_tensor(gt_path, gt)

    return {
        "num_views": len(cams),
        "l1": l1_sum / len(cams),
        "psnr": psnr_sum / len(cams),
        "ssim": ssim_sum / len(cams),
    }


def make_packet_windows(packet_ids: List[int], window_size: int, stride: int) -> List[Tuple[int, int, List[int], List[int]]]:
    """Return (window_id, start_pos, window_packet_ids, newly_inserted_packet_ids)."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(packet_ids) < window_size:
        raise ValueError(f"Need at least window_size packets, got {len(packet_ids)} < {window_size}")
    windows = []
    inserted_until = 0
    wid = 0
    start = 0
    while start + window_size <= len(packet_ids):
        end = start + window_size
        win_ids = packet_ids[start:end]
        new_ids = packet_ids[inserted_until:end]
        windows.append((wid, start, win_ids, new_ids))
        inserted_until = max(inserted_until, end)
        wid += 1
        start += stride
    return windows


def load_full_gaussian_params(initial_ply: str, sh_degree: int, optimizer_type: str, train_test_exp: bool) -> Dict[str, torch.Tensor]:
    g = GaussianModel(sh_degree, optimizer_type)
    g.load_ply(initial_ply, train_test_exp)
    return {
        "xyz": g._xyz.detach().cpu(),
        "features_dc": g._features_dc.detach().cpu(),
        "features_rest": g._features_rest.detach().cpu(),
        "opacity": g._opacity.detach().cpu(),
        "scaling": g._scaling.detach().cpu(),
        "rotation": g._rotation.detach().cpu(),
    }


def slice_params_by_indices(params: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[indices].clone() for k, v in params.items()}


def concat_param_dicts(param_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not param_dicts:
        raise ValueError("No parameter dictionaries to concatenate")
    keys = list(param_dicts[0].keys())
    return {k: torch.cat([d[k] for d in param_dicts], dim=0) for k in keys}


def current_params_to_cpu(gaussians: GaussianModel) -> Dict[str, torch.Tensor]:
    return {
        "xyz": gaussians._xyz.detach().cpu().clone(),
        "features_dc": gaussians._features_dc.detach().cpu().clone(),
        "features_rest": gaussians._features_rest.detach().cpu().clone(),
        "opacity": gaussians._opacity.detach().cpu().clone(),
        "scaling": gaussians._scaling.detach().cpu().clone(),
        "rotation": gaussians._rotation.detach().cpu().clone(),
    }


def assign_params_to_gaussians(gaussians: GaussianModel, params: Dict[str, torch.Tensor]) -> None:
    device = "cuda"
    gaussians._xyz = nn.Parameter(params["xyz"].to(device).float().contiguous().requires_grad_(True))
    gaussians._features_dc = nn.Parameter(params["features_dc"].to(device).float().contiguous().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(params["features_rest"].to(device).float().contiguous().requires_grad_(True))
    gaussians._opacity = nn.Parameter(params["opacity"].to(device).float().contiguous().requires_grad_(True))
    gaussians._scaling = nn.Parameter(params["scaling"].to(device).float().contiguous().requires_grad_(True))
    gaussians._rotation = nn.Parameter(params["rotation"].to(device).float().contiguous().requires_grad_(True))
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=device)
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)


def build_packet_index_map(packet_ids: List[int], gaussians_per_packet: int) -> Dict[int, Tuple[int, int]]:
    return {pid: (i * gaussians_per_packet, (i + 1) * gaussians_per_packet) for i, pid in enumerate(packet_ids)}


def build_cpu_map_from_inserted(
    inserted_packet_ids: List[int],
    optimized_inserted_params: Optional[Dict[str, torch.Tensor]],
    newly_inserted_ids: List[int],
    full_params: Dict[str, torch.Tensor],
    full_index_map: Dict[int, Tuple[int, int]],
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """
    Preserve optimized params for already inserted packets and append raw params for newly inserted packets.
    inserted_packet_ids is the current map order before insertion.
    """
    if optimized_inserted_params is None:
        pieces = []
        new_order = []
    else:
        pieces = [optimized_inserted_params]
        new_order = list(inserted_packet_ids)

    for pid in newly_inserted_ids:
        if pid in new_order:
            continue
        s, e = full_index_map[pid]
        idx = torch.arange(s, e, dtype=torch.long)
        pieces.append(slice_params_by_indices(full_params, idx))
        new_order.append(pid)

    return concat_param_dicts(pieces), new_order


def active_mask_for_packets(current_packet_order: List[int], active_packet_ids: List[int], gaussians_per_packet: int, device: str = "cuda") -> torch.Tensor:
    total = len(current_packet_order) * gaussians_per_packet
    mask = torch.zeros((total,), dtype=torch.bool, device=device)
    active_set = set(active_packet_ids)
    for i, pid in enumerate(current_packet_order):
        if pid in active_set:
            mask[i * gaussians_per_packet:(i + 1) * gaussians_per_packet] = True
    return mask


def choose_active_packet_ids(inserted_packet_order: List[int], win_ids: List[int], active_mode: str) -> List[int]:
    """Return which packet-owned Gaussian blocks are trainable in the current window.

    window:       only packets inside the current sliding window are updated.
    all_inserted: all currently inserted packets are updated. This is closer to
                  the original mapping-style behavior and lets old Gaussians be
                  refined by later-window losses.
    """
    if active_mode == "window":
        return list(win_ids)
    if active_mode == "all_inserted":
        return list(inserted_packet_order)
    raise ValueError(f"Unknown active_mode: {active_mode}")


def apply_grad_mask(gaussians: GaussianModel, active_mask: torch.Tensor, trainable_mode: str) -> None:
    allowed = TRAINABLE_GROUPS[trainable_mode]
    for group_name, attr in PARAM_ATTR_BY_GROUP.items():
        p = getattr(gaussians, attr, None)
        if p is None or p.grad is None:
            continue
        if group_name not in allowed:
            p.grad.zero_()
        else:
            # zero frozen rows. Works for [N,*] tensors.
            p.grad[~active_mask] = 0


def render_mapping_loss_for_camera(
    viewpoint_cam,
    gaussians: GaussianModel,
    pipe,
    bg: torch.Tensor,
    dataset,
    opt,
    depth_l1_weight,
    global_iteration: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, Dict]:
    """Render one camera and compute original 3DGS-style mapping loss.

    This script does not modify GraphDECO's renderer or GaussianModel.  It reuses
    the original modules and defines the SLAM-style window/replay loss externally.
    """
    render_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg,
        use_trained_exp=dataset.train_test_exp,
        separate_sh=SPARSE_ADAM_AVAILABLE,
    )
    image = render_pkg["render"]
    if viewpoint_cam.alpha_mask is not None:
        image = image * viewpoint_cam.alpha_mask.cuda()
    gt_image = viewpoint_cam.original_image.cuda()

    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

    Ll1depth_item = 0.0
    if depth_l1_weight(global_iteration) > 0 and getattr(viewpoint_cam, "depth_reliable", False):
        inv_depth = render_pkg["depth"]
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()
        Ll1depth_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
        Ll1depth = depth_l1_weight(global_iteration) * Ll1depth_pure
        loss = loss + Ll1depth
        Ll1depth_item = Ll1depth.item()

    return loss, Ll1, ssim_value, Ll1depth_item, image, gt_image, render_pkg


def sample_replay_cameras(
    cams_all: Sequence,
    packet_ids: List[int],
    inserted_packet_order: List[int],
    win_ids: List[int],
    args,
) -> List:
    """Select past replay cameras following the MonoGS-style random replay idea."""
    if args.replay_mode == "none" or args.num_replay_views <= 0 or args.lambda_replay <= 0:
        return []

    win_set = set(win_ids)
    if args.replay_pool == "past":
        # Strict past views before the current window start.  This is the cleanest
        # retention objective for online sliding-window refinement.
        start_pid = win_ids[0]
        pool_ids = [pid for pid in inserted_packet_order if pid < start_pid]
    elif args.replay_pool == "inserted_outside_window":
        # All already inserted views that are not in the current active window.
        pool_ids = [pid for pid in inserted_packet_order if pid not in win_set]
    else:
        raise ValueError(f"Unknown replay_pool: {args.replay_pool}")

    if not pool_ids:
        return []

    if args.replay_mode == "random":
        chosen = random.sample(pool_ids, k=min(args.num_replay_views, len(pool_ids)))
    elif args.replay_mode == "anchor":
        # Deterministic sparse anchors, e.g. 0, 5, 10, ... for anchor_stride=5.
        anchor_stride = max(1, int(args.replay_anchor_stride))
        anchors = [pid for pid in pool_ids if (pid - packet_ids[0]) % anchor_stride == 0]
        if not anchors:
            anchors = pool_ids[::anchor_stride]
        if len(anchors) <= args.num_replay_views:
            chosen = anchors
        else:
            chosen = random.sample(anchors, k=args.num_replay_views)
    else:
        raise ValueError(f"Unknown replay_mode: {args.replay_mode}")

    pos_by_pid = {pid: i for i, pid in enumerate(packet_ids)}
    return [cams_all[pos_by_pid[pid]] for pid in chosen if pid in pos_by_pid]


def active_isotropic_loss(gaussians: GaussianModel, active_mask: torch.Tensor) -> torch.Tensor:
    """MonoGS-style isotropic scale regularization on active Gaussians only.

    Uses activated scaling values, i.e. positive ellipsoid axis lengths.  Penalizes
    anisotropy inside each Gaussian, not absolute size.
    """
    if active_mask is None or int(active_mask.sum().item()) == 0:
        return torch.tensor(0.0, device="cuda")
    s = gaussians.get_scaling[active_mask]
    mean_s = s.mean(dim=1, keepdim=True)
    return torch.abs(s - mean_s).mean()


def camera_subset_by_packet_positions(cams_all: Sequence, packet_ids: List[int], selected_packet_ids: List[int]) -> List:
    # cameras are assumed sorted in the same order as packet_ids.
    pos_by_pid = {pid: i for i, pid in enumerate(packet_ids)}
    return [cams_all[pos_by_pid[pid]] for pid in selected_packet_ids if pid in pos_by_pid]


def run_eval(
    scene: Scene,
    pipe,
    background,
    dataset,
    args,
    stage_name: str,
    metrics_extra: Dict,
    cams_inserted: Sequence,
    cams_window: Optional[Sequence],
    eval_time_accum: float,
    t0_wall: float,
    optimization_time_override: Optional[float] = None,
    save_renders: bool = True,
) -> Tuple[Dict, float]:
    eval_t0 = time.time()
    entry: Dict = dict(metrics_extra)
    entry["stage"] = stage_name
    entry["num_gaussians"] = int(scene.gaussians.get_xyz.shape[0])

    # Always evaluate currently inserted trajectory.
    if args.eval_inserted:
        rendered_dir = None
        gt_dir = None
        if save_renders and "inserted" in args.eval_render_splits:
            rendered_dir = os.path.join(dataset.model_path, "eval_renders", stage_name, "inserted", "rendered")
            gt_dir = os.path.join(dataset.model_path, "eval_renders", "gt", "inserted") if args.save_eval_gt_once else os.path.join(dataset.model_path, "eval_renders", stage_name, "inserted", "gt")
        entry["inserted"] = evaluate_camera_list(
            cams_inserted, scene.gaussians, pipe, background, dataset, args.eval_max_views,
            rendered_dir, gt_dir, save_gt=(gt_dir is not None), overwrite_gt=(not args.save_eval_gt_once)
        )

    if cams_window is not None and args.eval_window:
        rendered_dir = None
        gt_dir = None
        if save_renders and "window" in args.eval_render_splits:
            rendered_dir = os.path.join(dataset.model_path, "eval_renders", stage_name, "window", "rendered")
            gt_dir = os.path.join(dataset.model_path, "eval_renders", "gt", "window") if args.save_eval_gt_once else os.path.join(dataset.model_path, "eval_renders", stage_name, "window", "gt")
        entry["window"] = evaluate_camera_list(
            cams_window, scene.gaussians, pipe, background, dataset, args.eval_max_views,
            rendered_dir, gt_dir, save_gt=(gt_dir is not None), overwrite_gt=(not args.save_eval_gt_once)
        )

    # At final, optionally evaluate all cameras, which should equal inserted when all packets inserted.
    if args.eval_all_train:
        cams_all = sorted_cameras(scene, "train")
        rendered_dir = None
        gt_dir = None
        if save_renders and "train" in args.eval_render_splits:
            rendered_dir = os.path.join(dataset.model_path, "eval_renders", stage_name, "train", "rendered")
            gt_dir = os.path.join(dataset.model_path, "eval_renders", "gt", "train") if args.save_eval_gt_once else os.path.join(dataset.model_path, "eval_renders", stage_name, "train", "gt")
        entry["train"] = evaluate_camera_list(
            cams_all, scene.gaussians, pipe, background, dataset, args.eval_max_views,
            rendered_dir, gt_dir, save_gt=(gt_dir is not None), overwrite_gt=(not args.save_eval_gt_once)
        )

    eval_dt = time.time() - eval_t0
    eval_time_accum += eval_dt
    wall = time.time() - t0_wall
    opt_time = optimization_time_override if optimization_time_override is not None else (wall - eval_time_accum)
    total_iters = max(1, int(entry.get("global_iteration", 0)))
    entry["optimization_time_sec"] = opt_time
    entry["optimization_time_per_iter_sec"] = opt_time / total_iters if total_iters > 0 else 0.0
    entry["eval_time_sec"] = eval_dt
    entry["eval_time_sec_accum"] = eval_time_accum
    entry["wall_time_sec"] = wall
    entry["elapsed_sec"] = opt_time
    return entry, eval_time_accum


def training_online_sliding(dataset, opt, pipe, args):
    if not os.path.exists(args.initial_ply):
        raise FileNotFoundError(f"--initial_ply not found: {args.initial_ply}")
    if args.trainable_mode not in TRAINABLE_GROUPS:
        raise ValueError(f"Unknown trainable_mode: {args.trainable_mode}")
    if args.densification_mode != "off":
        raise ValueError("This online sliding-window prototype currently supports only --densification_mode off")
    if opt.optimizer_type == "sparse_adam" and not SPARSE_ADAM_AVAILABLE:
        sys.exit("Sparse Adam requested but unavailable. Use --optimizer_type default.")

    packet_ids = parse_int_ranges(args.packet_range_spec)
    if not packet_ids:
        raise ValueError("--packet_range_spec must not be empty, e.g. 0-29")
    windows = make_packet_windows(packet_ids, args.window_size, args.stride)
    print("[windows]")
    for wid, start_pos, win_ids, new_ids in windows:
        print(f"  W{wid}: window={win_ids[0]}-{win_ids[-1]}, newly_insert={new_ids[0] if new_ids else '-'}-{new_ids[-1] if new_ids else '-'}")

    tb_writer = prepare_output_and_logger(dataset)
    wandb_run = init_wandb(args)

    # Important: keep train/test split separated.  Original 3DGS defaults eval=False,
    # which merges test cameras into train cameras for Blender/NeRF scenes.  Since our
    # prepare script often uses transforms_test.json = transforms_train.json, leaving
    # eval=False creates duplicated train cameras and breaks window indexing.
    if not args.allow_train_test_merge:
        setattr(dataset, "eval", True)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    cams_all = sorted_cameras(scene, "train", deduplicate=True)
    print(f"[cameras] unique train cameras={len(cams_all)}, packet_ids={len(packet_ids)}")
    print(f"[cameras] first names={[c.image_name for c in cams_all[:min(5, len(cams_all))]]}")
    print(f"[cameras] last names={[c.image_name for c in cams_all[-min(5, len(cams_all)):]]}")
    if len(cams_all) < len(packet_ids):
        raise ValueError(f"Not enough unique train cameras: got {len(cams_all)}, need at least {len(packet_ids)}")
    if len(cams_all) > len(packet_ids):
        print(f"[cameras] Warning: got {len(cams_all)} unique train cameras but only {len(packet_ids)} packet ids; truncating.")
    cams_all = cams_all[:len(packet_ids)]

    print(f"[init] Loading full fused Gaussian PLY for slicing: {args.initial_ply}")
    full_params = load_full_gaussian_params(args.initial_ply, dataset.sh_degree, opt.optimizer_type, dataset.train_test_exp)
    total_full = int(full_params["xyz"].shape[0])
    if args.gaussians_per_packet > 0:
        gpp = args.gaussians_per_packet
    else:
        if total_full % len(packet_ids) != 0:
            raise ValueError(f"Cannot infer gaussians_per_packet: {total_full} not divisible by {len(packet_ids)}")
        gpp = total_full // len(packet_ids)
    print(f"[init] full gaussians={total_full:,}, packets={len(packet_ids)}, gaussians_per_packet={gpp:,}")
    full_index_map = build_packet_index_map(packet_ids, gpp)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=max(1, args.iterations_per_window * len(windows)))

    metrics_log: List[Dict] = []
    eval_time_accum = 0.0
    t0_wall = time.time()
    global_iteration = 0

    inserted_packet_order: List[int] = []
    optimized_params_cpu: Optional[Dict[str, torch.Tensor]] = None

    for wid, start_pos, win_ids, new_ids in windows:
        # Insert new packets. Preserve previously optimized parameters.
        cpu_map_params, inserted_packet_order = build_cpu_map_from_inserted(
            inserted_packet_order,
            optimized_params_cpu,
            new_ids,
            full_params,
            full_index_map,
        )
        assign_params_to_gaussians(scene.gaussians, cpu_map_params)
        scene.gaussians.training_setup(opt)

        active_packet_ids = choose_active_packet_ids(inserted_packet_order, win_ids, args.active_mode)
        active_mask = active_mask_for_packets(inserted_packet_order, active_packet_ids, gpp, "cuda")
        active_count = int(active_mask.sum().item())
        cams_inserted = camera_subset_by_packet_positions(cams_all, packet_ids, inserted_packet_order)
        cams_window = camera_subset_by_packet_positions(cams_all, packet_ids, win_ids)

        if args.replay_pool == "past":
            replay_pool_ids = [pid for pid in inserted_packet_order if pid < win_ids[0]]
        else:
            replay_pool_ids = [pid for pid in inserted_packet_order if pid not in set(win_ids)]

        print(f"\n[W{wid}] inserted={inserted_packet_order[0]}-{inserted_packet_order[-1]} ({len(inserted_packet_order)} packets), "
              f"window={win_ids[0]}-{win_ids[-1]}, active_mode={args.active_mode}, active_packets={len(active_packet_ids)}, "
              f"active_gaussians={active_count:,}, total_gaussians={scene.gaussians.get_xyz.shape[0]:,}, "
              f"replay_pool={len(replay_pool_ids)}")

        if args.save_after_insert:
            scene.save(global_iteration)

        # Evaluate after insertion before optimizing this window.
        if args.eval_before_each_window:
            extra = {
                "iteration": global_iteration,
                "global_iteration": global_iteration,
                "window_id": wid,
                "window_start": win_ids[0],
                "window_end_exclusive": win_ids[-1] + 1,
                "inserted_packets": len(inserted_packet_order),
                "active_packets": len(active_packet_ids),
                "active_packet_ids": active_packet_ids,
                "active_mode": args.active_mode,
                "active_gaussians": active_count,
                "replay_mode": args.replay_mode,
                "num_replay_views": args.num_replay_views,
                "lambda_replay": args.lambda_replay,
                "lambda_iso": args.lambda_iso,
                "replay_pool_size": len(replay_pool_ids),
            }
            entry, eval_time_accum = run_eval(
                scene, pipe, background, dataset, args,
                f"window_{wid:02d}_before", extra, cams_inserted, cams_window,
                eval_time_accum, t0_wall, save_renders=args.save_eval_renders,
            )
            metrics_log.append(entry)
            log_wandb_eval(wandb_run, entry, step=global_iteration, prefix=f"window_{wid:02d}_before")

        viewpoint_stack = list(cams_window)
        ema_loss_for_log = 0.0
        ema_depth_for_log = 0.0
        pbar = tqdm(range(1, args.iterations_per_window + 1), desc=f"SW W{wid} {win_ids[0]}-{win_ids[-1]}")

        for local_iter in pbar:
            global_iteration += 1
            scene.gaussians.update_learning_rate(global_iteration)
            if (not args.freeze_sh_growth) and global_iteration % 1000 == 0:
                scene.gaussians.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = list(cams_window)
            rand_idx = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            (current_loss, Ll1, ssim_value, Ll1depth_item, image, gt_image, render_pkg) = render_mapping_loss_for_camera(
                viewpoint_cam, scene.gaussians, pipe, bg, dataset, opt, depth_l1_weight, global_iteration
            )

            replay_cams = sample_replay_cameras(cams_all, packet_ids, inserted_packet_order, win_ids, args)
            replay_loss = torch.tensor(0.0, device="cuda")
            if replay_cams:
                replay_losses = []
                replay_bg = background if not opt.random_background else bg
                for replay_cam in replay_cams:
                    r_loss, *_ = render_mapping_loss_for_camera(
                        replay_cam, scene.gaussians, pipe, replay_bg, dataset, opt, depth_l1_weight, global_iteration
                    )
                    replay_losses.append(r_loss)
                replay_loss = torch.stack(replay_losses).mean()

            iso_loss = torch.tensor(0.0, device="cuda")
            if args.lambda_iso > 0:
                iso_loss = active_isotropic_loss(scene.gaussians, active_mask)

            loss = current_loss + args.lambda_replay * replay_loss + args.lambda_iso * iso_loss

            loss.backward()
            apply_grad_mask(scene.gaussians, active_mask, args.trainable_mode)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_depth_for_log = 0.4 * Ll1depth_item + 0.6 * ema_depth_for_log
                if local_iter % 10 == 0:
                    pbar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.6f}",
                        "Replay": f"{float(replay_loss.item()):.5f}",
                        "Iso": f"{float(iso_loss.item()):.5f}",
                        "#G": f"{scene.gaussians.get_xyz.shape[0]}",
                        "active": f"{active_count}",
                    })

                if tb_writer:
                    tb_writer.add_scalar("train_loss/l1", Ll1.item(), global_iteration)
                    tb_writer.add_scalar("train_loss/total", loss.item(), global_iteration)
                    tb_writer.add_scalar("scene/total_points", scene.gaussians.get_xyz.shape[0], global_iteration)
                    tb_writer.add_scalar("scene/active_points", active_count, global_iteration)

                if wandb_run is not None and global_iteration % args.wandb_log_interval == 0:
                    sampled_psnr = psnr(image.detach(), gt_image.detach()).mean().double().item()
                    wandb_run.log({
                        "train/l1_sample": Ll1.item(),
                        "train/ssim_sample": float(ssim_value.item()) if hasattr(ssim_value, "item") else float(ssim_value),
                        "train/psnr_sample": sampled_psnr,
                        "train/loss_total": loss.item(),
                        "train/loss_current": current_loss.item(),
                        "train/loss_replay": float(replay_loss.item()),
                        "train/loss_iso": float(iso_loss.item()),
                        "train/num_replay_views": len(replay_cams),
                        "scene/num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
                        "scene/active_gaussians": active_count,
                        "window/window_id": wid,
                        "window/local_iter": local_iter,
                        "window/global_iteration": global_iteration,
                    }, step=global_iteration)

                if use_sparse_adam:
                    radii = render_pkg["radii"]
                    visible = radii > 0
                    scene.gaussians.optimizer.step(visible, radii.shape[0])
                else:
                    scene.gaussians.optimizer.step()
                scene.gaussians.optimizer.zero_grad(set_to_none=True)
                scene.gaussians.exposure_optimizer.step()
                scene.gaussians.exposure_optimizer.zero_grad(set_to_none=True)

        # Evaluate after optimizing current window.
        extra = {
            "iteration": global_iteration,
            "global_iteration": global_iteration,
            "window_id": wid,
            "window_start": win_ids[0],
            "window_end_exclusive": win_ids[-1] + 1,
            "inserted_packets": len(inserted_packet_order),
            "active_packets": len(active_packet_ids),
            "active_packet_ids": active_packet_ids,
            "active_mode": args.active_mode,
            "active_gaussians": active_count,
            "replay_mode": args.replay_mode,
            "num_replay_views": args.num_replay_views,
            "lambda_replay": args.lambda_replay,
            "lambda_iso": args.lambda_iso,
            "replay_pool_size": len(replay_pool_ids),
        }
        entry, eval_time_accum = run_eval(
            scene, pipe, background, dataset, args,
            f"window_{wid:02d}_after", extra, cams_inserted, cams_window,
            eval_time_accum, t0_wall, save_renders=args.save_eval_renders,
        )
        metrics_log.append(entry)
        log_wandb_eval(wandb_run, entry, step=global_iteration, prefix=f"window_{wid:02d}_after")
        print(f"\n[W{wid}] eval: {json.dumps(entry, indent=2)}")
        with open(os.path.join(dataset.model_path, "metrics_log.json"), "w") as f:
            json.dump(metrics_log, f, indent=2)

        if args.save_after_each_window:
            scene.save(global_iteration)

        # Commit current optimized global map to CPU before inserting the next window's packets.
        optimized_params_cpu = current_params_to_cpu(scene.gaussians)
        torch.cuda.empty_cache()

    # Final eval over all train cameras.
    final_extra = {
        "iteration": global_iteration,
        "global_iteration": global_iteration,
        "inserted_packets": len(inserted_packet_order),
        "active_packets": 0,
        "active_gaussians": 0,
    }
    old_eval_all = args.eval_all_train
    args.eval_all_train = True
    final_entry, eval_time_accum = run_eval(
        scene, pipe, background, dataset, args,
        "final", final_extra, camera_subset_by_packet_positions(cams_all, packet_ids, inserted_packet_order), None,
        eval_time_accum, t0_wall, save_renders=args.save_final_renders,
    )
    args.eval_all_train = old_eval_all
    final_entry["stage"] = "final"
    metrics_log.append(final_entry)
    log_wandb_eval(wandb_run, final_entry, step=global_iteration, prefix="final")
    with open(os.path.join(dataset.model_path, "metrics_log.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    scene.save(global_iteration)
    summary = {
        "packet_ids": packet_ids,
        "window_size": args.window_size,
        "stride": args.stride,
        "iterations_per_window": args.iterations_per_window,
        "num_windows": len(windows),
        "total_local_iterations": global_iteration,
        "gaussians_per_packet": gpp,
        "trainable_mode": args.trainable_mode,
        "active_mode": args.active_mode,
        "densification_mode": args.densification_mode,
        "final_entry": final_entry,
    }
    with open(os.path.join(dataset.model_path, "online_sliding_window_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if wandb_run is not None:
        wandb_run.finish()
    print("[done] final eval:", json.dumps(final_entry, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="Online-style sliding-window fused 3DGS refinement")
    parser.add_argument("--gs_repo", type=str, default="/home/shiyo/Desktop/gaussian-splatting", help="Path to graphdeco gaussian-splatting repo")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--initial_ply", type=str, required=True)
    parser.add_argument("--packet_range_spec", type=str, required=True, help="Example: 0-29")
    parser.add_argument("--gaussians_per_packet", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--iterations_per_window", type=int, default=400)

    parser.add_argument("--densification_mode", choices=["off"], default="off")
    parser.add_argument("--trainable_mode", choices=list(TRAINABLE_GROUPS.keys()), default="all",
                        help="Which Gaussian parameter groups are trainable for active Gaussians. Use all for the original-style isotropic-prior experiment.")
    parser.add_argument("--active_mode", choices=["window", "all_inserted"], default="all_inserted",
                        help="window = update only current-window packet Gaussians; all_inserted = update all currently inserted Gaussians.")
    parser.add_argument("--freeze_sh_growth", action="store_true")

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--disable_viewer", action="store_true", default=True)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--eval_initial", action="store_true")  # kept for CLI symmetry; first window before/after is more useful
    parser.add_argument("--eval_before_each_window", action="store_true", default=False)
    parser.add_argument("--eval_after_each_window", action="store_true", default=True)
    parser.add_argument("--eval_inserted", action="store_true", default=True)
    parser.add_argument("--eval_window", action="store_true", default=True)
    parser.add_argument("--eval_all_train", action="store_true", default=False)
    parser.add_argument("--eval_render_splits", nargs="+", choices=["inserted", "window", "train"], default=["inserted", "window"])
    parser.add_argument("--save_eval_renders", action="store_true")
    parser.add_argument("--save_final_renders", action="store_true")
    parser.add_argument("--save_eval_gt_once", action="store_true", default=True)
    parser.add_argument("--save_initial", action="store_true")
    parser.add_argument("--save_after_insert", action="store_true", default=False)
    parser.add_argument("--save_after_each_window", action="store_true")
    parser.add_argument("--eval_max_views", type=int, default=0)
    parser.add_argument("--allow_train_test_merge", action="store_true",
                        help="Allow original 3DGS behavior that merges test cameras into train cameras when eval=False. Do not use for packet-indexed sliding-window experiments.")

    # MonoGS-inspired anti-forgetting / shape-regularization options.
    # These are implemented in this external training script; original 3DGS source files are not modified.
    parser.add_argument("--replay_mode", choices=["none", "random", "anchor"], default="none",
                        help="Add past-view replay loss. random = sample past views per iteration; anchor = sample sparse deterministic anchors.")
    parser.add_argument("--replay_pool", choices=["past", "inserted_outside_window"], default="past",
                        help="past = packets before current window start; inserted_outside_window = all inserted views outside current active window.")
    parser.add_argument("--num_replay_views", type=int, default=2)
    parser.add_argument("--lambda_replay", type=float, default=0.0,
                        help="Weight for replay loss. Use 0 to disable even if replay_mode is set.")
    parser.add_argument("--replay_anchor_stride", type=int, default=5)
    parser.add_argument("--lambda_iso", type=float, default=0.0,
                        help="Weight for MonoGS-style isotropic scale regularization on active Gaussians.")

    parser.add_argument("--wandb_mode", choices=["disabled", "online", "offline"], default="disabled")
    parser.add_argument("--wandb_project", type=str, default="fused-3dgs-online-sliding-window")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_log_interval", type=int, default=10)

    args = parser.parse_args(sys.argv[1:])
    print("Online sliding-window optimization: " + args.model_path)
    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_online_sliding(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nTraining complete.")
