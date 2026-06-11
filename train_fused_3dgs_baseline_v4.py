#!/usr/bin/env python3
"""
global 3DGS optimization baseline
支持:
  standard 3DGS
  opt-only
  prune-only
  eval render
  optimization time / eval time / wall time 分离

读取ply文件然后全局优化
这个脚本主要作为对照组，可以知道优化的上限在哪里，但是实际online模型不能进行全局优化

命令：
python ../gaussian-splatting/train_fused_3dgs_baseline_v4.py \
  -s /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/3dgs_scene_packetcam \
  --initial_ply /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/3dgs_scene_packetcam/initial_fused_3dgs.ply \
  -m /home/shiyo/Desktop/ZipMap/outputs/api_demo_P000_0_80/exp1_standard_stochastic_2400 \
  --iterations 2400 \
  --sh_degree 3 \
  --densification_mode standard \
  --test_iterations 800 1600 2400 \
  --save_iterations 2400 \
  --eval_splits train \
  --final_render_splits train \
  --eval_render_splits train \
  --eval_initial \
  --save_final_renders \
  --save_initial \
  --eval_max_views 0 \
  --disable_viewer \
  --wandb_mode online \
  --wandb_project fused-3dgs-exp1 \
  --wandb_run_name P000_0_80_standard_stochastic_2400
"""
"""
Train / purify a fused ReSplat/MVSplat Gaussian map with the original
GraphDECO 3DGS backend.

Place this file at the root of https://github.com/graphdeco-inria/gaussian-splatting
and run it from that repo's Python environment.

This script intentionally reuses original 3DGS modules:
  - scene.Scene / scene.GaussianModel
  - gaussian_renderer.render
  - original loss, optimizer, densification, pruning mechanisms

Main difference from train.py:
  - Scene is still initialized from a Blender/NeRF-style dataset.
  - Immediately after Scene construction, GaussianModel is overwritten by --initial_ply,
    which is a full original-3DGS-style Gaussian PLY created by
    prepare_resplat_fused_3dgs_scene.py.
  - densification_mode controls whether to run standard 3DGS densify+prune,
    no densification/pruning, or prune-only purification.
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Dict, List, Optional

import torch
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


def init_wandb(args):
    """Initialize W&B only when explicitly requested."""
    if getattr(args, 'wandb_mode', 'disabled') == 'disabled':
        return None
    try:
        import wandb
    except ImportError:
        print('[wandb] wandb is not installed; skip W&B logging.')
        return None

    run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.model_path))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    return run


def log_wandb_eval(wandb_run, entry: Dict, step: int, prefix: str = 'eval') -> None:
    if wandb_run is None:
        return
    log_data = {
        # elapsed_sec is kept as a backward-compatible alias for optimization_time_sec.
        f'{prefix}/elapsed_sec': entry.get('elapsed_sec'),
        f'{prefix}/optimization_time_sec': entry.get('optimization_time_sec'),
        f'{prefix}/optimization_time_per_iter_sec': entry.get('optimization_time_per_iter_sec'),
        f'{prefix}/wall_time_sec': entry.get('wall_time_sec'),
        f'{prefix}/eval_time_sec': entry.get('eval_time_sec'),
        f'{prefix}/eval_time_sec_accum': entry.get('eval_time_sec_accum'),
        f'{prefix}/num_gaussians': entry.get('num_gaussians'),
    }
    for split, metrics in entry.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log_data[f'{prefix}/{split}_{k}'] = v
    wandb_run.log(log_data, step=step)

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


def prepare_output_and_logger(args):
    if not args.model_path:
        raise ValueError('--model_path is required for this baseline script')
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print('Tensorboard not available: not logging progress')
    return None


def save_image_tensor(path: str, image: torch.Tensor) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (torch.clamp(image.detach(), 0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def print_camera_name_stats(scene: Scene) -> None:
    from collections import Counter
    for split, cams in [('train', scene.getTrainCameras()), ('test', scene.getTestCameras())]:
        names = [c.image_name for c in cams]
        dup = [name for name, count in Counter(names).items() if count > 1]
        print(f'[camera] {split}: total={len(names)}, unique={len(set(names))}, duplicates={len(dup)}')
        if dup:
            print(f'[camera] {split} duplicate examples: {dup[:10]}')


@torch.no_grad()
def evaluate_cameras(
    scene: Scene,
    pipe,
    background,
    split: str,
    render_args_extra,
    max_views: int = 0,
    save_dir: Optional[str] = None,
    rendered_dir: Optional[str] = None,
    gt_dir: Optional[str] = None,
    save_gt: bool = True,
    overwrite_gt: bool = False,
) -> Dict[str, float]:
    cams = scene.getTrainCameras() if split == 'train' else scene.getTestCameras()

    # Recover chronological order.
    cams = sorted(cams, key=lambda c: c.image_name)

    if max_views and max_views > 0:
        cams = cams[:max_views]

    if len(cams) == 0:
        return {'num_views': 0}

    # Backward-compatible path layout for final renders.
    if save_dir is not None:
        if rendered_dir is None:
            rendered_dir = os.path.join(save_dir, split, 'rendered')
        if gt_dir is None:
            gt_dir = os.path.join(save_dir, split, 'gt')

    l1_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0

    for i, viewpoint in enumerate(cams):
        image = torch.clamp(
            render(viewpoint, scene.gaussians, pipe, background, *render_args_extra)['render'],
            0.0,
            1.0,
        )
        gt = torch.clamp(viewpoint.original_image.to('cuda'), 0.0, 1.0)

        l1_sum += l1_loss(image, gt).mean().double().item()
        psnr_sum += psnr(image, gt).mean().double().item()
        ssim_sum += ssim(image, gt).mean().double().item()

        filename = f'{i:04d}_{viewpoint.image_name}.png'
        if rendered_dir is not None:
            save_image_tensor(os.path.join(rendered_dir, filename), image)
        if save_gt and gt_dir is not None:
            gt_path = os.path.join(gt_dir, filename)
            if overwrite_gt or (not os.path.exists(gt_path)):
                save_image_tensor(gt_path, gt)

    return {
        'num_views': len(cams),
        'l1': l1_sum / len(cams),
        'psnr': psnr_sum / len(cams),
        'ssim': ssim_sum / len(cams),
    }


def prune_only(gaussians: GaussianModel, min_opacity: float, extent: float, max_screen_size: Optional[float], radii: torch.Tensor, world_scale_frac: float) -> int:
    before = int(gaussians.get_xyz.shape[0])
    prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
    if max_screen_size is not None and max_screen_size > 0:
        big_points_vs = gaussians.max_radii2D > max_screen_size
        big_points_ws = gaussians.get_scaling.max(dim=1).values > world_scale_frac * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    if prune_mask.numel() == 0 or int(prune_mask.sum().item()) == 0:
        return 0
    gaussians.tmp_radii = radii
    gaussians.prune_points(prune_mask)
    gaussians.tmp_radii = None
    torch.cuda.empty_cache()
    after = int(gaussians.get_xyz.shape[0])
    return before - after


def make_timing_entry(iteration: int, t0_wall: float, eval_time_accum: float, eval_time_current: float, num_gaussians: int, stage: str) -> Dict:
    """Create a metrics entry with optimization time separated from evaluation/render time.

    - wall_time_sec: total elapsed wall-clock time, including periodic full evaluation and image saving.
    - eval_time_sec_accum: accumulated time spent inside full-view evaluation/render export.
    - optimization_time_sec: wall_time_sec - eval_time_sec_accum. This is the value to use
      when reporting the optimization/training cost.
    - elapsed_sec: backward-compatible alias for optimization_time_sec. Older table scripts can
      keep reading elapsed_sec, but new tables should use optimization_time_sec explicitly.
    """
    wall_time = time.time() - t0_wall
    optimization_time = max(0.0, wall_time - eval_time_accum)
    return {
        'iteration': iteration,
        'stage': stage,
        'elapsed_sec': optimization_time,
        'optimization_time_sec': optimization_time,
        'optimization_time_per_iter_sec': (optimization_time / iteration) if iteration > 0 else 0.0,
        'eval_time_sec': eval_time_current,
        'eval_time_sec_accum': eval_time_accum,
        'wall_time_sec': wall_time,
        'num_gaussians': int(num_gaussians),
    }


def training_fused(dataset, opt, pipe, args):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == 'sparse_adam':
        sys.exit('Sparse Adam requested but unavailable. Install the accelerated rasterizer or use --optimizer_type default.')
    if not os.path.exists(args.initial_ply):
        raise FileNotFoundError(f'--initial_ply not found: {args.initial_ply}')

    tb_writer = prepare_output_and_logger(dataset)
    wandb_run = init_wandb(args)

    # IMPORTANT: original 3DGS merges transforms_test into train when eval=False.
    # For this baseline we usually want train/test lists to stay separate,
    # even if transforms_test is identical to transforms_train.
    if not args.merge_train_test_into_train:
        dataset.eval = True

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    print_camera_name_stats(scene)

    # Overwrite the random/COLMAP initialization with the fused ReSplat map.
    print(f"[init] Loading fused initial Gaussian map: {args.initial_ply}")
    gaussians.load_ply(args.initial_ply, dataset.train_test_exp)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda')
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device='cuda')
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device='cuda')
    print(f"[init] Gaussian count after loading fused map: {gaussians.get_xyz.shape[0]:,}")

    gaussians.training_setup(opt)
    if args.save_initial:
        scene.save(0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    use_sparse_adam = opt.optimizer_type == 'sparse_adam' and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    metrics_log: List[Dict] = []
    t0_wall = time.time()
    eval_time_accum = 0.0

    if args.eval_initial:
        torch.cuda.empty_cache()
        eval_t0 = time.time()
        initial_entry = None
        split_metrics = {}
        for split in args.eval_splits:
            init_rendered_dir = None
            init_gt_dir = None
            if args.save_eval_renders and split in args.eval_render_splits:
                init_rendered_dir = os.path.join(
                    dataset.model_path, 'eval_renders', 'iteration_0', split, 'rendered'
                )
                if args.save_eval_gt_once:
                    init_gt_dir = os.path.join(dataset.model_path, 'eval_renders', 'gt', split)
                else:
                    init_gt_dir = os.path.join(
                        dataset.model_path, 'eval_renders', 'iteration_0', split, 'gt'
                    )
            split_metrics[split] = evaluate_cameras(
                scene,
                pipe,
                background,
                split,
                (1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                args.eval_max_views,
                rendered_dir=init_rendered_dir,
                gt_dir=init_gt_dir,
                save_gt=(init_gt_dir is not None),
                overwrite_gt=(not args.save_eval_gt_once),
            )
        eval_time_current = time.time() - eval_t0
        eval_time_accum += eval_time_current
        initial_entry = make_timing_entry(
            0, t0_wall, eval_time_accum, eval_time_current, gaussians.get_xyz.shape[0], stage='initial'
        )
        initial_entry.update(split_metrics)
        metrics_log.append(initial_entry)
        log_wandb_eval(wandb_run, initial_entry, 0, prefix='eval')
        print(f"\n[ITER 0] initial eval: {json.dumps(initial_entry, indent=2)}")
        with open(os.path.join(dataset.model_path, 'metrics_log.json'), 'w') as f:
            json.dump(metrics_log, f, indent=2)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_depth_for_log = 0.0
    first_iter = 1
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc='Training fused 3DGS')

    for iteration in range(first_iter, opt.iterations + 1):
        if not args.disable_viewer:
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                    if custom_cam is not None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifier,
                                           use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)['render']
                        net_image_bytes = memoryview((torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if (not args.freeze_sh_growth) and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        bg = torch.rand((3), device='cuda') if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg['render']
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            inv_depth = render_pkg['depth']
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth_item = Ll1depth.item()
        else:
            Ll1depth_item = 0.0

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_for_log = 0.4 * Ll1depth_item + 0.6 * ema_depth_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{ema_loss_for_log:.7f}',
                    'Depth': f'{ema_depth_for_log:.7f}',
                    '#G': f'{gaussians.get_xyz.shape[0]}'
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss/l1', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss/total', loss.item(), iteration)
                tb_writer.add_scalar('scene/total_points', gaussians.get_xyz.shape[0], iteration)
            if wandb_run is not None:
                wandb_run.log({
                    'train/l1': Ll1.item(),
                    'train/ssim': float(ssim_value.item()) if hasattr(ssim_value, 'item') else float(ssim_value),
                    'train/loss_total': loss.item(),
                    'train/loss_ema': ema_loss_for_log,
                    'train/depth_loss': Ll1depth_item,
                    'scene/num_gaussians': int(gaussians.get_xyz.shape[0]),
                    'time/wall_time_sec': time.time() - t0_wall,
                    'time/eval_time_sec_accum': eval_time_accum,
                    'time/optimization_time_sec': max(0.0, time.time() - t0_wall - eval_time_accum),
                }, step=iteration)

            if iteration in args.test_iterations:
                torch.cuda.empty_cache()
                eval_t0 = time.time()
                split_metrics = {}

                for split in args.eval_splits:
                    eval_rendered_dir = None
                    eval_gt_dir = None
                    if args.save_eval_renders and split in args.eval_render_splits:
                        eval_rendered_dir = os.path.join(
                            dataset.model_path,
                            'eval_renders',
                            f'iteration_{iteration}',
                            split,
                            'rendered',
                        )
                        if args.save_eval_gt_once:
                            eval_gt_dir = os.path.join(dataset.model_path, 'eval_renders', 'gt', split)
                        else:
                            eval_gt_dir = os.path.join(
                                dataset.model_path,
                                'eval_renders',
                                f'iteration_{iteration}',
                                split,
                                'gt',
                            )

                    split_metrics[split] = evaluate_cameras(
                        scene,
                        pipe,
                        background,
                        split,
                        (1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                        args.eval_max_views,
                        rendered_dir=eval_rendered_dir,
                        gt_dir=eval_gt_dir,
                        save_gt=(eval_gt_dir is not None),
                        overwrite_gt=(not args.save_eval_gt_once),
                    )
                eval_time_current = time.time() - eval_t0
                eval_time_accum += eval_time_current
                eval_entry = make_timing_entry(
                    iteration, t0_wall, eval_time_accum, eval_time_current, gaussians.get_xyz.shape[0], stage='eval'
                )
                eval_entry.update(split_metrics)
                metrics_log.append(eval_entry)
                log_wandb_eval(wandb_run, eval_entry, iteration, prefix='eval')
                print(f"\n[ITER {iteration}] eval: {json.dumps(eval_entry, indent=2)}")
                with open(os.path.join(dataset.model_path, 'metrics_log.json'), 'w') as f:
                    json.dump(metrics_log, f, indent=2)

            # Densification / purification.
            if args.densification_mode != 'off' and iteration < opt.densify_until_iter and iteration < opt.iterations:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if args.densification_mode == 'standard':
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, args.prune_min_opacity, scene.cameras_extent, size_threshold, radii)
                elif args.densification_mode == 'prune_only':
                    if iteration > opt.densify_from_iter and iteration % args.prune_interval == 0:
                        max_screen = args.prune_max_screen_size if args.prune_max_screen_size > 0 else None
                        removed = prune_only(gaussians, args.prune_min_opacity, scene.cameras_extent, max_screen, radii, args.prune_world_scale_frac)
                        if removed > 0:
                            print(f"\n[ITER {iteration}] prune_only removed {removed:,}; remain {gaussians.get_xyz.shape[0]:,}")
                else:
                    raise ValueError(f'Unknown densification_mode: {args.densification_mode}')

                if args.opacity_reset and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f'chkpnt{iteration}.pth'))

    scene.save(opt.iterations)
    final_eval_dir = os.path.join(dataset.model_path, 'final_renders') if args.save_final_renders else None

    eval_t0 = time.time()
    split_metrics = {}

    for split in args.eval_splits:
        save_dir_for_split = None
        if final_eval_dir is not None and split in args.final_render_splits:
            save_dir_for_split = final_eval_dir

        split_metrics[split] = evaluate_cameras(
            scene,
            pipe,
            background,
            split,
            (1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
            args.eval_max_views,
            save_dir=save_dir_for_split,
            save_gt=True,
            overwrite_gt=True,
        )
    eval_time_current = time.time() - eval_t0
    eval_time_accum += eval_time_current
    final_entry = make_timing_entry(
        opt.iterations, t0_wall, eval_time_accum, eval_time_current, gaussians.get_xyz.shape[0], stage='final'
    )
    final_entry.update(split_metrics)
    metrics_log.append(final_entry)
    log_wandb_eval(wandb_run, final_entry, opt.iterations, prefix='final')
    with open(os.path.join(dataset.model_path, 'metrics_log.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"[done] final eval: {json.dumps(final_entry, indent=2)}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    parser = ArgumentParser(description='Fused ReSplat/MVSplat map optimization baseline with original 3DGS backend')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--initial_ply', type=str, required=True, help='Full 3DGS Gaussian PLY created from fused ReSplat packets')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[1000, 3000, 7000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')

    parser.add_argument('--densification_mode', choices=['standard', 'off', 'prune_only'], default='standard',
                        help='standard = original 3DGS densify_and_prune; off = pure optimization; prune_only = no new Gaussians, only pruning')
    parser.add_argument('--prune_interval', type=int, default=100)
    parser.add_argument('--prune_min_opacity', type=float, default=0.005)
    parser.add_argument('--prune_max_screen_size', type=float, default=20.0,
                        help='Only used in prune_only. <=0 disables screen-size pruning.')
    parser.add_argument('--prune_world_scale_frac', type=float, default=0.1)
    parser.add_argument('--opacity_reset', action='store_true', help='Enable original opacity reset schedule')
    parser.add_argument('--save_initial', action='store_true')
    parser.add_argument('--save_final_renders', action='store_true')
    parser.add_argument('--save_eval_renders', action='store_true',
                        help='Save rendered images whenever full evaluation is executed at --test_iterations.')
    parser.add_argument('--eval_initial', action='store_true',
                        help='Evaluate and optionally render the initial fused map before optimization.')
    parser.add_argument('--save_eval_gt_once', action='store_true', default=True,
                        help='For --save_eval_renders, save GT images once under eval_renders/gt/<split>.')
    parser.add_argument('--no_save_eval_gt_once', dest='save_eval_gt_once', action='store_false',
                        help='For --save_eval_renders, save GT inside every iteration folder instead.')
    parser.add_argument('--eval_max_views', type=int, default=0, help='0 = all views')
    parser.add_argument('--freeze_sh_growth', action='store_true')

    parser.add_argument(
    '--eval_splits',
    nargs='+',
    choices=['train', 'test'],
    default=['train'],
    help='Splits to evaluate. For observed-view reconstruction baseline, use train only.',
    )

    parser.add_argument(
        '--final_render_splits',
        nargs='+',
        choices=['train', 'test'],
        default=['train'],
        help='Splits to save final rendered images for.',
    )

    parser.add_argument(
        '--eval_render_splits',
        nargs='+',
        choices=['train', 'test'],
        default=['train'],
        help='Splits to save rendered images for when --save_eval_renders is enabled.',
    )

    parser.add_argument(
        '--merge_train_test_into_train',
        action='store_true',
        help='Mimic original 3DGS behavior: when eval=False, merge transforms_test into train. Usually keep this OFF for this baseline.',
    )

    parser.add_argument('--wandb_mode', choices=['disabled', 'online', 'offline'], default='disabled')
    parser.add_argument('--wandb_project', type=str, default='fused-3dgs-baseline')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    print('Optimizing fused map: ' + args.model_path)

    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_fused(lp.extract(args), op.extract(args), pp.extract(args), args)
    print('\nTraining complete.')
