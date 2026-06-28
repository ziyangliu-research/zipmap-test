#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-/home/shiyo/Desktop/ZipMap/train_resplat_3dgs_incremental_local_map_v4_maintenance_visualization.py}"
SOURCE_SCENE="${SOURCE_SCENE:-/home/shiyo/Desktop/ZipMap/outputs/gt_resplat_P000_0_50/3dgs_scene_packetcam_refine_0_strict_split}"
PACKET_DIR="${PACKET_DIR:-/home/shiyo/Desktop/ZipMap/outputs/gt_resplat_P000_0_50/gaussian_packets_api/refine_0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/shiyo/Desktop/ZipMap/outputs/gt_resplat_P000_0_50}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-resplat-3dgs-incremental-maintenance-ablation}"

COMMON=(
  -s "$SOURCE_SCENE"
  --packet_dir "$PACKET_DIR"
  --packet_range_spec 0-49
  --dataset_start_index 0
  --internal_split
  --split_every 5
  --split_offset 4
  --split_index_mode local_index
  --local_map_size 10
  --sh_degree 3
  --reset_new_packet_opacity
  --new_packet_reset_max_opacity 0.01
  --eval_before_optimization
  --eval_every_train_packets 1
  --save_every_train_packets 5
  --disable_viewer
  --wandb_mode "$WANDB_MODE"
  --wandb_project "$PROJECT"
)

run_exp() {
  local name="$1"
  shift
  echo "============================================================"
  echo "Running: $name"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$SCRIPT" \
    "${COMMON[@]}" \
    -m "$OUTPUT_ROOT/$name" \
    --wandb_run_name "$name" \
    "$@"
}

TARGET="${1:-all}"

run_exp1() {
  run_exp incremental_i100_maintenance_off \
    --iterations_per_packet 100 \
    --iterations 4000 \
    --packet_maintenance_mode off
}

run_exp2() {
  run_exp incremental_i100_maintenance_at100 \
    --iterations_per_packet 100 \
    --iterations 4000 \
    --packet_maintenance_mode standard \
    --packet_maintenance_after_local_iteration 100 \
    --packet_maintenance_grad_threshold -1 \
    --packet_maintenance_min_opacity 0.005 \
    --packet_maintenance_max_screen_size -1
}

run_exp3() {
  run_exp incremental_i200_maintenance_off \
    --iterations_per_packet 200 \
    --iterations 8000 \
    --packet_maintenance_mode off
}

run_exp4() {
  run_exp incremental_i200_maintenance_at100 \
    --iterations_per_packet 200 \
    --iterations 8000 \
    --packet_maintenance_mode standard \
    --packet_maintenance_after_local_iteration 100 \
    --packet_maintenance_grad_threshold -1 \
    --packet_maintenance_min_opacity 0.005 \
    --packet_maintenance_max_screen_size -1
}

case "$TARGET" in
  all)
    run_exp1
    run_exp2
    run_exp3
    run_exp4
    ;;
  exp1) run_exp1 ;;
  exp2) run_exp2 ;;
  exp3) run_exp3 ;;
  exp4) run_exp4 ;;
  *)
    echo "Usage: $0 [all|exp1|exp2|exp3|exp4]" >&2
    exit 2
    ;;
esac
