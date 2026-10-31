#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 06:00:00
#SBATCH -x hawk-gpu1,hawk-gpu5,hawk-gpu6
#SBATCH -o outputs/slurm_logs/%x_%j.log

# fin-scaling-laws params-axis sweep: one config over all folds x seeds.
# Usage: sbatch -J scal_<tag> scripts/scaling_sweep.sbatch <config.yaml> <output-dir>
# Node exclusions keep the global our-jobs footprint at <=3 nodes
# (gpu2/gpu3 carry other running projects; gpu4 is the third).
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}
bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "$1" \
  --test-years 2021,2022,2023,2024,2025 \
  --seeds 7,8,9,10,11 \
  --output-dir "$2"
