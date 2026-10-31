#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Day-level damage attribution: re-evaluate the stress-fold rescues
# (eval-only, checkpoints reused) dumping per-day IC series per cell, so
# the p99-vs-absmax daily damage gap can be attributed to contaminated
# (envelope-break-exposed) vs clean days. Array index -> (model, year).
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

MODELS=(dlinear_sp500_vol_opt itransformer_sp500_vol_tuned patchtst_sp500_vol_tuned \
        segrnn_sp500_vol_tuned timemixer_sp500_vol_tuned transformer_sp500_vol_tuned \
        tsmixer_sp500_vol_tuned)
YEARS=(2018 2020)
cfg=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
y=${YEARS[$((SLURM_ARRAY_TASK_ID % 2))]}

bash scripts/apptainer_exec.sh python scripts/rescue_check.py \
  --config "configs/${cfg}.yaml" \
  --folds "$y" --seeds 7,8,9,10,11,12,13,14,15,16 \
  --percentiles 0.999,0.99 \
  --dump-daily outputs/daily_attrib
echo "[DONE] daily attrib $cfg fold$y"
