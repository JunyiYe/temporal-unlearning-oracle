#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign B: seeds 12-16 for the main fold sweep, doubling the grid to
# 10 seeds x 5 folds = 50 cells per architecture. skip-existing (default)
# makes this resumable and skips the patchtst cells a previous chain finished.
# Array index -> (model, fold).
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

MODELS=(dlinear_sp500_vol_opt itransformer_sp500_vol_tuned patchtst_sp500_vol_tuned \
        segrnn_sp500_vol_tuned timemixer_sp500_vol_tuned transformer_sp500_vol_tuned \
        tsmixer_sp500_vol_tuned)
FOLDS=(2021 2022 2023 2024 2025)
cfg=${MODELS[$((SLURM_ARRAY_TASK_ID / 5))]}
y=${FOLDS[$((SLURM_ARRAY_TASK_ID % 5))]}

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${cfg}.yaml" \
  --test-years "$y" --seeds 12,13,14,15,16
echo "[DONE] extra seeds $cfg fold$y"
