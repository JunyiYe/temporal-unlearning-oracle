#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 20:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign D/F: calibration rescue (abs-max / p99 / p999, + mixed precision
# where a sensitivity map exists) for ALL 7 models on folds 2021-2025 at
# 10 seeds (7-16; seed 12-16 checkpoints from the extra-seeds campaign).
# Completes the money figure's p99 bars (dlinear/itransformer/timemixer were
# missing) and upgrades every rescue entry from 25 to 50 cells.
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

MODELS=(dlinear_sp500_vol_opt itransformer_sp500_vol_tuned patchtst_sp500_vol_tuned \
        segrnn_sp500_vol_tuned timemixer_sp500_vol_tuned transformer_sp500_vol_tuned \
        tsmixer_sp500_vol_tuned)
cfg=${MODELS[$SLURM_ARRAY_TASK_ID]}

EXTRA=()
case "$cfg" in
  patchtst_*|transformer_*|tsmixer_*) EXTRA=(--protect-auto) ;;
esac

bash scripts/apptainer_exec.sh python scripts/rescue_check.py \
  --config "configs/${cfg}.yaml" \
  --folds 2021,2022,2023,2024,2025 \
  --seeds 7,8,9,10,11,12,13,14,15,16 \
  --percentiles 0.999,0.99 ${EXTRA[@]+"${EXTRA[@]}"}
echo "[DONE] rescue 10 seeds $cfg"
