#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign G step 1: seeds 12-16 for both stress folds (2018, 2020) so the
# stress-fold rescue matches the main grid's 10 seeds.
# Array index -> (model, stress year).
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

MODELS=(dlinear_sp500_vol_opt itransformer_sp500_vol_tuned patchtst_sp500_vol_tuned \
        segrnn_sp500_vol_tuned timemixer_sp500_vol_tuned transformer_sp500_vol_tuned \
        tsmixer_sp500_vol_tuned)
YEARS=(2018 2020)
cfg=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
y=${YEARS[$((SLURM_ARRAY_TASK_ID % 2))]}

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${cfg}.yaml" \
  --test-years "$y" --seeds 12,13,14,15,16
echo "[DONE] stress extra seeds $cfg fold$y"
