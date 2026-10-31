#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# ORACLE calibration diagnostic (look-ahead; not deployable): calibrate
# scales on the TEST year itself (random windows across the year) and
# re-evaluate the stress folds. Separates the cost of the temporal bet
# from the cost of quantization itself. Array index -> (model, year).
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

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
  --calib-split test --policy random \
  --dump-daily outputs/daily_attrib_oracle
echo "[DONE] oracle calib $cfg fold$y"
