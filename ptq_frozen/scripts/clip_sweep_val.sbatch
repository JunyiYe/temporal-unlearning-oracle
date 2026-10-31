#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Validation-proxy clip sweep: same 560 frozen v2 checkpoints, same
# percentile grid as the test-year clip sweep (outputs/v2/clip_sweep*),
# but EVALUATED on the validation year (Y-1) -- everything a deployer
# sees before the test year. Purpose: measure how well validation-side
# R/L predicts the retrospective test-side R/L quoted in Sec. 7.1.
# Array index -> model. Throttle %3 (cluster rule: <=3 GPU nodes).
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

MODELS=(dlinear_sp500v2_vol_opt itransformer_sp500v2_vol_tuned \
        patchtst_sp500v2_vol_tuned segrnn_sp500v2_vol_tuned \
        timemixer_sp500v2_vol_tuned transformer_sp500v2_vol_tuned \
        tsmixer_sp500v2_vol_tuned)
cfg=${MODELS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/rescue_check.py \
  --config "configs/${cfg}.yaml" \
  --folds 2018,2019,2020,2021,2022,2023,2024,2025 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --percentiles 0.9,0.95,0.98,0.99,0.995,0.999,0.9999 \
  --train-years 7 \
  --sweep-dir outputs/v2/sweep \
  --calib-split val --eval-split val \
  --output-dir outputs/v2/clip_sweep_val
echo "[DONE] clip sweep val-eval $cfg"
