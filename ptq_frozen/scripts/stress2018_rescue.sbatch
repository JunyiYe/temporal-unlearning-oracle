#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign C step 2: calibration rescue on the 2018 stress fold
# (abs-max baseline + p99 + p999), mirroring the COVID-2020 analysis.
# Submit with --dependency=afterok:<stress2018_sweep array job id>.
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

MODELS=(dlinear_sp500_vol_opt itransformer_sp500_vol_tuned patchtst_sp500_vol_tuned \
        segrnn_sp500_vol_tuned timemixer_sp500_vol_tuned transformer_sp500_vol_tuned \
        tsmixer_sp500_vol_tuned)
cfg=${MODELS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/rescue_check.py \
  --config "configs/${cfg}.yaml" \
  --folds 2018 --seeds 7,8,9,10,11 \
  --percentiles 0.999,0.99
echo "[DONE] stress2018 rescue $cfg"
