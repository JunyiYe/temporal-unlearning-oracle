#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH -x hawk-gpu7
#SBATCH -J t3_trainwin
#SBATCH -o outputs/slurm_logs/t3_trainwin_%A_%a.log

# T3 training-window contrast: test 2020 / val 2019 FIXED, training window
# extended from the standard 7 years (2012-2018, already in the main grid)
# to 11 years (2008-2018, includes the GFC). Isolates whether training
# exposure to a dislocation changes the quantization tax on 2020.
# Array index -> architecture. Output kept separate from the main grid.
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

CFGS=(dlinear_sp500v2_vol_opt itransformer_sp500v2_vol_tuned patchtst_sp500v2_vol_tuned \
      segrnn_sp500v2_vol_tuned timemixer_sp500v2_vol_tuned transformer_sp500v2_vol_tuned \
      tsmixer_sp500v2_vol_tuned)
cfg=${CFGS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${cfg}.yaml" \
  --test-years 2020 --seeds 1,2,3,4,5 \
  --train-years 11 \
  --output-dir outputs/v2/t3_trainwin
echo "[DONE] t3_trainwin $cfg"
