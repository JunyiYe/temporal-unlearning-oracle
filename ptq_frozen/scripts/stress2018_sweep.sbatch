#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign C: second stress fold. Fold 2018 = train <= 2016-12-31,
# val/calibrate 2017 (calm), test 2018 (Feb Volmageddon + Q4 selloff).
# Short training window (2015-2016) is a disclosed property of the design.
# Array index -> model. Rescue eval (absmax/p99/p999) runs afterwards via
# scripts/stress2018_rescue.sbatch with a dependency on this array.
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

MODELS=(dlinear_sp500_vol_opt itransformer_sp500_vol_tuned patchtst_sp500_vol_tuned \
        segrnn_sp500_vol_tuned timemixer_sp500_vol_tuned transformer_sp500_vol_tuned \
        tsmixer_sp500_vol_tuned)
cfg=${MODELS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${cfg}.yaml" \
  --test-years 2018 --seeds 7,8,9,10,11
echo "[DONE] stress2018 sweep $cfg"
