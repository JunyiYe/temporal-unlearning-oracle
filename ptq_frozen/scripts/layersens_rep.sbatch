#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 20:00:00
#SBATCH -x hawk-gpu7
#SBATCH -J lsrep_v2
#SBATCH -o outputs/slurm_logs/lsrep_v2_%A_%a.log

# Layer-sensitivity replication: 6 extra cells per architecture
# (seeds 2,3 of fold 2024 + seed 1 of folds 2019/2021/2022/2025),
# supporting the map-stability claim. Array index -> architecture.
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

CFGS=(dlinear_sp500v2_vol_opt itransformer_sp500v2_vol_tuned patchtst_sp500v2_vol_tuned \
      segrnn_sp500v2_vol_tuned timemixer_sp500v2_vol_tuned transformer_sp500v2_vol_tuned \
      tsmixer_sp500v2_vol_tuned)
cfg=${CFGS[$SLURM_ARRAY_TASK_ID]}

run_cell () {
  local year=$1 seed=$2
  bash scripts/apptainer_exec.sh python scripts/layer_sensitivity.py \
    --config "configs/${cfg}.yaml" \
    --checkpoint "outputs/v2/sweep/${cfg}_fold${year}_seed${seed}_model.pt" \
    --test-year "$year" --train-years 7 --seed "$seed" --policy random \
    --output-dir outputs/v2/mechanism
}

run_cell 2024 2
run_cell 2024 3
run_cell 2019 1
run_cell 2021 1
run_cell 2022 1
run_cell 2025 1
echo "[DONE] lsrep_v2 $cfg"
