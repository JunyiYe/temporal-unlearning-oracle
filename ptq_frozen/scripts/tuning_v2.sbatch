#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# T1, v2 design: equal-budget hyperparameter tuning on the extended panel.
# Tuning pseudo-fold matches the main-grid fold shape: FIXED 7-year train
# window 2009-2015 (train_start exclusive 2008-12-31), val 2016.
# Grid: d_model {16,32,64,128,256} x dropout {0.1,0.3,0.5}, depth 2,
# 3 random seeds {379,404,639} drawn once (see paper/icaif_v2 campaign log).
# Array index -> model; per-model output dir avoids jsonl write races.
# Resumable: finished cells are skipped from the jsonl.
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

MODELS=(itransformer patchtst segrnn timemixer transformer tsmixer)
model=${MODELS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/run_tuning.py \
  --models "$model" \
  --template "configs/{model}_sp500v2_vol_opt.yaml" \
  --d-models 16,32,64,128,256 \
  --dropouts 0.1,0.3,0.5 \
  --seeds 379,404,639 \
  --train-start 2008-12-31 --train-end 2015-12-31 --val-end 2016-12-31 \
  --output-dir "outputs/v2/tuning/$model"
echo "[DONE] tuning_v2 $model"
