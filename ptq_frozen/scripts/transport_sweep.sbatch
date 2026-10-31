#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# External-validity transport check: Nikkei 225 panel, FROZEN S&P-tuned
# configs (no re-tuning), folds 2022 (bear) and 2024 (calm), 5 seeds.
# Array index -> (model, fold).
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

MODELS=(dlinear itransformer patchtst segrnn timemixer transformer tsmixer)
FOLDS=(2022 2024)
m=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
y=${FOLDS[$((SLURM_ARRAY_TASK_ID % 2))]}

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${m}_nikkei_vol.yaml" \
  --test-years "$y" --seeds 7,8,9,10,11 \
  --output-dir outputs/transport
echo "[DONE] transport sweep $m fold$y"
