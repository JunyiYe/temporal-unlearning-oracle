#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Transport check step 2: percentile rescue on the Nikkei panel
# (submit with --dependency=afterok:<transport_sweep array>).
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

MODELS=(dlinear itransformer patchtst segrnn timemixer transformer tsmixer)
m=${MODELS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/rescue_check.py \
  --config "configs/${m}_nikkei_vol.yaml" \
  --folds 2022,2024 --seeds 7,8,9,10,11 \
  --sweep-dir outputs/transport \
  --percentiles 0.999,0.99
echo "[DONE] transport rescue $m"
