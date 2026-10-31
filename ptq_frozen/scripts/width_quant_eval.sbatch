#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 20:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign A: evaluate the full PTQ method set on the existing width/scaling
# checkpoints (runner reuses <label>_model.pt; --no-skip-existing overwrites
# the fp32-only metrics the scaling ladder wrote, upgrading them to supersets).
# Array index -> (model, width); folds/seeds fixed inside.
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

MODELS=(itransformer patchtst transformer tsmixer)
WIDTHS=(8 16 32 64 128 256 512)
m=${MODELS[$((SLURM_ARRAY_TASK_ID / 7))]}
w=${WIDTHS[$((SLURM_ARRAY_TASK_ID % 7))]}

# fold 2024 w16-256 already has complete (tracked) metrics for
# patchtst/itransformer — skip it there; everywhere else include all folds.
if { [ "$m" = patchtst ] || [ "$m" = itransformer ]; } && [ "$w" != 8 ] && [ "$w" != 512 ]; then
  YEARS=2021,2022,2023,2025
else
  YEARS=2021,2022,2023,2024,2025
fi

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${m}_sp500_vol_w${w}.yaml" \
  --test-years "$YEARS" --seeds 7,8,9,10,11 \
  --output-dir outputs/width --no-skip-existing
echo "[DONE] width quant eval $m w$w"
