#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%A_%a.log

# Campaign E: replicate the only-one / leave-one-out layer-sensitivity map
# across folds and seeds (the original rested on fold2024 seed7 alone).
# 4 models x 6 cells: folds {2021,2022,2023,2025} @ seed7 + fold2024 @ seeds {8,9}.
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq

MODELS=(patchtst transformer tsmixer segrnn)
CELLS=("2021 7" "2022 7" "2023 7" "2025 7" "2024 8" "2024 9")
m=${MODELS[$((SLURM_ARRAY_TASK_ID / 6))]}
read -r y s <<< "${CELLS[$((SLURM_ARRAY_TASK_ID % 6))]}"

bash scripts/apptainer_exec.sh python scripts/layer_sensitivity.py \
  --config "configs/${m}_sp500_vol_tuned.yaml" \
  --checkpoint "outputs/sweep/${m}_sp500_vol_tuned_fold${y}_seed${s}_model.pt" \
  --test-year "$y" --seed "$s" \
  --label "${m}_fold${y}_seed${s}"
echo "[DONE] layer sensitivity $m fold$y seed$s"
