#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu1,hawk-gpu5,hawk-gpu6
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/v2/outputs/slurm_logs/%x_%j.log

# Per-day deletion effect for the mechanism figure. Runs from the
# fin-ts-ptq repo root so config data paths and the sweep checkpoint dir
# resolve; the fin-unlearning tree is bind-mounted into the container.
# Usage:
#   sbatch -J dd_tr_L1 scripts/daily_delta.sbatch \
#     --config configs/transformer_sp500_vol_v2.yaml --mode strict
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/daily_delta.py "$@"
