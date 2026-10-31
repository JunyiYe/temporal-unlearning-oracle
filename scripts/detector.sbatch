#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu1,hawk-gpu6
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/outputs/slurm_logs/%x_%j.log

# Phase-2 seed: per-window loss extraction over all completed pilot cells.
# Usage: sbatch -J unl_detect scripts/detector.sbatch --config configs/transformer_sp500_vol_tuned.yaml
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/loss_gap_detector.py "$@"
