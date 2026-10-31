#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -x hawk-gpu1,hawk-gpu6
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/outputs/slurm_logs/%x_%j.log

# Phase-2 granularity sweep over detector NPZs (CPU-bound numpy/pandas;
# GPU allocation only because compute must not run on the login node).
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/granularity_auc.py "$@"
