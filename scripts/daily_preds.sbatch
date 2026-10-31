#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH -x hawk-gpu1,hawk-gpu5,hawk-gpu6
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/v2/outputs/slurm_logs/%x_%j.log
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/daily_preds.py "$@"
