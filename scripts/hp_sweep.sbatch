#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -x hawk-gpu1,hawk-gpu6
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/outputs/slurm_logs/%x_%j.log

# Appendix method-hyperparameter sweep on one cell per config.
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/hp_sweep.py "$@"
