#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/v2/outputs/slurm_logs/%x_%j.log
# v2 reproduction dry-run from the frozen environment (fresh out-dir
# forces a from-scratch oracle retrain of one cell).
set -euo pipefail
FROZEN=/nfs/storage1/home/ye/fin-unlearning/v2/fin-ts-ptq-frozen
cd "$FROZEN"
export PROJECT_HOME="$FROZEN"
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/era_pilot.py "$@"
