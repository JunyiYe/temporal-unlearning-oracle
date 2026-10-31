#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/outputs/slurm_logs/%x_%j.log

# Reproduction check under the FROZEN dependency set (FROZEN.md):
# runs entirely from the pinned worktree; fresh out-dir forces a from-
# scratch oracle retrain.
set -euo pipefail
FROZEN=/nfs/storage1/home/ye/fin-unlearning/fin-ts-ptq-frozen
cd "$FROZEN"
export PROJECT_HOME="$FROZEN"
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/era_pilot.py "$@"
