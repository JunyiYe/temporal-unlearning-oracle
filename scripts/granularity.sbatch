#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -x hawk-gpu1,hawk-gpu6
#SBATCH -o outputs/slurm_logs/%x_%j.log

# Phase-2 granularity sweep over detector NPZs (CPU-bound numpy/pandas;
# GPU allocation only because compute must not run on the login node).
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/ptq_frozen"
export PROJECT_HOME="$REPO/ptq_frozen"
export APPTAINER_BINDPATH="$REPO"
export PYTHONPATH="$REPO/src"
bash scripts/apptainer_exec.sh python $REPO/scripts/granularity_auc.py "$@"
