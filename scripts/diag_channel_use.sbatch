#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o outputs/slurm_logs/%x_%j.log
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/forecasting"
export PROJECT_HOME="$REPO/forecasting"
export APPTAINER_BINDPATH="$REPO"
bash scripts/apptainer_exec.sh python $REPO/scripts/diag_channel_use.py
