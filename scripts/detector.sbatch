#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu1,hawk-gpu6
#SBATCH -o outputs/slurm_logs/%x_%j.log

# Phase-2 seed: per-window loss extraction over all completed pilot cells.
# Usage: sbatch -J unl_detect scripts/detector.sbatch --config configs/transformer_sp500_vol_tuned.yaml
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/ptq_frozen"
export PROJECT_HOME="$REPO/ptq_frozen"
export APPTAINER_BINDPATH="$REPO"
export PYTHONPATH="$REPO/src"
bash scripts/apptainer_exec.sh python $REPO/scripts/loss_gap_detector.py "$@"
