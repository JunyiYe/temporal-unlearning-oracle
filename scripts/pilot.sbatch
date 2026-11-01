#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -o outputs/slurm_logs/%x_%j.log

# Era pilot. Runs from the forecasting/ directory so config data paths and
# the sweep checkpoint dir resolve; the repo tree is bind-mounted into the container and its src/ appended to PYTHONPATH
# (apptainer_exec.sh prepends /work/src and keeps existing PYTHONPATH).
# Usage:
#   sbatch -J unl_smoke scripts/pilot.sbatch --config configs/... [args]
# Longer runs: sbatch -p gpu-day-long -t 08:00:00 -J unl_pilot ...
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/forecasting"
export PROJECT_HOME="$REPO/forecasting"
export APPTAINER_BINDPATH="$REPO"
export PYTHONPATH="$REPO/src"
bash scripts/apptainer_exec.sh python $REPO/scripts/era_pilot.py "$@"
