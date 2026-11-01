#!/bin/bash
#SBATCH -p gpu-short
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o outputs/slurm_logs/%x_%j.log
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/forecasting"
export PROJECT_HOME="$REPO/forecasting"
export APPTAINER_BINDPATH="$REPO"
bash scripts/apptainer_exec.sh python $REPO/scripts/har_deletion.py "$@"
