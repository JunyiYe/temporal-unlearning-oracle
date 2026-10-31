#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o outputs/slurm_logs/%x_%j.log
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/ptq_frozen"
export PROJECT_HOME="$REPO/ptq_frozen"
export APPTAINER_BINDPATH="$REPO"
bash scripts/apptainer_exec.sh python $REPO/scripts/diag_probe_task.py
