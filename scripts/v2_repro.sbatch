#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o outputs/slurm_logs/%x_%j.log
# v2 reproduction dry-run from the frozen environment (fresh out-dir
# forces a from-scratch oracle retrain of one cell).
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
FROZEN="$REPO/ptq_frozen"
cd "$FROZEN"
export PROJECT_HOME="$FROZEN"
export APPTAINER_BINDPATH="$REPO"
export PYTHONPATH="$REPO/src"
bash scripts/apptainer_exec.sh python $REPO/scripts/era_pilot.py "$@"
