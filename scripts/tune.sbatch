#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o outputs/slurm_logs/%x_%j.log

# Era-clean tuning: yf2008, train 2010-2014 (5y), val 2015.
# Tuning seeds 1,2,3 — disjoint from evaluation seeds 4-13.
set -euo pipefail
# Submit from the repository root: `sbatch scripts/<job>.sbatch ...`
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO/forecasting"
export PROJECT_HOME="$REPO/forecasting"
export APPTAINER_BINDPATH="$REPO"
bash scripts/apptainer_exec.sh python scripts/run_tuning.py \
  --models "$1" \
  --train-start 2009-12-31 --train-end 2014-12-31 --val-end 2015-12-31 \
  --data-path data/sp500_yf2008_returns.csv \
  --seeds 1,2,3 \
  --output-dir "$REPO/outputs/tuning/$1"
