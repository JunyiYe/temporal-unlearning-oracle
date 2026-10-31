#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -x hawk-gpu7
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/v2/outputs/slurm_logs/%x_%j.log

# P0 fine-grained: one (arch, d_model, dropout) tuning cell, seeds 1,2,3.
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
bash scripts/apptainer_exec.sh python scripts/run_tuning.py \
  --models "$1" --d-models "$2" --dropouts "$3" \
  --train-start 2009-12-31 --train-end 2014-12-31 --val-end 2015-12-31 \
  --data-path data/sp500_yf2008_returns.csv \
  --seeds 1,2,3 \
  --output-dir "/nfs/storage1/home/ye/fin-unlearning/v2/outputs/tuning/$1/d$2_dr$3"
