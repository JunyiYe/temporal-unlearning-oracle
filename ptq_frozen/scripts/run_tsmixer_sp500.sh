#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o outputs/tsmixer_sp500_opt.log

export PATH="/usr/local/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/anaconda3/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/Time-Series-Library:${SLURM_SUBMIT_DIR:-$PWD}/src"
cd ${SLURM_SUBMIT_DIR:-$PWD}
python scripts/run_experiment.py --config configs/tsmixer_sp500_opt.yaml
