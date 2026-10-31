#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o outputs/%x_opt.log

# Usage: sbatch -J <model>_sp500[_<variant>] scripts/run_sp500.sh <model> [variant]
# variant is one of: (empty)=next-day return, weekly=5-day forward return, vol=5-day forward realized vol
export PATH="/usr/local/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/anaconda3/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/nfs/storage1/home/ye/fin-ts-ptq/Time-Series-Library:/nfs/storage1/home/ye/fin-ts-ptq/src"
cd /nfs/storage1/home/ye/fin-ts-ptq
variant_suffix="${2:+_$2}"
python scripts/run_experiment.py --config "configs/${1}_sp500${variant_suffix}_opt.yaml"
