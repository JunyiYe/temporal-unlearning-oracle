#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o outputs/tsmixer_sp500_opt.log

export PATH="/usr/local/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/anaconda3/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/nfs/storage1/home/ye/fin-ts-ptq/Time-Series-Library:/nfs/storage1/home/ye/fin-ts-ptq/src"
cd /nfs/storage1/home/ye/fin-ts-ptq
python scripts/run_experiment.py --config configs/tsmixer_sp500_opt.yaml
