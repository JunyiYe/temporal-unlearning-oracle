#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o outputs/%x.log

# Usage: sbatch -J significance_sig_<model>[_<variant>] scripts/run_significance.sh <model> [variant]
# variant is one of: (empty)=daily return, vol=5d realized vol, momentum=21d momentum, weekly=5d forward return
# Output JSON is written to outputs/significance_sig_<model>[_<variant>].log via SBATCH -o above
export PATH="/usr/local/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/anaconda3/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/nfs/storage1/home/ye/fin-ts-ptq/Time-Series-Library:/nfs/storage1/home/ye/fin-ts-ptq/src"
cd /nfs/storage1/home/ye/fin-ts-ptq
variant_suffix="${2:+_$2}"
python scripts/significance_test.py --config "configs/${1}_sp500${variant_suffix}_opt.yaml"
