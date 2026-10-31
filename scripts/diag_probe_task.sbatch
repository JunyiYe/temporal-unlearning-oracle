#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/v2/outputs/slurm_logs/%x_%j.log
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/diag_probe_task.py
