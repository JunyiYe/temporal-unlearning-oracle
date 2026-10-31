#!/bin/bash
#SBATCH -p gpu-short
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/v2/outputs/slurm_logs/%x_%j.log
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/har_deletion.py "$@"
