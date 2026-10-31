#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -x hawk-gpu1,hawk-gpu5,hawk-gpu6
#SBATCH -o /nfs/storage1/home/ye/fin-unlearning/outputs/slurm_logs/%x_%j.log

# fin-unlearning era pilot. Runs from the fin-ts-ptq repo root so config
# data paths and the sweep checkpoint dir resolve; the fin-unlearning tree
# is bind-mounted into the container and its src/ appended to PYTHONPATH
# (apptainer_exec.sh prepends /work/src and keeps existing PYTHONPATH).
# Usage:
#   sbatch -J unl_smoke scripts/era_pilot.sbatch --config configs/... [args]
# Longer runs: sbatch -p gpu-day-long -t 08:00:00 -J unl_pilot ...
# Account discipline: <=2 unlearning jobs in flight; node exclusions keep
# our global footprint on gpu2/3/4.
set -euo pipefail
cd /nfs/storage1/home/ye/fin-ts-ptq
export APPTAINER_BINDPATH="/nfs/storage1/home/ye/fin-unlearning"
export PYTHONPATH="/nfs/storage1/home/ye/fin-unlearning/src"
bash scripts/apptainer_exec.sh python /nfs/storage1/home/ye/fin-unlearning/scripts/era_pilot.py "$@"
