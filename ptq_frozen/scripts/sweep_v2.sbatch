#!/bin/bash
#SBATCH -p gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 20:00:00
#SBATCH -x hawk-gpu7
#SBATCH -J sweep_v2
#SBATCH -o outputs/slurm_logs/sweep_v2_%A_%a.log

# T2, v2 main grid: one architecture per submission ($1 = config stem),
# array index -> test year 2018..2025. Fixed 7-year training window
# (train [Y-8, Y-2], val Y-1, test Y) on the extended panel.
# Seeds 1-5 (first pass; top-up 6-10 on demand). Resumable (skip-existing).
# Usage: sbatch --array=0-7 scripts/sweep_v2.sbatch <config_stem>
set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

cfg=${1:?usage: sbatch --array=0-7 scripts/sweep_v2.sbatch <config_stem> [seeds]}
seeds=${2:-1,2,3,4,5}
YEARS=(2018 2019 2020 2021 2022 2023 2024 2025)
year=${YEARS[$SLURM_ARRAY_TASK_ID]}

bash scripts/apptainer_exec.sh python scripts/run_sweep.py \
  --config "configs/${cfg}.yaml" \
  --test-years "$year" --seeds "$seeds" \
  --train-years 7 \
  --output-dir outputs/v2/sweep
echo "[DONE] sweep_v2 $cfg fold$year seeds$seeds"
