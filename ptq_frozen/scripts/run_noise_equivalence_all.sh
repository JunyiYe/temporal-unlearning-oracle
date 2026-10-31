#!/usr/bin/env bash
# Full noise-equivalence sweep over the volatility-task checkpoints.
# Each model runs as its own srun job (sequentially by default; pass a model
# name to run just one). Full sweep per model: 5 multipliers x 8 seeds + fp32
# + 3 quant points = 44 test-set evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(${1:-dlinear tsmixer itransformer patchtst transformer timemixer})
MULTIPLIERS="${MULTIPLIERS:-0.25,0.5,1,2,4}"
NOISE_SEEDS="${NOISE_SEEDS:-8}"
QUANT_METHODS="${QUANT_METHODS:-rtn_w4,static_int8_recent,static_int4_recent}"
export TIME="${TIME:-02:00:00}"
# the apptainer sandbox is node-local; hawk-gpu1 and hawk-gpu4 have it (2026-07)
export NODELIST="${NODELIST:-hawk-gpu4}"

for model in "${MODELS[@]}"; do
  config="configs/${model}_sp500_vol_opt.yaml"
  echo "=== ${model}: ${config} ==="
  bash scripts/srun_apptainer.sh bash -lc '
    export PYTHONPATH="/work/src:/work/Time-Series-Library:${PYTHONPATH:-}"
    python scripts/noise_equivalence.py \
      --config '"$config"' \
      --multipliers '"$MULTIPLIERS"' \
      --noise-seeds '"$NOISE_SEEDS"' \
      --quant-methods '"$QUANT_METHODS"'
  '
done
