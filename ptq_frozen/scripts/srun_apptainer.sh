#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-fin-ts-ptq}"
PROJECT_HOME="${PROJECT_HOME:-$HOME/fin-ts-ptq}"
PROJECT_SCRATCH="${PROJECT_SCRATCH:-/scratch/$USER/$PROJECT_NAME}"

# Prefer an existing sandbox on Hawk. Pulling a large PyTorch SIF inside an
# interactive srun can be slow and hard to diagnose on this cluster.
CONTAINER_PATH="${CONTAINER_PATH:-/scratch/$USER/fin-ts-ptq/pytorch_sbx_cu121}"
IMAGE_URI="${IMAGE_URI:-docker://pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
SIF="${SIF:-$PROJECT_SCRATCH/images/pytorch_2.4.0_cuda12.4.sif}"

PARTITION="${PARTITION:-gpu-short}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48G}"
TIME="${TIME:-01:00:00}"
# /scratch sandboxes are node-local; pin to a node that has one (e.g.
# NODELIST=hawk-gpu4) when the default scheduling lands on a bare node.
NODELIST="${NODELIST:-}"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

srun -p "$PARTITION" --gres="gpu:$GPUS" -c "$CPUS" --mem="$MEM" -t "$TIME" \
  ${NODELIST:+--nodelist="$NODELIST"} \
  env \
    PROJECT_NAME="$PROJECT_NAME" \
    PROJECT_HOME="$PROJECT_HOME" \
    PROJECT_SCRATCH="$PROJECT_SCRATCH" \
    CONTAINER_PATH="$CONTAINER_PATH" \
    IMAGE_URI="$IMAGE_URI" \
    SIF="$SIF" \
  bash -lc '
    set -euo pipefail

    export PATH="$HOME/apptainer/bin:$PATH"
    if [ -d "$HOME/.conda/envs/apptainer-fuse/bin" ]; then
      export PATH="$HOME/.conda/envs/apptainer-fuse/bin:$PATH"
    fi

    if mkdir -p "$PROJECT_SCRATCH/tmp" "$PROJECT_SCRATCH/cache" "$PROJECT_SCRATCH/work" "$PROJECT_SCRATCH/images" 2>/dev/null; then
      work_base="$PROJECT_SCRATCH"
    else
      work_base="/dev/shm/$USER/$PROJECT_NAME"
      mkdir -p "$work_base/tmp" "$work_base/cache" "$work_base/work" "$work_base/images"
    fi

    export APPTAINER_TMPDIR="$work_base/tmp"
    export APPTAINER_CACHEDIR="$work_base/cache"
    export APPTAINER_WORKDIR="$work_base/work"

    if [ -n "$CONTAINER_PATH" ] && [ -e "$CONTAINER_PATH" ]; then
      container="$CONTAINER_PATH"
    elif [ -f "$SIF" ]; then
      container="$SIF"
    else
      echo "[ERROR] No container found. Expected existing sandbox: $CONTAINER_PATH" >&2
      echo "[ERROR] Or existing SIF: $SIF" >&2
      echo "[ERROR] Build/pull images in a dedicated prepare step before running experiments." >&2
      exit 2
    fi

    echo "[INFO] Node: $(hostname)"
    echo "[INFO] Container: $container"
    echo "[INFO] Project: $PROJECT_HOME -> /work"

    apptainer exec --nv \
      --workdir "$APPTAINER_WORKDIR" \
      --bind "$PROJECT_HOME:/work" \
      --pwd /work \
      "$container" \
      bash -lc '\''
        set -euo pipefail
        export PYTHONUSERBASE=/work/.local/python-userbase
        export PATH="$PYTHONUSERBASE/bin:$PATH"
        export PYTHONPATH="/work/src:${PYTHONPATH:-}"
        "$@"
      '\'' container-command "$@"
  ' apptainer-runner "$@"
