#!/usr/bin/env bash
# Node-side apptainer exec wrapper. Run ON a compute node (inside sbatch or an
# existing srun allocation) — unlike srun_apptainer.sh, this does not submit
# anything, so batch jobs built on it survive SSH/login-session death.
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-fin-ts-ptq}"
PROJECT_HOME="${PROJECT_HOME:-$HOME/fin-ts-ptq}"
PROJECT_SCRATCH="${PROJECT_SCRATCH:-/scratch/$USER/$PROJECT_NAME}"
CONTAINER_PATH="${CONTAINER_PATH:-/scratch/$USER/fin-ts-ptq/pytorch_sbx_cu121}"
SIF="${SIF:-$PROJECT_SCRATCH/images/pytorch_2.4.0_cuda12.4.sif}"

export PATH="$HOME/apptainer/bin:$PATH"
if [ -d "$HOME/.conda/envs/apptainer-fuse/bin" ]; then
  export PATH="$HOME/.conda/envs/apptainer-fuse/bin:$PATH"
fi

if mkdir -p "$PROJECT_SCRATCH/tmp" "$PROJECT_SCRATCH/cache" "$PROJECT_SCRATCH/work" 2>/dev/null; then
  work_base="$PROJECT_SCRATCH"
else
  work_base="/dev/shm/$USER/$PROJECT_NAME"
  mkdir -p "$work_base/tmp" "$work_base/cache" "$work_base/work"
fi
export APPTAINER_TMPDIR="$work_base/tmp"
export APPTAINER_CACHEDIR="$work_base/cache"
export APPTAINER_WORKDIR="$work_base/work"

if [ -n "$CONTAINER_PATH" ] && [ -e "$CONTAINER_PATH" ]; then
  container="$CONTAINER_PATH"
elif [ -f "$SIF" ]; then
  container="$SIF"
else
  echo "[ERROR] No container found on $(hostname): $CONTAINER_PATH / $SIF" >&2
  exit 2
fi

echo "[INFO] Node: $(hostname)  Container: $container"
exec apptainer exec --nv \
  --workdir "$APPTAINER_WORKDIR" \
  --bind "$PROJECT_HOME:/work" \
  --pwd /work \
  "$container" \
  bash -lc '
    export PYTHONUSERBASE=/work/.local/python-userbase
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PYTHONPATH="/work/src:/work/Time-Series-Library:${PYTHONPATH:-}"
    "$@"
  ' inner "$@"
