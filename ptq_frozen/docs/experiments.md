# Running Experiments

All experiments run inside an Apptainer container on GPU nodes via `srun`. The host Python is too old to run this project.

## One-time setup (do this once per environment)

### 1. Clone Time-Series-Library

```bash
git clone https://github.com/thuml/Time-Series-Library.git Time-Series-Library
```

### 2. Build the Apptainer sandbox

```bash
srun -p gpu-short --gres=gpu:1 -c 4 --mem=16G -t 01:00:00 bash -c '
  export APPTAINER_TMPDIR=/scratch/$USER/tmp
  export APPTAINER_CACHEDIR=/scratch/$USER/cache
  mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
  apptainer build --sandbox /scratch/$USER/fin-ts-ptq/pytorch_sbx \
    docker://pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
'
```

### 3. Install dependencies (inside the container)

**Important**: the container provides its own PyTorch/CUDA. The `rm -rf` line at the end deletes any PyTorch that pip installed from PyPI, so the container's version takes over.

```bash
CONTAINER_PATH=/scratch/$USER/fin-ts-ptq/pytorch_sbx \
  bash scripts/srun_apptainer.sh bash -lc '
    set -euo pipefail
    export PYTHONUSERBASE=/work/.local/python-userbase
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PYTHONPATH="/work/src:/work/Time-Series-Library:${PYTHONPATH:-}"

    grep -v "^git+\|^torch" /work/env/requirements.txt \
      | xargs python -m pip install --user --break-system-packages

    grep -v "^#" /work/Time-Series-Library/requirements.txt | grep -v "^$" \
      | xargs python -m pip install --user --break-system-packages

    rm -rf "$PYTHONUSERBASE/lib/python3.11/site-packages/torch"*
    rm -rf "$PYTHONUSERBASE/lib/python3.11/site-packages/functorch"

    python -m pip install --user --break-system-packages --no-deps -e /work

    python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
    python -c "from models.PatchTST import Model; print('TSLib OK')"
  '
```

## Running an experiment (every time)

Dependencies are cached in `.local/`, so subsequent runs just need:

```bash
bash scripts/srun_apptainer.sh bash -lc '
  export PYTHONUSERBASE=/work/.local/python-userbase
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="/work/src:/work/Time-Series-Library:${PYTHONPATH:-}"
  python scripts/run_experiment.py --config CONFIG_NAME
'
```

The model checkpoint is saved to `outputs/<config>_model.pt` after first training.
Subsequent runs skip training and only re-run PTQ evaluation.

The container path defaults to `/scratch/$USER/fin-ts-ptq/pytorch_sbx_cu121` (set in `scripts/srun_apptainer.sh`).

## Known issue: node-local scratch

`/scratch` is per-node, not shared. If the sandbox doesn't exist on the node that `srun` lands on, the script will error out. Solutions:

1. Build the sandbox on every GPU node you might land on.
2. Or force a specific node: add `--nodelist=hawk-gpu1` to the `srun` line in `scripts/srun_apptainer.sh`.
3. Or rebuild on the fly (adds ~2 min per new node):

```bash
srun -p gpu-short --gres=gpu:1 -c 4 --mem=16G -t 00:10:00 bash -c '
  if [ ! -e /scratch/$USER/fin-ts-ptq/pytorch_sbx_cu121/opt/conda/bin/python ]; then
    export APPTAINER_TMPDIR=/scratch/$USER/tmp APPTAINER_CACHEDIR=/scratch/$USER/cache
    mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR /scratch/$USER/fin-ts-ptq
    apptainer build --sandbox /scratch/$USER/fin-ts-ptq/pytorch_sbx_cu121 \
      docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
  fi
'
```

## Available configs

| Config | Model | Purpose |
|--------|-------|---------|
| `configs/patchtst_us_large_liquid.yaml` | PatchTST | Main PTQ baseline |
| `configs/dlinear_us_large_liquid.yaml` | DLinear | Alternative time-series baseline |
| `configs/lstm_us_large_liquid.yaml` | LSTM | Sequence model baseline |
| `configs/smoke_synthetic.yaml` | PatchTST | Quick validation |

## Common issues

### Torch shows wrong version / CUDA not available

Pip may install PyTorch from PyPI into `~/.local`, shadowing the container's built-in PyTorch. Check:

```bash
python -c "import torch; print(torch.__version__, torch.__file__)"
```

If it shows a version like `2.12.1+cu130` from `/work/.local/python-userbase/...`, delete the user torch and re-run. The fix is already included in the install step above (the `rm -rf` line).

### Container not found

Set `CONTAINER_PATH` to point to your sandbox, or symlink it:

```bash
ln -s /scratch/$USER/fin-ts-ptq/pytorch_sbx /scratch/$USER/easyrobust/run/pytorch_sbx
```

Then the default path in `scripts/srun_apptainer.sh` will work.
