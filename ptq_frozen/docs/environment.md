# Execution Environment

This repository is meant to run on the Hawk cluster. There are two supported paths:

1. **Host `/usr/local/anaconda3` Python directly** — what the current
   `configs/*_sp500_*_opt.yaml` experiments actually use (see
   `scripts/run_sp500.sh`). This anaconda install ships an older torch
   (1.13.1+cu117) that was built compatible with this cluster's glibc 2.17,
   so it runs fine for the project's current models — all thin adapters
   around classic TSLib architectures (DLinear, Transformer, PatchTST,
   iTransformer, TimeMixer, TSMixer, SegRNN) that only need basic
   `nn.Linear`/`nn.LSTM`/`nn.GRU`/attention ops. Just clone
   [Time-Series-Library](https://github.com/thuml/Time-Series-Library) into
   the project root and put both it and `src/` on `PYTHONPATH` (see
   `scripts/run_sp500.sh` for the exact env vars). No container needed.
2. **Apptainer containers** — needed if a future model requires a newer
   torch/transformers than anaconda's 1.13.1 provides. TSLib also includes
   newer foundation-model-style backbones (Chronos2, TimesFM, Moirai, Mamba,
   TimeMoE, etc.) that likely need this path; the project's current model
   set does not.

Recommended setup for the Apptainer path:

- Launch jobs with `srun` or `sbatch` on a GPU partition.
- Use the helper wrapper in `scripts/srun_apptainer.sh`.
- Keep the container image path and scratch directories on local or writable storage.
- If the cluster needs overlay support, use the Conda environment `apptainer-fuse` so `squashfuse` and `fuse-overlayfs` are available.

The wrapper script already sets:

- `APPTAINER_TMPDIR`
- `APPTAINER_CACHEDIR`
- `APPTAINER_WORKDIR`
- `PYTHONUSERBASE` inside the container

This matters for future agents and reruns: prefer the Apptainer workflow on Hawk, and avoid assuming the system Python is usable for this repo.
