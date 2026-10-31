# forecasting

Training and evaluation code that every experiment in this repo runs on:
panel loading and windowing, the training loop, metrics, and the seven
Time-Series-Library model adapters. It is the `fin_ts_ptq` package, taken from
an earlier project (post-training quantization of financial forecasters) and
frozen here so results do not drift with upstream changes.

- `src/fin_ts_ptq/` the package (`pip install -e forecasting`)
- `configs/*_sp500_vol_opt.yaml` base per-architecture configs; the final configs in `../configs/` are derived from them
- `scripts/apptainer_exec.sh` container wrapper used by every job
- `scripts/run_tuning.py` hyperparameter search driver
- `env/` pinned Python requirements
- `data/` symlink to `../data`

Clone [Time-Series-Library](https://github.com/thuml/Time-Series-Library) into
`forecasting/Time-Series-Library` before running anything.
