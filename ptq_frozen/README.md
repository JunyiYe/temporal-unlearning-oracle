# ptq_frozen

Frozen snapshot of the `fin_ts_ptq` package (training loop, panel/windowing,
metrics, and the seven Time-Series-Library model adapters) that every
experiment in this repo runs on. It is pinned so results do not drift with
upstream changes.

- `src/fin_ts_ptq/` the package (`pip install -e ptq_frozen`)
- `configs/*_sp500_vol_opt.yaml` base per-architecture configs; the final configs in `../configs/` are derived from them
- `scripts/apptainer_exec.sh` container wrapper used by every job
- `scripts/run_tuning.py` hyperparameter search driver
- `env/` pinned Python requirements
- `data/` symlink to `../data`

Clone [Time-Series-Library](https://github.com/thuml/Time-Series-Library) into
`ptq_frozen/Time-Series-Library` before running anything.
