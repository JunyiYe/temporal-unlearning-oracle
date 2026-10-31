# Reproduction dry-run — verdict: REPRODUCED

Date: 2026-07-20. Slurm job 47570 (gpu node, apptainer container),
runtime 8m20s.

## What was run

One full cell re-executed from the frozen environment
(`ptq_frozen`, PTQ repo pinned @be876a6, vendored
python-userbase + Time-Series-Library + yf2008 CSV), writing to a
fresh out-dir (`outputs/repro_check`) so nothing could be reused
from cache:

- config: `transformer_sp500_vol.yaml`, era 2020, strict (L1),
  fold2024, seed 4, `--train-start-years 5`
- retrained from scratch: reference, oracle, SCRUB

## Comparison vs original campaign JSONs

| model | field | original | repro |
|---|---|---|---|
| reference | test_daily_ic_mean | 0.493162 | 0.493162 |
| reference | forget_mse | 0.749344 | 0.749344 |
| reference | retain_sample_mse | 0.728429 | 0.728429 |
| oracle | test_daily_ic_mean | 0.491386 | 0.491386 |
| oracle | forget_mse | 1.138875 | 1.138875 |
| oracle | retain_sample_mse | 0.680735 | 0.680735 |
| scrub | test_daily_ic_mean | 0.501164 | 0.501164 |
| scrub | forget_mse | 0.957557 | 0.957557 |
| scrub | retain_sample_mse | 0.701773 | 0.701773 |

All 9 fields identical to 6 decimal places (tolerance was 5e-4;
observed difference 0). The pipeline is bit-reproducible from the
frozen environment; any cell reproduces the same way via
`scripts/repro.sbatch` with the cell's arguments.
