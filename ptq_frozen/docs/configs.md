# Config Plan

This repo is centered on PTQ for financial time-series models. All forecasting
models are thin adapters around the official
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib)
implementations, so PTQ coverage is consistent across models (TSLib's
attention layers use explicit `nn.Linear` for Q/K/V/out, unlike
`torch.nn.MultiheadAttention`).

## Active config family: `configs/{model}_sp500_{variant}_opt.yaml`

Primary dataset: `data/sp500_kaggle_returns.csv` (~500 tickers). Models:
`dlinear`, `itransformer`, `patchtst`, `timemixer`, `transformer`, `tsmixer`
(via `src/fin_ts_ptq/models/factory.py`).

Shared settings across all three variants:

- `data.train_end/val_end/test_end`: `2022-12-31` / `2023-12-31` / `2025-12-31`
  (train ~7y, val 1y, test 2y — a 2-year test window avoids the inflated
  results seen with shorter test periods).
- `task.context_length: 64`, 11 engineered features (`value`, `ret_ma5/10/20`,
  `vol_20`, `rsi_14`, `macd`, `macd_signal`, `max_20`, `min_20`, `skew_20`).
- `task.target_transform: cross_sectional_zscore` with `target_clip: "-3,3"` —
  without clipping, the heavy-tailed target lets a handful of extreme days
  dominate the MSE loss and stalls training (val_loss stuck near the
  predict-zero baseline).

Three task variants per model:

1. **`{model}_sp500_opt.yaml`** — next-day return (`forward_period: 1`,
   `target_agg: sum`). Empirically this task has ~zero learnable signal at
   the 1-day horizon (see `scripts/significance_test.py` output) — consistent
   with near-efficient-market daily returns, not a pipeline bug.
2. **`{model}_sp500_weekly_opt.yaml`** — 5-day forward return
   (`forward_period: 5`, `target_agg: sum`). Captures a weak but
   statistically real short-term reversal effect.
3. **`{model}_sp500_vol_opt.yaml`** — 5-day forward realized volatility
   (`forward_period: 5`, `target_agg: realized_vol`, `target_log: true`).
   Volatility clustering is a much stronger, more robust signal than return
   predictability. `evaluation.portfolio.enabled: false` here — the
   long-short backtest assumes the raw target is a return stream, which
   isn't meaningful for a volatility target.

When comparing significance across `forward_period > 1` variants, use
`scripts/significance_test.py`, which applies a Newey-West/Hansen-Hodrick HAC
correction (lag = `forward_period - 1`) to the daily-IC t-test, since
overlapping forward windows make consecutive days' IC values serially
correlated — the naive Fama-MacBeth SE understates the true standard error
for these variants.

## Secondary config family: `configs/{model}_us_large_liquid_opt.yaml`

Smaller 50-ticker universe (`data/us_large_liquid_returns.csv`,
`configs/us_large_liquid_tickers.txt`). Currently de-prioritized: early runs
showed `val_loss` stuck near the predict-zero baseline, most likely because
the cross-sectional sample size (50 names/day) is too small for the model to
learn reliable cross-sectional structure — not a feature/clip/task-design
bug, since the same pipeline learns fine on the 500-ticker sp500 panel. Kept
around as a possible robustness/generalization check once the sp500 results
are in.

## Workflow scripts

- `scripts/run_sp500.sh` — sbatch entrypoint: `sbatch -J <model>_sp500[_<variant>] scripts/run_sp500.sh <model> [variant]`, where `variant` is empty (next-day return), `weekly`, or `vol`. Runs directly via the host's `/usr/local/anaconda3` Python (no Apptainer container needed for this workload — see `docs/environment.md`).
- `scripts/significance_test.py --config configs/X.yaml` — loads a trained checkpoint, re-evaluates on the test set, and reports daily-IC HAC significance plus a block-bootstrap MSE-gain-vs-zero-baseline test.
