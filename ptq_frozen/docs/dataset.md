# Dataset Overview

This project uses panel-style return data, where each row is one asset on one trading day.

Core fields:

- `date`: trading date
- `symbol`: asset identifier
- `return`: daily return target

Included datasets:

1. `data/synthetic_returns.csv`
   - Synthetic smoke-test panel
   - Built from a shared market factor plus asset-specific noise
   - A simple generative form is:

```text
r_{i,t} = alpha_i + beta_i * m_t + 0.15 * mom_t + eps_{i,t}
```

   - `m_t` is the common market factor
   - `mom_t` is short-term momentum of the market factor
   - `alpha_i` and `beta_i` are asset-specific parameters
   - `eps_{i,t}` is idiosyncratic noise
   - Date span: `2015-01-01` to `2022-12-30`
   - Total tickers: `40`
   - Fully balanced: 40 names per date

2. `data/sp500_kaggle_returns.csv`
   - Primary dataset for current experiments.
   - Real S&P 500 daily-return panel (~500 tickers, sourced from a public
     Kaggle dataset).
   - Date span: `2015-12-22` to `2025-12-19` (2514 unique trading dates).
   - `data.min_history: 1000` in the sp500 configs filters out tickers with
     short total history.
   - Used by all `configs/{model}_sp500_{,weekly_,vol_}opt.yaml` configs —
     see `docs/configs.md`.

3. `data/sp500_yf2008_returns.csv`
   - Extended panel for the fixed-window experimental design (test years 2018--2025,
     fixed 7-year training windows).
   - Same 501-ticker universe as the Kaggle panel
     (`configs/sp500_current_tickers.txt`), simple daily returns from
     dividend+split adjusted closes.
   - Date span: `2008-06-02` to `2025-12-31` (~2.06M rows). Unbalanced:
     420 names in 2009 rising to 501 by 2025 (late starters are true
     IPO/listing dates).
   - Sources: Yahoo Finance via `scripts/download_yfinance_panel.py`
     (496 tickers; validated against the Kaggle panel on the 2016--2025
     overlap: 100% row match, 494/496 with max daily diff < 1e-3).
     BK/DAY/HOLX/MMC from Tiingo adjClose (Yahoo broke these symbols in
     July 2026 when BK renamed to BNY and MMC to MRSH; Tiingo series
     validated against the panel overlap at max diff < 4e-4). CTRA is a
     hybrid: Kaggle rows for 2015-12-22..2025-12-19 (kept for
     convention consistency; Tiingo and Yahoo disagree on Coterra's
     2022+ variable-dividend ex-day amounts), Tiingo for the 2008--2015
     deep history (Cabot era, negligible dividends) and the last days
     of 2025.
   - `scripts/backfill_missing_tickers.py` re-fetches individual
     tickers and merges with an overlap-validation guard.

4. `data/us_large_liquid_returns.csv`
   - Real US large-cap liquid equity panel
   - Date span: `2010-01-05` to `2024-12-31`
   - Total tickers: `50`
   - Slightly unbalanced at the start because some names have shorter histories
   - Ticker universe is in `configs/us_large_liquid_tickers.txt`
   - The universe is a manually curated liquid large-cap list
   - Built from yfinance adjusted close prices, then converted to returns
   - Currently de-prioritized vs. sp500: the 50-name cross-section is likely
     too small for the models to learn stable cross-sectional structure
     (early runs stalled near the predict-zero baseline). Kept for a
     possible robustness check once the sp500 results are in.

Notes:

- The main task is time-series forecasting on per-asset history windows.
- Evaluation is cross-sectional at the decision layer, using IC, rank IC, and long-short backtests.
- Regime splits are used to study tail and high-volatility behavior.
- `scripts/make_synthetic_panel.py` generates the synthetic file.
- `scripts/download_yfinance_panel.py` builds the real market panel.
