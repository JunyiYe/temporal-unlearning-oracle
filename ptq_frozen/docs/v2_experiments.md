# v2 Experiment Record (ICAIF'26 rewrite)

Reproduction reference for the v2 campaign. Everything here is generated
from artifacts on disk — regenerate numbers, never hand-edit them.

## Design

- **Panel**: `data/sp500_yf2008_returns.csv` — 501 current S&P 500
  constituents, June 2008 – Dec 2025, ~2.06M rows, simple daily returns
  from adjusted closes (Yahoo Finance; BK/CTRA/DAY/HOLX/MMC via Tiingo
  backfill, see `docs/dataset.md`). Ticker list:
  `configs/sp500_current_tickers.txt`.
- **Folds**: walk-forward, FIXED 7-year training window: train
  [Y−8, Y−2], val Y−1 (early stopping), test Y. Test years 2018–2025
  (8 folds). Implemented via `run_sweep.py --train-years 7`
  (`data.train_start`, exclusive bound).
- **Seeds**: evaluation seeds **1–10** (first pass 1–5, top-up 6–10);
  tuning seeds **379, 404, 639** (drawn once via SystemRandom,
  deliberately disjoint from evaluation seeds).
- **Trainer** (identical for all models): AdamW, lr 5e-4,
  weight_decay 0.01, batch 1024, MSE loss on the cross-sectionally
  z-scored log-RV target, max 300 epochs, early stopping patience 20
  (best-epoch weights restored). IC is evaluation-only.
- **Baselines** (same folds, fixed 7y window): HAR mean daily IC 0.408,
  persistence 0.315 → `outputs/v2/har_baseline.json`
  (`scripts/har_baseline.py --train-years 7`).

## Tuning (T1)

Equal-budget grid per tunable architecture: d_model {16, 32, 64, 128,
256} × dropout {0.1, 0.3, 0.5}, depth fixed 2, d_ff = 4·d_model,
3 seeds each (45 cells/arch; 6 archs × 45 = **270 runs** — DLinear has
no capacity knobs and is not tuned). Development pseudo-fold: train
2009–2015, validate 2016. Selection: lowest mean validation MSE over
the 3 seeds. All 270 cells early-stopped (zero hit the 300-epoch cap).

| Model | d_model | dropout | d_ff | layers | params | val MSE | runner-up |
|---|---|---|---|---|---|---|---|
| TSMixer | 16 | 0.3 | 64 | 2 | 5,079 | 0.51396 | dm32/do0.5 (0.51429) |
| TimeMixer | 16 | 0.1 | 64 | 2 | 39,737 | 0.60149 | dm256/do0.5 (0.60181) |
| Transformer | 16 | 0.5 | 64 | 2 | 8,353 | 0.50961 | dm64/do0.1 (0.51524) |
| iTransformer | 16 | 0.1 | 64 | 2 | 7,649 | 0.62736 | dm32/do0.5 (0.62766) |
| PatchTST | 32 | 0.1 | 128 | 2 | 26,241 | 0.62604 | dm16/do0.1 (0.62612) |
| SegRNN | 128 | 0.1 | 512 | 2 | 100,225 | 0.52312 | dm128/do0.5 (0.52364) |
| DLinear | — (not tuned; moving_avg 25, individual) | | | | 1,430 | | |

Frozen configs: `configs/{model}_sp500v2_vol_tuned.yaml`
(DLinear: `configs/dlinear_sp500v2_vol_opt.yaml`). Model-specific fixed
settings (patch lengths, segment lengths, etc.) live in those YAMLs.

## Artifact paths

| Artifact | Path |
|---|---|
| Tuning raw cells (per arch) | `outputs/v2/tuning/{model}/tuning_results.jsonl` |
| Tuning selection + full grid scores | `outputs/v2/tuning/{model}/chosen_configs.json` |
| Main grid (T2): metrics/ckpt/preds per cell | `outputs/v2/sweep/{config}_fold{Y}_seed{S}_{metrics.json,model.pt,preds.npz}` |
| Rescue (p99/p999, val-calibrated, 10 seeds) | `outputs/v2/mechanism/{model}_rescue_folds*.json` |
| Oracle (calib = test year, 10 seeds) | `outputs/v2/mechanism/{model}_rescue_folds*_oracle.json` |
| Mixed precision (protect top layer) | `outputs/v2/mechanism_mixed/{model}_rescue_folds*.json` |
| Layer sensitivity (7 cells/arch) | `outputs/v2/mechanism/{config}_fold{Y}_seed{S}_layer_sensitivity.json` |
| Calibration-size sweep | `outputs/v2/mechanism/{model}_calibration_size.json` |
| Matched-noise falsification | `outputs/v2/mechanism/{model}_noise_equiv.json` |
| Rangeprobe (activation exceedance) | `outputs/v2/mechanism/*_rangecheck.json` |
| Bet matrix (calib-year × test-year) | `outputs/v2/bet_matrix/{model}_anchor2018_bet_matrix.csv` (+ per-seed daily dumps) |
| T3 train-window contrast (11y incl. GFC) | `outputs/v2/t3_trainwin/` |
| Daily IC dumps (rescue / oracle) | `outputs/v2/daily_attrib/`, `outputs/v2/daily_attrib_oracle/` |
| Validation-proxy clip sweep (eval on val year Y-1, same 8 clip points, same calib; job 47019, GPU+apptainer 2026-07-13) | `outputs/v2/clip_sweep_val/{model}_cells_*_evalval.csv` (+ summary jsons); protocol spot-check vs test-side CSV in `outputs/v2/clip_sweep_val_spotcheck/` (job 47020; max abs diff 0.004 on absmax, R aligned to 1pt) |
| SLURM logs | `outputs/slurm_logs/` (tuning_v2 / sweep_v2 / rescue10 / oracle10 / mech_v2 / betmx_v2 / t3_trainwin / lsrep_v2 / mixed* / clipswval) |
| Paper tables generator | `scripts/make_tables_v2.py` → `paper/icaif_v2/tables/` |

## Run counts

T1 tuning 270 · T2 main grid 560 (7×8×10) · T3 35 · everything else
eval-only on the T2 checkpoints. Every metrics JSON embeds the git
commit that produced it.
