# v2 complete parameter specification

*Every fixed setting in the campaign. Anything not listed here is a
code default frozen in the repos (commits recorded per JSON).*

## Data & task (identical to v1 protocol)

| parameter | value |
|---|---|
| main panel | `data/sp500_kaggle_returns.csv` (S&P 500, 2015-12-22..2025-12-19, md5 fae3606f4364) |
| tuning panel | `data/sp500_yf2008_returns.csv` (2008-06..2025) — used ONLY for P0 |
| min_history | 1000 rows per symbol |
| features (11) | value, ret_ma5, ret_ma10, ret_ma20, vol_20, rsi_14, macd, macd_signal, max_20, min_20, skew_20 |
| context length C | 64 trading days |
| forward period F | 5 days |
| target | 5-day forward realized vol, log-transformed, cross-sectionally z-scored per day, clipped to [-3, 3] |
| feature normalization | panel-level z-score, stats from the TRAINING window only (era-excluded under \L1) |
| window assignment | by TARGET date; contexts may span backward across boundaries |

## Splits (uniform five-year rule)

| split | train (exclusive lower, inclusive upper) | val | test |
|---|---|---|---|
| tuning | (2009-12-31, 2014-12-31] | 2015 | — (test_end=val_end) |
| fold 2022 | (2015-12-31, 2020-12-31] | 2021 | 2022 |
| fold 2024 | (2017-12-31, 2022-12-31] | 2023 | 2024 |

## Seeds

| use | seeds |
|---|---|
| tuning | 1, 2, 3 |
| every experimental cell | 4–13 (uniform 10) |
| two-definitions sweep | seed 4 only (single cell, HP grid is the object) |

## Training (all reference/oracle/tuning runs; = v1 trainer)

batch 1024 · max epochs 300 · AdamW lr 5e-4 · weight decay 0.01 ·
patience 20 on val MSE · MSE loss · early-stopped best-val state.

## P0 tuning grid (equal budget)

d_model ∈ {32, 64, 128} × dropout ∈ {0.1, 0.3}; n_layers=2;
d_ff = 4×d_model; selection = argmin of mean val loss over seeds 1–3;
per-arch winner emitted to `v2/configs/{arch}_sp500_vol_v2.yaml` with
`ptq.methods=[fp32]` (no PTQ machinery in this study). Gate P0 also
records winner-vs-runner-up margin relative to seed spread.

## Deletion semantics

| level | rule | where used |
|---|---|---|
| \L0 "target" | forward days ∩ era ≠ ∅ | strictness-price cells only |
| \L1 "strict" (DEFAULT) | (context ∪ forward) ∩ era ≠ ∅ AND norm stats recomputed excluding era rows | everywhere |
| halo math | \L1 halo = C+F−1 = 68 trading days past era end | — |
| placebo | random window set, size-matched to the 2020-\L1 forget count, rng seed 90000+seed | quantity control |

## Unlearning methods (protocol-fixed, no per-arch tuning)

| method | hyperparameters |
|---|---|
| finetune | 20 epochs, lr 1e-4, wd 0.01, patience 5 (val early stop) |
| NegGrad+hinge | 5 epochs, lr 1e-4, α 0.5, margin 1.0, grad-clip 1.0, final-epoch selection |
| SCRUB | 5 epochs, msteps 3, lr 1e-4, α=γ=1.0, grad-clip 1.0, final-epoch |
| TS-Unlearn (faithful) | 5 epochs, lr 1e-4, MGDA analytic α per step, forget targets = teacher(N(0,1) inputs), retain = teacher distillation, final-epoch |
| (plain NegGrad+) | run once to document divergence; one sentence, no table row |
| two-definitions sweep | TS-Unlearn: lr ∈ {1e-5, 1e-4, 1e-3} × epochs ∈ {5, 15} |

## Evaluation quantities

| symbol | definition |
|---|---|
| IC | mean daily cross-sectional Pearson IC on the test year |
| paired tests | per-day IC differences, Newey–West lag F−1 = 4 |
| G_F (memorization gap) | MSE_F(oracle) − MSE_F(reference), forget set F |
| Δ (deletion effect) | IC_test(oracle) − IC_test(reference); positive = better without the era |
| κ (gap closed) | (MSE_F(m) − MSE_F(ref)) / (MSE_F(ora) − MSE_F(ref)) |
| forget-IC | daily IC computed on the forget windows |
| audit d′/AUC | era-level mean forget loss, reference vs oracle populations, n=10/side, within fold 2024; AUC = Φ(d′/√2); display rule: d′ primary, AUC capped at ">0.99", complete separation reported via exact permutation p = 1/C(20,10) |
| granularity grains | window / day / week / month / era; day-level NW lag 4 |
| cost | wall-clock seconds vs the cell's oracle retrain |

## Job/output conventions

Reference: `v2/outputs/sweep/{stem}_fold{F}_seed{S}_model.pt`.
Cells: `v2/outputs/pilot/{stem}_era{E}_{mode}_fold{F}_seed{S}_unlearn.json`
(+ ckpts under `pilot/ckpt/`). Detector: `v2/outputs/detector/`.
Job names: `u2t_*` (P0), `u2r_*` (P1), `u2q1_*`/`u2q2_*`/`u2q3_*`
(P2–P4). Every JSON records both repos' commits.
