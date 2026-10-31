# Results master (exhibit-ready numbers)

## T1 master rows
| arch | G 2020 | G 2019 | Delta 2022-stress | best method (k, dIC) | d' 2020 (AUC) | d' 2019 (AUC) | ref IC f2024 |
|---|---|---|---|---|---|---|---|
| transformer | +0.363 | +0.064 | +0.0413±0.0136 | scrub (0.69, -0.001) | 11.1 (>0.99) | 5.2 (>0.99) | +0.489 |
| itransformer | +0.046 | +0.009 | -0.0095±0.0225 | none close (SCRUB 0.25, -0.011) | 3.8 (>0.99) | 3.1 (0.99) | +0.221 |
| segrnn | +0.390 | +0.054 | +0.0787±0.0239 | none close (hinge 0.50, +0.004) | 11.3 (>0.99) | 7.7 (>0.99) | +0.473 |
| patchtst | +0.080 | +0.023 | -0.0675±0.0237 | none reliable (tested) | 5.3 (>0.99) | 3.1 (0.99) | +0.169 |

## T2 ladder (transformer f2024 L1)
| era | G ± sd | t |
|---|---|---|
| 2018 | +0.0582±0.0146 | 12.6 |
| 2019 | +0.0643±0.0195 | 10.4 |
| 2020 | +0.3626±0.0483 | 23.7 |
| 2021 | +0.0372±0.0141 | 8.3 |
| 2022 | +0.1206±0.0385 | 9.9 |
| 2019 L0 | +0.0529 | — |
| 2020 L0 | +0.5119 | — |

## F1 exposure (fold2022 L1, n=10)
| arch | era 2020 | era 2019 |
|---|---|---|
| segrnn | +0.0787±0.0239 | -0.0307±0.0147 |
| transformer | +0.0413±0.0136 | -0.0070±0.0072 |
| itransformer | -0.0095±0.0225 | -0.0006±0.0294 |
| patchtst | -0.0675±0.0237 | +0.0099±0.0187 |

## F2 two-definitions (transformer 2020 strict seed4)
- lr1e-3_ep15: forget_ic=+0.187 test_ic=+0.456
- lr1e-3_ep5: forget_ic=+0.182 test_ic=+0.411
- lr1e-4_ep15: forget_ic=+0.420 test_ic=+0.492
- lr1e-4_ep5: forget_ic=+0.479 test_ic=+0.495
- lr1e-5_ep15: forget_ic=+0.524 test_ic=+0.496
- lr1e-5_ep5: forget_ic=+0.530 test_ic=+0.496
- reference: forget_ic=+0.533; oracle: forget_ic=+0.501

## Sec-6 granularity (naive t / NW t / win-frac means)
- itransformer: naive 8.7 / NW 4.7 / win-frac 0.531 (n=40)
- patchtst: naive 9.2 / NW 5.0 / win-frac 0.525 (n=40)
- segrnn: naive 8.5 / NW 4.5 / win-frac 0.536 (n=50)
- transformer: naive 16.1 / NW 7.9 / win-frac 0.577 (n=110)

## DEVIATIONS from the earlier design recorded before paper writing
1. The earlier design's SegRNN calm-strict "coin flip" (AUC 0.55) DID NOT replicate:
   final-design calm d' = 3.1-7.7 — every era-level deletion here is population-
   auditable (weakest AUC 0.99). RQ3 narrative becomes "d' spans 3.6x
   between calm and crisis, all detectable at era level"; the
   chance-level corner was an earlier-design (expanding/other-HP) phenomenon
   — goes to the extended version as a design-sensitivity note. The
   sample-level hopelessness (win-frac 0.52-0.58) and iid inflation
   (~2x) DO replicate.
2. Methods column: RESOLVED 2026-07-20 — segrnn/itransformer methods
   supplement run (jobs 47571-80, docs/m8_methods_supplement.md); all
   four architectures now tested at identical protocol.
3. Era-position finding (P3 gate) reshapes the fold-2022 "control"
   column: 2019 deltas are real small effects, not nulls.

## Data & compute numbers quoted in Setup (added 2026-07-20)

- Panel: 501 unique symbols, 2015-12-22..2025-12-19 (streamed count of
  data/sp500_kaggle_returns.csv; min_history=1000 filter leaves ~480
  names in the daily cross-section: n_train / (5y*252d) ~= 483).
- Window counts (from pilot JSON `data` fields, era2020 strict):
  fold2022 n_train=562,325 forget=122,422 (21.8%); fold2024
  n_train=608,526 forget=155,453 (25.5%).
- Oracle retrain wall-clock (median over cells with recorded timings,
  incl. backup_pre_m8/oracle_timings.json): iTransformer 2.7-2.9 min,
  transformer 5.0 min, PatchTST 6.7 min, SegRNN 12.1-17.7 min.
- Campaign total: 35 h of recorded model-fit time across 454 fits;
  refs (80), tuning (72 fits), and P3-overwritten oracle timings
  (~40) are unrecorded, estimated ~+20 h -> paper states "under 60
  A100-hours".

## Data-construction facts quoted in Setup (added 2026-07-20, advisor Q&A)

- Target (windowing.py:101): y = std(next 5 daily returns) (centered,
  ddof=0, daily close-to-close) -> log(x+eps) -> per-day
  cross-sectional z-score, clip +-3. NOT sqrt(sum r^2), NOT intraday.
- 11 features (features.py DEFAULT_FEATURE_COLUMNS, all computed on
  the RETURN series, no prices/volume): value, ret_ma5/10/20, vol_20,
  rsi_14, macd, macd_signal, max_20, min_20, skew_20. Panel z-score
  w/ training stats (recomputed w/o E at L1), clip +-3, per-window
  z-score, warmup rows dropped.
- Universe: configs/sp500_current_tickers.txt = 501 CURRENT
  constituents backfilled -> NOT point-in-time -> survivorship bias
  (disclosed in paper; defense: all claims are paired contrasts on
  the same panel).
- Returns: simple pct-change of dividend+split adjusted closes.
  yf2008 rebuild (Yahoo + Tiingo for BK/DAY/HOLX/MMC, CTRA hybrid)
  validated vs Kaggle panel: 100% row match on 2016-2025 overlap,
  494/496 max daily diff < 1e-3 (docs/dataset.md in ptq repo).
- Missing/halts = absent rows (stack().dropna()); windows built on
  each asset's own consecutive trading days; min_history=1000 filter;
  delisted names never enter (current-list universe).
- Sources all PUBLIC (Kaggle + Yahoo/Tiingo rebuild, scripts
  released) -> naming them is anonymity-safe; "licensed vendor"
  would be FALSE. Exact Kaggle dataset name NOT recorded in repo —
  user to recover URL from download records for camera-ready.

## Anti-coincidence evidence for the ladder (computed 2026-07-21)

Cross-fold x cross-arch replication of crisis-vs-calm gap (strict, n=10):
| arch | fold | G2020 | G2019 | ratio |
|---|---|---|---|---|
| transformer | 2022 | +0.402±0.052 (t 24.3) | +0.044±0.013 (t 10.5) | 9.1x |
| transformer | 2024 | +0.363±0.048 (t 23.7) | +0.064±0.019 (t 10.4) | 5.6x |
| segrnn | 2022 | +0.444±0.099 (t 14.2) | +0.040±0.019 (t 6.7) | 11.1x |
| segrnn | 2024 | +0.390±0.056 (t 22.2) | +0.054±0.009 (t 18.4) | 7.3x |
| patchtst | 2022 | +0.043±0.024 (t 5.6) | +0.008±0.006 (t 4.5) | 5.5x |
| patchtst | 2024 | +0.080±0.019 (t 13.5) | +0.023±0.010 (t 7.5) | 3.5x |
| itransformer | 2022 | +0.071±0.018 (t 12.6) | +0.013±0.004 (t 10.0) | 5.4x |
| itransformer | 2024 | +0.046±0.018 (t 8.3) | +0.009±0.004 (t 6.4) | 5.3x |
=> 8/8, ratio 3.5-11.1x, all t 4.5-24.3.

External anchor (model-free, from the return panel): annualized vol of
the equal-weighted market per year vs the transformer ladder gap —
2018 15.7%/0.058, 2019 12.5%/0.064, 2020 37.6%/0.363, 2021 13.5%/0.037,
2022 23.7%/0.121. Pearson r=+0.97 (p=0.006); leave-one-out r stays
+0.91..+0.99 (dropping 2020: +0.91, p=0.087). Spearman +0.70 — the
three calm years' internal rank is noise (vol 12.5-15.7%, gap
0.037-0.064), so report Pearson + the three-level dose-response, not
the fine ranking.

## SIGN TRAP in the cell JSONs (verified 2026-07-21)

`models.<m>.vs_reference.delta_ic_mean` is stored with the OPPOSITE
sign to the paper's Eq. 3. Verified on transformer era2020 strict
fold2022, n=10: direct IC(oracle)-IC(reference) = +0.0413±0.0136
(= Table 2, correct); the JSON field reads -0.0413±0.0136.
ALWAYS compute Delta as
  models['oracle']['test_daily_ic_mean'] - models['reference']['test_daily_ic_mean']
and never read vs_reference.delta_ic_mean for a signed quantity (its
|tstat| is a per-seed daily HAC statistic and is fine to use as such).
Paper t-values for Delta are ACROSS-SEED: mean/(sd/sqrt(10)); e.g.
0.0413/(0.0136/sqrt(10)) = 9.6, quoted as "t ~ 10".

## Calm-fold deletion effects (computed 2026-07-21; CORRECTS an overclaim)

Delta = IC(oracle) - IC(reference), strict, n=10, computed directly
(NOT from the sign-flipped vs_reference field):

| arch | era | fold2024 (calm test) | fold2022 (stress test) |
|---|---|---|---|
| segrnn | 2020 | +0.0105±0.0025 (t 13.1) | +0.0787±0.0239 (t 10.4) |
| segrnn | 2019 | -0.0113±0.0065 (t -5.5) | -0.0307±0.0147 (t -6.6) |
| transformer | 2020 | +0.0079±0.0070 (t 3.6) | +0.0413±0.0136 (t 9.6) |
| transformer | 2019 | -0.0081±0.0082 (t -3.1) | -0.0070±0.0072 (t -3.1) |
| itransformer | 2020/2019 | n.s. | n.s. |
| patchtst | 2020 | +0.0183±0.0330 n.s. | -0.0675±0.0237 (t -9.0) |
| patchtst | 2019 | -0.0099±0.0204 n.s. | +0.0099±0.0187 n.s. |

CONSEQUENCE: "deletion moves nothing in the calm test year" was FALSE
for the two heavy memorizers (t 3.1-13.1) and was stated in abstract,
intro and RQ1. Corrected 2026-07-21 to "at most 2% of skill, though
detectably nonzero for the heavier memorizers; stress fold moves
5-8x more". BONUS: the era-position sign structure (2020 positive,
2019 negative) replicates in BOTH folds for segrnn and transformer,
at ~1/5 the magnitude in the calm year — now stated in RQ1.

## Skill / memorization / architecture collinearity (2026-07-21, user question)

IC levels, era-2020 deletion, n=10:
| fold | arch | ref IC | ora IC | Delta | rel |
|---|---|---|---|---|---|
| 2024 | transformer | 0.489 | 0.497 | +0.0079 | +1.6% |
| 2024 | segrnn | 0.473 | 0.484 | +0.0105 | +2.2% |
| 2024 | itransformer | 0.221 | 0.218 | -0.0030 | -1.4% |
| 2024 | patchtst | 0.169 | 0.187 | +0.0183 | +10.8% (n.s.) |
| 2022 | transformer | 0.546 | 0.587 | +0.0413 | +7.6% |
| 2022 | segrnn | 0.458 | 0.536 | +0.0787 | +17.2% |
| 2022 | itransformer | 0.130 | 0.120 | -0.0095 | -7.3% (n.s.) |
| 2022 | patchtst | 0.168 | 0.101 | -0.0675 | -40.1% |

FINDING: baseline skill splits the four archs on the same line as
memorization G (Pearson r = +0.97 between ref IC and G) -> architecture,
skill and memorization are COLLINEAR in this design; cannot be
disentangled at n=4. Delta2020 vs ref IC r=+0.81, vs G r=+0.88 in the
stress fold, but n=4 => p ~ 0.12-0.19, NOT significant either way.
COUNTEREXAMPLE that rules out skill-alone: patchtst has ref IC 0.169
(f2024) vs 0.168 (f2022) — same skill — yet Delta flips +0.018 n.s. to
-0.068 sig. Regime interacts independently of baseline.
CONSEQUENCE: Table 2 caption's "No column predicts another" was an
independence claim n=4 cannot support; softened 2026-07-21 to concrete
non-redundancy examples + explicit collinearity disclosure.

## Implementation validation: are the weak architectures mis-wired? (2026-07-21)

Trigger: user observed iTransformer/PatchTST look unfit for the task. They sit
at val 0.60 vs transformer 0.50 and are BELOW the HAR baseline at test time
(HAR daily IC 0.502 in 2022, 0.392 in 2024, from the PTQ outputs har_baseline.json;
persistence 0.400/0.293).

Rescue tuning (scripts/rescue_tune.py, same era-clean split, lr x structure):
NO improvement -- itransformer 0.6017 -> 0.6016, patchtst 0.6006 -> 0.6001
across 12-19 configs each. Not a tuning-budget failure.

Channel-use probe (scripts/diag_channel_use.py, TRAINED checkpoints):
| arch | d(out)/d(chan0) | d(out)/d(chans1-10) | verdict |
|---|---|---|---|
| transformer | 1.95e-1 | 6.72e-2 | uses all 11 |
| tsmixer | 4.13e-2 | 7.03e-2 | uses all 11 |
| segrnn | 5.0e-1 | 0 | channel-independent |
| itransformer | 5.0e-1 | 3.6e-8 | effectively channel-independent |
| patchtst | 5.0e-1 | 0 | channel-independent |
=> PAPER FACTUAL ERROR: Setup says "11 backward-looking features"; that holds
   only for transformer (and tsmixer). Three of the four paper architectures
   forecast the return channel from that channel alone.

Learnability probes (scripts/diag_probe_task.py; target variance = 1.0):
| probe | transf | tsmixer | segrnn | itrans | patchtst |
|---|---|---|---|---|---|
| A recent vol, raw-scale input | 0.049 | 0.038 | 0.077 | 0.603 | 0.540 |
| B recent mean, raw-scale input | 0.041 | 0.012 | 0.068 | 0.030 | 0.026 |
| C shape vol, z-scored input (MATCHED to our pipeline) | 0.235 | 0.074 | 0.084 | 0.060 | 0.032 |

READING: adapters are correct (probe B passes everywhere). Probe A exposes a
real property of RevIN-style models (TSL PatchTST/iTransformer de-normalise by
adding back the window mean, so a scale-type target carries an irreducible
additive term) -- but our pipeline z-scores each window, making RevIN a no-op,
and probe C (matched condition) shows both recover shape information BEST of
all five. Their weak real-task fit is therefore a genuine forecasting-ability
difference, NOT an implementation defect. Note probe C does not predict
real-task rank (patchtst best on C, worst on the task), so the task's signal
is not in the standardised return channel's recent shape.

TWO HYPOTHESES I RAISED AND FALSIFIED (recorded so they are not re-run):
1. "channel independence explains the weakness" -- refuted: segrnn is
   channel-independent and fits well (0.5208).
2. "RevIN de-normalisation pollutes the output" -- refuted for our pipeline by
   probe C.

## Stress-fold era ladder — replaces the companion-study citation (2026-07-21)

Tier-1 experiment (user-requested): delete each of the stress fold's five
training years (2016-2020), test 2022, L1, n=10, transformer + segrnn.
100 cells, jobs c_* , 0 failures.

| era | transformer D | segrnn D | transformer G | segrnn G |
|---|---|---|---|---|
| 2016 | +0.0010±0.0059 (t 0.5) n.s. | -0.0024±0.0147 (t -0.5) n.s. | +0.024 | -0.003 |
| 2017 | +0.0026±0.0060 (t 1.4) n.s. | +0.0062±0.0107 (t 1.8) n.s. | +0.029 | -0.009 |
| 2018 | -0.0210±0.0090 (t -7.3) | -0.0342±0.0105 (t -10.3) | +0.047 | +0.008 |
| 2019 | -0.0070±0.0072 (t -3.1) | -0.0307±0.0147 (t -6.6) | +0.044 | +0.040 |
| 2020 | +0.0413±0.0136 (t 9.6) | +0.0787±0.0239 (t 10.4) | +0.402 | +0.444 |

THREE-LEVEL STRUCTURE, replicated across both architectures:
distant years (2016-17) inert -> years adjacent to the crisis (2018-19)
significantly NEGATIVE -> crisis year (2020) POSITIVE.
Entirely within one training window and one test year, so nothing else varies.
The paper no longer cites the companion expanding-window study for this.

L0 confirmation (distant-calm anchors): transformer era2016 +0.0009 (t 0.3),
era2017 +0.0095 (t 5.3); segrnn era2016 -0.0044 (t -1.7), era2017 +0.0120
(t 2.6). NOTE: at L0 the 2017 deletion is small-but-significant positive for
the transformer while L1 is null -- disclose if the L0 rows are reported.

## TSMixer: predictive test of the fit-quality hypothesis (2026-07-21)

TSMixer was added because the four architectures split 2-2 into "beats HAR"
and "far below HAR", leaving the exposure-sign claim resting on two weak
models. It was tuned on the same era-clean split (lr 6e-3, d64, 2 layers,
dropout 0.1; val 0.5085) and its references trained on both folds
(test IC 0.488 calm / 0.560 stress -> beats HAR 0.392 / 0.502).

PREDICTION MADE BEFORE RUNNING ITS CELLS: if exposure sign tracks fit quality,
TSMixer should GAIN from deleting the crisis year, like transformer and segrnn.

RESULT: confirmed. Delta(era2020) = +0.0048 (t 2.5) in the calm test year and
+0.0595 (t 10.1) in the stress year.

### Five-architecture profile (era 2020 deletion)
| arch | test IC 2024 | val loss | G 2020 | D stress | rel |
|---|---|---|---|---|---|
| transformer | 0.489 | 0.5036 | +0.363 | +0.0413 | +7.2% |
| tsmixer | 0.488 | 0.5085 | +0.302 | +0.0595 | +10.6% |
| segrnn | 0.476 | 0.5208 | +0.390 | +0.0787 | +15.2% |
| itransformer | 0.221 | 0.6017 | +0.046 | -0.0095 | -7.6% |
| patchtst | 0.169 | 0.6006 | +0.080 | -0.0675 | -40.2% |

3/3 architectures above the HAR baseline gain; 2/2 below it do not.

### Correlations (n=5) -- and why we do not name a single driver
Pearson(test IC, D stress) = +0.94; Pearson(val loss, D) = -0.87;
Pearson(memorization G, D) = +0.88. But Pearson(test IC, G) = +0.96: skill and
memorization remain collinear across the five architectures, so the paper
reports the axis (deployable vs not) and does not attribute the sign to skill,
memorization or architecture family individually.

### TSMixer stress-fold ladder (test 2022, L1, n=10) -- third replication
era2016 -0.0029 (t -0.7) n.s. | era2017 +0.0075 (t 2.6) | era2018 -0.0245
(t -4.6) | era2019 -0.0190 (t -5.7) | era2020 +0.0595 (t 10.1)
=> distant inert, adjacent-to-crisis negative, crisis positive: the same
three-level structure as transformer and segrnn.

## HAR control: does the crisis-deletion gain need model capacity? (2026-07-21)

Same protocol applied to HAR-RV (4-parameter OLS on log daily/weekly/monthly
realised vol), matched training windows (5y), matched test years, L0 and L1,
placebo. Script: scripts/har_deletion.py.

| fold | ref IC | delete 2020, D | neural D (deployable archs) |
|---|---|---|---|
| 2022 (stress) | +0.5085 | **-0.0074** | +0.041 / +0.060 / +0.079 |
| 2024 (calm) | +0.3964 | +0.0009 | +0.008 / +0.005 / +0.011 |

HEADLINE: on the stress fold HAR moves in the OPPOSITE direction from every
deployable neural model, and an order of magnitude less. So the gain from
deleting the crisis year is NOT a data-level fact (if 2020 were simply
anti-informative for 2022, the benchmark would gain too). HAR uses the crisis
year productively; only models with capacity to absorb the regime are better
off without it. This rules out the strongest alternative reading of RQ1.

Full stress-fold ladder (L1): 2016 +0.0085, 2017 -0.0009, 2018 -0.0003,
2019 -0.0049, 2020 -0.0074. Calm fold: all |D| <= 0.0016.

Memorization gap: HAR G(2020) = +0.035 (stress) / +0.009 (calm), about a tenth
of the transformer's +0.363, with a size-matched retain control at -0.004
(i.e. the gap is real, not a refit artifact). Even four parameters show a
nonzero gap: G measures influence, not storage.

Placebo: random deletion of the same row count gives D = +0.0001.

CAUTION recorded: HAR is deterministic, so paired daily differences have tiny
variance and EVERY cell is "significant" (the placebo itself has t = 15.9 at
D = +0.0001). Report magnitudes only. The bootstrap d' analogue (5-37) is NOT
comparable to the neural seed populations and is excluded from the paper.

## Two falsified explanations for the deletion effect (2026-07-21)

Tested because "the deletion changed the train/test distribution match" is the
first explanation any reader will reach for.

1) MARGINAL DISTRIBUTION. Wasserstein distance between the training set's
log-RV distribution and the test year's, before vs after each deletion,
correlated with Delta (3 deployable archs, n=5 eras per fold):
   fold2024 r = -0.96 (consistent with distribution matching, but every
   effect there is <= 0.009, i.e. noise scale)
   fold2022 r = +0.69 -> WRONG SIGN. Deleting 2020 moves the training
   distribution FURTHEST from the test year (+0.112 Wasserstein) and is the
   ONLY deletion that helps (+0.060). Refuted where it matters.

2) CONDITIONAL STRUCTURE. Per-year HAR coefficient vectors, distance from the
test year's, correlated with Delta: fold2022 r = +0.04 (nothing),
fold2024 r = +0.59. Also refuted on the fold with large effects.

3) ALSO NOT ROBUST: within-fold "more absorption -> more gain".
fold2022 Pearson(G, Delta) = +0.89, but drops to -0.87 when era 2020 is
removed -- the correlation is a single-point artifact. The G attribution is
supported ACROSS MODELS (6 points incl. HAR) and NOT across eras.

STANDING POSITION: the paper claims the phenomenon, the cross-model sign
attribution (HAR decisive against the skill explanation), and states that the
distribution explanation fails. It does NOT claim a mechanism for why 2020
specifically flips the sign -- n = 1 crisis.

## Factual characterisation of each year (2026-07-21) — CORRECTS the paper's labels

Equal-weighted market, from the return panel:
| year | ann. vol | max drawdown | worst day | days |ret|>2% |
|---|---|---|---|---|
| 2016 | 14.4% | -10.1% | -3.9% | 14 |
| 2017 | 7.2% | -2.4% | -1.7% | 0 |
| 2018 | 15.7% | -18.8% | -3.6% | 16 |
| 2019 | 12.5% | -6.1% | -2.9% | 7 |
| 2020 | 37.6% | -38.2% | -12.7% | 63 |
| 2021 | 13.5% | -6.4% | -2.6% | 10 |
| 2022 | 23.7% | -19.5% | -4.2% | 43 |
| 2023 | 14.1% | -11.9% | -2.2% | 8 |
| 2024 | 11.7% | -7.0% | -3.0% | 4 |

CORRECTION: the paper called 2018 a calm year. It is not -- its drawdown
(-18.8%) nearly matches 2022's (-19.5%): Volmageddon in February plus the Q4
selloff. 2016 is also not quiet (-10.1%, 14 tail days). Only 2017 is genuinely
calm (zero days with |return| > 2% in the entire year).

WHAT SURVIVES: G tracks the year's VOLATILITY LEVEL, not its drawdown --
2018 (-18.8% DD, 16% vol) has G +0.058 while 2022 (-19.5% DD, 24% vol) has
+0.121, twice as much at nearly identical drawdown. So "substitutability
(volatility level) sets the gap" holds; the "calm year" label does not.

FIXED IN PAPER: "the three calm years" -> named by measured volatility;
"distant-calm inert, recent-calm negative" -> "the earliest years inert, the
two immediately preceding the crisis negative"; per-year vol/drawdown now in
the ladder caption; Setup states folds are named by their test year's regime
while deleted years are described by measurement.

## kappa (gap-closed) is uninterpretable on small-gap eras (2026-07-21)

Ran the four methods on era 2019 as well as 2020 (calm fold, 5 archs, n=10),
to check whether the RQ2 conclusions are era-specific. They are, but not for
the reason expected: the metric breaks, not the methods.

kappa = [MSE_F(method) - MSE_F(ref)] / [MSE_F(oracle) - MSE_F(ref)].

| arch | era | denominator (oracle gap) | hinge numerator | kappa |
|---|---|---|---|---|
| transformer | 2020 | +0.363 | +0.458 | 1.3 |
| transformer | 2019 | +0.064 | +0.511 | 8.0 |
| tsmixer | 2019 | +0.057 | +0.382 | 6.8 |
| segrnn | 2019 | +0.054 | +0.356 | 6.6 |
| itransformer | 2019 | +0.009 | +0.263 | 30.2 |
| patchtst | 2019 | +0.023 | +0.291 | 12.6 |

The numerator is roughly era-invariant (0.19-0.51) because the hinge ascends
to a fixed margin; the denominator falls 6-10x on a low-volatility era. So
kappa inflates for EVERY architecture on calm eras with no change in method
behaviour.

CONSEQUENCE FOR THE PAPER: the sentence citing PatchTST's calm-era SCRUB
(0.54 +- 0.40) as evidence of unreliability was misleading -- every
architecture is noisy there (transformer 0.38+-0.17, segrnn 0.46+-0.55,
tsmixer -1.11+-4.22). Removed. RQ2 now states explicitly that methods are
reported on crisis-era deletions because gap-closed needs a large
counterfactual gap to be interpretable.

## Overnight campaign 2026-07-21/22: four regimes, L2, method era-2019

FOUR TEST REGIMES (delete era 2020, L1, n=10). Added folds test-2023
(train 2017-2021) and test-2025 (train 2019-2023) to the existing two.
| arch | 2022 (24%) | 2025 (18%) | 2023 (14%) | 2024 (12%) |
|---|---|---|---|---|
| transformer | +0.041 (t 9.6) | +0.031 (t 8.5) | +0.019 (t 4.4) | +0.008 (t 3.6) |
| tsmixer | +0.060 (t 10.1) | +0.017 (t 6.1) | +0.028 (t 4.6) | +0.005 (t 2.5) |
| segrnn | +0.079 (t 10.4) | +0.021 (t 10.5) | +0.036 (t 7.4) | +0.011 (t 13.1) |
| itransformer | -0.010 n.s. | +0.012 n.s. | +0.027 (t 5.2) | -0.003 n.s. |
| patchtst | -0.068 (t -9.0) | +0.016 n.s. | -0.005 n.s. | +0.018 n.s. |
=> 12/12 significant positive for the deployable architectures; the two
below HAR never hold a sign. Transformer is monotone in test-year
volatility; the other two are not (2025 < 2023), and the folds differ in
where 2020 sits in the window as well as in test regime, so we claim the
uniform sign and the magnitude ordering stress > calm, not strict monotonicity.

L2 (rolling-feature halo) COMPLETES THE LATTICE, fold2024:
| arch | era | L0 | L1 | L2 |
|---|---|---|---|---|
| transformer | 2019 | +0.053 | +0.064 | +0.101 |
| transformer | 2020 | +0.512 | +0.363 | +0.334 |
| tsmixer | 2019 | +0.052 | +0.057 | +0.132 |
| tsmixer | 2020 | +0.467 | +0.302 | +0.322 |
| segrnn | 2019 | +0.021 | +0.054 | +0.240 |
| segrnn | 2020 | +0.604 | +0.390 | +0.362 |
| itransformer | 2019/2020 | +0.005/+0.056 | +0.009/+0.046 | +0.015/+0.048 |
| patchtst | 2019/2020 | +0.016/+0.097 | +0.023/+0.080 | +0.037/+0.086 |
=> the L0/L1 era-dependent direction EXTENDS to L2 and is monotone in the
level: calm eras rise with strictness, the crisis era falls.

BUG CAUGHT: the first placebo batch used --modes strict, but placebo supports
target only; era_pilot rejected all 20 jobs cleanly (no data written).
Resubmitted with --modes target.

## RQ3 CORRECTED by full audit coverage (2026-07-22)

An earlier draft claimed "every era-level deletion in the study is detectable
(d' 3.1-11.3)". That rested on 10 cells: 2 eras on 1 fold. Computing d' over
ALL 100 cells (5 archs x 5 eras x 4 folds, from the existing cell JSONs --
no new runs needed) gives:

  d' range -0.4 to +14.0, median 3.8
  d' < 1.0: 7 cells (7%)   d' < 2.0: 17 cells (17%)   d' < 3.0: 35 cells (35%)
  Pearson(G, d') = +0.62, Spearman +0.64
  every cell with d' < 1 has |G| < 0.01

Hardest cells: segrnn era2017 fold2022 (-0.4), segrnn era2018 fold2023 (-0.4),
segrnn era2016 fold2022 (-0.2), tsmixer era2017 fold2022 (+0.3),
patchtst era2022 fold2025 (+0.6).

This RESTORES the earlier design's "coin-flip corner" that an earlier draft had reported as
non-replicating -- it did replicate; we had only measured the two easiest
eras. Claim rewritten in abstract, intro, RQ3, practice and conclusion:
auditability is a property of what the model absorbed, and a legitimate
deletion can be unverifiable.

## Cross-model G comparisons need normalisation (2026-07-22, user question)

G_F carries the units of the target the model was fit to. Our architectures
predict a cross-sectionally z-scored target (Var = 1 by construction); HAR
predicts log realised volatility (Var = 0.5427 on era 2020). A raw ratio of
G values across the two is therefore meaningless.

Defined in the paper: the share G_F / Var(y_E).
  HAR crisis era: 0.0353 / 0.5427 = 6.5%
  transformer stress fold: 0.4022 / 1.0 = 40.2%
  => a sixth, not a tenth. An earlier draft sentence said "a tenth of the
  transformer's" from the raw ratio (0.035/0.402); corrected.

## Target variance is NOT 1 (2026-07-22) — corrects a claim I had just written

The neural target is cross-sectionally z-scored per day (variance 1) and THEN
clipped at +-3, which removes variance:
  era 2019: 1.0000 before clipping -> 0.9191 after
  era 2020: 1.0000 before clipping -> 0.9558 after
So "Var(y_E) = 1 by construction" was false. Shares on the crisis era:
  HAR 0.0353 / 0.5427 = 6.5%
  transformer (stress fold) 0.4022 / 0.9558 = 42.1%
=> HAR's share is about a sixth of the transformer's.

DECISION: no separate symbol for the share. It is needed only where a model
with a different target (HAR) is compared; among our five architectures the
denominator is common and cancels, and across eras it varies by 4% against
ratios of 6-10x. The normalisation is stated inline where it is used.

## Full-paper claim audit (2026-07-22)

Every quantitative claim re-derived from the cell data. Verified correct:
  window counts 562,325 / 608,526 (exact); forget shares 21.8% / 25.5%
  ("22-26%" ok); L1/L0 window ratio 1.25 ("~25% more" ok); placebo
  G=+0.0169+-0.0233 and dIC=+0.0046 ("+0.017+-0.023", "+0.005" ok);
  two-definitions ref forget-IC 0.5332 / oracle 0.5014 ("0.53" / "0.50" ok);
  SCRUB-transformer kappa 0.693 dIC -0.0009; hinge-TSMixer 0.813 dIC +0.0036;
  PatchTST SCRUB 73.6+-214.7 dIC +0.272; hinge across archs 0.50/0.81/1.27/
  3.46/4.45; |dIC| <= 0.011 on segrnn/tsmixer/itransformer; G cluster
  0.037-0.064; G2022 +0.1206; cross-arch gap span 8.5x; 12/12 deployable
  architectures gain; every d'<1 cell has |G|<0.01; 7 cells d'<=1; 17 cells
  (17%) d'<2; d' range -0.4 to +14.0.

ERRORS FOUND AND FIXED:
1. "the largest effect across all five architectures is 0.018 IC, under 4% of
   baseline" in the calm test year -- FALSE. PatchTST gains 0.036 IC (21% of
   its baseline) from deleting 2022. The 0.018 figure came from considering
   era-2020 deletions only. Rewritten to bound the deployable architectures
   (<= 0.011, under 3%) and disclose PatchTST separately.
2. "crisis year is 6-10x the cluster" -- actual range 5.6-9.8x. Corrected.
3. "more than seven hundred exact retraining oracles" -- actual count is 2297
   and still growing. TO BE UPDATED once the queue drains.

METHOD NOTE FOR MYSELF: three errors this session came from writing a number
or a "by construction" claim without re-deriving it first (unit-variance
target, raw cross-scale ratio, calm-year maximum). Any number or universal
claim entering the paper gets computed from the cell data in the same commit.

## RQ3 window-level and i.i.d. numbers, full coverage (2026-07-22)

Detector rerun over all 1,000 L1 cells (5 archs x 5 eras x 4 folds x 10 seeds),
up from the 240 that the earlier numbers rested on.

Window-level win fraction: range 0.333-0.692, MEDIAN 0.538, 10-90 percentile
0.501-0.610. Earlier claim "0.52-0.58 across all architectures" understated
the spread. It tracks absorption like the population statistic does:
Pearson(G, win-frac) = +0.57; median 0.602 where G > 0.25 (n=106) vs 0.526
where G < 0.05 (n=628); best cell 0.667. Rewritten as "close to a coin flip"
with the median and the absorption dependence, not a flat range.

i.i.d. inflation: median naive/NW t ratio 1.90x (per-arch medians 1.58-2.14x),
range 0.86-2.22x. Earlier claim quoted absolute t ranges (8.5-16.1 vs 4.5-7.9)
from the small sample; those ranges are now 0.1-62.9 and 0.0-32.4, so the
paper reports the ratio instead, which is what the claim is about.

BUG CAUGHT: the first detector submission omitted --pilot-dir, so it defaulted
to the earlier pilot directory. Jobs exited 0 having processed nothing (the
config stems differ, so the glob matched no files -- had the stems matched,
old cells would have been silently written into the detector directory). Exit code 0
is not evidence of work done; check artifact counts.

## Deletion volume is not near-identical, and does not confound (2026-07-22)

The paper claimed "deletion counts are near-identical across eras". Measured
across 20 (fold, era) cells at L1: 20.1% to 27.3% of windows, a 7.2pp spread.
The pattern is positional -- the first and last era of a window lose ~20%
because the L1 halo is truncated at the window boundary, middle eras 25-27%.

But volume does NOT confound the era-position result:
  Pearson(forget_fraction, Delta) = -0.21 over 60 cells (3 deployable archs
  x 5 eras x 4 folds)
  At matched volume (20-22%): segrnn era2020 fold2022 Delta=+0.0787,
  tsmixer +0.0595, transformer +0.0413, versus transformer era2023 fold2025
  -0.0136 and segrnn era2021 fold2023 -0.0103. Same volume, opposite signs.

CONSEQUENCE: the placebo does not need to be repeated per era. It controls
volume and the act of retraining; volume has been shown not to drive the
effect, and the placebo already uses a middle-era (25.5%) window count, which
over-controls for the ~20% edge eras. Claim rewritten with the measured range
and the matched-volume counterexample.

## Two-definitions sweep at n=10 (2026-07-22) — the single seed was misleading

The sweep had been run at seed 4 only ("the HP grid is the object"). Rerun at
n=10 for the transformer, and separately for TSMixer:

| config | forget-IC | test IC |
|---|---|---|
| lr1e-5 ep5 | 0.537±0.006 | 0.492±0.004 |
| lr1e-5 ep15 | 0.523±0.009 | 0.492±0.004 |
| lr1e-4 ep5 | 0.450±0.028 | 0.488±0.007 |
| lr1e-4 ep15 | 0.369±0.037 | 0.485±0.008 |
| lr1e-3 ep5 | 0.232±0.039 | 0.423±0.049 |
| lr1e-3 ep15 | 0.199±0.060 | 0.422±0.062 |
| reference | 0.545±0.011 | 0.489±0.005 |
| oracle | 0.509±0.004 | 0.497±0.003 |

CRITICAL: at seed 4 the oracle's test IC (0.491) was BELOW the reference's
(0.496). At n=10 it is ABOVE (0.497 vs 0.489), consistent with the paper's own
finding that deleting the crisis year helps this architecture. The single-seed
figure contradicted the headline result and nobody had noticed.

Also new at n=10: the seed spread widens tenfold along the sweep (±0.004 at
the mild end to ±0.062 at the aggressive end), so configurations approaching
the privacy target are also the ones that stop behaving predictably. Added to
the paper; Figure 3 regenerated with error bars.

## Placebo comparison by fold — the calm-regime effect is not era-specific (2026-07-22)

Real era-2020 deletion vs equal-sized random deletion, same fold, same arch:
| arch | f2022 | f2025 | f2023 | f2024 |
|---|---|---|---|---|
| transformer | 7.9x | 2.8x | 15.7x | **1.7x** |
| tsmixer | 21.6x | 6.6x | 4.9x | **1.3x** |
| segrnn | opposite sign | opposite sign | opposite sign | 8.9x |

In the calmest test year the real deletion moves the transformer and TSMixer
only 1.3-1.7x more than deleting the same number of random windows. So the
small calm-regime effect cannot be attributed to the era's identity for two of
three deployable architectures. The stressed regimes are unambiguous
(2.8-21.6x, or opposite sign for segrnn where the placebo is negative).

CONSEQUENCE: "deletion is nearly inert in calm regimes" stands; "the residue
is detectably nonzero for the heavier memorizers" was tested against zero, not
against the placebo, and does not survive the right comparison. RQ1 now says
so explicitly.

## Final claim sweep (2026-07-22) — ninth correction

"Neither architecture below the HAR baseline gains" was FALSE: iTransformer's
era-2020 deletion at fold2023 is +0.0267 (t = 5.2), significantly positive.
Across four regimes iTransformer is null three times and positive once;
PatchTST is negative once (-0.068, t = -9.0) and null otherwise. Rewritten
everywhere as "twelve consistent gains above the line against one isolated
gain below it".

Verified unchanged in the same sweep: Table 2 values (G 2020 per arch, stress
Delta per arch) all match the current data; the "5-12x stress-to-calm
multiplier" is 5.2x / 12.4x / 7.5x for transformer / tsmixer / segrnn.
Practice section's counts updated to the four-regime numbers.

## Full read-through of the compiled paper (2026-08-07)

First end-to-end read of the 8-page PDF since the day's surgical edits.
Every number below was re-measured on the 3,200-cell grid with
`scripts/verify_paper_readthrough.py` (read-only; reproduces from
`outputs/pilot`).

Corrected (was wrong):

1. **Abstract contradicted the body.** "neither architecture below that
   baseline gains" was the pre-correction-9 claim; iTransformer's
   era-2020 deletion at fold2023 is +0.0267 (across-seed t = 5.2).
   -> "the two below that baseline hold no consistent sign".
2. **Compute total off by ~10x.** "under 60 A100-hours" was written when
   the campaign was 454 fits. sacct over the 1,316 campaign jobs
   (1 GPU each, gpu-short + gpu-day-long) totals **600 GPU-hours**.
   Sanity check the old number failed: 3,200 retrains x 3 min minimum
   = 155 h. -> "about 600 A100-hours".
3. **Broken sentence** in RQ1: "...is era-specific is not. in the 2022
   bear market..." (lower-case sentence start left by an insertion).
   The PatchTST parenthetical was also stranded after the bear-market
   clause while qualifying the calm-market one; moved.
4. **Era-position range wrong.** "the two years preceding the crisis are
   significantly negative (-1.9 to -3.4 points)". Measured, stress fold,
   three deployable archs: delete-2018 -2.10/-2.45/-3.42, delete-2019
   -0.70/-1.90/-3.07. Range is **-0.7 to -3.4**. The companion
   "|t| = 3.1 to 10.3" is correct (across-seed t 3.06 to 10.32).
5. **"at a fifth of the magnitude"** held only for the crisis year.
   calm/stress ratios: era2020 0.19/0.08/0.13; era2019 1.16/0.43/0.37 --
   the transformer's 2019 loss is *larger* in the calm test year.
   Rewritten to say only the gain shrinks.
6. **Table 2 vs Tables 3/4 disagreed** on two cells whose third decimal
   is a 5: TSMixer stress Delta 0.0595 (0.060 vs 0.059) and PatchTST
   -0.0675 (-0.068 vs -0.067). Harmonized on 0.059 / -0.067; prose
   "+6.0" -> "+5.9", "6.8 of its 16.8" -> "6.7 of its 16.8".
7. **HAR baseline was rounded down, model IC rounded up.** Measured
   reference test IC: transformer 0.489/0.567, TSMixer 0.487/0.557,
   SegRNN 0.476/**0.511**, iTransformer 0.221/0.125, PatchTST
   0.169/0.168. HAR: 0.396 (calm 2024), **0.5085** (stress 2022).
   The table printed SegRNN 0.52 and HAR 0.50, turning a **0.003**
   margin into an apparent 0.02. Now 3 decimals, and the caption
   discloses the margin. The split itself is safe: all three clear HAR
   in all four test years (HAR 2023 0.3841, 2025 0.4122), by
   +0.003/+0.055/+0.080/+0.040 for SegRNN, comfortably for the other two.
8. **kappa denominator sentence was self-inconsistent**: "6-10x smaller"
   next to "falls from 0.36 to 0.009" (= 40x). Measured per
   (arch, fold): crisis G / mean non-crisis G spans 2.6x to 47.9x,
   median ~7x; crisis G 0.036-0.444; the lowest non-crisis cell is
   -0.009 (segrnn/fold2022/era2017). -> "5 to 10x on a low-volatility
   era and up to 48x at the extreme; from 0.363 to below 0.01".
9. **Placebo was not size-matched to the paper's default.** The placebo
   removes 124,362 windows = exactly the **L0** count (20.4%); the L1
   deletion removes 155,453 (25.5%). The published ratios (1.3-1.7x
   calm, 2.8-22x stress) compared an L1 deletion against an L0-sized
   placebo. Like-for-like at L0: **1.4-1.8x** calm, **2.5-13x** stress.
   Conclusion unchanged; the paper now states the comparison is at L0.
10. **Missing / wrong setup facts.** TSMixer was absent from the
    channel-mixing list (it mixes all 11 channels) and from the
    hyperparameter paragraph, and it is the one architecture *not* at
    lr 5e-4 (it uses **6e-3**, adopted in the rescue round).
11. Figure 1 caption said "Both collapse onto L2" with a singular
    subject (L3) -> "It collapses onto L2".
12. Table 2 caption promised an AUC column the table does not have.
13. "3,100 oracles" -> **3,200** (outputs/pilot holds 3,200 cells, each
    with one from-scratch retrain: 3,000 main grid + 200 placebo).
    The 1,050 files in outputs/detector re-read those models and are
    NOT additional retrains.
14. Conclusion said the extended version adds "three further
    architectures"; the pool is seven and five are reported -> **two**
    (TimeMixer, DLinear).
15. Abstract/intro said calm-regime effects are "at most 2%" / "under
    2%" of skill; max deployable |Delta| in the calm fold is SegRNN
    -0.0113 on 0.476 = 2.4% -> "under 3%", matching RQ1.
16. Practice said the deletion is "inert for a fourth" citing Table 2
    (a single regime) as if it were architecture-level; scoped to the
    bear market. "Recently ingested data carries forward value, so
    obligations on recent eras are the expensive ones" ignored that the
    *most* recent era in the stress fold (2020) is a windfall; rewritten.

Verified unchanged (checked, correct):

- "t = 2.5 to 13.1" over the twelve above-HAR crisis cells: measured
  across-seed t 2.50 to 13.14.
- "every one at t ~ 10" for the three 2022 cells: 9.58, 10.08, 10.40.
- "5-12x more in the bear market": 5.2x / 12.4x / 7.5x.
- Table 3 and Table 4 every cell and every significance mark
  (* |t|>2.3, ** |t|>3.5, across-seed t).
- G ladder, calm fold, transformer: 2021 0.0372, 2018 0.0582,
  2019 0.0643, 2022 0.1206, 2020 0.3626 -> cluster 0.037-0.064,
  crisis 5.6-9.8x. r = 0.91 with the crisis year removed.
- HAR under the same protocol: delete-2020 costs -0.0074 (0.7 IC
  points), never moves more than +0.0085, G = +0.0353 (6% of a 0.54
  variance) vs the transformer's 0.4022 (42% of 0.956).
- Window counts: L1 2020 deletion surrenders 21.8% (stress) to 25.5%
  (calm); L1 removes 25% more windows than L0.
- Placebo gap +0.0169 +- 0.0233 and skill +0.0046 (transformer, calm).

Two lines of body text were cut to hold 8 pages, both exact duplicates
of a table caption they sat next to (the "two below the baseline agree
on nothing" sentence in RQ1, the "nothing predicts which case it is"
sentence in RQ2). No result was dropped.

### Read-through, second pass (2026-08-07)

The user asked for no content cuts and no page-limit constraint, so the two
caption-duplicating sentences removed in the first pass were restored (paper
is now 9 pages, body ends on 8). Further re-measurements, all from
`scripts/verify_rq2_rq3.py` and ad-hoc passes over `outputs/pilot` and
`outputs/detector`:

Reproduced exactly, no change needed:

- Table 5, all 20 kappa cells: **0 mismatches** against the printed values.
- Cross-regime kappa ranges: transformer SCRUB 0.40-0.69, TSMixer hinge
  0.74-1.18 (never more than 0.26 from 1), transformer hinge 1.12-1.36,
  iTransformer hinge 2.61-4.98, PatchTST SCRUB 0.54 (fold2022) to 73.64
  (fold2024). Exactly two pairs stay within half the gap in all four
  regimes; only TSMixer's hinge brackets the oracle.
- Population d': Table 2's ten values reproduce to the printed decimal
  (11.1/5.2, 5.2/3.4, 11.3/7.7, 3.8/3.1, 5.3/3.1); over 100 cells
  -0.45 to +13.97; seven cells at or below d'=1; 17% below d'=2;
  r(G, d') = +0.62; max |G| among d'<1 cells = 0.0091.
- Window-level: 1,000 strict cells, median 0.538, 80% inside [0.50, 0.61];
  i.i.d. inflation median 1.90, per-architecture 1.55 to 2.07.
- Mechanism correlations (+0.69 marginal, +0.04 conditional, stress fold).
- Every year's realized vol / drawdown in the Table 3 caption.
- Two-definitions figure (n=10 sweep): oracle 0.509/0.497, reference
  0.545/0.489, forcing IC_F to 0.20 costs 0.492 -> 0.42 with the seed
  spread widening from +-0.004 to +-0.062.

Corrected in this pass:

17. **"the crisis-era gap falls as the level tightens ... in all twenty
    cells" was false.** L0 -> L1 falls in **20/20**. L1 -> L2 falls in
    only 10, ties in the 5 where the era sits at the training window's
    edge (there L2 = L1 by construction, as the lattice section itself
    says), and **rises in 5** (transformer/2025, tsmixer/2023,
    tsmixer/2024, itransformer/2024, patchtst/2024). Claim narrowed to
    the L0->L1 step and the L2 behaviour stated honestly.
18. **"r = -0.21 against Delta"** for removal volume: measured **-0.06**
    over the 100 L1 cells (per-arch -0.23 to +0.40). -0.21 matched no
    scope. Replaced; -0.06 is also the stronger number for the claim.
19. **"at a matched 20-22% the deletion effect ranges from +0.079 to
    -0.014"** mixed scopes: +0.079 is SegRNN's max over all
    architectures, -0.014 is the transformer's own min. Over the 25
    cells in that band the true range is **+0.079 to -0.068**.
20. **"crisis gap exceeding the calm one every time, by 3.0-11.1x
    (t = 4.5-25.9)"**. Against the largest non-crisis era in the same
    fold (the definition whose max, 11.1x, the paper already used) the
    range is **1.3 to 11.1x**, Welch t **1.6 to 20.9**, and one of the
    twenty (patchtst/fold2024, t = 1.6) is not significant. Restated.
21. Window-level audit: best cell **0.69**, not 0.67; r(window frac, G)
    **+0.51**, not 0.57; median where G>0.25 **0.59**, not 0.60.
22. hinge kappa on non-crisis eras **6.7 to 36.3** (paper had 6.6-30.2);
    hinge numerator per cell **0.15 to 0.51** (paper had 0.19).
23. **Survivorship bias was never disclosed.** The universe is the 501
    end-of-sample constituents carried back, i.e. not point in time.
    Two sentences added to Setup, with the paired-contrast defence.
    (An earlier docs note claimed this was "disclosed in paper"; it was
    not.)
24. "one architecture-method pair of twenty" in Practice and Conclusion
    contradicted RQ2, which says two pairs stay within half the gap.
    Reworded to "one brackets the oracle, a second stays within half
    the gap by overshooting".
25. The distribution-matching refutation reported only the stress fold.
    In the calm fold both correlations point the expected way (-0.96
    marginal, +0.59 conditional). Added, with the reason it does not
    rescue the explanation (every effect there is at noise scale).
26. refs.bib: "Rethinking Machine Unlearning for LLMs" (Liu et al.) was
    keyed to arXiv:2503.01854; the paper with that title and author is
    arXiv:2402.08787 / Nature Machine Intelligence. **Verify before
    camera-ready.**

## Narrative pass: findings and contributions (2026-08-07)

Diagnosis: the numbers were fine, the framing was one level off. The paper
was written as "we can finally measure unlearning properly" (methodological)
when its content is "measuring it properly overturns what the field assumed"
(substantive). Six structural changes, no data touched.

1. **Contribution list added to the Introduction.** Previously the only
   contribution sentence in the paper was in Related Work: "our
   contribution is not a new method: it is the inversion of that
   constraint". That is a method-of-study claim, not a findings claim,
   and it sat in section 2. Five contributions now stated: lattice,
   sign result, target result, audit result, released oracle matrix.
2. **The thesis moved from Related Work to the Introduction.** The
   Feldman-inversion argument (a clean legitimate year, deletion a gain
   for the models that absorbed it most, invisible to the field because
   approximate unlearning can only show utility fall) was the fourth
   paragraph of section 2. It is now the third paragraph of section 1.
   Related Work's version was rewritten to place the two literatures and
   name what each assumes, so the two do not repeat each other.
3. **The three RQ paragraphs now open with a claim** in italics, before
   the supporting detail. RQ headings kept for navigation.
4. **"The sign follows the deployability line" reframed as a screen.**
   The law reading needs the two below-baseline models to misbehave, and
   they are worse than a four-parameter OLS, so a reviewer can dismiss
   it as "broken models behave erratically". The screen reading does not
   depend on them: every model good enough to deploy gains, and the only
   two that do not are two nobody should deploy. Same fact, and the n=5
   objection no longer bites.
5. **Practice marks the decision rule as a prior from five
   architectures, not a law.**
6. **Limitations now says what determines the sign: we do not know.**
   Within a window, absorption orders eras but does not predict the
   sign; the within-fold G-Delta correlation is carried by 2020 alone
   (+0.89, falls to -0.87 without it). The sign result rests on the
   cross-model contrast, where HAR is decisive, and on n = 1 crisis.
   Stated explicitly rather than left to be asked.

Also: the two-definitions result moved to the front of RQ2 (it is the
most transferable idea in the paper and was the back half of the
section); the abstract's RQ1 sentence now leads with the gain instead of
the deployability line; a volatility-not-drawdown sentence added to RQ1
(2018 and 2022 drew down almost equally, -19% and -20%, but the gap on
2022 is twice the gap on 2018, in step with 24% vs 16% volatility).

## Merge with the Overleaf lineage (2026-08-07)

The user's Overleaf draft turned out to be a different lineage: much
better formal writing, but the scope of an earlier campaign (4
architectures, 2 folds, 2 deletion levels, 240 oracles, no HAR, no
TSMixer). Several of its stated conclusions were falsified by the
experiments run afterwards, most importantly the RQ3 heading "every
era-level deletion is detectable" (over 100 cells d' spans -0.4 to
14.0 and 17% fall below 2).

Direction chosen: **local as the base** (current scope, 26 corrections
already applied), porting the Overleaf material item by item. The
reverse would have meant rewriting every number.

New structure: Intro / Related Work / **Methodology** / **Experimental
Setup** / RQ1 / RQ2 / RQ3 / Practice / Conclusion. The old combined
`03_setup.tex` is superseded and archived under
`paper_overleaf/_superseded/`.

Ported from Overleaf:
- A standalone Methodology section: formal $A, D, F$, reference,
  oracle, unlearning operator $U$; deletion-level table; the nested
  forget sets $F_0(T) \subseteq F_1(T) \subseteq F_2(T)$ with
  equations; $G_F$, closure, $\Delta s_{\text{test}}$, $d'$ as
  numbered definitions; the audit access model.
- Notation unified on the Overleaf convention: $\mathcal{L}_F$,
  $s_{\text{test}}$, $\Delta s_{\text{test}}$, "closure" for kappa,
  with $\Delta$ and $G$ declared as shorthands.
- Seven architecture citations added to refs.bib (the local draft had
  none at all: Transformer, PatchTST, iTransformer, SegRNN, TSMixer,
  TimeMixer, DLinear).
- IC stated as the Spearman rank correlation.
- Target formula, forget-loss formula, $d'$ formula.
- MDE symbol glossary after Eq. 14.
- "A risk task rather than an alpha contest."
- Three reader-facing tables: the $G$ ladder (Table 4), the method
  guide (Table 7), the audit-object guide (Table 9).

Added while merging (gaps in both drafts):
- **The HAR benchmark is now defined.** It carries the entire RQ1
  split and neither draft ever wrote down the regression: log 5-day
  forward RV on a constant and log RV over the trailing 1, 5 and 22
  days, OLS, four parameters, subject to the same deletions.
- Placebo design moved to Experimental Setup, placebo numbers moved to
  RQ1 where the claim is made.

Verified before printing: every value in the new $G$ ladder table
reproduces exactly (2018 +0.0582+-0.0146 t=12.6; 2019 +0.0643+-0.0195
t=10.4; 2020 +0.3626+-0.0483 t=23.7; 2021 +0.0372+-0.0141 t=8.3;
2022 +0.1206+-0.0385 t=9.9; L0 2019 +0.0529, L0 2020 +0.5119;
L2 2019 +0.1006, L2 2020 +0.3339).

Data provenance settled: the evaluation folds use
`sp500_kaggle_returns.csv` (2015-12-22..2025-12-19) and the tuning
split uses `sp500_yf2008_returns.csv` (Yahoo + Tiingo for BK/DAY/
HOLX/MMC, back to 2008). The two agree on 1,220,223 of 1,220,224
overlapping observations, 34 differing by more than 1e-4. The paper
states this in one sentence since the artifact is released.

Paper is now 12 pages (body ends on 11).

## Results sections rewritten to question / what-to-look-at / finding (2026-08-07)

Author's verdict on the previous draft: the bold lead-ins were titles
for conclusions, not signposts, so the reader never knew where to look.
All three results sections now use one template: a bold question, one
sentence naming what we look at and which table holds it, then the
finding as a trend with few numbers.

The master table was dropped. Its unique content moved to where it
belongs: reference skill and the HAR benchmark into the four-regime
table, d' into a new audit table in RQ3.

New tables, every value recomputed from the grid before printing:
- Table 3, memorization gap G by deleted year x architecture, calm
  fold, L1 (5x5). Replaces both the old G-ladder and the master
  table's G columns.
- Table 5, two blocks: reference skill per architecture per test year
  with HAR on the same years, and the effect of deleting 2020. The
  above/below-benchmark split is now visible in one place.
- Table 7 (RQ3), d' beside G for both eras, which shows the audit
  result directly: the two columns order the architectures the same way.

CORRECTION found while building Table 5. The reference model's test IC
is NOT era-independent at L1, because L1 recomputes normalization
statistics without the deleted era, so a reference is retrained per
(era, level). The previously printed test IC (0.567 / 0.557 / 0.511 for
the three top architectures in the stress fold) averaged over all eras
AND all levels, mixing plain references with renormalized ones. The
clean quantity is the L0 reference, which is era-independent (verified:
identical to four decimals across all five eras). Corrected stress-fold
values: transformer 0.570, TSMixer 0.560, SegRNN 0.517, against HAR
0.508. SegRNN clears HAR by 0.009, not the 0.003 previously reported,
so the split is slightly safer than the paper had claimed.

Numbers deliberately turned into prose trends, with the values now
carried by the tables: the per-year G values, the per-architecture
effect sizes, the seed standard deviations, the closure values, and the
window-level distribution statistics. Eight statistics were restored
after the pass because they ARE the evidence for their claim and the
sentence is unfalsifiable without them: r = 0.97 and 0.91 (gap against
volatility), r = -0.06 (volume), the placebo ratios 1.4-1.8x and
2.5-13x, HAR's 6% against the transformer's 42% of target variance,
r = 0.96 (collinearity), r = +0.69 and +0.04 (the distribution
mechanism refutation), d' spanning -0.4 to 14.0, and r = 0.62 (audit
strength against memorization).

## v3 RQ1: the volatility law becomes a 100-cell exhibit (2026-08-08)

Author's diagnosis of the memorization block: the section's main
regularity (G tracks the deleted year's volatility) was asserted in
prose from one fold, with the per-year volatilities nowhere visible.

Fix: a new scatter, fig_gap, plots G for ALL 100 (architecture, fold,
deleted-year) cells at L1 against the realized volatility of the
deleted year. New statistic verified for it: within-architecture
correlation between G and deleted-year volatility over each
architecture's 20 cells:
  transformer +0.960, tsmixer +0.934, segrnn +0.933,
  itransformer +0.910, patchtst +0.761  (pooled +0.724)
Paper states the range 0.76-0.96. The old 5-point transformer-ladder
r = 0.97/0.91 is superseded by this stronger version in the final draft (kept in
the earlier draft).

Also verified: the era-position sign pattern replicates in the
intermediate folds (2023 fold: deleting 2020 positive for all three
deployable archs, 2018/2019/2021 negative; 2025 fold: 2020 positive,
2021/2023 negative, 2022 ~null). v3 states it in one sentence.

fig_ladder reduced to two panels (the per-year deletion effect in the
2022 and 2024 folds), with each year's realized volatility printed
under its x-tick so the reader can see which years are volatile
without leaving the figure.

Writing convention finalized after author feedback: every result
block is (1) what the exhibit reports plus a one-sentence re-gloss of
the quantity, (2) the factual context the reader needs, (3) the
observation, (4) the interpretation. Complete sentences, formal
register; sentence fragments ("Of the model.", "It gained.")
eliminated.

### fig_gap final form (2026-08-08)

Author-directed revisions: shared y scale across both panels;
iTransformer/PatchTST recoloured purple (#9467bd) and brown (#8c564b)
globally after the two greys proved indistinguishable; points at alpha
0.55; per-architecture least-squares fit lines with slopes in the
legend. Slopes of G on deleted-year volatility (G per percentage
point, computed from the same 100 cells):
  transformer 0.0123, tsmixer 0.0099, segrnn 0.0137,
  itransformer 0.0018, patchtst 0.0020
The deployable trio's slopes are 5-7x the below-benchmark pair's,
which is the architecture result in one number. fig_ladder gained a
proper legend (it lost its labels when the G panel was removed).

## Why does G track volatility? Difficulty ruled out (2026-08-08)

Author asked whether the mechanism is "volatile years have larger
loss, so the model memorizes them". Tested on the grid:

- Reference (trained WITH the era) per-era forget MSE, transformer:
  2020 is only mildly harder than ordinary years (fold2022: 0.821 vs
  0.584-0.707; fold2024: 0.760 vs 0.684-0.720), and calm 2017 (0.682)
  is HARDER to fit than 2016 (0.584). Per-day z-scoring equalizes the
  target scale, so no variance mechanics.
- Across all 100 cells, r(reference per-era MSE, G) = +0.16. The year
  a model finds hardest is not the year it memorizes most. Difficulty
  does not drive the gap.
- The gap opens on the ORACLE side: fold2024 transformer, era2019
  oracle 0.784 vs ref 0.720 (nearly recovered from neighbours);
  era2020 oracle 1.122 vs ref 0.760 (unrecoverable). Substitutability,
  not difficulty: calm years have in-window substitutes, the crisis
  has none, and volatility is the observable mark of irreplaceability.
- No decisive test possible within this sample: one crisis (n=1), so
  volatility and uniqueness cannot be separated. Stated as the
  boundary; the paper claims the law and the interpretation, not a
  mechanism.

Added to 5.1 as a three-sentence check after the volume paragraph.

## Year market stats: script saved, 2025 added, 2016 drawdown corrected (2026-08-08)

The per-year volatility/drawdown numbers now come from a saved script,
scripts/year_market_stats.py (equal-weighted mean of all names present
per day in the kaggle panel; sample sd x sqrt(252); compounded max
drawdown within the year). It reproduces the previously recorded
values exactly for 2017-2024 and:
- adds 2025: 17.9% vol, -16.3% drawdown (the paper's "(18%)" for the
  2025 fold header was already consistent);
- CORRECTS 2016's drawdown to -11.3% (the older hand-recorded -10.1%
  cannot be reproduced; nine of ten years match, so the script wins).
  tab:years now prints -11 for 2016.
tab:years extended to all ten years with a deletable / test-only
header split, so the fold headers' parenthetical volatilities and the
figure axes all trace to one script.

## Mechanism probe: activation hypothesis REFUTED, pervasiveness found (2026-08-08)

Pre-registered criterion (fixed before data): if deleting 2020 removes
a crisis template that stress cues invoke, the daily deletion gain
should concentrate on the test year's stressed days and be flat in the
calm fold.

Experiment: daily_delta.py, pure inference. Reference (sweep ckpt,
evaluated under mode-appropriate normalization -- discovery: the strict
reference IS the sweep weights under era-excluded normalization, no
retraining needed) vs era-2020 oracle ckpt, per-day cross-sectional IC
on each fold's test year. 3 deployable archs x 2 levels x 4 folds x 10
seeds. 24 JSONs in outputs/daily_delta/.

VERDICT: refuted. Daily gain is FLAT across stress quintiles in every
fold, both levels, both stress proxies (|eq-weight return| and 5-day
rolling vol): r(day stress, daily delta) between -0.19 and +0.07 with
no consistent sign. If anything the calmest quintile is slightly the
largest in the 2022 fold.

WHAT REPLACED IT (stronger finding): the gain is pervasive. Share of
individual test days on which the oracle beats the reference (L1,
3-arch pooled): 2022 fold 99%, 2023 fold 90%, 2025 fold 85%, 2024 fold
73%. Deleting 2020 improves the model on virtually every day of the
bear year, not on crash days only. The regime sets the LEVEL of the
daily gain (0.06 vs 0.01), not its day pattern.

Interpretation update: the memorized crisis content acts as a standing
distortion of the learned feature-to-rank mapping, not an episodically
triggered template. "When it costs" is a property of the year as a
whole, not of the day. The content-level story (irreplaceability,
oracle-side gap opening, HAR attribution) is untouched.

Analysis + figure: scripts/daily_delta_analysis.py -> fig_daily
(quintile profile, four flat lines ordered by fold volatility,
positive-day shares in the legend). Not yet in the paper; author to
judge.

## Direction probe: the tilt is architecture-specific, not universal (2026-08-08)

Pre-registered hypothesis (persistence inflation): training on 2020
makes the reference over-rank recently volatile stocks relative to its
oracle, stably across days, architectures and folds.

Result: REFUTED as a universal direction. Mean daily cross-sectional
Spearman of (pred_ref - pred_ora) against raw trailing 20-day vol:
  transformer: +0.10 / +0.23 / +0.32 / +0.17  (folds 22/23/24/25) all +
  tsmixer:     -0.06 / +0.18 / +0.31 / +0.26  mostly +
  segrnn:      -0.33 / -0.15 / +0.15 / +0.00  mostly - or 0
The Transformer's absorbed 2020 tilts it TOWARD recently volatile
names, SegRNN's tilts it AWAY in the stress folds, TSMixer mixed --
yet all three lose skill from the tilt in every fold (the daily IC
result). So the harm is universal but its direction onto observable
characteristics is architecture-idiosyncratic: what is shared is that
fitting 2020 bends each function away from the later regime's optimum,
not which way it bends.

(Bug log: predictions saved as (n,1); argsort along last axis produced
zero ranks -> empty correlations. Fixed by ravel. Join needed str()
casts on numpy scalar keys.)

Complete causal ladder now, every rung tested:
1. Phenomenon: 12/12 gains (iron).
2. Attribution: model property, not data (HAR control).
3. Form: standing daily bias, not stress-triggered (activation probe:
   flat quintiles, 99% positive days in the 2022 fold).
4. Direction: architecture-specific, opposite signs possible
   (this probe); no single characteristic-level mechanism exists.
5. Boundary: n = 1 crisis; parameter-level why is future work.

### HAR's own memorization spike (2026-08-08)

Verified from har_deletion.json (strict, share = G / target variance
0.54): deleting 2020 opens HAR's largest gap in every fold -- shares
6.5 / 4.7 / 1.6 / 1.9 per cent (folds 2022/2023/2024/2025) against
largest-ordinary-year shares 1.6 / 1.4 / 0.5 / 1.0, i.e. 1.9-4.1x.
One sentence added to 5.1: the volatility law reaches down to four
parameters, capacity sets its magnitude. HAR stays OUT of fig_gap
(different target scale would force renormalizing the whole axis).

## tab:method_choice verified and filled (2026-08-08)

Worst-fold |kappa-1| criterion, era-2020 strict cells:
  transformer: hinge 1.12-1.36 (worst dist 0.36), worst skill cost
    +0.011, cost 0.43x; runner-up SCRUB
  tsmixer: hinge 0.74-1.18 (0.26), +0.004, 0.42x
  segrnn: hinge 0.44-0.79 (0.56), +0.004, 0.22x
  itransformer: SCRUB 0.21-0.33 (0.79), +0.014, 0.33x
  patchtst: mechanical winner TS-Unlearn (kappa 0.07-0.17) only
    because it barely forgets; table says 'none reliable'.
Cost methodology: mean method seconds over era-2020 strict cells /
median oracle seconds of the same (arch, fold) over all cells with a
recorded oracle timing (timings_seconds key; the old 0.2-0.4x column
was calm-fold-only). TSMixer's cost updated 0.3x -> 0.4x accordingly;
skill cost 'none' -> 0.004 for uniformity.

Also logged: TS-Unlearn at the fixed budget is not harmless in the
2022 fold: dIC vs oracle +0.13/+0.07/+0.13 for the trio (vs +0.009/
+0.003/+0.009 in the 2024 fold), i.e. it forfeits the deletion gain
AND damages the model, while kappa stays near 0.1. The old claim
'methods fail by not forgetting rather than damaging' holds for
finetune/hinge/SCRUB but NOT for TS-Unlearn outside the calm fold.

## Three author-requested checks (2026-08-08)

1. kappa denominator G(2020) per (arch, fold), strict:
     transformer 0.402/0.368/0.363/0.367 (hi/lo 1.11)
     tsmixer     0.313/0.249/0.302/0.361 (1.45)
     segrnn      0.444/0.351/0.390/0.383 (1.26)
     itransformer 0.071/0.061/0.046/0.039 (1.79)
     patchtst    0.043/0.098/0.080/0.036 (2.73)
   "0.36-0.40 nearly constant" was the TRANSFORMER only. RQ2 sentence
   corrected to "varies by less than half across folds on the
   deployable architectures".
2. SegRNN-hinge: fold means 0.556/0.787/0.496/0.437 -- no fold mean
   crosses 1, the figure and the under-forgets verdict stand. Seed
   tail: 3 of 40 seeds exceed 1 (max 1.59), logged here only.
3. Five-architecture framing removed from abstract, intro and
   conclusion: the substitution claim is now stated over the twelve
   deployable architecture-method pairs (exactly one brackets, one
   more within half the gap), with the non-deployable pair as
   released-artifact support. The abstract's "overshoots fivefold"
   (iTransformer) replaced by within-trio contrasts (Transformer
   overshoot, SegRNN half the required forgetting).

## Two author-requested checks on RQ2 (2026-08-08, second batch)

1. TSMixer-hinge skill cost significance (vs oracle, per fold):
     fold2022 +0.0019 (mean NW t +0.62, across-seed t +0.63)
     fold2023 -0.0047 (NW t -1.59, across-seed t -1.26)
     fold2024 +0.0036 (NW t +1.56, across-seed t +1.71)  <- worst fold
     fold2025 +0.0006 (NW t +0.02, across-seed t +0.19)
   VERDICT: no fold-level cost is distinguishable from zero (all
   aggregate t < 2; the worst fold reaches only 1.71). "At no
   measurable cost" in the prose is supported and stays. Wrinkle
   logged: 5-7 of 10 individual seeds are NW-significant per fold but
   with MIXED signs, cancelling in the mean; measurable per seed,
   not at the fold level.
2. fig_twodefs sweep provenance confirmed from its artifacts
   (outputs/sweep_2def_n10/*/transformer_..._fold2024_...): the
   TS-Unlearn budget sweep is the 2024 fold, transformer, era-2020,
   strict. "2024 fold" added to the caption and the first prose
   mention; leverage sentence added (the 0.49 -> 0.42 fall happens in
   the calmest fold, where deletion itself moves at most 3%).

## Level-table measure check (2026-08-08, author query)

The author read 0.363 as a cross-fold mean. It is NOT: 0.363 is the
2024-FOLD value. Transformer G(2020) per level per fold, verified:
  L0: 0.524 / 0.495 / 0.512 / 0.575  (folds 22/23/24/25; mean 0.527)
  L1: 0.402 / 0.368 / 0.363 / 0.367  (mean 0.375)
  L2: 0.402 / 0.343 / 0.334 / 0.394  (mean 0.368)
Figure-table consistency: fig_gap plots the four per-fold L1 points
(0.363-0.402), so the author's eyeball of 0.37-0.40 matches the
figure exactly; no discrepancy. The familiar triple 0.512/0.363/0.334
is the 2024 fold throughout (its L0/L2 companions are also fold2024).
Note L2=L1 exactly in the 2022 fold (era at window edge), and the
L0->L1 drop holds in every fold and dominates the L1->L2 step.
DECISION NEEDED for the planned 5.4 mini-table: either label the G
column '2024 fold' explicitly, or use the cross-fold means
(0.527/0.375/0.368). Recommendation: cross-fold means, since the
sign-count column (12/12, 12/12, 11/12) is already cross-fold and a
mixed-scope table invites exactly this question.

## Normalization level: definition table corrected (2026-08-08, author query)

Implementation ground truth (src/fin_unlearning/era.py:165-167):
normalization statistics are recomputed WITHOUT the era for modes
strict and l2halo, i.e. from L1 upward; L0 (target) keeps full-data
statistics. stats_excluded_era flags in every cell JSON agree
(target False, strict/l2halo True).

The author's new 5.4 already states it correctly ("the step from L0
to L1 ... adds the overlapping context windows and recomputes
normalization"). The definition table (tab:deletion_levels) was the
inconsistent one: its L2 examples listed "normalization". Fixed both
rows: L1 now reads "...and use in the normalization statistics that
scale them"; L2's examples drop normalization and keep rolling
statistics, EMAs, imputation. One story everywhere: normalization is
an L1 obligation in this paper.

## fig_audit scope check: author chose to KEEP all five (2026-08-09)

Verified both scopes in case the choice is revisited. Correction to
my own earlier argument: the unverifiable cells do NOT concentrate in
the non-deployable ARCHITECTURES -- below d'=2 the split is patchtst
7, segrnn 7, tsmixer 2, transformer 1, itransformer 0. They
concentrate in low-G CELLS (weak absorptions), wherever they occur;
all five deployable cells at d'<=1 have |G|<0.01 (segrnn 2016/17/18
deletions in early folds, tsmixer 2017).

Deployable-only (60 cells) would preserve every headline: d' range
-0.45..14.0 unchanged (the floor is segrnn|2022|2017), 10/60 = 17%
below d'=2 unchanged, r(G,d') 0.62 -> 0.70 (stronger). Window-level
deployable-only: median 0.549, best 0.692, iid inflation median 2.00
(vs 0.54 / 0.69 / 1.9 over all five) -- prose would survive nearly
verbatim. Author decision: keep all five for now; the caption's
justification (an auditor cannot presume deployability) stands.
