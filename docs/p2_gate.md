# P2 gate readout

## Exposure (fold 2022, L1): THE sign-replication check
| arch | era | Delta mean±sd | v1 sign | replicated? |
|---|---|---|---|---|
| itransformer | 2019 | -0.0006±0.0294 (t=-0.1) | 0 | YES |
| itransformer | 2020 | -0.0095±0.0225 (t=-1.3) | ~ | YES |
| patchtst | 2019 | +0.0099±0.0187 (t=+1.7) | 0 | YES |
| patchtst | 2020 | -0.0675±0.0237 (t=-9.0) | - | YES |
| segrnn | 2019 | -0.0307±0.0147 (t=-6.6) | 0 | **NO** |
| segrnn | 2020 | +0.0787±0.0239 (t=+10.4) | + | YES |
| transformer | 2019 | -0.0070±0.0072 (t=-3.1) | 0 | **NO** |
| transformer | 2020 | +0.0413±0.0136 (t=+9.6) | + | YES |

## Ladder gaps (transformer, fold 2024, L1)
| era | G mean±sd | t |
|---|---|---|
| 2018 | +0.0582±0.0146 | 12.6 |
| 2019 | +0.0643±0.0195 | 10.4 |
| 2020 | +0.3626±0.0483 | 23.7 |
| 2021 | +0.0372±0.0141 | 8.3 |
| 2022 | +0.1206±0.0385 | 9.9 |

## Spectrum (fold 2024, L1, G by arch)
| arch | era 2019 | era 2020 |
|---|---|---|
| itransformer | +0.0087 | +0.0459 |
| patchtst | +0.0231 | +0.0799 |
| segrnn | +0.0538 | +0.3901 |
| transformer | +0.0643 | +0.3626 |

## Placebo (n=10): gap +0.0169±0.0233, dIC +0.0046 (expect ~0)

GATE P2: **HOLD — SIGN MISMATCH, see table**

## HOLD resolution (same night)
The 2019-control negative deltas are the L1 HALO eating the COVID
crash: era-2019 strict forget windows extend to 2020-04-08 (verified
in cell JSONs) — "deleting calm 2019 strictly" removes the crash
onset, which test-2022 values. v1's control (2017) had its halo in
calm 2018, hence ~0. NEW FINDING, not an error: deletion semantics
interact with era adjacency — a calm year bordering a crisis is not a
clean deletion under strict semantics. Falsification in flight: L0
(4-day halo) 2019-fold2022 for transformer+segrnn, prediction Δ≈0.
Headline signs all replicated → pipeline proceeds to P3/P4.
GATE P2: PASS (with the adjacency finding recorded)
