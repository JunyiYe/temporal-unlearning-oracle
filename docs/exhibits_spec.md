# v2 exhibit specification — 3 tables + 2 figures, each unique

*Consolidated after redundancy review: the master table already carries
per-architecture memorization and audit columns, so the spectrum panel
and the AUC table were duplicate outings of the same numbers. 5
exhibits total; nothing gets added without deleting something.*

## T1 — MASTER: compliance profile per architecture  (Sec 4 opener)

| Architecture | Memorization G (2020 / 2019) | Stress-year Δ (fold 2022, \L1) | Best approx. method (κ, cost) | Audit d′ (2020 / 2019; AUC in parens, saturated cells shown as >0.99) |
|---|---|---|---|---|
| Transformer / iTransformer / SegRNN / PatchTST | ... | ... | ... | ... |

Absorbs: architecture spectrum (old F1b) and the AUC table (old T5).
Caption: four verbs — memorize / exposed / fixable / auditable; no
column predicts another. Audit column reporting rule: d′ is primary
(does not saturate); AUC = Φ(d′/√2) shown only below 0.99, else
">0.99"; complete-separation cells carry the exact permutation
p = 1/C(20,10) ≈ 5.4e-6 (assumption-free); 'in-sample separability'
stated. NEVER print AUC = 1.00.

## T2 — deletion measurement  (Sec 4)

Block A (ladder): deleted year {2018,2019,2020,2021,2022} × {G, t},
transformer, fold 2024, \L1, n=10. Block B (strictness price): eras
2019/2020 × {windows vs \L0, G, Δ}. Absorbs old F1a — the calm-band
vs 2020 contrast reads directly off five numbers; t-stats need a table.

## T3 — methods vs the oracle  (Sec 5; formerly T4)

method × {transformer, PatchTST} × {κ(2020), κ(2019), ΔIC vs oracle,
cost vs retrain}; plain-NegGrad divergence = footnote line.
Over-forgetting (κ>1) flagged as a detectable signature.

## F1 — two-sided exposure bars  (Sec 4; formerly F2)

Horizontal Δ bars per architecture, fold 2022 era 2020 \L1, ±sd
whiskers, zero line. Absorbs old T3: exact values + significance tiers
in the caption; 2019-control ("|Δ| ≤ ..., n.s.") and \L0 comparison as
one sentence each in text. The money figure: bars on both sides of
zero.

## F2 — the two definitions of forgetting  (Sec 5; formerly F3)

Scatter: x = forget-IC, y = test-IC; TS-Unlearn budget sweep (6
points), reference (star), oracle (diamond). The noise target is a
state the counterfactual does not occupy.

## NOT in the paper (extended version)

Granularity power figure (numbers in Sec 6 text), AUC d′ per-fold
detail, exposure table form, spectrum bars, formal propositions, HP
grids, fixed-width/L2/placebo detail (placebo = one sentence),
v1 original-HP replication.
