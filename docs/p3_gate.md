# P3 gate readout (methods + halo + two-definitions)

## Halo falsification OVERTURNED -> better finding
L0 era-2019 fold-2022 deltas remain negative (segrnn -0.025 t=-4.4 =
80% of the L1 effect; transformer -0.006 t=-2.2): the halo explains
little — 2019 ITSELF carries forward value for test-2022. Combined
with v1's same-window 2017 deletion (~0) and v2's 2020 (+0.079/+0.041):
a clean WITHIN-FOLD era-position curve (distant-calm ~ 0, recent-calm
negative, crisis positive) — the identification the cross-fold arc
could never deliver. Paper: fold-2022 exposure table gains a second
real column; the inert-calm control role moves to fold-2024.

## Methods (n=10, strict, fold 2024)
transformer replicates v1: scrub k=0.69 dIC-0.001 (best), hinge 1.27
overshoot, finetune 0.23, ts_unlearn 0.12, neggrad diverges.
PatchTST: pre-registered falsifier TOUCHED — scrub-2019 k=0.54+-0.40
with IC intact — but seed variance is 4x and 2020 scrub explodes
(k=74+-215). Claim SOFTENED per protocol: "no method works RELIABLY on
PatchTST" (mean acceptable once, seed-level never; crisis cells
explosive). Architecture-gating survives in reliability form.

## Two-definitions sweep
Lockstep replicated: lr1e-3 pushes k only to 0.49 while test IC falls
0.49->0.41; gentle settings stay at the reference. forget-IC values for
F2 computed in P4.

GATE P3: PASS (with the era-position reinterpretation and the PatchTST
softening recorded before any paper text is written)
