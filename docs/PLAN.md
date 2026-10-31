# PLAN v2 — ICAIF-focused clean redo (2026-07-19)

*Why v2: (1) era-clean hyperparameters — v1's tuning split (2016-18/19)
overlapped deletable eras 2016-2019 (HP channel; measured ≈0 by control
47372, but v2 removes it by design); (2) ICAIF focus — 3 RQs, 4
architectures, 2 folds, one deletion level as default; reader load cut
roughly in half. The entire v1 grid becomes the extended version's
"original-HP replication" appendix.*

---

## 0. Locked design decisions (with rationale)

| decision | choice | rationale |
|---|---|---|
| HP tuning split | yf2008 panel, train 2010-01-01..2014-12-31 (5y), val 2015, test_end=val_end; tuning seeds 1,2,3 (disjoint from evaluation seeds 4-13) | temporally disjoint from ALL deletable eras (2018-2022) and all test years; UNIFORM WINDOW RULE: every training window in the study — tuning and both folds — is five years; GFC core (2008-09) avoided |
| architectures | transformer, iTransformer, SegRNN, PatchTST | each carries a distinct headline: high-memo/methods-work; low-memo control; RNN persistence + audit coin-flip; exposure negative pole + methods-fail. DLinear/TimeMixer/TSMixer → extended version |
| folds | 2022 (stress; train 2016-2020) + 2024 (calm; train 2018-2022) — both FIXED five-year windows | folds are contrast scenarios, not samples; inference units are ~250 paired days x seeds; fold 2023 duplicated 2024's role and created pooling heterogeneity. 2023 replication lives in the extended version |
| window style | FIXED five-year rolling windows (main) | deployment-common; the two folds become structurally symmetric (same train length, era-2020 share ~22-26% in both); framing sentence kept: rolling pipelines have scheduled data expiry, so within-window deletion is the live case; EXPANDING results (entire v1 grid + fixed-vs-expanding comparison) = extended-version robustness |
| deletion level | \L1 (strict) as the paper-wide default | the compliance-honest semantics; \L0 appears only in the strictness-price cells; \L2 halo + \L3 = one sentence each + analytic note |
| seeds | UNIFORM 10 seeds (4-13) for every experimental cell (sole exception: the single-cell two-definitions sweep) | compute cap lifted; uniform n kills all per-row caption complexity — one sentence in the paper: "every cell: 10 seeds"; disjoint from tuning seeds 1-3 |
| terminology | "audit" umbrella; detect = method; verify = only after power accounting (one definitional sentence) | avoids H0-proof and crypto-verification landmines; the MDE machinery is what licenses limited verification statements |
| flagship exhibit | master table "compliance profile per architecture" (memorize / exposure / fixable / auditable) | the paper in one screenshot; four detail tables hang off it |

Output roots: `~/fin-unlearning/v2/` — the ENTIRE v2 campaign (configs/, outputs/{tuning,sweep,pilot,detector,slurm_logs}, docs/) lives under this one folder, fully separated from v1 artifacts for post-hoc clarity. Reference sweep naming: `{stem}_fold{F}_seed{S}_model.pt`
with v2 config stems (`*_v2` configs emitted into `v2/configs/`; never overwrite v1 tuned configs). Every JSON carries both repos' commits.
Freeze (worktree + vendor) refreshed at campaign end; repro dry-run
before submission.

---

**Companion documents**: `docs/parameters.md` — every fixed
parameter (data/task, splits, seeds, trainer, tuning grid, deletion
semantics, method HPs, evaluation quantities, naming conventions);
`docs/exhibits_spec.md` — column-level mockups of all 5 tables + 3
figures with captions and data sources; `docs/design_log.md` — the
decision history.

## 1. Research questions, experiments, expected results

### RQ1 — What does deleting a licensed era mean, and what does exact deletion do?

| experiment | cells | seeds |
|---|---|---|
| era ladder | transformer x eras 2018..2022 x \L1 x fold2024 (calm 2018/2019/2021, crisis 2020, bear 2022) | 10 |
| architecture spectrum | 4 archs x eras 2019,2020 x \L1 x fold2024 | 10 |
| strictness price | transformer x 2019,2020 x \L0 x fold2024 | 10 |
| two-sided exposure | 4 archs x 2019,2020 x \L1 x fold2022 | 10 |
| quantity placebo | transformer x 2020-matched random x fold2024 | 10 |

**Expected (v1 priors as falsifiable predictions):** calm-band gaps
0.02-0.05 (2018/2019/2021) with 2020 at 10-20x the band and era 2022
(the in-window bear year) expected elevated (falsifier: 2020 inside the
band kills the substitutability claim); spectrum ordering iTr < PatchTST
< SegRNN ~ transformer (falsifier: ordering scrambles => memorization is
HP-dependent after all); exposure signs SegRNN/transformer positive,
PatchTST ~ -0.1, 2019 ~ 0 everywhere (falsifier: sign flips vs v1 =>
the v1 signs were HP artifacts — this is the single most important
replication in v2); \L1 = 1.25x \L0 windows; placebo gap ~ 0.

### RQ2 — Can approximate unlearning substitute for retraining, and what is the right target?

| experiment | cells | seeds |
|---|---|---|
| methods matrix (10 seeds) | {finetune, NegGrad+hinge, SCRUB, TS-Unlearn} x {transformer, PatchTST} x {2019, 2020} x \L1 x fold2024 | 10 |
| two-definitions sweep | TS-Unlearn lr x budget grid, one cell (transformer 2020) | 1 |

(plain NegGrad divergence: one sentence, not tabulated; relabel_noise
dropped — superseded by faithful TS-Unlearn.)

**Expected:** transformer: SCRUB kappa ~ 0.8 with IC ~ oracle at ~1/5
retrain cost; hinge overshoots; PatchTST: no method reaches kappa in
[0.5, 1.5] with IC intact (falsifier: a method works on PatchTST =>
"only exact unlearning is real on some architectures" weakens to a
tuning claim); oracle forget-IC ~ 0.5 vs reference ~ 0.53 (the
two-definitions gulf); TS-Unlearn pushes forget-IC down only by
dragging test-IC in lockstep.

### RQ3 — Can deletion be audited?

| experiment | cells | source |
|---|---|---|
| per-window loss extraction | all RQ1 fold-2024 ref/oracle pairs | inference pass |
| population AUC | 4 archs x 2019,2020 x \L1 x fold2024, n=10/side | reuses RQ1 oracles |

**Expected:** crisis cells at complete in-sample separation (report d′
+ exact permutation p = 5.4e-6, AUC shown as >0.99 — never 1.00);
calm-\L1 minimum on SegRNN (0.5-0.7 = the coin-flip bookend; falsifier: calm-year SegRNN cleanly
auditable => the unverifiability half collapses); day-week power
plateau, month-grain failure for crisis on >= 2 archs; iid window
treatment inflates t severalfold vs day-level HAC.

---

## 2. Paper outline (ICAIF, 8 pp)

**Title (v2)**: *Can You Delete a Year of Market Data?
Machine Unlearning Against Exact Retraining Oracles* — v1's "Data
Feed" overpromised (feed-level units deferred) and "Verification"
broke the audit-terminology discipline; "market data" carries the
domain, the oracle gets title billing, and the three RQ nouns live in
the abstract, not the title.


| # | section | pages | exhibits |
|---|---|---|---|
| 1 | Introduction: the license-expiry problem; oracle affordability; 3 RQs; contributions | 1.1 | — |
| 2 | Related work: unlearning; TS unlearning + TS MIA; evaluation at financial SNR | 0.5 | — |
| 3 | Setup: primer (one para); task/protocol + inference-units statement + era-clean tuning statement; membership lattice (\L1 default); oracle + kappa (Eqs 1-4); quantity note (2 sentences) | 1.4 | — |
| 4 | RQ1 — deletion measured: master table first, then ladder/spectrum + exposure; how-to-read-the-sequence para (regime bundle) | 1.4 | **T1 master**, T2 ladder+strictness, F1 exposure bars |
| 5 | RQ2 — methods vs the oracle: matrix; two-definitions (Eqs 5-6) + quantization-undoes-unlearning warning | 1.2 | T3 methods, F2 two-definitions |
| 6 | RQ3 — can deletion be audited: verify-definition sentence; two auditors; AUC columns live in T1; granularity numbers in text | 1.2 | (uses T1) |
| 7 | Practice: contracts / model owners / auditors | 0.8 | — |
| 8 | Conclusion + limitations (regime-contrast claims, one panel, protocol released) + repro para | 0.4 | — |

Cut to extended version: formal propositions + proofs, HP grids,
fixed-width + \L2 experiments, 3 extra architectures, fold 2023, DLinear
control, TS-Unlearn failure-mode taxonomy, granularity figure, sign-
audit trail. Each gets <= 1 sentence + pointer in the main text.

---

## 3. Execution phases

| phase | content | jobs | est. GPU-h |
|---|---|---|---|
| P0 | tuning, 4 archs (6-config grid x 3 seeds (1,2,3) each) on yf2008 2012-14/15 | unl2_tune_{arch} x4 | ~10 |
| P1 | emit configs_v2; reference sweep: 4 archs x (both folds x seeds 4-13) = 80 refs | unl2_ref_{arch} | ~14 |
| P2 | RQ1 oracles: ladder 50 + spectrum 60 + \L0 20 + exposure 80 + placebo 10 | unl2_rq1_* | ~38 |
| P3 | RQ2 methods 80 runs + TS-Unlearn sweep 1 cell | unl2_rq2_* | ~10 |
| P4 | RQ3 detector + AUC passes | unl2_rq3 | ~3 |
| P5 | fill paper v2 numbers; refresh freeze; repro dry-run; Overleaf | — | ~1 |

Total ~78 GPU-h => ~2.5-3 days under the 6-GPU cap. Gate after P0:
tuned models must reach val-2015 losses comparable to v1's tuning run
(sanity that yf2008 pre-period is trainable); gate after P1: reference
fold-2024 test ICs within ~0.05 of v1's (models are healthy under new
HPs). Gate after P2-exposure: SIGN REPLICATION vs v1 (the key check).

## 4. Risks

- yf2008 2012-14 coverage/quality — P0 gate catches it; fallback:
  us_large_liquid panel same years.
- New HPs shift absolute ICs — fine; all claims are within-cell
  contrasts; P1 gate bounds the drift.
- PatchtST training cost dominates P2/P3 (~30 min/train) — schedule its
  jobs first in each phase.
- Optional armor (post-P4, if time): one Nikkei cross-market cell.
