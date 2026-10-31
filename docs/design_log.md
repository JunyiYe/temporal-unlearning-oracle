# v2 design log — every decision, in order, with rationale

*Chronological record of the 2026-07-19 design session (user + assistant).
The final state is in ../PLAN.md; this log preserves WHY each choice
was made and what it replaced, so future-us can audit the path.*

1. **v2 exists because of the HP channel.** User caught that v1's
   tuning split (train 2016-18, val 2019) overlaps deletable eras
   2016-2019. A default-HP control measured the channel ≈ negligible
   (2017's gap shift under different HPs matched clean-2021's generic
   shift), but the user chose design purity: full redo with era-clean
   HPs. v1 becomes the extended version's original-HP replication.

2. **ICAIF focus → 3 RQs, 4 architectures, 2 folds.** RQ1 deletion
   semantics+effects / RQ2 methods+target definition / RQ3 audit.
   Architectures = transformer, iTransformer, SegRNN, PatchTST (each
   carries a distinct headline; DLinear/TimeMixer/TSMixer → extended).
   Folds = 2022 (stress) + 2024 (calm): folds are contrast scenarios,
   not samples — inference lives in paired days × seeds; fold 2023
   duplicated 2024's role and caused pooling headaches in v1.

3. **Fixed five-year training windows, both folds.** User rejected the
   5y/7y expanding asymmetry ("怎么还是用5年和七年啊"). fold2022 trains
   2016-2020, fold2024 trains 2018-2022. Assistant's earlier claim that
   expanding is "what deployment does" was corrected by the user:
   rolling windows are equally common in practice; framing sentence
   kept (rolling = scheduled data expiry). Expanding-window results
   (all of v1 + the fixed-vs-expanding comparison) = extended-version
   robustness.

4. **Consequences of fixed windows:** era 2017 left fold2024's window;
   calm representative is now **2019** in BOTH folds (uniform); ladder
   becomes 2018-2022 (calm 2018/2019/2021 + crisis 2020 + bear 2022);
   deletable set = 2018-2022; era-2020 share ~22-26% in both folds
   (dilution asymmetry gone by construction).

5. **Tuning split: yf2008 panel, train 2010-2014 (5y), val 2015,
   seeds 1,2,3.** Evolution: first proposed 2012-2014 (3y, matching
   v1's tuning volume); user then chose the UNIFORM WINDOW RULE — every
   training window in the study is five years — trading the
   v1-volume-matching argument for a cleaner uniformity statement.
   Tuning seeds set to 1,2,3 by the user (3-seed selection; was the
   script default 7,8), disjoint from evaluation seeds.

6. **Evaluation seeds 4-13, uniform 10 per cell.** User set experiments
   to start at seed 4; later lifted the compute cap, so the tiered
   design (10/5/3) was collapsed to UNIFORM 10 seeds for every cell
   (sole exception: the single-cell two-definitions sweep). One caption
   sentence — "every cell: 10 seeds" — replaces all per-row n
   bookkeeping (the class of Table-1 problem the user caught in v1).

7. **Folder isolation.** All v2 artifacts under `~/fin-unlearning/v2/`
   (this tree), physically separated from v1 for post-hoc clarity.

8. **Terminology discipline (from v1 review discussions, inherited):**
   "audit" umbrella; detect = method; verify = only after power
   accounting. Flagship exhibit = master compliance-profile table.

9. **Parallelization.** Cap lifted → P0 exploded into 24 per-config
   jobs; P1-P4 to be submitted at maximum dependency-allowed width;
   whole-campaign wall-clock target ≈ 1 day of compute.

Key v1 lessons carried forward (see ../../docs/ for the audits):
sign convention lives in the equation, not captions; significance
tiers, not tallies; every selection step's information set documented;
placebo for the quantity confound; reproduction dry-run before
submission.
