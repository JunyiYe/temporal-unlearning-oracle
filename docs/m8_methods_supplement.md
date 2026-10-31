# Methods supplement: iTransformer + SegRNN (fills T1 "not tested")

Date 2026-07-20. Jobs 47571-47580 (10 jobs, all COMPLETED, 0 failures;
iTransformer ~16 min/job, SegRNN ~33-37 min/job). Protocol identical
to P3: era 2020, strict (L1), calm fold (train 2018-22, test 2024),
seeds 4-13, methods finetune/neggrad/neggrad_hinge/scrub/ts_unlearn,
references from v2 sweep, oracles reused from P2 checkpoints
(`--no-skip-existing` rewrote the P2 JSONs with methods added;
originals in outputs/pilot/backup_pre_m8/). Aggregation:
scripts/m8_aggregate.py, cross-validated against published T3 rows
(transformer/patchtst kappa reproduced digit-for-digit).

## Results (n=10; dIC = oracle - method, positive = skill lost)

| arch | method | kappa +- sd | dIC vs ora | cost |
|---|---|---|---|---|
| iTransformer | finetune | 0.21 +- 0.23 | +0.004 | 0.43x |
| iTransformer | hinge | 4.45 +- 1.03 | -0.007 | 0.44x |
| iTransformer | SCRUB | 0.25 +- 0.16 | -0.011 | 0.33x |
| iTransformer | TS-Unlearn | 0.07 +- 0.05 | -0.003 | 0.55x |
| SegRNN | finetune | 0.07 +- 0.14 | +0.003 | 0.20x |
| SegRNN | hinge | 0.50 +- 0.08 | +0.004 | 0.16x |
| SegRNN | SCRUB | 0.28 +- 0.36 | +0.004 | 0.11x |
| SegRNN | TS-Unlearn | 0.06 +- 0.05 | +0.009 | 0.26x |

(plain neggrad diverges, as on the other archs; not tabulated)

## Reading

- SCRUB-on-transformer (0.69, IC intact) is now the ONLY reliable
  near-oracle combination among 4 architectures.
- SegRNN (the other heavy memorizer, G=0.390): everything
  under-forgets; hinge is seed-stable at exactly half the gap
  (0.50 +- 0.08). SCRUB does NOT transfer from the transformer
  (0.28 +- 0.36, cv > 1).
- iTransformer (low memorizer, G=0.046): hinge overshoots 4.5x;
  SCRUB/finetune close ~a quarter of an already-small gap.
- Skill costs on both new archs are tiny (|dIC| <= 0.011): the
  failure mode is not collateral damage but failure to forget.
- Hinge kappa across archs: 0.50 (SegRNN) / 1.27 (transformer) /
  3.46 (PatchTST) / 4.45 (iTransformer) -- rankings do not transfer,
  now shown on all four.

Paper updates: T1 method column (2 rows), T3 (+8 rows), RQ2
feasibility paragraph, abstract/intro/conclusion/practice one-liners.
