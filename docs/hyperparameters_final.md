# Final hyperparameters (v2), and how they were selected

Two rounds of tuning, both on the SAME era-clean split (yf2008 panel,
train 2010-2014, validate 2015, seeds 1,2,3; no test year is ever touched).

**Round 1 (P0)** searched d_model x dropout at fixed lr 5e-4, depth 2 --
24 cells, 4 architectures.

**Round 2 (rescue, 2026-07-21)** was triggered by the observation that two
architectures sit below a HAR baseline at test time. It searched the axes P0
never touched: the learning rate for every architecture, plus the structural
knob each one is actually sensitive to (patch geometry for PatchTST, depth
for the transformers, d_model/depth/dropout for the mixers, decomposition
window for DLinear). 351 runs total.

## Selection rule

Adopt a round-2 winner only if it beats the incumbent config by MORE than the
across-seed standard deviation. A hyperparameter change invalidates every cell
already computed for that architecture, so a change must be justified by more
than seed noise.

## Outcome

| arch | incumbent val | rescue best val | gain | seed noise | decision |
|---|---|---|---|---|---|
| transformer | 0.5036±0.0015 | 0.5028±0.0012 | 0.0008 | 0.0015 | KEEP |
| segrnn | 0.5208±0.0011 | 0.5208±0.0011 | 0.0000 | 0.0011 | KEEP |
| patchtst | 0.6006±0.0002 | 0.6001±0.0008 | 0.0005 | 0.0008 | KEEP |
| itransformer | 0.6017±0.0003 | 0.6016±0.0003 | 0.0001 | 0.0003 | KEEP |
| tsmixer | (not tuned in v2) | 0.5085±0.0023 | -- | -- | ADOPT |
| timemixer | (not tuned in v2) | 0.5669±0.0009 | -- | -- | ADOPT |
| dlinear | (not tuned in v2) | 0.6043±0.0004 | -- | -- | ADOPT |

**No existing cell was invalidated**: all four architectures already in the
paper keep their hyperparameters, so the 473 cells on disk stay valid.

## Final configuration per architecture

Shared trainer for every architecture: AdamW, weight decay 0.01, batch 1024,
max 300 epochs, patience 20 on validation MSE, best-val checkpoint.

| arch | lr | structure | val loss | test IC (calm/stress) |
|---|---|---|---|---|
| transformer | 5e-4 | d_model 64, 2 layers, dropout 0.1 | 0.5036 | 0.489 / 0.546 |
| tsmixer | 6e-3 | d_model 64, 2 layers, dropout 0.1 | 0.5085 | not yet measured |
| segrnn | 5e-4 | d_model 64, 2 layers, dropout 0.3 | 0.5208 | 0.473 / 0.458 |
| timemixer | 1e-2 | d_model 64, 3 layers, dropout 0.1 | 0.5669 | not yet measured |
| patchtst | 5e-4 | d_model 32, 2 layers, dropout 0.1, patch 8 / stride 4 | 0.6006 | 0.169 / 0.168 |
| itransformer | 5e-4 | d_model 32, 2 layers, dropout 0.1 | 0.6017 | 0.221 / 0.130 |
| dlinear | 3e-3 | moving_avg 25 | 0.6043 | not yet measured |

HAR baseline on the same test years: 0.392 (calm 2024), 0.502 (stress 2022);
naive persistence 0.293 / 0.400.

Raw sweep records: `v2/outputs/rescue_tune/{arch}_lr{lr}.jsonl`
(one line per config x seed). Config fingerprints: `v2/outputs/config_manifest.json`.
