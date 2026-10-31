# P1 gate readout (reference health)

| arch | fold | n | test IC mean±sd | IC range | val loss | epochs | n_train |
|---|---|---|---|---|---|---|---|
| itransformer | 2022 | 10 | +0.1246±0.0162 | [+0.103, +0.148] | 0.9072 | 21-23 | 562,325 |
| itransformer | 2024 | 10 | +0.2209±0.0157 | [+0.201, +0.254] | 0.9333 | 21-23 | 608,526 |
| patchtst | 2022 | 10 | +0.1681±0.0210 | [+0.140, +0.215] | 0.9072 | 21-36 | 562,325 |
| patchtst | 2024 | 10 | +0.1691±0.0146 | [+0.146, +0.195] | 0.9285 | 25-52 | 608,526 |
| segrnn | 2022 | 10 | +0.5169±0.0104 | [+0.492, +0.529] | 0.7595 | 23-63 | 562,325 |
| segrnn | 2024 | 10 | +0.4764±0.0024 | [+0.472, +0.480] | 0.7728 | 48-65 | 608,526 |
| transformer | 2022 | 10 | +0.5696±0.0049 | [+0.562, +0.579] | 0.7409 | 21-25 | 562,325 |
| transformer | 2024 | 10 | +0.4892±0.0056 | [+0.480, +0.496] | 0.7548 | 22-28 | 608,526 |

GATE P1: PASS (all 10-seed complete, no degenerate references)

## Noted deviations vs v1 (recorded before proceeding)
- transformer/segrnn at v1-comparable IC. itransformer fold2024 0.22
  (v1 ~0.49) and patchtst ~0.17 (v1 ~0.3-0.4): joint effect of fixed
  5y windows + new HPs (itransformer's tuning surface was flat to
  5e-5). Not blocking: all claims are within-cell contrasts; models
  are healthy and seed-stable. PAPER NOTE: v1's "at equal skill"
  phrasing must NOT be reused in v2 — architectures differ in skill
  under the uniform-window design; report per-arch IC in T1.
