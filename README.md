# Temporal Unlearning Oracle

Code, configs, and paper source for our camera-ready study of machine
unlearning for financial forecasters: can approximate unlearning remove a
data era from a trained model, and can removal be verified at financial SNR,
using exact retraining as the oracle?

## Layout

| path | contents |
|---|---|
| `paper/` | camera-ready LaTeX source (`main.tex`, `sections/`, `figs/`, `refs.bib`) and compiled `main.pdf` |
| `src/fin_unlearning/` | unlearning methods, oracle, and audit/detector code |
| `scripts/`, `scripts_v2/` | experiment drivers, SLURM job scripts, aggregation, and figure scripts |
| `configs/` | tuned per-architecture configs (`*_v2.yaml`) |
| `ptq_frozen/` | frozen snapshot of the `fin_ts_ptq` training/eval package the experiments run on |
| `outputs/` | small aggregated result files (`figure_series.json`, etc.) that the figures are built from |
| `docs/` | plan, design log, phase gates, and the master results table |

## Not included

Large artifacts are left out: raw return panels (`ptq_frozen/data`), reference
checkpoints, per-cell predictions/detector arrays (tens of GB), the vendored
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) clone and
Python user packages. `docs/repro_dryrun.md` documents the bit-level
reproduction check; `docs/results_master.md` lists every reported number.

## Notes

Scripts were written for the Hawk SLURM cluster and contain absolute paths
(e.g. `/nfs/storage1/home/ye/fin-unlearning/...`); adjust them to your checkout.
Figures can be rebuilt from `outputs/figure_series.json` with
`scripts_v2/make_figures.py`.
