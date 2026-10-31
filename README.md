# Temporal Unlearning Oracle

**Can You Delete a Year of Market Data? Machine Unlearning Against Exact Retraining Oracles**
Junyi Ye · ICAIF '26 · [paper PDF](paper/icaif26-189.pdf) · [doi:10.1145/3858063.3858875](https://doi.org/10.1145/3858063.3858875)

This repo contains the code, data, and results behind the paper. It asks
what happens to a financial forecaster when one year of its training data
must be deleted, and scores approximate unlearning methods against the
only ground truth that matters: a model retrained from scratch without
that year (the *oracle*).

## Why this paper

When a data license expires, deleting the stored records does not remove
what a trained model has already learned from them. Machine unlearning
tries to remove that influence without retraining. But retraining is the
ground truth, and it is usually too expensive to run, so unlearning
methods are normally scored against proxies such as low accuracy on the
deleted data.

Financial forecasters train in minutes, so here we can afford the real
thing. We run 3,200 paired comparisons between a reference model trained
on all data and an oracle retrained without a deleted period, on an
S&P 500 five-day volatility panel. The grid covers five architectures,
four rolling test folds, five deletable years, and three deletion levels.

What the oracle shows:

- **Deleting the 2020 COVID year is not a cost.** It produces the largest
  memorization gap for every architecture, and removing it improves all
  three deployable models in every fold, most in the 2022 bear market.
- **The target is the oracle, not noise.** An oracle that never saw 2020
  still predicts it at an information coefficient (IC) of 0.51, versus
  0.55 for the reference. Pushing predictions toward noise lowers test skill.
- **Method rankings do not transfer.** Of twelve deployable
  architecture-method pairs, only TSMixer with the hinge method stays near
  the oracle in every fold (74-118% of the gap closed, worst-fold cost of
  0.004 IC). Rankings change across architectures and windows.
- **Auditing is weak.** At the window level the reference beats its oracle
  on at most 69% of deleted windows (chance is 50%), and treating
  stock-level windows as independent inflates the absolute t-statistic by
  a median factor of 1.9.

## Quick start

Experiments run on a SLURM GPU cluster inside an Apptainer container
(see `ptq_frozen/scripts/apptainer_exec.sh` and `ptq_frozen/env/`).

```bash
pip install -e ptq_frozen          # the frozen training/evaluation package
export PYTHONPATH=$PWD/src:$PYTHONPATH   # unlearning methods (src/fin_unlearning)
# plus Time-Series-Library (https://github.com/thuml/Time-Series-Library),
# cloned to ptq_frozen/Time-Series-Library
```

Create the log directory once (`mkdir -p outputs/slurm_logs`) if missing, then follow the phases in order (`pilot.sbatch` wraps `era_pilot.py`, `detector.sbatch` wraps `loss_gap_detector.py`, `train_reference.sbatch` wraps `train_reference.py`):

```bash
sbatch scripts/train_reference.sbatch   # reference models
sbatch scripts/pilot.sbatch             # oracles + unlearning methods
sbatch scripts/detector.sbatch          # audit / detector arrays
mkdir -p figs && python scripts/make_figures.py   # rebuild figures (figs/) from results/figure_series.json
```

## What's inside

```text
src/fin_unlearning/    Unlearning methods (SCRUB, hinge, ...) and era-deletion logic
scripts/               Experiment drivers, SLURM jobs, aggregation, and figure scripts
configs/               Tuned per-architecture configs (`<arch>_sp500_vol.yaml`)
ptq_frozen/            Frozen snapshot of the fin_ts_ptq package the experiments run on
data/                  S&P 500 daily returns panel used in the paper
results/               Aggregated outputs the paper's figures are built from
docs/                  Final hyperparameters and the reproduction check
paper/                 Paper PDF
```

## Notes

- `docs/repro_dryrun.md` documents a bit-level reproduction check: one cell
  retrained from scratch matched the stored result to six decimals.
- Raw checkpoints and per-cell predictions (tens of GB) are not included.
  `results/` holds only the aggregated series.
- All paths are relative to the repo root. Submit SLURM jobs from the repo
  root (`sbatch scripts/<job>.sbatch ...`); logs go to `outputs/slurm_logs/`.
  The `#SBATCH` partition and node flags and the container paths in
  `ptq_frozen/scripts/apptainer_exec.sh` are Hawk-specific; edit them for
  your cluster.
- `outputs/` (git-ignored) is where checkpoints, oracle cells, and detector
  arrays are written. `ptq_frozen/data` is a symlink to `data/`.
- Hyperparameter tuning (`tune.sbatch`) reads `data/sp500_yf2008_returns.csv`
  (70 MB, not shipped); the main experiments use the Kaggle panel in `data/`.
- Yahoo Finance and Kaggle price data are provided for research
  reproducibility only.

## Citation

If you find this repo useful, please consider giving it a star ⭐ and citing the paper:

```bibtex
@inproceedings{ye2026temporalunlearning,
  title     = {Can You Delete a Year of Market Data? Machine Unlearning Against Exact Retraining Oracles},
  author    = {Ye, Junyi},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM},
  doi       = {10.1145/3858063.3858875}
}
```

## License

MIT (see `LICENSE`). The paper PDF is CC BY 4.0.
