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
(see `forecasting/scripts/apptainer_exec.sh` and `forecasting/env/`).

```bash
pip install -e forecasting          # training/evaluation package (fin_ts_ptq)
export PYTHONPATH=$PWD/src:$PYTHONPATH   # unlearning methods (src/fin_unlearning)
# plus Time-Series-Library (https://github.com/thuml/Time-Series-Library),
# cloned to forecasting/Time-Series-Library
```

Create the log directory once, then run the phases in order:

```bash
mkdir -p outputs/slurm_logs figs

sbatch scripts/train_reference.sbatch   # reference models
sbatch scripts/pilot.sbatch             # oracles + unlearning methods (era_pilot.py)
sbatch scripts/detector.sbatch          # audit arrays (loss_gap_detector.py)
python scripts/make_figures.py          # figures from results/figure_series.json
```

## What's inside

```text
src/fin_unlearning/    Unlearning methods (SCRUB, hinge, ...) and era-deletion logic
scripts/               Experiment drivers, SLURM jobs, aggregation, and figure scripts
configs/               Tuned per-architecture configs (`<arch>_sp500_vol.yaml`)
forecasting/           Training and evaluation package the experiments run on (frozen)
data/                  S&P 500 daily returns panel used in the paper
results/               Aggregated outputs the paper's figures are built from
paper/                 Paper PDF
```

## Notes

- Raw checkpoints and per-cell predictions (tens of GB) are not included;
  `results/` holds only the aggregated series. One cell retrained from
  scratch matched the stored result to six decimals.
- Paths are relative to the repo root: submit jobs from there. The `#SBATCH`
  partition and node flags and the container paths in
  `forecasting/scripts/apptainer_exec.sh` are Hawk-specific; edit them for your
  cluster. Outputs go to `outputs/` (git-ignored); `forecasting/data` is a
  symlink to `data/`.
- `data/` holds the S&P 500 panel used for the main experiments. Hyperparameter
  tuning (`tune.sbatch`) also reads `data/sp500_yf2008_returns.csv` (70 MB, not
  shipped). Price data derive from Kaggle and Yahoo Finance and are provided
  for research reproducibility only.

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
