# fin-ts-ptq

Post-training quantization experiments for financial time-series models.

This project studies whether standard PTQ methods preserve not only forecasting accuracy, but also finance-specific utility such as rank correlation, long-short portfolio performance, drawdown, turnover, and regime sensitivity.

The main setup is panel time-series forecasting with cross-sectional evaluation: models predict next-day returns from each asset's history, while IC, rank IC, long-short backtests, and regime splits assess decision quality and tail robustness.

For runtime setup on Hawk, see [docs/environment.md](docs/environment.md).

## What’s inside

- Forecasting models: DLinear, Transformer, PatchTST, iTransformer, TimeMixer, TSMixer, SegRNN — all thin adapters around the official [Time-Series-Library](https://github.com/thuml/Time-Series-Library) implementations (`src/fin_ts_ptq/models/`), so PTQ coverage is consistent across models.
- Three task variants per model on the primary sp500 dataset: next-day return, 5-day forward return, and 5-day forward realized volatility (see `docs/configs.md`).
- Quantization methods: fp32 baseline, dynamic INT8, static INT8/INT4, weight-only INT4 (RTN)
- Evaluation: forecast metrics, finance metrics, long-short backtests, regime breakdowns, and HAC-corrected significance testing (`scripts/significance_test.py`)
- Experiment configs for synthetic smoke tests and market-panel runs

## Dataset

See [docs/dataset.md](docs/dataset.md) for a short description of the datasets used in this project.

## Execution Environment

This project runs on the Hawk cluster. The current model set (TSLib classic architectures) runs fine directly on the host `/usr/local/anaconda3` Python — see `scripts/run_sp500.sh`. Apptainer containers are also supported and become necessary if a future model needs a newer torch/transformers than anaconda's 1.13.1. See [docs/environment.md](docs/environment.md) for both paths.

## Repository Layout

```text
configs/                 Experiment configs and ticker lists
env/                     Container and dependency files
paper/                   Paper outline and research notes
scripts/                 Training, evaluation, and data-prep entrypoints
src/fin_ts_ptq/          Library code
tests/                   Focused unit tests
```

## Requirements

- Python 3.7+ (host anaconda) or 3.10+ (Apptainer container)
- PyTorch 1.13+ is sufficient for the current model set; newer models would need PyTorch 2.4+ via the Apptainer container
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (cloned into project root)
- Optional: `yfinance` for download scripts

## Quick Start

The current sp500 experiments run directly via the host anaconda Python and `sbatch` — no container needed (see `docs/environment.md` for why). `scripts/run_sp500.sh` is the entrypoint:

```bash
# variant is empty (next-day return), "weekly" (5-day forward return), or "vol" (5-day forward realized vol)
sbatch -J patchtst_sp500 scripts/run_sp500.sh patchtst
sbatch -J patchtst_sp500_weekly scripts/run_sp500.sh patchtst weekly
sbatch -J patchtst_sp500_vol scripts/run_sp500.sh patchtst vol
```

For models that need a newer torch/transformers than anaconda provides, use the Apptainer path instead — see [docs/experiments.md](docs/experiments.md) for the full container setup workflow:

```bash
bash scripts/srun_apptainer.sh bash -lc '
  export PYTHONUSERBASE=/work/.local/python-userbase
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="/work/src:/work/Time-Series-Library:${PYTHONPATH:-}"
  python scripts/run_experiment.py --config configs/patchtst_sp500_opt.yaml
'
```

## Metrics

The main experiment runner reports:

- Forecast accuracy such as MAE and MSE
- Directional and correlation-based metrics
- Finance-oriented summaries including IC, rank IC, Sharpe, and max drawdown
- Long-short backtest results with transaction costs
- Performance broken out by volatility or absolute-return regimes

## Notes on PTQ

The repo is intentionally focused on financial time-series forecasting rather than LLM compression. It includes a small PTQ evaluation stack designed to compare calibration policies and measure downstream decision impact.

## License

MIT. See [LICENSE](LICENSE).
