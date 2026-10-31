from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fin_ts_ptq.backtest.long_short import long_short_backtest
from fin_ts_ptq.config import load_config
from fin_ts_ptq.data.panel import PanelSpec, chronological_split, load_csv_panel
from fin_ts_ptq.data.windowing import TimeSeriesWindowDataset, WindowConfig
from fin_ts_ptq.metrics.finance import summarize_finance
from fin_ts_ptq.metrics.forecast import summarize_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Check model metrics against simple baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    d = cfg["data"]
    panel = load_csv_panel(
        PanelSpec(
            path=d["path"],
            date_col=d.get("date_col", "date"),
            asset_col=d.get("asset_col", "symbol"),
            value_col=d.get("value_col", "return"),
            min_history=int(d.get("min_history", 0)),
        )
    )
    train_df, val_df, test_df = chronological_split(panel, d["train_end"], d["val_end"], d.get("test_end"))
    w = WindowConfig(
        context_length=int(cfg["task"]["context_length"]),
        horizon=int(cfg["task"]["horizon"]),
        normalize=cfg["task"].get("normalize", "zscore"),
    )
    train_ds = TimeSeriesWindowDataset(train_df, w)
    val_ds = TimeSeriesWindowDataset(val_df, w)
    test_ds = TimeSeriesWindowDataset(test_df, w)

    print("split,target_mean,target_std,mse_zero,mae_zero,pos_rate,n")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        y = ds.y.reshape(-1)
        print(
            f"{name},{y.mean():.10g},{y.std():.10g},{np.mean(y * y):.10g},"
            f"{np.mean(np.abs(y)):.10g},{np.mean(y > 0):.10g},{len(y)}"
        )

    baselines = {
        "zero": np.zeros_like(test_ds.y),
        "train_mean": np.full_like(test_ds.y, float(train_ds.y.mean())),
        "val_mean": np.full_like(test_ds.y, float(val_ds.y.mean())),
    }
    print("\nbaseline,metric,value")
    for name, pred in baselines.items():
        out = {}
        out.update(summarize_forecast(test_ds.y, pred))
        out.update(summarize_finance(test_ds.y, pred))
        out.update(long_short_backtest(test_ds.meta, test_ds.y, pred))
        for key in ["mse", "mae", "direction_accuracy", "ic", "rank_ic", "sharpe", "portfolio_return_mean", "max_drawdown"]:
            print(f"{name},{key},{out[key]:.10g}")

    print("\nmethod,metric,value,delta_vs_train_mean")
    train_mean_metrics = {}
    train_mean_metrics.update(summarize_forecast(test_ds.y, baselines["train_mean"]))
    train_mean_metrics.update(summarize_finance(test_ds.y, baselines["train_mean"]))
    for method, payload in metrics["methods"].items():
        overall = payload["overall"]
        for key in ["mse", "mae", "direction_accuracy", "ic", "rank_ic", "sharpe"]:
            base = train_mean_metrics.get(key, float("nan"))
            value = overall[key]
            delta = value - base if np.isfinite(base) else float("nan")
            print(f"{method},{key},{value:.10g},{delta:.10g}")


if __name__ == "__main__":
    main()
