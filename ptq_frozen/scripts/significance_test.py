from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_datasets
from fin_ts_ptq.metrics.significance import daily_ic_frame, newey_west_mean_test
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.training.loop import predict


def block_bootstrap_mse_gain(
    meta: pd.DataFrame,
    target_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 1,
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Circular moving-block bootstrap over trading days.

    block_size should be >= forward_period so that resampled blocks don't
    artificially split apart days whose targets overlap.
    """
    target_true = target_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    baseline_se = target_true**2
    model_se = (target_true - y_pred) ** 2
    gain = baseline_se - model_se  # positive => model beats zero baseline

    dates = meta["date"].to_numpy()
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    idx_by_date = {d: np.where(dates == d)[0] for d in unique_dates}

    n_blocks = int(np.ceil(n_dates / block_size))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        block_starts = rng.integers(0, n_dates, size=n_blocks)
        sampled_dates = []
        for s in block_starts:
            sampled_dates.extend(unique_dates[(s + np.arange(block_size)) % n_dates])
        sampled_dates = sampled_dates[:n_dates]
        idxs = np.concatenate([idx_by_date[d] for d in sampled_dates])
        boot_means[b] = gain[idxs].mean()

    point = float(gain.mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = cfg["task"]
    forward_period = int(task_cfg.get("forward_period", 1))
    hac_lag = max(0, forward_period - 1)  # Hansen-Hodrick rule for k-period overlapping forecasts

    data = build_datasets(cfg, splits=("test",))
    test_ds = data.splits["test"]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

    model = build_model(cfg["model"], task_cfg, num_features=len(data.feature_columns))
    checkpoint_path = Path("outputs") / f"{Path(args.config).stem}_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_true, y_pred = predict(model, test_loader, device)

    daily_ic = daily_ic_frame(test_ds.meta, target_true, {"pred": y_pred})["pred"].to_numpy()
    mean_ic, tstat_nw, pval_nw, se_nw = newey_west_mean_test(daily_ic, lag=hac_lag)
    # naive (i.i.d.-days) Fama-MacBeth SE, reported alongside for comparison
    se_naive = float(daily_ic.std(ddof=1) / np.sqrt(len(daily_ic)))
    tstat_naive = mean_ic / se_naive if se_naive > 0 else 0.0
    pval_naive = float(2 * stats.t.sf(abs(tstat_naive), df=len(daily_ic) - 1))

    block_size = max(1, forward_period)
    gain_point, gain_lo, gain_hi = block_bootstrap_mse_gain(
        test_ds.meta, target_true, y_pred, block_size=block_size, n_boot=args.n_boot
    )

    report = {
        "config": args.config,
        "forward_period": forward_period,
        "hac_lag": hac_lag,
        "n_test_obs": int(len(target_true)),
        "n_test_days": int(len(daily_ic)),
        "daily_ic_mean": mean_ic,
        "daily_ic_tstat_hac": tstat_nw,
        "daily_ic_pvalue_hac": pval_nw,
        "daily_ic_tstat_naive": tstat_naive,
        "daily_ic_pvalue_naive": pval_naive,
        "mse_gain_vs_zero_mean": gain_point,
        "mse_gain_vs_zero_ci95": [gain_lo, gain_hi],
        "mse_gain_significant": bool(gain_lo > 0 or gain_hi < 0),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
