"""Era (calendar-year) forget units on top of fin_ts_ptq window datasets.

A licensed data era E is one calendar year inside the training period.
Because samples are overlapping sliding windows, "delete era E" is ambiguous;
we implement two membership levels (the window-overlap gradient the paper
formalizes):

  L0 "target": a window is forgotten iff its LABEL uses E — any of the
      forward_period days that generate the target falls inside E.
  L1 "strict": a window is forgotten iff ANY timestep it touches (context
      day or forward day) falls inside E, and the panel z-score
      normalization statistics are recomputed on training rows excluding E
      (an exact-deletion pipeline never saw E at all, including in its
      preprocessing constants).

  L2 (documented, not implemented): rolling derived features (ret_ma20,
      vol_20, rsi_14, macd, min/max/skew_20) embed raw E returns for up to
      ~26 trading days after E ends, so a maximal reading of deletion
      extends the halo further and recomputes features. The pilot measures
      L0 vs L1 first; L2 is a paper section, not a pilot blocker.

Dataset assembly mirrors fin_ts_ptq.experiment.build_datasets step for step
(windows built on the full panel, split by TARGET date) so that retained
windows are bit-identical to what the reference sweep checkpoints saw.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fin_ts_ptq.data.features import add_return_features, compute_panel_stats, normalize_panel
from fin_ts_ptq.data.panel import PanelSpec, chronological_split, load_csv_panel
from fin_ts_ptq.data.windowing import TimeSeriesWindowDataset, WindowConfig
from fin_ts_ptq.experiment import parse_range

MODES = ("target", "strict", "l2halo")
# l2halo: L2 lower bound — strict membership plus the rolling-feature halo
# (~26 trading days ≈ 38 calendar days past the era end during which
# derived features still embed raw era returns). Feature recomputation
# itself is NOT done (that would be full L2); documented as a lower bound.
L2_HALO = np.timedelta64(38, "D")


def build_feature_panel(cfg: dict[str, Any]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Raw panel + derived features, identical to fin_ts_ptq.experiment."""
    data_cfg = cfg["data"]
    if data_cfg.get("kind") != "csv_panel":
        raise ValueError("Only data.kind=csv_panel is supported.")
    task_cfg = cfg["task"]
    panel = load_csv_panel(
        PanelSpec(
            path=data_cfg["path"],
            date_col=data_cfg.get("date_col", "date"),
            asset_col=data_cfg.get("asset_col", "symbol"),
            value_col=data_cfg.get("value_col", "return"),
            min_history=int(data_cfg.get("min_history", 0)),
        )
    )
    feat_cols = tuple(task_cfg.get("feature_columns", ["value"]))
    if feat_cols != ("value",):
        panel = add_return_features(panel, feature_columns=feat_cols)
    return panel, feat_cols


def window_spans(
    panel: pd.DataFrame,
    ds: TimeSeriesWindowDataset,
    window_cfg: WindowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window (context_start, first_forward_day, target_day) dates.

    make_windows only records the target date; the other two are recovered
    from each asset's trading-day array in the SAME panel the windows were
    built on. Positions: target_idx = end + fwd - 1, forward days are
    dates[end : end+fwd], context is dates[end-ctx : end].
    """
    ctx, fwd = window_cfg.context_length, window_cfg.forward_period
    sorted_panel = panel.sort_values(["asset", "date"])
    asset_dates = {
        a: g["date"].to_numpy(dtype="datetime64[ns]")
        for a, g in sorted_panel.groupby("asset", sort=False)
    }
    tgt = ds.meta["date"].to_numpy(dtype="datetime64[ns]")
    assets = ds.meta["asset"].to_numpy()
    ctx_start = np.empty(len(ds), dtype="datetime64[ns]")
    first_fwd = np.empty(len(ds), dtype="datetime64[ns]")
    for asset in np.unique(assets):
        m = assets == asset
        d = asset_dates[asset]
        ti = np.searchsorted(d, tgt[m])
        if not np.array_equal(d[ti], tgt[m]):
            raise ValueError(f"Window target dates not found in panel for {asset}.")
        end = ti - fwd + 1
        if (end - ctx).min() < 0:
            raise ValueError(f"Window context underruns panel start for {asset}.")
        ctx_start[m] = d[end - ctx]
        first_fwd[m] = d[end]
    return ctx_start, first_fwd, tgt


def era_forget_mask(
    ctx_start: np.ndarray,
    first_fwd: np.ndarray,
    tgt: np.ndarray,
    era_start: pd.Timestamp,
    era_end: pd.Timestamp,
    mode: str,
) -> np.ndarray:
    es, ee = np.datetime64(era_start), np.datetime64(era_end)
    if mode == "target":
        # forward window [first_fwd, tgt] overlaps E
        return (tgt >= es) & (first_fwd <= ee)
    if mode == "strict":
        # whole window [ctx_start, tgt] overlaps E
        return (tgt >= es) & (ctx_start <= ee)
    if mode == "l2halo":
        # strict + feature halo: context starting within L2_HALO of era end
        # still consumes era returns through rolling features
        return (tgt >= es) & (ctx_start <= ee + L2_HALO)
    raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")


def subset_by_mask(ds: TimeSeriesWindowDataset, mask: np.ndarray) -> TimeSeriesWindowDataset:
    return type(ds).from_arrays(ds.x[mask], ds.y[mask], ds.raw_y[mask], ds.meta.loc[mask], ds.config)


def build_era_datasets(
    cfg: dict[str, Any],
    era_year: int,
    mode: str,
    placebo: bool = False,
) -> tuple[dict[str, TimeSeriesWindowDataset], dict[str, Any]]:
    """Splits for one forget cell: train_retain / train_forget / val / test.

    In strict mode the normalization statistics exclude era rows, so ALL
    windows (including val/test) are normalized as an era-free pipeline
    would have — that is the exact-deletion semantics, and it means strict
    forget/retain tensors differ slightly from the reference sweep's.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")
    data_cfg, task_cfg = cfg["data"], cfg["task"]
    train_end = pd.Timestamp(data_cfg["train_end"])
    val_end = pd.Timestamp(data_cfg["val_end"])
    test_end = pd.Timestamp(data_cfg["test_end"])
    # optional EXCLUSIVE lower training bound (fixed-width folds). Applied
    # inline so this works under both the frozen (pre-train_start) and the
    # live fin_ts_ptq code.
    train_start = pd.Timestamp(data_cfg["train_start"]) if data_cfg.get("train_start") else None
    era_start = pd.Timestamp(f"{era_year}-01-01")
    era_end = pd.Timestamp(f"{era_year}-12-31")
    if era_end > train_end:
        raise ValueError(f"Era {era_year} must lie inside the training period (<= {train_end.year}).")

    panel, feat_cols = build_feature_panel(cfg)

    panel_norm = task_cfg.get("normalize", "zscore")
    stats_excluded_era = False
    if len(feat_cols) > 1 and panel_norm != "none":
        train_df, _, _ = chronological_split(panel, train_end=train_end, val_end=val_end, test_end=test_end)
        if train_start is not None:
            train_df = train_df[train_df["date"] > train_start]
        if mode in ("strict", "l2halo"):
            train_df = train_df[(train_df["date"] < era_start) | (train_df["date"] > era_end)]
            stats_excluded_era = True
        stats = compute_panel_stats(train_df, list(feat_cols))
        panel = normalize_panel(panel, stats, list(feat_cols))

    use_per_window = len(feat_cols) == 1 and panel_norm != "none"
    window_cfg = WindowConfig(
        context_length=int(task_cfg["context_length"]),
        horizon=1,
        forward_period=int(task_cfg.get("forward_period", 1)),
        normalize=panel_norm if use_per_window else "none",
        target_transform=task_cfg.get("target_transform", "raw"),
        target_clip=parse_range(task_cfg.get("target_clip")),
        feature_columns=feat_cols,
        target_agg=task_cfg.get("target_agg", "sum"),
        target_log=bool(task_cfg.get("target_log", False)),
    )
    panel = panel[panel["date"] <= test_end]
    full_ds = TimeSeriesWindowDataset(panel, window_cfg)
    train_ds = full_ds.subset_by_date(train_start, train_end)
    val_ds = full_ds.subset_by_date(train_end, val_end)
    test_ds = full_ds.subset_by_date(val_end, test_end)

    ctx_start, first_fwd, tgt = window_spans(panel, train_ds, window_cfg)
    fmask = era_forget_mask(ctx_start, first_fwd, tgt, era_start, era_end, mode)

    if placebo:
        # Quantity-matched placebo: TRAIN with the same COUNT of windows
        # removed, but chosen uniformly at random instead of by era.
        # EVALUATION forget set stays the era's windows, so the placebo
        # oracle answers "is the memorization gap just data volume?"
        # (expected: no gap, no skill cost). Target mode only — random
        # window removal has no coherent stats-exclusion semantics.
        if mode != "target":
            raise ValueError("placebo supports mode='target' only.")
        rng = np.random.default_rng(90000 + int(cfg.get("seed", 7)))
        pmask = np.zeros(len(train_ds), dtype=bool)
        pmask[rng.choice(len(train_ds), size=int(fmask.sum()), replace=False)] = True
        datasets = {
            "retain": subset_by_mask(train_ds, ~pmask),
            "forget": subset_by_mask(train_ds, fmask),  # eval on era windows
            "val": val_ds,
            "test": test_ds,
        }
        info = {
            "era": era_year,
            "mode": "placebo",
            "stats_excluded_era": False,
            "n_train": len(train_ds),
            "n_forget": int(fmask.sum()),
            "n_removed_random": int(pmask.sum()),
            "n_retain": int((~pmask).sum()),
            "removed_overlap_with_era": float((pmask & fmask).sum() / max(1, fmask.sum())),
        }
        return datasets, info

    datasets = {
        "retain": subset_by_mask(train_ds, ~fmask),
        "forget": subset_by_mask(train_ds, fmask),
        "val": val_ds,
        "test": test_ds,
    }
    info = {
        "era": era_year,
        "mode": mode,
        "train_start": str(train_start.date()) if train_start is not None else None,
        "stats_excluded_era": stats_excluded_era,
        "n_train": len(train_ds),
        "n_forget": int(fmask.sum()),
        "n_retain": int((~fmask).sum()),
        "forget_fraction": float(fmask.mean()),
        "forget_target_dates": [str(pd.Timestamp(tgt[fmask].min()).date()), str(pd.Timestamp(tgt[fmask].max()).date())] if fmask.any() else None,
    }
    return datasets, info
