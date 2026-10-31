import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Time-Series-Library"))

import pandas as pd
import numpy as np
from fin_ts_ptq.data.features import add_return_features
from fin_ts_ptq.data.panel import PanelSpec, load_csv_panel
from fin_ts_ptq.data.windowing import WindowConfig, make_windows

cfg_path = sys.argv[1]

from fin_ts_ptq.config import load_config
cfg = load_config(cfg_path)
data_cfg = cfg["data"]

panel = load_csv_panel(PanelSpec(
    path=data_cfg["path"],
    date_col=data_cfg.get("date_col", "date"),
    asset_col=data_cfg.get("asset_col", "symbol"),
    value_col=data_cfg.get("value_col", "return"),
    min_history=int(data_cfg.get("min_history", 0)),
))

panel = add_return_features(panel)
print("NaNs per column:")
print(panel.isna().sum())
print()

feat_cols = ["value", "ret_ma5", "ret_ma10", "ret_ma20", "vol_20", "rsi_14"]
a = panel[panel["asset"] == panel["asset"].unique()[0]]
print(f"First asset: {panel['asset'].unique()[0]}, rows={len(a)}")
print(a[["date"] + feat_cols].head(30))
print()

# Check windows for NaN
task_cfg = cfg["task"]
window_cfg = WindowConfig(
    context_length=int(task_cfg["context_length"]),
    horizon=int(task_cfg["horizon"]),
    normalize=task_cfg.get("normalize", "zscore"),
    target_transform=task_cfg.get("target_transform", "raw"),
    feature_columns=tuple(feat_cols),
)
xs, ys, raw_y, meta = make_windows(panel, window_cfg)
print(f"xs shape: {xs.shape}")
print(f"xs has NaN: {np.isnan(xs).any()}")
print(f"ys has NaN: {np.isnan(ys).any()}")
if np.isnan(xs).any():
    nan_idx = np.where(np.isnan(xs))
    print(f"NaN locations (first 10): {list(zip(nan_idx[0][:10], nan_idx[1][:10], nan_idx[2][:10]))}")
