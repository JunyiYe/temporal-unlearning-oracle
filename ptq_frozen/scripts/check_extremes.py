import sys
sys.path.insert(0, "src")

import numpy as np
from fin_ts_ptq.config import load_config
from fin_ts_ptq.data.panel import PanelSpec, load_csv_panel, chronological_split
from fin_ts_ptq.data.features import add_return_features, compute_panel_stats, normalize_panel

cfg = load_config(sys.argv[1])
data_cfg = cfg["data"]

panel = load_csv_panel(PanelSpec(path=data_cfg["path"], min_history=int(data_cfg.get("min_history", 0))))
panel = add_return_features(panel)
train_df, val_df, test_df = chronological_split(panel, train_end=data_cfg["train_end"], val_end=data_cfg["val_end"], test_end=data_cfg.get("test_end"))

feat_cols = list(cfg["task"].get("feature_columns", ["value"]))
stats = compute_panel_stats(train_df, feat_cols)
train_norm = normalize_panel(train_df, stats, feat_cols, clip_range=None)

for name, df in [("train", train_norm)]:
    print(f"\n=== {name} (z-scored, no clip) ===")
    arr = df[feat_cols].values
    print(f"{'feature':<15} {'max|val|':<10} {'% > 3':<10}")
    print("-" * 35)
    for i, c in enumerate(feat_cols):
        col = arr[:, i]
        max_abs = np.nanmax(np.abs(col))
        pct = np.mean(np.abs(col) > 3) * 100
        print(f"{c:<15} {max_abs:<10.2f} {pct:<10.2f}")
