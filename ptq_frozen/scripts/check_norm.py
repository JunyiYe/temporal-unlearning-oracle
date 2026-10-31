import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from fin_ts_ptq.config import load_config
from fin_ts_ptq.data.panel import PanelSpec, load_csv_panel, chronological_split
from fin_ts_ptq.data.features import add_return_features, compute_panel_stats, normalize_panel

cfg = load_config(sys.argv[1])
data_cfg = cfg["data"]

panel = load_csv_panel(PanelSpec(
    path=data_cfg["path"],
    min_history=int(data_cfg.get("min_history", 0)),
))
panel = add_return_features(panel)

train_df, val_df, test_df = chronological_split(
    panel,
    train_end=data_cfg["train_end"],
    val_end=data_cfg["val_end"],
    test_end=data_cfg.get("test_end"),
)

task_cfg = cfg["task"]
feat_cols = list(task_cfg.get("feature_columns", ["value"]))

stats = compute_panel_stats(train_df, feat_cols)
train_norm = normalize_panel(train_df, stats, feat_cols)

print("=== 归一化前训练集统计 ===")
for c in feat_cols:
    v = train_df[c].dropna().values
    print(f"{c:15s}  mean={v.mean():+.6f}  std={v.std():.4f}")

print("\n=== 归一化后训练集统计 (应接近 mean=0, std=1) ===")
for c in feat_cols:
    v = train_norm[c].dropna().values
    print(f"{c:15s}  mean={v.mean():+.6f}  std={v.std():.4f}")

print("\n=== 归一化前 val/test 集统计 ===")
for name, df in [("val", val_df), ("test", test_df)]:
    print(f"\n{name}:")
    for c in feat_cols:
        v = df[c].dropna().values
        print(f"  {c:15s}  mean={v.mean():+.6f}  std={v.std():.4f}")
