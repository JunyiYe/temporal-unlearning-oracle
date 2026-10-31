"""Boundary/leakage tests for target-date split assignment with cross-boundary context."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from fin_ts_ptq.data.windowing import TimeSeriesWindowDataset
from fin_ts_ptq.experiment import build_datasets

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


rng = np.random.default_rng(0)
dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
assets = [f"A{i}" for i in range(6)]
rows = []
for a in assets:
    vals = rng.standard_normal(len(dates)) * 0.02
    rows.append(pd.DataFrame({"date": dates, "symbol": a, "return": vals}))
panel_csv = pd.concat(rows, ignore_index=True)
csv_path = str(Path(tempfile.mkdtemp()) / "toy_panel.csv")
panel_csv.to_csv(csv_path, index=False)

CTX, FWD = 16, 3
cfg = {
    "data": {
        "kind": "csv_panel",
        "path": csv_path,
        "train_end": "2020-12-31",
        "val_end": "2021-06-30",
        "test_end": "2021-12-31",
    },
    "task": {
        "context_length": CTX,
        "horizon": 1,
        "forward_period": FWD,
        "target_agg": "realized_vol",
        "target_log": True,
        "normalize": "zscore",
        "target_transform": "cross_sectional_zscore",
        "feature_columns": ["value", "ret_ma5", "vol_20"],
    },
}

data = build_datasets(cfg)
train_ds, val_ds, test_ds = data.splits["train"], data.splits["val"], data.splits["test"]
train_end = pd.Timestamp("2020-12-31")
val_end = pd.Timestamp("2021-06-30")
test_end = pd.Timestamp("2021-12-31")

check("train targets <= train_end", (train_ds.meta["date"] <= train_end).all())
check("val targets in (train_end, val_end]",
      (val_ds.meta["date"] > train_end).all() and (val_ds.meta["date"] <= val_end).all())
check("test targets in (val_end, test_end]",
      (test_ds.meta["date"] > val_end).all() and (test_ds.meta["date"] <= test_end).all())

first_val = val_ds.meta["date"].min()
first_test = test_ds.meta["date"].min()
bdays_val = len(pd.bdate_range(train_end, first_val)) - 1
bdays_test = len(pd.bdate_range(val_end, first_test)) - 1
check("val starts right after boundary", bdays_val <= FWD, f"{first_val.date()} ({bdays_val} bdays)")
check("test starts right after boundary", bdays_test <= FWD, f"{first_test.date()} ({bdays_test} bdays)")

for name, ds in [("val", val_ds), ("test", test_ds)]:
    per_day = ds.meta.groupby("date").size()
    check(f"{name} has all assets every day", (per_day == len(assets)).all(),
          f"min={per_day.min()} days={len(per_day)}")

w0 = val_ds.meta["date"].idxmin()
target0 = val_ds.meta.loc[w0, "date"]
t_idx = int(np.where(dates == target0)[0][0])
last_ctx_date = dates[t_idx - FWD]
check("earliest val window context ends in train period", last_ctx_date <= train_end,
      f"last ctx {last_ctx_date.date()}")

# z-scored targets for interior dates identical to old within-split construction
from fin_ts_ptq.data.features import add_return_features, compute_panel_stats, normalize_panel
from fin_ts_ptq.data.panel import PanelSpec, chronological_split, load_csv_panel

raw_panel = load_csv_panel(PanelSpec(path=csv_path))
raw_panel = add_return_features(raw_panel, feature_columns=("value", "ret_ma5", "vol_20"))
tr_df, va_df, te_df = chronological_split(raw_panel, "2020-12-31", "2021-06-30", "2021-12-31")
stats = compute_panel_stats(tr_df, ["value", "ret_ma5", "vol_20"])
te_df = normalize_panel(te_df, stats, ["value", "ret_ma5", "vol_20"])
old_test = TimeSeriesWindowDataset(te_df, data.window_config)
deep_date = sorted(set(old_test.meta["date"]))[old_test.meta["date"].nunique() // 2]
old_mask = (old_test.meta["date"] == deep_date).to_numpy()
new_mask = (test_ds.meta["date"] == deep_date).to_numpy()
old_y = old_test.y[old_mask][np.argsort(old_test.meta.loc[old_mask, "asset"].to_numpy())]
new_y = test_ds.y[new_mask][np.argsort(test_ds.meta.loc[new_mask, "asset"].to_numpy())]
check("interior-date targets match old construction", np.allclose(old_y, new_y, atol=1e-6),
      f"date={deep_date.date()}")
check("new test covers more days than old",
      test_ds.meta["date"].nunique() > old_test.meta["date"].nunique())

data_t = build_datasets(cfg, splits=("test",))
tt = data_t.splits["test"]
check("trimmed build matches full build", len(tt) == len(test_ds)
      and np.allclose(tt.y, test_ds.y, atol=1e-7) and np.allclose(tt.x, test_ds.x, atol=1e-7))

print()
print("FAILED:" if failures else "ALL TESTS PASSED", failures if failures else "")
sys.exit(1 if failures else 0)
