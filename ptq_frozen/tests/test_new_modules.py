"""Unit tests for significance metrics, selective features, calibration fix, noise baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch import nn

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


# ---------- 1. daily_ic_frame + paired_delta_tests ----------
from fin_ts_ptq.metrics.significance import daily_ic_frame, newey_west_mean_test, paired_delta_tests

rng = np.random.default_rng(0)
n_days, n_assets = 40, 50
dates = np.repeat(pd.date_range("2024-01-01", periods=n_days, freq="B"), n_assets)
meta = pd.DataFrame({"date": dates, "asset": np.tile([f"A{i}" for i in range(n_assets)], n_days)})
y = rng.standard_normal(n_days * n_assets)
good = 0.5 * y + rng.standard_normal(len(y))
bad = rng.standard_normal(len(y))
frame = daily_ic_frame(meta, y, {"good": good, "bad": bad})
check("daily_ic_frame shape", frame.shape == (n_days, 2), f"got {frame.shape}")
check("daily_ic good>bad", frame["good"].mean() > 0.3 and abs(frame["bad"].mean()) < 0.1)
tests = paired_delta_tests(frame, baseline="good", lag=0)
check("paired delta positive for degraded", tests["bad"]["delta_ic_mean"] > 0.3 and tests["bad"]["pvalue"] < 0.01)

frame2 = daily_ic_frame(meta, y, {"a": good, "b": good.copy()})
t2 = paired_delta_tests(frame2, "a", lag=0)
check("paired delta zero for identical", abs(t2["b"]["delta_ic_mean"]) < 1e-12)

bad_day = bad.copy()
bad_day[:n_assets] = 7.0
frame3 = daily_ic_frame(meta, y, {"good": good, "bad": bad_day})
check("degenerate day dropped for all", frame3.shape == (n_days - 1, 2))

m, t, p, se = newey_west_mean_test(np.ones(100) * 0.1 + rng.standard_normal(100) * 0.01, lag=4)
check("newey_west detects nonzero mean", p < 1e-6 and abs(m - 0.1) < 0.01)

# ---------- 2. features: selective computation ----------
from fin_ts_ptq.data.features import add_return_features

np.random.seed(1)
n_rows = 400
fdf = pd.DataFrame({
    "date": np.tile(pd.date_range("2020-01-01", periods=n_rows, freq="B"), 2),
    "asset": np.repeat(["X", "Y"], n_rows),
    "value": np.random.standard_normal(2 * n_rows) * 0.02,
})
sub = add_return_features(fdf, feature_columns=("value", "ret_ma5", "vol_20", "rsi_14"))
full = add_return_features(fdf)
check("subset keeps more rows than full", len(sub) > len(full))
check("subset drops only its own warmup", len(sub) == 2 * (n_rows - 14))
check("subset has requested cols only", {"ret_ma5", "vol_20", "rsi_14"}.issubset(sub.columns)
      and "mom_12m" not in sub.columns and "macd" not in sub.columns)
check("full has mom_12m", "mom_12m" in full.columns and len(full) == 2 * (n_rows - 272))
try:
    add_return_features(fdf, feature_columns=("value", "nope"))
    check("unknown feature raises", False)
except ValueError:
    check("unknown feature raises", True)
merged = sub.merge(full, on=["date", "asset"], suffixes=("_s", "_f"))
check("shared feature values identical", np.allclose(merged["ret_ma5_s"], merged["ret_ma5_f"])
      and np.allclose(merged["vol_20_s"], merged["vol_20_f"]))

# ---------- 3. calibration: input_window_volatility on 3D windows ----------
from fin_ts_ptq.quantization.calibration import input_window_volatility

n_win, T, F = 100, 64, 11
x3 = np.zeros((n_win, T, F), dtype=np.float32)
true_vol = np.linspace(0.1, 2.0, n_win)
x3[:, :, 0] = true_vol[:, None] * np.sign(np.random.standard_normal((n_win, T)))
x3[:, :, 1:] = 100.0  # huge non-return features that the old flat slice would pick up
v = input_window_volatility(x3, 20)
check("vol proxy uses feature 0 only", np.allclose(v, true_vol, atol=1e-5))

# ---------- 4. noise: perturb_weights_like_rtn ----------
from fin_ts_ptq.quantization.noise import count_perturbed_parameters, perturb_weights_like_rtn, rtn_noise_sigma
from fin_ts_ptq.quantization.rtn import rtn_w4


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.conv = nn.Conv1d(4, 8, 1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


torch.manual_seed(0)
tiny = Tiny()
orig = {k: v.clone() for k, v in tiny.state_dict().items()}
noised = perturb_weights_like_rtn(tiny, num_bits=4, multiplier=1.0, seed=3)
check("original model untouched", all(torch.equal(orig[k], v) for k, v in tiny.state_dict().items()))
check("noised weights differ", not torch.equal(noised.fc1.weight, tiny.fc1.weight)
      and not torch.equal(noised.conv.weight, tiny.conv.weight))
check("bias untouched", torch.equal(noised.fc1.bias, tiny.fc1.bias))
delta = (noised.fc1.weight - tiny.fc1.weight).detach()
sigma = rtn_noise_sigma(tiny.fc1.weight, 4).squeeze()
ratio = (delta.std(dim=1) / sigma).mean().item()
check("noise std matches RTN error scale", 0.85 < ratio < 1.15, f"mean ratio={ratio:.3f}")
noised2 = perturb_weights_like_rtn(tiny, num_bits=4, multiplier=1.0, seed=3)
check("noise reproducible by seed", torch.equal(noised.fc1.weight, noised2.fc1.weight))
noised3 = perturb_weights_like_rtn(tiny, num_bits=4, multiplier=1.0, seed=4)
check("different seed differs", not torch.equal(noised.fc1.weight, noised3.fc1.weight))
q = rtn_w4(tiny)
qerr = (q.fc1.weight_qdq - tiny.fc1.weight).detach().std(dim=1)
qratio = (qerr / sigma).mean().item()
check("RTN error magnitude ~ sigma", 0.5 < qratio < 1.5, f"qerr/sigma={qratio:.3f}")
info = count_perturbed_parameters(tiny)
check("count_perturbed_parameters", info["perturbed_modules"] == 3
      and info["perturbed_weights"] == 64 * 128 + 4 * 8 + 128, str(info))

print()
print("FAILED:" if failures else "ALL TESTS PASSED", failures if failures else "")
sys.exit(1 if failures else 0)
