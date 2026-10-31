"""Aggregate method cells into Table-3 rows (kappa, dIC vs oracle, cost).

Usage: python m8_aggregate.py <stem> [<stem> ...]
Reads v2/outputs/pilot/{stem}_era2020_strict_fold2024_seed{4..13}_unlearn.json.
kappa per Eq. 4; dIC = test_daily_ic_mean(method) - test_daily_ic_mean(oracle);
cost = method wall-clock / oracle retrain wall-clock (oracle timings from the
cell JSON itself if present, else backup_pre_m8/oracle_timings.json).
"""
import json
import statistics
import sys
from pathlib import Path

PILOT = Path("/nfs/storage1/home/ye/fin-unlearning/v2/outputs/pilot")
ORA_T = PILOT / "backup_pre_m8" / "oracle_timings.json"
METHODS = ["finetune", "neggrad_hinge", "scrub", "ts_unlearn"]
SEEDS = range(4, 14)

ora_timings = json.loads(ORA_T.read_text()) if ORA_T.exists() else {}

for stem in sys.argv[1:]:
    rows = {m: {"kappa": [], "dic": [], "cost": []} for m in METHODS}
    missing = []
    for s in SEEDS:
        cell = f"{stem}_era2020_strict_fold2024_seed{s}"
        f = PILOT / f"{cell}_unlearn.json"
        if not f.exists():
            missing.append(cell)
            continue
        d = json.load(open(f))
        mo, tim = d["models"], d.get("timings_seconds", {})
        ref_mse, ora_mse = mo["reference"]["forget_mse"], mo["oracle"]["forget_mse"]
        ora_ic = mo["oracle"]["test_daily_ic_mean"]
        ora_sec = tim.get("oracle") or ora_timings.get(cell)
        for m in METHODS:
            if m not in mo:
                missing.append(f"{cell}:{m}")
                continue
            rows[m]["kappa"].append((mo[m]["forget_mse"] - ref_mse) / (ora_mse - ref_mse))
            rows[m]["dic"].append(mo[m]["test_daily_ic_mean"] - ora_ic)
            if ora_sec and tim.get(m):
                rows[m]["cost"].append(tim[m] / ora_sec)
    print(f"\n=== {stem} (n per method: {len(rows['finetune']['kappa'])}) ===")
    if missing:
        print(f"  MISSING: {missing}")
    for m in METHODS:
        k, dic, c = rows[m]["kappa"], rows[m]["dic"], rows[m]["cost"]
        if not k:
            continue
        ksd = statistics.stdev(k) if len(k) > 1 else 0.0
        cost = f"{statistics.mean(c):.2f}x" if c else "n/a"
        print(f"  {m:14s} kappa {statistics.mean(k):+7.2f} +- {ksd:6.2f}   "
              f"dIC vs ora {statistics.mean(dic):+.3f}   cost {cost}")
