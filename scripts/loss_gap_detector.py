"""Per-window loss extraction for the verification chapter (Phase 2 seed).

For every completed pilot cell, loads the cached checkpoints (reference
from the fin-ts-ptq sweep, oracle + approximate methods from the pilot
ckpt dir), recomputes per-window squared errors on the forget set and the
size-matched retain control, and writes one NPZ per cell:

  date_ns, asset, y (targets), plus loss::<model> arrays for both splits.

Also writes a summary JSON with the paired reference-vs-oracle gap
aggregated per day: naive i.i.d. t-stat vs Newey-West t-stat — the spread
between the two is the autocorrelation-inflation number quoted in
docs/verification_mde.md. Runs inside the cluster container:

  sbatch -J unl_detect scripts/era_pilot.sbatch  # NO — use own entry:
  sbatch -p gpu-short -J unl_detect scripts/detector.sbatch [args]
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from fin_ts_ptq.config import load_config
from fin_ts_ptq.metrics.significance import newey_west_mean_test
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.device import resolve_device

from fin_unlearning.era import build_era_datasets

UNLEARN_ROOT = (_REPO)


def fold_cfg(base: dict, test_year: int, seed: int, train_start_years: int = 0) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    if train_start_years > 0:
        cfg["data"]["train_start"] = f"{test_year - 2 - train_start_years}-12-31"
    return cfg


def per_window_losses(model, ds, device, batch_size) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    y, p = predict(model, loader, device)
    return ((p - y) ** 2).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot-dir", default=str(UNLEARN_ROOT / "outputs" / "pilot"))
    ap.add_argument("--sweep-dir", default="outputs/sweep")
    ap.add_argument("--out-dir", default=str(UNLEARN_ROOT / "outputs" / "detector"))
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--train-start-years", type=int, default=0)
    ap.add_argument("--folds", default=None, help="comma list; default all")
    ap.add_argument("--eras", default=None, help="comma list; default all")
    ap.add_argument("--modes", default=None, help="comma list; default all")
    args = ap.parse_args()
    keep_folds = {int(x) for x in args.folds.split(",")} if args.folds else None
    keep_eras = {int(x) for x in args.eras.split(",")} if args.eras else None
    keep_modes = set(args.modes.split(",")) if args.modes else None

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    hac_lag = max(0, int(base_cfg["task"].get("forward_period", 1)) - 1)
    device = resolve_device("auto")
    pilot_dir = Path(args.pilot_dir)
    ckpt_dir = pilot_dir / "ckpt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cache: dict[tuple, dict] = {}  # datasets depend on (fold, era, mode), not seed
    for cell_json in sorted(pilot_dir.glob(f"{stem}_era*_unlearn.json")):
        cell = json.loads(cell_json.read_text(encoding="utf-8"))
        name = cell["cell"]
        out_npz = out_dir / f"{name}_losses.npz"
        out_sum = out_dir / f"{name}_daygap.json"
        if out_npz.exists() and out_sum.exists():
            print(f"[SKIP] {name}", flush=True)
            continue
        fold, era, mode, seed = cell["fold"], cell["era"], cell["mode"], cell["seed"]
        if ((keep_folds and fold not in keep_folds)
                or (keep_eras and era not in keep_eras)
                or (keep_modes and mode not in keep_modes)):
            continue
        cfg = fold_cfg(base_cfg, fold, seed, train_start_years=args.train_start_years)
        key = (fold, era, mode)
        if key not in ds_cache:
            ds_cache[key] = build_era_datasets(cfg, era, mode)[0]
        datasets = ds_cache[key]
        forget, retain = datasets["forget"], datasets["retain"]
        rng = np.random.default_rng(seed)  # same control sample as era_pilot.py
        sample_idx = rng.choice(len(retain), size=min(len(forget), len(retain)), replace=False)
        control = Subset(retain, sample_idx.tolist())
        num_features = forget.num_features

        models = {}
        ref_ckpt = Path(args.sweep_dir) / f"{stem}_fold{fold}_seed{seed}_model.pt"
        model_names = ["reference", "oracle"] + [
            m for m in cell["models"] if m not in ("reference", "oracle")]
        for mname in model_names:
            path = ref_ckpt if mname == "reference" else ckpt_dir / f"{name}_{mname}.pt"
            if not path.exists():
                print(f"[MISS] {name}: {path}", flush=True)
                continue
            model = build_model(cfg["model"], cfg["task"], num_features=num_features)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            models[mname] = model

        arrays = {
            "date_ns": forget.meta["date"].to_numpy().astype("datetime64[ns]").astype(np.int64),
            "asset": forget.meta["asset"].to_numpy().astype(str),
            "y": forget.y.reshape(-1).astype(np.float32),
        }
        for mname, model in models.items():
            arrays[f"forget_loss::{mname}"] = per_window_losses(model, forget, device, args.batch_size).astype(np.float32)
            arrays[f"control_loss::{mname}"] = per_window_losses(model, control, device, args.batch_size).astype(np.float32)
        np.savez_compressed(out_npz, **arrays)

        summary = {"cell": name, "n_forget": len(forget), "models": {}}
        if "reference" in models and "oracle" in models:
            gap = arrays["forget_loss::oracle"] - arrays["forget_loss::reference"]
            days = pd.Series(gap).groupby(pd.to_datetime(arrays["date_ns"])).mean()
            naive_t = float(days.mean() / (days.std(ddof=1) / np.sqrt(len(days)))) if len(days) > 1 else float("nan")
            mean, nw_t, nw_p, se = newey_west_mean_test(days.to_numpy(), lag=hac_lag)
            summary["day_gap"] = {"mean": float(days.mean()), "n_days": int(len(days)),
                                  "naive_t": naive_t, "nw_t": nw_t, "nw_p": nw_p,
                                  "window_frac_ref_better": float((gap > 0).mean())}
        for mname in models:
            summary["models"][mname] = {
                "forget_mean": float(arrays[f"forget_loss::{mname}"].mean()),
                "control_mean": float(arrays[f"control_loss::{mname}"].mean()),
            }
        out_sum.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OK] {name}: day_gap={summary.get('day_gap', {})}", flush=True)


if __name__ == "__main__":
    main()
