"""P1: train reference models (fixed five-year windows).

One invocation trains references for (config, fold) over the given
seeds, saves checkpoints in sweep naming, and records an inline
test-IC health metric for the P1 gate.

  python scripts/train_reference.py --config <config yaml> --fold 2024 --seeds 4,5,6
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_datasets
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.runner import read_git_commit
from fin_ts_ptq.training.loop import TrainConfig, predict, train_model
from fin_ts_ptq.utils.device import resolve_device
from fin_ts_ptq.utils.seed import set_seed

UNL = (_REPO)
TRAIN_YEARS = 5


def daily_ic(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    df = pd.DataFrame({"d": meta["date"].to_numpy(), "y": y.ravel(), "p": p.ravel()})
    per_day = df.groupby("d").apply(lambda g: g["y"].corr(g["p"]))
    return float(per_day.dropna().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out-dir", default=str(UNL / "outputs/sweep"))
    args = ap.parse_args()

    base = load_config(args.config)
    stem = Path(args.config).stem
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")

    for seed in [int(s) for s in args.seeds.split(",") if s]:
        label = f"{stem}_fold{args.fold}_seed{seed}"
        ckpt = out / f"{label}_model.pt"
        meta_path = out / f"{label}_refmetrics.json"
        if ckpt.exists() and meta_path.exists():
            print(f"[SKIP] {label}", flush=True)
            continue
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["data"]["train_start"] = f"{args.fold - 2 - TRAIN_YEARS}-12-31"
        cfg["data"]["train_end"] = f"{args.fold - 2}-12-31"
        cfg["data"]["val_end"] = f"{args.fold - 1}-12-31"
        cfg["data"]["test_end"] = f"{args.fold}-12-31"

        set_seed(seed)
        data = build_datasets(cfg)
        nf = len(data.feature_columns)
        model = build_model(cfg["model"], cfg["task"], num_features=nf)
        tc = cfg.get("train", {})
        train_cfg = TrainConfig(
            batch_size=int(tc.get("batch_size", 1024)),
            epochs=int(tc.get("epochs", 300)),
            lr=float(tc.get("lr", 5e-4)),
            weight_decay=float(tc.get("weight_decay", 0.01)),
            patience=int(tc.get("patience", 20)),
            log_every=int(tc.get("log_every", 1)),
        )
        bs = train_cfg.batch_size
        t0 = time.perf_counter()
        res = train_model(
            model,
            DataLoader(data.splits["train"], batch_size=bs, shuffle=True),
            DataLoader(data.splits["val"], batch_size=bs, shuffle=False),
            train_cfg, device)
        yt, pt = predict(model, DataLoader(data.splits["test"], batch_size=bs, shuffle=False), device)
        ic = daily_ic(data.splits["test"].meta, yt, pt)
        torch.save(model.state_dict(), ckpt)
        meta_path.write_text(json.dumps({
            "label": label,
            "train_start": cfg["data"]["train_start"],
            "n_train": len(data.splits["train"]),
            "best_val_loss": res.best_val_loss,
            "epochs_ran": res.epochs_ran,
            "test_daily_ic": ic,
            "seconds": time.perf_counter() - t0,
            "git": {"fin_ts_ptq": read_git_commit("."),
                    "fin_unlearning": read_git_commit(UNL)},
        }, indent=2), encoding="utf-8")
        print(f"[REF] {label}: ic={ic:+.4f} val={res.best_val_loss:.4f} "
              f"epochs={res.epochs_ran} n_train={len(data.splits['train'])}", flush=True)


if __name__ == "__main__":
    main()
