"""Appendix hyperparameter sweep: does tuning rescue the approximate methods?

One cell per config (default: the sharpest — era2020 strict fold2024
seed7), cached oracle reused. Grid over neggrad_hinge (lr x alpha) and
scrub (lr x msteps), protocol methods otherwise unchanged. Reports per
variant: test daily IC, forget MSE, gap_closed, retain-sample MSE.

  sbatch -J unl_hp scripts/hp_sweep.sbatch --configs \
      configs/transformer_sp500_vol_tuned.yaml,configs/patchtst_sp500_vol_tuned.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "/nfs/storage1/home/ye/fin-unlearning/src")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fin_ts_ptq.config import load_config
from fin_ts_ptq.metrics.significance import daily_ic_frame
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.device import resolve_device
from fin_ts_ptq.utils.seed import set_seed

from fin_unlearning.era import build_era_datasets
from fin_unlearning.methods import neggrad_plus, scrub, ts_unlearn

UNLEARN_ROOT = Path("/nfs/storage1/home/ye/fin-unlearning")


def fold_cfg(base: dict, test_year: int, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma-separated config paths")
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--era", type=int, default=2020)
    ap.add_argument("--mode", default="strict")
    ap.add_argument("--sweep-dir", default="outputs/sweep")
    ap.add_argument("--pilot-ckpt-dir", default=str(UNLEARN_ROOT / "outputs" / "pilot" / "ckpt"))
    ap.add_argument("--out-dir", default=str(UNLEARN_ROOT / "outputs" / "hp_sweep"))
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    device = resolve_device("auto")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_path in args.configs.split(","):
        base_cfg = load_config(config_path)
        stem = Path(config_path).stem
        cfg = fold_cfg(base_cfg, args.fold, args.seed)
        cell = f"{stem}_era{args.era}_{args.mode}_fold{args.fold}_seed{args.seed}"
        datasets, info = build_era_datasets(cfg, args.era, args.mode)
        batch_size = int(cfg.get("train", {}).get("batch_size", 128))
        num_features = datasets["test"].num_features
        loaders = {
            "retain_train": DataLoader(datasets["retain"], batch_size=batch_size, shuffle=True),
            "forget_train": DataLoader(datasets["forget"], batch_size=batch_size, shuffle=True),
            "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
            "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
            "forget": DataLoader(datasets["forget"], batch_size=batch_size, shuffle=False),
        }
        rng = np.random.default_rng(args.seed)
        sample_idx = rng.choice(len(datasets["retain"]), size=min(len(datasets["forget"]), len(datasets["retain"])), replace=False)
        loaders["retain_sample"] = DataLoader(Subset(datasets["retain"], sample_idx.tolist()), batch_size=batch_size, shuffle=False)

        def load(path):
            m = build_model(cfg["model"], cfg["task"], num_features=num_features)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            return m

        reference = load(Path(args.sweep_dir) / f"{stem}_fold{args.fold}_seed{args.seed}_model.pt")
        oracle = load(Path(args.pilot_ckpt_dir) / f"{cell}_oracle.pt")

        def eval_model(model):
            yt, pt = predict(model, loaders["test"], device)
            frame = daily_ic_frame(datasets["test"].meta, yt, {"m": pt})
            yf, pf = predict(model, loaders["forget"], device)
            yr, pr = predict(model, loaders["retain_sample"], device)
            # TS-Unlearn's own success criterion: forget predictions should be
            # noise-like, i.e. daily IC ON THE FORGET WINDOWS -> 0. The
            # compliance criterion instead wants it ~= the oracle's (positive,
            # via generalization). One number separates the two framings.
            forget_frame = daily_ic_frame(datasets["forget"].meta, yf, {"m": pf})
            return {
                "test_daily_ic_mean": float(frame["m"].mean()),
                "forget_daily_ic_mean": float(forget_frame["m"].mean()) if len(forget_frame) else float("nan"),
                "forget_mse": float(np.mean((pf - yf) ** 2)),
                "retain_sample_mse": float(np.mean((pr - yr) ** 2)),
            }

        ref_m, ora_m = eval_model(reference), eval_model(oracle)
        gap = ora_m["forget_mse"] - ref_m["forget_mse"]
        results = {"cell": cell, "data": info, "reference": ref_m, "oracle": ora_m, "variants": {}}

        grid = []
        for lr in (1e-5, 1e-4, 1e-3):
            for alpha in (0.1, 0.5, 1.0, 2.0):
                grid.append(("neggrad_hinge", {"lr": lr, "alpha": alpha, "hinge_margin": 1.0}))
        for lr in (1e-5, 1e-4, 1e-3):
            for msteps in (1, 3):
                grid.append(("scrub", {"lr": lr, "msteps": msteps}))
        # fairness check: give TS-Unlearn its own lr AND budget axis
        for lr in (1e-5, 1e-4, 1e-3):
            for epochs in (5, 15):
                grid.append(("ts_unlearn", {"lr": lr, "epochs": epochs}))

        for method, hp in grid:
            tag = method + "_" + "_".join(f"{k}{v:g}" for k, v in hp.items())
            set_seed(args.seed)
            model = copy.deepcopy(reference)
            if method == "neggrad_hinge":
                neggrad_plus(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                             device, epochs=args.epochs, **hp)
            elif method == "ts_unlearn":
                ts_unlearn(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                           device, **hp)
            else:
                scrub(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                      device, epochs=args.epochs, **hp)
            m = eval_model(model)
            m["gap_closed"] = (m["forget_mse"] - ref_m["forget_mse"]) / gap if abs(gap) > 1e-9 else None
            m["hp"] = hp
            results["variants"][tag] = m
            print(f"[HP] {cell} {tag}: ic={m['test_daily_ic_mean']:+.4f} "
                  f"closed={m['gap_closed'] if m['gap_closed'] is None else round(m['gap_closed'], 2)} "
                  f"retain={m['retain_sample_mse']:.3f}", flush=True)

        out = out_dir / f"{cell}_hp_sweep.json"
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OK] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
