"""Dump per-window test-year predictions of reference and oracle.

Feeds the distortion-direction analysis: with per-window predictions in
hand, the difference pred_ref - pred_ora can be regressed on raw asset
characteristics (recomputed from the returns panel, since the network
inputs are window-normalized) to ask in WHICH DIRECTION training on
2020 bends the learned ranking.

One invocation = one (config, fold) at strict mode across all seeds.
Output: one npz per (config, fold) under v2/outputs/daily_preds/ with
date, asset, y, and pred_{ref,ora}_seed{N} arrays.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/nfs/storage1/home/ye/fin-unlearning/src")
sys.path.insert(0, "/nfs/storage1/home/ye/fin-unlearning/scripts")

import numpy as np
import torch
from torch.utils.data import DataLoader

from era_pilot import fold_cfg  # noqa: E402
from fin_ts_ptq.config import load_config  # noqa: E402
from fin_ts_ptq.models.factory import build_model  # noqa: E402
from fin_ts_ptq.training.loop import predict  # noqa: E402
from fin_ts_ptq.utils.device import resolve_device  # noqa: E402

from fin_unlearning.era import build_era_datasets  # noqa: E402

UNLEARN_ROOT = Path("/nfs/storage1/home/ye/fin-unlearning")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--era", default="2020")
    ap.add_argument("--mode", default="strict")
    ap.add_argument("--seeds", default="4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--sweep-dir",
                    default=str(UNLEARN_ROOT / "v2" / "outputs" / "sweep"))
    ap.add_argument("--ckpt-dir",
                    default=str(UNLEARN_ROOT / "v2" / "outputs" / "pilot" / "ckpt"))
    ap.add_argument("--out-dir",
                    default=str(UNLEARN_ROOT / "v2" / "outputs" / "daily_preds"))
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    device = resolve_device("auto")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    out_path = out_dir / f"{stem}_era{args.era}_{args.mode}_fold{args.fold}_preds.npz"
    if out_path.exists():
        print(f"[SKIP] {out_path.name}", flush=True)
        return

    cfg = fold_cfg(base_cfg, args.fold, seeds[0], train_start_years=5)
    datasets, info = build_era_datasets(cfg, int(args.era), args.mode)
    print(f"[DATA] {info}", flush=True)
    test = datasets["test"]
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                        drop_last=False)

    arrays = {
        "date": test.meta["date"].to_numpy().astype("datetime64[D]").astype(str),
        "asset": test.meta["asset"].to_numpy().astype(str),
    }
    y_saved = False
    for seed in seeds:
        ref = Path(args.sweep_dir) / f"{stem}_fold{args.fold}_seed{seed}_model.pt"
        ora = Path(args.ckpt_dir) / (f"{stem}_era{args.era}_{args.mode}_"
                                     f"fold{args.fold}_seed{seed}_oracle.pt")
        if not (ref.exists() and ora.exists()):
            print(f"[MISS] seed{seed}", flush=True)
            continue
        for tag, path in (("ref", ref), ("ora", ora)):
            model = build_model(cfg["model"], cfg["task"],
                                num_features=test.num_features)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            y, preds = predict(model, loader, device)
            if not y_saved:
                arrays["y"] = np.asarray(y, dtype=np.float32)
                y_saved = True
            arrays[f"pred_{tag}_seed{seed}"] = np.asarray(preds, dtype=np.float32)
        print(f"[OK] seed{seed}", flush=True)
    np.savez_compressed(out_path, **arrays)
    print(f"[WROTE] {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
