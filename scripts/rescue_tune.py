"""Rescue tuning for the two under-fitting architectures.

The tuning grid searched d_model x dropout only, with the learning rate fixed at
5e-4 for every architecture. That grid barely moved iTransformer or PatchTST
(validation-loss spread 0.0012-0.0015 across six configs, versus 0.0065 for
the transformer), and both land far below a HAR baseline at test time. Before
attributing anything to "architecture", they need a search over the knobs
that actually bind for them: the learning rate, and the structural parameter
each architecture is sensitive to (patch geometry for PatchTST, depth for
iTransformer).

Protocol-legal: identical era-clean tuning split to P0 (train 2010-2014,
validate 2015, seeds 1-3, yf2008 panel). No test year is touched.

    python scripts/rescue_tune.py --model patchtst
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_datasets
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.runner import pick_keys
from fin_ts_ptq.training.loop import TrainConfig, train_model
from fin_ts_ptq.utils.device import resolve_device
from fin_ts_ptq.utils.seed import set_seed

LRS = [1e-4, 5e-4, 1e-3, 3e-3]
# structural axis per architecture: (config keys, list of value tuples)
STRUCT = {
    "patchtst": (("patch_length", "stride"), [(4, 2), (8, 4), (16, 8), (16, 16)]),
    "itransformer": (("n_layers",), [(2,), (3,), (4,)]),
    "transformer": (("n_layers",), [(2,), (3,)]),
    "segrnn": (("n_layers",), [(2,), (3,)]),
    # tsmixer had no P0 tuning, so it gets the wider structural axis
    "tsmixer": (("d_model", "n_layers", "dropout"),
                [(64, 2, 0.1), (64, 2, 0.3), (128, 2, 0.1), (64, 3, 0.1)]),
    "timemixer": (("d_model", "n_layers", "dropout"),
                  [(64, 2, 0.1), (64, 2, 0.3), (128, 2, 0.1), (64, 3, 0.1)]),
    # DLinear has no capacity knobs; its only structural axis is the
    # trend/seasonal decomposition window
    "dlinear": (("moving_avg",), [(5,), (13,), (25,), (51,)]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="the tuned config for this arch")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", default=None, help="comma list; default = the full LR axis")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["cell"])

    base = load_config(args.config)
    data_cfg = copy.deepcopy(base)
    data_cfg["data"].update(path="data/sp500_yf2008_returns.csv",
                            train_start="2009-12-31", train_end="2014-12-31",
                            val_end="2015-12-31", test_end="2015-12-31")
    print(f"[DATA] tuning split train (2009-12-31, 2014-12-31], val<=2015-12-31", flush=True)
    data = build_datasets(data_cfg, splits=("train", "val"))
    train_ds, val_ds = data.splits["train"], data.splits["val"]
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)}", flush=True)

    keys, values = STRUCT[args.model]
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LRS
    combos = list(itertools.product(lrs, values))
    print(f"[PLAN] {len(combos)} configs x {len(seeds)} seeds", flush=True)

    with out.open("a", encoding="utf-8") as sink:
        for i, (lr, struct) in enumerate(combos, 1):
            for seed in seeds:
                tag = "_".join(f"{k}{v}" for k, v in zip(keys, struct))
                cell = f"{args.model}_lr{lr:g}_{tag}_seed{seed}"
                if cell in done:
                    print(f"[SKIP] {cell}", flush=True)
                    continue
                cfg = copy.deepcopy(base)
                cfg["seed"] = seed
                cfg["model"].update(dict(zip(keys, struct)))
                cfg["train"]["lr"] = lr
                set_seed(seed)
                train_cfg = TrainConfig(**pick_keys(cfg.get("train", {}), TrainConfig.__annotations__.keys()))
                bs = int(cfg.get("train", {}).get("batch_size", train_cfg.batch_size))
                device = resolve_device("auto")
                model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
                res = train_model(
                    model,
                    DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False),
                    DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False),
                    train_cfg, device)
                rec = {"cell": cell, "model": args.model, "lr": lr,
                       **dict(zip(keys, struct)), "seed": seed,
                       "n_params": sum(p.numel() for p in model.parameters()),
                       "best_val_loss": res.best_val_loss, "epochs_ran": res.epochs_ran}
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                print(f"[{i}/{len(combos)}] {cell}: val={res.best_val_loss:.4f} ep={res.epochs_ran}", flush=True)


if __name__ == "__main__":
    main()
