"""Era-unlearning pilot: exact-retraining oracle vs approximate methods.

For each (fold, seed, era, mode) cell:
  - reference  = the existing fin-ts-ptq sweep checkpoint (trained WITH era E),
  - oracle     = retrain from scratch on the retain set (exact unlearning),
  - finetune   = keep training the reference on retain only,
  - neggrad    = NegGrad+ plain ascent,
  - neggrad_hinge = NegGrad+ with a forget-loss ceiling at the uninformative level.

Metrics per model: test daily-IC (paired vs oracle and vs reference with
NW-HAC), val/forget/retain-sample MSE (the loss-gap forgetting signal),
prediction-space agreement with the oracle, wall-clock cost.

Run inside the cluster container from the fin-ts-ptq repo root, e.g.
  sbatch -J unl_smoke scripts/pilot.sbatch \
      --config configs/transformer_sp500_vol.yaml \
      --folds 2024 --seeds 7 --eras 2020 --modes target,strict
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

sys.path.insert(0, str(_REPO / "src"))  # container-safe: PYTHONPATH may not survive login-shell init

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fin_ts_ptq.config import load_config
from fin_ts_ptq.metrics.significance import daily_ic_frame, paired_delta_tests
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.runner import read_git_commit
from fin_ts_ptq.training.loop import TrainConfig, predict, train_model
from fin_ts_ptq.utils.device import resolve_device
from fin_ts_ptq.utils.seed import set_seed

from fin_unlearning.era import build_era_datasets
from fin_unlearning.methods import finetune_retain, neggrad_plus, relabel_noise, scrub, ts_unlearn

UNLEARN_ROOT = (_REPO)



def _config_fingerprint(cfg: dict) -> str:
    """Stable hash of the hyperparameters a cell was produced with.

    Cells are only comparable within one fingerprint; a config edit must be
    followed by purging that architecture's cells (scripts/purge_stale_cells.py).
    """
    import hashlib as _h
    payload = json.dumps({k: cfg.get(k) for k in ("model", "train", "task")}, sort_keys=True)
    return _h.sha256(payload.encode()).hexdigest()[:16]


def parse_list(v: str, cast=str) -> list:
    return [cast(p) for p in str(v).split(",") if p != ""]


def fold_cfg(base: dict, test_year: int, seed: int, train_start_years: int = 0) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    if train_start_years > 0:
        cfg["data"]["train_start"] = f"{test_year - 2 - train_start_years}-12-31"
    return cfg


def mse_of(model, loader, device) -> float:
    y, p = predict(model, loader, device)
    return float(np.mean((p - y) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Era-unlearning pilot.")
    parser.add_argument("--config", required=True, help="fin-ts-ptq config (path relative to its repo root)")
    parser.add_argument("--folds", default="2024")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--eras", default="2020")
    parser.add_argument("--modes", default="target,strict")
    parser.add_argument("--methods", default="finetune,neggrad,neggrad_hinge")
    parser.add_argument("--sweep-dir", default="outputs/sweep", help="reference checkpoints (fin-ts-ptq relative)")
    parser.add_argument("--out-dir", default=str(UNLEARN_ROOT / "outputs" / "pilot"))
    parser.add_argument("--ckpt-dir", default=None,
                        help="Checkpoint cache dir (default <out-dir>/ckpt); point at the main pilot's to reuse oracles.")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = use config train.batch_size")
    parser.add_argument("--ft-epochs", type=int, default=20)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--ng-epochs", type=int, default=5)
    parser.add_argument("--ng-lr", type=float, default=1e-4)
    parser.add_argument("--ng-alpha", type=float, default=0.5)
    parser.add_argument("--hinge-margin", type=float, default=1.0)
    parser.add_argument("--placebo", action="store_true",
                        help="Train with a size-matched RANDOM window removal; evaluate on the era windows.")
    parser.add_argument("--train-start-years", type=int, default=0,
                        help="Fixed-width training window of N years (0 = expanding from panel start).")
    parser.add_argument("--train-reference", action="store_true",
                        help="Train the reference from scratch when no sweep checkpoint matches (fixed-width folds).")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    methods = parse_list(args.methods)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    hac_lag = max(0, int(base_cfg["task"].get("forward_period", 1)) - 1)
    git_info = {
        "fin_ts_ptq": read_git_commit("."),
        "fin_unlearning": read_git_commit(UNLEARN_ROOT),
    }
    failures: list[str] = []

    for fold in parse_list(args.folds, int):
        for era in parse_list(args.eras, int):
            for mode in parse_list(args.modes):
                for seed in parse_list(args.seeds, int):
                    tag = f"{mode}_placebo" if args.placebo else mode
                    cell = f"{stem}_era{era}_{tag}_fold{fold}_seed{seed}"
                    out_path = out_dir / f"{cell}_unlearn.json"
                    if args.skip_existing and out_path.exists():
                        print(f"[SKIP] {cell} (metrics exist)", flush=True)
                        continue
                    ref_ckpt = Path(args.sweep_dir) / f"{stem}_fold{fold}_seed{seed}_model.pt"
                    if args.train_start_years > 0 and args.train_reference:
                        # Fixed-width runs against an EXPANDING sweep:
                        # train a per-cell reference. (When the sweep itself is
                        # fixed-width, omit --train-reference and the
                        # sweep checkpoints are used directly.)
                        ref_ckpt = ckpt_dir / f"{cell}_reference.pt"
                    if not ref_ckpt.exists() and not args.train_reference:
                        print(f"[SKIP] {cell}: missing reference {ref_ckpt}", flush=True)
                        continue
                    print(f"[CELL] {cell}", flush=True)
                    try:
                        run_cell(args, base_cfg, stem, fold, era, mode, seed, ref_ckpt,
                                 ckpt_dir, out_path, device, hac_lag, git_info, methods)
                    except Exception as exc:  # keep sweeping; cells are independent
                        failures.append(cell)
                        print(f"[FAIL] {cell}: {exc}", flush=True)

    print(json.dumps({"failures": failures}))
    if failures:
        raise SystemExit(1)


def run_cell(args, base_cfg, stem, fold, era, mode, seed, ref_ckpt,
             ckpt_dir, out_path, device, hac_lag, git_info, methods) -> None:
    cfg = fold_cfg(base_cfg, fold, seed, train_start_years=args.train_start_years)
    datasets, info = build_era_datasets(cfg, era, mode, placebo=args.placebo)
    print(f"[DATA] {info}", flush=True)

    train_cfg_src = cfg.get("train", {})
    batch_size = args.batch_size or int(train_cfg_src.get("batch_size", 128))
    num_features = datasets["test"].num_features
    loaders = {
        "retain_train": DataLoader(datasets["retain"], batch_size=batch_size, shuffle=True, drop_last=False),
        "forget_train": DataLoader(datasets["forget"], batch_size=batch_size, shuffle=True, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, drop_last=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, drop_last=False),
        "forget": DataLoader(datasets["forget"], batch_size=batch_size, shuffle=False, drop_last=False),
    }
    # size-matched retain sample as the loss-gap control group
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(datasets["retain"]), size=min(len(datasets["forget"]), len(datasets["retain"])), replace=False)
    loaders["retain_sample"] = DataLoader(
        Subset(datasets["retain"], sample_idx.tolist()), batch_size=batch_size, shuffle=False, drop_last=False)

    def fresh_model():
        return build_model(cfg["model"], cfg["task"], num_features=num_features)

    models: dict[str, torch.nn.Module] = {}
    timings: dict[str, float] = {}
    method_stats: dict[str, dict] = {}

    reference = fresh_model()
    if ref_ckpt.exists():
        reference.load_state_dict(torch.load(ref_ckpt, map_location="cpu"))
    else:  # fixed-width fold with --train-reference: train on the FULL window
        from torch.utils.data import ConcatDataset
        set_seed(seed)
        reference = fresh_model()
        full_loader = DataLoader(ConcatDataset([datasets["retain"], datasets["forget"]]),
                                 batch_size=batch_size, shuffle=True, drop_last=False)
        ref_train_cfg = TrainConfig(
            batch_size=batch_size,
            epochs=int(train_cfg_src.get("epochs", 20)),
            lr=float(train_cfg_src.get("lr", 5e-4)),
            weight_decay=float(train_cfg_src.get("weight_decay", 1e-4)),
            patience=int(train_cfg_src.get("patience", 5)),
            log_every=int(train_cfg_src.get("log_every", 1)),
        )
        t0 = time.perf_counter()
        res = train_model(reference, full_loader, loaders["val"], ref_train_cfg, device)
        timings["reference"] = time.perf_counter() - t0
        method_stats["reference"] = {"epochs_ran": res.epochs_ran, "best_val_loss": res.best_val_loss}
        torch.save(reference.state_dict(), ref_ckpt)
    models["reference"] = reference

    cell = out_path.stem.replace("_unlearn", "")
    oracle_ckpt = ckpt_dir / f"{cell}_oracle.pt"
    oracle = fresh_model()
    if oracle_ckpt.exists():
        print(f"[LOAD] oracle from {oracle_ckpt}", flush=True)
        oracle.load_state_dict(torch.load(oracle_ckpt, map_location="cpu"))
    else:
        set_seed(seed)
        oracle = fresh_model()
        train_cfg = TrainConfig(
            batch_size=batch_size,  # informational only; loaders are prebuilt
            epochs=int(train_cfg_src.get("epochs", 20)),
            lr=float(train_cfg_src.get("lr", 5e-4)),
            weight_decay=float(train_cfg_src.get("weight_decay", 1e-4)),
            patience=int(train_cfg_src.get("patience", 5)),
            log_every=int(train_cfg_src.get("log_every", 1)),
        )
        t0 = time.perf_counter()
        res = train_model(oracle, loaders["retain_train"], loaders["val"], train_cfg, device)
        timings["oracle"] = time.perf_counter() - t0
        method_stats["oracle"] = {"epochs_ran": res.epochs_ran, "best_val_loss": res.best_val_loss}
        torch.save(oracle.state_dict(), oracle_ckpt)
    models["oracle"] = oracle

    for method in methods:
        set_seed(seed)
        model = copy.deepcopy(reference)
        if method == "finetune":
            stats = finetune_retain(model, loaders["retain_train"], loaders["val"], device,
                                    epochs=args.ft_epochs, lr=args.ft_lr,
                                    weight_decay=float(train_cfg_src.get("weight_decay", 0.01)))
        elif method in ("neggrad", "neggrad_hinge"):
            stats = neggrad_plus(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                                 device, epochs=args.ng_epochs, lr=args.ng_lr, alpha=args.ng_alpha,
                                 hinge_margin=args.hinge_margin if method == "neggrad_hinge" else None)
        elif method == "scrub":
            stats = scrub(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                          device, epochs=args.ng_epochs, lr=args.ng_lr)
        elif method == "relabel_noise":
            stats = relabel_noise(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                                  device, epochs=args.ng_epochs, lr=args.ng_lr, alpha=args.ng_alpha)
        elif method == "ts_unlearn":
            stats = ts_unlearn(model, loaders["retain_train"], loaders["forget_train"], loaders["val"],
                               device, epochs=args.ng_epochs, lr=args.ng_lr)
        else:
            raise ValueError(f"Unknown method {method!r}")
        timings[method] = stats.pop("seconds")
        method_stats[method] = stats
        torch.save(model.state_dict(), ckpt_dir / f"{cell}_{method}.pt")
        models[method] = model

    # ---- evaluation ----
    test_preds: dict[str, np.ndarray] = {}
    per_model: dict[str, dict] = {}
    y_test = None
    for name, model in models.items():
        y_test, test_preds[name] = predict(model, loaders["test"], device)
    frame = daily_ic_frame(datasets["test"].meta, y_test, test_preds)
    vs_oracle = paired_delta_tests(frame, baseline="oracle", lag=hac_lag)
    vs_reference = paired_delta_tests(frame, baseline="reference", lag=hac_lag)

    yf, oracle_forget_pred = predict(models["oracle"], loaders["forget"], device)
    for name, model in models.items():
        _, pf = predict(model, loaders["forget"], device)
        fic_frame = daily_ic_frame(datasets["forget"].meta, yf, {"m": pf})
        per_model[name] = {
            "test_daily_ic_mean": float(frame[name].mean()),
            "forget_daily_ic_mean": float(fic_frame["m"].mean()) if len(fic_frame) else None,
            "val_mse": mse_of(model, loaders["val"], device),
            "forget_mse": float(np.mean((pf - yf) ** 2)),
            "retain_sample_mse": mse_of(model, loaders["retain_sample"], device),
            "test_pred_corr_oracle": float(np.corrcoef(test_preds[name].ravel(), test_preds["oracle"].ravel())[0, 1]),
            "test_pred_rmse_oracle": float(np.sqrt(np.mean((test_preds[name] - test_preds["oracle"]) ** 2))),
            "forget_pred_corr_oracle": float(np.corrcoef(pf.ravel(), oracle_forget_pred.ravel())[0, 1]),
            "vs_oracle": vs_oracle.get(name),
            "vs_reference": vs_reference.get(name),
        }
        print(f"[EVAL] {name}: ic={per_model[name]['test_daily_ic_mean']:+.4f} "
              f"forget_mse={per_model[name]['forget_mse']:.4f} "
              f"retain_mse={per_model[name]['retain_sample_mse']:.4f}", flush=True)

    result = {
        "cell": out_path.stem.replace("_unlearn", ""),
        "config_fingerprint": _config_fingerprint(base_cfg),
        "config": str(args.config),
        "fold": fold,
        "era": era,
        "mode": mode,
        "seed": seed,
        "git": git_info,
        "data": info,
        "hyperparams": {
            "ft_epochs": args.ft_epochs, "ft_lr": args.ft_lr,
            "ng_epochs": args.ng_epochs, "ng_lr": args.ng_lr,
            "ng_alpha": args.ng_alpha, "hinge_margin": args.hinge_margin,
            "batch_size": batch_size,
        },
        "timings_seconds": timings,
        "method_stats": method_stats,
        "models": per_model,
        "n_test_days": int(len(frame)),
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
