"""Independent end-to-end verification of Finding 2's sign chain.

Deliberately avoids the code paths that produced the paper numbers:
  - datasets built via fin_ts_ptq.experiment.build_datasets DIRECTLY
    (target-mode leg uses zero fin_unlearning code);
  - daily IC reimplemented inline with pandas (no fin_ts_ptq.metrics);
  - prediction loop reimplemented inline (no fin_ts_ptq.training.loop);
  - reference side cross-checked against outputs/sweep/*_preds.npz,
    predictions written by runner.py months before this project existed.

Identity fingerprints (is "oracle" really the era-free model?):
  F1 file provenance: reference ckpt from outputs/sweep (PTQ-era sweep),
     oracle ckpt from outputs/pilot/ckpt (this project's retrains);
  F2 parameter distance ||theta_ora - theta_ref|| > 0;
  F3 behavioral: the model trained WITHOUT 2020 must be systematically
     WORSE on 2020 windows (direction fixed by training, not by any
     evaluation code).

Sign chain under test:  delta = IC_test(oracle) - IC_test(reference);
positive delta  <=>  oracle better  <=>  deletion IMPROVES skill.
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_datasets
from fin_ts_ptq.models.factory import build_model

UNL = (_REPO)
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default="configs/transformer_sp500_vol_tuned.yaml")
_ap.add_argument("--fold", type=int, default=2022)
_ap.add_argument("--era", type=int, default=2020)
_args = _ap.parse_args()
CONFIG = _args.config
STEM = Path(CONFIG).stem
FOLD, ERA = _args.fold, _args.era


def inline_daily_ic(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    df = pd.DataFrame({"d": meta["date"].to_numpy(), "y": y.ravel(), "p": p.ravel()})
    per_day = df.groupby("d").apply(lambda g: g["y"].corr(g["p"]))
    return float(per_day.dropna().mean())


@torch.no_grad()
def inline_predict(model, ds, device):
    model.to(device).eval()
    ys, ps = [], []
    for x, y in DataLoader(ds, batch_size=2048, shuffle=False):
        ps.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fold_cfg(base, test_year, seed):
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["train_end"] = f"{test_year - 2}-12-31"
    cfg["data"]["val_end"] = f"{test_year - 1}-12-31"
    cfg["data"]["test_end"] = f"{test_year}-12-31"
    return cfg


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_config(CONFIG)
    print(f"[CONFIG] {CONFIG} stem={STEM} fold={FOLD} era={ERA}")
    ok = True

    print("=" * 72)
    print("LEG A: target mode — datasets via fin_ts_ptq.build_datasets ONLY")
    print("=" * 72)
    for seed in (7, 8, 9):
        cfg = fold_cfg(base, FOLD, seed)
        data = build_datasets(cfg)  # PTQ-paper code path, untouched by this project
        test_ds, train_ds = data.splits["test"], data.splits["train"]
        nf = len(data.feature_columns)

        ref = build_model(cfg["model"], cfg["task"], num_features=nf)
        ref.load_state_dict(torch.load(f"outputs/sweep/{STEM}_fold{FOLD}_seed{seed}_model.pt", map_location="cpu"))
        ora = build_model(cfg["model"], cfg["task"], num_features=nf)
        ora.load_state_dict(torch.load(
            UNL / "outputs/pilot/ckpt" / f"{STEM}_era{ERA}_target_fold{FOLD}_seed{seed}_oracle.pt", map_location="cpu"))

        # F2: parameter distance
        pd_ = sum(float(((a - b) ** 2).sum()) for a, b in zip(ref.state_dict().values(), ora.state_dict().values())) ** 0.5
        # F3: behavioral fingerprint on 2020 training windows (target date in era)
        m2020 = (train_ds.meta["date"] >= pd.Timestamp("2020-01-01")).to_numpy() & \
                (train_ds.meta["date"] <= pd.Timestamp("2020-12-31")).to_numpy()
        sub = torch.utils.data.Subset(train_ds, np.nonzero(m2020)[0].tolist())
        yf, pf_ref = inline_predict(ref, sub, device)
        _, pf_ora = inline_predict(ora, sub, device)
        mse_ref, mse_ora = float(((pf_ref - yf) ** 2).mean()), float(((pf_ora - yf) ** 2).mean())

        # the quantity under test
        yt, pt_ref = inline_predict(ref, test_ds, device)
        _, pt_ora = inline_predict(ora, test_ds, device)
        ic_ref = inline_daily_ic(test_ds.meta, yt, pt_ref)
        ic_ora = inline_daily_ic(test_ds.meta, yt, pt_ora)
        delta = ic_ora - ic_ref

        # reference cross-check vs months-old sweep predictions
        npz = np.load(f"outputs/sweep/{STEM}_fold{FOLD}_seed{seed}_preds.npz", allow_pickle=False)
        sweep_meta = pd.DataFrame({"date": pd.to_datetime(npz["date_ns"])})
        ic_ref_npz = inline_daily_ic(sweep_meta, npz["target_true"], npz["pred::fp32"])

        f3_ok = mse_ora > mse_ref
        xchk_ok = abs(ic_ref - ic_ref_npz) < 0.005
        ok &= f3_ok and xchk_ok and pd_ > 0
        print(f"seed{seed}: ||dtheta||={pd_:.1f} | 2020-window MSE ref={mse_ref:.4f} ora={mse_ora:.4f} "
              f"(oracle worse on era: {'OK' if f3_ok else 'FAIL'})")
        print(f"        IC_ref={ic_ref:+.4f} (sweep-npz cross-check {ic_ref_npz:+.4f}: "
              f"{'OK' if xchk_ok else 'FAIL'}) | IC_ora={ic_ora:+.4f} | delta=IC_ora-IC_ref={delta:+.4f}")

    print()
    print("=" * 72)
    print("LEG B: strict mode — era datasets, but inline IC/predict/fingerprints")
    print("=" * 72)
    from fin_unlearning.era import build_era_datasets
    for seed in (7, 8, 9):
        cfg = fold_cfg(base, FOLD, seed)
        ds, _ = build_era_datasets(cfg, ERA, "strict")
        nf = ds["test"].num_features
        ref = build_model(cfg["model"], cfg["task"], num_features=nf)
        ref.load_state_dict(torch.load(f"outputs/sweep/{STEM}_fold{FOLD}_seed{seed}_model.pt", map_location="cpu"))
        ora = build_model(cfg["model"], cfg["task"], num_features=nf)
        ora.load_state_dict(torch.load(
            UNL / "outputs/pilot/ckpt" / f"{STEM}_era{ERA}_strict_fold{FOLD}_seed{seed}_oracle.pt", map_location="cpu"))
        yf, pf_ref = inline_predict(ref, ds["forget"], device)
        _, pf_ora = inline_predict(ora, ds["forget"], device)
        mse_ref, mse_ora = float(((pf_ref - yf) ** 2).mean()), float(((pf_ora - yf) ** 2).mean())
        yt, pt_ref = inline_predict(ref, ds["test"], device)
        _, pt_ora = inline_predict(ora, ds["test"], device)
        ic_ref = inline_daily_ic(ds["test"].meta, yt, pt_ref)
        ic_ora = inline_daily_ic(ds["test"].meta, yt, pt_ora)
        f3_ok = mse_ora > mse_ref
        ok &= f3_ok
        print(f"seed{seed}: forget MSE ref={mse_ref:.4f} ora={mse_ora:.4f} ({'OK' if f3_ok else 'FAIL'}) | "
              f"IC_ref={ic_ref:+.4f} IC_ora={ic_ora:+.4f} delta={ic_ora - ic_ref:+.4f}")

    print()
    print("VERDICT:", "SIGN CHAIN VERIFIED — positive delta = oracle better = deletion improves"
          if ok else "*** DISCREPANCY FOUND — DO NOT SUBMIT ***")


if __name__ == "__main__":
    main()
