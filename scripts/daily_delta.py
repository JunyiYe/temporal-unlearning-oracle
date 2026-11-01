"""Per-day deletion effect on the test year, for the mechanism figure.

For each (fold, mode, seed): load the reference (sweep checkpoint; at
strict mode it is evaluated under the era-excluded normalization, which
is exactly how era_pilot defines the strict reference) and the era-2020
oracle checkpoint, predict the fold's test year, and store each model's
daily cross-sectional IC. The daily deletion effect is
IC_day(oracle) - IC_day(reference); the activation hypothesis predicts
it concentrates on the test year's stressed days.

Run inside the container via sbatch (see daily_delta.sbatch). One
invocation covers one config (architecture) and one mode across folds
and seeds. Output: one JSON per (config, fold, mode) under
outputs/daily_delta/.
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import torch
from torch.utils.data import DataLoader

from era_pilot import fold_cfg  # noqa: E402
from fin_ts_ptq.config import load_config  # noqa: E402
from fin_ts_ptq.metrics.significance import daily_ic_frame  # noqa: E402
from fin_ts_ptq.models.factory import build_model  # noqa: E402
from fin_ts_ptq.training.loop import predict  # noqa: E402
from fin_ts_ptq.utils.device import resolve_device  # noqa: E402

from fin_unlearning.era import build_era_datasets  # noqa: E402

UNLEARN_ROOT = (_REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config yaml path (relative to forecasting/)")
    ap.add_argument("--folds", default="2022,2023,2024,2025")
    ap.add_argument("--era", default="2020")
    ap.add_argument("--mode", required=True, choices=["target", "strict"])
    ap.add_argument("--seeds", default="4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--sweep-dir", default=str(UNLEARN_ROOT / "outputs" / "sweep"),
                    help="reference checkpoints")
    ap.add_argument("--ckpt-dir",
                    default=str(UNLEARN_ROOT / "outputs" / "pilot" / "ckpt"))
    ap.add_argument("--out-dir",
                    default=str(UNLEARN_ROOT / "outputs" / "daily_delta"))
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    stem = Path(args.config).stem
    device = resolve_device("auto")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    seeds = [int(s) for s in args.seeds.split(",")]

    for fold in [int(f) for f in args.folds.split(",")]:
        out_path = out_dir / f"{stem}_era{args.era}_{args.mode}_fold{fold}_daily.json"
        if out_path.exists():
            print(f"[SKIP] {out_path.name} exists", flush=True)
            continue
        cfg = fold_cfg(base_cfg, fold, seeds[0])
        datasets, info = build_era_datasets(cfg, int(args.era), args.mode)
        print(f"[DATA] fold{fold} {args.mode}: {info}", flush=True)
        test_loader = DataLoader(datasets["test"], batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
        num_features = datasets["test"].num_features

        result = {"config": stem, "fold": fold, "era": int(args.era),
                  "mode": args.mode, "seeds": {}}
        dates = None
        for seed in seeds:
            ref_ckpt = Path(args.sweep_dir) / f"{stem}_fold{fold}_seed{seed}_model.pt"
            ora_ckpt = ckpt_dir / (f"{stem}_era{args.era}_{args.mode}_"
                                   f"fold{fold}_seed{seed}_oracle.pt")
            missing = [p for p in (ref_ckpt, ora_ckpt) if not p.exists()]
            if missing:
                print(f"[MISS] fold{fold} seed{seed}: {missing}", flush=True)
                continue
            preds = {}
            y_test = None
            for name, path in (("reference", ref_ckpt), ("oracle", ora_ckpt)):
                model = build_model(cfg["model"], cfg["task"],
                                    num_features=num_features)
                model.load_state_dict(torch.load(path, map_location="cpu"))
                y_test, preds[name] = predict(model, test_loader, device)
            frame = daily_ic_frame(datasets["test"].meta, y_test, preds)
            if dates is None:
                dates = [str(d)[:10] for d in frame.index]
                result["dates"] = dates
            result["seeds"][seed] = {
                "ref": [round(float(v), 6) for v in frame["reference"]],
                "ora": [round(float(v), 6) for v in frame["oracle"]],
            }
            print(f"[OK] fold{fold} seed{seed}: "
                  f"ref {frame['reference'].mean():+.4f} "
                  f"ora {frame['oracle'].mean():+.4f}", flush=True)
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"[WROTE] {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
