"""Noise-equivalence experiment: is quantization robustness just redundancy?

For a trained checkpoint, compares test-set daily IC under
  (a) actual PTQ (rtn_w4 / static_int4 / static_int8), and
  (b) iid Gaussian weight noise whose per-channel std matches the second
      moment of RTN quantization error, swept over a multiplier grid.

If the model tolerates matched noise as well as it tolerates quantization,
its PTQ robustness is generic perturbation robustness (parameter redundancy).
If quantization degrades IC far outside the matched-noise distribution, the
quantization error is structured (outlier clipping, systematic bias) and the
redundancy explanation is ruled out.

Usage (inside the cluster container):
    python scripts/noise_equivalence.py --config configs/tsmixer_sp500_vol_opt.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_calibration_loader, build_datasets
from fin_ts_ptq.metrics.significance import daily_ic_frame, paired_delta_tests
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.noise import count_perturbed_parameters, perturb_weights_like_rtn
from fin_ts_ptq.quantization.ptq import quantize_model
from fin_ts_ptq.training.loop import predict
from fin_ts_ptq.utils.seed import set_seed


def parse_floats(v: str) -> list[float]:
    return [float(p) for p in str(v).split(",") if p != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight-noise vs quantization comparison.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bits", type=int, default=4, help="Bit-width whose RTN error the noise matches.")
    parser.add_argument("--multipliers", default="0.25,0.5,1,2,4", help="Noise std multipliers.")
    parser.add_argument("--noise-seeds", type=int, default=8, help="Noise draws per multiplier.")
    parser.add_argument(
        "--quant-methods",
        default="rtn_w4,static_int8_recent,static_int4_recent",
        help="Comma list; static entries are <method>_<calibration policy>.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--train-years", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.test_year is not None:
        cfg["data"]["train_end"] = f"{args.test_year - 2}-12-31"
        cfg["data"]["val_end"] = f"{args.test_year - 1}-12-31"
        cfg["data"]["test_end"] = f"{args.test_year}-12-31"
        if args.train_years is not None:
            cfg["data"]["train_start"] = f"{args.test_year - 2 - args.train_years}-12-31"
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    set_seed(int(cfg.get("seed", 7)))
    stem = Path(args.config).stem

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("outputs") / f"{stem}_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint required: {checkpoint_path}. Train it with run_experiment.py first.")

    quant_methods = [m for m in str(args.quant_methods).split(",") if m]
    needs_val = any(m.startswith("static_") for m in quant_methods)
    data = build_datasets(cfg, splits=("val", "test") if needs_val else ("test",))
    test_ds = data.splits["test"]
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds: dict[str, np.ndarray] = {}
    target_true, preds["fp32"] = predict(quantize_model(model, "fp32"), test_loader, device)
    print(f"[EVAL] fp32 done ({len(target_true)} windows)", flush=True)

    calibration_cfg = cfg.get("ptq", {}).get("calibration", {})
    for label in quant_methods:
        if label.startswith("static_"):
            # policy names may contain underscores (volatility_stratified),
            # so split on the known method prefix rather than the last "_"
            method = next((m for m in ("static_int8", "static_int4") if label.startswith(m + "_")), None)
            if method is None:
                raise ValueError(f"Cannot parse static method label: {label}")
            policy = label[len(method) + 1 :]
            calibration_loader = build_calibration_loader(
                data.splits["val"],
                policy=policy,
                calibration_cfg=calibration_cfg,
                seed=int(cfg.get("seed", 7)),
                batch_size=int(args.batch_size),
            )
            qmodel = quantize_model(model, method, calibration_loader)
            # fake-quant modules run fine on GPU; dynamic_int8 would not
            _, preds[label] = predict(qmodel, test_loader, device)
        elif label == "dynamic_int8":
            qmodel = quantize_model(model, label)
            _, preds[label] = predict(qmodel, test_loader, torch.device("cpu"))
        else:
            qmodel = quantize_model(model, label)
            _, preds[label] = predict(qmodel, test_loader, device)
        print(f"[EVAL] {label} done", flush=True)

    multipliers = parse_floats(args.multipliers)
    noise_labels: dict[float, list[str]] = {m: [] for m in multipliers}
    for multiplier in multipliers:
        for seed in range(args.noise_seeds):
            label = f"noise_b{args.bits}_x{multiplier:g}_s{seed}"
            nmodel = perturb_weights_like_rtn(model, num_bits=args.bits, multiplier=multiplier, seed=seed)
            _, preds[label] = predict(nmodel, test_loader, device)
            noise_labels[multiplier].append(label)
        print(f"[EVAL] noise x{multiplier:g} done ({args.noise_seeds} seeds)", flush=True)

    # paired-by-day IC comparison on the model's actual forecast target
    frame = daily_ic_frame(test_ds.meta, target_true, preds)
    forward_period = int(cfg["task"].get("forward_period", 1))
    hac_lag = max(0, forward_period - 1)
    tests = paired_delta_tests(frame, baseline="fp32", lag=hac_lag)

    noise_summary: dict[str, dict[str, float]] = {}
    for multiplier in multipliers:
        deltas = np.array([tests[l]["delta_ic_mean"] for l in noise_labels[multiplier]])
        noise_summary[f"x{multiplier:g}"] = {
            "delta_ic_mean": float(deltas.mean()),
            "delta_ic_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "delta_ic_max": float(deltas.max()),
            "ic_mean": float(np.mean([tests[l]["ic_mean"] for l in noise_labels[multiplier]])),
        }

    # where does actual quantization fall inside the matched (x1) noise distribution?
    matched = noise_summary.get("x1", {})
    quant_summary: dict[str, dict[str, float]] = {}
    for label in quant_methods:
        entry = dict(tests[label])
        if matched and matched.get("delta_ic_std", 0.0) > 0:
            entry["z_vs_matched_noise"] = (
                (entry["delta_ic_mean"] - matched["delta_ic_mean"]) / matched["delta_ic_std"]
            )
        quant_summary[label] = entry

    report = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "bits": int(args.bits),
        "noise_seeds": int(args.noise_seeds),
        "hac_lag": hac_lag,
        "n_test_windows": int(len(target_true)),
        "n_days": int(len(frame)),
        "model_size": count_perturbed_parameters(model),
        "fp32_ic_mean": float(frame["fp32"].mean()),
        "quant": quant_summary,
        "noise": noise_summary,
    }

    output = Path(args.output) if args.output else Path("outputs") / f"{stem}_noise_equiv.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    frame.to_csv(output.with_suffix(".daily_ic.csv"), index_label="date")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[OK] Wrote {output} and {output.with_suffix('.daily_ic.csv')}")


if __name__ == "__main__":
    main()
