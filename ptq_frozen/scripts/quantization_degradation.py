from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    "mse",
    "mae",
    "direction_accuracy",
    "ic",
    "rank_ic",
    "portfolio_return_mean",
    "portfolio_return_sum",
    "sharpe",
    "max_drawdown",
    "turnover_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report quantization-induced downstream degradation.")
    parser.add_argument("metrics_json")
    parser.add_argument("--baseline", default="fp32")
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    methods = payload["methods"]
    base = methods[args.baseline]["overall"]

    print(f"config: {payload.get('config')}")
    print(f"baseline: {args.baseline}")
    if payload.get("baselines"):
        print("\nnaive_baselines")
        print("baseline,metric,value,delta_vs_fp32")
        fp32 = base
        for baseline_name, baseline_payload in payload["baselines"].items():
            overall = baseline_payload["overall"]
            for metric in ["mse", "mae", "direction_accuracy", "ic", "rank_ic", "sharpe", "portfolio_return_mean"]:
                if metric not in overall:
                    continue
                value = float(overall[metric])
                fp = float(fp32.get(metric, float("nan")))
                delta = value - fp if fp == fp else float("nan")
                print(f"{baseline_name},{metric},{value:.12g},{delta:.12g}")

    print("\nquantized_vs_fp32")
    print("method,metric,baseline,quantized,absolute_delta,relative_delta_pct")
    for method, data in methods.items():
        if method == args.baseline:
            continue
        overall = data["overall"]
        for metric in METRICS:
            if metric not in base or metric not in overall:
                continue
            b = float(base[metric])
            q = float(overall[metric])
            delta = q - b
            rel = 100.0 * delta / abs(b) if abs(b) > 1e-12 else float("nan")
            print(f"{method},{metric},{b:.12g},{q:.12g},{delta:.12g},{rel:.6g}")

    print("\nforecast_loss_summary")
    print("method,mse_increase_pct,mae_increase_pct,direction_acc_delta,rank_ic_delta")
    for method, data in methods.items():
        if method == args.baseline:
            continue
        overall = data["overall"]
        mse_pct = 100.0 * (overall["mse"] - base["mse"]) / abs(base["mse"])
        mae_pct = 100.0 * (overall["mae"] - base["mae"]) / abs(base["mae"])
        da_delta = overall["direction_accuracy"] - base["direction_accuracy"]
        ric_delta = overall["rank_ic"] - base["rank_ic"]
        print(f"{method},{mse_pct:.6g},{mae_pct:.6g},{da_delta:.12g},{ric_delta:.12g}")

    if "calibration" in payload and payload["calibration"]:
        print("\ncalibration_context")
        print("policy,tail_coverage_rate,abs_target_mean,date_coverage_ratio")
        for policy, stats in payload["calibration"].items():
            print(
                f"{policy},{stats['tail_coverage_rate']:.12g},"
                f"{stats['abs_target_mean']:.12g},{stats['date_coverage_ratio']:.12g}"
            )


if __name__ == "__main__":
    main()
