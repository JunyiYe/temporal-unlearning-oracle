from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PTQ metrics JSON.")
    parser.add_argument("metrics_json")
    parser.add_argument("--baseline", default="fp32")
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    methods = payload["methods"]
    baseline = methods[args.baseline]["overall"]

    print(f"config: {payload.get('config')}")
    print(f"baseline: {args.baseline}")

    if payload.get("baselines"):
        print("\nnaive_baselines")
        print("baseline,mse,mae,direction_accuracy,ic,rank_ic,sharpe")
        for name, baseline_payload in payload["baselines"].items():
            o = baseline_payload["overall"]
            print(f"{name},{o['mse']:.10g},{o['mae']:.10g},{o['direction_accuracy']:.10g},{o['ic']:.10g},{o['rank_ic']:.10g},{o['sharpe']:.10g}")

    calibration = payload.get("calibration", {})
    if calibration:
        print("\ncalibration_policy_summary")
        print("policy,num_windows,date_start,date_end,date_coverage_ratio,abs_target_mean,abs_target_p90,abs_target_p99,tail_coverage_rate,unique_assets")
        for policy, metrics in calibration.items():
            print(
                f"{policy},{metrics['num_windows']:.0f},{metrics['date_start']},{metrics['date_end']},"
                f"{metrics['date_coverage_ratio']:.6g},{metrics['abs_target_mean']:.10g},"
                f"{metrics['abs_target_p90']:.10g},{metrics['abs_target_p99']:.10g},"
                f"{metrics['tail_coverage_rate']:.6g},{metrics['unique_assets']:.0f}"
            )

    print("\noverall_method_metrics")
    print("metric,method,value,delta_vs_baseline,relative_delta")
    for method, method_payload in methods.items():
        overall = method_payload["overall"]
        for metric, value in sorted(overall.items()):
            if not isinstance(value, int | float):
                continue
            base = baseline.get(metric)
            delta = value - base if isinstance(base, int | float) else float("nan")
            rel = delta / abs(base) if isinstance(base, int | float) and abs(base) > 1e-12 else float("nan")
            print(f"{metric},{method},{value:.10g},{delta:.10g},{rel:.10g}")

    print("\nregime_delta_by_abs_return_decile")
    print("decile,metric,method,value,baseline,delta")
    baseline_regimes = methods[args.baseline]["abs_return_regimes"]
    for method, method_payload in methods.items():
        if method == args.baseline:
            continue
        for decile, metrics in sorted(method_payload["abs_return_regimes"].items()):
            base_metrics = baseline_regimes[decile]
            for metric in ["mse", "mae", "direction_accuracy", "ic", "rank_ic"]:
                value = metrics[metric]
                base = base_metrics[metric]
                print(f"{decile},{metric},{method},{value:.10g},{base:.10g},{value - base:.10g}")


if __name__ == "__main__":
    main()
