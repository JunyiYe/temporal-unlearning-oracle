"""Aggregate sweep metrics JSONs into one long-form CSV for analysis/plots.

Each row = (label, fold, seed, method) with all scalar `overall` metrics.
Baselines are included with method names prefixed `baseline_`.

    python scripts/aggregate_sweep.py --glob "outputs/sweep/*_metrics.json" \
        --output outputs/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

LABEL_RE = re.compile(r"^(?P<model>.+?)_fold(?P<fold>\d{4})_seed(?P<seed>\d+)$")


def rows_from_file(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    label = payload.get("label") or Path(path).name.replace("_metrics.json", "")
    m = LABEL_RE.match(label)
    base = {
        "label": label,
        "config": payload.get("config"),
        "model": m.group("model") if m else label,
        "fold": int(m.group("fold")) if m else None,
        "seed": int(m.group("seed")) if m else payload.get("seed"),
        "best_val_loss": payload.get("train", {}).get("best_val_loss"),
        "epochs_ran": payload.get("train", {}).get("epochs_ran"),
    }
    rows = []
    sections = [("", payload.get("methods", {})), ("baseline_", payload.get("baselines", {}))]
    for prefix, section in sections:
        for method, data in section.items():
            row = dict(base)
            row["method"] = f"{prefix}{method}"
            for k, v in data.get("overall", {}).items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sweep metrics into a CSV.")
    parser.add_argument("--glob", default="outputs/sweep/*_metrics.json")
    parser.add_argument("--output", default="outputs/sweep_summary.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No metrics files match {args.glob}")
    rows = [row for f in files for row in rows_from_file(f)]
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"[OK] {len(df)} rows from {len(files)} files -> {args.output}")

    # quick fp32 sanity pivot: mean daily IC by model x fold
    fp32 = df[df["method"] == "fp32"]
    if len(fp32) and "daily_ic_mean" in fp32.columns and fp32["fold"].notna().any():
        pivot = fp32.pivot_table(index="model", columns="fold", values="daily_ic_mean", aggfunc="mean")
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
