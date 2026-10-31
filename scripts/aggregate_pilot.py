"""Aggregate era-pilot cell JSONs -> G1 gate verdict + method table.

Stdlib-only on purpose (json/statistics): reads a couple dozen small JSONs,
no panel data touched, so it can run anywhere including a login shell.

  python scripts/aggregate_pilot.py [--pilot-dir outputs/pilot] [--out docs/p11_results.md]

G1 gate: is the reference-vs-oracle forget-MSE gap large relative to its
across-cell (fold x seed) spread, per (era, mode)? Report mean gap, sd,
gap/sd, and a paired t across cells. Method table: per (era, mode, method)
mean +- sd of dIC-vs-oracle, forget-MSE positioning between reference and
oracle (fraction of the gap closed; >1 = overshoot), retain MSE, cost.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tstat(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    sd = st.stdev(xs)
    return st.mean(xs) / (sd / math.sqrt(len(xs))) if sd > 0 else float("inf")


def fmt(m: float, s: float, d: int = 4) -> str:
    return f"{m:+.{d}f}±{s:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default=str(ROOT / "outputs" / "pilot"))
    ap.add_argument("--stem", default="", help="config stem prefix filter, e.g. transformer_sp500_vol_tuned")
    ap.add_argument("--out", default=str(ROOT / "docs" / "p11_results.md"))
    args = ap.parse_args()

    cells = []
    for f in sorted(Path(args.pilot_dir).glob(f"{args.stem}*_unlearn.json")):
        cells.append(json.loads(f.read_text(encoding="utf-8")))
    if not cells:
        raise SystemExit("no cell JSONs found")

    lines = [f"# P1.1 era-pilot aggregation ({len(cells)} cells)", ""]

    # ---- G1 gate: forget-MSE gap reference vs oracle ----
    lines += ["## G1 gate: is the memorization gap detectable above seed noise?", "",
              "| era | mode | cells | ref forget MSE | oracle forget MSE | gap | gap/sd | t |",
              "|---|---|---|---|---|---|---|---|"]
    by_em: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        by_em[(c["era"], c["mode"])].append(c)
    for (era, mode), cs in sorted(by_em.items()):
        gaps = [c["models"]["oracle"]["forget_mse"] - c["models"]["reference"]["forget_mse"] for c in cs]
        refs = [c["models"]["reference"]["forget_mse"] for c in cs]
        oras = [c["models"]["oracle"]["forget_mse"] for c in cs]
        gm = st.mean(gaps)
        gs = st.stdev(gaps) if len(gaps) > 1 else float("nan")
        lines.append(
            f"| {era} | {mode} | {len(cs)} | {st.mean(refs):.4f} | {st.mean(oras):.4f} "
            f"| {gm:+.4f} | {gm/gs if gs and gs > 0 else float('nan'):.1f} | {tstat(gaps):.1f} |")
    lines.append("")

    # ---- skill cost of exact deletion ----
    lines += ["## Deletion effect (oracle IC - reference IC; POSITIVE = better without the era)", "",
              "| era | mode | dIC mean±sd | t |", "|---|---|---|---|"]
    for (era, mode), cs in sorted(by_em.items()):
        d = [c["models"]["oracle"]["test_daily_ic_mean"] - c["models"]["reference"]["test_daily_ic_mean"] for c in cs]
        lines.append(f"| {era} | {mode} | {fmt(st.mean(d), st.stdev(d) if len(d)>1 else float('nan'))} | {tstat(d):.1f} |")
    lines.append("")

    # ---- method table ----
    methods = sorted({m for c in cells for m in c["models"] if m not in ("reference", "oracle")})
    lines += ["## Approximate methods vs oracle", "",
              "(gap_closed: 0 = still at reference forget-MSE, 1 = exactly oracle, >1 = overshoot)", "",
              "| era | mode | method | dIC vs oracle | gap_closed | retain MSE | mean s |",
              "|---|---|---|---|---|---|---|"]
    for (era, mode), cs in sorted(by_em.items()):
        for m in methods:
            have = [c for c in cs if m in c["models"]]
            if not have:
                continue
            dic = [c["models"][m]["vs_oracle"]["delta_ic_mean"] for c in have]
            closed = []
            for c in have:
                ref, ora = c["models"]["reference"]["forget_mse"], c["models"]["oracle"]["forget_mse"]
                closed.append((c["models"][m]["forget_mse"] - ref) / (ora - ref) if ora != ref else float("nan"))
            rmse = [c["models"][m]["retain_sample_mse"] for c in have]
            secs = [c["timings_seconds"].get(m, float("nan")) for c in have]
            lines.append(
                f"| {era} | {mode} | {m} | {fmt(st.mean(dic), st.stdev(dic) if len(dic)>1 else float('nan'))} "
                f"| {st.mean(closed):.2f}±{st.stdev(closed) if len(closed)>1 else float('nan'):.2f} "
                f"| {st.mean(rmse):.4f} | {st.mean(secs):.0f} |")
    lines.append("")

    # ---- cost ----
    osec = [c["timings_seconds"]["oracle"] for c in cells if "oracle" in c["timings_seconds"]]
    if osec:
        lines.append(f"Oracle retrain: mean {st.mean(osec):.0f}s over {len(osec)} trained (rest cached).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    main()
