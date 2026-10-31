"""Granularity sweep over the per-window loss NPZs (Phase 2, claim 3).

Two layers, matching docs/detector_notes.md:

1. ORACLE-ACCESS (paired) detector, per cell: block-mean reference-vs-
   oracle loss gaps on the forget set at granularity g in {window, day,
   week, month, era}; report the detection t at each g (NW at day level,
   naive above) and the implied minimum-detectable-gap at 80% power.
   This locates g* — the granularity that minimizes the MDE.
2. POPULATION layer, per (era, mode) group: separability d' between the
   {reference} and {oracle} model populations using era-level mean forget
   loss (3 seeds x 2 folds each); AUC = Phi(d'/sqrt(2)).

Run inside the container (numpy/pandas over ~100k-row arrays — srun, not
login):  sbatch -J unl_gran scripts/detector.sbatch  # wrong entry
Use:      sbatch -J unl_gran scripts/granularity.sbatch [--stem ...]
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from fin_ts_ptq.metrics.significance import newey_west_mean_test

UNLEARN_ROOT = (_REPO)
GRAINS = ("window", "day", "week", "month", "era")
Z80 = norm.ppf(0.8) + norm.ppf(0.975)  # (z_beta + z_alpha/2) for 80% power two-sided


def block_means(gap: np.ndarray, dates: pd.Series, grain: str) -> np.ndarray:
    if grain == "window":
        return gap
    if grain == "era":
        return np.array([gap.mean()])
    freq = {"day": "D", "week": "W", "month": "M"}[grain]
    return pd.Series(gap).groupby(dates.dt.to_period(freq).to_numpy()).mean().to_numpy()


def detection_t(blocks: np.ndarray, grain: str) -> tuple[float, float]:
    """(t, mde_gap at 80% power). NW at day level, naive elsewhere."""
    n = len(blocks)
    if n < 2:
        return float("nan"), float("nan")
    if grain == "day":
        mean, t, _, se = newey_west_mean_test(blocks, lag=4)
    else:
        se = blocks.std(ddof=1) / math.sqrt(n)
        t = blocks.mean() / se if se > 0 else float("inf")
    return float(t), float(Z80 * se)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-dir", default=str(UNLEARN_ROOT / "outputs" / "detector"))
    ap.add_argument("--stems", default="transformer_sp500_vol_tuned,patchtst_sp500_vol_tuned")
    ap.add_argument("--out", default=str(UNLEARN_ROOT / "docs" / "granularity_results.md"))
    args = ap.parse_args()

    lines = ["# Granularity sweep (oracle-access t / MDE per grain; population d')", ""]
    for stem in args.stems.split(","):
        per_grain: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        era_scores: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        n_blocks: dict[tuple, dict[str, float]] = defaultdict(dict)
        for f in sorted(Path(args.detector_dir).glob(f"{stem}_era*_losses.npz")):
            z = np.load(f, allow_pickle=False)
            if "forget_loss::reference" not in z or "forget_loss::oracle" not in z:
                continue
            name = f.stem.replace("_losses", "")
            era = [p for p in name.split("_") if p.startswith("era")][0][3:]
            mode = "strict" if "_strict_" in name else "target"
            dates = pd.to_datetime(pd.Series(z["date_ns"]))
            gap = z["forget_loss::oracle"] - z["forget_loss::reference"]
            for grain in GRAINS:
                blocks = block_means(gap, dates, grain)
                t, mde = detection_t(blocks, grain)
                per_grain[(era, mode)][grain].append((t, mde, gap.mean()))
                n_blocks[(era, mode)][grain] = len(blocks)
            era_scores[(era, mode)]["reference"].append(float(z["forget_loss::reference"].mean()))
            era_scores[(era, mode)]["oracle"].append(float(z["forget_loss::oracle"].mean()))

        if not per_grain:
            continue
        lines += [f"## {stem}", "",
                  "| era | mode | grain | n_blocks | mean t | mde_gap | actual gap | detectable? |",
                  "|---|---|---|---|---|---|---|---|"]
        for (era, mode), grains in sorted(per_grain.items()):
            for grain in GRAINS:
                vals = grains[grain]
                ts = [v[0] for v in vals if not math.isnan(v[0])]
                mdes = [v[1] for v in vals if not math.isnan(v[1])]
                gaps = [v[2] for v in vals]
                if not ts:
                    lines.append(f"| {era} | {mode} | {grain} | 1 | n/a | n/a | {np.mean(gaps):+.4f} | single block |")
                    continue
                ok = "YES" if np.mean(gaps) > np.mean(mdes) else "no"
                lines.append(f"| {era} | {mode} | {grain} | {n_blocks[(era, mode)][grain]} "
                             f"| {np.mean(ts):.1f} | {np.mean(mdes):.4f} | {np.mean(gaps):+.4f} | {ok} |")
        lines += ["", "### Population layer (era-level mean forget loss, ref vs oracle models)", "",
                  "| era | mode | n_ref/n_ora | d' | implied AUC |", "|---|---|---|---|---|"]
        for (era, mode), s in sorted(era_scores.items()):
            r, o = np.array(s["reference"]), np.array(s["oracle"])
            pooled = math.sqrt((r.var(ddof=1) + o.var(ddof=1)) / 2)
            d = (o.mean() - r.mean()) / pooled if pooled > 0 else float("inf")
            auc = norm.cdf(d / math.sqrt(2))
            lines.append(f"| {era} | {mode} | {len(r)}/{len(o)} | {d:.2f} | {auc:.3f} |")
        lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
