"""Generate paper/icaif_v2 tables from outputs/v2 artifacts. Never hand-edit the .tex outputs.

    PYTHONPATH=src python scripts/make_tables_v2.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

SWEEP = "outputs/v2/sweep"
MECH = "outputs/v2/mechanism"
OUT = Path("paper/icaif_v2/tables")

ARCHS = {
    "dlinear": ("DLinear", "linear"),
    "patchtst": ("PatchTST", "attention"),
    "itransformer": ("iTransformer", "attention"),
    "tsmixer": ("TSMixer", "mixer"),
    "transformer": ("Transformer", "attention"),
    "timemixer": ("TimeMixer", "mixer"),
    "segrnn": ("SegRNN", "recurrent"),
}


def arch_stats(a: str) -> dict:
    fps, d8, dw, d4 = [], [], [], []
    n_params = None
    for f in glob.glob(f"{SWEEP}/{a}*_metrics.json"):
        m = json.load(open(f))
        me = m["methods"]
        g = lambda k: me[k]["overall"]["daily_ic_mean"]
        n_params = m["quantization_coverage"]["total_parameters"]
        fps.append(g("fp32"))
        d8.append(g("fp32") - g("static_int8_random"))
        dw.append(g("fp32") - g("rtn_w4"))
        d4.append(g("fp32") - g("static_int4_random"))
    groups = {"w4a4_p0.99": [], "w4a4_p0.999": []}  # (n, mean, sd) per rescue file
    for rf in glob.glob(f"{MECH}/{a}_rescue_folds*.json"):
        if "oracle" in rf:
            continue
        r = json.load(open(rf))
        for k in groups:
            groups[k].append((r["cells"], r["delta_ic"][k]["mean"], r["delta_ic"][k]["std"]))

    def pooled(parts):
        # combine per-file (n, mean, sd) into overall mean and sd (ddof=1)
        if not parts:
            return None, None
        n_tot = sum(n for n, _, _ in parts)
        mean = sum(n * m for n, m, _ in parts) / n_tot
        ss = sum(s * s * (n - 1) + n * (m - mean) ** 2 for n, m, s in parts)
        return mean, (ss / (n_tot - 1)) ** 0.5

    p99, p99_sd = pooled(groups["w4a4_p0.99"])
    p999, p999_sd = pooled(groups["w4a4_p0.999"])
    return {
        "n_cells": len(fps), "n_params": n_params,
        "fp32": st.mean(fps), "fp32_sd": st.stdev(fps),
        "d_w8a8": st.mean(d8), "d_w4only": st.mean(dw),
        "d_w4a4": st.mean(d4), "d_w4a4_sd": st.stdev(d4),
        "p99": p99, "p99_sd": p99_sd, "p999": p999, "p999_sd": p999_sd,
    }


def fmt_params(n: int) -> str:
    return f"{n/1000:.1f}K"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    har = json.load(open("outputs/v2/har_baseline.json"))
    har_ic = st.mean(v["har"]["daily_ic_mean"] for v in har["folds"].values())
    per_ic = st.mean(v["persistence"]["daily_ic_mean"] for v in har["folds"].values())

    rows = {a: arch_stats(a) for a in ARCHS}
    order = sorted(ARCHS, key=lambda a: rows[a]["d_w4a4"])

    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & Family & Params & fp32 IC $\uparrow$ & $\Delta$W8A8 $\downarrow$ & "
        r"$\Delta$W4 $\downarrow$ & $\Delta$W4A4 $\downarrow$ & $\Delta$p999 $\downarrow$ & $\Delta$p99 $\downarrow$ \\",
        r"\midrule",
        rf"HAR & econometric & 3 & {har_ic:.3f} & -- & -- & -- & -- & -- \\",
        rf"Persistence & na\"ive & 0 & {per_ic:.3f} & -- & -- & -- & -- & -- \\",
        r"\midrule",
    ]
    for a in order:
        r = rows[a]
        name, fam = ARCHS[a]
        has_base = r["fp32"] > 0.05

        def cell(mean, sd):
            if mean is None:
                return "--"
            rel = f" ({mean/r['fp32']*100:.0f}\\%)" if has_base else ""
            return f"{mean:+.3f}$\\pm${sd:.3f}{rel}"

        lines.append(
            rf"{name} & {fam} & {fmt_params(r['n_params'])} & "
            rf"{r['fp32']:.3f}$\pm${r['fp32_sd']:.3f} & "
            rf"{r['d_w8a8']:+.4f} & {r['d_w4only']:+.4f} & "
            rf"{cell(r['d_w4a4'], r['d_w4a4_sd'])} & {cell(r['p999'], r['p999_sd'])} & {cell(r['p99'], r['p99_sd'])} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {OUT/'main_results.tex'} ({len(order)} archs; cells per arch: "
          f"{sorted(set(rows[a]['n_cells'] for a in order))})")


if __name__ == "__main__":
    main()
