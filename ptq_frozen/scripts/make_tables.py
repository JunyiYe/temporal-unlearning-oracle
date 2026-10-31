"""Generate the paper's LaTeX tables directly from result artifacts.

Numbers are never hand-copied: every table cell traces to a JSON/CSV in
outputs/. Regenerate with:  python scripts/make_tables.py
Outputs land in paper/icaif/tables/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fin_ts_ptq.metrics.significance import daily_ic_frame  # noqa: E402

OUT = Path("paper/icaif/tables")
MODEL_ORDER = ["dlinear", "itransformer", "patchtst", "transformer", "tsmixer", "timemixer", "segrnn"]
PRETTY = {
    "dlinear": "DLinear", "itransformer": "iTransformer", "patchtst": "PatchTST",
    "transformer": "Transformer", "tsmixer": "TSMixer", "timemixer": "TimeMixer", "segrnn": "SegRNN",
}


def model_key(label: str) -> str:
    return label.replace("_sp500_vol_tuned", "").replace("_sp500_vol_opt", "").replace("_opt", "").replace("_tuned", "")


def fmt(x: float, digits: int = 3, plus: bool = True) -> str:
    s = f"{x:+.{digits}f}" if plus else f"{x:.{digits}f}"
    return s


def main_table() -> str:
    df = pd.read_csv("outputs/sweep_summary.csv")
    df = df[df["fold"].between(2021, 2025)]  # exclude the 2018/2020 stress folds
    piv = df.pivot_table(index=["model", "fold", "seed"], columns="method", values="daily_ic_mean")
    rows = []
    for m in MODEL_ORDER:
        sub = piv[piv.index.get_level_values("model").map(model_key) == m]
        if not len(sub):
            continue
        cov_txt, par_txt = "--", "--"
        cov_all = {}
        cov_file = Path("outputs/coverage.json")
        if cov_file.exists():
            cov_all = json.loads(cov_file.read_text())
        c = cov_all.get(m)
        if c is None:
            for f in Path("outputs/sweep").glob(f"{m}_sp500_vol_*_fold2024_seed7_metrics.json"):
                c = json.loads(f.read_text()).get("quantization_coverage")
        if c:
            cov_txt = f"{100 * c['weight_coverage']:.0f}\\%"
            par_txt = f"{c['total_parameters'] / 1e3:.0f}K"
        fp32 = sub["fp32"]
        cells = [PRETTY[m], par_txt, cov_txt, fmt(fp32.mean())]
        for col in ["static_int8_recent", "rtn_w4", "static_int4_recent"]:
            d = fp32 - sub[col]
            cells.append(f"{fmt(d.mean())} $\\pm$ {d.std():.3f}")
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{lrrrrrr}" "\n" r"\toprule" "\n"
        r"Model & Params & Coverage & fp32 IC & $\Delta$ W8A8 & $\Delta$ W4 (w-only) & $\Delta$ W4A4 \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


RESCUE_MAIN = "outputs/mechanism/{m}_rescue_folds2021_2022_2023_2024_2025.json"


def verdict_table() -> str:
    """Qualitative map: per (architecture, setting) verdict, thresholded on
    relative damage (safe < 5% of fp32 signal, degraded 5-30%, broken > 30%);
    DLinear judged on absolute delta (its fp32 base is statistically zero)."""
    df = pd.read_csv("outputs/sweep_summary.csv")
    df = df[df["fold"].between(2021, 2025)]
    piv = df.pivot_table(index=["model", "fold", "seed"], columns="method", values="daily_ic_mean")

    def verdict(delta: float, fp32: float) -> str:
        if fp32 < 0.05:
            return "safe" if abs(delta) < 0.01 else "degraded"
        rel = delta / fp32
        if rel < 0.05:
            return "safe"
        return "degraded" if rel < 0.30 else "broken"

    rows = []
    for m in MODEL_ORDER:
        sub = piv[piv.index.get_level_values("model").map(model_key) == m]
        if not len(sub):
            continue
        fp32 = float(sub["fp32"].mean())
        cells = [PRETTY[m]]
        for col in ["static_int8_recent", "rtn_w4", "static_int4_recent"]:
            cells.append(verdict(float((sub["fp32"] - sub[col]).mean()), fp32))
        p = Path(RESCUE_MAIN.format(m=m))
        if p.exists():
            d = json.loads(p.read_text())["delta_ic"]
            cells.append(verdict(d["w4a4_p0.99"]["mean"], fp32))
        else:
            cells.append("--")
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
        r"Model & W8A8 & W4 (w-only) & W4A4 abs-max & W4A4 p99 \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def rescue_table() -> str:
    rows = []
    for m in MODEL_ORDER:
        path = Path(RESCUE_MAIN.format(m=m))
        if not path.exists():
            continue
        d = json.loads(path.read_text())["delta_ic"]
        cells = [PRETTY[m]]
        for k in ["w4a4_absmax", "w4a4_p0.99", "w4a4_p0.999", "mixed_protect"]:
            v = d.get(k)
            cells.append(fmt(v["mean"]) if v else "--")
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{lrrrr}" "\n" r"\toprule" "\n"
        r"Model & abs-max & p99 & p999 & mixed \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def stress_table() -> str:
    rows = []
    for m in MODEL_ORDER:
        p18 = Path(f"outputs/mechanism/{m}_rescue_folds2018.json")
        p20 = Path(f"outputs/mechanism/{m}_rescue_folds2020.json")
        if not p18.exists() and not p20.exists():
            continue
        cells = [PRETTY[m]]
        for path in (p18, p20):
            if path.exists():
                d = json.loads(path.read_text())["delta_ic"]
                for k in ["w4a4_absmax", "w4a4_p0.99", "w4a4_p0.999"]:
                    v = d.get(k)
                    cells.append(fmt(v["mean"]) if v else "--")
            else:
                cells.extend(["--"] * 3)
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{lrrrrrr}" "\n" r"\toprule" "\n"
        r" & \multicolumn{3}{c}{Test 2018 (Volmageddon, Q4)} & \multicolumn{3}{c}{Test 2020 (COVID)} \\"
        "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
        r"Model & abs-max & p99 & p999 & abs-max & p99 & p999 \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def calibration_size_table() -> str:
    rows = []
    for m in ["transformer", "tsmixer", "segrnn"]:
        p = Path(f"outputs/mechanism/{m}_calibration_size.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())["delta_ic"]
        for tag, label in (("absmax", "abs-max"), ("p99", "p99")):
            cells = [PRETTY[m] if tag == "absmax" else "", label]
            for n in [16, 64, 256, 1024, 4096]:
                v = d.get(f"n{n}_{tag}")
                cells.append(fmt(v["mean"]) if v else "--")
            rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{llrrrrr}" "\n" r"\toprule" "\n"
        r"Model & Scale stat. & $n{=}16$ & $64$ & $256$ & $1024$ & $4096$ \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def transport_table() -> str:
    """Nikkei 225 transport check: frozen S&P-tuned configs, folds 2022+2024."""
    import glob
    import re
    rows_raw = []
    for f in glob.glob("outputs/transport/*_metrics.json"):
        m = re.match(r"(\w+?)_nikkei_vol_fold(\d+)_seed(\d+)", Path(f).name)
        if not m:
            continue
        d = json.loads(Path(f).read_text())["methods"]
        rows_raw.append({"model": m.group(1),
                         "fp32": d["fp32"]["overall"]["daily_ic_mean"],
                         "w4a4": d["static_int4_recent"]["overall"]["daily_ic_mean"]})
    df = pd.DataFrame(rows_raw)
    rows = []
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        if not len(sub):
            continue
        fp32 = sub["fp32"].mean()
        cells = [PRETTY[m], fmt(fp32), fmt((sub["fp32"] - sub["w4a4"]).mean())]
        p = Path(f"outputs/mechanism/{m}_nikkei_vol_rescue_folds2022_2024.json")
        if p.exists():
            d = json.loads(p.read_text())["delta_ic"]
            cells += [fmt(d["w4a4_p0.99"]["mean"]), fmt(d["w4a4_p0.999"]["mean"])]
        else:
            cells += ["--", "--"]
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{lrrrr}" "\n" r"\toprule" "\n"
        r"Model & fp32 IC & $\Delta$ abs-max & $\Delta$ p99 & $\Delta$ p999 \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def _days_fmt(days: float) -> str:
    if days < 1:
        return "$<$1"
    if days > 20000:
        return "$>$20{,}000"
    return f"{days:,.0f}".replace(",", "{,}")


def mde_table() -> str:
    rows = []
    for m in ["transformer", "tsmixer", "timemixer", "patchtst", "itransformer"]:
        cells = [PRETTY[m]]
        npz = Path(f"outputs/exhibit/{m}_sp500_opt_preds.npz")
        if npz.exists():
            z = np.load(npz)
            meta = pd.DataFrame({"date": pd.to_datetime(z["date_ns"])})
            frame = daily_ic_frame(meta, z["target_true"], {"fp32": z["pred::fp32"]})
            ic, sd = float(frame["fp32"].mean()), float(frame["fp32"].std())
            cells += [f"{ic:+.3f}", _days_fmt((2 * sd / abs(ic)) ** 2 if ic else float("inf"))]
        else:
            cells += ["--", "--"]
        csv = Path(f"outputs/{m}_sp500_vol_tuned_noise_equiv.daily_ic.csv")
        if csv.exists():
            df = pd.read_csv(csv)
            ic, sd = float(df["fp32"].mean()), float(df["fp32"].std())
            sd_d = float((df["fp32"] - df["static_int4_recent"]).std())
            target = 0.10 * abs(ic)
            cells += [
                f"{ic:+.3f}", _days_fmt((2 * sd / abs(ic)) ** 2),
                _days_fmt((2 * sd_d / target) ** 2),
                _days_fmt((2 * np.sqrt(2) * sd / target) ** 2),
            ]
        else:
            cells += ["--"] * 4
        rows.append(" & ".join(cells) + r" \\")
    header = (
        r"\begin{tabular}{lrrrrrr}" "\n" r"\toprule" "\n"
        r" & \multicolumn{2}{c}{Next-day return} & \multicolumn{2}{c}{Volatility} "
        r"& \multicolumn{2}{c}{Detect 10\% W4A4 loss} \\" "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
        r"Model & IC & days & IC & days & paired & unpaired \\"
        "\n" r"\midrule" "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("main_results", main_table),
        ("verdicts", verdict_table),
        ("rescue", rescue_table),
        ("stress", stress_table),
        ("transport", transport_table),
        ("calibration_size", calibration_size_table),
        ("mde", mde_table),
    ):
        (OUT / f"{name}.tex").write_text(builder(), encoding="utf-8")
        print(f"[OK] {OUT / f'{name}.tex'}")


if __name__ == "__main__":
    main()
