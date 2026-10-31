"""Generate the paper's figures directly from result artifacts (PDF, vector).

Entity->color map is fixed across all figures (validated categorical palette,
slots in fixed order; identity is never color-alone: every series also gets a
distinct marker and a direct label).

    python scripts/make_figures.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("paper/icaif/figures")

# fixed entity -> (color, marker) assignment, reused in every figure
STYLE = {
    "transformer": ("#2a78d6", "o", "Transformer"),
    "tsmixer": ("#1baf7a", "s", "TSMixer"),
    "patchtst": ("#eda100", "^", "PatchTST"),
    "itransformer": ("#008300", "D", "iTransformer"),
    "segrnn": ("#4a3aa7", "v", "SegRNN"),
}

LAYER_NAMES = {
    "model.head.linear": "head",
    "model.enc_embedding.value_embedding.tokenConv": "input conv",
    "model.projection": "head",
    "model.rnn": "GRU",
    "model.valueEmbedding.0": "input embed",
    "model.predict.1": "head",
    "model.patch_embedding.value_embedding": "patch embed",
}


def pretty_layer(name: str) -> str:
    if name in LAYER_NAMES:
        return LAYER_NAMES[name]
    if "out_projection" in name:
        return "attn out-proj"
    if ".channel." in name:
        return "channel MLP"
    if ".temporal." in name:
        return "temporal MLP"
    return name.split(".")[-1]

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})


PRETTY = {
    "dlinear": "DLinear", "itransformer": "iTransformer", "patchtst": "PatchTST",
    "transformer": "Transformer", "tsmixer": "TSMixer", "timemixer": "TimeMixer",
    "segrnn": "SegRNN",
}


def model_key(label: str) -> str:
    return (label.replace("_sp500_vol_tuned", "").replace("_sp500_vol_opt", "")
            .replace("_opt", "").replace("_tuned", ""))


def money_figure() -> None:
    """Figure 1: relative signal loss per architecture under the three PTQ
    settings that define the paper's storyline (W8A8, W4A4 abs-max, W4A4 p99)."""
    df = pd.read_csv("outputs/sweep_summary.csv")
    df = df[df["fold"].between(2021, 2025)]  # exclude the stress folds
    df["model"] = df["model"].map(model_key)
    piv = df.pivot_table(index=["model", "fold", "seed"], columns="method",
                         values="daily_ic_mean")
    rows = []
    for m, sub in piv.groupby(level="model"):
        fp32 = sub["fp32"].mean()
        rec = {"model": m, "fp32": fp32,
               "w8a8": 100 * (sub["fp32"] - sub["static_int8_recent"]).mean() / fp32,
               "w4a4": 100 * (sub["fp32"] - sub["static_int4_recent"]).mean() / fp32}
        p = Path(f"outputs/mechanism/{m}_rescue_folds2021_2022_2023_2024_2025.json")
        if p.exists():
            d = json.loads(p.read_text())["delta_ic"]
            rec["p99"] = 100 * d["w4a4_p0.99"]["mean"] / fp32
        rows.append(rec)
    data = pd.DataFrame(rows).sort_values("w4a4").reset_index(drop=True)
    # drop models whose fp32 skill is itself statistically ~zero (DLinear:
    # significant in 1 of 5 folds) — a share-of-signal bar on a no-skill
    # base is not meaningful; the paper quotes their damage in absolute IC
    data = data[data["fp32"] >= 0.05].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    x = range(len(data))
    w = 0.27
    ax.bar([i - w for i in x], data["w8a8"], w, color="#b8c4cc", label="W8A8")
    ax.bar(list(x), data["w4a4"], w, color="#c0392b", label="W4A4 abs-max")
    if "p99" not in data:
        data["p99"] = float("nan")
    # show the p99 bar only where abs-max damage is material (>10% of
    # signal): a "rescue" of an undamaged model is meaningless, and on
    # DLinear's tiny base the relative bar would mislead
    p99_idx = [i for i in data.index
               if pd.notna(data.loc[i, "p99"]) and data.loc[i, "w4a4"] > 10]
    ax.bar([i + w for i in p99_idx], [data.loc[i, "p99"] for i in p99_idx], w,
           color="#2874a6", hatch="///", edgecolor="white", linewidth=0.0,
           label="W4A4 p99 calib.")
    for i in data.index:
        v = data.loc[i, "w4a4"]
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.2, color="#c0392b")
        if i in p99_idx:
            ax.text(i + w, data.loc[i, "p99"] + 1.5, f"{data.loc[i, 'p99']:.0f}",
                    ha="center", fontsize=6.2, color="#2874a6")
    ax.set_xticks(list(x), [PRETTY[m] for m in data["model"]], fontsize=6.6,
                  rotation=18, ha="right")
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
    ax.set_ylabel("signal lost to PTQ (% of fp32 IC)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "money.pdf")
    plt.close(fig)
    print("[OK] figures/money.pdf")


def width_figure() -> None:
    rows = []
    for f in glob.glob("outputs/width/*_metrics.json"):
        m = re.match(r"(\w+?)_sp500_vol_w(\d+)_fold(\d+)_seed(\d+)", Path(f).name)
        if not m:
            continue
        d = json.loads(Path(f).read_text())
        fp = d["methods"]["fp32"]["overall"]["daily_ic_mean"]
        q4 = d["methods"].get("static_int4_recent", {}).get("overall", {}).get("daily_ic_mean")
        if q4 is None:
            continue
        rows.append({"model": m.group(1), "width": int(m.group(2)), "delta": fp - q4})
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "width"])["delta"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    for model, (color, marker, label) in STYLE.items():
        sub = agg[agg["model"] == model].sort_values("width")
        if not len(sub):
            continue
        ax.errorbar(
            sub["width"], sub["mean"], yerr=sub["std"],
            color=color, marker=marker, markersize=3.5, linewidth=1.2,
            elinewidth=0.6, capsize=1.5, label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512"])
    ax.axhline(0.0, color="0.75", linewidth=0.6, zorder=0)
    ax.set_xlabel(r"model width $d_{\mathrm{model}}$")
    ax.set_ylabel(r"W4A4 $\Delta$IC (paired)")
    ax.set_ylim(-0.035, 0.245)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, handlelength=1.3, columnspacing=0.8, borderaxespad=0.0,
              fontsize=6.3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "width.pdf")
    plt.close(fig)
    print("[OK] figures/width.pdf")


def layer_sensitivity_figure() -> None:
    """Horizontal bars: only-one damage share for the top-3 layers per model."""
    models = ["patchtst", "transformer", "tsmixer", "segrnn"]
    fig, axes = plt.subplots(1, len(models), figsize=(7.0, 1.75), sharex=True)
    for ax, m in zip(axes, models):
        d = json.loads(Path(f"outputs/mechanism/{m}_fold2024_seed7_layer_sensitivity.json").read_text())
        full = d["full_delta"]["delta_ic_mean"]
        layers = sorted(d["layers"].items(), key=lambda kv: -kv[1]["only_one"]["delta_ic_mean"])[:3]
        names = [pretty_layer(k) for k, _ in layers][::-1]
        shares = [100 * v["only_one"]["delta_ic_mean"] / full for _, v in layers][::-1]
        color = STYLE.get(m, ("#4a3aa7", "o", m))[0]
        # positional y with explicit tick labels: duplicate names must not merge
        ax.barh(range(len(shares)), shares, color=color, height=0.55)
        ax.set_yticks(range(len(names)), names)
        for y, s in enumerate(shares):
            ax.text(min(s, 100) + 3, y, f"{s:.0f}%", va="center", fontsize=6.5)
        ax.axvline(100, color="0.4", linewidth=0.6, linestyle=":")
        ax.set_title(STYLE.get(m, (None, None, m.capitalize()))[2], fontsize=7.5)
        ax.set_xlim(0, 125)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6.2)
    axes[0].set_xlabel("")
    fig.supxlabel("share of full W4A4 damage when quantizing this layer alone (%)",
                  fontsize=7.2, y=0.02)
    fig.tight_layout(pad=0.4, rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "layer_sensitivity.pdf")
    plt.close(fig)
    print("[OK] figures/layer_sensitivity.pdf")


def regime_figure() -> None:
    """Data-space regime check: daily cross-sectional dispersion (CSSD)
    against each stress fold's calibration-year maximum (the data-space
    analogue of an abs-max calibration range). Also dumps the stats the
    paper quotes to outputs/regime_stats.json."""
    ret = pd.read_csv("data/sp500_kaggle_returns.csv", parse_dates=["date"])
    col = "value" if "value" in ret.columns else "return"
    disp = ret.groupby("date")[col].std() * 100.0
    disp = disp[disp.index >= "2016-01-01"]
    all_days = list(disp.index)
    pos = {d: i for i, d in enumerate(all_days)}

    stats = {}
    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    ax.plot(disp.index, disp.values, lw=0.4, color="0.5", zorder=1)
    ax.set_yscale("log")

    for test_year, cal_year, color in ((2018, 2017, "#b07500"), (2020, 2019, "#c23b22")):
        cal = disp[disp.index.year == cal_year]
        cur = disp[disp.index.year == test_year]
        cal_max = float(cal.max())
        exc = cur[cur > cal_max]
        x0 = pd.Timestamp("%d-01-01" % test_year)
        x1 = pd.Timestamp("%d-12-31" % test_year)
        ax.axvspan(x0, x1, color=color, alpha=0.10, zorder=0)
        ax.hlines(cal_max, x0, x1, color=color, ls="--", lw=1.0, zorder=2)
        ax.plot(exc.index, exc.values, ".", ms=3.5, color=color, zorder=3)
        contaminated = set()
        for d in exc.index:
            contaminated |= set(all_days[pos[d]:pos[d] + 64])
        contaminated = {d for d in contaminated if d.year == test_year}
        stats[test_year] = {
            "cal_year": cal_year,
            "cal_max_pct": round(cal_max, 3),
            "exceed_days": int(len(exc)),
            "exceed_share": round(len(exc) / len(cur), 4),
            "peak_ratio": round(float(cur.max()) / cal_max, 3),
            "share_above_cal_p99": round(float((cur > cal.quantile(0.99)).mean()), 4),
            "contaminated_window_share": round(len(contaminated) / len(cur), 4),
        }

    a18, a20 = stats[2018], stats[2020]
    ax.annotate("2018: %d days above\ncal-2017 max, peak %.2fx" % (a18["exceed_days"], a18["peak_ratio"]),
                xy=(pd.Timestamp("2017-08-01"), 4.6), fontsize=6, ha="center", color="#b07500")
    ax.annotate("2020: %d days above\ncal-2019 max, peak %.2fx" % (a20["exceed_days"], a20["peak_ratio"]),
                xy=(pd.Timestamp("2021-10-01"), 8.8), fontsize=6, ha="center", color="#c23b22")
    ax.set_ylabel(r"cross-sectional dispersion $\sigma_t$ (%)", fontsize=7)
    ax.set_ylim(0.45, 16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "regime.pdf")
    plt.close(fig)
    Path("outputs/regime_stats.json").write_text(json.dumps(stats, indent=2))
    print("[OK] figures/regime.pdf + outputs/regime_stats.json")


if __name__ == "__main__":
    money_figure()
    width_figure()
    layer_sensitivity_figure()
    regime_figure()
