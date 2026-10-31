"""Generate paper/icaif_v2 figures from outputs/v2 + panel data.

    PYTHONPATH=src python scripts/make_figures_v2.py regime
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

FIGDIR = Path("paper/icaif_v2/figures")
PANEL = "data/sp500_yf2008_returns.csv"

# Okabe-Ito CVD-safe hues; identity is also carried by mark shape.
C_SIGMA = "#7f7f7f"      # daily dispersion line (recessive)
C_ENV = "#0072B2"        # calibration envelope step (blue, dashed)
C_EXC = "#D55E00"        # exceedance days (vermillion dots)

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
})


def regime_figure() -> None:
    df = pd.read_csv(PANEL, parse_dates=["date"])
    sig = df.groupby("date")["return"].std()
    sig.index = pd.to_datetime(sig.index)
    sig = sig["2017":"2025"]

    stats = {}
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    ax.plot(sig.index, sig.values, color=C_SIGMA, lw=0.5,
            label=r"$\sigma_t$")

    env_handle = None
    exc_handle = None
    for Y in range(2018, 2026):
        cal = sig[str(Y - 1)]
        test = sig[str(Y)]
        env = cal.max()
        exceed = test[test > env]
        stats[Y] = {"envelope": float(env), "peak_ratio": float(test.max() / env),
                    "n_exceed": int(len(exceed)), "share": float(len(exceed) / len(test))}
        (env_handle,) = ax.plot(
            [pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-12-31")], [env, env],
            color=C_ENV, ls="--", lw=1.0)
        if len(exceed):
            exc_handle = ax.scatter(exceed.index, exceed.values, s=5, zorder=3,
                                    color=C_EXC, edgecolors="white", linewidths=0.3)

    # single annotation: the one true envelope break
    ax.annotate("2.03$\\times$ the envelope", xy=(pd.Timestamp("2020-03-20"), 0.081),
                xytext=(pd.Timestamp("2021-06-01"), 0.072), fontsize=6.8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"))

    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_t$ (log)")
    ax.set_yticks([0.01, 0.02, 0.04, 0.08])
    ax.set_yticklabels(["0.01", "0.02", "0.04", "0.08"])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, length=2)
    ax.legend(handles=[
        ax.lines[0],
        matplotlib.lines.Line2D([], [], color=C_ENV, ls="--", lw=1.0,
                                label="calibration envelope"),
        matplotlib.lines.Line2D([], [], color=C_EXC, marker="o", ls="none", ms=3,
                                label="exceedance"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
       handlelength=1.2, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout(pad=0.3)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "regime.pdf")
    fig.savefig(FIGDIR / "regime.png", dpi=220)
    Path("outputs/v2/regime_stats.json").write_text(json.dumps(stats, indent=2))
    print("[OK] regime.pdf +", json.dumps({k: round(v["peak_ratio"], 2) for k, v in stats.items()}))


def ucurve_figure() -> None:
    import pandas as pd
    df = pd.read_csv("outputs/v2/clip_sweep/clip_sweep_cells.csv")
    KEYS = ["w4a4_p0.9", "w4a4_p0.95", "w4a4_p0.98", "w4a4_p0.99",
            "w4a4_p0.995", "w4a4_p0.999", "w4a4_p0.9999", "w4a4_absmax"]
    XLBL = ["p90", "p95", "p98", "p99", "p99.5", "p99.9", "p99.99", "max"]
    ARCHS = [("tsmixer", "TSMixer"), ("transformer", "Transformer"),
             ("segrnn", "SegRNN"), ("timemixer", "TimeMixer")]
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True)
    x = range(len(KEYS))
    for ax, (a, name) in zip(axes.flat, ARCHS):
        m = df[df.arch == a].groupby("method")["delta"].mean()
        ax.plot(x, [m[k] for k in KEYS], marker="o", ms=2.4, lw=1.1, color=C_ENV)
        ax.set_title(name, fontsize=7, pad=2)
        ax.axhline(0, color="#bbbbbb", lw=0.4, zorder=0)
        ax.set_xticks(list(x))
        ax.set_xticklabels(XLBL, rotation=60, fontsize=5.5)
        ax.tick_params(length=2)
    axes[0][0].set_ylabel(r"$\Delta$ daily IC")
    axes[1][0].set_ylabel(r"$\Delta$ daily IC")
    fig.tight_layout(pad=0.3)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "ucurve.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "ucurve.png", dpi=220, bbox_inches="tight")
    print("[OK] ucurve.pdf (pooled, all folds)")


def ucurve_years_figure() -> None:
    import pandas as pd
    df = pd.read_csv("outputs/v2/clip_sweep/clip_sweep_cells.csv")
    KEYS = ["w4a4_p0.9", "w4a4_p0.95", "w4a4_p0.98", "w4a4_p0.99",
            "w4a4_p0.995", "w4a4_p0.999", "w4a4_p0.9999", "w4a4_absmax"]
    XLBL = ["p90", "p95", "p98", "p99", "p99.5", "p99.9", "p99.99", "max"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharex=True, sharey=True)
    x = range(len(KEYS))
    for ax, (a, name) in zip(axes, [("tsmixer", "TSMixer"), ("segrnn", "SegRNN")]):
        sub = df[df.arch == a]
        for fold in range(2018, 2026):
            m = sub[sub.fold == fold].groupby("method")["delta"].mean()
            y = [m.get(k) for k in KEYS]
            if fold == 2020:
                ax.plot(x, y, color=C_EXC, lw=1.4, marker="o", ms=2, zorder=5, label="2020")
            elif fold == 2021:
                ax.plot(x, y, color=C_ENV, lw=1.4, marker="s", ms=2, zorder=4,
                        label="2021 (calib. on 2020)")
            else:
                ax.plot(x, y, color="#aaaaaa", lw=0.5, zorder=2,
                        label="other years" if fold == 2018 else None)
        ax.set_title(name, fontsize=7, pad=2)
        ax.axhline(0, color="#cccccc", lw=0.4, zorder=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(XLBL, rotation=60, fontsize=5.2)
        ax.set_ylim(-0.03, 0.36)
        ax.tick_params(length=2)
    axes[0].set_ylabel(r"$\Delta$ daily IC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=6.2, columnspacing=0.8,
               handlelength=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.savefig(FIGDIR / "ucurve_years.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "ucurve_years.png", dpi=220, bbox_inches="tight")
    print("[OK] ucurve_years.pdf")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "regime"
    {"regime": regime_figure, "ucurve": ucurve_figure,
     "ucurve_years": ucurve_years_figure}[which]()
