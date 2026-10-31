"""Publication figures for the v2 paper, built from outputs/figure_series.json.

Every series in that file was exported directly from the 3,200-cell grid by
the export block in the same commit; nothing here is typed by hand.

Colour encodes the paper's own distinction: the three architectures above the
HAR benchmark get saturated hues, the two below get greys, HAR itself is a
black dashed reference. That way the reader sees the split before reading a
caption.

Outputs (vector PDF + PNG preview) into figs/:
  fig_ladder.pdf   what a model carried, and what happened when it went
  fig_regimes.pdf  the crisis-year deletion across four test regimes
  fig_methods.pdf  closure of each architecture-method pair across regimes
  fig_audit.pdf    audit separation against memorization, all 100 cells
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

S = json.load(open(str(_REPO / "results" / "figure_series.json")))
FIGS = str(_REPO / "figs") + "/"

ABOVE = ["transformer", "tsmixer", "segrnn"]
BELOW = ["itransformer", "patchtst"]
NAME = {"transformer": "Transformer", "tsmixer": "TSMixer", "segrnn": "SegRNN",
        "itransformer": "iTransformer", "patchtst": "PatchTST", "har": "HAR"}
COL = {"transformer": "#2a78d6", "tsmixer": "#eb6834", "segrnn": "#2e9e5b",
       "itransformer": "#d0489c", "patchtst": "#8c564b", "har": "#111111"}
MRK = {"transformer": "o", "tsmixer": "s", "segrnn": "^",
       "itransformer": "v", "patchtst": "D", "har": None}


def style(a):
    return dict(color=COL[a], marker=MRK[a], markersize=3.2, linewidth=1.3,
                markeredgewidth=0, zorder=3 if a in ABOVE else 2,
                alpha=1.0 if a in ABOVE else 0.85)


def save(fig, stem):
    fig.savefig(FIGS + stem + ".pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGS + stem + ".png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", stem)


# ------------------------------------------------------------------ figure 1
# All four folds: the per-year deletion effect, 2x2, rows share a scale.
VOLS = {"2016": 14.4, "2017": 7.2, "2018": 15.7, "2019": 12.5,
        "2020": 37.6, "2021": 13.5, "2022": 23.7, "2023": 14.1}
L4 = S["ladder4"]
fig, axg = plt.subplots(2, 2, figsize=(3.35, 2.45), sharey=True)
order = ["2022", "2023", "2024", "2025"]
for k, fold in enumerate(order):
    ax = axg[k // 2][k % 2]
    yrs = L4["years"][fold]
    ax.axhline(0, color="#444444", linewidth=0.7, zorder=1)
    AJIT_L = {"transformer": -0.18, "tsmixer": 0.0, "segrnn": 0.18}
    for a in ABOVE:
        ax.errorbar([i + AJIT_L[a] for i in range(5)],
                    [L4["data"][fold][a]["mean"][y] for y in yrs],
                    yerr=[L4["data"][fold][a]["sem"][y] for y in yrs],
                    linestyle="none", marker=MRK[a], markersize=3.0,
                    color=COL[a], markeredgewidth=0,
                    elinewidth=0.6, capsize=1.2, zorder=3)
    ax.plot(range(5), [L4["data"][fold]["har"]["mean"][y] for y in yrs],
            color=COL["har"], linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([y[2:] for y in yrs], fontsize=6.3)
    TESTLBL = {"2022": "bear-market test (24%)", "2023": "mid-vol test (14%)",
               "2024": "low-vol test (12%)", "2025": "mid-vol test (18%)"}
    ax.set_title(f"{fold} fold: {TESTLBL[fold]}", fontsize=5.9, loc="left",
                 pad=2)
    ax.tick_params(axis="y", labelsize=6.5)
axg[0][0].set_ylim(-0.045, 0.095)
axg[0][0].set_yticks([-0.04, 0.0, 0.04, 0.08])
for row in axg:
    row[0].set_ylabel(r"$\Delta$ (IC)")
hs = [plt.Line2D([], [], color=COL[a], marker=MRK[a], linestyle="none",
                 markersize=3.0, label=NAME[a])
      for a in ABOVE] + \
     [plt.Line2D([], [], color=COL["har"], linestyle=(0, (3, 2)),
                 linewidth=0.9, label="HAR")]
axg[0][0].legend(handles=hs, frameon=False, loc="upper left", ncol=1,
                 handletextpad=0.3, handlelength=1.3, borderpad=0.1,
                 labelspacing=0.12, fontsize=5.2)
fig.supxlabel("deleted year ('16 = 2016)", fontsize=7.5, y=0.015)
fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.6, rect=(0, 0.03, 1, 1))
save(fig, "fig_ladder")


# ------------------------------------------------------------------ figure 2
# Scatter: effect of deleting 2020 vs test-year volatility, deployable + HAR.
fig, ax = plt.subplots(figsize=(3.35, 2.3))
folds = sorted(S["regimes"]["vol"], key=lambda f: -S["regimes"]["vol"][f])
xs = {f: S["regimes"]["vol"][f] for f in folds}
ax.axhline(0, color="#444444", linewidth=0.8, zorder=1)
AJIT = {"transformer": -0.45, "tsmixer": -0.15, "segrnn": 0.15}
for a in ABOVE:
    ax.errorbar([xs[f] + AJIT[a] for f in folds],
                [S["regimes"]["delta"][a][f] for f in folds],
                yerr=[S["regimes"]["sd"][a][f] for f in folds],
                linestyle="none", marker=MRK[a], markersize=3.6,
                color=COL[a], markeredgewidth=0, elinewidth=0.7,
                capsize=1.6, zorder=3)
ax.scatter([xs[f] + 0.45 for f in folds],
           [S["regimes"]["delta"]["har"][f] for f in folds],
           s=16, facecolor=COL["har"], marker="D", edgecolor="none",
           zorder=3)
guide = [sum(S["regimes"]["delta"][a][f] for a in ABOVE) / 3 for f in folds]
ax.plot([xs[f] for f in folds], guide, color="#8a8a8a", linewidth=1.1,
        linestyle=(0, (4, 2)), alpha=0.75, zorder=2)
hs = [plt.Line2D([], [], color=COL[a], marker=MRK[a], linestyle="none",
                 markersize=3.4, label=NAME[a]) for a in ABOVE] + \
     [plt.Line2D([], [], color=COL["har"], marker="D", linestyle="none",
                 markersize=3.4, label="HAR")]
ax.legend(handles=hs, frameon=False, loc="upper right", handletextpad=0.25,
          borderpad=0.15, labelspacing=0.2, fontsize=6.0)
ax.set_xlabel("realized volatility of the test year (%)")
ax.set_ylabel(r"$\Delta$ from deleting 2020 (IC)")
ax.invert_xaxis()
ax.set_xticks([xs[f] for f in folds])
ax.set_xticklabels([f"{S['regimes']['vol'][f]:.0f}\n{f} fold" for f in folds],
                   fontsize=6.8)
fig.tight_layout(pad=0.4)
save(fig, "fig_regimes")


# ------------------------------------------------------------------ figure 3
# kappa per pair: colour = architecture, marker shape = fold.
MNAME = {"finetune": "finetune", "neggrad_hinge": "hinge",
         "scrub": "SCRUB", "ts_unlearn": "TS-Unlearn"}
FMRK = {"2022": "o", "2023": "s", "2024": "^", "2025": "D"}
rows = []
for a in ABOVE:
    for meth in ["finetune", "neggrad_hinge", "scrub", "ts_unlearn"]:
        d = S["kappa"][a][meth]
        if d:
            rows.append((a, meth, d))
fig, ax = plt.subplots(figsize=(3.35, 1.95))
ax.axvline(1.0, color="#444444", linewidth=0.9, zorder=1)
ax.text(1.15, -0.45, "counterfactual", fontsize=6.5, color="#444444")
for i, (a, meth, d) in enumerate(rows):
    y = len(rows) - 1 - i
    cl = {f: max(v, 0.02) for f, v in d.items()}
    ax.plot([min(cl.values()), max(cl.values())], [y, y], color=COL[a],
            linewidth=0.7, alpha=0.45, zorder=2)
    for f, v in cl.items():
        ax.plot([v], [y], linestyle="none", marker=FMRK[f], markersize=2.9,
                color=COL[a], markeredgewidth=0, alpha=0.9, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{NAME[a]} $\\cdot$ {MNAME[m]}" for a, m, _ in rows][::-1],
                   fontsize=6.8)
ax.set_xscale("log")
ax.set_xlabel(r"closure $\kappa$ (one point per fold)", labelpad=2)
ax.set_xlim(0.02, 20)
ax.set_ylim(-0.8, len(rows) - 0.2)
hs = [plt.Line2D([], [], color="#666666", marker=FMRK[f], linestyle="none",
                 markersize=3.2, label=f"{f} fold") for f in FMRK]
ax.legend(handles=hs, frameon=False, loc="upper right", handletextpad=0.25,
          borderpad=0.15, labelspacing=0.2, fontsize=6.0)
fig.tight_layout(pad=0.4)
save(fig, "fig_methods")


# ------------------------------------------------------------------ figure 4
fig, ax = plt.subplots(figsize=(3.35, 1.75))
ax.axhline(1.0, color="#999999", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
ax.axhline(2.0, color="#999999", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
for key, v in S["audit"].items():
    a, fold, era = key.split("|")
    crisis = era == "2020"
    ax.scatter(v["G"], v["dprime"], s=18 if crisis else 11,
               facecolor=COL[a], edgecolor="none",
               alpha=0.95 if a in ABOVE else 0.7,
               marker=MRK[a], zorder=3 if crisis else 2)
import matplotlib.patheffects as _pe
_halo = [_pe.withStroke(linewidth=2.4, foreground="white")]
ax.text(0.245, 1.0, "$d'=1$", fontsize=6.0, color="#777777",
        va="center", path_effects=_halo)
ax.text(0.155, 2.0, "$d'=2$", fontsize=6.0, color="#777777",
        va="center", path_effects=_halo)
ax.set_xlabel("memorization gap $G$")
ax.set_ylabel("audit separation $d'$")
hs = [plt.Line2D([], [], color=COL[a], marker=MRK[a], linestyle="none",
                 markersize=3.6, label=NAME[a]) for a in ABOVE + BELOW]
ax.legend(handles=hs, frameon=False, loc="upper center", ncol=2,
          bbox_to_anchor=(0.56, 1.04), handletextpad=0.25, borderpad=0.1,
          labelspacing=0.2, columnspacing=0.7, fontsize=6.0)
fig.tight_layout()
save(fig, "fig_audit")


# ------------------------------------------------------------------ figure: G vs volatility, all cells
fig, (axL, axR) = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=True)
VOLS_G = S["gap_scatter"]["vol"]
JIT = {"2022": -0.36, "2023": -0.12, "2024": 0.12, "2025": 0.36}
grp = {"L": [], "R": []}
for c in S["gap_scatter"]["cells"]:
    a = c["arch"]
    ax = axL if a in ABOVE else axR
    grp["L" if a in ABOVE else "R"].append((VOLS_G[c["year"]], c["G"]))
    ax.scatter(VOLS_G[c["year"]] + JIT[c["fold"]], c["G"], s=11,
               facecolor=COL[a], edgecolor="none", marker=MRK[a],
               alpha=0.8, zorder=3)


def cluster_curve(pts):
    """mean G per volatility cluster: <10, 10-20, 2022, 2020."""
    out = []
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45)]:
        sel = [(x, y) for x, y in pts if lo <= x < hi]
        n = len(sel)
        mx = sum(x for x, _ in sel) / n
        my = sum(y for _, y in sel) / n
        sd = (sum((y - my) ** 2 for _, y in sel) / (n - 1)) ** 0.5
        out.append((mx, my, sd))
    return out


for ax, key in [(axL, "L"), (axR, "R")]:
    cc = cluster_curve(grp[key])
    ax.errorbar([x for x, _, _ in cc], [y for _, y, _ in cc],
                yerr=[e for _, _, e in cc], color="#8a8a8a",
                linewidth=1.1, linestyle=(0, (4, 2)), marker="o",
                markersize=2.6, alpha=0.75, elinewidth=0.7,
                capsize=1.8, zorder=2)
for ax, title in [(axL, "deployable"),
                  (axR, "non-deployable")]:
    ax.set_xlim(4, 41)
    ax.set_title(title, fontsize=7.0, loc="left", pad=3)
axL.set_ylim(-0.012, 0.50)
axL.set_ylabel("memorization gap $G$")
axL.annotate("low-med. vol.", (14.5, 0.135), fontsize=6.0, ha="center",
             color="#555555")
axL.annotate("2022", (23.7, 0.225), fontsize=6.0, ha="center", color="#555555")
axL.annotate("2020", (37.6, 0.47), fontsize=6.0, ha="center", va="top",
             color="#555555")
axR.annotate("low-med. vol.", (12.0, 0.105), fontsize=6.0, ha="center",
             color="#555555")
axR.annotate("2022", (23.7, 0.115), fontsize=6.0, ha="center", color="#555555")
axR.annotate("2020", (37.6, 0.135), fontsize=6.0, ha="center", color="#555555")
hsL = [plt.Line2D([], [], color=COL[a], marker=MRK[a], linestyle="none",
                  markersize=3.2, label=NAME[a]) for a in ABOVE]
hsR = [plt.Line2D([], [], color=COL[a], marker=MRK[a], linestyle="none",
                  markersize=3.2, label=NAME[a]) for a in BELOW]
axL.legend(handles=hsL, frameon=False, loc="upper left", handletextpad=0.25,
           borderpad=0.15, labelspacing=0.2, fontsize=6.0)
axR.legend(handles=hsR, frameon=False, loc="upper left", handletextpad=0.25,
           borderpad=0.15, labelspacing=0.2, fontsize=6.0)
fig.supxlabel("realized volatility of the deleted year (%)", fontsize=7.5, y=0.02)
fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
save(fig, "fig_gap")
