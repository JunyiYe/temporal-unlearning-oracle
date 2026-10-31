"""Publication figure: the two definitions of forgetting, measured.

Scatter: TS-Unlearn budget sweep (lr x epochs grid, transformer, era 2020,
L1) in the (forget-window IC, test IC) plane, with the reference and the
exact oracle as anchors. Every point is the mean of 10 seeds with 1-sd bars.
The noise target (privacy) sits at IC_F = 0; the counterfactual sits at 0.51.

Output: figs/fig_two_definitions.pdf (vector) + .png preview.
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# validated categorical palette, fixed slot order (dataviz reference instance)
C_SWEEP = "#2a78d6"   # slot 1 blue   -> TS-Unlearn sweep
C_ORACLE = "#eb6834"  # slot 2 orange -> exact oracle
C_REF = "#3a3a3a"     # neutral dark  -> reference anchor

# means over 10 seeds; xerr/yerr are 1 sd
SWEEP_X  = [0.537, 0.523, 0.450, 0.369, 0.232, 0.199]
SWEEP_XE = [0.006, 0.009, 0.028, 0.037, 0.039, 0.060]
SWEEP_Y  = [0.492, 0.492, 0.488, 0.485, 0.423, 0.422]
SWEEP_YE = [0.004, 0.004, 0.007, 0.008, 0.049, 0.062]
REF, REF_E = (0.545, 0.489), (0.011, 0.005)
ORACLE, ORACLE_E = (0.509, 0.497), (0.004, 0.003)

fig, ax = plt.subplots(figsize=(3.35, 1.86))

# noise target (privacy goal) at IC_F = 0
ax.axvline(0.0, color="#888888", linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)
ax.text(0.008, 0.452, "noise target", rotation=90, ha="left",
        va="center", fontsize=6.5, color="#777777", style="italic")

eb = dict(fmt="none", ecolor="#9a9a9a", elinewidth=0.7, capsize=1.6, zorder=3)
ax.errorbar(SWEEP_X, SWEEP_Y, xerr=SWEEP_XE, yerr=SWEEP_YE, **eb)
ax.scatter(SWEEP_X, SWEEP_Y, s=26, facecolor=C_SWEEP, edgecolor="white",
           linewidth=0.6, zorder=4, label="TS-Unlearn (lr $\\times$ budget)")
ax.errorbar([REF[0]], [REF[1]], xerr=[REF_E[0]], yerr=[REF_E[1]], **eb)
ax.scatter(*REF, marker="*", s=110, facecolor=C_REF, edgecolor="white",
           linewidth=0.6, zorder=5, label="reference")
ax.errorbar([ORACLE[0]], [ORACLE[1]], xerr=[ORACLE_E[0]], yerr=[ORACLE_E[1]], **eb)
ax.scatter(*ORACLE, marker="D", s=42, facecolor=C_ORACLE, edgecolor="white",
           linewidth=0.6, zorder=5, label="exact oracle")

# anchor annotations (top-right region is crowded; leader lines disambiguate)
ax.text(0.548, 0.4845, "reference", fontsize=6.5, color="#3a3a3a",
        ha="center", va="top")
ax.annotate("exact oracle\n$\\mathrm{IC}_{\\mathcal{F}} = 0.51$", ORACLE,
            xytext=(0.455, 0.476), fontsize=6.5, color="#3a3a3a",
            ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6))

# the lockstep reading, once, muted
ax.text(0.235, 0.445, "moving $\\mathrm{IC}_{\\mathcal{F}}$ toward 0\nreduces test skill",
        fontsize=6.5, color="#777777", style="italic", ha="left", va="center")

ax.set_xlim(-0.015, 0.60)
ax.set_ylim(0.398, 0.510)
ax.set_xlabel(r"forget-window IC ($\mathrm{IC}_{\mathcal{F}}$)")
ax.set_ylabel("test-year IC")

ax.xaxis.grid(True, color="#e3e3e3", linewidth=0.5, zorder=0)
ax.yaxis.grid(True, color="#e3e3e3", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#9a9a9a")
ax.spines["left"].set_color("#9a9a9a")
ax.tick_params(colors="#3a3a3a")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncols=3,
          frameon=False, borderpad=0.1, handlelength=1.0,
          columnspacing=1.0, handletextpad=0.4, scatterpoints=1)

fig.tight_layout(pad=0.3)

import pathlib
out = (_REPO / "figs")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_two_definitions.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(out / "fig_two_definitions.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
print("wrote", out / "fig_two_definitions.pdf")
