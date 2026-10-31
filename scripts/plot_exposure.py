"""Publication figure: license-expiry exposure across test regimes.

Two stacked panels sized for one column, shared x-scale: the same two
deletions (era 2019 recent-calm, era 2020 crisis) evaluated in a calm test
year (2024) and in a bear-market test year (2022), four architectures, L1,
n=10 seeds.

Delta = IC(oracle) - IC(reference), computed directly from the cell JSONs.
NOTE: never read models[m]['vs_reference']['delta_ic_mean'] -- that field is
stored with the opposite sign (see docs/results_master.md).

Output: figs/fig_exposure_two_regimes.pdf (vector) + .png preview.
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import json
import pathlib
import statistics

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

C2020 = "#2a78d6"  # categorical slot 1
C2019 = "#eb6834"  # categorical slot 2
LABEL = {2020: "delete 2020 (crisis era)", 2019: "delete 2019 (calm era)"}

PILOT = (_REPO / "outputs" / "pilot")
# ordered by test IC: the three above the HAR baseline, then the two below
ARCHS = ["Transformer", "TSMixer", "SegRNN", "iTransformer", "PatchTST"]
STEM = {"Transformer": "transformer", "TSMixer": "tsmixer", "SegRNN": "segrnn",
        "iTransformer": "itransformer", "PatchTST": "patchtst"}


def delta(arch, era, fold):
    d = []
    for s in range(4, 14):
        f = PILOT / f"{STEM[arch]}_sp500_vol_v2_era{era}_strict_fold{fold}_seed{s}_unlearn.json"
        m = json.load(open(f))["models"]
        d.append(m["oracle"]["test_daily_ic_mean"] - m["reference"]["test_daily_ic_mean"])
    return statistics.mean(d), statistics.stdev(d)


y = np.arange(len(ARCHS))[::-1]
H, OFF = 0.34, 0.19
err_kw = dict(ecolor="#3a3a3a", elinewidth=0.7, capsize=1.8, capthick=0.7)

fig, axes = plt.subplots(2, 1, figsize=(3.33, 4.1), sharex=True, sharey=True)

for ax, (fold, title) in zip(axes, [(2024, "calm test year (2024)"),
                                    (2022, "bear-market test year (2022)")]):
    for era, colour, off, hatch in [(2020, C2020, +OFF, None),
                                    (2019, C2019, -OFF, "///")]:
        vals = [delta(a, era, fold) for a in ARCHS]
        ax.barh(y + off, [v[0] for v in vals], height=H, color=colour,
                edgecolor="white", linewidth=0.4, hatch=hatch,
                xerr=[v[1] for v in vals], error_kw=err_kw, zorder=3,
                label=LABEL[era] if fold == 2024 else None)
    ax.axvline(0.0, color="#888888", linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)
    ax.set_title(title, fontsize=7.5, pad=3, loc="left")
    ax.xaxis.grid(True, color="#e3e3e3", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9a9a9a")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#3a3a3a")

axes[0].set_yticks(y, ARCHS)
axes[0].set_xlim(-0.105, 0.105)
axes[0].set_ylim(-0.55, 4.95)
axes[0].set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])

# the line that separates architectures beating the HAR baseline from the rest
for ax in axes:
    ax.axhline(1.5, color="#b0b0b0", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
axes[0].text(0.10, 2.05, "above HAR", ha="right", va="bottom",
             fontsize=6, color="#888888", style="italic")
axes[0].text(0.10, 0.95, "below HAR", ha="right", va="top",
             fontsize=6, color="#888888", style="italic")

# sign semantics stated once, in the free strip above the top panel
axes[0].text(-0.10, 4.62, "← worse without it", ha="left", va="center",
             fontsize=6.5, color="#777777", style="italic")
axes[0].text(0.10, 4.62, "better without it →", ha="right", va="center",
             fontsize=6.5, color="#777777", style="italic")
fig.supxlabel(r"deletion effect $\Delta$ = IC(oracle) $-$ IC(reference)",
              fontsize=8, y=0.02)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
           ncols=2, frameon=False, handlelength=1.2, handleheight=0.9,
           columnspacing=1.5, handletextpad=0.4)

fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.945))

out = (_REPO / "figs")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_exposure_two_regimes.pdf")
fig.savefig(out / "fig_exposure_two_regimes.png", dpi=300)
print("wrote", out / "fig_exposure_two_regimes.pdf")
