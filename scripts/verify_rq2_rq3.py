"""Re-measure the RQ2 (methods) and RQ3 (audit) numbers quoted in the v2 paper.

Read-only. Population d' is recomputed from the per-cell forget-set MSE of the
10 reference and 10 oracle models, which is the statistic the paper describes.
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import json, glob, os, math, re
from collections import defaultdict

ROOT = str(_REPO / "outputs")
PAT = re.compile(r"^(?P<arch>[a-z]+)_sp500_vol_v2_era(?P<era>\d{4})_"
                 r"(?P<mode>strict|target|l2halo)_fold(?P<fold>\d{4})_seed(?P<seed>\d+)_unlearn\.json$")
ARCHS = ["transformer", "tsmixer", "segrnn", "itransformer", "patchtst"]
FOLDS = ["2022", "2023", "2024", "2025"]
METHODS = ["finetune", "neggrad_hinge", "scrub", "ts_unlearn"]

cells = defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "pilot", "*_unlearn.json"))):
    m = PAT.match(os.path.basename(f))
    if m:
        cells[(m["arch"], m["fold"], m["era"], m["mode"])].append(json.load(open(f)))


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return 0.0
    mu = mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


def med(v):
    s = sorted(v)
    n = len(s)
    return None if not s else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


# ------------------------------------------------------------------ population d'
print("=" * 78)
print("RQ3 population audit: d' = (mean_ora - mean_ref)/pooled sd of forget MSE")
dprime, gapof = {}, {}
for key, ds in cells.items():
    if key[3] != "strict":
        continue
    r = [d["models"]["reference"]["forget_mse"] for d in ds]
    o = [d["models"]["oracle"]["forget_mse"] for d in ds]
    if len(r) < 2:
        continue
    pooled = math.sqrt((sd(r) ** 2 + sd(o) ** 2) / 2)
    dprime[key[:3]] = (mean(o) - mean(r)) / pooled if pooled else float("nan")
    gapof[key[:3]] = mean(o) - mean(r)

print("  Table 2 check (calm fold 2024, L1):  arch     d'2020   d'2019   "
      "[paper: 11.1/5.2, 5.2/3.4, 11.3/7.7, 3.8/3.1, 5.3/3.1]")
for a in ARCHS:
    print(f"    {a:13s} {dprime.get((a,'2024','2020'), float('nan')):6.1f}  "
          f"{dprime.get((a,'2024','2019'), float('nan')):6.1f}")

vals = [v for v in dprime.values() if v == v]
print(f"\n  over {len(vals)} cells:  min {min(vals):+.2f}  max {max(vals):+.2f}"
      f"   [paper: -0.4 to 14.0 over 100 cells]")
print(f"  cells with d' <= 1: {sum(1 for v in vals if v <= 1)}   [paper: seven]")
print(f"  cells below d' = 2: {sum(1 for v in vals if v < 2)} "
      f"= {100*sum(1 for v in vals if v < 2)/len(vals):.0f}%   [paper: 17%]")
ks = [k for k in dprime if dprime[k] == dprime[k]]
print(f"  r(G, d') = {pearson([gapof[k] for k in ks], [dprime[k] for k in ks]):+.2f}"
      f"   [paper: 0.62]")
low = [k for k in ks if dprime[k] < 1]
print(f"  max |G| among cells with d' < 1: "
      f"{max(abs(gapof[k]) for k in low) if low else float('nan'):.4f}   [paper: < 0.01]")

# ------------------------------------------------------------------ window level
print("\n" + "=" * 78)
print("RQ3 window-level audit and the i.i.d. trap (outputs/detector/*_daygap.json)")
wf, infl, byarch = [], [], defaultdict(list)
wfg = []
for f in sorted(glob.glob(os.path.join(ROOT, "detector", "*_daygap.json"))):
    d = json.load(open(f))
    g = d.get("day_gap", {})
    if "window_frac_ref_better" not in g:
        continue
    stem = os.path.basename(f)
    m = re.match(r"^([a-z]+)_sp500_vol_v2_era(\d{4})_(\w+?)_fold(\d{4})_seed(\d+)_daygap\.json$", stem)
    if not m:
        continue
    arch, era, mode, fold = m.group(1), m.group(2), m.group(3), m.group(4)
    wf.append(g["window_frac_ref_better"])
    key = (arch, fold, era)
    if key in gapof:
        wfg.append((g["window_frac_ref_better"], gapof[key]))
    nt, wt = g.get("naive_t"), g.get("nw_t")
    if nt and wt and abs(wt) > 1e-9:
        infl.append(abs(nt / wt))
        byarch[arch].append(abs(nt / wt))
print(f"  cells: {len(wf)}   [paper: over 1,000]")
print(f"  window frac ref better: median {med(wf):.3f}  max {max(wf):.3f}"
      f"   [paper: median 0.54, best 0.67]")
inband = sum(1 for v in wf if 0.50 <= v <= 0.61)
print(f"  fraction in [0.50, 0.61]: {inband/len(wf):.2f}   [paper: eight in ten]")
if wfg:
    print(f"  r(window frac, G) = {pearson([a for a,_ in wfg],[b for _,b in wfg]):+.2f}"
          f"   [paper: 0.57]")
    hi = [a for a, b in wfg if b > 0.25]
    lo = [a for a, b in wfg if b < 0.05]
    print(f"  median where G>0.25: {med(hi):.3f} (n={len(hi)});  "
          f"where G<0.05: {med(lo):.3f} (n={len(lo)})   [paper: 0.60 vs 0.53]")
print(f"  i.i.d. inflation |naive t / NW t|: median {med(infl):.2f}   [paper: 1.9]")
pa = sorted(med(v) for v in byarch.values())
print(f"  per-architecture medians: {pa[0]:.2f} to {pa[-1]:.2f}   [paper: 1.6 to 2.1]")

# ------------------------------------------------------------------ RQ2 kappa
print("\n" + "=" * 78)
print("RQ2: gap closed (kappa) per method, era 2020, L1, across the four regimes")
NAME = {"finetune": "finetune", "neggrad_hinge": "hinge", "scrub": "SCRUB",
        "ts_unlearn": "TS-Unlearn"}


def kappa(ds, meth):
    out = []
    for d in ds:
        try:
            mr = d["models"]["reference"]["forget_mse"]
            mo = d["models"]["oracle"]["forget_mse"]
            mm = d["models"][meth]["forget_mse"]
        except (KeyError, TypeError):
            continue
        if abs(mo - mr) > 1e-12:
            out.append((mm - mr) / (mo - mr))
    return out


print("  arch          method       " + "  ".join(f"fold{f}" for f in FOLDS) + "     range")
for a in ARCHS:
    for meth in METHODS:
        row, allk = [], []
        for f in FOLDS:
            k = kappa(cells.get((a, f, "2020", "strict"), []), meth)
            row.append(f"{mean(k):7.2f}" if k else "     --")
            if k:
                allk.append(mean(k))
        rng = f"{min(allk):.2f}-{max(allk):.2f}" if allk else "--"
        print(f"  {a:13s} {NAME[meth]:11s} " + " ".join(row) + f"   {rng}")

print("\n  Table 5 column check (calm fold 2024 only, vs printed values):")
PRINTED = {("transformer", "finetune"): 0.23, ("transformer", "neggrad_hinge"): 1.27,
           ("transformer", "scrub"): 0.69, ("transformer", "ts_unlearn"): 0.12,
           ("tsmixer", "finetune"): -0.12, ("tsmixer", "neggrad_hinge"): 0.81,
           ("tsmixer", "scrub"): 0.17, ("tsmixer", "ts_unlearn"): -0.09,
           ("segrnn", "finetune"): 0.07, ("segrnn", "neggrad_hinge"): 0.50,
           ("segrnn", "scrub"): 0.28, ("segrnn", "ts_unlearn"): 0.06,
           ("itransformer", "finetune"): 0.21, ("itransformer", "neggrad_hinge"): 4.45,
           ("itransformer", "scrub"): 0.25, ("itransformer", "ts_unlearn"): 0.07,
           ("patchtst", "finetune"): 0.03, ("patchtst", "neggrad_hinge"): 3.46,
           ("patchtst", "scrub"): 73.6, ("patchtst", "ts_unlearn"): 0.15}
bad = 0
for (a, meth), want in PRINTED.items():
    k = kappa(cells.get((a, "2024", "2020", "strict"), []), meth)
    got = mean(k)
    if got is None:
        continue
    flag = "" if abs(got - want) <= max(0.02, 0.02 * abs(want)) else "   <-- MISMATCH"
    if flag:
        bad += 1
    print(f"    {a:13s} {NAME[meth]:11s} printed {want:8.2f}  measured {got:8.2f}"
          f"  sd {sd(k):7.2f}{flag}")
print(f"  mismatches: {bad}")

# hinge numerator / kappa on calm eras
print("\n  hinge numerator MSE_F(m)-MSE_F(ref) across ALL cells (paper: 0.19 to 0.51):")
num = []
for key, ds in cells.items():
    if key[3] != "strict":
        continue
    for d in ds:
        try:
            num.append(d["models"]["neggrad_hinge"]["forget_mse"]
                       - d["models"]["reference"]["forget_mse"])
        except (KeyError, TypeError):
            pass
cm = defaultdict(list)
for key, ds in cells.items():
    if key[3] != "strict":
        continue
    v = kappa(ds, "neggrad_hinge")
    if v:
        cm[key[:3]].append(mean(v))
hinge_calm = [mean(v) for k, v in cm.items() if k[2] != "2020"]
print(f"    numerator: min {min(num):.2f} max {max(num):.2f} "
      f"(cell means {min(mean(kappa(ds,'neggrad_hinge')) or 0 for ds in [[]] ):.0f})"
      if False else f"    numerator per run: min {min(num):.2f}  max {max(num):.2f}")
print(f"    hinge kappa on non-crisis eras: min {min(hinge_calm):.1f} "
      f"max {max(hinge_calm):.1f}   [paper: 6.6 to 30.2]")
