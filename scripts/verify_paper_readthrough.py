"""Verify the specific numbers questioned by the full read-through of the v2 paper.

Read-only. Prints each claim with the measured value.
NOTE: models[m]['vs_reference']['delta_ic_mean'] is stored with the OPPOSITE
sign to Eq. 3; never read it. Always compute IC(ora) - IC(ref) directly from
test_daily_ic_mean.  The HAC t-stat in oracle.vs_reference.tstat is likewise
signed against Eq. 3, so it is reported with the sign flipped.
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import json, glob, os, math, re
from collections import defaultdict

PILOT = str(_REPO / "outputs" / "pilot")
FILES = sorted(glob.glob(os.path.join(PILOT, "*_unlearn.json")))
PAT = re.compile(r"^(?P<arch>[a-z]+)_sp500_vol_v2_era(?P<era>\d{4})_"
                 r"(?P<mode>strict|target|l2halo)_fold(?P<fold>\d{4})_seed(?P<seed>\d+)_unlearn\.json$")

rows = []
for f in FILES:
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    try:
        rows.append((m.groupdict(), json.load(open(f))))
    except Exception:
        pass
print(f"cells on disk {len(FILES)}, parsed {len(rows)}")

ARCHS = ["transformer", "tsmixer", "segrnn", "itransformer", "patchtst"]
DEPLOY = ["transformer", "tsmixer", "segrnn"]


def get(d, model, key):
    try:
        v = d["models"][model][key]
    except (KeyError, TypeError):
        return None
    return v if isinstance(v, (int, float)) else None


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# index: (arch, fold, era, mode) -> list of per-seed dicts
IDX = defaultdict(list)
for g, d in rows:
    IDX[(g["arch"], g["fold"], g["era"], g["mode"])].append(d)


def delta(a, f, e, mode="strict"):
    ds = IDX.get((a, f, e, mode), [])
    v = [get(d, "oracle", "test_daily_ic_mean") - get(d, "reference", "test_daily_ic_mean")
         for d in ds
         if get(d, "oracle", "test_daily_ic_mean") is not None
         and get(d, "reference", "test_daily_ic_mean") is not None]
    return v


def gapF(a, f, e, mode="strict"):
    ds = IDX.get((a, f, e, mode), [])
    return [get(d, "oracle", "forget_mse") - get(d, "reference", "forget_mse") for d in ds
            if get(d, "oracle", "forget_mse") is not None
            and get(d, "reference", "forget_mse") is not None]


def hac_t(a, f, e, mode="strict"):
    """Newey-West t for Delta, sign-corrected to Eq. 3."""
    ds = IDX.get((a, f, e, mode), [])
    out = []
    for d in ds:
        try:
            out.append(-d["models"]["oracle"]["vs_reference"]["tstat"])
        except (KeyError, TypeError):
            pass
    return out


print("\n" + "=" * 78)
print("A. Table 2 'Stress Delta (2022)' vs Table 4 col 2022  (era2020, L1)")
print("   arch          n   Delta      sd     HAC t (mean over seeds)")
for a in ARCHS:
    v = delta(a, "2022", "2020")
    t = hac_t(a, "2022", "2020")
    if v:
        print(f"   {a:13s} {len(v):2d}  {mean(v):+.4f}  {sd(v):.4f}   "
              f"{mean(t):+.1f}" if t else "")

print("\n" + "=" * 78)
print("B. reference test IC per arch per fold  (Table 2 'test IC calm/stress')")
for a in ARCHS:
    out = []
    for fold in ["2022", "2023", "2024", "2025"]:
        v = [get(d, "reference", "test_daily_ic_mean")
             for (aa, ff, ee, mm), ds in IDX.items() if aa == a and ff == fold and mm == "strict"
             for d in ds if get(d, "reference", "test_daily_ic_mean") is not None]
        out.append(f"{fold}:{mean(v):.3f}(n={len(v)})" if v else f"{fold}:--")
    print(f"   {a:13s} " + "  ".join(out))

print("\n" + "=" * 78)
print("C. era-position ladder, stress fold 2022, L1 (paper: earliest inert |D|<=0.008;")
print("   the two preceding the crisis '-1.9 to -3.4 points, |t| = 3.1 to 10.3')")
for a in DEPLOY:
    parts = []
    for e in ["2016", "2017", "2018", "2019", "2020"]:
        v, t = delta(a, "2022", e), hac_t(a, "2022", e)
        parts.append(f"{e}:{mean(v):+.4f}(t{mean(t):+.1f})" if v and t else f"{e}:--")
    print(f"   {a:13s} " + " ".join(parts))
print("   L0 (target mode) check for eras 2018/2019:")
for a in DEPLOY:
    parts = []
    for e in ["2018", "2019"]:
        v = delta(a, "2022", e, "target")
        parts.append(f"{e}:{mean(v):+.4f}" if v else f"{e}:--")
    print(f"   {a:13s} " + " ".join(parts))

print("\n" + "=" * 78)
print("D. calm-fold vs stress-fold magnitude ('at a fifth of the magnitude')")
for e in ["2019", "2020"]:
    for a in DEPLOY:
        s, c = mean(delta(a, "2022", e)), mean(delta(a, "2024", e))
        if s and c:
            print(f"   era{e} {a:13s} stress {s:+.4f}  calm {c:+.4f}  calm/stress {c/s:.2f}")

print("\n" + "=" * 78)
print("E. transformer G ladder, calm fold 2024, L1")
print("   (paper: cluster 0.037-0.064; 2022 +0.121; 2020 +0.363; ratio 5.6-9.8x)")
gl = {}
for e in ["2018", "2019", "2020", "2021", "2022"]:
    g = gapF("transformer", "2024", e)
    if g:
        gl[e] = mean(g)
        print(f"   era{e}: G={mean(g):+.4f} (n={len(g)})")
if len(gl) == 5:
    cl = [gl[e] for e in ["2018", "2019", "2021"]]
    print(f"   crisis/cluster ratio: {gl['2020']/max(cl):.1f}x to {gl['2020']/min(cl):.1f}x")

print("\n" + "=" * 78)
print("F. kappa denominator across eras (RQ2: '6-10x smaller on a low-volatility era';")
print("   'the denominator falls from 0.36 to 0.009')")
allg = {}
for (a, f, e, m), ds in IDX.items():
    if m != "strict":
        continue
    g = gapF(a, f, e)
    if g:
        allg[(a, f, e)] = mean(g)
cr = {k: v for k, v in allg.items() if k[2] == "2020"}
nc = {k: v for k, v in allg.items() if k[2] != "2020"}
print(f"   crisis(2020) G over {len(cr)} (arch,fold): min {min(cr.values()):+.4f} "
      f"max {max(cr.values()):+.4f}")
ncs = sorted(nc.items(), key=lambda kv: kv[1])
print(f"   non-crisis G over {len(nc)} cells: min {ncs[0][1]:+.4f} {ncs[0][0]} "
      f"max {ncs[-1][1]:+.4f} {ncs[-1][0]}")
print("   per (arch,fold) crisis / mean non-crisis:")
for a in ARCHS:
    for f in ["2022", "2023", "2024", "2025"]:
        c = allg.get((a, f, "2020"))
        o = [v for (aa, ff, ee), v in allg.items() if aa == a and ff == f and ee != "2020"]
        if c and o:
            print(f"     {a:13s} fold{f}: crisis {c:+.4f}  non-crisis mean {mean(o):+.4f}"
                  f"  ratio {c/mean(o):5.1f}x   (non-crisis min {min(o):+.4f})")

print("\n" + "=" * 78)
print("G. compute accounting")
n_or = sum(1 for _, d in rows if get(d, "oracle", "test_daily_ic_mean") is not None)
print(f"   parsed cells with an oracle: {n_or}")
tk = set()
for _, d in rows:
    for k in d.get("models", {}).get("oracle", {}) or {}:
        tk.add(k)
print(f"   oracle keys present: {sorted(tk)}")
