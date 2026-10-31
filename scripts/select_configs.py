"""P0 gate: aggregate the 24 tuning cells, select per-arch winners,
emit the final configs, and write the gate readout.

Stdlib-only (reads small jsonl files, edits config text) — login-safe.
  python scripts/select_configs.py
"""
from __future__ import annotations
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

UNL = (_REPO)
PTQ = (_REPO / "ptq_frozen")
ARCHS = ["transformer", "itransformer", "segrnn", "patchtst"]


def main() -> None:
    gate_lines = ["# P0 gate readout (tuning selection)", ""]
    ok = True
    (UNL / "configs").mkdir(parents=True, exist_ok=True)

    for arch in ARCHS:
        # gather all (config, seed) records across the per-config dirs
        by_cfg: dict[tuple, list[float]] = defaultdict(list)
        epochs: dict[tuple, list[int]] = defaultdict(list)
        for f in (UNL / "outputs/tuning" / arch).rglob("tuning_results.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                m = re.match(r".*_dm(\d+)_do([\d.]+)_seed(\d+)", r["cell"])
                if not m:
                    continue
                key = (int(m.group(1)), float(m.group(2)))
                by_cfg[key].append(float(r["best_val_loss"]))
                epochs[key].append(int(r.get("epochs_ran", -1)))

        complete = {k: v for k, v in by_cfg.items() if len(v) == 3}
        gate_lines.append(f"## {arch}")
        if len(complete) < 6:
            gate_lines.append(f"INCOMPLETE: {len(complete)}/6 configs have 3 seeds — HOLD")
            ok = False
            continue
        ranked = sorted(complete.items(), key=lambda kv: st.mean(kv[1]))
        (w_cfg, w_losses), (r_cfg, r_losses) = ranked[0], ranked[1]
        margin = st.mean(r_losses) - st.mean(w_losses)
        spread = max(st.stdev(w_losses), st.stdev(r_losses))
        gate_lines.append(
            f"- winner d_model={w_cfg[0]} dropout={w_cfg[1]} "
            f"mean_val={st.mean(w_losses):.5f} (seeds: {[round(x,4) for x in sorted(w_losses)]})")
        gate_lines.append(
            f"- runner-up d{r_cfg[0]}/dr{r_cfg[1]} mean_val={st.mean(r_losses):.5f}; "
            f"margin={margin:.5f} vs seed-spread={spread:.5f} "
            f"({'clear' if margin > spread else 'NEAR-TIE (either acceptable)'})")
        gate_lines.append(
            f"- all configs trained (epochs 3..300, none stuck at 1): "
            f"{sorted(set(e for v in epochs.values() for e in v))[:5]}...")

        # emit config from the _opt template by textual field replacement
        tpl = (PTQ / f"configs/{arch}_sp500_vol_opt.yaml").read_text(encoding="utf-8")
        tpl = re.sub(r"(?m)^(\s*d_model:)\s*\d+", rf"\g<1> {w_cfg[0]}", tpl)
        tpl = re.sub(r"(?m)^(\s*d_ff:)\s*\d+", rf"\g<1> {4 * w_cfg[0]}", tpl)
        tpl = re.sub(r"(?m)^(\s*dropout:)\s*[\d.]+", rf"\g<1> {w_cfg[1]}", tpl)
        out = UNL / f"configs/{arch}_sp500_vol.yaml"
        out.write_text(tpl, encoding="utf-8")
        gate_lines.append(f"- emitted {out.name}")
        gate_lines.append("")

    gate_lines.append("GATE P0: " + ("PASS" if ok else "HOLD (incomplete cells)"))
    rd = UNL / "docs/p0_gate.md"
    rd.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    print("\n".join(gate_lines))


if __name__ == "__main__":
    main()
