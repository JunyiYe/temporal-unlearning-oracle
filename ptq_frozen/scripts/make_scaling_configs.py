"""Generate the fin-scaling-laws params-axis ladder configs.

Vol task: extends the existing width convention ({arch}_sp500_vol_w{N}) to
w8 and w512 and clones the mid-ladder levels with fp32-only PTQ so new
cells skip the quantization eval (existing outputs/width cells are reused
via run_sweep --skip-existing; the fp32 training path is identical).

Return task ({arch}_sp500_ret_w{N}): same model/train blocks, task block
swapped to next-day cross-sectional-zscore return (the low-SNR contrast),
matching the original *_sp500_opt.yaml task definition.

Everything except d_model/d_ff stays frozen at the arch's tuned values —
the same freeze the ICAIF width figure used.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "scaling"
ARCHS = ["transformer", "tsmixer", "patchtst", "itransformer"]
LADDER = [8, 16, 32, 64, 128, 256, 512]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for arch in ARCHS:
        for width in LADDER:
            template_w = 16 if width == 8 else 256 if width == 512 else width
            tpl_path = ROOT / "configs" / f"{arch}_sp500_vol_w{template_w}.yaml"
            cfg = yaml.safe_load(tpl_path.read_text())

            cfg["model"]["d_model"] = width
            if "d_ff" in cfg["model"]:
                cfg["model"]["d_ff"] = 4 * width
            heads = cfg["model"].get("n_heads")
            assert heads is None or width % heads == 0, (arch, width, heads)
            cfg["ptq"] = {"enabled": True, "methods": ["fp32"]}

            vol_path = OUT / f"{arch}_sp500_vol_w{width}.yaml"
            vol_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

            ret = copy.deepcopy(cfg)
            task = ret["task"]
            ret["task"] = {
                "context_length": task["context_length"],
                "horizon": 1,
                "target": "return",
                "normalize": task.get("normalize", "zscore"),
                "target_transform": task.get("target_transform", "cross_sectional_zscore"),
                "target_clip": task.get("target_clip", "-3,3"),
                "feature_columns": task["feature_columns"],
            }
            ret_path = OUT / f"{arch}_sp500_ret_w{width}.yaml"
            ret_path.write_text(yaml.safe_dump(ret, sort_keys=False))
            n += 2
    print(f"[OK] wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
