"""Build every configs/scaling model on CPU and dump trainable-param counts.

Catches config errors (head divisibility, missing keys) before any sbatch
spend, and produces the params-vs-width map used as the scaling x-axis.
Run via srun on a CPU node.
"""
from __future__ import annotations

import json
from pathlib import Path

from fin_ts_ptq.config import load_config
from fin_ts_ptq.models.factory import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "scaling_param_counts.json"


def main() -> None:
    rows = {}
    for path in sorted((ROOT / "configs" / "scaling").glob("*.yaml")):
        cfg = load_config(str(path))
        n_features = len(cfg["task"]["feature_columns"])
        model = build_model(cfg["model"], cfg["task"], num_features=n_features)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows[path.stem] = n_params
        print(f"{path.stem}: {n_params}")
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"[OK] {len(rows)} configs -> {OUT}")


if __name__ == "__main__":
    main()
