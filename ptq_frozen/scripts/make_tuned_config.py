"""Emit configs/<model>_sp500_vol_tuned.yaml from tuning selection results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Write tuned config from chosen_configs.json.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", default="configs/{model}_sp500_vol_opt.yaml")
    parser.add_argument("--chosen", default="outputs/tuning/{model}/chosen_configs.json")
    parser.add_argument("--output", default="configs/{model}_sp500_vol_tuned.yaml")
    args = parser.parse_args()

    template_path = Path(str(args.template).format(model=args.model))
    chosen_path = Path(str(args.chosen).format(model=args.model))
    output_path = Path(str(args.output).format(model=args.model))

    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    if args.model not in chosen:
        raise SystemExit(f"{chosen_path} has no entry for {args.model!r}")
    c = chosen[args.model]
    cfg["model"].update(
        d_model=int(c["d_model"]),
        dropout=float(c["dropout"]),
        d_ff=int(c["d_ff"]),
        n_layers=int(c["n_layers"]),
    )
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[OK] {output_path}: d_model={c['d_model']} dropout={c['dropout']} "
          f"(mean_val_loss={c['mean_val_loss']:.5f})")


if __name__ == "__main__":
    main()
