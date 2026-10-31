from __future__ import annotations

import argparse
from pathlib import Path

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import parse_range
from fin_ts_ptq.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a financial time-series PTQ experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics/preds/checkpoint.")
    parser.add_argument("--clip", default=None, help="Clip predictions to [low,high], e.g. --clip -3,3")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(
        cfg,
        label=Path(args.config).stem,
        output_dir=args.output_dir,
        clip_range=parse_range(args.clip),
        config_name=str(args.config),
    )


if __name__ == "__main__":
    main()
