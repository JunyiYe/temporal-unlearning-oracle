"""Equal-budget hyperparameter tuning on a dedicated tuning split.

Protocol (frozen for the paper): every tunable model searches the same grid
(d_model x dropout, depth fixed at 2, d_ff = 4*d_model) with the same trainer
settings, selected by mean validation loss over seeds on a tuning split that
never touches any evaluation fold's test year. DLinear has no capacity knobs
and enters the paper as-is.

The window datasets are built once and shared across all grid cells and
models (they only depend on the task/data sections, which are asserted
identical across model templates).

    python scripts/run_tuning.py \
        --models tsmixer,timemixer,itransformer,patchtst,transformer,segrnn

Resumable: finished cells are read back from the results JSONL and skipped.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from torch.utils.data import DataLoader

from fin_ts_ptq.config import load_config
from fin_ts_ptq.experiment import build_datasets
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.runner import pick_keys
from fin_ts_ptq.training.loop import TrainConfig, train_model
from fin_ts_ptq.utils.device import resolve_device
from fin_ts_ptq.utils.seed import set_seed


def parse_list(v: str, cast=str) -> list:
    return [cast(p) for p in str(v).split(",") if p != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Equal-budget hyperparameter search on the tuning split.")
    parser.add_argument("--models", default="tsmixer,timemixer,itransformer,patchtst,transformer,segrnn")
    parser.add_argument("--template", default="configs/{model}_sp500_vol_opt.yaml")
    parser.add_argument("--d-models", default="32,64,128")
    parser.add_argument("--dropouts", default="0.1,0.3")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seeds", default="7,8")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--val-end", default="2019-12-31")
    parser.add_argument("--train-start", default=None,
                        help="Exclusive lower bound of the training window (fixed-window tuning).")
    parser.add_argument("--data-path", default=None,
                        help="Override data.path in every model template (e.g. the extended panel).")
    parser.add_argument("--output-dir", default="outputs/tuning")
    parser.add_argument("--epochs-override", type=int, default=None, help="Smoke-testing only.")
    args = parser.parse_args()

    models = parse_list(args.models)
    d_models = parse_list(args.d_models, int)
    dropouts = parse_list(args.dropouts, float)
    seeds = parse_list(args.seeds, int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tuning_results.jsonl"

    done: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["cell"]] = rec

    # load templates; the shared data build below is only valid if every
    # model's task/data sections agree
    templates = {m: load_config(str(args.template).format(model=m)) for m in models}
    reference = next(iter(templates.values()))
    for m, cfg in templates.items():
        for section in ("task", "data"):
            if cfg[section] != reference[section]:
                raise ValueError(f"{m} template differs from reference in [{section}]; cannot share datasets.")

    data_cfg_tuning = copy.deepcopy(reference)
    data_cfg_tuning["data"]["train_end"] = args.train_end
    data_cfg_tuning["data"]["val_end"] = args.val_end
    data_cfg_tuning["data"]["test_end"] = args.val_end  # tuning never sees any test year
    if args.train_start:
        data_cfg_tuning["data"]["train_start"] = args.train_start
    if args.data_path:
        data_cfg_tuning["data"]["path"] = args.data_path
    print(f"[DATA] building tuning datasets (train ({args.train_start}, {args.train_end}], "
          f"val<= {args.val_end}, path={data_cfg_tuning['data']['path']})", flush=True)
    data = build_datasets(data_cfg_tuning, splits=("train", "val"))
    train_ds, val_ds = data.splits["train"], data.splits["val"]
    print(f"[DATA] train_windows={len(train_ds)} val_windows={len(val_ds)}", flush=True)

    total = len(models) * len(d_models) * len(dropouts) * len(seeds)
    n = 0
    with results_path.open("a", encoding="utf-8") as sink:
        for model_name in models:
            for d_model in d_models:
                for dropout in dropouts:
                    for seed in seeds:
                        n += 1
                        cell = f"{model_name}_dm{d_model}_do{dropout:g}_seed{seed}"
                        if cell in done:
                            print(f"[SKIP {n}/{total}] {cell}", flush=True)
                            continue
                        print(f"[RUN {n}/{total}] {cell}", flush=True)
                        cfg = copy.deepcopy(templates[model_name])
                        cfg["seed"] = seed
                        cfg["model"].update(
                            d_model=d_model,
                            dropout=dropout,
                            d_ff=4 * d_model,
                            n_layers=int(args.n_layers),
                        )
                        if args.epochs_override:
                            cfg["train"]["epochs"] = args.epochs_override
                            cfg["train"]["patience"] = args.epochs_override
                        set_seed(seed)
                        train_cfg = TrainConfig(**pick_keys(cfg.get("train", {}), TrainConfig.__annotations__.keys()))
                        batch_size = int(cfg.get("train", {}).get("batch_size", train_cfg.batch_size))
                        device = resolve_device(cfg.get("train", {}).get("device", "auto"))
                        model = build_model(cfg["model"], cfg["task"], num_features=len(data.feature_columns))
                        n_params = sum(p.numel() for p in model.parameters())
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
                        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
                        result = train_model(model, train_loader, val_loader, train_cfg, device)
                        rec = {
                            "cell": cell,
                            "model": model_name,
                            "d_model": d_model,
                            "dropout": dropout,
                            "n_layers": int(args.n_layers),
                            "seed": seed,
                            "n_params": n_params,
                            "best_val_loss": result.best_val_loss,
                            "epochs_ran": result.epochs_ran,
                        }
                        sink.write(json.dumps(rec) + "\n")
                        sink.flush()
                        done[cell] = rec

    # selection: per model, argmin of mean val loss over seeds
    chosen = {}
    for model_name in models:
        combos: dict[tuple[int, float], list[float]] = {}
        for rec in done.values():
            if rec["model"] != model_name:
                continue
            combos.setdefault((rec["d_model"], rec["dropout"]), []).append(rec["best_val_loss"])
        scored = {k: sum(v) / len(v) for k, v in combos.items() if len(v) == len(seeds)}
        if not scored:
            continue
        (best_dm, best_do), best_loss = min(scored.items(), key=lambda kv: kv[1])
        chosen[model_name] = {
            "d_model": best_dm,
            "dropout": best_do,
            "d_ff": 4 * best_dm,
            "n_layers": int(args.n_layers),
            "mean_val_loss": best_loss,
            "all_combos": {f"dm{k[0]}_do{k[1]:g}": v for k, v in sorted(scored.items())},
        }
    chosen_path = output_dir / "chosen_configs.json"
    chosen_path.write_text(json.dumps(chosen, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({m: {k: v for k, v in c.items() if k != "all_combos"} for m, c in chosen.items()}, indent=2))
    print(f"[OK] Wrote {chosen_path}")


if __name__ == "__main__":
    main()
