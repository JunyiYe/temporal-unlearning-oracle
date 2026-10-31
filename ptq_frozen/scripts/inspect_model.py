import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fin_ts_ptq.config import load_config
from fin_ts_ptq.models.factory import build_model


def show_model(model, label):
    total = linear = conv1d = other = 0
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")
    print(f"{'Name':<58} {'Type':<30} {'Shape':<28} {'Params':<8} Quantized?")
    print("-" * 150)
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        p = sum(p.numel() for p in module.parameters())
        total += p
        t = type(module).__name__
        is_lin = isinstance(module, torch.nn.Linear)
        is_conv = isinstance(module, torch.nn.Conv1d)
        if is_lin:
            linear += p
            q = "YES(Linear)"
        elif is_conv:
            conv1d += p
            q = "YES(Conv1d)" if "RTN" in t else "NO(Conv1d)"
        else:
            other += p
            q = "NO"
        shapes = ", ".join(str(tuple(p.shape)) for p in module.parameters()) or "-"
        print(f"{name:<58} {t:<30} {shapes:<28} {p:<8} {q}")

    print("-" * 150)
    print(f"{'Total':<58} {'':30} {'':28} {total:<8}")
    print(f"  Linear: {linear:>6,} ({100*linear/total:.1f}%)")
    print(f"  Conv1d: {conv1d:>6,} ({100*conv1d/total:.1f}%)")
    print(f"  Other:  {other:>6,} ({100*other/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Inspect PatchTST model layers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--q", action="store_true", help="Also show quantized model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = cfg["task"]
    model_cfg = cfg["model"]

    model = build_model(model_cfg, task_cfg)
    ckpt_path = Path("outputs") / f"{Path(args.config).stem}_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    show_model(model, "Original model")

    if args.q:
        from fin_ts_ptq.quantization.rtn import rtn_w4
        qmodel = rtn_w4(model, num_bits=4)
        show_model(qmodel, "After RTN W4")


if __name__ == "__main__":
    main()
