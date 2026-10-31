"""CPU latency/memory for fp32 vs dynamic INT8 — real deployment numbers.

Dynamic INT8 uses torch's actual quantized kernels (unlike the fake-quant
simulation used for accuracy experiments), so these are honest wall-clock and
model-size measurements for the practical-guidance section.

    python scripts/latency_bench.py --configs configs/tsmixer_sp500_vol_tuned.yaml,...
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch

from fin_ts_ptq.config import load_config
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.ptq import quantize_model


def state_bytes(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def bench(model: torch.nn.Module, x: torch.Tensor, warmup: int = 3, iters: int = 20) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        return (time.perf_counter() - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU latency/memory: fp32 vs dynamic INT8.")
    parser.add_argument("--configs", required=True, help="Comma-separated config paths.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    report = {"batch_size": args.batch_size, "threads": args.threads, "models": {}}
    for cfg_path in [c for c in str(args.configs).split(",") if c]:
        cfg = load_config(cfg_path)
        stem = Path(cfg_path).stem
        model = build_model(cfg["model"], cfg["task"], num_features=len(cfg["task"]["feature_columns"]))
        ckpt = Path("outputs") / f"{stem}_model.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        x = torch.randn(args.batch_size, int(cfg["task"]["context_length"]), len(cfg["task"]["feature_columns"]))

        fp32_s = bench(model.cpu(), x)
        qd = quantize_model(model, "dynamic_int8")
        int8_s = bench(qd, x)
        entry = {
            "fp32_ms_per_batch": 1e3 * fp32_s,
            "dyn_int8_ms_per_batch": 1e3 * int8_s,
            "speedup": fp32_s / int8_s,
            "fp32_bytes": state_bytes(model),
            "dyn_int8_bytes": state_bytes(qd),
        }
        entry["size_ratio"] = entry["fp32_bytes"] / max(entry["dyn_int8_bytes"], 1)
        report["models"][stem] = entry
        print(f"{stem:36s} fp32={entry['fp32_ms_per_batch']:7.1f}ms  int8={entry['dyn_int8_ms_per_batch']:7.1f}ms  "
              f"speedup={entry['speedup']:.2f}x  size {entry['fp32_bytes']/1e3:.0f}K->{entry['dyn_int8_bytes']/1e3:.0f}K")

    out = Path("outputs") / "latency_bench.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()
