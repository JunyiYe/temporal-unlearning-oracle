from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch import nn

from fin_ts_ptq.quantization.rtn import fake_static_int8, rtn_w4


LINEAR_DYNAMIC_SET = {nn.Linear, nn.GRU}


def quantize_model(
    model: nn.Module,
    method: str,
    calibration_batches: Iterable[torch.Tensor] | None = None,
) -> nn.Module:
    """Return an eval-mode PTQ copy of a trained model."""
    if method == "fp32":
        return copy.deepcopy(model).eval()
    if method == "dynamic_int8":
        return dynamic_int8(model)
    if method == "static_int8":
        if calibration_batches is None:
            raise ValueError("static_int8 requires calibration_batches.")
        return fake_static_int8(model, calibration_batches, num_bits=8)
    if method == "static_int4":
        if calibration_batches is None:
            raise ValueError("static_int4 requires calibration_batches.")
        return fake_static_int8(model, calibration_batches, num_bits=4)
    if method == "rtn_w4":
        return rtn_w4(model)
    raise ValueError(f"Unsupported PTQ method: {method}")


def dynamic_int8(model: nn.Module) -> nn.Module:
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    qmodel = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(qmodel, LINEAR_DYNAMIC_SET, dtype=torch.qint8)
