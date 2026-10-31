"""Tests for GRU quantization support (FakeQuantGRU, weight-only, dynamic, noise)."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn

from fin_ts_ptq.quantization.noise import count_perturbed_parameters, perturb_weights_like_rtn
from fin_ts_ptq.quantization.ptq import quantize_model
from fin_ts_ptq.quantization.rtn import (
    FakeQuantGRU,
    collect_linear_input_amax,
    fake_static_int8,
    rtn_w4,
)

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


class TinySegRNN(nn.Module):
    """Mirror of SegRNN's structure: Linear -> GRU -> (second GRU call w/ hx) -> Linear."""

    def __init__(self, d=32):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.rnn = nn.GRU(input_size=d, hidden_size=d, num_layers=1, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, x):  # x: (B, T, 1)
        e = torch.relu(self.embed(x))
        out, hn = self.rnn(e)
        out2, hy = self.rnn(e[:, :1, :], hn)  # second call with explicit hx, like SegRNN
        return self.head(hy[0])


torch.manual_seed(0)
model = TinySegRNN().eval()
x = torch.randn(64, 16, 1)

with torch.no_grad():
    ref = model(x)

# --- 1. high-bit FakeQuantGRU ~= exact nn.GRU (numerical correctness of unroll) ---
gru = model.rnn
qgru_hi = FakeQuantGRU(gru, input_amax=10.0, hidden_amax=10.0, num_bits=24)
e = torch.relu(model.embed(x)).detach()
with torch.no_grad():
    out_ref, hn_ref = gru(e)
    out_hi, hn_hi = qgru_hi(e)
check("unrolled GRU matches nn.GRU at high bits",
      torch.allclose(out_ref, out_hi, atol=1e-4) and torch.allclose(hn_ref, hn_hi, atol=1e-4),
      f"max diff={float((out_ref-out_hi).abs().max()):.2e}")

# hx path
with torch.no_grad():
    o1, h1 = gru(e[:, :3, :], hn_ref)
    o2, h2 = qgru_hi(e[:, :3, :], hn_ref)
check("hx path matches", torch.allclose(o1, o2, atol=1e-4), f"diff={float((o1-o2).abs().max()):.2e}")

# --- 2. calibration collects GRU amax keys ---
batches = [(x[i : i + 16], None) for i in range(0, 64, 16)]
amax = collect_linear_input_amax(model, [b[0] for b in batches])
check("GRU amax keys collected", "rnn.x" in amax and "rnn.h" in amax,
      f"keys={sorted(k for k in amax if 'rnn' in k)}")
check("GRU amax values positive", amax.get("rnn.x", 0) > 0 and amax.get("rnn.h", 0) > 0)

# --- 3. fake_static_int8 replaces the GRU and still runs end to end ---
q8 = fake_static_int8(model, [b[0] for b in batches], num_bits=8)
check("static quant replaces GRU", isinstance(q8.rnn, FakeQuantGRU))
with torch.no_grad():
    out8 = q8(x)
check("W8A8 output close to fp32", float((out8 - ref).abs().max()) < 0.1,
      f"max diff={float((out8-ref).abs().max()):.3f}")
q4 = fake_static_int8(model, [b[0] for b in batches], num_bits=4)
with torch.no_grad():
    out4 = q4(x)
check("W4A4 differs more than W8A8",
      float((out4 - ref).abs().mean()) > float((out8 - ref).abs().mean()))

# --- 4. rtn_w4 quantizes GRU weights ---
qw = rtn_w4(model)
check("rtn_w4 changes GRU weights", not torch.equal(qw.rnn.weight_ih_l0, model.rnn.weight_ih_l0))
check("rtn_w4 keeps GRU biases", torch.equal(qw.rnn.bias_ih_l0, model.rnn.bias_ih_l0))
with torch.no_grad():
    outw = qw(x)
check("rtn_w4 runs and stays close", float((outw - ref).abs().max()) < 0.5)

# --- 5. dynamic_int8 includes GRU ---
qd = quantize_model(model, "dynamic_int8")
dyn_types = {type(m).__name__ for m in qd.modules()}
check("dynamic quantizes GRU", "GRU" in str(dyn_types) and any("quantized" in str(type(m)).lower() for m in qd.modules()),
      f"types={sorted(t for t in dyn_types if 'GRU' in t or 'Linear' in t)}")
with torch.no_grad():
    outd = qd(x)
check("dynamic_int8 runs", outd.shape == ref.shape)

# --- 6. noise perturbs GRU weights; coverage counts them ---
noised = perturb_weights_like_rtn(model, num_bits=4, multiplier=1.0, seed=1)
check("noise perturbs GRU weights", not torch.equal(noised.rnn.weight_hh_l0, model.rnn.weight_hh_l0))
cov = count_perturbed_parameters(model)
gru_w = model.rnn.weight_ih_l0.numel() + model.rnn.weight_hh_l0.numel()
lin_w = model.embed.weight.numel() + model.head.weight.numel()
check("coverage counts GRU weights", cov["perturbed_weights"] == gru_w + lin_w,
      f"{cov['perturbed_weights']} vs {gru_w + lin_w}, coverage={cov['weight_coverage']:.2%}")

# ---------- Conv1d (TokenEmbedding-style, circular padding) ----------
from fin_ts_ptq.quantization.rtn import FakeQuantConv1d

conv = nn.Conv1d(11, 32, kernel_size=3, padding=1, padding_mode="circular", bias=False).eval()
xc = torch.randn(8, 11, 64)
with torch.no_grad():
    conv_ref = conv(xc)

fq_hi = FakeQuantConv1d(conv, activation_amax=10.0, num_bits=24)
with torch.no_grad():
    out_hi = fq_hi(xc)
check("FakeQuantConv1d matches conv at high bits (circular pad preserved)",
      torch.allclose(conv_ref, out_hi, atol=1e-4), f"diff={float((conv_ref-out_hi).abs().max()):.2e}")


class TinyConvEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(11, 32, 3, padding=1, padding_mode="circular", bias=False)
        self.head = nn.Linear(32, 1)

    def forward(self, x):  # (B, T, C)
        h = self.embed(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.head(torch.relu(h)).mean(dim=1)


cm = TinyConvEmbed().eval()
xb = torch.randn(32, 64, 11)
with torch.no_grad():
    cref = cm(xb)
q8c = fake_static_int8(cm, [xb[:16], xb[16:]], num_bits=8)
check("static quant replaces Conv1d", isinstance(q8c.embed, FakeQuantConv1d))
with torch.no_grad():
    c8 = q8c(xb)
check("conv-embed W8A8 close to fp32", float((c8 - cref).abs().max()) < 0.1,
      f"diff={float((c8-cref).abs().max()):.3f}")

qwc = rtn_w4(cm)
check("rtn_w4 qdq's conv weights in place", isinstance(qwc.embed, nn.Conv1d)
      and not torch.equal(qwc.embed.weight, cm.embed.weight))
check("rtn_w4 conv keeps circular padding", qwc.embed.padding_mode == "circular")
covc = count_perturbed_parameters(cm)
check("conv counted in coverage", covc["perturbed_weights"] == cm.embed.weight.numel() + cm.head.weight.numel())

print()
print("FAILED:" if failures else "ALL TESTS PASSED", failures if failures else "")
sys.exit(1 if failures else 0)
