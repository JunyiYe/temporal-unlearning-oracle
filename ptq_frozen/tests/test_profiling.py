import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fin_ts_ptq.quantization.profiling import profile_clipping, profile_fp32
from fin_ts_ptq.quantization.rtn import fake_static_int8

failures = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: failures.append(name)

class FoldModel(nn.Module):
    """Folds channels into the leading dim like SegRNN/PatchTST do."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 16)
        self.rnn = nn.GRU(16, 16, batch_first=True)
        self.head = nn.Linear(16, 1)
    def forward(self, x):            # (B, T=24, C=8)
        B = x.shape[0]
        z = x.reshape(B * 4, 6, 8)   # fold: leading dim = B*4
        e = torch.relu(self.embed(z))
        _, hn = self.rnn(e)
        return self.head(hn[0]).reshape(B, 4).mean(dim=1, keepdim=True)

torch.manual_seed(0)
m = FoldModel().eval()
N = 100
xs = torch.randn(N, 24, 8); ys = torch.randn(N, 1)
loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=False)

stats, amax, preds = profile_fp32(m, loader, torch.device("cpu"))
check("stats cover all 3 layers", set(stats) == {"embed", "rnn", "head"}, str(sorted(stats)))
check("kurtosis finite", all(np.isfinite(v["excess_kurtosis"]) for v in stats.values()))
check("per-window amax aligned despite fold", all(len(a) == N for a in amax.values()),
      {k: len(v) for k, v in amax.items()})
check("preds shape", preds.shape == (N, 1))

q4 = fake_static_int8(m, [xs[:50], xs[50:]], num_bits=4)
clip, qpreds = profile_clipping(q4, loader, torch.device("cpu"))
check("clip arrays for all quant layers", len(clip) == 3, str(sorted(clip)))
check("clip fractions in [0,1]", all((a >= 0).all() and (a <= 1).all() for a in clip.values()))
check("clip aligned per window", all(len(a) == N for a in clip.values()))
# abs-max calibration on the same data → clipping should be rare but can exceed 0 on new data;
# here calibration == test data so clip fraction must be ~0
check("no clipping when calibrated on same data", max(float(a.mean()) for a in clip.values()) < 1e-3,
      f"max mean clip={max(float(a.mean()) for a in clip.values()):.2e}")

# ---------- multi-call alignment + layer_filter ----------
from fin_ts_ptq.quantization.rtn import FakeQuantLinear

class DoubleCall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.head = nn.Linear(8, 1)
    def forward(self, x):  # x: (B, 8); fc called twice
        return self.head(torch.relu(self.fc(torch.relu(self.fc(x)))))

dm = DoubleCall().eval()
xs2 = torch.randn(90, 8); loader2 = DataLoader(TensorDataset(xs2, torch.randn(90,1)), batch_size=32)
q = fake_static_int8(dm, [xs2], num_bits=4)
clip2, _ = profile_clipping(q, loader2, torch.device("cpu"))
check("double-call layer aligned to windows", len(clip2["fc"]) == 90, f"len={len(clip2['fc'])}")
stats2, amax2, _ = profile_fp32(dm, loader2, torch.device("cpu"))
check("double-call fp32 amax aligned", len(amax2["fc"]) == 90)

qf = fake_static_int8(dm, [xs2], num_bits=4, layer_filter=lambda n: n == "head")
check("layer_filter: only head quantized",
      isinstance(qf.head, FakeQuantLinear) and isinstance(qf.fc, nn.Linear) and not isinstance(qf.fc, FakeQuantLinear))
qn = fake_static_int8(dm, [xs2], num_bits=4, layer_filter=lambda n: n != "head")
check("layer_filter: loo excludes head",
      isinstance(qn.fc, FakeQuantLinear) and not isinstance(qn.head, FakeQuantLinear))

print(); print("FAILED:" if failures else "ALL TESTS PASSED", failures if failures else "")
sys.exit(1 if failures else 0)
