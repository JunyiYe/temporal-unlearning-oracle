"""Can each architecture learn a target that is trivially present in its input?

PatchTST and iTransformer plateau at val ~0.60 on the real task while SegRNN,
which consumes the same single channel, reaches ~0.52. Hyperparameters, depth,
patch geometry and channel access have all been ruled out. This probe asks a
narrower question: given a target that is a deterministic function of the
input window's channel 0, can the architecture recover it at all?

  probe A  in-window realised vol of the LAST 5 steps of channel 0
  probe B  mean of channel 0 over the last 5 steps

Both are computable from the input by construction, so a healthy architecture
should drive the loss far below the variance of the target. An architecture
that cannot is structurally impaired in how we call it.
"""
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]
import sys, torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, str(_REPO / "ptq_frozen" / "src"))
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.config import load_config

CFG = str(_REPO / "configs") + "/{}_sp500_vol.yaml"
task = {"context_length": 64, "horizon": 1}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
N = 8192
x = torch.randn(N, 64, 11)
# make channel 0 heteroskedastic so "recent volatility" genuinely varies
scale = torch.rand(N, 1) * 2.0 + 0.2
x[:, :, 0] *= scale
targets = {
    "A_recent_vol": x[:, -5:, 0].std(dim=1, keepdim=True),
    "B_recent_mean": x[:, -5:, 0].mean(dim=1, keepdim=True),
}

# Probe C runs the MATCHED condition: the real pipeline z-scores every window
# per feature before the model sees it, which makes RevIN a no-op. The target
# is then relative recent volatility -- pure shape, no scale.
xz = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-8)
targets_matched = {"C_shape_vol_zscored_input": xz[:, -5:, 0].std(dim=1, keepdim=True)}

for tname, y in list(targets.items()) + list(targets_matched.items()):
    xin = xz if tname.startswith("C_") else x
    y = (y - y.mean()) / y.std()                     # unit-variance target
    print(f"\n=== probe {tname}  (predict-the-mean baseline MSE = 1.000) ===")
    ds = TensorDataset(xin, y)
    tr = DataLoader(torch.utils.data.Subset(ds, range(6144)), batch_size=512, shuffle=True)
    va = DataLoader(torch.utils.data.Subset(ds, range(6144, N)), batch_size=512)
    for name in ('transformer', 'tsmixer', 'segrnn', 'itransformer', 'patchtst'):
        cfg = load_config(CFG.format(name))
        torch.manual_seed(0)
        m = build_model(cfg['model'], cfg['task'], num_features=11).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        best = float('inf')
        for ep in range(30):
            m.train()
            for xb, yb in tr:
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(m(xb.to(dev)), yb.to(dev))
                loss.backward(); opt.step()
            m.eval(); tot = n = 0
            with torch.no_grad():
                for xb, yb in va:
                    p = m(xb.to(dev)); tot += torch.nn.functional.mse_loss(
                        p, yb.to(dev), reduction='sum').item(); n += len(yb)
            best = min(best, tot / n)
        verdict = "recovers it" if best < 0.2 else ("partial" if best < 0.7 else "CANNOT LEARN IT")
        print(f"  {name:14s} best val MSE = {best:.4f}   {verdict}")
