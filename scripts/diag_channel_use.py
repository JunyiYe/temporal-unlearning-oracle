"""Do the trained models actually use the engineered feature channels?

The first version of this probe used untrained models and printed 5 decimals;
an architecture with weak cross-channel coupling at initialisation would read
as exactly 0. This version loads real trained checkpoints and reports the
relative sensitivity, so "independent" means structurally independent rather
than below print precision.
"""
import sys, glob, torch
sys.path.insert(0, '/nfs/storage1/home/ye/fin-ts-ptq/src')
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.config import load_config

CFG = '/nfs/storage1/home/ye/fin-unlearning/v2/configs/{}_sp500_vol_v2.yaml'
SWEEP = '/nfs/storage1/home/ye/fin-unlearning/v2/outputs/sweep/{}_sp500_vol_v2_fold2024_seed4_model.pt'
task = {"context_length": 64, "horizon": 1}

torch.manual_seed(0)
x = torch.randn(64, 64, 11)
print(f"{'arch':14s}{'|d out| chan0':>15s}{'|d out| chans1-10':>19s}{'ratio':>10s}   verdict")
for name in ('transformer', 'tsmixer', 'segrnn', 'itransformer', 'patchtst'):
    try:
        cfg = load_config(CFG.format(name))
        m = build_model(cfg['model'], cfg['task'], num_features=11)
        ck = SWEEP.format(name)
        trained = False
        if glob.glob(ck):
            m.load_state_dict(torch.load(ck, map_location='cpu')); trained = True
        m.eval()
        with torch.no_grad():
            base = m(x)
            x0 = x.clone(); x0[:, :, 0] += 0.5
            d0 = (m(x0) - base).abs().mean().item()
            xr = x.clone(); xr[:, :, 1:] += 0.5
            dr = (m(xr) - base).abs().mean().item()
        ratio = dr / d0 if d0 > 0 else float('nan')
        verdict = ("CHANNEL-INDEPENDENT" if dr < 1e-9 else
                   "uses features weakly" if ratio < 0.01 else "uses all channels")
        tag = "" if trained else "  (UNTRAINED - no ckpt)"
        print(f"{name:14s}{d0:>15.3e}{dr:>19.3e}{ratio:>10.2e}   {verdict}{tag}")
    except Exception as e:
        print(f"{name:14s}  FAILED: {type(e).__name__}: {e}")
