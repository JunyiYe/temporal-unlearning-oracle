"""Compute quantization coverage for every architecture -> outputs/coverage.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import yaml
from fin_ts_ptq.models.factory import build_model
from fin_ts_ptq.quantization.noise import count_perturbed_parameters

out = {}
for m in ["dlinear", "tsmixer", "timemixer", "transformer", "itransformer", "patchtst", "segrnn"]:
    suffix = "opt" if m == "dlinear" else "tuned"
    cfg = yaml.safe_load(open(f"configs/{m}_sp500_vol_{suffix}.yaml"))
    model = build_model(cfg["model"], cfg["task"], num_features=len(cfg["task"]["feature_columns"]))
    out[m] = count_perturbed_parameters(model)
Path("outputs/coverage.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print(json.dumps({k: f"{100*v['weight_coverage']:.1f}%" for k, v in out.items()}, indent=1))
