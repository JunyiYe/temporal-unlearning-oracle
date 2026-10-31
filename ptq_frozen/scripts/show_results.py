import sys
import json

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

methods = d["methods"]
print(f"{'Method':<30} {'MSE':<12} {'MAE':<12} {'IC':<10} {'Rank IC':<10} {'Sharpe':<10}")
print("=" * 90)
for name in sorted(methods.keys()):
    o = methods[name]["overall"]
    print(f"{name:<30} {o['mse']:.6f}  {o['mae']:.6f}  {o['ic']:+.6f}  {o['rank_ic']:+.6f}  {o['sharpe']:+.4f}")

print()
bl = d["baselines"]
print(f"{'Baselines':<30} {'MSE':<12} {'MAE':<12} {'IC':<10} {'Rank IC':<10} {'Sharpe':<10}")
print("=" * 90)
for name in ["train_mean", "zero"]:
    o = bl[name]["overall"]
    print(f"{name:<30} {o['mse']:.6f}  {o['mae']:.6f}  {o['ic']:+.6f}  {o['rank_ic']:+.6f}  {o['sharpe']:+.4f}")

print()
print(f"Training best_val_loss: {d['train']['best_val_loss']:.6f}")
print(f"Test MSE (fp32): {methods['fp32']['overall']['mse']:.6f}")
print(f"Test MSE (baseline): {bl['train_mean']['overall']['mse']:.6f}")
