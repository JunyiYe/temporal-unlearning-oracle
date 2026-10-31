from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic return panel for smoke tests.")
    parser.add_argument("--output", default="data/synthetic_returns.csv")
    parser.add_argument("--assets", type=int, default=40)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range(args.start, args.end)
    market = simulate_garch_like_returns(len(dates), rng, base_vol=0.008)
    rows: list[dict[str, object]] = []

    for asset_idx in range(args.assets):
        beta = rng.normal(1.0, 0.25)
        alpha = rng.normal(0.0, 0.0002)
        idio = simulate_garch_like_returns(len(dates), rng, base_vol=rng.uniform(0.006, 0.015))
        momentum = pd.Series(market).rolling(5, min_periods=1).mean().shift(1).fillna(0.0).to_numpy()
        returns = alpha + beta * market + 0.15 * momentum + idio
        symbol = f"S{asset_idx:03d}"
        rows.extend(
            {"date": date.date().isoformat(), "symbol": symbol, "return": float(ret)}
            for date, ret in zip(dates, returns, strict=True)
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {out} rows={len(rows)} assets={args.assets}")


def simulate_garch_like_returns(n: int, rng: np.random.Generator, base_vol: float) -> np.ndarray:
    returns = np.zeros(n, dtype=np.float64)
    variance = base_vol**2
    for t in range(n):
        shock = rng.standard_t(df=5)
        returns[t] = np.clip(np.sqrt(variance) * shock, -0.2, 0.2)
        variance = min(0.05**2, 0.000002 + 0.08 * returns[t] ** 2 + 0.9 * variance)
    return returns


if __name__ == "__main__":
    main()
