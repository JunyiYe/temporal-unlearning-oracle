"""Build the Nikkei 225 transport panel: data/nikkei225_returns.csv.

External-validity check requested in review: does the damage taxonomy
transport to a different market? Downloads adjusted closes for current
Nikkei 225 constituents (Wikipedia list, 2026-07; survivorship-tilted like
the S&P panel, and disclosed as such), computes simple daily returns, and
writes the same schema as data/sp500_kaggle_returns.csv
(date,symbol,return).

    python scripts/build_nikkei_panel.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CODES = """9202 9201 543A 7267 7202 7261 7211 7201 7270 7269 7203 7272 8304
8331 8354 8306 8411 8308 5831 8316 8309 7186 3407 4061 4901 4452 3405 4188
4183 4021 6988 4004 4063 4911 4005 4043 4042 4208 9433 9432 9434 9984 1721
1925 1808 1963 1812 1802 1928 1803 1801 6857 6770 7751 6902 6954 6504 6702
6501 6861 285A 6971 6920 6479 6503 6981 6701 6594 6645 6752 6723 7752 6963
7735 6724 6753 6758 6526 6976 6762 8035 6506 6841 9502 9503 9501 1332 2802
2502 2914 2801 2503 2269 2282 2871 2002 2501 9532 9531 5201 5333 5214 5233
5301 5332 8750 8725 8630 8795 8766 9147 9064 6113 6367 6361 6305 7004 7013
5631 6473 6301 6326 7011 6471 6472 6103 6302 6273 9107 9104 9101 1605 5714
5803 5801 5711 5706 3436 5802 5713 8253 8697 8591 7832 7912 7911 7951 5020
5019 4503 4519 4568 4523 4151 4578 4506 4507 4502 6146 7741 4902 7731 7733
4543 3861 9022 9020 9008 9009 9007 9001 9005 9021 8802 8801 8830 8804 3289
8267 9983 3099 3086 8252 7453 9843 7532 3382 8233 3092 5108 5101 8601 8604
6532 4751 2432 4324 6178 9766 4689 4385 2413 3659 7974 4307 4661 4755 6098
9735 3697 9602 4704 7012 5411 5406 5401 3401 3402 8001 8002 8058 8031 2768
8053 8015""".split()

START, END = "2015-01-01", "2025-12-31"
OUT = Path("data/nikkei225_returns.csv")


def main() -> None:
    tickers = [f"{c}.T" for c in CODES]
    frames = []
    batch = 40
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        for attempt in range(3):
            try:
                d = yf.download(" ".join(chunk), start=START, end=END,
                                progress=False, auto_adjust=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[RETRY {attempt}] {e}", flush=True)
                time.sleep(20)
        close = d["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(chunk[0])
        frames.append(close)
        print(f"[OK] {i + len(chunk)}/{len(tickers)}", flush=True)
        time.sleep(2)
    px = pd.concat(frames, axis=1)
    px = px.dropna(axis=1, how="all")
    ret = px.pct_change()
    long = (ret.stack(future_stack=True).rename("return").reset_index())
    long.columns = ["date", "symbol", "return"]
    long["symbol"] = long["symbol"].str.replace(".T", "", regex=False)
    long = long.dropna(subset=["return"])
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long = long.sort_values(["date", "symbol"])
    OUT.parent.mkdir(exist_ok=True)
    long.to_csv(OUT, index=False)
    n_assets = long["symbol"].nunique()
    obs = long.groupby("symbol").size()
    print(f"[DONE] {OUT}: {len(long)} rows, {n_assets} assets, "
          f"{long['date'].min()} -> {long['date'].max()}; "
          f"assets with >=1000 obs: {(obs >= 1000).sum()}")


if __name__ == "__main__":
    main()
