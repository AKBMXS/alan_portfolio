#!/usr/bin/env python3
"""
full_data.py

Download raw OHLCV CSVs for each ticker×interval, using the same
pre-scan + retry logic as dataset_generation.py.
"""

import os
import time
import random
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import yfinance as yf
import pandas as pd

# ─── CONFIG (copied from dataset_generation.py) ─────────────────────────────────

tickers = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK.B","C",
    "CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS",
    "CVX","DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD",
    "GM","GOOG","GOOGL","GS","HD","HON","IBM","INTU","ISRG","JNJ",
    "JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT",
    "MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE",
    "NOW","NVDA","OXY","PEP","PFE","PG","PLTR","PM","PYPL","QCOM",
    "RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM",
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "INFY.NS","ITC.NS","BHARTIARTL.NS","SBIN.NS","BAJFINANCE.NS",
    "ABB.NS","ACC.NS","APLAPOLLO.NS","DABUR.NS","CUMMINSIND.NS",
]

timeframes = {
    "5m":  {"window": 96,  "step":  8},
    "15m": {"window": 64,  "step":  4},
    "30m": {"window": 48,  "step":  4},
    "1h":  {"window": 32,  "step":  2},
    "4h":  {"window": 20,  "step":  1},
    "1d":  {"window": 100, "step":  5},
    "1wk": {"window": 100, "step":  5},
    "1mo": {"window": 20,  "step":  1},
}

expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

output_dir = "full_data"
os.makedirs(output_dir, exist_ok=True)

WORKERS = min(8, os.cpu_count() or 4)


# ─── FETCH & SAVE (with pre-scan logic) ─────────────────────────────────────────

def fetch_and_save_full_data(ticker: str, interval: str, cfg: dict):
    win  = cfg["window"]
    step = cfg["step"]

    # 1) Pre-scan: count any existing CSVs to decide how far back we ask
    pattern = f"{output_dir}/{ticker}_{interval}_*.csv"
    existing = glob.glob(pattern)
    try:
        # same period logic as in your old code
        period = ("7d"  if interval.endswith("m")
                  else "60d" if interval.endswith("h")
                  else "7y")
        df_tmp = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=False, timeout=30
        )
        # flatten MultiIndex if needed
        if isinstance(df_tmp.columns, pd.MultiIndex):
            df_tmp.columns = df_tmp.columns.get_level_values('Price')
        df_tmp.dropna(subset=expected_cols, inplace=True)
        total = len(df_tmp)
    except Exception:
        total = None

    # if we got enough rows to cover one window, compute how many windows we'd get
    if total and total >= win:
        expected_windows = (total - win)//step + 1
        print(f"[{ticker}@{interval}] scanned {total} rows → {len(existing)}/{expected_windows} windows exist")
    else:
        print(f"[{ticker}@{interval}] only {total or 0} rows (<{win}); skipping download")
        return

    # 2) Now fetch full history (we’ll just re-use that df_tmp if it covers full period)
    #    If df_tmp is None or too small, we still try once here.
    try:
        df = df_tmp if (df_tmp is not None and not df_tmp.empty) else yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=False, timeout=30
        )
    except Exception as e:
        print(f"[{ticker}@{interval}] download failed: {e}")
        return

    # 3) Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values('Price')

    # 4) Validate & clean
    if not all(c in df.columns for c in expected_cols):
        print(f"[{ticker}@{interval}] missing cols {set(expected_cols)-set(df.columns)}; skipping")
        return
    df.dropna(subset=expected_cols, inplace=True)

    # 5) Save CSV (overwrite any existing)
    out_csv = os.path.join(output_dir, f"{ticker}_{interval}.csv")
    df.index.name = 'Date'
    df.reset_index().to_csv(out_csv, index=False)
    print(f"[{ticker}@{interval}] ✅ saved {len(df)} rows → {out_csv}")

    # 6) Polite pause
    time.sleep(random.uniform(1, 3))


# ─── MAIN: PARALLEL ORCHESTRATION ──────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    tasks = [
        (t, iv, cfg)
        for t in tickers
        for iv, cfg in timeframes.items()
    ]

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(fetch_and_save_full_data, t, iv, cfg): (t, iv)
            for t, iv, cfg in tasks
        }
        for fut in as_completed(futures):
            t, iv = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[{t}@{iv}] ERROR: {e!r}")

    elapsed_min = (time.time()-start)/60
    print(f"\n✅ All done in {elapsed_min:.1f} minutes")
