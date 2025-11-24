#!/usr/bin/env python3
import os
import time
import random
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import yfinance as yf
import mplfinance as mpf
import pandas as pd
import ta  # Technical Analysis Library

# ─── CONFIG ────────────────────────────────────────────────────────────────────

tickers = [
    # ─── S&P 100 Constituents (115 total including India examples) ─────────────
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

    # ─── India: Top 10 Large-Caps by Market Cap ───────────────────────────────
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "INFY.NS","ITC.NS","BHARTIARTL.NS","SBIN.NS","BAJFINANCE.NS",

    # ─── India: Selected NIFTY 500 Mid-Caps (5 examples) ────────────────────────
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
os.makedirs('charts', exist_ok=True)
os.makedirs('csv',    exist_ok=True)

# ─── HOW MANY WORKERS? ─────────────────────────────────────────────────────────
# Tune this to your CPU count / network capacity.
WORKERS = min(8, os.cpu_count() or 4)


# ─── INDICATOR COMPUTATION ───────────────────────────────────────────────────

def compute_indicators(df):
    df['SMA_20']  = df['Close'].rolling(20).mean()
    df['EMA_9']   = df['Close'].ewm(span=9,  adjust=False).mean()
    df['EMA_20']  = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI_14']  = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff']   = macd.macd_diff()

    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'],
        close=df['Close'], volume=df['Volume']
    ).vwap

    df['ATR_14'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'],
        close=df['Close'], window=14
    ).average_true_range()

    bb = ta.volatility.BollingerBands(
        close=df['Close'], window=20, window_dev=2
    )
    df['BB_High']  = bb.bollinger_hband()
    df['BB_Low']   = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()

    try:
        df['ADX_14'] = ta.trend.ADXIndicator(
            high=df['High'], low=df['Low'],
            close=df['Close'], window=14
        ).adx()
    except Exception:
        df['ADX_14'] = 0.0

    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    ).on_balance_volume()

    return df


# ─── FETCH, SLIDE & SAVE ───────────────────────────────────────────────────────

def fetch_and_process_data(ticker, interval, cfg):
    win, step = cfg["window"], cfg["step"]

    # 1) Pre-scan: skip if already done
    existing = glob.glob(f"csv/{ticker}_{interval}_*_{win}_ind.csv")
    try:
        period = ("7d" if interval.endswith("m")
                  else "60d" if interval.endswith("h")
                  else "7y")
        df_tmp = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=False, timeout=30
        )
        if isinstance(df_tmp.columns, pd.MultiIndex):
            df_tmp.columns = df_tmp.columns.get_level_values('Price')
        df_tmp.dropna(subset=expected_cols, inplace=True)
        total = len(df_tmp)
    except Exception:
        total = None

    if total and total >= win:
        expected = (total - win) // step + 1
        if len(existing) >= expected:
            print(f"[{ticker}@{interval}] {len(existing)}/{expected} windows exist → skip")
            return

    # 2) Download with retries
    df = None
    for attempt in range(1, 6):
        try:
            print(f"[{ticker}@{interval}] Download attempt {attempt}…")
            df = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=False, timeout=30
            )
            if df is not None and not df.empty:
                break
            print("  → Download empty, skipping interval.")
            return
        except Exception as e:
            msg = str(e)
            if "Rate limited" in msg or "429" in msg:
                wait = 2**attempt * 5
                print(f"  → Rate limited; sleeping {wait}s…")
                time.sleep(wait)
            else:
                print(f"  → Download error: {e}")
                return
    else:
        print("  → Failed after retries, skipping interval.")
        return

    # 3) Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values('Price')

    # 4) Guard & clean
    if not all(col in df.columns for col in expected_cols):
        print(f"  → Missing OHLCV for {ticker}@{interval}, skipping")
        return
    df.dropna(subset=expected_cols, inplace=True)
    total = len(df)
    if total < win:
        print(f"  → Only {total} rows (<{win}), skipping")
        return

    # 5) Slide & save
    new_cnt = skipped = 0
    for start in range(0, total - win + 1, step):
        end = start + win
        csv_f = f"csv/{ticker}_{interval}_{start}_{end}_ind.csv"
        img_f = f"charts/{ticker}_{interval}_{start}_{end}.png"
        if os.path.exists(csv_f) and os.path.exists(img_f):
            skipped += 1
            continue

        window = df.iloc[start:end]
        compute_indicators(window.copy()).to_csv(csv_f, index=False)
        mpf.plot(
            window, type='candle', style='charles',
            title=f'{ticker} {interval} [{start}:{end}]',
            ylabel='Price', figsize=(8,5),
            savefig=dict(fname=img_f, dpi=150)
        )
        new_cnt += 1

    print(f"[{ticker}@{interval}] New: {new_cnt}, Skipped: {skipped}")

    # 6) Politeness pause
    time.sleep(random.uniform(1, 3))


# ─── MAIN: PARALLEL ORCHESTRATION ──────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    tasks = [
        (t, interval, cfg)
        for t in tickers
        for interval, cfg in timeframes.items()
    ]

    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = {
            exe.submit(fetch_and_process_data, t, iv, cf): (t, iv)
            for t, iv, cf in tasks
        }
        for fut in as_completed(futures):
            t, iv = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[{t}@{iv}] ERROR: {e!r}")

    print(f"\n✅ Everything complete in {(time.time()-start)/60:.1f} minutes")
