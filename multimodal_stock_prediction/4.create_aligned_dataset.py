#!/usr/bin/env python3
import os
import pickle
import pandas as pd

aligned_data = []

# ─── Directory paths ───────────────────────────────────────────────────────────
chart_dir     = "charts"
csv_dir       = "csv"        # your indicator CSVs live here
full_data_dir = "full_data"  # raw OHLCV history

# ─── Helper: parse chart filenames ──────────────────────────────────────────────
def parse_filename(fname):
    """
    Given e.g. "AAPL_1d_220_320.png", return
       ticker="AAPL", tf="1d", start=220, end=320
    """
    base = fname.rsplit(".", 1)[0]
    ticker, tf, start, end = base.split("_")
    return ticker, tf, int(start), int(end)

# ─── Main loop: for each chart window ──────────────────────────────────────────
for filename in os.listdir(chart_dir):
    if not filename.endswith(".png"):
        continue

    ticker, timeframe, start_idx, end_idx = parse_filename(filename)
    image_path = os.path.join(chart_dir, filename)

    # the indicator CSV naming convention in your pipeline was:
    #   csv/<TICKER>_<tf>_<start>_<end>_ind.csv
    csv_fname = f"{ticker}_{timeframe}_{start_idx}_{end_idx}_ind.csv"
    csv_path  = os.path.join(csv_dir, csv_fname)
    if not os.path.exists(csv_path):
        # skip if indicators missing
        continue

    # load and immediately pick *only* the numeric columns
    df = pd.read_csv(csv_path)
    # Select dtypes float/int and drop any leftover NaN/Inf
    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        continue

    # grab the *last* row of just those numeric columns
    last_row = numeric.iloc[-1]
    tech_feats = last_row.values.astype(float).tolist()

    # now determine your binary up/down label from the full price history
    full_csv = os.path.join(full_data_dir, f"{ticker}_{timeframe}.csv")
    if not os.path.exists(full_csv):
        continue
    full_df = pd.read_csv(full_csv).dropna(subset=["Close"])
    # guard against running off the end
    if end_idx + 1 >= len(full_df):
        continue

    last_close = full_df.iloc[end_idx]["Close"]
    next_close = full_df.iloc[end_idx + 1]["Close"]
    label = int(next_close > last_close)

    # sanity‐check print
    idx = len(aligned_data)
    if idx % 50 == 0:
        print(f"\nSample {idx}:")
        print(" Chart file:", filename)
        print(" Window   :", start_idx, "-", end_idx)
        print(" LastClose:", last_close)
        print(" NextClose:", next_close)
        print(" Label    :", label)

    aligned_data.append({
        "image_path":           image_path,
        "technical_indicators": tech_feats,
        "label":                label
    })

# ─── Save everything ────────────────────────────────────────────────────────────
with open("aligned_dataset.pkl", "wb") as f:
    pickle.dump(aligned_data, f)

print(f"\n✅ Aligned dataset created with {len(aligned_data)} samples.")
