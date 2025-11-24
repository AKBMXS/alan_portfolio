#!/usr/bin/env python
import os
import pickle
import pandas as pd

def parse_filename(fn):
    # expects: TICKER_TIMEFRAME_window_START_END.png
    parts     = fn.split("_")
    ticker    = parts[0]
    timeframe = parts[1]
    start_idx = int(parts[3])
    end_idx   = int(parts[4].split(".")[0])
    return ticker, timeframe, start_idx, end_idx

# 1) load your cleaned aligned data
with open("aligned_dataset_clean.pkl","rb") as f:
    aligned = pickle.load(f)

enhanced = []
for rec in aligned:
    fn = os.path.basename(rec["image_path"])
    T, tf, start_idx, end_idx = parse_filename(fn)

    # 2) re-load the exact window CSV to get its date index
    csv_fname = f"{T}_{tf}_window_{start_idx}_{end_idx}_indicators.csv"
    csv_path  = os.path.join("csv", csv_fname)
    if not os.path.exists(csv_path):
        rec["sentiment_features"] = [0,0,0,0.0]
        enhanced.append(rec)
        continue

    window = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    start_date, end_date = window.index[0], window.index[-1]

    # 3) load that ticker’s sentiment file
    sent_path = os.path.join("sentiment_all_promuse", f"{T}_sentiment.csv")
    if not os.path.exists(sent_path):
        feats = [0,0,0,0.0]
    else:
        sent = pd.read_csv(sent_path)
        # parse and then **drop any tz** so it’s comparable to window.index
        sent['date'] = (
            pd.to_datetime(sent['date'], errors='coerce', utc=True)
              .dt.tz_convert(None)
        )
        sent = sent.dropna(subset=['date'])
        sent = sent[sent.is_relevant]

        # now filter to the same date window
        mask = (sent.date >= start_date) & (sent.date <= end_date)
        sel  = sent.loc[mask]

        # build your four features: counts + mean score
        pos = int((sel.sentiment_label=="positive").sum())
        neg = int((sel.sentiment_label=="negative").sum())
        neu = int((sel.sentiment_label=="neutral").sum())
        avg = float(sel.sentiment_score.mean()) if not sel.empty else 0.0
        feats = [pos, neg, neu, avg]

    rec["sentiment_features"] = feats
    enhanced.append(rec)

# 4) write out a new pickle with your added sentiment
with open("aligned_dataset_with_sent_promuse.pkl","wb") as f:
    pickle.dump(enhanced, f)

print(f"✅ Saved {len(enhanced)} samples → aligned_dataset_with_sent_promuse.pkl")
