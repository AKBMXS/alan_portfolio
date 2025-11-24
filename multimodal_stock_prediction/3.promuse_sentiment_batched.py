#!/usr/bin/env python3
# promuse_sentiment_batched.py
# macOS M1 fixes: spawn, disable tokenizer parallelism,
# force PyTorch, enable MPS fallback, import torch.

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
# 1) Enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# 2) Disable Rust-parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import re
import argparse
import csv
import requests
import pandas as pd
import feedparser
import torch
from transformers import pipeline

def clean_text(txt):
    txt = re.sub(r'http\S+|www\.\S+', '', txt)
    txt = re.sub(r'\$[A-Za-z]+', '', txt)
    txt = re.sub(r'[^\x00-\x7F]+', '', txt)
    return " ".join(txt.split())

def scrape_yahoo_news(ticker, max_articles=50):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en"
    feed = feedparser.parse(url)
    out = []
    for entry in feed.entries[:max_articles]:
        title = entry.get("title","").strip()
        date  = entry.get("published","").strip()
        if len(title.split()) > 4:
            out.append({"source":"yahoo","text":title,"date":date})
    return out

def fetch_stocktwits_messages(ticker, max_messages=500):
    msgs, max_id = [], None
    headers = {"User-Agent":"Mozilla/5.0"}
    while len(msgs) < max_messages:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        if max_id:
            url += f"?max={max_id}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            break
        batch = r.json().get("messages", [])
        if not batch:
            break
        msgs.extend(batch)
        max_id = batch[-1]["id"] - 1
    out = []
    for m in msgs:
        body = m.get("body","").strip()
        date_str = m.get("created_at","").strip()
        if body:
            out.append({"source":"stocktwits","text":body,"date":date_str})
    return out

def scrape_google_rss(ticker, days=7, max_articles=50):
    q = requests.utils.quote(ticker)
    url = f"https://news.google.com/rss/search?q={q}%20when:{days}d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    out = []
    for entry in feed.entries[:max_articles]:
        title = entry.get("title","").strip()
        date  = entry.get("published","").strip()
        if len(title.split()) > 4:
            out.append({"source":"google","text":title,"date":date})
    return out

def analyze_sentiments(records, model, batch_size=32, min_len=10):
    valid, skipped, texts, metas = [], [], [], []
    for rec in records:
        txt = rec["text"].strip()
        if len(txt) < min_len:
            skipped.append(f"{rec['source']} too short")
            continue
        clean = clean_text(txt)[:512]
        if len(clean) < min_len:
            skipped.append(f"{rec['source']} cleaned too short")
            continue
        texts.append(clean)
        metas.append((rec['source'], rec.get('date',''), clean))

    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        res = model(
            batch,
            candidate_labels=["positive","negative","neutral"],
            multi_label=False
        )
        if isinstance(res, dict):
            outputs.append(res)
        else:
            outputs.extend(res)

    for (src, date, txt), out in zip(metas, outputs):
        valid.append({
            "source":           src,
            "date":             date,
            "text":             txt,
            "sentiment_label":  out["labels"][0],
            "sentiment_score":  round(out["scores"][0],4)
        })

    if skipped:
        print("⚠️ Skipped:", skipped)
    return valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers",   nargs="+", required=True)
    parser.add_argument("--max_yahoo", type=int, default=50)
    parser.add_argument("--max_st",    type=int, default=500)
    parser.add_argument("--rss_days",  type=int, default=7)
    parser.add_argument("--min_len",   type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--output_dir",default="sentiment_all_promuse")
    parser.add_argument("--model_name",default="facebook/bart-large-mnli")
    parser.add_argument("--device",    type=int, default=-1,
                        help="-1 for CPU or GPU ID")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Device set to", "cpu" if args.device<0 else f"cuda:{args.device}")

    # Force PyTorch-only zero-shot pipeline (no TF)
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model_name,
        tokenizer=args.model_name,
        framework="pt",
        device=args.device
    )

    for T in args.tickers:
        print(f"\n▶ Processing {T}")
        recs = (
            scrape_yahoo_news(T, args.max_yahoo) +
            fetch_stocktwits_messages(T, args.max_st) +
            scrape_google_rss(T, args.rss_days, args.max_yahoo)
        )
        if not recs:
            print(f"⚠️ No data for {T}")
            continue

        print(f"🔍 analyzing {len(recs)} records…")
        df = pd.DataFrame(analyze_sentiments(
            recs, classifier, args.batch_size, args.min_len
        ))

        df["is_relevant"] = False
        df.loc[df.source.isin(["yahoo","google"]), "is_relevant"] = True
        pattern = rf"\b(?:{T}|{T.lower()}|{T.upper()}|{T.capitalize()})\b"
        mask = df.source.eq("stocktwits") & df.text.str.contains(pattern, case=False)
        df.loc[mask, "is_relevant"] = True

        out_path = os.path.join(args.output_dir, f"{T}_sentiment.csv")
        print(f"→ Writing {out_path}")
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"✅ {T}: {len(df)} rows → {out_path}")

if __name__ == "__main__":
    main()
