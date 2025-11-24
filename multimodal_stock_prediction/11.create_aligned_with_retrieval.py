#!/usr/bin/env python3
"""
create_aligned_with_retrieval.py (Optimized)

1) Loads cleaned aligned samples (image + tech + label).
2) Precomputes top-K FAISS retrieval + FinBERT sentiment per ticker.
3) Broadcasts retrieval_features to each sample.
4) Saves the fully-augmented dataset.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import faiss
from transformers import pipeline
from collections import defaultdict

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
CLEAN_PICKLE  = "aligned_dataset_clean.pkl"
OUTPUT_PICKLE = "aligned_with_retrieval.pkl"
EMBED_DIR      = "fnspid_shards_full"
PARQUET_DIR    = "fnspid_shards_full"
TOP_K          = 10
FINBERT_MODEL  = "ProsusAI/finbert"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def load_faiss_and_metadata(embed_dir, parquet_dir):
    """
    Loads every .npz shard (embeddings + symbols + dates) and the matching
    .parquet shards (titles), returns:
      - all_embeddings: (N, D) float32
      - symbols:        list[str], length N
      - dates:          list[str], length N
      - titles:         list[str], length N
    """
    all_embs = []
    symbols  = []
    dates    = []
    titles   = []

    for fn in sorted(os.listdir(embed_dir)):
        if not fn.endswith(".npz"):
            continue

        # handle embeddings file
        if fn.endswith("_embeddings.npz"):
            shard_base = fn[:-len("_embeddings.npz")]
        else:
            shard_base = fn[:-4]

        data = np.load(os.path.join(embed_dir, fn))
        emb  = data["embeddings"].astype("float32")
        sy   = data["symbol"].tolist()
        dt   = data["date"].tolist()

        # load titles from parquet
        pq_path = os.path.join(parquet_dir, f"{shard_base}.parquet")
        df = pd.read_parquet(pq_path, columns=["title"])
        shard_titles = df["title"].fillna("").astype(str).tolist()

        if len(shard_titles) != emb.shape[0]:
            raise ValueError(
                f"Mismatch for {shard_base}: "
                f"{len(shard_titles)} titles vs {emb.shape[0]} embeddings"
            )

        all_embs.append(emb)
        symbols.extend(sy)
        dates.extend(dt)
        titles.extend(shard_titles)

    all_embeddings = np.vstack(all_embs)
    return all_embeddings, symbols, dates, titles

def fast_compute_retrieval_features(samples, embeddings, symbols, titles, k):
    """
    1) Groups embedding indices by symbol.
    2) For each symbol:
       - Builds a FAISS HNSW index once,
       - Retrieves top-k titles,
       - Runs FinBERT as a single batch,
       - Aggregates pos/neg/mean_score.
    3) Assigns those features to every sample of that symbol.
    """
    # 1) Group indices by ticker symbol
    sym_to_idxs = defaultdict(list)
    for idx, sym in enumerate(symbols):
        sym_to_idxs[sym].append(idx)

    # 2) Prepare batched FinBERT
    finbert = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        device=0 if DEVICE=="cuda" else -1,
        batch_size=32
    )

    features_by_symbol = {}
    for sym, idxs in tqdm(sym_to_idxs.items(), desc="Precomputing per‐ticker"):
        sub_embs   = embeddings[idxs]
        sub_titles = [titles[i] for i in idxs]

        # Build & query FAISS sub‐index
        sub_idx = faiss.IndexHNSWFlat(sub_embs.shape[1], 32)
        sub_idx.hnsw.efConstruction = 200
        sub_idx.add(sub_embs)
        D, I = sub_idx.search(sub_embs[:1], k)

        top_texts = [sub_titles[i] for i in I[0]]
        if top_texts:
            outputs = finbert(top_texts)
            pos = sum(1 for out in outputs if out["label"].lower()=="positive")
            neg = sum(1 for out in outputs if out["label"].lower()=="negative")
            mean_scr = float(sum(out["score"] for out in outputs) / len(outputs))
        else:
            pos = neg = 0
            mean_scr = 0.0

        features_by_symbol[sym] = [pos, neg, mean_scr]

    # 3) Broadcast to all samples
    for rec in tqdm(samples, desc="Assigning to samples"):
        sym = os.path.basename(rec["image_path"]).split("_")[0]
        rec["retrieval_features"] = features_by_symbol.get(sym, [0, 0, 0.0])

    return samples

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_pickle", default=CLEAN_PICKLE)
    parser.add_argument("--output_pickle", default=OUTPUT_PICKLE)
    parser.add_argument("--embed_dir",     default=EMBED_DIR)
    parser.add_argument("--parquet_dir",   default=PARQUET_DIR)
    parser.add_argument("--top_k",         type=int, default=TOP_K)
    args = parser.parse_args()

    print("⏳ Loading cleaned aligned samples…")
    with open(args.clean_pickle, "rb") as f:
        samples = pickle.load(f)

    print("⏳ Loading embeddings + metadata…")
    embeddings, symbols, dates, titles = load_faiss_and_metadata(
        args.embed_dir, args.parquet_dir
    )

    # Ensure placeholder for original scraped sentiment
    for rec in samples:
        rec.setdefault("sentiment_features", [0.0, 0.0, 0.0, 0.0])

    print(f"⏳ Computing retrieval features (top-{args.top_k})…")
    samples = fast_compute_retrieval_features(
        samples, embeddings, symbols, titles, args.top_k
    )

    print(f"✅ Saving merged dataset to {args.output_pickle} …")
    with open(args.output_pickle, "wb") as f:
        pickle.dump(samples, f)

    print(f"✅ Done: wrote {len(samples)} samples → {args.output_pickle}")

if __name__ == "__main__":
    main()
