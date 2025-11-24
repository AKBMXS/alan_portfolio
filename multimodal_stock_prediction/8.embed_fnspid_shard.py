#!/usr/bin/env python3
# Embed one Parquet shard of headlines, on GPU if available,
# suppress TF/XLA warnings.

import os
# 1) Suppress TensorFlow/XLA noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_args():
    p = argparse.ArgumentParser(
        description="Embed headlines in one FNSPID Parquet shard"
    )
    p.add_argument("-i", "--input",     required=True,
                   help="Input Parquet shard (e.g. fnspid_0000.parquet)")
    p.add_argument("-o", "--output",    required=True,
                   help="Output .npz path (e.g. fnspid_0000_embeddings.npz)")
    p.add_argument("-m", "--model",     default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model name")
    p.add_argument("-b", "--batch_size", type=int, default=1024,
                   help="Number of titles per embedding batch")
    return p.parse_args()

def main():
    args = parse_args()

    # Load only the columns we need
    df = pd.read_parquet(args.input, columns=["symbol","date","title"])
    titles  = df["title"].fillna("").astype(str).tolist()
    symbols = df["symbol"].tolist()
    dates   = df["date"].tolist()

    # Pick device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"→ Using device: {device}")

    # Initialize model
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = 512

    # Embed in batches
    embeddings = []
    for start in tqdm(range(0, len(titles), args.batch_size),
                      desc=os.path.basename(args.input)):
        batch = titles[start : start + args.batch_size]
        emb   = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    # Save embeddings + metadata
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        symbol=np.array(symbols, dtype="U32"),
        date=np.array(dates,   dtype="U32")
    )
    print(f"✅ Saved {args.output} ({embeddings.shape[0]}×{embeddings.shape[1]})")

if __name__ == "__main__":
    main()