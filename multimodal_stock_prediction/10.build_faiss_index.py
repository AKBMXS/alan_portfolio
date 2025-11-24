#!/usr/bin/env python3
import os, numpy as np, faiss

EMBED_DIR   = "fnspid_shards_full"
INDEX_PATH  = "faiss_index/stock_news.idx"
META_PATH   = "faiss_index/stock_news_meta.npz"
DIM         = 384
M_PARAM     = 32
EF_CONSTR   = 500

def main():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    index = faiss.IndexHNSWFlat(DIM, M_PARAM)
    index.hnsw.efConstruction = EF_CONSTR

    all_syms, all_dates = [], []
    for fn in sorted(os.listdir(EMBED_DIR)):
        if not fn.endswith("_embeddings.npz"):
            continue
        data = np.load(os.path.join(EMBED_DIR, fn))
        emb   = data["embeddings"].astype("float32")
        syms  = data["symbol"].tolist()
        dates = data["date"].tolist()
        index.add(emb)
        all_syms.extend(syms)
        all_dates.extend(dates)
        print(f"Added {emb.shape[0]} vectors from {fn}")

    # save index & metadata
    faiss.write_index(index, INDEX_PATH)
    np.savez_compressed(META_PATH,
                        symbol=np.array(all_syms, dtype="U32"),
                        date =np.array(all_dates, dtype="U32"))
    print(f"Index saved → {INDEX_PATH}")
    print(f"Meta saved  → {META_PATH}")

if __name__=="__main__":
    main()
