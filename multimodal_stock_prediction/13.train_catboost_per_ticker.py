#!/usr/bin/env python3
import pickle
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os, argparse, tqdm

def load_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def extract_features_labels(samples, ticker):
    # filter for this ticker
    filt = [s for s in samples if os.path.basename(s["image_path"]).startswith(ticker + "_")]
    tech = [np.asarray(s["technical_indicators"], np.float32) for s in filt]
    T = max(arr.shape[0] for arr in tech)
    tech = np.vstack([np.pad(arr, (0, T - arr.shape[0])) for arr in tech])
    sent = np.array([s["sentiment_features"]     for s in filt], np.float32)
    retr = np.array([s["retrieval_features"]     for s in filt], np.float32)
    X = np.concatenate([tech, sent, retr], axis=1)
    y = np.array([s["label"] for s in filt], np.int32)
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", required=True,
                   help="aligned_with_retrieval.pkl")
    p.add_argument("--out_dir", default="catboost_per_ticker")
    p.add_argument("--test_size",  type=float, default=0.1)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.aligned)

    # find all tickers from image_path filenames
    tickers = sorted({os.path.basename(s["image_path"]).split("_")[0] for s in samples})

    for ticker in tqdm.tqdm(tickers, desc="Tickers"):
        X, y = extract_features_labels(samples, ticker)
        if len(y) < 100:            # skip tiny tickers
            continue

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y
        )
        pool_tr  = Pool(X_tr, y_tr)
        pool_val = Pool(X_val, y_val)

        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=args.random_state,
            verbose=False,
            early_stopping_rounds=50
        )
        model.fit(pool_tr, eval_set=pool_val, verbose=50)

        acc = accuracy_score(y_val, model.predict(pool_val))
        fname = os.path.join(args.out_dir, f"catboost_{ticker}.cbm")
        model.save_model(fname)
        print(f" → {ticker}: val_acc={acc*100:.2f}%  saved {fname}")

if __name__ == "__main__":
    main()
