#!/usr/bin/env python3
import pickle, numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

def load_data(aligned_pkl):
    with open(aligned_pkl,'rb') as f:
        samples = pickle.load(f)
    # features: concat technical + sentiment + retrieval
    tech = [np.asarray(s['technical_indicators'],np.float32) for s in samples]
    maxlen = max(arr.shape[0] for arr in tech)
    tech = np.vstack([np.pad(arr,(0,maxlen-arr.shape[0])) for arr in tech])
    sent = np.array([s['sentiment_features'] for s in samples],np.float32)
    retr = np.array([s['retrieval_features'] for s in samples],np.float32)
    X = np.concatenate([tech, sent, retr], axis=1)
    y = np.array([s['label'] for s in samples], np.int32)
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aligned', default='aligned_with_retrieval.pkl')
    p.add_argument('--test_size', type=float, default=0.1)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning_rate', type=float, default=0.1)
    args = p.parse_args()

    print("Loading data…")
    X, y = load_data(args.aligned)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    train_pool = Pool(X_train, y_train)
    val_pool   = Pool(X_val,   y_val)

    print("Training CatBoostClassifier…")
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=args.random_state,
        verbose=50,
        early_stopping_rounds=50
    )
    model.fit(train_pool, eval_set=val_pool)

    preds = model.predict(val_pool)
    acc = accuracy_score(y_val, preds)
    print(f"\n✅ Validation accuracy: {acc*100:.2f}%")

    model.save_model("catboost_retrieval.cbm")
    print("Model saved to catboost_retrieval.cbm")

if __name__ == '__main__':
    main()
