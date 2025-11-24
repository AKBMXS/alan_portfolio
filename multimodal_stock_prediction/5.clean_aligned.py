import pickle, numpy as np

# 1) load
with open("aligned_dataset.pkl","rb") as f:
    data = pickle.load(f)

# 2) filter
cleaned = []
for sample in data:
    feats = sample["technical_indicators"]
    if np.isnan(feats).any() or np.isinf(feats).any():
        continue
    cleaned.append(sample)

print(f"Kept {len(cleaned)}/{len(data)} samples")

# 3) save
with open("aligned_dataset_clean.pkl","wb") as f:
    pickle.dump(cleaned, f)
