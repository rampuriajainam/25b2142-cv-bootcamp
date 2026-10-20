"""Task 3: split train_idx into K folds (stratified by the training label). Writes, for each fold k,
folds/fit_k.npy  = training indices EXCLUDING fold k  (pass to train.py --fit_idx)
folds/hold_k.npy = fold k (scored out-of-fold)

  python make_folds.py --data data --out folds --label_col noisy_label
"""
import argparse, os
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--out", default="folds")
p.add_argument("--label_col", default="noisy_label")
p.add_argument("--k", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(os.path.join(a.data, "labels.csv")).set_index("index")
tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=np.int64)
y = df.loc[tr, a.label_col].values
rng = np.random.default_rng(a.seed)
fold = np.empty(len(tr), np.int64)
for c in np.unique(y):
    pos = rng.permutation(np.where(y == c)[0])
    fold[pos] = np.arange(len(pos)) % a.k
for k in range(a.k):
    np.save(os.path.join(a.out, f"hold_{k}.npy"), np.sort(tr[fold == k]))
    np.save(os.path.join(a.out, f"fit_{k}.npy"), np.sort(tr[fold != k]))
    print(f"fold {k}: hold {np.sum(fold == k)}  fit {np.sum(fold != k)}")
