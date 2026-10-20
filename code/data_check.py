"""Task 1: noise rate + 10x10 confusion matrix (clean rows -> noisy cols) on the 45k training images.
Usage: python data_check.py --data data --out outputs/data_check
"""
import argparse, json, os
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--out", default="outputs/data_check")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

classes = [l.strip() for l in open(os.path.join(a.data, "classes.txt")) if l.strip()]
df = pd.read_csv(os.path.join(a.data, "labels.csv")).set_index("index")
tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=int)
d = df.loc[tr]

res = {
    "noise_rate_noisy": float((d.noisy_label != d.clean_label).mean()),
    "noise_rate_synthetic": float((d.synthetic_label != d.clean_label).mean()),
}
for col in ["noisy_label", "synthetic_label"]:
    cm = pd.crosstab(d.clean_label, d[col]).reindex(index=range(10), columns=range(10), fill_value=0)
    cm.index = [f"clean_{c}" for c in classes]; cm.columns = [f"{col}_{c}" for c in classes]
    cm.to_csv(os.path.join(a.out, f"confusion_{col}.csv"))
    off = cm.values.copy(); np.fill_diagonal(off, 0)
    flips = sorted(((int(off[i, j]), classes[i], classes[j]) for i in range(10) for j in range(10)), reverse=True)
    res[f"top_flips_{col}"] = [f"{c}->{n}: {k}" for k, c, n in flips[:8]]
    res[f"top8_share_of_noise_{col}"] = float(sum(k for k, _, _ in flips[:8]) / off.sum())
    res[f"per_class_noise_{col}"] = {classes[i]: float(1 - cm.values[i, i] / cm.values[i].sum()) for i in range(10)}

json.dump(res, open(os.path.join(a.out, "data_check.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

try:  # optional heatmap for the report
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cm = pd.crosstab(d.clean_label, d.noisy_label).reindex(index=range(10), columns=range(10), fill_value=0).values
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(10), classes, rotation=45, ha="right"); ax.set_yticks(range(10), classes)
    ax.set_xlabel("noisy label"); ax.set_ylabel("clean label"); ax.set_title("P(noisy | clean), train split")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(a.out, "confusion_noisy.png"), dpi=150)
except ImportError:
    pass
