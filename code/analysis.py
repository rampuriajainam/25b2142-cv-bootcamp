"""Figures + numbers for ANALYSIS.md and report.md. No training; reads the run folders only.

Clean labels are used here ONLY to measure (confusion matrix, Task 2 curves, own Task 3 check, how pure the DivideMix
split is). Nothing computed here feeds back into training, filtering, relabeling or model selection.

  python analysis.py --data data --outputs outputs --figs figures

Needs: outputs/<run>/log_full.csv for every run, outputs/t3/all_signals.csv, and outputs/t1_noisy_ce/train_probs.npy +
outputs/t5_baseline_synth/train_probs.npy (for the per-noise-type memorization curve).
"""
import argparse, json, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--outputs", default="outputs")
p.add_argument("--figs", default="figures")
a = p.parse_args()
os.makedirs(a.figs, exist_ok=True)

# ---------------- style: one fixed colour per method, real = solid, synthetic = dashed ----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"CE": "#2a78d6", "GCE": "#eb6834", "DivideMix": "#1baf7a", "Filter": "#eda100", "Relabel": "#e87ba4"}
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(a.figs, name), dpi=160)
    plt.close(fig)


def log(run):
    return pd.read_csv(os.path.join(a.outputs, run, "log_full.csv"))


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.plot([x], [y], "o", ms=4, color=color)


cls = [l.strip() for l in open(os.path.join(a.data, "classes.txt"))]
lab = pd.read_csv(os.path.join(a.data, "labels.csv")).set_index("index")
tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=np.int64)
yn, yc, ys = (lab.loc[tr, c].values for c in ["noisy_label", "clean_label", "synthetic_label"])
wrong, wrong_s = yn != yc, ys != yc
num = {}

# ======================= Fig 1: noise structure, real vs synthetic =======================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"wspace": 0.32})
for ax, y, title in [(axes[0], yn, "Human labels (noisy_label)"), (axes[1], ys, "Synthetic labels (synthetic_label)")]:
    M = np.zeros((10, 10))
    np.add.at(M, (yc, y), 1)
    R = M / M.sum(1, keepdims=True) * 100
    off = R.copy(); np.fill_diagonal(off, np.nan)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES)
    im = ax.imshow(off, cmap=cmap, vmin=0, vmax=30)
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center", fontsize=7, color=INK2, style="italic")
            elif R[i, j] >= 3:
                ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if R[i, j] > 14 else INK)
    ax.set_xticks(range(10), cls, rotation=45, ha="right"); ax.set_yticks(range(10), cls)
    ax.set_xlabel("given label"); ax.grid(False)
    if ax is axes[0]:
        ax.set_ylabel("true label")
    ax.set_title(f"{title}: {(y != yc).mean() * 100:.1f}% wrong", loc="left")
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cb.set_label("% of the true class (off-diagonal)")
fig.text(0.01, -0.07, "Diagonal (italic) = % of the class whose label is correct. Cells under 3% left blank.", fontsize=8.5, color=INK2)
fig.savefig(os.path.join(a.figs, "fig1_noise_structure.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

pairs = pd.Series(list(zip(yc[wrong], yn[wrong]))).value_counts()
num["top_pairs"] = {f"{cls[i]}->{cls[j]}": int(v) for (i, j), v in pairs.head(8).items()}
num["top8_share_real"] = float(pairs.head(8).sum() / wrong.sum())
num["top8_share_synth"] = float(pd.Series(list(zip(yc[wrong_s], ys[wrong_s]))).value_counts().head(8).sum() / wrong_s.sum())
num["truck_auto_both_ways"] = float((pairs[(9, 1)] + pairs[(1, 9)]) / wrong.sum())
num["cat_dog_both_ways"] = float((pairs[(3, 5)] + pairs[(5, 3)]) / wrong.sum())

# ======================= Fig 2: Task 2 training dynamics (CE on human labels) =======================
d = log("t1_noisy_ce")
mem_start, early = 37, int(d.loc[d.val_acc_noisy.idxmax(), "epoch"])
fig, ax = plt.subplots(figsize=(8.6, 5.2))
series = [("train_acc_correct", "train acc, correctly labeled", "#2a78d6"),
          ("train_acc_mislabeled", "train acc, mislabeled (fits the wrong label)", "#eb6834"),
          ("val_acc_clean", "val acc, clean labels", "#1baf7a"),
          ("val_acc_noisy", "val acc, noisy labels", "#eda100")]
offs = {"val_acc_clean": 6, "train_acc_mislabeled": -6}
for col, name, c in series:
    ax.plot(d.epoch, d[col], color=c, label=name)
    label_end(ax, 60, d[col].iloc[-1], f"{d[col].iloc[-1]:.2f}", c, dy=offs.get(col, 0))
ax.axvspan(mem_start, 60, color="#eb6834", alpha=0.06, lw=0)
ax.axvline(early, color=INK2, lw=1, ls="--")
ax.text(early - 0.6, 1.0, f"early stop: epoch {early}\n(best noisy-val acc)", ha="right", va="top", fontsize=8.5, color=INK2)
ax.text(mem_start + 0.6, 0.06, f"memorization from epoch {mem_start}", fontsize=8.5, color="#b84a1e")
ax.set_xlim(0, 64); ax.set_ylim(0, 1.03); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
ax.set_title("Task 2 - cross-entropy on human labels: learn first, memorize later", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
save(fig, "fig2_dynamics.png")
num["t2"] = {"early_stop_epoch": early, "memorization_start_epoch": mem_start,
             "mislabeled_fit_at_36": float(d.train_acc_mislabeled[35]), "mislabeled_fit_at_60": float(d.train_acc_mislabeled.iloc[-1]),
             "val_clean_at_early": float(d.val_acc_clean[early - 1]), "val_clean_final": float(d.val_acc_clean.iloc[-1])}

# ======================= Fig 3: human noise is partly *learnable*, synthetic noise is not =======================
top8 = set(pairs.head(8).index)
sysm = np.array([(i, j) in top8 for i, j in zip(yc, yn)]) & wrong
oth = wrong & ~sysm
P = np.load(os.path.join(a.outputs, "t1_noisy_ce", "train_probs.npy"), mmap_mode="r")
Ps = np.load(os.path.join(a.outputs, "t5_baseline_synth", "train_probs.npy"), mmap_mode="r")
E = np.arange(1, len(P) + 1)
f_sys, f_oth, f_syn = [], [], []
for e in range(len(P)):
    pr, prs = P[e].argmax(1), Ps[e].argmax(1)
    f_sys.append((pr[sysm] == yn[sysm]).mean()); f_oth.append((pr[oth] == yn[oth]).mean()); f_syn.append((prs[wrong_s] == ys[wrong_s]).mean())
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for f, name, c, ls in [(f_sys, f"human, 8 most common confusions ({sysm.sum():,} imgs)", "#2a78d6", "-"),
                       (f_oth, f"human, all other mistakes ({oth.sum():,} imgs)", "#eb6834", "-"),
                       (f_syn, f"synthetic, uniform random ({wrong_s.sum():,} imgs)", "#1baf7a", "--")]:
    ax.plot(E, f, color=c, ls=ls, label=name)
    label_end(ax, 60, f[-1], f"{f[-1]:.2f}", c)
ax.axvline(early, color=INK2, lw=1, ls="--")
ax.text(early + 0.6, 0.02, f"early stop (epoch {early})", fontsize=8, color=INK2)
ax.set_xlim(0, 64); ax.set_ylim(0, 0.85); ax.set_xlabel("epoch")
ax.set_ylabel("share of mislabeled images\npredicted as their WRONG label")
ax.set_title("Cross-entropy agrees with plausible human mistakes long before memorization", loc="left")
ax.legend(loc="upper left")
save(fig, "fig3_learnable_noise.png")
num["learnable"] = {"sys_fit_ep5": float(f_sys[4]), "oth_fit_ep5": float(f_oth[4]), "syn_fit_ep5": float(f_syn[4]),
                    "sys_fit_ep35": float(f_sys[34]), "oth_fit_ep35": float(f_oth[34]), "syn_fit_ep35": float(f_syn[34]),
                    "sys_fit_ep60": float(f_sys[-1]), "oth_fit_ep60": float(f_oth[-1]), "syn_fit_ep60": float(f_syn[-1])}

# ======================= Fig 4: Task 3 noise detection =======================
S = pd.read_csv(os.path.join(a.outputs, "t3", "all_signals.csv")).set_index("index").loc[tr]
names = {"oof_avg": "out-of-fold confidence, epoch-avg (submitted)", "loss_avg": "in-sample loss, epoch-avg",
         "loss_early": "in-sample loss at early-stop epoch", "oof_final": "out-of-fold confidence, final epoch"}
cols4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
k = int(wrong.sum())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
num["t3"] = {}
for (sig, name), c in zip(names.items(), cols4):
    fpr, tpr, _ = roc_curve(wrong, S[sig])
    au = roc_auc_score(wrong, S[sig]); pk = wrong[np.argsort(-S[sig].values)[:k]].mean()
    num["t3"][sig] = {"auroc": float(au), "precision_at_k": float(pk)}
    ax1.plot(fpr, tpr, color=c, lw=2 if sig == "oof_avg" else 1.5, label=f"{name}\nAUROC {au:.3f}, P@k {pk:.3f}")
ax1.plot([0, 1], [0, 1], color=GRID, lw=1)
ax1.set_xlabel("false positive rate (correct labels flagged)"); ax1.set_ylabel("true positive rate (wrong labels caught)")
ax1.set_title("Noise-score signals (clean labels used only to measure)", loc="left", fontsize=10)
ax1.legend(loc="lower right", fontsize=7.5)
flag = np.zeros(len(tr), bool); flag[np.argsort(-S["oof_avg"].values)[:k]] = True
w = pd.DataFrame({"c": yc[wrong], "n": yn[wrong], "flag": flag[wrong]})
g = w.groupby(["c", "n"]).flag.agg(["size", "mean"]).sort_values("size", ascending=False)
top = g.head(8)
labels = [f"{cls[i]}→{cls[j]}  (n={int(r['size'])})" for (i, j), r in top.iterrows()]
rest_rec = w.flag[~w.set_index(["c", "n"]).index.isin(top.index)].mean()
vals = list(top["mean"]) + [rest_rec]
labels.append(f"all other mistakes  (n={int(len(w) - top['size'].sum())})")
yy = np.arange(len(vals))[::-1]
bars = ax2.barh(yy, vals, color=["#2a78d6"] * 8 + ["#eb6834"], height=0.68)
for b, v in zip(bars, vals):
    ax2.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=8.5, color=INK2)
ax2.set_yticks(yy, labels); ax2.set_xlim(0, 1.08); ax2.grid(axis="y", visible=False)
ax2.set_xlabel("share caught in the top-k flagged images (k = true # wrong)")
ax2.set_title("Where the submitted score fails: plausible confusions", loc="left", fontsize=10)
save(fig, "fig4_noise_detection.png")
m = w.set_index(["c", "n"]).index.isin(top.index)
fp = pd.Series(flag[~wrong]).groupby(yc[~wrong]).mean()
num["t3"]["recall_top8_pairs"] = float(w.flag[m].mean()); num["t3"]["recall_other"] = float(rest_rec)
num["t3"]["share_of_missed_in_top8"] = float((~w.flag[m]).sum() / (~w.flag).sum())
num["t3"]["false_flag_rate_by_class"] = {cls[i]: float(v) for i, v in fp.items()}

# ======================= Fig 5: memorization by method, real vs synthetic =======================
runs = {"CE": ("t1_noisy_ce", "t5_baseline_synth"), "GCE": ("t4_gce", "t5_gce_synth"),
        "DivideMix": ("t4_dividemix", "t5_dividemix_synth")}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
num["t4t5"] = {}
for mname, (rr, rs) in runs.items():
    dr, ds = log(rr), log(rs)
    ax1.plot(dr.epoch, dr.train_acc_mislabeled, color=C[mname], label=f"{mname}, human")
    ax1.plot(ds.epoch, ds.train_acc_mislabeled, color=C[mname], ls="--", label=f"{mname}, synthetic")
    ax2.plot(dr.epoch, dr.val_acc_clean, color=C[mname], label=f"{mname}, human")
    ax2.plot(ds.epoch, ds.val_acc_clean, color=C[mname], ls="--", label=f"{mname}, synthetic")
    for d_, kind in [(dr, "real"), (ds, "synth")]:
        b = d_.val_acc_noisy.idxmax()
        num["t4t5"][f"{mname}_{kind}"] = {"best_epoch": int(d_.epoch[b]), "sel_val_best": float(d_.val_acc_noisy[b]),
                                          "sel_val_final": float(d_.val_acc_noisy.iloc[-1]),
                                          "clean_val_best": float(d_.val_acc_clean[b]), "clean_val_final": float(d_.val_acc_clean.iloc[-1]),
                                          "mislabeled_fit_final": float(d_.train_acc_mislabeled.iloc[-1])}
ref = log("t1_clean_ref").val_acc_clean.iloc[-1]
ax2.axhline(ref, color=INK2, lw=1, ls=":")
ax2.text(1, ref + 0.006, f"clean-label reference {ref:.3f}", fontsize=8, color=INK2)
ax1.set_title("Fit to wrong training labels", loc="left"); ax1.set_ylabel("share of mislabeled images fitted")
ax2.set_title("Validation accuracy on clean labels (measurement only)", loc="left"); ax2.set_ylabel("accuracy")
ax2.set_ylim(0.3, 1.0)
for ax in (ax1, ax2):
    ax.set_xlabel("epoch"); ax.set_xlim(0, 61)
ax1.legend(loc="upper left", ncol=1)
save(fig, "fig5_methods_real_vs_synth.png")
for mname, rr in [("Filter", "t4_filter"), ("Relabel", "t4_relabel")]:
    d_ = log(rr); b = d_.val_acc_noisy.idxmax()
    num["t4t5"][f"{mname}_real"] = {"best_epoch": int(d_.epoch[b]), "sel_val_best": float(d_.val_acc_noisy[b]),
                                    "sel_val_final": float(d_.val_acc_noisy.iloc[-1]),
                                    "clean_val_best": float(d_.val_acc_clean[b]), "clean_val_final": float(d_.val_acc_clean.iloc[-1])}
num["clean_ref_clean_val"] = float(ref)

# ======================= Fig 6: how well DivideMix separates clean from noisy =======================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 3.9))
for run, ls, kind in [("t4_dividemix", "-", "human"), ("t5_dividemix_synth", "--", "synthetic")]:
    d_ = log(run)
    d_ = d_[d_.phase == "dividemix"]
    for ax, col, title in [(ax1, "labeled_frac", "share of images kept as 'clean'"),
                           (ax2, "labeled_purity", "purity of the 'clean' set"),
                           (ax3, "noisy_recall", "share of wrong labels set aside")]:
        v = (d_[f"net0_{col}"] + d_[f"net1_{col}"]) / 2
        ax.plot(d_.epoch, v, color=C["DivideMix"], ls=ls, label=kind)
        label_end(ax, d_.epoch.iloc[-1], v.iloc[-1], f"{v.iloc[-1]:.2f}", C["DivideMix"])
        ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("epoch"); ax.set_xlim(10, 66)
        num.setdefault("dividemix_split", {})[f"{kind}_{col}_final"] = float(v.iloc[-1])
ax1.axhline(1 - wrong.mean(), color=INK2, lw=1, ls=":"); ax1.text(11, 1 - wrong.mean() + 0.005, "true clean share 0.60", fontsize=8, color=INK2)
ax1.legend(loc="lower right")
fig.suptitle("DivideMix co-divide (average of the two networks; clean labels used only to measure)", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
save(fig, "fig6_dividemix_split.png")

json.dump(num, open(os.path.join(a.figs, "analysis_numbers.json"), "w"), indent=2)
print(json.dumps(num, indent=2))
