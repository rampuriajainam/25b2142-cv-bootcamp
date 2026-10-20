"""Task 3: per-image noise scores for all 45,000 training images, plus Task 4 inputs.

Signals (higher = label more likely wrong), all from models trained on the given label column only:
  oof_final  1 - p(label) out-of-fold, final epoch of each fold model
  oof_avg    1 - p(label) out-of-fold, averaged over all epochs           <- submitted (fixed in advance)
  oof_disagree  1 if out-of-fold argmax != label (ties broken by oof_avg)
  loss_early 1 - p(label) in-sample, full-data run at --early_epoch (before memorization)
  loss_avg   1 - p(label) in-sample, full-data run averaged over epochs (AUM-like)
  combo      mean rank of oof_avg and loss_avg

Task 4 inputs (no clean labels used): estimated noisy count n_est = #(oof argmax != label);
  t4_filter_fit.npy   train indices minus the n_est highest-scored images  (train.py --fit_idx)
  t4_relabel.npy      length-50000 labels; the n_est highest-scored images get the oof argmax (train.py --relabel)

--check prints AUROC / precision@k against clean labels: a MEASUREMENT only (allowed "own Task 3 check");
it never changes what is written.

  python score_noise.py --data data --folds folds --fold_runs outputs/t3_fold{k} --full_run outputs/t1_noisy_ce \
      --early_epoch 15 --out outputs/t3 --pred predictions/t3_noise_scores.csv --check
"""
import argparse, json, os
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--folds", default="folds")
p.add_argument("--fold_runs", default="outputs/t3_fold{k}")
p.add_argument("--k", type=int, default=5)
p.add_argument("--full_run", default="outputs/t1_noisy_ce", help="full-data run on the same label column")
p.add_argument("--early_epoch", type=int, required=True, help="Task 2 early-stop epoch (chosen with noisy val only)")
p.add_argument("--label_col", default="noisy_label")
p.add_argument("--signal", default="oof_avg")
p.add_argument("--out", default="outputs/t3")
p.add_argument("--pred", default="predictions/t3_noise_scores.csv")
p.add_argument("--check", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(os.path.dirname(a.pred) or ".", exist_ok=True)

df = pd.read_csv(os.path.join(a.data, "labels.csv")).set_index("index")
tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=np.int64)   # row order of train_probs.npy
pos_of = {int(i): n for n, i in enumerate(tr)}
y = df.loc[tr, a.label_col].values
N = len(tr); ar = np.arange(N)

# ---- out-of-fold probabilities ----
oof_final = np.full((N, 10), np.nan, np.float32)
oof_avg = np.full((N, 10), np.nan, np.float32)
for k in range(a.k):
    hold = np.load(os.path.join(a.folds, f"hold_{k}.npy"))
    pos = np.array([pos_of[int(i)] for i in hold])
    run = a.fold_runs.format(k=k)
    args = json.load(open(os.path.join(run, "args.json")))
    assert args["fit_idx"] and os.path.basename(args["fit_idx"]) == f"fit_{k}.npy", f"{run} was not trained on fit_{k}.npy"
    assert args["label_col"] == a.label_col, f"{run} trained on {args['label_col']}"
    fit = np.load(os.path.join(a.folds, f"fit_{k}.npy"))
    assert len(np.intersect1d(fit, hold)) == 0, f"fold {k}: held-out images were trained on"
    P = np.load(os.path.join(run, "train_probs.npy"), mmap_mode="r")      # E, N, 10
    oof_final[pos] = P[-1, pos].astype(np.float32)
    oof_avg[pos] = P[:, pos].astype(np.float32).mean(0)
assert not np.isnan(oof_avg).any(), "some images have no out-of-fold prediction"

Pf = np.load(os.path.join(a.full_run, "train_probs.npy"), mmap_mode="r")
p_early = Pf[a.early_epoch - 1].astype(np.float32)
p_fullavg = np.zeros((N, 10), np.float32)
for e in range(len(Pf)):
    p_fullavg += Pf[e].astype(np.float32)
p_fullavg /= len(Pf)


def rank01(s):
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(len(s)); return r / (len(s) - 1)


S = {
    "oof_final": 1 - oof_final[ar, y],
    "oof_avg": 1 - oof_avg[ar, y],
    "oof_disagree": (oof_avg.argmax(1) != y) + 1e-3 * (1 - oof_avg[ar, y]),
    "loss_early": 1 - p_early[ar, y],
    "loss_avg": 1 - p_fullavg[ar, y],
}
S["combo"] = (rank01(S["oof_avg"]) + rank01(S["loss_avg"])) / 2

score = S[a.signal]
pd.DataFrame({"index": tr, "score": score.astype(np.float64)}).to_csv(a.pred, index=False)
pd.DataFrame({"index": tr, **S}).to_csv(os.path.join(a.out, "all_signals.csv"), index=False)
np.save(os.path.join(a.out, "oof_avg_probs.npy"), oof_avg)

# ---- Task 4 inputs ----
oof_pred = oof_avg.argmax(1)
n_est = int((oof_pred != y).sum())
top = np.argsort(-score)[:n_est]
keep = np.ones(N, bool); keep[top] = False
np.save(os.path.join(a.out, "t4_filter_fit.npy"), np.sort(tr[keep]))
relabel = df[a.label_col].values.copy()          # length 50000, indexed by CIFAR index
relabel[tr[top]] = oof_pred[top]
np.save(os.path.join(a.out, "t4_relabel.npy"), relabel)
info = {"signal": a.signal, "n_est_noisy": n_est, "est_noise_rate": n_est / N,
        "filter_keep": int(keep.sum()), "relabel_changed": int((relabel[tr] != y).sum())}
print(json.dumps(info, indent=2))

# ---- own check with clean labels (measurement only) ----
if a.check:
    wrong = (df.loc[tr, "clean_label"].values != y)
    k_true = int(wrong.sum())

    def auroc(s, t):
        r = rank01(s) * (len(s) - 1) + 1
        n1 = t.sum(); n0 = len(t) - n1
        return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    rep = {name: {"auroc": auroc(s, wrong),
                  "precision_at_k": float(wrong[np.argsort(-s)[:k_true]].mean())} for name, s in S.items()}
    rep["_true_noisy_count"] = k_true
    rep["_filter_removed_precision"] = float(wrong[top].mean())
    rep["_relabel_correct_on_changed"] = float(
        (relabel[tr][top] == df.loc[tr, "clean_label"].values[top])[relabel[tr][top] != y[top]].mean())
    info["check"] = rep
    print(json.dumps(rep, indent=2))
json.dump(info, open(os.path.join(a.out, "t3_info.json"), "w"), indent=2)
