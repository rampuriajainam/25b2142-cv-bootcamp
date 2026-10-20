"""One training script for every run (Tasks 1, 2, 3, 4, 5).

Each run writes to --out:
  log_full.csv      every per-epoch metric
  dynamics.csv      exactly the Task 2 columns (epoch,train_acc_correct,train_acc_mislabeled,val_acc_noisy,val_acc_clean)
  test_logits.npy   [E, 10000, 10] float32  -> pick any epoch later, no retraining
  val_logits.npy    [E, 5000, 10]  float32
  train_probs.npy   [E, 45000, 10] float16  softmax on ALL train_idx images (no aug) -> Task 3 signals
  summary.json      final + best-by-selection-label epoch and accuracies
  model_final.pt

Clean labels are used ONLY for logged metrics (allowed by the rules), never for training or selection.
Model selection uses --val_label_col (noisy_label by default; synthetic_label for Task 5).
"""
import argparse, json, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import ResNet18


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="folder with train_images.npy, labels.csv, *_idx.txt, test_images.npy")
    p.add_argument("--out", required=True)
    p.add_argument("--label_col", default="noisy_label", choices=["noisy_label", "clean_label", "synthetic_label"])
    p.add_argument("--val_label_col", default=None,
                   help="label used for model selection (default: synthetic_label if training on it, else noisy_label)")
    p.add_argument("--loss", default="ce", choices=["ce", "ls", "gce", "sce"])
    p.add_argument("--ls_eps", type=float, default=0.1)
    p.add_argument("--gce_q", type=float, default=0.7)
    p.add_argument("--sce_alpha", type=float, default=0.1)
    p.add_argument("--sce_beta", type=float, default=1.0)
    p.add_argument("--fit_idx", default=None,
                   help="optional .npy/.txt of training indices to fit on (k-fold fold or filtered subset); default train_idx.txt")
    p.add_argument("--relabel", default=None, help="optional .npy of length 50000 replacing --label_col labels")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--eval_bs", type=int, default=1000)
    return p.parse_args()


def load_idx(path):
    return np.load(path) if path.endswith(".npy") else np.loadtxt(path, dtype=np.int64)


# ---------------- augmentation on GPU: random crop (pad 4) + horizontal flip ----------------
def augment(x):
    B, C, H, W = x.shape
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    rows = (i[:, None] + ar)[:, None, :, None]          # B,1,H,1
    cols = (j[:, None] + ar)[:, None, None, :]          # B,1,1,W
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    x = xp[b, c, rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


# ---------------- losses ----------------
def make_loss(a):
    def ce(logits, y):
        return F.cross_entropy(logits, y)

    def ls(logits, y):
        return F.cross_entropy(logits, y, label_smoothing=a.ls_eps)

    def gce(logits, y):  # Generalized CE (Zhang & Sabuncu 2018)
        p_y = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return ((1 - p_y ** a.gce_q) / a.gce_q).mean()

    def sce(logits, y):  # Symmetric CE (Wang et al. 2019); RCE = -log(1e-4) * (1 - p_y)
        p_y = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        rce = -np.log(1e-4) * (1 - p_y)
        return a.sce_alpha * F.cross_entropy(logits, y) + a.sce_beta * rce.mean()

    return {"ce": ce, "ls": ls, "gce": gce, "sce": sce}[a.loss]


def main():
    a = get_args()
    if a.val_label_col is None:
        a.val_label_col = "synthetic_label" if a.label_col == "synthetic_label" else "noisy_label"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda" and not a.no_amp
    torch.backends.cudnn.benchmark = True

    # ---------------- data ----------------
    df = pd.read_csv(os.path.join(a.data, "labels.csv")).sort_values("index").reset_index(drop=True)
    assert (df["index"].values == np.arange(len(df))).all()
    train_idx = load_idx(os.path.join(a.data, "train_idx.txt"))
    val_idx = load_idx(os.path.join(a.data, "val_idx.txt"))
    fit_idx = load_idx(a.fit_idx) if a.fit_idx else train_idx
    assert len(np.intersect1d(fit_idx, val_idx)) == 0, "fit_idx overlaps validation"

    y_train_col = df[a.label_col].values.copy()
    if a.relabel:
        y_train_col = np.load(a.relabel).astype(np.int64)
        assert len(y_train_col) == len(df)
    y_noisy = df["noisy_label"].values
    y_clean = df["clean_label"].values               # metrics only
    y_valsel = df[a.val_label_col].values             # model selection

    X = torch.from_numpy(np.load(os.path.join(a.data, "train_images.npy"))).to(dev)       # N,32,32,3 uint8
    Xt = torch.from_numpy(np.load(os.path.join(a.data, "test_images.npy"))).to(dev)
    mean = X[torch.from_numpy(train_idx).to(dev)].float().mean((0, 1, 2)) / 255
    std = X[torch.from_numpy(train_idx).to(dev)].float().std((0, 1, 2)) / 255
    mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)

    def prep(xb, aug):
        x = (xb.permute(0, 3, 1, 2).float() / 255 - mean) / std
        if aug:
            x = augment(x)
        return x.contiguous(memory_format=torch.channels_last)

    Y = torch.from_numpy(y_train_col).long().to(dev)
    fit_t = torch.from_numpy(fit_idx).long().to(dev)

    # ---------------- model / optim ----------------
    net = ResNet18(num_classes=10).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=True)
    steps_per_epoch = max(1, len(fit_idx) // a.bs)                    # drop last partial batch (avoids BN size-1 batches)
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = make_loss(a)

    @torch.no_grad()
    def predict(images, idx=None):
        net.eval()
        n = len(idx) if idx is not None else len(images)
        out = []
        for s in range(0, n, a.eval_bs):
            sl = idx[s:s + a.eval_bs] if idx is not None else torch.arange(s, min(s + a.eval_bs, n), device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out.append(net(prep(images[sl], False)).float())
        return torch.cat(out).cpu()

    E = a.epochs
    tr_t, va_t = torch.from_numpy(train_idx).long().to(dev), torch.from_numpy(val_idx).long().to(dev)
    test_logits = np.zeros((E, len(Xt), 10), np.float32)
    val_logits = np.zeros((E, len(val_idx), 10), np.float32)
    train_probs = np.zeros((E, len(train_idx), 10), np.float16)

    # masks for dynamics, computed on the images actually fitted
    assert np.isin(fit_idx, train_idx).all(), "fit_idx must be a subset of train_idx"
    order = np.argsort(train_idx); train_sorted = train_idx[order]
    fit_pos = order[np.searchsorted(train_sorted, fit_idx)]
    fit_correct = y_train_col[fit_idx] == y_clean[fit_idx]

    log = []
    for ep in range(1, E + 1):
        t0 = time.time()
        net.train()
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        run_loss = 0.0
        for s in range(steps_per_epoch):
            b = perm[s * a.bs:(s + 1) * a.bs]
            x, y = prep(X[b], True), Y[b]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = net(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item()

        # ---- evaluation (no augmentation) ----
        lt = predict(X, tr_t); lv = predict(X, va_t); lte = predict(Xt)
        pt = F.softmax(lt, 1)
        train_probs[ep - 1] = pt.numpy(); val_logits[ep - 1] = lv.numpy(); test_logits[ep - 1] = lte.numpy()

        pred_fit = pt.argmax(1).numpy()[fit_pos]
        hit = pred_fit == y_train_col[fit_idx]           # fits the label it was trained on
        pv = lv.argmax(1).numpy()
        row = dict(
            epoch=ep,
            train_acc_correct=float(hit[fit_correct].mean()) if fit_correct.any() else float("nan"),
            train_acc_mislabeled=float(hit[~fit_correct].mean()) if (~fit_correct).any() else float("nan"),
            val_acc_noisy=float((pv == y_valsel[val_idx]).mean()),
            val_acc_clean=float((pv == y_clean[val_idx]).mean()),
            # extras (not in dynamics.csv)
            train_mislabeled_pred_clean=float((pred_fit[~fit_correct] == y_clean[fit_idx][~fit_correct]).mean())
            if (~fit_correct).any() else float("nan"),
            val_acc_noisy_label_col=float((pv == y_noisy[val_idx]).mean()),
            train_loss=run_loss / steps_per_epoch,
            lr=opt.param_groups[0]["lr"],
            sec=time.time() - t0,
        )
        log.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        # save every epoch so a crashed/killed session still leaves usable outputs
        pd.DataFrame(log).to_csv(os.path.join(a.out, "log_full.csv"), index=False)
        dyn = pd.DataFrame(log)[["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]]
        dyn.to_csv(os.path.join(a.out, "dynamics.csv"), index=False)
        np.save(os.path.join(a.out, "test_logits.npy"), test_logits[:ep])
        np.save(os.path.join(a.out, "val_logits.npy"), val_logits[:ep])
        np.save(os.path.join(a.out, "train_probs.npy"), train_probs[:ep])

    torch.save(net.state_dict(), os.path.join(a.out, "model_final.pt"))
    df_log = pd.DataFrame(log)
    best = int(df_log["val_acc_noisy"].idxmax())
    summary = dict(
        selection_label=a.val_label_col,
        final_epoch=E,
        final_val_acc_sel=log[-1]["val_acc_noisy"],
        best_epoch=int(df_log.loc[best, "epoch"]),
        best_val_acc_sel=float(df_log.loc[best, "val_acc_noisy"]),
    )
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
