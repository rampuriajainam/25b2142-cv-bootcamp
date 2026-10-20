"""Task 4/5 method: DivideMix (Li et al., ICLR 2020) — two networks, co-divide + MixMatch-style semi-supervised training.

Each epoch (after warm-up), every network fits a 2-component GMM to its per-sample loss on the training labels;
the low-loss component = "probably clean" (labeled set, with clean-probability w), the rest = unlabeled (labels dropped).
Each network is trained on the split produced by its PEER (co-divide), so the two don't confirm their own mistakes.
Training step: label refinement (labeled), co-guessing (unlabeled), sharpening, MixUp, + uniform-prior penalty.

Budget/rules: 2 networks (allowed max), 60 epochs each (incl. warm-up), ResNet-18 from model.py.
Optimizer / lr / schedule / weight decay / crop+flip / steps per epoch are IDENTICAL to train.py; the method adds MixUp.
Clean labels are used only for logged metrics (purity of the labeled set, val_acc_clean) — never for training/selection.
Predictions = average softmax of the two networks; selection label = noisy (or synthetic for Task 5) validation labels.

Writes the same files as train.py (log_full.csv, dynamics.csv, test_logits.npy, val_logits.npy, train_probs.npy,
summary.json) so make_preds.py works unchanged.

  python dividemix.py --data data --out outputs/t4_dividemix --label_col noisy_label
  python dividemix.py --data data --out outputs/t5_best_synth --label_col synthetic_label
"""
import argparse, json, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from model import ResNet18
from train import augment, load_idx


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--label_col", default="noisy_label", choices=["noisy_label", "synthetic_label"])
    p.add_argument("--val_label_col", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10, help="plain CE (+confidence penalty) epochs before co-divide")
    p.add_argument("--bs", type=int, default=128, help="labeled batch = unlabeled batch = bs")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--p_threshold", type=float, default=0.5)
    p.add_argument("--T", type=float, default=0.5, help="sharpening temperature")
    p.add_argument("--alpha", type=float, default=4.0, help="MixUp Beta(alpha, alpha)")
    p.add_argument("--lambda_u", type=float, default=0.0, help="unlabeled loss weight (DivideMix CIFAR-10 setting)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_bs", type=int, default=1000)
    p.add_argument("--smoke", action="store_true", help="tiny CPU/GPU pipeline test; outputs are not results")
    return p.parse_args()


def main():
    a = get_args()
    if a.val_label_col is None:
        a.val_label_col = "synthetic_label" if a.label_col == "synthetic_label" else "noisy_label"
    if a.smoke:
        a.epochs, a.warmup, a.eval_bs = 4, 2, 512
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    torch.backends.cudnn.benchmark = True

    # ---------------- data (same loading / normalisation as train.py) ----------------
    df = pd.read_csv(os.path.join(a.data, "labels.csv")).sort_values("index").reset_index(drop=True)
    assert (df["index"].values == np.arange(len(df))).all()
    train_idx = load_idx(os.path.join(a.data, "train_idx.txt"))
    val_idx = load_idx(os.path.join(a.data, "val_idx.txt"))
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    X = torch.from_numpy(np.load(os.path.join(a.data, "train_images.npy"))).to(dev)
    Xt = torch.from_numpy(np.load(os.path.join(a.data, "test_images.npy"))).to(dev)
    mean = X[torch.from_numpy(train_idx).to(dev)].float().mean((0, 1, 2)) / 255
    std = X[torch.from_numpy(train_idx).to(dev)].float().std((0, 1, 2)) / 255
    mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    if a.smoke:
        train_idx, val_idx, Xt = train_idx[:4096], val_idx[:512], Xt[:512]

    y_train = df[a.label_col].values                      # the ONLY labels used for training
    y_clean = df["clean_label"].values                    # metrics only
    y_valsel = df[a.val_label_col].values                 # model selection
    N = len(train_idx)
    tr_t = torch.from_numpy(train_idx).long().to(dev)
    va_t = torch.from_numpy(val_idx).long().to(dev)
    Y = torch.from_numpy(y_train).long().to(dev)
    correct_mask = y_train[train_idx] == y_clean[train_idx]  # metrics only

    def prep(xb, aug):
        x = (xb.permute(0, 3, 1, 2).float() / 255 - mean) / std
        if aug:
            x = augment(x)
        return x.contiguous(memory_format=torch.channels_last)

    # ---------------- two networks, identical optimiser/schedule to train.py ----------------
    steps_per_epoch = max(1, N // a.bs)
    total = steps_per_epoch * a.epochs

    def make():
        net = ResNet18(num_classes=10).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=True)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * s / total)))
        return net, opt, sched, torch.amp.GradScaler("cuda", enabled=use_amp)

    nets = [make(), make()]

    def ac():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    def fin(z):          # fp16 overflow guard: inf/NaN logits -> large finite values (softmax/CE stay defined)
        return torch.nan_to_num(z.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    @torch.no_grad()
    def logits_of(net, images, idx=None):
        net.eval()
        n = len(idx) if idx is not None else len(images)
        out = []
        for s in range(0, n, a.eval_bs):
            sl = idx[s:s + a.eval_bs] if idx is not None else torch.arange(s, min(s + a.eval_bs, n), device=dev)
            with ac():
                out.append(fin(net(prep(images[sl], False))))
        return torch.cat(out)

    bad = {"run": 0, "total": 0}

    def step(net, opt, sched, scaler, loss):
        if not torch.isfinite(loss):                     # skip the update (as GradScaler does for inf grads)
            bad["run"] += 1; bad["total"] += 1
            if bad["run"] >= 50:
                raise RuntimeError("50 consecutive non-finite losses; aborting instead of wasting the run")
            sched.step()                                 # keep the lr schedule aligned with train.py
            return
        bad["run"] = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()

    # ---------------- warm-up: CE + confidence penalty (DivideMix, for asymmetric noise) ----------------
    def warmup_epoch(k):
        net, opt, sched, scaler = nets[k]
        net.train()
        perm = tr_t[torch.randperm(N, device=dev)]
        tot = 0.0
        for s in range(steps_per_epoch):
            b = perm[s * a.bs:(s + 1) * a.bs]
            with ac():
                out = net(prep(X[b], True))
            out = out.float()
            p = F.softmax(out, 1)
            loss = F.cross_entropy(out, Y[b]) + (p * F.log_softmax(out, 1)).sum(1).mean()   # CE - entropy
            step(net, opt, sched, scaler, loss)
            tot += loss.item()
        return tot / steps_per_epoch

    # ---------------- co-divide: per-sample loss -> 2-component GMM -> clean probability ----------------
    def clean_prob(k):
        lg = logits_of(nets[k][0], X, tr_t)
        loss = F.cross_entropy(lg, Y[tr_t], reduction="none").cpu().numpy().astype(np.float64)
        fin = np.isfinite(loss)
        if not fin.any():
            raise RuntimeError("all per-sample losses are non-finite")
        if not fin.all():                                   # fp16 overflow -> treat as maximally suspicious
            print(f"warning: {int((~fin).sum())} non-finite per-sample losses replaced by max", flush=True)
            loss[~fin] = loss[fin].max()
        loss = (loss - loss.min()) / (loss.max() - loss.min() + 1e-12)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=a.seed)
        gmm.fit(loss.reshape(-1, 1))
        prob = gmm.predict_proba(loss.reshape(-1, 1))[:, gmm.means_.argmin()]
        if a.smoke and min((prob > a.p_threshold).sum(), (prob <= a.p_threshold).sum()) < a.bs:
            prob = (loss <= np.median(loss)).astype(np.float64) * 0.5 + 0.25   # smoke only: force the DivideMix code path
        return prob

    def sharpen(p):
        p = p ** (1 / a.T)
        return p / p.sum(1, keepdim=True)

    # ---------------- one DivideMix epoch for net k, using the split computed by its peer ----------------
    def dividemix_epoch(k, prob):
        net, opt, sched, scaler = nets[k]
        peer = nets[1 - k][0]
        lab = np.where(prob > a.p_threshold)[0]
        unl = np.where(prob <= a.p_threshold)[0]
        if len(lab) < a.bs or len(unl) < a.bs:          # degenerate split -> fall back to warm-up style epoch
            return warmup_epoch(k), len(lab), True
        lab_t = torch.from_numpy(lab).long().to(dev)
        unl_t = torch.from_numpy(unl).long().to(dev)
        w_all = torch.from_numpy(prob).float().to(dev)
        net.train(); peer.eval()
        tot = 0.0
        for s in range(steps_per_epoch):                 # same number of optimiser steps as train.py
            li = lab_t[torch.randint(0, len(lab_t), (a.bs,), device=dev)]
            ui = unl_t[torch.randint(0, len(unl_t), (a.bs,), device=dev)]
            gi_l, gi_u = tr_t[li], tr_t[ui]              # global CIFAR indices
            x1, x2 = prep(X[gi_l], True), prep(X[gi_l], True)
            u1, u2 = prep(X[gi_u], True), prep(X[gi_u], True)
            yx = F.one_hot(Y[gi_l], 10).float()
            w = w_all[li].view(-1, 1)
            with torch.no_grad():
                with ac():
                    ou = fin(torch.cat([net(u1), net(u2), peer(u1), peer(u2)]))
                    ox = fin(torch.cat([net(x1), net(x2)]))
                pu = F.softmax(ou, 1).view(4, a.bs, 10).mean(0)
                tu = sharpen(pu)                          # co-guessing
                px = F.softmax(ox, 1).view(2, a.bs, 10).mean(0)
                tx = sharpen(w * yx + (1 - w) * px)       # label refinement
            lam = float(rng.beta(a.alpha, a.alpha)); lam = max(lam, 1 - lam)
            inp = torch.cat([x1, x2, u1, u2]); tgt = torch.cat([tx, tx, tu, tu])
            perm = torch.randperm(len(inp), device=dev)
            mi = lam * inp + (1 - lam) * inp[perm]
            mt = lam * tgt + (1 - lam) * tgt[perm]
            with ac():
                lo = net(mi.contiguous(memory_format=torch.channels_last))
            lo = lo.float()
            nx = 2 * a.bs
            Lx = -(F.log_softmax(lo[:nx], 1) * mt[:nx]).sum(1).mean()
            Lu = ((F.softmax(lo[nx:], 1) - mt[nx:]) ** 2).mean()
            pm = F.softmax(lo, 1).mean(0)
            penalty = (torch.full_like(pm, 0.1) * torch.log(0.1 / pm)).sum()   # uniform prior
            loss = Lx + a.lambda_u * Lu + penalty
            step(net, opt, sched, scaler, loss)
            tot += loss.item()
        return tot / steps_per_epoch, len(lab), False

    # ---------------- training loop ----------------
    E = a.epochs
    test_logits = np.zeros((E, len(Xt), 10), np.float32)
    val_logits = np.zeros((E, len(val_idx), 10), np.float32)
    train_probs = np.zeros((E, N, 10), np.float16)
    log = []
    for ep in range(1, E + 1):
        t0 = time.time()
        extra = {}
        if ep <= a.warmup:
            losses = [warmup_epoch(0), warmup_epoch(1)]
        else:
            probs = [clean_prob(0), clean_prob(1)]
            losses = []
            for k in (0, 1):
                l, nl, fallback = dividemix_epoch(k, probs[1 - k])   # net k trained on peer's split
                losses.append(l)
                extra[f"net{k}_fallback"] = int(fallback)
                sel = probs[1 - k] > a.p_threshold
                extra[f"net{k}_labeled_frac"] = float(sel.mean())
                extra[f"net{k}_labeled_purity"] = float(correct_mask[sel].mean()) if sel.any() else float("nan")  # metric only
                extra[f"net{k}_noisy_recall"] = float((~sel[~correct_mask]).mean())                             # metric only

        # ---- evaluation: ensemble = mean softmax of the two networks (no augmentation) ----
        def ens(images, idx=None):
            return (F.softmax(logits_of(nets[0][0], images, idx), 1) + F.softmax(logits_of(nets[1][0], images, idx), 1)) / 2
        pt, pv, pte = ens(X, tr_t).cpu(), ens(X, va_t).cpu(), ens(Xt).cpu()
        train_probs[ep - 1] = pt.numpy()
        val_logits[ep - 1] = torch.log(pv.clamp_min(1e-8)).numpy()      # log of ensemble probs (argmax-equivalent)
        test_logits[ep - 1] = torch.log(pte.clamp_min(1e-8)).numpy()
        pred_tr = pt.argmax(1).numpy(); pred_v = pv.argmax(1).numpy()
        hit = pred_tr == y_train[train_idx]
        row = dict(
            epoch=ep,
            train_acc_correct=float(hit[correct_mask].mean()),
            train_acc_mislabeled=float(hit[~correct_mask].mean()) if (~correct_mask).any() else float("nan"),
            val_acc_noisy=float((pred_v == y_valsel[val_idx]).mean()),
            val_acc_clean=float((pred_v == y_clean[val_idx]).mean()),
            train_mislabeled_pred_clean=float((pred_tr[~correct_mask] == y_clean[train_idx][~correct_mask]).mean())
            if (~correct_mask).any() else float("nan"),
            val_acc_noisy_label_col=float((pred_v == df["noisy_label"].values[val_idx]).mean()),
            train_loss=float(np.mean(losses)),
            lr=nets[0][1].param_groups[0]["lr"],
            phase="warmup" if ep <= a.warmup else "dividemix",
            skipped_steps_total=bad["total"],
            **extra,
            sec=time.time() - t0,
        )
        log.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        pd.DataFrame(log).to_csv(os.path.join(a.out, "log_full.csv"), index=False)
        pd.DataFrame(log)[["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]] \
            .to_csv(os.path.join(a.out, "dynamics.csv"), index=False)
        np.save(os.path.join(a.out, "test_logits.npy"), test_logits[:ep])
        np.save(os.path.join(a.out, "val_logits.npy"), val_logits[:ep])
        np.save(os.path.join(a.out, "train_probs.npy"), train_probs[:ep])

    for k in (0, 1):
        torch.save(nets[k][0].state_dict(), os.path.join(a.out, f"model_final_net{k}.pt"))
    d = pd.DataFrame(log)
    best = int(d["val_acc_noisy"].idxmax())
    summary = dict(selection_label=a.val_label_col, final_epoch=E, final_val_acc_sel=log[-1]["val_acc_noisy"],
                   best_epoch=int(d.loc[best, "epoch"]), best_val_acc_sel=float(d.loc[best, "val_acc_noisy"]))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
