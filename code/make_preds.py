"""Turn a run folder into submission files.

  python make_preds.py --run outputs/t1_noisy_ce --epoch final --out predictions/t1_noisy_ce.csv
  python make_preds.py --run outputs/t1_noisy_ce --epoch best  --out predictions/t2_early_stop.csv
  python make_preds.py --run outputs/t1_noisy_ce --epoch 17    --out predictions/t2_early_stop.csv
  python make_preds.py --run outputs/t1_noisy_ce --dynamics predictions/t2_dynamics.csv

'best' = epoch with highest validation accuracy on the run's selection label (noisy, or synthetic for Task 5).
"""
import argparse, json, os, shutil
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--epoch", default=None, help="final | best | <int, 1-based>")
p.add_argument("--out", default=None)
p.add_argument("--dynamics", default=None)
a = p.parse_args()

if a.dynamics:
    os.makedirs(os.path.dirname(a.dynamics) or ".", exist_ok=True)
    shutil.copy(os.path.join(a.run, "dynamics.csv"), a.dynamics)
    print("wrote", a.dynamics)

if a.epoch:
    logits = np.load(os.path.join(a.run, "test_logits.npy"))
    log = pd.read_csv(os.path.join(a.run, "log_full.csv"))
    if a.epoch == "final":
        ep = len(logits)
    elif a.epoch == "best":
        ep = int(log.loc[log["val_acc_noisy"].idxmax(), "epoch"])
    else:
        ep = int(a.epoch)
    pred = logits[ep - 1].astype(np.float32).argmax(1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame({"id": np.arange(len(pred)), "pred": pred}).to_csv(a.out, index=False)
    sel = log.loc[log.epoch == ep, "val_acc_noisy"].item()
    print(f"wrote {a.out} from epoch {ep} (val acc on selection label = {sel:.4f})")
