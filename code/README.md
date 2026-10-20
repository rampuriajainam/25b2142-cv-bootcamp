# CIFAR-10N "worst": learning from messy human labels

Jainam Rampuria · 25b2142. For the analysis, figures and what I learned, see **[ANALYSIS.md](ANALYSIS.md)**.

## One fixed training recipe

Every network in every task uses the same recipe. Only the method under test changes.

| | |
|---|---|
| Model | ResNet-18 from the provided `model.py`, from scratch |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 |
| LR schedule | 0.1 → 0, cosine, stepped every batch, over 60 epochs |
| Batch / epochs | 128 / 60 (351 optimizer steps per epoch on 45k images) |
| Augmentation | random crop (pad 4) + horizontal flip, done on the GPU |
| Other | fp16 mixed precision, seed 0, per-channel normalisation from `train_idx` images |

Clean labels are used **only to measure**: the Task 1 confusion matrix, the Task 2 curves, my own Task 3 check, and diagnostics such as how pure the DivideMix split is. They never feed into training, filtering, relabeling or model selection. Model selection uses the noisy validation labels in Tasks 1–4 and the synthetic validation labels in Task 5.

## Files

| File | What it does |
|---|---|
| `train.py` | Every single-network run (Tasks 1, 2, 3 folds, GCE, filter, relabel, Task 5 baseline). Flags: `--label_col`, `--loss {ce,ls,gce,sce}`, `--fit_idx`, `--relabel`. Saves per-epoch test/val logits and softmax on all 45k training images, so any epoch can be exported later without retraining. |
| `dividemix.py` | DivideMix (2 networks): co-divide with a 2-component Gaussian mixture on per-image loss, label refinement, co-guessing, MixUp. Same optimizer, schedule, augmentation and steps per epoch as `train.py`; prediction = mean softmax of the two networks. |
| `data_check.py` | Task 1 noise rate, confusion matrices, per-class noise. |
| `make_preds.py` | Writes `id,pred` from the `final`, `best` (highest selection-label val acc) or N-th epoch; copies the Task 2 dynamics. |
| `make_folds.py` | Task 3: stratified 5-fold split of `train_idx` (by noisy label, seed 0). |
| `score_noise.py` | Task 3 scores from the 5 fold runs, and the Task 4 filter/relabel inputs. `--check` only prints AUROC / precision@k against clean labels. |
| `analysis.py` | All figures in `figures/` and `figures/analysis_numbers.json` (no training). |
| `notebooks/` | The exact Kaggle / Colab notebooks that produced the submitted runs (see "Where each run happened"). |

Layout assumed by the commands: `data/` holds the provided files; each run writes `outputs/<run>/`; submissions go to `predictions/`. Run the commands from this folder.

## Exact commands

### Task 1: data and baselines
```bash
python data_check.py --data data --out outputs/data_check
python train.py --data data --out outputs/t1_clean_ref --label_col clean_label
python train.py --data data --out outputs/t1_noisy_ce  --label_col noisy_label --loss ce
python make_preds.py --run outputs/t1_clean_ref --epoch final --out predictions/t1_clean_ref.csv
python make_preds.py --run outputs/t1_noisy_ce  --epoch final --out predictions/t1_noisy_ce.csv
```

### Task 2: training dynamics (no extra training; reuses the Task 1 noisy run)
```bash
python make_preds.py --run outputs/t1_noisy_ce --dynamics predictions/t2_dynamics.csv
python make_preds.py --run outputs/t1_noisy_ce --epoch best --out predictions/t2_early_stop.csv   # epoch 35
```

### Task 3: noise scores (exactly 5 training runs: 5-fold cross-validation on noisy labels)
```bash
python make_folds.py --data data --out folds --label_col noisy_label
for k in 0 1 2 3 4; do
  python train.py --data data --out outputs/t3_fold$k --label_col noisy_label --fit_idx folds/fit_$k.npy
done
python score_noise.py --data data --folds folds --fold_runs "outputs/t3_fold{k}" --full_run outputs/t1_noisy_ce \
    --early_epoch 35 --out outputs/t3 --pred predictions/t3_noise_scores.csv --check
```
Submitted signal: `oof_avg` = 1 − (softmax probability of the noisy label from the fold model that never trained on the image), averaged over its 60 epochs. I fixed this choice before running the check. `score_noise.py` also writes `outputs/t3/t4_filter_fit.npy` and `outputs/t3/t4_relabel.npy`. Both are sized by the number of training images whose out-of-fold prediction disagrees with the noisy label (19,689), so no clean labels are involved.

### Task 4: robust training (3 submitted methods)
```bash
python train.py     --data data --out outputs/t4_gce    --label_col noisy_label --loss gce                          # GCE, q = 0.7
python train.py     --data data --out outputs/t4_filter --label_col noisy_label --fit_idx outputs/t3/t4_filter_fit.npy
python dividemix.py --data data --out outputs/t4_dividemix --label_col noisy_label
python make_preds.py --run outputs/t4_dividemix --epoch best --out predictions/t4_dividemix.csv   # epoch 55
python make_preds.py --run outputs/t4_gce       --epoch best --out predictions/t4_gce.csv         # epoch 51
python make_preds.py --run outputs/t4_filter    --epoch best --out predictions/t4_filter.csv      # epoch 52
```
Best by noisy validation accuracy: **DivideMix (0.5928)**, ahead of GCE (0.5724) and filter (0.5722).

### Task 5: real vs synthetic noise (cross-entropy baseline + best Task 4 method, on `synthetic_label`)
```bash
python train.py     --data data --out outputs/t5_baseline_synth  --label_col synthetic_label --loss ce
python dividemix.py --data data --out outputs/t5_dividemix_synth --label_col synthetic_label
python make_preds.py --run outputs/t5_baseline_synth  --epoch best --out predictions/t5_baseline_synth.csv   # epoch 40
python make_preds.py --run outputs/t5_dividemix_synth --epoch best --out predictions/t5_best_synth.csv       # epoch 58
```
With `--label_col synthetic_label`, both scripts select on the synthetic validation labels automatically. For these runs, the `val_acc_noisy` column and `best` in `make_preds.py` mean the synthetic labels.

### Figures and analysis
```bash
python analysis.py --data data --outputs outputs --figs figures
```

## Which epoch each submitted file uses

| File | Run | Epoch | Why |
|---|---|---|---|
| `t1_clean_ref.csv`, `t1_noisy_ce.csv` | t1_clean_ref, t1_noisy_ce | 60 | Task 1 asks for the final epoch |
| `t2_early_stop.csv` | t1_noisy_ce | 35 | best noisy validation accuracy |
| `t4_dividemix.csv` / `t4_gce.csv` / `t4_filter.csv` | t4_* | 55 / 51 / 52 | best noisy validation accuracy (model selection) |
| `t5_baseline_synth.csv` / `t5_best_synth.csv` | t5_baseline_synth / t5_dividemix_synth | 40 / 58 | best synthetic validation accuracy |

## Where each run happened (all NVIDIA T4)

| Notebook | Runs |
|---|---|
| `kaggle_t1_t2_baselines.ipynb` (Kaggle, T4 ×2) | data_check, t1_clean_ref, t1_noisy_ce (+ Task 2 exports) |
| `kaggle_t3_folds_and_scores.ipynb` (Kaggle, T4 ×2) | t3_fold0–4, score_noise |
| `colab_t4_gce_t5_baseline.ipynb` (Colab, T4) | t4_gce, t5_baseline_synth |
| `kaggle_t4_filter_relabel.ipynb` (Kaggle, T4 ×2) | t4_filter, t4_relabel |
| `kaggle_t5_gce_synth.ipynb` (Kaggle, T4) | GCE on synthetic labels (analysis only) |
| `kaggle_dividemix_t4_t5.ipynb` (Kaggle, T4 ×2) | smoke test, then t4_dividemix and t5_dividemix_synth in parallel |

The notebooks find the provided files at the start of the run and copy this code from a private Kaggle dataset. Nothing outside the provided files is used. The Task 1 run used an earlier copy of `train.py` that differs only in `steps_per_epoch = len(fit_idx) // bs`, now `max(1, …)`, which has no effect with 45k images. Timing: about 17 s per epoch for a `train.py` run and about 150 s per epoch for DivideMix (two networks).

## Budget and rules, accounted for

* **Epochs:** every network ran exactly 60 epochs. DivideMix's 10 warm-up epochs are part of its 60.
* **Networks per method:** 1 for each `train.py` method; 2 for DivideMix (the maximum).
* **Task 3:** exactly 5 training runs (the 5 folds). The full-data run used as an extra signal is the Task 1 noisy run.
* **Task 4:** I trained four methods (GCE, filter, relabel, DivideMix) and submit three. Relabel (noisy-val 0.5662, the lowest) was dropped when DivideMix was added, to stay within three submitted files. It is reported in ANALYSIS.md for completeness.
* **Task 5:** the two submitted models are the cross-entropy baseline and DivideMix (the best Task 4 method). Before DivideMix existed, GCE was the best Task 4 method, so I also trained GCE on synthetic labels. That run is not submitted; it appears only in the analysis.
* **DivideMix development:** the first two notebook versions stopped in a 3-epoch smoke test on a few thousand images, before any real training. The cause was fp16 overflow with immature BatchNorm statistics. The fix clamps non-finite logits and skips non-finite steps; the submitted runs skipped **0** steps.
* **Validation images** were never trained on (`train.py` and `dividemix.py` assert it). The test set was only predicted.
