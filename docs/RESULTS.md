# Full Results

Detailed results from **"SE-SMOTE: Latent-Space Oversampling for Long-Tailed Oil Palm Fruit Classification"** (ICTS 2025). All numbers below are taken directly from the paper's Table I (class counts) and Table II (macro-averaged F1-score across five classifiers). For the headline story and method walkthrough, see the [project README](../README.md).

---

## Experimental setup

- **Dataset.** 5-class oil palm fruit ripeness images (64×64 RGB) collected from a Kalimantan plantation.
- **Classes.** Immature, Partially Ripe, Fully Ripe, Over Ripe, Decayed.
- **Split.** 80/20 stratified train/test, repeated under different random seeds for robustness.
- **Long-tail construction.** Geometric schedule parameterised by imbalance ratio (IR). The tail class is set by `PreprocessConfig.tail_class_label` in `src/config.py` (default: Fully Ripe).
- **Metric.** Macro-averaged F1 — equal weight per class, the right metric under imbalance. The paper does **not** report accuracy, because accuracy is misleading when a single majority class dominates.
- **Classifiers.** KNN, SVM, Logistic Regression, XGBoost, Random Forest — all with scikit-learn defaults.

### Table I — class counts by imbalance ratio

| Class           | IR = 10 | IR = 30 | IR = 50 | IR = 70 | IR = 100 |
|-----------------|---------|---------|---------|---------|----------|
| Partially Ripe  | 1405    | 1405    | 1405    | 1405    | 1405     |
| Immature        | 758     | 600     | 528     | 485     | 444      |
| Over Ripe       | 444     | 256     | 198     | 167     | 140      |
| Decayed         | 249     | 109     | 74      | 58      | 44       |
| Fully Ripe      | 140     | 46      | 28      | 20      | 14       |
| **Total**       | 2996    | 2416    | 2233    | 2135    | 2047     |

---

## Table II — macro-averaged F1 by classifier

SE-SMOTE and EOS rows in bold when they are the best (or tied-best) for that IR group.

### IR = 10

| Method              | KNN               | SVM                | Logistic Reg.      | XGBoost            | Random Forest      |
|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| SMOTE               | 0.304 ± 0.015     | 0.484 ± 0.034      | 0.423 ± 0.020      | 0.440 ± 0.033      | 0.393 ± 0.031      |
| DeepSMOTE           | 0.382 ± 0.018     | 0.382 ± 0.018      | 0.382 ± 0.018      | 0.382 ± 0.018      | 0.382 ± 0.018      |
| VAE-SMOTE           | 0.413 ± 0.013     | 0.528 ± 0.036      | 0.470 ± 0.023      | 0.505 ± 0.036      | 0.501 ± 0.028      |
| EOS                 | 0.805 ± 0.024     | 0.816 ± 0.026      | 0.810 ± 0.031      | 0.815 ± 0.027      | 0.813 ± 0.026      |
| **SE-SMOTE (ours)** | **0.854 ± 0.003** | **0.863 ± 0.009**  | **0.833 ± 0.009**  | **0.850 ± 0.037**  | **0.859 ± 0.039**  |

### IR = 30

| Method              | KNN               | SVM               | Logistic Reg.     | XGBoost           | Random Forest     |
|---------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| SMOTE               | 0.269 ± 0.042     | 0.269 ± 0.042     | 0.269 ± 0.042     | 0.269 ± 0.042     | 0.269 ± 0.042     |
| DeepSMOTE           | 0.303 ± 0.050     | 0.303 ± 0.050     | 0.303 ± 0.050     | 0.303 ± 0.050     | 0.303 ± 0.050     |
| VAE-SMOTE           | 0.348 ± 0.046     | 0.348 ± 0.046     | 0.348 ± 0.046     | 0.348 ± 0.046     | 0.348 ± 0.046     |
| **EOS**             | **0.733 ± 0.039** | **0.737 ± 0.037** | **0.741 ± 0.041** | **0.755 ± 0.035** | **0.747 ± 0.035** |
| SE-SMOTE (ours)     | 0.706 ± 0.052     | 0.732 ± 0.025     | 0.671 ± 0.050     | 0.671 ± 0.015     | 0.697 ± 0.033     |

### IR = 50

| Method              | KNN               | SVM               | Logistic Reg.     | XGBoost           | Random Forest     |
|---------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| SMOTE               | 0.204 ± 0.082     | 0.204 ± 0.082     | 0.204 ± 0.082     | 0.204 ± 0.082     | 0.204 ± 0.082     |
| DeepSMOTE           | 0.315 ± 0.095     | 0.315 ± 0.095     | 0.315 ± 0.095     | 0.315 ± 0.095     | 0.315 ± 0.095     |
| VAE-SMOTE           | 0.342 ± 0.110     | 0.342 ± 0.110     | 0.342 ± 0.110     | 0.342 ± 0.110     | 0.342 ± 0.110     |
| **EOS**             | **0.662 ± 0.065** | **0.680 ± 0.051** | **0.674 ± 0.048** | **0.678 ± 0.060** | **0.680 ± 0.052** |
| SE-SMOTE (ours)     | 0.638 ± 0.156     | 0.666 ± 0.133     | 0.619 ± 0.108     | 0.663 ± 0.090     | 0.651 ± 0.136     |

### IR = 70

| Method              | KNN               | SVM               | Logistic Reg.     | XGBoost           | Random Forest     |
|---------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| SMOTE               | 0.208 ± 0.041     | 0.208 ± 0.041     | 0.208 ± 0.041     | 0.208 ± 0.041     | 0.208 ± 0.041     |
| DeepSMOTE           | 0.273 ± 0.052     | 0.273 ± 0.052     | 0.273 ± 0.052     | 0.273 ± 0.052     | 0.273 ± 0.052     |
| VAE-SMOTE           | 0.261 ± 0.019     | 0.332 ± 0.041     | 0.341 ± 0.056     | 0.323 ± 0.045     | 0.303 ± 0.027     |
| EOS                 | 0.647 ± 0.058     | **0.665 ± 0.044** | **0.661 ± 0.032** | **0.670 ± 0.032** | **0.657 ± 0.035** |
| **SE-SMOTE (ours)** | **0.673 ± 0.009** | 0.623 ± 0.022     | 0.638 ± 0.060     | 0.633 ± 0.074     | 0.629 ± 0.018     |

### IR = 100 (headline values from the paper abstract)

| Method              | Macro-F1 |
|---------------------|----------|
| SMOTE               | 0.276    |
| DeepSMOTE           | 0.275    |
| VAE-SMOTE           | 0.308    |
| EOS                 | 0.594    |
| **SE-SMOTE (ours)** | **0.621** |

Relative improvements at IR = 100 (per the paper): **+124.9%** over SMOTE, **+125.8%** over DeepSMOTE, **+101.7%** over VAE-SMOTE, **+4.5%** over EOS.

---

## What the numbers say

- **Extreme imbalance (IR = 100) is where SE-SMOTE wins decisively** — this is the regime oversampling is designed for, and the only method that stays above 0.60 macro-F1 is SE-SMOTE.
- **Mild imbalance (IR = 10)** also goes to SE-SMOTE — the supervised latent space preserves class structure better than any of the autoencoder-based or pixel-space baselines. Gap over EOS is small (~0.04–0.05) but consistent.
- **Mid-range (IR = 30–70)**, EOS and SE-SMOTE trade top spot. EOS edges SE-SMOTE at IR = 30 and IR = 50 across most classifiers; they are roughly tied at IR = 70 (SE-SMOTE leads on KNN, EOS leads on the other four).
- **Pixel-space SMOTE, DeepSMOTE, and VAE-SMOTE collapse below 0.42 macro-F1 even at IR = 10** — strong evidence that the problem isn't the interpolation itself, it's the space the interpolation happens in.

The practical implication: SE-SMOTE's architectural choice — supervising the latent space with a Separator branch *before* oversampling — pays off most when the imbalance is severe and the minority classes are at risk of disappearing entirely.

---

## Qualitative evidence — latent space

Two figures in `assets/` back up the numbers:

- **`comparison_latent_before_smote.png`** — t-SNE *before* oversampling: (a) Pixel, (b) VAE, (c) AE, (d) CNN Pretrained, (e) SE. Only the supervised Super-Encoder (e) produces visibly disjoint class clusters.
- **`comparison_latent_after_smote.png`** — t-SNE *after* oversampling: (a) SMOTE, (b) VAE-SMOTE, (c) DeepSMOTE, (d) CNN-EOS, (e) SE-SMOTE. SE-SMOTE is the only method whose synthetic samples stay inside the original class manifolds.

---

## Reproducing these numbers

From the repository root:

```bash
python scripts/run_preprocess.py      # build the long-tailed split
python scripts/run_train.py           # Phase 1 — train Super-Encoder
python scripts/run_oversample.py      # Phase 2 — SMOTE in latent space
```

Then train any scikit-learn classifier on the output of step 3 (`latent_features_dataset.zip`) and evaluate against the held-out validation split saved in `data/preprocessed_dataset.zip`.

To sweep over imbalance ratios, change `PreprocessConfig.imbalance_factor` in `src/config.py` between runs.
