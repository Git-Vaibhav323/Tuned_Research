# DA2 Model Inventory (Definitive)

**Policy:** Count **distinct algorithms**, not every tuned clone, as separate methods.  
M5 tuned variants are marked Tuned?=Yes under the same algorithm family.

Splits (all supervised tasks): Train 1399 / Val 300 / Test 301.

---

## A. Open-access category (`oa_category`) — M4 defaults

Source: `DA2/m4_leaderboard.csv` (test metrics).

| Model | Task | Tuned? | Acc | Prec(macro) | Rec(macro) | Macro-F1 | ROC-AUC | AP(macro) | Artifact |
|-------|------|--------|----:|------------:|-----------:|---------:|--------:|----------:|----------|
| Dummy | OA | No | 0.419 | 0.140 | 0.333 | 0.197 | 0.500 | 0.333 | baseline |
| Logistic Regression | OA | No | 0.439 | 0.424 | 0.428 | 0.423 | 0.619 | 0.450 | m4 |
| Gaussian NB | OA | No | 0.442 | 0.441 | 0.450 | 0.431 | 0.596 | 0.422 | m4 |
| k-NN | OA | No | 0.389 | 0.377 | 0.342 | 0.333 | 0.582 | 0.400 | m4 |
| Linear SVC (calibrated) | OA | No | 0.465 | 0.441 | 0.422 | 0.415 | 0.619 | 0.455 | m4 |
| RBF SVC | OA | No | 0.422 | 0.406 | 0.402 | 0.404 | 0.604 | 0.430 | m4 |
| Decision Tree | OA | No | 0.369 | 0.357 | 0.358 | 0.350 | 0.517 | 0.348 | m4 |
| Random Forest | OA | No | 0.455 | 0.436 | 0.425 | 0.425 | 0.613 | 0.450 | m4 |
| Extra Trees | OA | No | 0.465 | 0.444 | 0.438 | 0.436 | 0.623 | 0.454 | m4 ckpt |
| **AdaBoost** | OA | No | **0.482** | **0.465** | **0.450** | **0.452** | 0.610 | 0.435 | `impact` N/A; `models/checkpoints/m4/` |
| Gradient Boosting | OA | No | 0.462 | 0.448 | 0.431 | 0.433 | 0.597 | 0.425 | m4 ckpt |
| XGBoost | OA | No | 0.432 | 0.414 | 0.403 | 0.405 | 0.599 | 0.434 | m4 |
| LightGBM | OA | No | 0.409 | 0.396 | 0.388 | 0.391 | 0.595 | 0.424 | m4 |
| MLP | OA | No | 0.439 | 0.422 | 0.397 | 0.389 | 0.613 | 0.440 | m4 |

**Distinct OA algorithms = 14** (meets 10–15 requirement).

---

## B. Open-access — M5 tuned variants

Source: `DA2/m5_tuning_leaderboard.csv` + `DA2/m6_comparative_metrics.csv`.

| Model | Tuned? | Acc | Macro-F1 | ROC-AUC | AP(macro) | Best params (short) |
|-------|--------|----:|---------:|--------:|----------:|---------------------|
| Extra Trees | Yes | 0.449 | 0.439 | **0.628** | 0.454 | n_estimators=500, max_depth=16, … |
| Logistic Regression | Yes | 0.449 | 0.433 | 0.621 | 0.453 | C=5.0, lbfgs, l2 |
| Random Forest | Yes | 0.445 | 0.429 | 0.624 | 0.447 | n_estimators=200, max_depth=8, … |
| AdaBoost | Yes | 0.482 | 0.452 | 0.610 | 0.435 | n_estimators=200, lr=0.8 (≈ M4) |
| Gradient Boosting | Yes | 0.458 | 0.428 | 0.595 | 0.425 | n_estimators=150, lr=0.08, … |

Checkpoints: `models/checkpoints/m5/*_tuned.joblib`.

---

## C. Impact-tier (`impact_tier`) — M7

Source: `DA2/m7_impact_leaderboard.csv` / `evaluation/reports/m7/m7_summary.json`.

| Model | Task | Tuned? | Acc | Macro-F1 | ROC-AUC | Notes |
|-------|------|--------|----:|---------:|--------:|-------|
| **AdaBoost** | Impact | No (M7 defaults) | **0.551** | **0.543** | 0.680 | **Val champion** |
| Random Forest | Impact | No | 0.555 | 0.552 | **0.728** | Higher test F1 than champion; not selected (val rule) |
| LightGBM | Impact | No | 0.542 | 0.541 | 0.711 | |
| XGBoost | Impact | No | 0.532 | 0.530 | 0.711 | CPU-fixed; status=ok |
| Extra Trees | Impact | No | 0.538 | 0.536 | 0.715 | |
| k-NN | Impact | No | 0.518 | 0.512 | 0.682 | |
| Logistic Regression | Impact | No | 0.528 | 0.519 | 0.711 | |
| MLP | Impact | No | 0.512 | 0.498 | 0.715 | |
| Gradient Boosting | Impact | No | 0.502 | 0.498 | 0.689 | |
| Decision Tree | Impact | No | 0.455 | 0.456 | 0.615 | |
| Gaussian NB | Impact | No | 0.455 | 0.403 | 0.655 | |
| Dummy | Impact | No | 0.306 | 0.156 | 0.500 | |

**12/12** models successful. Precision/Recall macros available in detailed metric JSON where exported; M7 leaderboard emphasizes Acc/F1/ROC/AP.

Checkpoints: `models/checkpoints/m7/impact_adaboost.joblib` (+ mlp, gradient_boosting).

---

## D. Unsupervised

| Method | Task | Key hyperparams | Metric | Artifact |
|--------|------|-----------------|--------|----------|
| KMeans | Topic clustering | k=8 (selected 3–8 by silhouette) | Silhouette **0.064** | `models/checkpoints/m7/topic_kmeans.joblib` |

---

## Multi-metric OA leaders (frozen)

| Criterion | Winner | Value |
|-----------|--------|------:|
| Accuracy | m4_adaboost | 0.482 |
| Macro-F1 | m4_adaboost | 0.452 |
| ROC-AUC | m5_extra_trees_tuned | 0.628 |
