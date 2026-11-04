# Phase 2 / M4 — Machine Learning Model Training Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M4 — Train 10–15 ML/DL algorithms (OA category)  
**Date:** 2026-08-08  
**Status:** Complete  
**Depends on:** M3 train-ready matrices (`data/ml/m3/oa_*.csv`)

---

## 1. Objective

Train a broad suite of classical and modern classifiers on the Phase 2 primary task — **open-access category prediction** — using the leakage-safe, consensus-selected feature set from M3.

M4 delivers:

1. **14** trained algorithms (within the DA2 10–15 requirement)
2. 5-fold CV on train + validation + held-out test metrics
3. Leaderboard and comparison plot
4. Top-3 model checkpoints
5. Experiment logging into SQLite (`experiment_runs`, `predictions`)

Hyperparameter search is deferred to **M5**. Comparative ROC/CM deep-dive plots are expanded in **M6** (per-model confusion matrices and probabilities are already saved here).

---

## 2. Task definition

| Item | Value |
|------|--------|
| Track | Primary — `oa_category` |
| Problem | Multiclass classification |
| Classes | `fully_open`, `partially_open`, `closed` |
| Features | 30 (M3 consensus + scaled structural / TF-IDF) |
| Train / Val / Test | 1399 / 300 / 301 |
| Split | Stratified by `oa_category`, `random_state=42` |
| Ranking metric | Test **macro-F1** (CV uses `f1_macro`) |

Baseline majority-class rate ≈ 41.7% (`fully_open`) — dummy accuracy on test ≈ 41.9%.

---

## 3. Algorithms trained (14)

| # | Model key | Family |
|---|-----------|--------|
| 1 | `dummy` | Baseline (most frequent) |
| 2 | `logistic_regression` | Linear |
| 3 | `gaussian_nb` | Probabilistic |
| 4 | `knn` | Instance-based |
| 5 | `linear_svc` | SVM (+ calibration for probabilities) |
| 6 | `svc_rbf` | Kernel SVM |
| 7 | `decision_tree` | Tree |
| 8 | `random_forest` | Bagging ensemble |
| 9 | `extra_trees` | Bagging ensemble |
| 10 | `adaboost` | Boosting |
| 11 | `gradient_boosting` | Boosting |
| 12 | `xgboost` | Gradient boosting |
| 13 | `lightgbm` | Gradient boosting |
| 14 | `mlp` | Neural net (sklearn MLP) |

Config: `configs/phase2/models.yaml`  
Factory: `src/researchpilot/models/registry.py`  
Trainer: `src/researchpilot/models/train_eval.py`  
Entrypoint: `scripts/phase2/04_train_models.py`

---

## 4. Evaluation protocol

| Stage | Purpose |
|-------|---------|
| 5-fold CV on train | Stable `cv_f1_macro` for ranking / DB `cv_score` |
| Validation set | Model selection signal (no tuning yet in M4) |
| Test set | Final reported generalization metrics |

**Metrics:** accuracy, precision_macro, recall_macro, f1_macro, f1_weighted, roc_auc_ovr (macro), average_precision_macro.

Default hyperparameters only (tuning = M5).

---

## 5. Results — test leaderboard

Sorted by **test macro-F1** (champion first):

| Rank | Model | CV F1 (macro) | Test Acc | Test F1 (macro) | Test ROC-AUC (OvR) |
|------|-------|---------------|----------|-----------------|--------------------|
| 1 | **adaboost** | 0.456 | **0.482** | **0.452** | 0.610 |
| 2 | extra_trees | 0.463 | 0.465 | 0.436 | 0.623 |
| 3 | gradient_boosting | 0.472 | 0.462 | 0.433 | 0.597 |
| 4 | gaussian_nb | 0.449 | 0.442 | 0.431 | 0.596 |
| 5 | random_forest | 0.460 | 0.455 | 0.425 | 0.613 |
| 6 | logistic_regression | 0.478 | 0.439 | 0.423 | 0.619 |
| 7 | linear_svc | 0.473 | 0.465 | 0.415 | 0.619 |
| 8 | xgboost | 0.469 | 0.432 | 0.405 | 0.599 |
| 9 | svc_rbf | 0.453 | 0.422 | 0.404 | 0.604 |
| 10 | lightgbm | 0.452 | 0.409 | 0.391 | 0.595 |
| 11 | mlp | 0.465 | 0.439 | 0.389 | 0.613 |
| 12 | decision_tree | 0.428 | 0.369 | 0.350 | 0.517 |
| 13 | knn | 0.401 | 0.389 | 0.333 | 0.582 |
| 14 | dummy | 0.196 | 0.419 | 0.197 | 0.500 |

Full table: `evaluation/reports/m4/leaderboard.csv`  
Markdown: `evaluation/reports/m4/leaderboard.md`  
Plot: `reports/figures/m4_model_comparison_f1.png`

### Champion

| Item | Value |
|------|--------|
| Model | **AdaBoost** |
| Test accuracy | 0.482 |
| Test macro-F1 | 0.452 |
| Test ROC-AUC (OvR macro) | 0.610 |
| Checkpoint | `models/checkpoints/m4/adaboost.joblib` |

Also saved: `extra_trees.joblib`, `gradient_boosting.joblib`.

### Interpretation

- All non-dummy models beat the dummy **macro-F1** (0.197); most also beat chance on ROC-AUC (~0.50).
- Predicting OA category from title/abstract TF-IDF + a few metadata signals is a **hard** multiclass problem; absolute accuracy in the mid-40% range is expected without deeper text embeddings.
- Tree ensembles (AdaBoost, Extra Trees, Gradient Boosting) lead the board — good candidates for **M5** tuning.
- Logistic regression remains competitive on CV/val and is a strong interpretable baseline for later explanation work.

---

## 6. Artifacts

| Artifact | Path |
|----------|------|
| Leaderboard CSV/MD | `evaluation/reports/m4/leaderboard.*` |
| Per-model metrics JSON | `evaluation/reports/m4/metrics_<model>.json` |
| Test predictions | `evaluation/reports/m4/predictions_test_<model>.csv` |
| Summary JSON | `evaluation/reports/m4/m4_training_summary.json` |
| Comparison figure | `reports/figures/m4_model_comparison_f1.png` |
| Top checkpoints | `models/checkpoints/m4/*.joblib` |
| DB runs | `experiment_runs` + `predictions` (14 runs) |

---

## 7. Database logging

Each successful model writes:

- `experiment_runs` — track=`oa_category`, metrics JSON, `cv_score`
- `predictions` — test-set `y_true` / `y_pred` / class probabilities

Query example:

```sql
SELECT model_name, cv_score, metrics_json
FROM experiment_runs
WHERE track = 'oa_category'
ORDER BY cv_score DESC;
```

---

## 8. How to reproduce

```powershell
cd E:\Tuned_Research
# prerequisites if needed
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py

# M4
python scripts/phase2/04_train_models.py
```

---

## 9. DA2 rubric mapping (M4)

| Requirement | M4 contribution |
|-------------|-----------------|
| 3. Implementation of 10–15 ML/DL algorithms (3 marks) | **14 models trained and compared** |
| 5. Comparative metrics (partial) | Leaderboard with Acc / F1 / ROC-AUC |
| 2. Database connectivity | Results logged to SQLite |
| 6. Visualizations (partial) | Macro-F1 comparison bar chart |

Still pending: hyperparameter tuning (**M5**), full ROC/PR/CM comparative figure pack (**M6**).

---

## 10. Next steps (M5)

1. Tune AdaBoost, Extra Trees, Gradient Boosting, Logistic Regression (and optionally XGBoost)
2. Use validation macro-F1 / RandomizedSearchCV or Optuna
3. Re-evaluate champion on the frozen test set once
4. Feed tuned models into M6 visualization suite

---

## 11. Conclusion

M4 is complete: ResearchPilot’s predictive layer now includes **14** algorithms for open-access category classification, with a reproducible leaderboard, checkpoints, and database-backed experiment logs. **AdaBoost** is the current champion under default hyperparameters and is the starting point for M5 optimization.
