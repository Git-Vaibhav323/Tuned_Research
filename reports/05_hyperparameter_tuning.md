# Phase 2 / M5 — Hyperparameter Tuning Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M5 — Hyperparameter tuning & model optimization  
**Date:** 2026-08-08  
**Status:** Complete  
**Depends on:** M3 matrices + M4 baseline leaderboard

---

## 1. Objective

Improve the top Phase 2 classifiers for **`oa_category`** prediction by searching hyperparameters with **RandomizedSearchCV**, while keeping the held-out **test set frozen** until final evaluation.

M5 delivers:

1. Tuned versions of 5 strong M4 models  
2. Validation-based champion selection  
3. One-shot test evaluation (no test-driven tuning)  
4. M4 vs M5 comparison plot  
5. Tuned checkpoints + SQLite experiment logs  
6. This report  

---

## 2. Protocol (no test leakage)

| Step | Data | Role |
|------|------|------|
| RandomizedSearchCV | **Train** only (3-fold CV) | Find best params (`scoring=f1_macro`) |
| Model selection | **Validation** macro-F1 | Choose M5 champion |
| Final score | **Test** once | Report generalization |
| Champion refit | Train + Val | Refit winning params before test |

| Setting | Value |
|---------|--------|
| `n_iter` | 20 |
| `cv_folds` | 3 |
| `random_state` | 42 |
| Config | `configs/phase2/tuning.yaml` |

---

## 3. Models tuned

| Model | Why included |
|-------|----------------|
| AdaBoost | M4 test champion |
| Extra Trees | Strong M4 ensemble |
| Gradient Boosting | Strong M4 ensemble |
| Random Forest | Competitive tree bagging |
| Logistic Regression | Strong linear / interpretable baseline |

Entrypoint: `scripts/phase2/05_tune_hyperparameters.py`  
Code: `src/researchpilot/models/tuning.py`

---

## 4. Best parameters found

### Extra Trees (M5 champion by validation F1)

```json
{
  "n_estimators": 500,
  "max_depth": 16,
  "min_samples_split": 10,
  "min_samples_leaf": 2,
  "max_features": "log2"
}
```

### Other tuned winners (summary)

| Model | Key best params |
|-------|-----------------|
| Logistic Regression | `C=5.0`, `solver=lbfgs`, `penalty=l2` |
| Random Forest | `n_estimators=200`, `max_depth=8`, `min_samples_split=10`, `max_features=sqrt` |
| AdaBoost | `n_estimators=200`, `learning_rate=0.8` |
| Gradient Boosting | `n_estimators=150`, `learning_rate=0.08`, `max_depth=3`, `subsample=0.85` |

Full JSON per model: `evaluation/reports/m5/tuned_<model>.json`

---

## 5. Results

### Tuning leaderboard (sorted by **validation** macro-F1)

| Rank | Model | CV best F1 | Val F1 | Test Acc | Test F1 | Test ROC-AUC |
|------|-------|------------|--------|----------|---------|--------------|
| 1 | **extra_trees** | 0.483 | **0.503** | 0.449 | 0.439 | 0.628 |
| 2 | logistic_regression | 0.461 | 0.477 | 0.449 | 0.433 | 0.621 |
| 3 | random_forest | 0.468 | 0.471 | 0.445 | 0.429 | 0.624 |
| 4 | adaboost | 0.463 | 0.444 | **0.482** | **0.452** | 0.610 |
| 5 | gradient_boosting | 0.459 | 0.442 | 0.458 | 0.428 | 0.595 |

### Comparison to M4 default champion (AdaBoost)

| Metric | M4 AdaBoost (default) | M5 Extra Trees (val-selected, refit train+val) |
|--------|------------------------|-----------------------------------------------|
| Test accuracy | **0.482** | 0.449 |
| Test macro-F1 | **0.452** | 0.439 |
| Test ROC-AUC | 0.610 | **0.628** |

Delta test macro-F1 (M5 val-champion vs M4 champion): **−0.012**

### Honest interpretation

1. Tuning **did** improve Extra Trees on CV/validation (val F1 **0.503** vs weaker M4 default behavior on val).  
2. On the **frozen test set**, the val-selected Extra Trees did **not** beat M4 AdaBoost on macro-F1 — a normal outcome when the signal is weak and the test set is small (n=301).  
3. Among tuned models, **AdaBoost** still has the best **test** macro-F1 (0.452), matching its M4 defaults (search recovered similar params).  
4. Extra Trees gained the best **test ROC-AUC** among the tuned set (0.628).  

**Selection rule for DA2 (declared a priori):** champion = best **validation** macro-F1 → **Extra Trees (tuned)**.  
**Practical deployment note:** if prioritizing test macro-F1, retain **AdaBoost**; if prioritizing ranking/probability quality, Extra Trees ROC-AUC is competitive.

---

## 6. Artifacts

| Artifact | Path |
|----------|------|
| Leaderboard | `evaluation/reports/m5/tuning_leaderboard.csv` |
| Summary | `evaluation/reports/m5/m5_tuning_summary.json` |
| Per-model tuned JSON | `evaluation/reports/m5/tuned_*.json` |
| Predictions | `evaluation/reports/m5/predictions_test_*_tuned.csv` |
| Checkpoints | `models/checkpoints/m5/*_tuned.joblib` |
| M4 vs M5 plot | `reports/figures/m5_vs_m4_test_f1.png` |
| DB | `experiment_runs` notes = `M5 tuned hyperparameters` |

---

## 7. How to reproduce

```powershell
cd E:\Tuned_Research
python scripts/phase2/05_tune_hyperparameters.py
```

Prerequisites: M3 feature files under `data/ml/m3/` and (for comparison) M4 summary at `evaluation/reports/m4/m4_training_summary.json`.

---

## 8. DA2 rubric mapping (M5)

| Requirement | M5 contribution |
|-------------|-----------------|
| 4. Hyperparameter tuning & optimization (1 mark) | **Satisfied** — RandomizedSearchCV on 5 models, documented params, val-based selection, test reporting |
| 5. Comparative analysis (partial) | M4 default vs M5 tuned table + plot |
| Database connectivity | Tuned runs logged to SQLite |

Next: **M6** comparative visualizations (ROC, PR, confusion matrices, feature importance) using M4/M5 artifacts.

---

## 9. Conclusion

M5 is complete. ResearchPilot now has optimized hyperparameters for five leading OA-category models, with a clear validation-based champion (**tuned Extra Trees**) and transparent test-set comparison against the M4 AdaBoost baseline. The tuning pipeline is reproducible and ready to feed M6 visualization work.
