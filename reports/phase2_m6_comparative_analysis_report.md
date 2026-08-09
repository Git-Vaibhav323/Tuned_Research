# Phase 2 / M6 — Comparative Visualizations Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M6 — Comparative performance visualizations  
**Date:** 2026-08-09  
**Status:** Complete  
**Depends on:** M4 prediction CSVs + M5 tuned predictions/checkpoints

---

## 1. Objective

Produce the DA2 comparative visualization pack for open-access category classification:

1. Confusion matrices  
2. ROC curves (one-vs-rest) + macro ROC-AUC comparison  
3. Precision–Recall curves + macro AP comparison  
4. Grouped metric bars  
5. Feature importance charts  

M6 does **not** retrain models; it consumes M4/M5 artifacts.

---

## 2. Models visualized (8)

| Label | Source |
|-------|--------|
| `m4_adaboost` | M4 default champion |
| `m4_extra_trees` | M4 top ensemble |
| `m4_gradient_boosting` | M4 top ensemble |
| `m5_extra_trees_tuned` | M5 val-selected champion |
| `m5_adaboost_tuned` | M5 tuned AdaBoost |
| `m5_logistic_regression_tuned` | M5 tuned LogReg |
| `m5_random_forest_tuned` | M5 tuned RF |
| `m5_gradient_boosting_tuned` | M5 tuned GB |

Config: `configs/phase2/comparative.yaml`  
Entrypoint: `scripts/phase2/06_comparative_visualizations.py`  
Code: `src/researchpilot/evaluation/comparative_plots.py`

---

## 3. Comparative metrics (test set)

| Model | Accuracy | F1 macro | ROC-AUC | Avg Precision |
|-------|----------|----------|---------|---------------|
| **m4_adaboost** | **0.482** | **0.452** | 0.610 | 0.435 |
| m5_adaboost_tuned | 0.482 | 0.452 | 0.610 | 0.435 |
| m5_extra_trees_tuned | 0.449 | 0.439 | **0.628** | 0.454 |
| m4_extra_trees | 0.465 | 0.436 | 0.623 | **0.454** |
| m5_logistic_regression_tuned | 0.449 | 0.433 | 0.621 | 0.453 |
| m4_gradient_boosting | 0.462 | 0.433 | 0.597 | 0.425 |
| m5_random_forest_tuned | 0.445 | 0.429 | 0.624 | 0.447 |
| m5_gradient_boosting_tuned | 0.458 | 0.428 | 0.595 | 0.425 |

Full table: `evaluation/reports/m6/comparative_metrics.csv`

### Takeaways

- **Best macro-F1 / accuracy:** M4 AdaBoost (and equivalent M5 AdaBoost tuned params).  
- **Best ranking quality (ROC-AUC):** M5 Extra Trees tuned (0.628).  
- Ensembles dominate; class confusion is hardest between `fully_open` and `partially_open` (see CM figures).

---

## 4. Figure catalog

All figures: `reports/figures/m6/`

### Confusion matrices
- Per-model: `cm_m4_*.png`, `cm_m5_*.png`
- Grid: `cm_grid_comparative.png`

### ROC (one-vs-rest)
- Per class: `roc_ovr_fully_open.png`, `roc_ovr_partially_open.png`, `roc_ovr_closed.png`
- Macro comparison: `roc_auc_comparison.png`

### Precision–Recall
- Per class: `pr_fully_open.png`, `pr_partially_open.png`, `pr_closed.png`
- Macro AP comparison: `pr_ap_comparison.png`

### Metrics overview
- `metrics_grouped_bars.png`

### Feature importance
- `feature_importance_adaboost_m4.png`
- `feature_importance_extra_trees_m5.png`
- `feature_importance_random_forest_m5.png`
- `feature_importance_gradient_boosting_m5.png`

Top importances CSV: `evaluation/reports/m6/feature_importance_top.csv`

---

## 5. Feature importance highlights

### AdaBoost (M4) — top signals
1. `tfidf_artificial intelligence`  
2. `tfidf_image`  
3. `abstract_length`  
4. `tfidf_chatgpt`  
5. `tfidf_approaches`  

### Extra Trees (M5 tuned) — top signals
1. `tfidf_image`  
2. `tfidf_finally`  
3. `tfidf_attention`  
4. `tfidf_existing`  
5. `tfidf_machine learning`  

Interpretation: topical/abstract language features dominate OA-category prediction; `abstract_length` is the strongest structural cue in AdaBoost.

Importances for Extra Trees M5 were also written to SQLite `feature_importance` (run_id `m6_feature_importance_extra_trees_m5`).

---

## 6. How to reproduce

```powershell
cd E:\Tuned_Research
# requires M4 + M5 artifacts
python scripts/phase2/06_comparative_visualizations.py
```

---

## 7. DA2 rubric mapping (M6)

| Requirement | M6 contribution |
|-------------|-----------------|
| 5. Comparative performance analysis (1 mark) | Metrics table across 8 model variants |
| 6. Comparative visualizations (1 mark) | ROC, PR, CM, feature importance, grouped bars |
| Database | Feature importance logged for M5 Extra Trees |

---

## 8. Conclusion

M6 completes the comparative analysis layer of ResearchPilot Phase 2: evaluators can inspect ROC/PR behavior by class, confusion patterns, and which text/metadata features drive the leading models. Together with M4–M5, the predictive OA-category pipeline is fully documented for demonstration.
