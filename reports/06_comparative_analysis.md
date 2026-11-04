# Phase 2 / M6 — Comparative Visualizations Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M6 — Comparative performance visualizations  
**Date:** 2026-08-15 (QA documentation refresh)  
**Status:** Complete / frozen after QA  
**Depends on:** M4 prediction CSVs + M5 tuned predictions/checkpoints  

M6 does **not** retrain models; it consumes existing M4/M5 artifacts. Test-set metrics below are unchanged from the verified M6 comparative table.

---

## 1. Objective

Produce the DA2 comparative visualization pack for open-access category classification:

1. Confusion matrices  
2. ROC curves (one-vs-rest) + macro ROC-AUC comparison  
3. Precision–Recall curves + macro Average Precision comparison  
4. Grouped metric bars  
5. Feature importance charts  

---

## 2. Evaluation protocol (test set untouched until final scoring)

Verified workflow:

```text
Train split
  → feature selection / TF-IDF / scalers fit (M2–M3; train only)
  → model training + CV on train (M4)
  → hyperparameter search on train CV (M5)
  → M5 champion selection by validation macro-F1
  → final scoring on held-out test
  → M6 plots from saved test predictions
```

| Stage | Uses test? |
|-------|------------|
| M2 TF-IDF / impact thresholds | No (train fit) |
| M3 feature selection / scaling / label encoders | No (train fit) |
| M5 RandomizedSearchCV | No (train CV) |
| M5 champion selection | No (**validation** macro-F1) |
| M4/M5 reported test metrics & M6 figures | Yes (final evaluation only) |

**Note:** M4’s summary string `champion` ranks default models by test macro-F1 for leaderboard labeling only. That ranking did **not** change fitted parameters, did **not** guide M5 tuning, and did **not** select the M5 champion. All M4 models used fixed default hyperparameters trained on the train split.

---

## 3. Models visualized (8)

| Label | Source |
|-------|--------|
| `m4_adaboost` | M4 default (highest test accuracy / macro-F1) |
| `m4_extra_trees` | M4 ensemble |
| `m4_gradient_boosting` | M4 ensemble |
| `m5_extra_trees_tuned` | M5 val-selected champion |
| `m5_adaboost_tuned` | M5 tuned AdaBoost |
| `m5_logistic_regression_tuned` | M5 tuned LogReg |
| `m5_random_forest_tuned` | M5 tuned RF |
| `m5_gradient_boosting_tuned` | M5 tuned GB |

Config: `configs/phase2/comparative.yaml`  
Entrypoint: `scripts/phase2/06_comparative_visualizations.py`  
Code: `src/researchpilot/evaluation/comparative_plots.py`

---

## 4. Comparative metrics (test set)

| Model | Accuracy | F1 macro | ROC-AUC | Avg Precision |
|-------|----------|----------|---------|---------------|
| m4_adaboost | 0.482 | 0.452 | 0.610 | 0.435 |
| m5_adaboost_tuned | 0.482 | 0.452 | 0.610 | 0.435 |
| m5_extra_trees_tuned | 0.449 | 0.439 | 0.628 | 0.454 |
| m4_extra_trees | 0.465 | 0.436 | 0.623 | 0.454 |
| m5_logistic_regression_tuned | 0.449 | 0.433 | 0.621 | 0.453 |
| m4_gradient_boosting | 0.462 | 0.433 | 0.597 | 0.425 |
| m5_random_forest_tuned | 0.445 | 0.429 | 0.624 | 0.447 |
| m5_gradient_boosting_tuned | 0.458 | 0.428 | 0.595 | 0.425 |

Source: `evaluation/reports/m6/comparative_metrics.csv`

### Best model by metric (no universal winner)

| Criterion | Model | Value |
|-----------|--------|-------|
| Best accuracy | `m4_adaboost` | 0.482 |
| Best macro-F1 | `m4_adaboost` | 0.452 |
| Best ROC-AUC | `m5_extra_trees_tuned` | 0.628 |
| Best average precision | `m5_extra_trees_tuned` / `m4_extra_trees` (tie) | 0.454 |

**Interpretation:** AdaBoost achieved the highest accuracy and macro-F1, while tuned Extra Trees achieved the highest ROC-AUC and tied for the highest average precision. Therefore, no single model dominates across every evaluation metric.

---

## 5. Interpretation of accuracy level (~0.445–0.482)

Test accuracy for the compared models falls approximately in **0.445–0.482**. This should be read as an empirical finding about task difficulty, not as evidence of a highly accurate classifier.

Supporting points:

- Predicting `oa_category` (`fully_open` / `partially_open` / `closed`) from the available metadata and title/abstract TF-IDF features is inherently difficult.  
- The three classes are not cleanly separable in this feature space (see confusion matrices).  
- Majority-class baseline accuracy is about **0.42** (`fully_open`); AdaBoost’s 0.482 is above that baseline but still modest in absolute terms.  
- Macro-F1, ROC-AUC, and average precision provide complementary information beyond raw accuracy and should be reported alongside it.  

No post-hoc metric inflation or undisclosed re-tuning was applied for this QA pass.

---

## 6. Confusion-matrix findings

Figures: `reports/figures/m6/cm_*.png`, `cm_grid_comparative.png`

For `m4_adaboost` on the test set (labels order: fully_open, partially_open, closed):

```text
[[81 32 13]
 [52 44 13]
 [19 27 20]]
```

Pairwise off-diagonal confusions:

| Pair | Count |
|------|------:|
| fully_open ↔ partially_open | 84 |
| partially_open ↔ closed | 40 |
| fully_open ↔ closed | 32 |

**Documented finding:** The main classification difficulty occurs between `fully_open` and `partially_open` (largest pairwise error mass). This pattern also appears for Extra Trees variants.

**Caveat:** `closed` is not uniformly “easy.” For AdaBoost, closed recall (≈0.30) is lower than fully_open recall (≈0.64). The dominant boundary error is still fully_open ↔ partially_open, consistent with those categories being adjacent OA states, while closed remains a non-trivial minority class.

---

## 7. ROC and Precision–Recall findings

- OvR ROC curves: `roc_ovr_fully_open.png`, `roc_ovr_partially_open.png`, `roc_ovr_closed.png`  
- Macro ROC-AUC bar chart: `roc_auc_comparison.png` — highest among compared models: **m5_extra_trees_tuned (0.628)**  
- PR curves: `pr_fully_open.png`, `pr_partially_open.png`, `pr_closed.png`  
- Macro AP chart: `pr_ap_comparison.png` — highest: **0.454** (`m5_extra_trees_tuned` and `m4_extra_trees`)  

ROC-AUC / AP above ~0.5 indicate ranking quality beyond chance even when hard accuracy remains limited.

---

## 8. Feature-importance findings

Figures: `feature_importance_*.png`  
Table: `evaluation/reports/m6/feature_importance_top.csv`

### AdaBoost (M4) — top 5
1. `tfidf_artificial intelligence`  
2. `tfidf_image`  
3. `abstract_length`  
4. `tfidf_chatgpt`  
5. `tfidf_approaches`  

### Extra Trees (M5 tuned) — top 5
1. `tfidf_image`  
2. `tfidf_finally`  
3. `tfidf_attention`  
4. `tfidf_existing`  
5. `tfidf_machine learning`  

**Interpretation:** Textual and topical features dominate the feature-importance rankings for the current OA-category classification task, while `abstract_length` is also an important structural feature for AdaBoost.

**Non-causal clarification:** Feature importance indicates predictive contribution within the trained model; it does not imply that these features causally determine open-access status.

SQLite: Extra Trees M5 importances logged under run_id `m6_feature_importance_extra_trees_m5`.

---

## 9. Figure catalog (verified present)

All under `reports/figures/m6/` (QA check: **0 missing**):

| Category | Files |
|----------|--------|
| Confusion matrices | per-model `cm_m4_*.png` / `cm_m5_*.png`, `cm_grid_comparative.png` |
| ROC | `roc_ovr_fully_open.png`, `roc_ovr_partially_open.png`, `roc_ovr_closed.png`, `roc_auc_comparison.png` |
| Precision–Recall | `pr_fully_open.png`, `pr_partially_open.png`, `pr_closed.png`, `pr_ap_comparison.png` |
| Metrics | `metrics_grouped_bars.png` |
| Feature importance | `feature_importance_adaboost_m4.png`, `feature_importance_extra_trees_m5.png`, `feature_importance_random_forest_m5.png`, `feature_importance_gradient_boosting_m5.png` |

---

## 10. How to reproduce (plots only)

```powershell
cd E:\Tuned_Research
python scripts/phase2/06_comparative_visualizations.py
```

Requires existing M4/M5 prediction CSVs and checkpoints; does not retrain.

---

## 11. DA2 rubric mapping

| Requirement | Status |
|-------------|--------|
| Comparative performance analysis | Satisfied (multi-model metrics table) |
| Comparative visualizations (ROC, PR, CM, importance, etc.) | Satisfied (full figure pack) |

Related QA write-up: `reports/m6_final_qa.md`

---

## 12. Conclusion

M6 is complete and documentation-frozen after QA. Comparative evidence shows metric-dependent leaders (AdaBoost vs Extra Trees), modest absolute accuracy reflecting task difficulty, and a clear fully_open ↔ partially_open confusion pattern, with feature importance dominated by textual signals under a non-causal reading.
