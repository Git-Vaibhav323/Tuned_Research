# M6 Final QA Report — ResearchPilot Phase 2

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M6 (Comparative Visualizations) — final quality assurance  
**Date:** 2026-08-15  
**Scope:** Documentation / verification only (no M7 work)

---

## 1. M6 status

| Item | Status |
|------|--------|
| Comparative metrics CSV/MD | Present |
| Confusion matrices | Present |
| OvR ROC + ROC-AUC comparison | Present |
| Precision–Recall + AP comparison | Present |
| Grouped metric bars | Present |
| Feature-importance plots + CSV | Present |
| SQLite feature-importance log | Present |
| Reproducible script | `scripts/phase2/06_comparative_visualizations.py` |
| **Final QA** | **Complete** |
| **M6 freeze** | **Yes — ready for M7** |

Primary narrative report (updated): `reports/phase2_m6_comparative_analysis_report.md`  
DA2 copy: `DA2/M6_Comparative_Analysis_Report.md`

---

## 2. Data leakage verification

Code inspected:

- `src/researchpilot/features/m2_pipeline.py` — TF-IDF and impact tertiles fit on **train**  
- `src/researchpilot/features/m3_selection.py` — selection / scaler / encoders fit on **train**  
- `src/researchpilot/models/train_eval.py` — M4 training on train; CV on train; test used for evaluation outputs  
- `src/researchpilot/models/tuning.py` — RandomizedSearchCV on **train**; champion by **validation** macro-F1; test scored after selection  

### Findings

| Risk | Result |
|------|--------|
| Test used in feature selection | **Not found** |
| Test used in hyperparameter tuning | **Not found** |
| Test used to choose M5 champion | **Not found** (val F1) |
| Test used for final M6 metrics/plots | **Yes — intended** |

**Conclusion:** The intended train → CV/val → tune → select → untouched test → M6 evaluation workflow is in place for M5 model selection and for all preprocessing fits. **No genuine leakage issue requiring pipeline changes or re-runs was found.**

### Documentation nuance (not a retrain trigger)

M4’s summary field `champion` ranks default models by **test** macro-F1 for leaderboard labeling. This does not alter model parameters and does not drive M5 selection. It is recorded here for methodological transparency.

**Pipeline code was not modified** during this QA.

---

## 3. Model-selection verification

| Selection | Criterion | Data used |
|-----------|-----------|-----------|
| M5 champion | macro-F1 | **Validation** |
| M5 hyperparameters | RandomizedSearchCV `f1_macro` | **Train CV** |
| M6 visualization set | Fixed config list | Predictions already scored on test |

Confirmed in `tuning.py`: `tuned_results.sort(key=lambda r: r["val"]["f1_macro"], reverse=True)`.

---

## 4. Final comparative metrics (unchanged)

| Model | Accuracy | F1 Macro | ROC-AUC | Average Precision |
|-------|----------|----------|---------|-------------------|
| m4_adaboost | 0.482 | 0.452 | 0.610 | 0.435 |
| m5_adaboost_tuned | 0.482 | 0.452 | 0.610 | 0.435 |
| m5_extra_trees_tuned | 0.449 | 0.439 | 0.628 | 0.454 |
| m4_extra_trees | 0.465 | 0.436 | 0.623 | 0.454 |
| m5_logistic_regression_tuned | 0.449 | 0.433 | 0.621 | 0.453 |
| m4_gradient_boosting | 0.462 | 0.433 | 0.597 | 0.425 |
| m5_random_forest_tuned | 0.445 | 0.429 | 0.624 | 0.447 |
| m5_gradient_boosting_tuned | 0.458 | 0.428 | 0.595 | 0.425 |

These values match `evaluation/reports/m6/comparative_metrics.csv`. **No metrics were altered** in this QA.

---

## 5. Best model by each metric

| Metric | Winner | Value |
|--------|--------|------:|
| Accuracy | m4_adaboost | 0.482 |
| Macro-F1 | m4_adaboost | 0.452 |
| ROC-AUC | m5_extra_trees_tuned | 0.628 |
| Average Precision | m5_extra_trees_tuned **and** m4_extra_trees (tie) | 0.454 |

**Statement used in the M6 report:**  
AdaBoost achieved the highest accuracy and macro-F1, while tuned Extra Trees achieved the highest ROC-AUC and tied for the highest average precision. Therefore, no single model dominates across every evaluation metric.

---

## 6. Confusion-matrix findings

Verified from test predictions (e.g. `m4_adaboost`):

- Largest pairwise error mass: **fully_open ↔ partially_open** (84 confusions).  
- Smaller pairwise masses involving closed: fully↔closed 32; partially↔closed 40.  

**Supported claim:** main difficulty is between `fully_open` and `partially_open`.  

**Not supported as a blanket claim:** “closed is comparatively easier” — AdaBoost closed recall ≈ 0.30 is lower than fully_open recall ≈ 0.64. The updated M6 report states the fo↔po pattern with this caveat.

---

## 7. ROC findings

- Per-class OvR ROC figures present for all three classes.  
- Highest macro ROC-AUC among compared models: **m5_extra_trees_tuned (0.628)**.  
- Values above 0.5 indicate better-than-chance ranking despite limited hard accuracy.

---

## 8. Precision–Recall findings

- Per-class PR figures present.  
- Highest macro average precision: **0.454** (tie: `m5_extra_trees_tuned`, `m4_extra_trees`).  
- Complements accuracy/F1 for imbalanced multiclass evaluation.

---

## 9. Feature-importance findings

AdaBoost M4 top features are dominated by TF-IDF terms plus `abstract_length`.  
Extra Trees M5 top features are dominated by TF-IDF terms.

Documented interpretation:

- Textual/topical features dominate; `abstract_length` is an important structural cue for AdaBoost.  
- Importance ≠ causal determination of open-access status.

---

## 10. Limitations / interpretation

1. Absolute test accuracy (~0.45–0.48) is modest; the task is hard with current features.  
2. Classes are not cleanly separable; fo↔po confusion dominates.  
3. Metric-dependent “winners” — no universal best model.  
4. M6 is visualization/reporting only; it does not improve underlying predictors.  
5. Small test set (n=301) implies metric variance; comparisons should stay descriptive.

---

## 11. DA2 comparative-analysis confirmation

| DA2 expectation | Evidence |
|-----------------|----------|
| Comparative metrics across models | `comparative_metrics.csv` + report tables |
| Confusion matrices | `cm_*.png`, `cm_grid_comparative.png` |
| ROC curves | `roc_ovr_*.png`, `roc_auc_comparison.png` |
| Precision–Recall | `pr_*.png`, `pr_ap_comparison.png` |
| Feature importance | `feature_importance_*.png` + CSV + SQLite |
| Honest interpretation | Updated M6 report + this QA doc |

**Confirmation:** M6 satisfies the DA2 comparative performance analysis and comparative visualization requirements for Phase 2.

---

## 12. QA actions taken

| Action | Done? |
|--------|-------|
| Leakage / selection code review | Yes |
| Pipeline code changes | **No** (none required) |
| Model retraining | **No** |
| Metric changes | **No** |
| Figure regeneration | **No** (all 22 expected figures present, non-empty) |
| M6 report wording update | Yes |
| This QA report created | Yes |
| DA2 pack sync | Yes |

---

## 13. Final statement

**M6 is frozen and ready. Proceed to M7.**
