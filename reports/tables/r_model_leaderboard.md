# ResearchPilot DA2 — R Model Leaderboard

> **Task**: OA Category Classification (fully_open / partially_open / closed)  
> **Test set**:  papers  |  **Metrics**: macro-averaged  |  **Date**: 2026-09-19

| Rank | Model | Type | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|------|-------|------|----------|-----------|--------|----------|---------|
| 1 | `mlp` | baseline | 0.4120 | 0.3792 | 0.3794 | **0.3783** | 0.5429 |
| 2 | `logistic_regression` | baseline | 0.4452 | 0.3875 | 0.3764 | **0.3345** | 0.5708 |
| 3 | `lda` | baseline | 0.4419 | 0.3565 | 0.3718 | **0.3257** | 0.5714 |
| 4 | `random_forest_tuned` | tuned | 0.3754 | — | — | **0.3251** | — |
| 5 | `gradient_boosting` | baseline | 0.3721 | 0.3407 | 0.3312 | **0.3233** | 0.5299 |
| 6 | `decision_tree` | baseline | 0.3787 | 0.3257 | 0.3323 | **0.3194** | 0.5273 |
| 7 | `random_forest` | baseline | 0.3654 | 0.3306 | 0.3262 | **0.3187** | 0.5040 |
| 8 | `xgboost_tuned` | tuned | 0.3189 | — | — | **0.3159** | — |
| 9 | `xgboost` | baseline | 0.2924 | 0.2957 | 0.3015 | **0.2909** | 0.4999 |
| 10 | `elastic_net` | baseline | 0.4452 | 0.2937 | 0.3648 | **0.2840** | 0.5582 |
| 11 | `svm_linear` | baseline | 0.4219 | 0.2885 | 0.3453 | **0.2661** | 0.5499 |
| 12 | `svm_tuned` | tuned | 0.4086 | — | — | **0.2510** | — |
| 13 | `svm_rbf` | baseline | 0.4086 | 0.3418 | 0.3338 | **0.2439** | 0.5357 |
| 14 | `naive_bayes` | baseline | 0.3920 | 0.2590 | 0.3190 | **0.2342** | 0.5396 |

## Notes
- All metrics computed on the **held-out test set** (15% of data).
- Precision / Recall / F1 are **macro-averaged** across all 3 classes.
- ROC-AUC uses **one-vs-rest (OVR)** macro averaging.
- `—` = metric not computed for tuned model variants.

## Python Baseline Reference
| Metric | Python M4/M5 | Note |
|--------|-------------|------|
| Best Accuracy | 0.4817 (AdaBoost) | R uses different random split |
| Best Macro-F1 | 0.4517 (AdaBoost) | Expected range: 0.40–0.50 |
| Best ROC-AUC  | 0.6278 (Extra Trees tuned) | Expected range: 0.55–0.65 |

*Generated: 2026-09-19 18:35:46.513384*
