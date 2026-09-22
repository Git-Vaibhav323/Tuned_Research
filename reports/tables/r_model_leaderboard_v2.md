# ResearchPilot DA2 — R Model Leaderboard v2 (Enriched Features)

> **Date**: 2026-09-22  |  **Features**: 80 enriched (TF-IDF + domain + publisher + structural)
> **Task**: OA Category Classification (fully_open / partially_open / closed)

| Rank | Model | Type | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|------|-------|------|----------|-----------|--------|----------|---------|
| 1 | `svm_tuned` | tuned | **0.5748** | — | — | 0.5701 | — |
| 2 | `lda` | baseline | **0.5748** | 0.5700 | 0.5694 | 0.5696 | 0.7577 |
| 3 | `mlp` | baseline | **0.5681** | 0.5650 | 0.5675 | 0.5657 | 0.7659 |
| 4 | `svm_rbf` | baseline | **0.5681** | 0.5597 | 0.5590 | 0.5559 | 0.7574 |
| 5 | `logistic_regression` | baseline | **0.5648** | 0.5602 | 0.5668 | 0.5631 | 0.7565 |
| 6 | `random_forest_tuned` | tuned | **0.5615** | — | — | 0.5475 | — |
| 7 | `elastic_net` | baseline | **0.5581** | 0.5559 | 0.5522 | 0.5539 | 0.7584 |
| 8 | `svm_linear` | baseline | **0.5515** | 0.5445 | 0.5438 | 0.5441 | 0.7587 |
| 9 | `random_forest` | baseline | **0.5482** | 0.5448 | 0.5296 | 0.5317 | 0.7572 |
| 10 | `gradient_boosting` | baseline | **0.5382** | 0.5415 | 0.5359 | 0.5384 | 0.7402 |
| 11 | `adaboost` | baseline | **0.5150** | 0.5126 | 0.5121 | 0.5123 | 0.6963 |
| 12 | `decision_tree` | baseline | **0.5083** | 0.5295 | 0.4962 | 0.5017 | 0.6924 |
| 13 | `naive_bayes` | baseline | **0.4186** | 0.4742 | 0.4909 | 0.4081 | 0.6840 |
| 14 | `xgboost_tuned` | tuned | **0.3555** | — | — | 0.3537 | — |
| 15 | `xgboost` | baseline | **0.3455** | 0.3464 | 0.3530 | 0.3428 | 0.5243 |

## Improvement Summary

| Metric | Original (9 features) | Enriched (80 features) | Improvement |
|--------|----------------------|----------------------|-------------|
| Accuracy | 0.4452 | 0.5748 | +0.1296 (+13.0%) |
| Macro-F1 | 0.3783 | 0.5701 | +0.1918 (+19.2%) |

## Key Insight
Publisher DOI prefix and domain binary features are the strongest predictors.
IEEE-published papers are predominantly CLOSED; MDPI/BMC are FULLY OPEN.
Medical domain papers skew FULLY OPEN (NIH/Wellcome Trust mandate).

*Generated: 2026-09-22 20:17:21.148268*
