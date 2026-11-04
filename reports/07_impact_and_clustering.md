# Phase 2 / M7 — Impact-Tier Classification & Topic Clustering

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M7 — Secondary research-intelligence track  
**Date:** 2026-08-15  
**Status:** Frozen  
**Depends on:** M3 impact matrices (`X_impact_scaled.csv`, `y_impact_final.csv`)

M7 does **not** retrain the OA-category (M4–M6) models and does **not** change Phase 1 data.

**Reproduce:**

```bash
python scripts/phase2/07_impact_and_clustering.py
```

---

## 1. M7 Objective

Extend ResearchPilot beyond open-access prediction with two complementary components:

1. **Impact-tier classification** — predict relative citation impact (`low` / `medium` / `high`) from leakage-safe text and metadata features  
2. **Research-topic clustering** — unsupervised K-Means on TF-IDF features to support similar-paper discovery and future research-gap analysis  

---

## 2. Impact-tier target definition

| Item | Value |
|------|--------|
| Source column | `citation_per_year` |
| Construction | Train-only tertile thresholds applied to all splits |
| Rule | `low` if ≤ q_low; `medium` if ≤ q_high; else `high` |
| Train thresholds | q_low ≈ **87.33**, q_high ≈ **134.0** (`data/ml/label_maps.json`) |
| Threshold type | **Relative** to the train split (quantiles ≈ 0.333 / 0.667) |
| Classes | `low`, `medium`, `high` |

Implementation: `assign_impact_tier()` in `src/researchpilot/features/m2_pipeline.py`. Thresholds are fit on **train only** and then applied to validation and test, so test labels do not influence cutpoints.

---

## 3. Impact-tier class distribution

| Impact tier | Count | Percentage |
|-------------|------:|----------:|
| low | 669 | 33.45% |
| medium | 657 | 32.85% |
| high | 674 | 33.70% |

Classes are approximately balanced (~33% each). Stratified M2 splits preserve tier balance. Macro-F1 is the primary selection metric. Several tree/linear models use `class_weight="balanced"` (or equivalent); AdaBoost does not reweight classes and remains appropriate given the near-equal prior.

Figure: `reports/figures/m7/m7_impact_class_distribution.png`

---

## 4. Feature matrix

| Item | Value |
|------|--------|
| Matrix | `data/ml/m3/X_impact_scaled.csv` |
| Labels | `data/ml/m3/y_impact_final.csv` |
| Features | **112** (structural / engineered numerical + OA one-hots + TF-IDF) |
| Excluded from inputs | `cited_by_count`, `citation_per_year`, `citation_log`, `impact_tier` |

Feature groups used as predictors:

- Engineered numerical (e.g., `paper_age`, `publication_year`, lengths, counts)  
- Structural metadata (e.g., OA category one-hots, `is_open_access`, `has_fulltext`)  
- TF-IDF textual features (`tfidf_*`)  

---

## 5. Train / validation / test split

| Split | n |
|-------|--:|
| Train | 1399 |
| Validation | 300 |
| Test | 301 |

Pipeline order:

```
Training data
    → train CV scores (reporting only)
    → Validation metrics → model selection (champion)
    → Untouched test set → final M7 metrics (once)
```

Verified:

- Target thresholds: train-only  
- Feature construction / scaling: train-fit (M2/M3)  
- Champion selection: **validation macro-F1**  
- Test set: not used for thresholds, selection, or tuning  

---

## 6. Models evaluated

12 models (config: `configs/phase2/m7_secondary.yaml`):

dummy, logistic_regression, gaussian_nb, knn, decision_tree, random_forest, extra_trees, adaboost, gradient_boosting, **xgboost**, lightgbm, mlp  

| Attempted | Successful | Failed |
|----------:|----------:|-------:|
| 12 | **12** | 0 |

Code: `src/researchpilot/models/m7_secondary.py`  
Factory: `src/researchpilot/models/registry.py`

---

## 7. Model performance

Full leaderboard: `evaluation/reports/m7/impact_leaderboard.csv`

| Model | Val macro-F1 | Test Acc | Test macro-F1 | Test ROC-AUC |
|-------|-------------:|---------:|--------------:|-------------:|
| **adaboost** | **0.600** | 0.551 | 0.543 | 0.680 |
| mlp | 0.589 | 0.512 | 0.498 | 0.715 |
| gradient_boosting | 0.567 | 0.502 | 0.498 | 0.689 |
| knn | 0.565 | 0.518 | 0.512 | 0.682 |
| random_forest | 0.562 | 0.555 | 0.552 | 0.728 |
| extra_trees | 0.556 | 0.538 | 0.536 | 0.715 |
| logistic_regression | 0.555 | 0.528 | 0.519 | 0.711 |
| xgboost | 0.552 | 0.532 | 0.530 | 0.711 |
| lightgbm | 0.539 | 0.542 | 0.541 | 0.711 |
| gaussian_nb | 0.453 | 0.455 | 0.403 | 0.655 |
| decision_tree | 0.449 | 0.455 | 0.456 | 0.615 |
| dummy | 0.179 | 0.306 | 0.156 | 0.500 |

Figure: `reports/figures/m7/m7_impact_model_comparison.png`

---

## 8. Validation champion

**AdaBoost** — highest validation macro-F1 (**0.5999**).

Note: Random Forest has a slightly higher *test* macro-F1 / ROC-AUC, but champion selection remains validation-based for honesty. Both are reported; selection is not rewritten post hoc.

---

## 9. Final test performance

| Metric | Value |
|--------|------:|
| Test accuracy | 0.5515 |
| Test macro-F1 | 0.5425 |
| Test ROC-AUC (OvR) | 0.6798 |

**Interpretation:** The impact-tier classification task achieved moderate predictive performance. AdaBoost was selected as the validation champion, with a test macro-F1 of approximately 0.543 and ROC-AUC of approximately 0.680. This indicates that the available text and metadata features contain useful information for relative impact profiling, although the three impact tiers remain only partially separable.

---

## 10. Confusion matrix interpretation

Figure: `reports/figures/m7/m7_impact_confusion_matrix.png`

Off-diagonal mass is concentrated between adjacent tiers (low↔medium, medium↔high), which is expected for an ordinal citation construct discretized into tertiles. Extreme confusions (low↔high) are less frequent than adjacent-tier errors, consistent with partial but imperfect separability.

---

## 11. ROC / Precision-Recall interpretation

Figures:

- `reports/figures/m7/m7_impact_roc_auc.png`  
- `reports/figures/m7/m7_impact_precision_recall.png`  

Macro ROC-AUC ≈ 0.68 exceeds chance (0.50) and indicates useful ranking discrimination across one-vs-rest tier probabilities. Precision-recall curves show moderate average precision; medium-tier discrimination is typically the hardest of the three classes.

---

## 12. Feature importance

Champion: AdaBoost (`feature_importances_`).

Figure: `reports/figures/m7/m7_impact_feature_importance.png`  
Table: `evaluation/reports/m7/impact_feature_importance_champion.csv`

Top contributors include:

| Feature | Group | Importance |
|---------|-------|----------:|
| paper_age | engineered numerical | 0.479 |
| recent_paper | engineered numerical | 0.161 |
| publication_year | engineered numerical | 0.130 |
| tfidf_based | TF-IDF textual | 0.042 |
| tfidf_comprehensive | TF-IDF textual | 0.029 |

Aggregated importance share (AdaBoost):

| Group | Share |
|-------|------:|
| Engineered numerical | ~0.79 |
| TF-IDF textual | ~0.21 |
| Structural metadata | ~0.00 |

Engineered temporal/length features dominate AdaBoost’s importance mass; TF-IDF terms contribute strongly among the remaining signal. These features contributed strongly to the model’s predictions. This does **not** imply that they cause high research impact.

---

## 13. XGBoost issue and resolution

**Prior failure (10:07 run):**

```text
Check failed: err == cudaGetLastError() (0 vs. 2)
```

(`cudaErrorMemoryAllocation` during array-interface GPU path on Windows.)

**Resolution (kept algorithm unchanged):**

- `CUDA_VISIBLE_DEVICES=-1` at M7 entrypoint and in the XGBoost factory  
- `device="cpu"`, `tree_method="hist"`, `n_jobs=1`  
- Fit/CV on `float32` NumPy arrays; serial CV for XGBoost  

**Final status:** `impact/xgboost` → **SUCCESS** (status=ok; val F1 ≈ 0.552, test F1 ≈ 0.530).

---

## 14. Leakage verification

Checked `X_impact_scaled.csv` feature columns:

- `cited_by_count` — **absent**  
- `citation_per_year` — **absent**  
- `citation_log` — **absent**  
- other `citation*` / `impact_tier` — **absent**  

Runtime loader also drops any residual citation-prefixed columns. **No leakage found; no feature-matrix change required.**

Target construction still uses citation information (as designed); that information is labels only, not predictors.

---

## 15. Topic clustering methodology

| Item | Value |
|------|--------|
| Algorithm | K-Means (`n_init=10`, `random_state=42`) |
| Representation | 100 TF-IDF columns only |
| Fit scope | Train rows; assign clusters to all papers |
| Visualization | PCA-2D of TF-IDF space |

---

## 16. k-selection

k searched over **3–8**; selected by **maximum train silhouette**.

---

## 17. Silhouette score

| Item | Value |
|------|------:|
| Best k | **8** |
| Train silhouette | **0.0645** |

Figures: `m7_silhouette_vs_k.png`, `m7_topic_clusters_pca.png`

The clustering experiment identified eight groups, but the low silhouette score indicates substantial overlap between research-topic clusters. Therefore, the clustering is treated as exploratory/supporting analysis rather than a strongly separated topic segmentation.

---

## 18. Cluster sizes

| Cluster | n | % of dataset |
|--------:|--:|-------------:|
| 0 | 79 | 3.95 |
| 1 | 272 | 13.60 |
| 2 | 623 | 31.15 |
| 3 | 82 | 4.10 |
| 4 | 321 | 16.05 |
| 5 | 182 | 9.10 |
| 6 | 237 | 11.85 |
| 7 | 204 | 10.20 |

Cluster 2 is a large mixed “general AI/CS” mass; smaller clusters (0, 3) are more distinctive but still overlap in PCA space.

---

## 19. Cluster interpretation

Derived from keywords, concepts, TF-IDF centroids, and sample titles (no invented topic labels). Artifacts: `evaluation/reports/m7/cluster_interpretation.json`.

| ID | n | Distinctive signals (keywords / TF-IDF / titles) |
|---:|--:|--------------------------------------------------|
| 0 | 79 | control / systems / robots (traffic control, soft robots) |
| 1 | 272 | generative AI, education, human/AI research discourse |
| 2 | 623 | broad data/research/analysis mass (heterogeneous) |
| 3 | 82 | ChatGPT / education / assessment (USMLE, higher-ed) |
| 4 | 321 | image / detection / segmentation / computer vision |
| 5 | 182 | language models / neural networks / NLP |
| 6 | 237 | machine learning / prediction / ML methods |
| 7 | 204 | deep learning / detection / XAI-style applications |

---

## 20. Limitations

- Impact tiers are corpus-internal relative tertiles, not absolute scientific prestige.  
- Temporal features (`paper_age`, `recent_paper`, `publication_year`) dominate AdaBoost importance; they are allowed predictors but are correlated with citation accumulation dynamics.  
- Moderate test F1 (~0.54) means tiers are only partially separable from non-citation features.  
- Clustering silhouette (~0.064) indicates weak geometric separation; clusters are exploratory.  
- OpenAlex AI/ML sampling bias limits external generalization.  

---

## 21. ResearchPilot relevance

M7 supplies secondary intelligence for:

- relative impact profiling of papers  
- exploratory topic neighborhoods for similar-paper discovery  
- scaffolding for Phase 3 research-gap / RAG narrative  

It does not replace the primary OA-category track (M4–M6).

---

## 22. Final M7 conclusion

Impact-tier classification was successfully implemented. Twelve ML models were compared; AdaBoost was the validation champion. Final test performance is reported honestly (macro-F1 ≈ 0.543; ROC-AUC ≈ 0.680): useful but imperfect impact-tier prediction. Topic clustering identified eight exploratory groups; the low silhouette score means topic boundaries overlap, so clustering serves as supporting research-discovery functionality. Together, M7 provides the foundation for future ResearchPilot capabilities such as similar-paper discovery and research-gap analysis.

---

## Reproducibility

| Item | Location |
|------|----------|
| Command | `python scripts/phase2/07_impact_and_clustering.py` |
| Config | `configs/phase2/m7_secondary.yaml` |
| Inputs | `data/ml/m3/X_impact_scaled.csv`, `data/ml/m3/y_impact_final.csv`, `data/final/final_dataset.csv` |
| Metrics / preds | `evaluation/reports/m7/` |
| Checkpoints | `models/checkpoints/m7/` |
| Figures | `reports/figures/m7/` |
| Reports | `DA2/M7_Impact_and_Clustering_Report.md`, `reports/phase2_m7_impact_and_clustering_report.md` |

### Final verified console snapshot

```text
Models attempted: 12
Models OK      : 12
Models failed  : 0
XGBoost status : ok
Champion (val) : adaboost
Val macro-F1   : 0.5998843880286179
Test macro-F1  : 0.5425104705592511
Test accuracy  : 0.5514950166112956
Test ROC-AUC   : 0.6798328093698903
Best k         : 8
Silhouette     : 0.06447164833411453
Status          : OK - M7 complete.
```
