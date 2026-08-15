# ResearchPilot — DA2 Final Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Course deliverable:** DA2 (Phase 1 + Phase 2)  
**Date:** 2026-08-15  
**Status:** Phase 2 complete through M8 documentation  

Architecture figure: `reports/figures/final/researchpilot_final_architecture.png`

---

## 1. Introduction

ResearchPilot is a data-science foundation for a research assistant that helps users explore AI/ML scholarly literature. Phase 1 builds a curated OpenAlex dataset with preprocessing and EDA. Phase 2 adds a SQLite database, feature engineering/selection, multi-algorithm machine learning, hyperparameter tuning, comparative evaluation, relative impact-tier prediction, and exploratory topic clustering.

This report consolidates verified Phase 1–2 work for DA2 assessment. It does not claim Phase 3 product features (RAG, chat UI, summarization) as implemented.

---

## 2. Project Objective

Build a reproducible pipeline that:

1. Collects and cleans AI/ML research metadata from OpenAlex  
2. Stores and retrieves records via SQL  
3. Engineers and selects leakage-safe predictive features  
4. Trains and compares 10–15 ML/DL algorithms for open-access category prediction  
5. Tunes strong models and reports honest before/after results  
6. Adds a secondary impact-tier classifier and exploratory topic clusters  

Primary predictive task: **`oa_category`** ∈ {`fully_open`, `partially_open`, `closed`}.  
Secondary task: **`impact_tier`** ∈ {`low`, `medium`, `high`} (relative tertiles).

---

## 3. Phase 1 Summary

| Step | Output |
|------|--------|
| OpenAlex collection | `data/raw/raw_papers.csv` |
| Preprocessing | `data/cleaned/cleaned_papers.csv` |
| Feature extraction | `data/processed/feature_extracted.csv` |
| Feature engineering | `data/final/final_dataset.csv` (**2000** papers, **27** columns) |
| EDA | Python + R figures under `reports/figures/` and `r/visualizations/` |

Phase 1 files are treated as **read-only** inputs to Phase 2.

---

## 4. Phase 2 Architecture

```text
OpenAlex API → Raw Dataset → Preprocessing → Feature Extraction
  → Feature Engineering → R/Python EDA → SQLite
  → Feature Selection → ML Feature Matrix
  → M4 Models → M5 Tuning → M6 Comparative Evaluation
       ├─ OA Classification
       └─ Impact-Tier Prediction → Topic Clustering → Research Insights
```

Milestones: **M1** database · **M2** features · **M3** selection · **M4** training · **M5** tuning · **M6** comparison · **M7** impact/clustering · **M8** integration/docs.

---

## 5. Feature Engineering

### Original / extracted fields (examples)

Title, abstract, publication year, cited_by_count, concepts, keywords, DOI, open-access JSON fields.

### Engineered numerical features

`paper_age`, `title_length`, `abstract_length`, `keyword_count`, `concept_count`, `citation_per_year`, `citation_log`, `recent_paper`, `has_doi`, OA flags/category.

### Text features

Title + abstract concatenated → **TF-IDF** (max 100 features, 1–2 grams, English stop words).

### Categorical / structural

`oa_category` (primary target); for impact track only, OA one-hots and related flags are allowed predictors.

Dictionary: `data/metadata/phase2_feature_dictionary.md`.

---

## 6. Feature Selection

**M3** builds consensus feature shortlists for the OA track (mutual information / model-based votes), drops near-zero variance and redundant constants (e.g., `has_doi`), and fits scalers/encoders on **train only**.

### Leakage prevention (critical)

**OA track excludes:** `is_open_access`, `oa_status`, `oa_url`, `oa_category`, `open_access`, `has_fulltext`, raw identifiers/text held out of the tabular matrix.

**Impact track excludes as inputs:** `cited_by_count`, `citation_per_year`, `citation_log`, `impact_tier`.

Impact thresholds (`q_low≈87.33`, `q_high≈134.0`) are computed from **training** `citation_per_year` tertiles only, then applied to val/test.

Splits: train **1399** / val **300** / test **301** (stratified by `oa_category`).

---

## 7. Database Connectivity

| Item | Value |
|------|--------|
| Technology | **SQLite** |
| File | `database/researchpilot.db` |
| Load | `python scripts/phase2/01_load_to_db.py` |
| Demo | `python scripts/phase2/08_da2_database_demo.py` |

### Tables

`papers` (PK `id`), `ml_features`, `experiment_runs`, `predictions`, `feature_importance`, `load_manifest`.

### Connectivity pattern

```text
Python (sqlite3 / pandas)
  → connect(researchpilot.db)
  → SQL query
  → DataFrame / printed research records
```

Example demo queries (real results available locally): recent papers (year ≥ 2023), top cited papers, OA category aggregates (834 fully_open / 723 partially_open / 443 closed).

R can connect via `RSQLite` to the same file for retrieval/EDA.

---

## 8. ML Model Development

### Distinct algorithms (OA track, M4) — **14**

Dummy, Logistic Regression, Gaussian NB, k-NN, Linear SVC, RBF SVC, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, MLP.

This satisfies the DA2 **10–15 algorithms** requirement. M5 tunes five of these (not counted as new algorithms). M7 reuses twelve algorithms on the impact track. KMeans provides unsupervised clustering support.

### Frozen OA leaders (M4/M6 comparative metrics)

| Leader | Model | Value |
|--------|-------|------:|
| Best Accuracy | m4_adaboost | **0.482** |
| Best Macro-F1 | m4_adaboost | **0.452** |
| Best ROC-AUC | m5_extra_trees_tuned | **0.628** |
| Best Average Precision | m4_linear_svc / strong ensembles ~0.45 | see comparative CSV |

Different metrics identify different leaders; there is **no single universal winner**.

Artifacts: `DA2/m4_leaderboard.csv`, `DA2/m6_comparative_metrics.csv`.

---

## 9. Hyperparameter Tuning

**Method:** `RandomizedSearchCV` · `n_iter=20` · 3-fold CV on **train** · score `f1_macro` · champion by **validation** F1 · test once.

| Model | Selected params (summary) | Val F1 | Test F1 |
|-------|---------------------------|-------:|--------:|
| Extra Trees (M5 val champion) | n_estimators=500, max_depth=16, … | 0.503 | 0.439 |
| Logistic Regression | C=5.0, lbfgs, l2 | 0.477 | 0.433 |
| Random Forest | n_estimators=200, max_depth=8, … | 0.471 | 0.429 |
| AdaBoost | n_estimators=200, lr=0.8 | 0.444 | **0.452** |
| Gradient Boosting | n_estimators=150, lr=0.08, … | 0.442 | 0.428 |

**Honest finding:** M5 Extra Trees won validation selection but did **not** beat M4 AdaBoost on test macro-F1. Tuning improved search/validation behavior for some models; it did not improve every model’s final test score.

---

## 10. Comparative Performance Analysis

Focus models (M6) report Accuracy, Precision, Recall, Macro-F1, ROC-AUC, Average Precision on the **held-out test set**.

Key pattern: open-access category is only moderately predictable from non-OA leakage-safe features. Macro-F1 near **0.45** and ROC-AUC near **0.61–0.63** indicate useful but imperfect signal.

---

## 11. Comparative Visualizations

Required packs exist under `reports/figures/m6/` and `reports/figures/m7/`:

- Confusion matrices, ROC OvR, Precision–Recall  
- Model comparison bars  
- Feature importance  
- Impact-tier CM / comparison / importance  
- Topic PCA clusters + silhouette vs k  
- Class distribution (impact)  

Final architecture: `reports/figures/final/researchpilot_final_architecture.png`.

---

## 12. Impact-Tier Prediction (M7, frozen)

| Item | Value |
|------|--------|
| Target | train-only tertiles of `citation_per_year` |
| Thresholds | q_low≈**87.33**, q_high≈**134.0** |
| Classes | low 669 (33.5%), medium 657 (32.9%), high 674 (33.7%) |
| Models | **12/12** OK (XGBoost CPU-fixed) |
| Champion | **AdaBoost** (val F1 **0.600**) |
| Test | Acc **0.551**, macro-F1 **0.543**, ROC-AUC **0.680** |

Performance is **moderate**: features help relative impact profiling, but tiers remain only partially separable. Temporal engineered features contribute strongly to AdaBoost importance; this is association, not causation.

---

## 13. Topic Clustering

| Item | Value |
|------|--------|
| Representation | TF-IDF (100 dims) |
| Method | KMeans, k ∈ [3,8], select max train silhouette |
| Best k | **8** |
| Silhouette | **0.064** |

The clustering experiment is exploratory because the low silhouette score indicates substantial overlap between topic groups. It can still support future similar-paper discovery, topic exploration, and research-gap analysis as a weak neighborhood signal—not as crisp topic segmentation.

---

## 14. Overall Findings

1. A full reproducible OpenAlex → ML pipeline is in place.  
2. OA prediction is a hard multiclass problem on this corpus; AdaBoost is the practical M4/M6 F1/accuracy leader.  
3. Tuning must be judged on held-out test honesty—validation winners may not dominate test F1.  
4. Impact-tier prediction is stronger than OA prediction here (test F1≈0.54) but still imperfect.  
5. Topic clusters are weak geometrically; treat as supporting analysis.

---

## 15. Limitations

- 2,000 OpenAlex AI/ML papers (domain- and size-limited)  
- OA category only partially predictable from allowed features  
- Impact tiers are relative to this corpus, not absolute prestige  
- Citations are time-dependent; age-related features correlate with impact labels  
- Clustering separation is weak (silhouette≈0.064)  
- TF-IDF does not capture full semantics  
- System is not yet a conversational research assistant  

---

## 16. Current System Capabilities

- OpenAlex data collection and cleaning  
- Feature extraction/engineering and EDA  
- SQLite storage and SQL retrieval (Python/R)  
- Feature selection with leakage controls  
- 14-algorithm OA classification + tuning + comparative viz  
- Impact-tier classification (12 models)  
- Exploratory topic clustering  

---

## 17. Future Scope (not implemented)

Semantic embeddings, RAG, summarization, similar-paper recommendation UI, research-gap finder, conversational interface, citation-aware assistant modules.

---

## 18. Conclusion

ResearchPilot Phase 1–2 delivers a coherent DA2 system: curated scholarly data, database connectivity, rigorous feature work, multi-model ML with tuning and comparative evaluation, plus secondary impact and exploratory clustering. Results are reported honestly—useful predictive signal without overstated accuracy. Documentation and demo materials in M8 make the work reproducible and examinable.

---

## 19. Reproducibility Instructions

See `DA2/REPRODUCIBILITY.md` for exact commands. Core Phase 2:

```powershell
cd E:\Tuned_Research
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py
python scripts/phase2/04_train_models.py
python scripts/phase2/05_tune_hyperparameters.py
python scripts/phase2/06_comparative_visualizations.py
python scripts/phase2/07_impact_and_clustering.py
python scripts/phase2/08_da2_database_demo.py
```

Do not retrain solely to refresh docs; frozen metrics are authoritative unless a genuine rebuild is requested.

---

## 20. References / Project Resources

- Milestone reports: `DA2/M1_*.md` … `DA2/M7_*.md`  
- Metrics CSVs: `DA2/m4_leaderboard.csv`, `m5_tuning_leaderboard.csv`, `m6_comparative_metrics.csv`, `m7_impact_leaderboard.csv`  
- Feature dictionary: `data/metadata/phase2_feature_dictionary.md`  
- Demo / viva / checklist: `DA2/DA2_Demo_Script.md`, `DA2/DA2_Viva_QA.md`, `DA2/DA2_FINAL_CHECKLIST.md`  
- OpenAlex: https://openalex.org/
