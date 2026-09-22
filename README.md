# ResearchPilot

**A Human-Centered AI Research Assistant — Data Science Foundation (Phase 1 + Phase 2)**

ResearchPilot builds a reproducible pipeline over OpenAlex AI/ML scholarly metadata: collect and clean papers, store them in a SQLite database, engineer leakage-safe features, train and compare 12 ML models for open-access category prediction, tune strong models, and add secondary impact-tier prediction with exploratory topic clustering.

> **Status:** Phase 1 complete · Phase 2 (M1–M8) complete and frozen · DA-2 faculty demo ready  
> Phase 3 (RAG / chat assistant / dashboard) is **roadmap only** — not implemented.

---

## Problem

Researchers need structured ways to explore large literature collections. Before conversational AI features, the project establishes trustworthy data, database access, and evaluated predictive models for:

1. **Open-access category** (`fully_open` / `partially_open` / `closed`)
2. **Relative impact tier** (`low` / `medium` / `high`)
3. **Exploratory topic clusters** (K-Means on TF-IDF, k=3)

---

## Dataset

| Item | Value |
|------|--------|
| Source | OpenAlex API |
| Final table | `data/final/final_dataset.csv` |
| Size | **2,000 papers × 27 columns** |
| Domain | AI / ML focused corpus |
| Languages | English only |

---

## Phase 1 — Data Foundation

OpenAlex → raw CSV → preprocessing → feature extraction → feature engineering → EDA → locked `data/final/final_dataset.csv`.

Scripts: `scripts/phase1/` · R scripts: `r/scripts/` · Figures: `reports/figures/`

---

## Phase 2 — Intelligence Layer (R Pipeline — v2)

| Milestone | Script | Focus |
|-----------|--------|--------|
| M1 | `03_database_setup.R` | SQLite load + SQL retrieval |
| M2 | `00_enriched_features.R` | Feature engineering (27 → 206) |
| M3 | `02_feature_selection_v2.R` | Feature selection (206 → 80) |
| M4 | `05_ml_models_v2.R` | 12 ML algorithms (OA classification) |
| M5 | `06_hyperparameter_tuning_v2.R` | Hyperparameter tuning (RF, SVM, XGBoost) |
| M6 | `07_model_evaluation_v2.R` + `08_visualizations_v2.R` | Comparative metrics + 15 visualizations |
| M7 | `09_impact_tier_v2.R` + `10_clustering.R` | Impact-tier + topic clustering |
| M8 | `DA2_Faculty_Demo_COMPLETE.R` | Integration, demo, DA-2 presentation |

All Phase 2 R scripts are in `r/scripts/phase2/`.

---

## Database

- Engine: **SQLite** (`database/researchpilot_r.db`)
- Tables: `papers`, `ml_features`, `oa_features`, `model_results`
- Queried via **RSQLite** in R
- 5 SQL queries demonstrated live including a JOIN across `papers` and `ml_features`

---

## Feature Engineering

| Stage | Count | Method |
|-------|-------|--------|
| Raw columns | 27 | OpenAlex fields |
| After engineering | 206 | Structural + domain binary + publisher one-hot + TF-IDF terms |
| After selection | **80** | NZV → Pearson corr filter → RF Gini importance top-80 |

### Feature categories engineered (27 → 206)

| Group | Count | Examples |
|-------|-------|---------|
| Structural / text-length | 12 | `title_length`, `abstract_length`, `text_richness`, `recency_score` |
| Domain binary indicators | 25 | `dom_medical`, `dom_llm`, `dom_education`, `dom_vision` |
| Publisher one-hot (DOI prefix) | 18 | `pub_ieee`, `pub_mdpi`, `pub_springer`, `pub_nature` |
| TF-IDF terms | ~150 | `tf_learning`, `tf_deep`, `tf_model`, `tf_detection` |

### Feature selection pipeline (206 → 80)

Three steps, all fitted on the **70% training split only** (no leakage):

1. **Near-Zero Variance (NZV) filter** — removes constant or near-constant columns (variance < 1e-10, or one value dominating >20× the second)
2. **Pearson correlation filter** — within the TF-IDF group only, removes one column from any pair with |r| > 0.95
3. **Random Forest Gini importance ranking** — 300-tree RF ranks remaining features by Mean Decrease Gini; top 80 are kept

> Note: Mutual Information was considered but not implemented in v2 due to computational cost with 205 features. RF Gini importance is the actual ranking method used.

---

## ML Models (R v2 Pipeline — verified results)

### Open-Access Classification (12 models)

All models evaluated on the **same stratified test split** (`ml_data_splits_v2.rds`, set.seed=42, 70/15/15 split, n=301 test rows).

| Rank | Model | Type | Accuracy | Macro-F1 | ROC-AUC |
|------|-------|------|----------|----------|---------|
| 1 (baseline) | **LDA** | Baseline | 0.5748 | **0.5696** | 0.7577 |
| 2 (baseline) | MLP | Baseline | 0.5681 | 0.5657 | 0.7659 |
| 3 (baseline) | Logistic Regression | Baseline | 0.5648 | 0.5631 | 0.7565 |
| 4 (baseline) | SVM (RBF) | Baseline | 0.5681 | 0.5559 | 0.7574 |
| **1 (overall)** | **SVM-tuned** | **Tuned** | **0.5748** | **0.5701** | — |
| 2 (tuned) | RF-tuned | Tuned | 0.5615 | 0.5475 | — |
| 3 (tuned) | XGBoost-tuned | Tuned | 0.3555 | 0.3537 | — |

**All 12 algorithms:** Logistic Regression, Elastic Net, Decision Tree, Random Forest, XGBoost, Gradient Boosting, AdaBoost, SVM-RBF, SVM-Linear, Naive Bayes, LDA, MLP

### Hyperparameter Tuning

| Model | Best Params | Base F1 | Tuned F1 | Improvement |
|-------|-------------|---------|----------|-------------|
| Random Forest | ntree=500, mtry=8 | 0.5317 | 0.5475 | +0.0158 |
| SVM (RBF) | cost=10, gamma=0.001 | 0.5559 | 0.5701 | +0.0142 |
| XGBoost | depth=6, eta=0.10, rounds=300 | 0.3428 | 0.3537 | +0.0109 |

Method: 5-fold cross-validation grid search on training set only; final evaluation on untouched test set.

### Random Forest — Confusion Matrix (proper test split, n=301)

```
                Actual
Predicted     closed  fully_open  partially_open
closed            31           7              18
fully_open        23          88              45
partly_open       13          30              46

Accuracy : 0.5482  |  Macro-F1 : 0.5317  |  ROC-AUC : 0.7572
```

### ROC-AUC (Random Forest, OVR macro)

| Class | AUC |
|-------|-----|
| closed | 0.8256 |
| fully_open | 0.7701 |
| partially_open | 0.6759 |
| **Mean (macro OVR)** | **0.7572** |

### Impact-Tier Classification (secondary task)

| Champion | Random Forest |
|----------|--------------|
| Test Accuracy | 0.5233 |
| Test Macro-F1 | 0.5134 |
| Test ROC-AUC | 0.6943 |
| Models evaluated | 9 |

### Topic Clustering (unsupervised)

| Item | Value |
|------|-------|
| Method | K-Means on TF-IDF (200 terms) with L2 normalisation |
| Best k | **3** (silhouette from k=3 to 10) |
| Cluster 1 (10%) | Education / GenAI / ChatGPT papers |
| Cluster 2 (71%) | General ML / Prediction / Robotics |
| Cluster 3 (19%) | Medical Imaging / Deep Learning / Computer Vision |

Clustering is fully independent of the supervised ML pipeline.

---

## Data integrity notes

- **No data leakage:** scaling, NZV, correlation filter, and RF importance ranking all fitted on training split only
- **Consistent test split:** CMD-6 (confusion matrix) and CMD-10 (ROC) both use the same `ml_data_splits_v2.rds` test set as the training pipeline — verified to 4 decimal places
- **TF-IDF built on full corpus** (standard practice; no label information used)

---

## How to run (R pipeline)

```r
setwd("e:/Tuned_Research")

# Run Phase 2 scripts in order:
source("r/scripts/phase2/00_enriched_features.R")      # feature engineering
source("r/scripts/phase2/02_feature_selection_v2.R")   # feature selection
source("r/scripts/phase2/03_database_setup.R")         # SQLite DB
source("r/scripts/phase2/04_database_queries.R")       # SQL queries
source("r/scripts/phase2/05_ml_models_v2.R")           # train 12 models
source("r/scripts/phase2/06_hyperparameter_tuning_v2.R") # tune RF/SVM/XGB
source("r/scripts/phase2/07_model_evaluation_v2.R")    # leaderboard
source("r/scripts/phase2/08_visualizations_v2.R")      # 15 plots
source("r/scripts/phase2/09_impact_tier_v2.R")         # impact tier
source("r/scripts/phase2/10_clustering.R")             # topic clusters
```

**For faculty demo only** (no retraining — uses saved model objects):
```r
source("DA2_Faculty_Demo_COMPLETE.R")
```

Full reproducibility detail: [`DA2/REPRODUCIBILITY.md`](DA2/REPRODUCIBILITY.md)

> Do not retrain unless explicitly needed. Model objects are saved in `data/ml_r/model_objects_v2.rds` and test splits in `data/ml_r/ml_data_splits_v2.rds`.

---

## Folder structure

```text
Tuned_Research/
├── data/
│   ├── final/               # Phase 1 locked CSV (2000 × 27)
│   └── ml_r/                # Phase 2 R matrices, model objects, splits
├── database/                # researchpilot_r.db (SQLite)
├── r/scripts/phase2/        # All Phase 2 R scripts
├── reports/
│   ├── figures/phase2_r/    # 15+ plots (ROC, CM, feature importance, etc.)
│   └── tables/              # leaderboard CSVs
├── DA2/                     # Faculty documentation pack
├── configs/                 # YAML configs
└── DA2_Faculty_Demo_COMPLETE.R  # Main demo script
```

---

## Limitations

- Moderate OA predictability (~57% accuracy) — 3-class OA classification from text/metadata has inherent overlap
- Tuning gain is small (+0.0005 F1 for best model) — LDA already captures linear separability well; each model improved over its own baseline
- Weak cluster separation (silhouette ~0.06) — expected for overlapping AI/ML research topics
- 2,000-paper AI/ML corpus; TF-IDF features (not deep semantic embeddings)
- No chat / RAG / conversational assistant implemented

---

## Future roadmap

Embeddings, RAG, summarization, similar-paper retrieval, gap analysis, conversational assistant — **not implemented** in this repository state.

---

## Documentation pack

| Document | Path |
|----------|------|
| DA-2 Final Report | [`DA2/ResearchPilot_DA2_Final_Report.md`](DA2/ResearchPilot_DA2_Final_Report.md) |
| Faculty Demo Script (R) | [`DA2_Faculty_Demo_COMPLETE.R`](DA2_Faculty_Demo_COMPLETE.R) |
| Demo walkthrough | [`DA2/DA2_Demo_Script.md`](DA2/DA2_Demo_Script.md) |
| Viva Q&A | [`DA2/DA2_Viva_QA.md`](DA2/DA2_Viva_QA.md) |
| DA-2 Checklist | [`DA2/DA2_FINAL_CHECKLIST.md`](DA2/DA2_FINAL_CHECKLIST.md) |
| Reproducibility | [`DA2/REPRODUCIBILITY.md`](DA2/REPRODUCIBILITY.md) |
| Model inventory | [`DA2/DA2_Model_Inventory.md`](DA2/DA2_Model_Inventory.md) |
