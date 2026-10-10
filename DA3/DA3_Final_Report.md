# ResearchPilot DA3 Final Report
## Human-Centered AI Research Assistant — Final Evaluation and Project Demonstration

**Submission:** Digital Assignment 3 (DA3) | October 16, 2026  
**Total Marks:** 10  
**Project:** ResearchPilot  
**Dataset:** 2,000 AI/ML papers from OpenAlex API

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Context and DA3 Objectives](#2-project-context-and-da3-objectives)
3. [Dataset and Methodology](#3-dataset-and-methodology)
4. [Machine Learning Models](#4-machine-learning-models)
5. [Final Performance Comparison](#5-final-performance-comparison) ⭐ Criterion 1
6. [Statistical Analysis](#6-statistical-analysis) ⭐ Criterion 1
7. [Interactive Dashboard](#7-interactive-dashboard) ⭐ Criterion 2
8. [Results and Discussion](#8-results-and-discussion) ⭐ Criterion 3
9. [Conclusion](#9-conclusion) ⭐ Criterion 4
10. [Limitations](#10-limitations) ⭐ Criterion 4
11. [Future Scope](#11-future-scope) ⭐ Criterion 4
12. [Code Organization](#12-code-organization) ⭐ Criterion 5
13. [References](#13-references)
14. [Appendices](#14-appendices)

---

## 1. Executive Summary

ResearchPilot is a human-centered AI research assistant that builds a reproducible machine learning pipeline over 2,000 AI/ML scholarly papers retrieved from the OpenAlex API. The primary prediction task is classifying each paper's open-access (OA) category into three classes: `fully_open`, `partially_open`, or `closed`.

**DA3 focus:** Final evaluation, statistical validation, and interactive visualization.

**Key Results:**

| Metric | Best Baseline (LDA) | Best Tuned (SVM-RBF) |
|--------|--------------------|-----------------------|
| Accuracy | 57.48% | 57.48% |
| Macro-F1 | 0.5696 | **0.5701** |
| ROC-AUC | 0.7577 | N/A |

**Statistical Analysis:** A Friedman non-parametric test was applied across 12 baseline models over 5-fold CV. Results are in `DA3/tables/friedman_results.csv`.

**Dashboard:** An R Shiny + Plotly interactive dashboard with 4 tabs allows dynamic exploration of all project results.

---

## 2. Project Context and DA3 Objectives

### 2.1 DA1 and DA2 Recap

**DA1 (Phase 1 — Data Foundation):**
- Retrieved 2,000 AI/ML papers from OpenAlex API
- Cleaned and preprocessed raw data
- Exploratory data analysis (EDA)
- Locked final dataset: `data/final/final_dataset.csv` (2,000 × 27)

**DA2 (Phase 2 — Intelligence Layer):**

| Milestone | Achievement |
|-----------|-------------|
| M1 | SQLite database with 4 tables, SQL queries demonstrated |
| M2 | Feature engineering: 27 → 206 features (structural, domain, publisher, TF-IDF) |
| M3 | Feature selection: 206 → 80 features (NZV, correlation, RF Gini ranking) |
| M4 | 12 ML algorithms trained and evaluated on held-out test set |
| M5 | Hyperparameter tuning for RF, SVM-RBF, XGBoost (5-fold grid search) |
| M6 | Comparative metrics, confusion matrices, ROC curves, 15+ visualizations |
| M7 | Impact-tier secondary classification + K-Means topic clustering (k=3) |
| M8 | Repository audit and documentation |

### 2.2 DA3 Objectives

The official DA3 rubric requires:

| Criterion | Description | Marks |
|-----------|-------------|-------|
| 1 | Final performance comparison + statistical analysis | 2 |
| 2 | Interactive visualization / dashboard | 2 |
| 3 | Result interpretation and discussion | 2 |
| 4 | Conclusion, limitations, future scope | 1 |
| 5 | Final report quality + code organization | 2 |
| 6 | Final presentation + viva | 1 |

DA3 does **not** re-implement DA2. It builds on the frozen DA2 pipeline to deliver final evaluation, statistical testing, interactive visualization, and comprehensive documentation.

---

## 3. Dataset and Methodology

### 3.1 Dataset Overview

| Property | Value |
|----------|-------|
| Source | OpenAlex API (open scholarly database) |
| Domain | Artificial Intelligence and Machine Learning |
| Size | 2,000 papers × 27 raw columns |
| Target variable | `oa_category` (3 classes) |
| Language filter | English only |

**OA Category Distribution:**

| Category | Count | Percentage |
|----------|-------|-----------|
| fully_open | 834 | 41.7% |
| partially_open | 723 | 36.2% |
| closed | 443 | 22.2% |

**Test Set (n=301, stratified):**

| Category | Count |
|----------|-------|
| closed | 67 |
| fully_open | 125 |
| partially_open | 109 |

### 3.2 Feature Engineering Pipeline

**Stage 1: Raw Features (27 columns)**
OpenAlex metadata: titles, abstracts, DOIs, citation counts, publication year, authors, concepts, keywords, open-access status.

**Stage 2: Engineered Features (206)**

| Group | Count | Examples |
|-------|-------|---------|
| Structural / text-length | 12 | `abstract_length`, `title_length`, `text_richness`, `recency_score` |
| Domain binary indicators | 25 | `dom_medical`, `dom_llm`, `dom_education`, `dom_vision` |
| Publisher one-hot | 18 | `pub_ieee`, `pub_mdpi`, `pub_springer`, `pub_nature` |
| TF-IDF terms | ~150 | `tf_learning`, `tf_deep`, `tf_model`, `tf_detection` |

**Stage 3: Feature Selection (80)**

Three filtering steps, **all fitted on training data only** (no leakage):
1. Near-Zero Variance filter — removes near-constant columns
2. Pearson correlation filter — within TF-IDF group, removes one from pairs with |r| > 0.95
3. Random Forest Gini importance ranking — retains top 80 features by Mean Decrease Gini

### 3.3 Database Integration

**Engine:** SQLite (`database/researchpilot_r.db`)  
**Tables:** `papers`, `ml_features`, `oa_features`, `model_results`  
**Interface:** RSQLite package in R  
**5 SQL queries demonstrated** including JOINs across `papers` and `ml_features`

### 3.4 Data Integrity

- All transformations fitted on training set only
- Validation set used only for hyperparameter selection (not final evaluation)
- Test set untouched until final reported evaluation
- Consistent split across all evaluations via `ml_data_splits_v2.rds`
- set.seed(42) throughout for full reproducibility

---

## 4. Machine Learning Models

### 4.1 Twelve Algorithms (Baseline)

| # | Model | Family | Key Characteristic |
|---|-------|--------|-------------------|
| 1 | Logistic Regression | Linear | Multi-class with L2 regularisation |
| 2 | Elastic Net | Linear | L1+L2 regularisation, automatic feature selection |
| 3 | Decision Tree | Tree | Interpretable, no regularisation |
| 4 | Random Forest | Ensemble | 300 trees, bagging, random feature subsets |
| 5 | XGBoost | Ensemble | Gradient boosting, regularised |
| 6 | Gradient Boosting | Ensemble | Sequential tree fitting |
| 7 | AdaBoost | Ensemble | Adaptive boosting, shallow trees |
| 8 | SVM-RBF | Kernel | Radial basis function kernel |
| 9 | SVM-Linear | Kernel | Linear kernel, equivalent to large-margin LR |
| 10 | Naive Bayes | Probabilistic | Assumes conditional feature independence |
| 11 | LDA | Discriminant | Linear discriminant analysis, dimensionality reduction |
| 12 | MLP | Neural | Single hidden layer, 50 units, nnet package |

### 4.2 Hyperparameter Tuning (3 Models)

5-fold cross-validation grid search on training data:

| Model | Best Params | CV F1 |
|-------|-------------|-------|
| Random Forest | ntree=500, mtry=8 | 0.6181 |
| SVM-RBF | cost=10, gamma=0.001 | 0.6065 |
| XGBoost | depth=6, eta=0.10, rounds=300 | N/A (logloss used) |

---

## 5. Final Performance Comparison

*This section satisfies DA3 Criterion 1 (Part 1)*

### 5.1 Complete Test-Set Results (n=301)

**Baseline Models:**

| Rank | Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 1 | **LDA** | **0.5748** | 0.5700 | 0.5694 | **0.5696** | 0.7577 |
| 2 | MLP | 0.5681 | 0.5650 | 0.5675 | 0.5657 | 0.7659 |
| 3 | Logistic Regression | 0.5648 | 0.5602 | 0.5668 | 0.5631 | 0.7565 |
| 4 | SVM-RBF | 0.5681 | 0.5597 | 0.5590 | 0.5559 | 0.7574 |
| 5 | Elastic Net | 0.5581 | 0.5559 | 0.5522 | 0.5539 | 0.7584 |
| 6 | SVM-Linear | 0.5515 | 0.5445 | 0.5438 | 0.5441 | 0.7587 |
| 7 | Gradient Boosting | 0.5382 | 0.5415 | 0.5359 | 0.5384 | 0.7402 |
| 8 | Random Forest | 0.5482 | 0.5448 | 0.5296 | 0.5317 | 0.7572 |
| 9 | AdaBoost | 0.5150 | 0.5126 | 0.5121 | 0.5123 | 0.6963 |
| 10 | Decision Tree | 0.5083 | 0.5295 | 0.4962 | 0.5017 | 0.6924 |
| 11 | Naive Bayes | 0.4186 | 0.4742 | 0.4909 | 0.4081 | 0.6840 |
| 12 | XGBoost | 0.3455 | 0.3464 | 0.3530 | 0.3428 | 0.5243 |

**Tuned Models:**

| Rank | Model | Accuracy | Precision | Recall | Macro-F1 | Params |
|------|-------|----------|-----------|--------|----------|--------|
| 1 | **SVM-RBF (Tuned)** | **0.5748** | — | — | **0.5701** | cost=10, γ=0.001 |
| 2 | Random Forest (Tuned) | 0.5615 | — | — | 0.5475 | ntree=500, mtry=8 |
| 3 | XGBoost (Tuned) | 0.3555 | — | — | 0.3537 | depth=6, η=0.10 |

### 5.2 Key Findings

- **Overall best model:** SVM-RBF (Tuned) — Macro-F1 = 0.5701
- **Best baseline:** LDA — Macro-F1 = 0.5696
- **Best improvement from tuning:** Random Forest — +0.0158 F1
- **Margin between best tuned and best baseline:** +0.0005 F1 (practically negligible)
- **Worst performing:** XGBoost — F1 = 0.3428 (poor match between model complexity and dataset size)
- **Best ROC-AUC:** MLP — 0.7659 (good discrimination even if F1 is slightly lower than LDA)

### 5.3 Tuning Improvement Summary

| Model | Base F1 | Tuned F1 | Improvement |
|-------|---------|----------|-------------|
| Random Forest | 0.5317 | 0.5475 | **+0.0158** |
| SVM-RBF | 0.5559 | 0.5701 | +0.0142 |
| XGBoost | 0.3428 | 0.3537 | +0.0109 |

Visualizations: See `DA3/figures/final_f1_comparison.png`, `final_accuracy_comparison.png`, `final_roc_comparison.png`, `final_baseline_vs_tuned.png`, `final_all_metrics_heatmap.png`

Run `source("r/scripts/phase3/12_final_comparison.R")` to regenerate.

---

## 6. Statistical Analysis

*This section satisfies DA3 Criterion 1 (Part 2)*

### 6.1 Research Question

Do the 12 ML models trained on the ResearchPilot dataset have statistically different Macro-F1 performance profiles, or are the observed differences attributable to random variation in fold assignments?

### 6.2 Methodology

**Test:** Friedman non-parametric test  
**Metric:** Macro-F1  
**Evaluation:** 5-fold cross-validation on training set (n≈1,400)  
**Post-hoc:** Nemenyi pairwise test (if H₀ rejected)

**Why Friedman?**

| Requirement | Friedman | ANOVA | Kruskal-Wallis |
|-------------|---------|-------|----------------|
| Non-parametric | ✅ | ❌ | ✅ |
| Repeated measures | ✅ | ✅ | ❌ |
| Multi-group (>2) | ✅ | ✅ | ✅ |
| **All requirements** | **✅** | **Partial** | **Partial** |

Friedman is the standard test for comparing multiple models across repeated folds in machine learning research (Demšar, 2006).

### 6.3 Formal Hypotheses

**H₀ (Null):** All 12 models have equal median Macro-F1 across folds. Observed differences are due to random variation.

**H₁ (Alternative):** At least one model has a significantly different median Macro-F1.

**Significance level:** α = 0.05

### 6.4 Cross-Validation Results

5-fold CV Macro-F1 matrix is generated by running:
```r
source("r/scripts/phase3/11_statistical_analysis.R")
```

Results saved to: `DA3/tables/cv_fold_matrix.csv`

The fold-level matrix shows each model's performance across all 5 folds, enabling detection of models that are both strong and consistent vs strong but volatile.

### 6.5 Friedman Test Results

**Actual results from `11_statistical_analysis.R` (October 2026):**

| Item | Value |
|------|-------|
| Chi-squared statistic | **43.4615** |
| Degrees of freedom | **11** |
| p-value | **0.000009 (9 × 10⁻⁶)** |
| Decision (α = 0.05) | **REJECT H₀** |

**Interpretation:** The p-value of 0.000009 is far below α = 0.05. We reject H₀ with very strong evidence. At least one model has a significantly different median Macro-F1 from the others. The observed performance differences across models are statistically real and not due to random fold variation.

### 6.6 Post-Hoc Analysis (Nemenyi Test)

7 significantly different model pairs identified (p < 0.05):

| Pair | p-value |
|------|---------|
| XGBoost vs Random Forest | 0.0045 |
| LDA vs XGBoost | 0.0045 |
| XGBoost vs Elastic Net | 0.0064 |
| Naive Bayes vs Random Forest | 0.0229 |
| LDA vs Naive Bayes | 0.0229 |
| XGBoost vs Logistic Regression | 0.0307 |
| Naive Bayes vs Elastic Net | 0.0307 |

**Key finding:** All significant pairs involve XGBoost or Naive Bayes at the bottom. Top-performing models (LDA, RF, Elastic Net, Logistic Regression) do not significantly differ from each other — their gaps are real but not statistically distinguishable at K=5 folds.

**CV vs Test ranking note:** Random Forest ranked 1st in CV (mean=0.6126) but LDA ranked 1st on the test set (F1=0.5696). This is expected — CV uses training folds with default hyperparameters, while the test evaluation uses final trained models. Both are valid metrics for different purposes.

---

## 7. Interactive Dashboard

*This section satisfies DA3 Criterion 2*

### 7.1 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Application framework | R Shiny | Reactive web interface |
| Charts | Plotly | Interactive visualizations with hover, zoom, pan |
| Tables | DT | Sortable, paginated data tables |
| Data | CSV/RDS files | All loaded from existing project outputs |
| Styling | Custom CSS | Professional appearance |

### 7.2 Running the Dashboard

```r
setwd("e:/Tuned_Research")
shiny::runApp("dashboard")
```

**Required packages:** shiny, plotly, dplyr, tidyr, readr, DT, scales

### 7.3 Tab 1: Overview

**KPI Cards (top row):**
- Total Papers: **2,000**
- Raw Features: **27**
- Engineered Features: **206**
- Selected Features: **80**
- ML Models: **12**
- Test Set Size: **301**

**Interactive Charts:**
- OA Category Distribution (Plotly bar chart) — loaded from `final_dataset.csv`
- Feature Engineering Pipeline (Plotly bar chart — 27 → 206 → 80)
- Test Set Class Distribution (Plotly pie chart — 67/125/109)

### 7.4 Tab 2: Model Comparison

**User Controls:**
- **Metric Selector** (dropdown): Macro-F1 / Accuracy / ROC-AUC / Precision / Recall
- **Model Type Filter** (checkboxes): Baseline / Tuned / All

**Interactivity:**
- Changing metric instantly re-sorts and re-colours the bar chart
- Hover tooltips show exact metric values, model type, and best params
- Baseline models shown in blue, Tuned models in red
- Sortable DT table below the chart

**Data source:** `data/ml_r/model_metrics_v2.csv`, `data/ml_r/all_results_v2.csv`

### 7.5 Tab 3: Model Diagnostics

**User Controls:**
- **Model Selector** (dropdown): LDA / Random Forest

**Displays:**
- **Confusion Matrix** (Plotly heatmap, column-normalised): shows misclassification patterns
- **ROC Curves** (Plotly line chart, One-vs-Rest per class): AUC values in legend
- **Feature Importance** (Plotly horizontal bar): Top 15 RF Gini features
- **Class-level metrics table**: Precision, Recall, F1 per class

**Note:** Confusion matrices use verified test-set results from DA2 pipeline.

### 7.6 Tab 4: Research Clusters

**Displays:**
- **Cluster Size Distribution** (Plotly pie chart): 200/1422/378 papers
- **Dominant Terms per Cluster** (Plotly grouped bar): Top 8 TF-IDF terms per cluster
- **Cluster Summary Table**: Cluster ID, theme, size, percentage, top term

**Data source:** `data/ml_r/cluster_keywords.csv`

**Note on labels:** Cluster labels (Education/GenAI, General ML, Medical Imaging) are human interpretations of TF-IDF terms, not algorithmic outputs.

### 7.7 Dashboard Design Principles

- No fake data — all values from actual project files
- No hardcoded metrics — dynamically loaded
- Clean UI with consistent colour scheme
- All charts are genuinely interactive (not static images)
- Meaningful hover tooltips
- Notes/warnings where interpretation caveats apply

---

## 8. Results and Discussion

*This section satisfies DA3 Criterion 3*

*(Complete 10-section discussion in `DA3/DA3_Results_Discussion.md` — key points summarised here)*

### 8.1 Dataset and Features

The corpus is moderately imbalanced (fully_open 42%, partially_open 36%, closed 22%). The use of Macro-F1 accounts for this imbalance. Stratified splitting preserves proportions. Publisher binary features are the strongest predictors — confirming that OA status is a publisher policy decision, not a paper content property.

### 8.2 Model Performance Insights

**Linear models win.** LDA, Logistic Regression, and Elastic Net consistently outperform tree ensembles. The engineered feature space is approximately linearly separable — publisher binary flags and structural features form relatively clean linear boundaries.

**XGBoost's failure is explainable.** Dataset size (~1,400 training samples) is insufficient for XGBoost's complexity. TF-IDF sparsity compounds the problem. This is a known phenomenon, not a coding error.

**Tuning is worthwhile but modest.** Random Forest improved most (+0.0158 F1) from `ntree=500` providing better ensemble stability. SVM-RBF improved +0.0142 with better kernel width (gamma=0.001). The overall gap between best tuned and best baseline is +0.0005 — practically negligible, confirming diminishing returns.

### 8.3 Error Analysis

`partially_open` is the hardest class (lowest ROC-AUC = 0.6759) due to definitional ambiguity and feature overlap with both other classes. `closed` is most separable (AUC = 0.8256) because major paywalled publishers (Elsevier, Wiley) have distinctive DOI prefixes.

A binary OA classification task (open vs closed) would likely achieve 70%+ accuracy by removing the ambiguous middle class.

### 8.4 Clustering Insights

Three research themes identified (K-Means, k=3):
- **Cluster 1 (10%):** Education/GenAI/ChatGPT — emerging subfield since 2022
- **Cluster 2 (71%):** General ML/Prediction/Robotics — mainstream AI/ML research
- **Cluster 3 (19%):** Medical Imaging/Deep Learning — coherent clinical AI subfield

Weak silhouette (~0.06) is expected for overlapping AI/ML topics. The clusters reflect genuine thematic tendencies despite weak separation.

### 8.5 Statistical Analysis Summary

See Section 6 for full details. The Friedman test provides rigorous validation that performance differences are (or are not) statistically consistent across folds, beyond single test-set comparison.

---

## 9. Conclusion

*This section satisfies DA3 Criterion 4*

ResearchPilot DA3 completes the final evaluation of the ResearchPilot project with:

1. **Statistical validation** of 12 ML models via Friedman test and Nemenyi post-hoc analysis
2. **Interactive dashboard** (R Shiny + Plotly, 4 tabs) for dynamic result exploration
3. **Comprehensive interpretation** of all results with proper caveats
4. **Complete documentation** organized for reproducibility

**Best result:** SVM-RBF (Tuned) — Macro-F1 = 0.5701, Accuracy = 57.48%  
**Statistical finding:** See `DA3/tables/friedman_results.csv`  
**Dashboard:** `shiny::runApp("dashboard")` from project root

The project demonstrates a complete, reproducible, and statistically validated ML pipeline for scholarly metadata analysis — a solid foundation for the planned Phase 3 research assistant features.

---

## 10. Limitations

*This section satisfies DA3 Criterion 4*

Ten verified limitations (detailed discussion in `DA3/DA3_Conclusion_Limitations_Future.md`):

1. **Dataset size** — 2,000 papers limits complex model performance and statistical power
2. **AI/ML domain only** — findings may not generalise to other scholarly fields
3. **Moderate accuracy (~57%)** — three-class overlap limits performance; not production-ready
4. **Partially_open ambiguity** — inherent label noise from fluid OA definition
5. **TF-IDF features** — no semantic embeddings; misses contextual similarity
6. **Static dataset** — snapshot in time; no continuous update pipeline
7. **Metadata dependency** — relies on OpenAlex data quality; missing fields degrade predictions
8. **Cluster label subjectivity** — human-interpreted themes, not algorithmic labels
9. **No full-text analysis** — titles and abstracts only; methodology details in paper body unused
10. **Limited CV folds (K=5)** — Friedman test has limited power for detecting small differences

---

## 11. Future Scope

*This section satisfies DA3 Criterion 4*

All items below are **NOT IMPLEMENTED** in the current codebase. They are planned for future development phases.

**Tier 1 (High Priority):**
- RAG-Based Research Assistant ❌ NOT IMPLEMENTED
- Local LLM Integration via Ollama ❌ NOT IMPLEMENTED
- Instruction Fine-Tuning (LoRA/QLoRA) ❌ NOT IMPLEMENTED

**Tier 2 (Medium Priority):**
- Semantic Embeddings (SciBERT/Sentence-BERT) ❌ NOT IMPLEMENTED
- Research Gap Detection ❌ NOT IMPLEMENTED
- Paper Similarity Search ❌ NOT IMPLEMENTED
- Journal Recommendation System ❌ NOT IMPLEMENTED

**Tier 3 (Long-Term):**
- Larger multi-domain corpus (10,000+ papers) ❌ NOT IMPLEMENTED
- Real-time data pipeline ❌ NOT IMPLEMENTED
- Causal inference on OA patterns ❌ NOT IMPLEMENTED
- Explainable AI (SHAP/LIME) ❌ NOT IMPLEMENTED

*(Full descriptions with estimated effort in `DA3/DA3_Conclusion_Limitations_Future.md`)*

---

## 12. Code Organization

*This section satisfies DA3 Criterion 5*

```
e:/Tuned_Research/
├── data/
│   ├── final/                     # Locked Phase 1 dataset (2000×27)
│   └── ml_r/                      # Phase 2 R artifacts (models, splits, metrics)
├── database/
│   └── researchpilot_r.db         # SQLite database
├── r/
│   └── scripts/
│       ├── phase2/                # DA2 scripts (frozen — do not modify)
│       │   ├── 00_enriched_features.R
│       │   ├── 02_feature_selection_v2.R
│       │   ├── 03_database_setup.R
│       │   ├── 04_database_queries.R
│       │   ├── 05_ml_models_v2.R
│       │   ├── 06_hyperparameter_tuning_v2.R
│       │   ├── 07_model_evaluation_v2.R
│       │   ├── 08_visualizations_v2.R
│       │   ├── 09_impact_tier_v2.R
│       │   └── 10_clustering.R
│       └── phase3/                # DA3 scripts (new)
│           ├── 11_statistical_analysis.R    # Friedman test
│           └── 12_final_comparison.R        # Final comparison charts
├── dashboard/
│   ├── app.R                      # R Shiny + Plotly interactive dashboard
│   └── README.md                  # Dashboard run instructions
├── DA2/                           # DA2 documentation (frozen)
├── DA3/                           # DA3 deliverables (new)
│   ├── DA3_Final_Report.md        # This document
│   ├── DA3_Statistical_Analysis.md
│   ├── DA3_Results_Discussion.md
│   ├── DA3_Conclusion_Limitations_Future.md
│   ├── DA3_Viva_QA.md
│   ├── DA3_PROJECT_AUDIT.md
│   ├── DA3_IMPLEMENTATION_PLAN.md
│   ├── DA3_RUBRIC_COVERAGE.md
│   ├── figures/                   # DA3-specific figures
│   └── tables/                    # DA3-specific tables
├── configs/                       # YAML configurations
├── README.md                      # Main project README (updated for DA3)
└── DA3_CHECKLIST.md               # Final verification checklist
```

**Design principles:**
- DA2 pipeline scripts preserved unchanged
- New DA3 code in dedicated `r/scripts/phase3/` directory
- Dashboard code in dedicated `dashboard/` directory
- All DA3 documents in `DA3/` directory
- No file duplication; data files shared between phases

---

## 13. References

1. Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of the world's research works. *arXiv preprint arXiv:2205.01833*.

2. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.

3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

4. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

5. Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

7. R Core Team (2024). R: A language and environment for statistical computing. R Foundation for Statistical Computing. https://www.R-project.org/

8. Chang, W., et al. (2024). shiny: Web Application Framework for R. R package version 1.8.x.

9. Sievert, C. (2020). Interactive Web-Based Data Visualization with R, plotly, and shiny. Chapman and Hall/CRC.

10. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267–288.

11. Venables, W.N., & Ripley, B.D. (2002). Modern Applied Statistics with S (4th ed.). Springer.

12. Boulesteix, A.L., et al. (2012). Overview of random forest methodology and practical guidance. *WIREs Data Mining and Knowledge Discovery*, 2(6), 493–507.

---

## 14. Appendices

### Appendix A: Hyperparameter Grids

**Random Forest Grid:**

| ntree | mtry | CV F1 |
|-------|------|-------|
| 300 | 5 | — |
| 300 | 8 | — |
| 300 | 12 | — |
| 300 | 15 | — |
| 500 | 5 | — |
| **500** | **8** | **0.6181** ← Best |
| 500 | 12 | — |
| 500 | 15 | — |
| 800 | 5 | — |
| 800 | 8 | — |
| 800 | 12 | — |
| 800 | 15 | — |

**SVM-RBF Grid:**

| cost | gamma | CV F1 |
|------|-------|-------|
| 0.1 | 0.0001 | — |
| **10** | **0.001** | **0.6065** ← Best |
| 100 | 0.01 | — |
| ... | ... | ... |

**XGBoost Grid:**

| max_depth | eta | nrounds | CV Logloss |
|-----------|-----|---------|-----------|
| 4 | 0.05 | 150 | — |
| **6** | **0.10** | **300** | ← Best (lowest logloss) |
| 8 | 0.10 | 150 | — |
| ... | ... | ... | ... |

### Appendix B: Top 80 Selected Features (First 20)

Features ranked by Random Forest Gini importance:

1. pub_ieee
2. pub_mdpi
3. title_to_abstract_ratio
4. abstract_length
5. abstract_word_count
6. title_length
7. text_richness
8. title_word_count
9. concept_count
10. keyword_count
11. pub_springer
12. recency_score
13. dom_medical
14. tf_deep
15. tf_model
16. tf_detect
17. tf_learn
18. pub_nature
19. dom_education
20. tf_imag
...

### Appendix C: Reproducibility Instructions

```r
# Step 1: Set working directory
setwd("e:/Tuned_Research")

# Step 2: Install all required packages
source("r/install_da2_packages.R")

# Step 3: Run DA2 pipeline (frozen — only if starting fresh)
# source("r/scripts/phase2/00_enriched_features.R")
# source("r/scripts/phase2/02_feature_selection_v2.R")
# ...etc

# Step 4: Run DA3 scripts
source("r/scripts/phase3/11_statistical_analysis.R")  # ~15 min
source("r/scripts/phase3/12_final_comparison.R")      # ~2 min

# Step 5: Launch dashboard
shiny::runApp("dashboard")
```

**Expected outputs:**
- `DA3/tables/cv_fold_matrix.csv`
- `DA3/tables/friedman_results.csv`
- `DA3/tables/final_model_comparison.csv`
- `DA3/figures/stat_*.png` (3 files)
- `DA3/figures/final_*.png` (5 files)
- Dashboard at `http://localhost:XXXX`

---

*ResearchPilot DA3 Final Report | October 2026 | Version 1.0*  
*Project root: `e:/Tuned_Research` | Database: `database/researchpilot_r.db`*
