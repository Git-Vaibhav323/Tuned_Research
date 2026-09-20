# DA2 Rubric Checklist — R Implementation
**Project**: ResearchPilot | **Language**: R | **Database**: SQLite

---

## Rubric Mapping

### ✅ 1. Feature Engineering & Feature Selection — 1 mark

| Evidence | Location |
|----------|----------|
| **R Script** | `r/scripts/phase2/01_feature_engineering.R` |
| **Output CSV** | `data/ml_r/engineered_features.csv` (2000 × 38) |
| **New features** | 11 new features added (title_word_count, abstract_word_count, title_to_abstract_ratio, keyword_diversity, concept_diversity, text_richness, recency_score, text_length_category, abstract_keyword_overlap, publication_year_norm, oa_category_encoded) |
| **Report** | `reports/phase2_r/feature_engineering_report.md` |
| **R Script** | `r/scripts/phase2/02_feature_selection.R` |
| **Output CSV** | `data/ml_r/selected_features_oa.csv` (9 features selected) |
| **Method 1** | Near-Zero Variance filter → removed 3 (has_doi, keyword_diversity, concept_diversity) |
| **Method 2** | Pairwise Correlation filter (r > 0.90) → removed 5 |
| **Method 3** | Mutual Information (discretised, 10 bins) → all 9 survivors pass threshold |
| **Method 4** | Random Forest Importance (Gini) |
| **Selection figure** | `reports/figures/phase2_r/feature_selection_agreement.png` |
| **MI figure** | `reports/figures/phase2_r/feature_mi_scores.png` |
| **Report** | `reports/phase2_r/feature_selection_report.md` |
| **Before/After** | 17 candidate → 9 selected features (47% reduction) |
| **Top features** | title_word_count, text_richness, title_to_abstract_ratio, abstract_word_count, keyword_count |
| **Leakage guard** | Excluded: is_open_access, oa_status, oa_url, cited_by_count, citation_per_year, citation_log |

---

### ✅ 2. Database Connectivity and Data Retrieval using R — 2 marks

| Evidence | Location |
|----------|----------|
| **Database** | `database/researchpilot_r.db` (SQLite) |
| **R packages** | `DBI` + `RSQLite` |
| **Setup script** | `r/scripts/phase2/03_database_setup.R` |
| **Query script** | `r/scripts/phase2/04_database_queries.R` |
| **Tables created** | `papers` (2000 rows), `ml_features` (2000 rows), `oa_features` (2000 rows), `model_results` |
| **Query 1** | Most recent papers (2024+) → `data/database_r/q1_most_recent_papers.csv` |
| **Query 2** | Most highly cited papers → `data/database_r/q2_most_cited_papers.csv` |
| **Query 3** | OA category distribution with % and avg citations → `data/database_r/q3_oa_category_distribution.csv` |
| **Query 4** | Aggregate: Papers per year → `data/database_r/q4_papers_by_year.csv` |
| **Query 5** | OA breakdown by year (cross-tab) → `data/database_r/q5_oa_by_year.csv` |
| **Query 6** | JOIN: papers + ml_features (high text-richness) → `data/database_r/q6_high_richness_papers.csv` |
| **Query 7** | JOIN + GROUP BY: avg features per OA class → `data/database_r/q7_avg_features_by_oa.csv` |
| **Visualisation** | `reports/figures/phase2_r/db_papers_by_year_oa.png` |
| **Connection demo** | `dbConnect()` → `dbListTables()` → `dbGetQuery()` → `dbDisconnect()` |

---

### ✅ 3. Implementation of 10–15 ML/DL Algorithms — 3 marks

| # | Algorithm | Family | R Package | Script |
|---|-----------|--------|-----------|--------|
| 1 | Logistic Regression (multinomial) | Linear | nnet::multinom | 05_ml_models.R |
| 2 | Elastic Net Logistic Regression | Regularised Linear | glmnet | 05_ml_models.R |
| 3 | CART Decision Tree | Tree | rpart | 05_ml_models.R |
| 4 | Random Forest (500 trees) | Ensemble Bagging | randomForest | 05_ml_models.R |
| 5 | Gradient Boosting Machine | Ensemble Boosting | gbm | 05_ml_models.R |
| 6 | XGBoost | Extreme Gradient Boosting | xgboost | 05_ml_models.R |
| 7 | AdaBoost | Adaptive Boosting | adabag | 05_ml_models.R |
| 8 | SVM (RBF Kernel) | Kernel | e1071 | 05_ml_models.R |
| 9 | SVM (Linear Kernel) | Kernel | e1071 | 05_ml_models.R |
| 10 | Naive Bayes (Gaussian) | Probabilistic | naivebayes / e1071 | 05_ml_models.R |
| 11 | Linear Discriminant Analysis | Discriminant | MASS | 05_ml_models.R |
| 12 | K-Nearest Neighbours (k=7) | Instance-based | kknn | 05_ml_models.R |
| 13 | MLP Neural Network (size=50) | Neural Network | nnet | 05_ml_models.R |

**Total: 13 genuinely different algorithms (not parameter variations)**

Evidence:
- Script: `r/scripts/phase2/05_ml_models.R`
- Output: `data/ml_r/model_metrics_oa.csv` (metrics for all models)
- Output: `data/ml_r/model_objects_oa.rds` (saved model objects)
- Leaderboard: `reports/tables/r_model_leaderboard_baseline.csv`

---

### ✅ 4. Hyperparameter Tuning and Model Optimization — 1 mark

| Evidence | Location |
|----------|----------|
| **Script** | `r/scripts/phase2/06_hyperparameter_tuning.R` |
| **Method** | Grid search with 5-fold cross-validation (training set only) |
| **Models tuned** | Random Forest (ntree × mtry grid), XGBoost (max_depth × eta × nrounds), SVM-RBF (cost × gamma) |
| **RF grid** | ntree ∈ {200,500} × mtry ∈ {2,3,4} = 6 combinations |
| **XGBoost grid** | depth ∈ {3,5} × eta ∈ {0.05,0.10} × nrounds ∈ {100,200} = 8 combinations |
| **SVM grid** | cost ∈ {0.1,1,10} × gamma ∈ {0.01,0.1,auto} = 9 combinations |
| **Output** | `data/ml_r/tuning_results.csv` |
| **Leaderboard** | `reports/tables/r_tuning_leaderboard.csv` |
| **Saved models** | `data/ml_r/rf_tuned.rds`, `data/ml_r/xgb_tuned.rds`, `data/ml_r/svm_tuned.rds` |
| **Test set integrity** | Test set NEVER used during tuning — CV on train only |

---

### ✅ 5. Comparative Performance Analysis — 1 mark

| Evidence | Location |
|----------|----------|
| **Script** | `r/scripts/phase2/07_model_evaluation.R` |
| **Metrics** | Accuracy, Macro-Precision, Macro-Recall, Macro-F1, OVR ROC-AUC |
| **Leaderboard CSV** | `reports/tables/r_model_leaderboard.csv` |
| **Leaderboard MD** | `reports/tables/r_model_leaderboard.md` |
| **Combined results** | `data/ml_r/all_results_combined.csv` (baseline + tuned) |
| **Database update** | Results written to `model_results` table in SQLite |
| **Best Accuracy** | see leaderboard |
| **Best Macro-F1** | see leaderboard |
| **Best ROC-AUC** | see leaderboard |
| **Baseline vs Tuned** | Delta comparison in evaluation output |
| **Note** | Different metrics have different leaders — no single champion claimed |

---

### ✅ 6. Comparative Visualizations — 1 mark

| # | Figure | File | Script |
|---|--------|------|--------|
| 1 | Model Accuracy Comparison | `01_model_accuracy_comparison.png` | 08 |
| 2 | Macro-F1 Comparison | `02_model_f1_comparison.png` | 08 |
| 3 | ROC Curves (OVR, 3 panels) | `03_roc_curves_ovr.png` | 08 |
| 4 | ROC-AUC Bar Chart | `04_roc_auc_comparison.png` | 08 |
| 5 | Precision-Recall Curves | `05_precision_recall_curves.png` | 08 |
| 6a | Confusion Matrix (best model) | `06a_confusion_matrix_best_model.png` | 08 |
| 6b | Confusion Matrix (Random Forest) | `06b_confusion_matrix_rf.png` | 08 |
| 7 | Feature Importance (RF Gini) | `07_feature_importance_rf.png` | 08 |
| 8 | Precision Comparison | `08_precision_comparison.png` | 08 |
| 9 | Recall Comparison | `09_recall_comparison.png` | 08 |
| 10 | Baseline vs Tuned | `10_baseline_vs_tuned.png` | 08 |
| + | Feature Selection Agreement | `feature_selection_agreement.png` | 02 |
| + | Mutual Information Scores | `feature_mi_scores.png` | 02 |
| + | Impact-Tier Comparison | `impact_tier_model_comparison.png` | 09 |
| + | Impact-Tier Distribution | `impact_tier_class_distribution.png` | 09 |
| + | Cluster PCA Scatter | `clustering_pca_scatter.png` | 10 |
| + | Silhouette vs k | `clustering_silhouette_vs_k.png` | 10 |
| + | Cluster Sizes | `clustering_size_distribution.png` | 10 |
| + | Papers by Year & OA | `db_papers_by_year_oa.png` | 04 |

**All figures directory**: `reports/figures/phase2_r/`
**All figures generated in R using ggplot2**

---

### ✅ 7. Progress Demonstration and Documentation — 1 mark

| Evidence | Location |
|----------|----------|
| **DA2 Final Report** | `DA2/DA2_R_Final_Report.md` (25 sections) |
| **Faculty Demo Script** | `DA2/DA2_R_Demo.md` (15 steps, 7-10 min) |
| **Rubric Checklist** | `DA2/DA2_Rubric_Checklist.md` (this file) |
| **Feature Engineering Report** | `reports/phase2_r/feature_engineering_report.md` |
| **Feature Selection Report** | `reports/phase2_r/feature_selection_report.md` |
| **Model Leaderboard** | `reports/tables/r_model_leaderboard.md` |
| **Requirements** | `r/requirements.md` |
| **Setup Script** | `r/phase2_setup.R` |
| **Python M1-M7** | Existing Python implementation preserved (not deleted) |

---

## Complete R Script Inventory

| Script | Milestone | Purpose |
|--------|-----------|---------|
| `r/phase2_setup.R` | Setup | Install + verify all packages |
| `r/scripts/phase2/01_feature_engineering.R` | M1 | Engineer 11 new features |
| `r/scripts/phase2/02_feature_selection.R` | M2 | NZV + Corr + MI + RF selection |
| `r/scripts/phase2/03_database_setup.R` | M3 | Create SQLite + load 4 tables |
| `r/scripts/phase2/04_database_queries.R` | M3 | 7 SQL queries + visualisation |
| `r/scripts/phase2/05_ml_models.R` | M4 | Train 13 ML algorithms |
| `r/scripts/phase2/06_hyperparameter_tuning.R` | M5 | Grid search + 5-fold CV |
| `r/scripts/phase2/07_model_evaluation.R` | M6 | Leaderboard + DB update |
| `r/scripts/phase2/08_comparative_visualizations.R` | M6 | 10+ ggplot2 figures |
| `r/scripts/phase2/09_impact_tier.R` | M7a | Impact-tier 10 classifiers |
| `r/scripts/phase2/10_clustering.R` | M7b | TF-IDF + KMeans clustering |

---

## How to Run the Complete Pipeline

```r
# In RStudio — set working directory to project root first
setwd("E:/Tuned_Research")
R_PATH <- "C:/Program Files/R/R-4.6.1/bin/Rscript.exe"

# Or run each script sequentially:
source("r/scripts/phase2/01_feature_engineering.R")
source("r/scripts/phase2/02_feature_selection.R")
source("r/scripts/phase2/03_database_setup.R")
source("r/scripts/phase2/04_database_queries.R")
source("r/scripts/phase2/05_ml_models.R")        # ~5-10 min
source("r/scripts/phase2/06_hyperparameter_tuning.R")  # ~15-30 min
source("r/scripts/phase2/07_model_evaluation.R")
source("r/scripts/phase2/08_comparative_visualizations.R")
source("r/scripts/phase2/09_impact_tier.R")       # ~5-10 min
source("r/scripts/phase2/10_clustering.R")         # ~5-10 min
```

---

*Last updated: 2026-09-19 | R 4.6.1 | SQLite via RSQLite*
