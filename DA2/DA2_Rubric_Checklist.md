# DA2 Rubric Checklist — R Implementation (Enhanced Pipeline v2)
**Project**: ResearchPilot | **Language**: R | **Database**: SQLite
**Version**: v2 (Enriched Features — TF-IDF + Domain + Publisher)

---

## ✅ 1. Feature Engineering & Feature Selection — 1 mark

### Feature Engineering

| Evidence | Location |
|----------|----------|
| **Script v1** | `r/scripts/phase2/01_feature_engineering.R` |
| **Script v2 (enhanced)** | `r/scripts/phase2/00_enriched_features.R` |
| **Output v1** | `data/ml_r/engineered_features.csv` (2000 × 38, 11 new features) |
| **Output v2** | `data/ml_r/enriched_features_v2.csv` (2000 × 206, 205 features) |
| **v2 Feature groups** | 12 structural + 25 domain + 18 publisher/DOI + 150 TF-IDF |
| **Key new features** | pub_ieee, pub_mdpi, dom_medical, dom_llm, tf_* (TF-IDF terms) |
| **Report** | `reports/phase2_r/feature_engineering_report.md` |

### Feature Selection

| Evidence | Location |
|----------|----------|
| **Script v1** | `r/scripts/phase2/02_feature_selection.R` |
| **Script v2** | `r/scripts/phase2/02_feature_selection_v2.R` |
| **Method 1** | Near-Zero Variance → removed 41 of 205 |
| **Method 2** | Pairwise Correlation (r > 0.95, TF-IDF only) |
| **Method 3** | Random Forest Gini Importance → top 80 retained |
| **Output v2** | `data/ml_r/selected_features_v2.csv` (80 features + target) |
| **Feature names** | `data/ml_r/selected_feature_names.rds` |
| **Top 2 features** | pub_ieee (37.84), pub_mdpi (34.89) |
| **Before/After** | 205 → 80 features (61% reduction) |
| **Report** | `reports/phase2_r/feature_selection_v2_report.md` |
| **Figure** | `reports/figures/phase2_r/feature_importance_enriched.png` |
| **Figure** | `reports/figures/phase2_r/v2_07_feature_importance_enriched.png` |

---

## ✅ 2. Database Connectivity and Data Retrieval using R — 2 marks

| Evidence | Location |
|----------|----------|
| **Database** | `database/researchpilot_r.db` (SQLite, 4.2 MB) |
| **Packages** | `DBI` + `RSQLite` |
| **Setup script** | `r/scripts/phase2/03_database_setup.R` |
| **Query script** | `r/scripts/phase2/04_database_queries.R` |
| **Tables** | `papers` (2000), `ml_features` (2000), `oa_features` (2000), `model_results` (live) |
| **Query 1** | Most recent papers (2024+) → `q1_most_recent_papers.csv` |
| **Query 2** | Most cited papers → `q2_most_cited_papers.csv` |
| **Query 3** | OA distribution (COUNT, AVG, Window function) → `q3_oa_category_distribution.csv` |
| **Query 4** | Papers per year → `q4_papers_by_year.csv` |
| **Query 5** | OA by year cross-tab (CASE WHEN) → `q5_oa_by_year.csv` |
| **Query 6** | High-richness papers (JOIN) → `q6_high_richness_papers.csv` |
| **Query 7** | Avg features by OA class (JOIN + GROUP BY) → `q7_avg_features_by_oa.csv` |
| **DB Visualisation** | `reports/figures/phase2_r/db_papers_by_year_oa.png` |
| **Connection demo** | `dbConnect()` → `dbListTables()` → `dbGetQuery()` → `dbDisconnect()` |

---

## ✅ 3. Implementation of 10–15 ML Algorithms — 3 marks

### OA Classification (Primary Task)

| # | Algorithm | Family | R Package | Script |
|---|-----------|--------|-----------|--------|
| 1 | Logistic Regression (multinomial) | Linear | nnet | 05_ml_models_v2.R |
| 2 | Elastic Net | Regularised Linear | glmnet | 05_ml_models_v2.R |
| 3 | CART Decision Tree | Tree | rpart | 05_ml_models_v2.R |
| 4 | Random Forest (500 trees) | Ensemble Bagging | randomForest | 05_ml_models_v2.R |
| 5 | XGBoost (150 rounds) | Extreme Gradient Boosting | xgboost | 05_ml_models_v2.R |
| 6 | Gradient Boosting Machine | Gradient Boosting | gbm | 05_ml_models_v2.R |
| 7 | AdaBoost (100 iters) | Adaptive Boosting | adabag | 05_ml_models_v2.R |
| 8 | SVM (RBF kernel) | Kernel | e1071 | 05_ml_models_v2.R |
| 9 | SVM (Linear kernel) | Kernel | e1071 | 05_ml_models_v2.R |
| 10 | Naive Bayes (Gaussian) | Probabilistic | e1071 | 05_ml_models_v2.R |
| 11 | LDA | Discriminant Analysis | MASS | 05_ml_models_v2.R |
| 12 | KNN (k=7) | Instance-Based | kknn | 05_ml_models_v2.R |
| 13 | MLP Neural Network (size=100) | Neural Network | nnet | 05_ml_models_v2.R |

**Total: 13 algorithms (7 families) — exceeds requirement of 10–15**

### Evidence files
- `data/ml_r/model_metrics_v2.csv` — all metrics
- `data/ml_r/model_objects_v2.rds` — saved model objects
- `reports/tables/r_model_leaderboard_v2_baseline.csv` — baseline leaderboard

---

## ✅ 4. Hyperparameter Tuning and Model Optimization — 1 mark

| Evidence | Location |
|----------|----------|
| **Script** | `r/scripts/phase2/06_hyperparameter_tuning_v2.R` |
| **Method** | Grid search + 5-fold CV (training set ONLY) |
| **RF grid** | ntree ∈ {300,500,800} × mtry ∈ {5,8,12,15} = 12 combinations |
| **XGBoost grid** | depth ∈ {4,6,8} × eta ∈ {0.05,0.10} × nrounds ∈ {150,300} = 12 |
| **SVM grid** | cost ∈ {1,10,100} × gamma ∈ {0.001,0.01,0.1} = 9 combinations |
| **RF best** | ntree=500, mtry=8, test-acc=0.5615 |
| **SVM best** | cost=10, gamma=0.001, CV-F1=0.6065, test-acc=0.5748 |
| **Output** | `data/ml_r/tuning_results_v2.csv` |
| **Leaderboard** | `reports/tables/r_tuning_leaderboard_v2.csv` |
| **Saved models** | `rf_tuned_v2.rds`, `xgb_tuned_v2.rds`, `svm_tuned_v2.rds` |
| **Test integrity** | Test set never used during tuning |

---

## ✅ 5. Comparative Performance Analysis — 1 mark

| Evidence | Location |
|----------|----------|
| **Script** | `r/scripts/phase2/07_model_evaluation_v2.R` |
| **Metrics** | Accuracy, Macro-Precision, Macro-Recall, Macro-F1, OVR ROC-AUC |
| **Leaderboard CSV** | `reports/tables/r_model_leaderboard_v2.csv` (15 models) |
| **Leaderboard MD** | `reports/tables/r_model_leaderboard_v2.md` |
| **Best Accuracy** | 0.5748 (svm_tuned) — up from 0.4452 (+13%) |
| **Best F1** | 0.5701 (svm_tuned) — up from 0.3783 (+19%) |
| **Best AUC** | 0.7659 (mlp) — up from 0.5714 (+19%) |
| **v1 vs v2** | `reports/figures/phase2_r/v2_03_old_vs_new_comparison.png` |
| **SQLite update** | model_results table updated with all v2 results |

---

## ✅ 6. Comparative Visualizations — 1 mark

### v2 Enhanced Figures (new)

| Figure | File |
|--------|------|
| Accuracy comparison (15 models) | `v2_01_accuracy_comparison.png` |
| Macro-F1 comparison | `v2_02_f1_comparison.png` |
| Old vs New feature set | `v2_03_old_vs_new_comparison.png` |
| ROC-AUC bar chart | `v2_04_roc_auc_comparison.png` |
| Precision/Recall/F1 grouped | `v2_05_precision_recall_f1.png` |
| Confusion matrix (RF) | `v2_cm_random_forest.png` |
| Confusion matrix (XGBoost) | `v2_cm_xgboost.png` |
| Feature importance (enriched) | `v2_07_feature_importance_enriched.png` |
| Baseline vs Tuned | `v2_08_baseline_vs_tuned.png` |
| Impact-tier comparison | `v2_impact_model_comparison.png` |
| Impact-tier distribution | `v2_impact_class_distribution.png` |

### v1 Original Figures (preserved)

| Figure | File |
|--------|------|
| Feature selection agreement | `feature_selection_agreement.png` |
| MI scores | `feature_mi_scores.png` |
| ROC curves (OVR, 3 panels) | `03_roc_curves_ovr.png` |
| Precision-Recall curves | `05_precision_recall_curves.png` |
| Confusion matrix (best model) | `06a_confusion_matrix_best_model.png` |
| Feature importance (v1 RF) | `07_feature_importance_rf.png` |
| Clustering PCA scatter | `clustering_pca_scatter.png` |
| Silhouette vs k | `clustering_silhouette_vs_k.png` |
| DB papers by year | `db_papers_by_year_oa.png` |

**All figures generated in R using ggplot2**  
**Total figures**: 29 (20 v1 + 9 v2)  
**Directory**: `reports/figures/phase2_r/`

---

## ✅ 7. Progress Demonstration and Documentation — 1 mark

| Evidence | Location |
|----------|----------|
| **DA2 Final Report** | `DA2/DA2_R_Final_Report.md` (25 sections, 700+ lines) |
| **Faculty Demo Script** | `DA2/DA2_R_Demo.md` (13 steps, ~9 min demo) |
| **Rubric Checklist** | `DA2/DA2_Rubric_Checklist.md` (this file) |
| **Feature Engineering Report** | `reports/phase2_r/feature_engineering_report.md` |
| **Feature Selection Report v2** | `reports/phase2_r/feature_selection_v2_report.md` |
| **Model Leaderboard v2** | `reports/tables/r_model_leaderboard_v2.md` |
| **Requirements** | `r/requirements.md` |
| **Pipeline Runner** | `r/run_enhanced_pipeline.R` |

---

## Complete R Script Inventory (v2 Pipeline)

| Script | Purpose | Run Time |
|--------|---------|----------|
| `r/scripts/phase2/00_enriched_features.R` | Build 205-feature enriched matrix | ~2 min |
| `r/scripts/phase2/01_feature_engineering.R` | v1 structural features | ~5 sec |
| `r/scripts/phase2/02_feature_selection.R` | v1 feature selection | ~15 sec |
| `r/scripts/phase2/02_feature_selection_v2.R` | v2 RF-based selection → 80 features | ~1 min |
| `r/scripts/phase2/03_database_setup.R` | Create SQLite + load 4 tables | ~5 sec |
| `r/scripts/phase2/04_database_queries.R` | 7 SQL queries + visualisation | ~5 sec |
| `r/scripts/phase2/05_ml_models.R` | v1 ML pipeline (11 models) | ~5 min |
| `r/scripts/phase2/05_ml_models_v2.R` | v2 ML pipeline (13 models) | ~5 min |
| `r/scripts/phase2/06_hyperparameter_tuning.R` | v1 tuning | ~20 min |
| `r/scripts/phase2/06_hyperparameter_tuning_v2.R` | v2 tuning (RF+XGB+SVM) | ~30 min |
| `r/scripts/phase2/07_model_evaluation.R` | v1 leaderboard | ~5 sec |
| `r/scripts/phase2/07_model_evaluation_v2.R` | v2 leaderboard | ~5 sec |
| `r/scripts/phase2/08_comparative_visualizations.R` | v1 figures | ~30 sec |
| `r/scripts/phase2/08_visualizations_v2.R` | v2 figures (9 new) | ~30 sec |
| `r/scripts/phase2/09_impact_tier.R` | v1 impact tier | ~5 min |
| `r/scripts/phase2/09_impact_tier_v2.R` | v2 impact tier | ~5 min |
| `r/scripts/phase2/10_clustering.R` | TF-IDF + KMeans clustering | ~5 min |

---

## How to Run the Full v2 Pipeline

```r
setwd("E:/Tuned_Research")

# Option A: Run the master script
source("r/run_enhanced_pipeline.R")

# Option B: Run each step individually
source("r/scripts/phase2/00_enriched_features.R")       # features
source("r/scripts/phase2/02_feature_selection_v2.R")    # selection
source("r/scripts/phase2/05_ml_models_v2.R")            # training
source("r/scripts/phase2/06_hyperparameter_tuning_v2.R") # tuning (~30 min)
source("r/scripts/phase2/07_model_evaluation_v2.R")     # leaderboard
source("r/scripts/phase2/08_visualizations_v2.R")       # figures
source("r/scripts/phase2/09_impact_tier_v2.R")          # impact
source("r/scripts/phase2/10_clustering.R")              # clustering
```

---

*Last updated: September 2026 | R 4.6.1 | Enhanced Pipeline v2*
