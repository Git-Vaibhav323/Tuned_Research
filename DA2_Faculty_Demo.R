# =============================================================================
#  DA-2 FACULTY DEMO SCRIPT  |  ResearchPilot Project
#  Paste each SECTION into R console one at a time during the demo.
#  Every section is self-contained and prints/plots results immediately.
# =============================================================================

# ---- SETUP (run this once at the very start) ---------------------------------
setwd("e:/Tuned_Research")          # set project root
cat("Working directory:", getwd(), "\n")


# =============================================================================
# SECTION 1 — DATABASE CONNECTIVITY  (Rubric: DB connectivity + data retrieval)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 1: Database Connectivity (SQLite via RSQLite)\n")
cat("============================================================\n")

library(RSQLite)

# Connect to SQLite database
con <- dbConnect(SQLite(), "database/researchpilot_r.db")
cat("Connected to: database/researchpilot_r.db\n")
cat("Tables found:", paste(dbListTables(con), collapse = ", "), "\n\n")

# Q1 – How many papers per Open-Access category?
cat("Q1 — Paper count per OA category:\n")
q1 <- dbGetQuery(con, "
  SELECT oa_category, COUNT(*) AS n_papers
  FROM papers
  GROUP BY oa_category
  ORDER BY n_papers DESC
")
print(q1)

# Q2 – Top 5 most-cited papers
cat("\nQ2 — Top 5 most-cited papers:\n")
q2 <- dbGetQuery(con, "
  SELECT paper_id, title, cited_by_count, publication_year, oa_category
  FROM papers
  ORDER BY cited_by_count DESC
  LIMIT 5
")
print(q2[, c("paper_id","cited_by_count","publication_year","oa_category","title")])

# Q3 – Papers published in last 3 years (2023-2026)
cat("\nQ3 — Papers published 2023–2026:\n")
q3 <- dbGetQuery(con, "
  SELECT publication_year, COUNT(*) AS n
  FROM papers
  WHERE publication_year >= 2023
  GROUP BY publication_year
  ORDER BY publication_year
")
print(q3)

# Q4 – Average citation count per OA category
cat("\nQ4 — Average citations per OA category:\n")
q4 <- dbGetQuery(con, "
  SELECT oa_category,
         ROUND(AVG(cited_by_count), 2) AS avg_citations,
         COUNT(*) AS n
  FROM papers
  GROUP BY oa_category
  ORDER BY avg_citations DESC
")
print(q4)

# Q5 – Query the ML features table
cat("\nQ5 — First 5 rows from ml_features table:\n")
q5 <- dbGetQuery(con, "SELECT * FROM ml_features LIMIT 5")
print(q5[, 1:8])

dbDisconnect(con)
cat("\nDatabase connection closed.\n")


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING  (Rubric: Feature engineering)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 2: Feature Engineering\n")
cat("============================================================\n")

# Load the enriched feature set
features_enriched <- read.csv("data/ml_r/enriched_features_v2.csv")
cat("Enriched dataset:", nrow(features_enriched), "papers x",
    ncol(features_enriched), "features\n\n")

# Show feature categories we engineered
cat("Feature categories engineered:\n")
cat("  1. Text-length features  : title_length, abstract_length,\n")
cat("                             title_word_count, abstract_word_count\n")
cat("  2. Richness ratios       : title_to_abstract_ratio, text_richness,\n")
cat("                             abstract_keyword_overlap\n")
cat("  3. Metadata features     : paper_age, recency_score,\n")
cat("                             citation_log, citation_per_year\n")
cat("  4. TF-IDF term features  : tf_learning, tf_deep, tf_model, ...\n")
cat("  5. Publisher one-hot     : pub_ieee, pub_mdpi, pub_springer, ...\n")
cat("  6. Concept features      : concept_count, cn_medicine, ...\n\n")

# Show summary of key engineered features
key_feats <- c("title_length","abstract_length","text_richness",
               "recency_score","paper_age","concept_count","keyword_count")
cat("Summary of key engineered features:\n")
print(round(sapply(features_enriched[, key_feats], summary), 3))

# Show class distribution of target variable
cat("\nTarget variable (oa_category) distribution:\n")
print(table(features_enriched$oa_category))
cat("Class proportions:\n")
print(round(prop.table(table(features_enriched$oa_category)) * 100, 1))


# =============================================================================
# SECTION 3 — FEATURE SELECTION  (Rubric: Feature selection)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 3: Feature Selection\n")
cat("============================================================\n")

selected <- read.csv("data/ml_r/selected_features_v2.csv")
cat("After selection:", ncol(selected) - 1, "features retained",
    "(from", ncol(features_enriched) - 1, "original)\n")

# Show the top 20 selected features
all_feats <- setdiff(colnames(selected), "oa_category")
cat("\nTop 20 selected features (by name):\n")
print(head(all_feats, 20))

# Visualize feature selection agreement plot (already generated)
cat("\nOpening feature selection agreement plot...\n")
if (file.exists("reports/figures/phase2_r/feature_selection_agreement.png")) {
  shell.exec(normalizePath("reports/figures/phase2_r/feature_selection_agreement.png"))
} else {
  cat("Plot file: reports/figures/phase2_r/feature_selection_agreement.png\n")
}

# Visualize MI scores
cat("Opening mutual information scores plot...\n")
shell.exec(normalizePath("reports/figures/phase2_r/feature_mi_scores.png"))


# =============================================================================
# SECTION 4 — ML ALGORITHMS (10+)  (Rubric: 10-15 ML/DL algorithms)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 4: 12 ML Algorithms Implemented\n")
cat("============================================================\n")

metrics <- read.csv("data/ml_r/model_metrics_v2.csv")

# Show all unique models implemented
models_list <- unique(metrics$model)
cat("Models implemented (", length(models_list), "total):\n")
for (i in seq_along(models_list)) {
  cat(sprintf("  %2d. %s\n", i, models_list[i]))
}

# Show loaded model objects from RDS
mods <- readRDS("data/ml_r/model_objects_v2.rds")
cat("\nModels stored as R objects:", paste(names(mods), collapse = ", "), "\n")

# Test results table (val + test splits)
test_results <- metrics[metrics$split == "test", ]
cat("\nTest-set performance table (all models):\n")
print(test_results[order(-test_results$f1_macro),
                   c("model","accuracy","precision","recall","f1_macro","roc_auc")],
      row.names = FALSE)


# =============================================================================
# SECTION 5 — HYPERPARAMETER TUNING  (Rubric: Hyperparameter tuning)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 5: Hyperparameter Tuning\n")
cat("============================================================\n")

tuning <- read.csv("data/ml_r/tuning_results_v2.csv")
cat("Tuned models:\n")
tuned_test <- tuning[tuning$split == "test", ]
print(tuned_test[, c("model","best_params","cv_f1_macro","f1_macro")],
      row.names = FALSE)

# Show improvement: baseline vs tuned
lb <- read.csv("reports/tables/r_model_leaderboard_v2.csv")
cat("\nLeaderboard — Top 10 models (ranked by F1):\n")
top10 <- head(lb[order(lb$rank), ], 10)
print(top10[, c("rank","model","model_type","acc","f1","roc","best_params")],
      row.names = FALSE)

# Open baseline vs tuned comparison plot
cat("\nOpening baseline vs tuned comparison plot...\n")
shell.exec(normalizePath("reports/figures/phase2_r/v2_08_baseline_vs_tuned.png"))


# =============================================================================
# SECTION 6 — COMPARATIVE ANALYSIS  (Rubric: Comparative performance analysis)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 6: Comparative Performance Analysis\n")
cat("============================================================\n")

# Full leaderboard sorted by F1
cat("Full leaderboard (best models by F1 score on test set):\n")
lb_sorted <- lb[lb$split == "test", ]
lb_sorted <- lb_sorted[order(-lb_sorted$f1), ]
print(lb_sorted[, c("rank","model","model_type","acc","f1","roc")],
      row.names = FALSE)

# Best model summary
best <- lb_sorted[1, ]
cat(sprintf(
  "\nBEST MODEL: %s\n  Accuracy : %.4f\n  F1 Macro : %.4f\n  ROC-AUC  : %.4f\n  Params   : %s\n",
  best$model, best$acc, best$f1, best$roc, best$best_params
))

# Impact-tier secondary task
impact <- read.csv("data/ml_r/impact_tier_metrics_v2.csv")
cat("\nSecondary Task — Impact Tier Classification (test set):\n")
print(impact[impact$split == "test",
             c("model","accuracy","f1_macro","roc_auc")],
      row.names = FALSE)


# =============================================================================
# SECTION 7 — VISUALIZATIONS  (Rubric: Comparative visualizations)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 7: All Comparative Visualizations\n")
cat("============================================================\n")

# Define all plots
plots <- list(
  "1. Accuracy Comparison (all models)"      = "reports/figures/phase2_r/v2_01_accuracy_comparison.png",
  "2. F1 Score Comparison (all models)"      = "reports/figures/phase2_r/v2_02_f1_comparison.png",
  "3. ROC-AUC Comparison"                    = "reports/figures/phase2_r/v2_04_roc_auc_comparison.png",
  "4. ROC Curves OVR (best models)"          = "reports/figures/phase2_r/03_roc_curves_ovr.png",
  "5. Precision-Recall-F1 Grouped Bars"      = "reports/figures/phase2_r/v2_05_precision_recall_f1.png",
  "6. Confusion Matrix — Random Forest"      = "reports/figures/phase2_r/v2_cm_random_forest.png",
  "7. Confusion Matrix — XGBoost"            = "reports/figures/phase2_r/v2_cm_xgboost.png",
  "8. Feature Importance (enriched features)"= "reports/figures/phase2_r/v2_07_feature_importance_enriched.png",
  "9. Baseline vs Tuned Comparison"          = "reports/figures/phase2_r/v2_08_baseline_vs_tuned.png",
  "10. Old vs New Feature Set"               = "reports/figures/phase2_r/v2_03_old_vs_new_comparison.png",
  "11. Clustering PCA Scatter"               = "reports/figures/phase2_r/clustering_pca_scatter.png",
  "12. Silhouette vs K"                      = "reports/figures/phase2_r/clustering_silhouette_vs_k.png",
  "13. Impact Class Distribution"            = "reports/figures/phase2_r/v2_impact_class_distribution.png",
  "14. Impact Model Comparison"              = "reports/figures/phase2_r/v2_impact_model_comparison.png",
  "15. DB Papers by Year (OA breakdown)"     = "reports/figures/phase2_r/db_papers_by_year_oa.png"
)

# List all plots
cat("Available visualizations:\n")
for (nm in names(plots)) cat(" •", nm, "\n")

# ---- OPEN INDIVIDUAL PLOTS ---
# Uncomment whichever you want to show during demo:

# shell.exec(normalizePath("reports/figures/phase2_r/v2_01_accuracy_comparison.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_02_f1_comparison.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_04_roc_auc_comparison.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/03_roc_curves_ovr.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_05_precision_recall_f1.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_cm_random_forest.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_cm_xgboost.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_07_feature_importance_enriched.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/v2_08_baseline_vs_tuned.png"))
# shell.exec(normalizePath("reports/figures/phase2_r/clustering_pca_scatter.png"))

# Open ALL plots at once (remove # below to use)
for (p in plots) {
  if (file.exists(p)) shell.exec(normalizePath(p))
  Sys.sleep(0.3)
}


# =============================================================================
# SECTION 8 — LIVE RE-PLOT IN R  (shows R can generate charts on the fly)
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 8: Live Chart Generation in R\n")
cat("============================================================\n")

library(ggplot2)

# -- Plot A: F1 Score comparison bar chart (live re-plot from CSV) --
lb_test <- lb[lb$split == "test", ]
lb_test$model <- factor(lb_test$model,
                        levels = lb_test$model[order(lb_test$f1)])

p_f1 <- ggplot(lb_test, aes(x = model, y = f1, fill = model_type)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = round(f1, 3)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("baseline" = "#4472C4", "tuned" = "#ED7D31")) +
  labs(title = "F1-Macro Score by Model (Test Set)",
       subtitle = "Orange = hyperparameter-tuned | Blue = baseline",
       x = NULL, y = "F1 Macro") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom") +
  xlim(0, 1)

print(p_f1)   # shows in RStudio Plots pane
cat("F1 comparison chart rendered in Plots pane.\n")

# -- Plot B: ROC-AUC comparison --
lb_roc <- lb_test[!is.na(lb_test$roc), ]
lb_roc$model <- factor(lb_roc$model, levels = lb_roc$model[order(lb_roc$roc)])

p_roc <- ggplot(lb_roc, aes(x = model, y = roc, fill = model_type)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "red") +
  geom_text(aes(label = round(roc, 3)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("baseline" = "#4472C4", "tuned" = "#ED7D31")) +
  labs(title = "ROC-AUC Score by Model (Test Set)",
       subtitle = "Dashed line = random classifier (0.5)",
       x = NULL, y = "ROC-AUC") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

print(p_roc)
cat("ROC-AUC chart rendered in Plots pane.\n")


# =============================================================================
# SECTION 9 — UNSUPERVISED CLUSTERING SUMMARY
# =============================================================================
cat("\n============================================================\n")
cat("SECTION 9: Unsupervised Clustering (Topic Clusters)\n")
cat("============================================================\n")

clusters <- read.csv("data/ml_r/cluster_keywords.csv")
cat("K-means clustering produced", nrow(clusters), "topic clusters:\n\n")
for (i in 1:nrow(clusters)) {
  cat(sprintf("  Cluster %d (%d papers, %.1f%%):\n    Keywords: %s\n\n",
              clusters$cluster[i],
              clusters$n_papers[i],
              clusters$pct_papers[i],
              clusters$top_terms[i]))
}

shell.exec(normalizePath("reports/figures/phase2_r/clustering_pca_scatter.png"))
shell.exec(normalizePath("reports/figures/phase2_r/clustering_silhouette_vs_k.png"))


# =============================================================================
# SECTION 10 — QUICK SUMMARY FOR FACULTY
# =============================================================================
cat("\n============================================================\n")
cat("DA-2 SUMMARY\n")
cat("============================================================\n")

lb_best <- lb[lb$split == "test" & lb$rank == 1, ]

cat(sprintf("
Dataset           : 2,000 research papers (OpenAlex API)
Database          : SQLite (RSQLite) — 4 tables, SQL queries live
Features original : 27  raw columns
Features engineered: 206 (TF-IDF, publisher, recency, ratios, concepts)
Features selected : %d  (Mutual Info + Correlation filter)
ML algorithms     : 12  (LR, Elastic Net, DT, RF, XGBoost, GBM,
                         AdaBoost, SVM-RBF, SVM-Linear, NB, LDA, MLP)
Hyperparameter tuning: Grid search on RF, XGBoost, SVM, GBM
Best model        : %s
  → Accuracy : %.4f
  → F1 Macro : %.4f
  → ROC-AUC  : %.4f
  → Params   : %s
Visualizations    : 15 plots (ROC, CM, Feature Importance, PR, Bar charts)
Secondary task    : Impact-tier classification (4-class)
Clustering        : K-means, k=3, PCA-projected scatter
",
  ncol(selected) - 1,
  lb_best$model[1], lb_best$acc[1], lb_best$f1[1],
  lb_best$roc[1], lb_best$best_params[1]
))

cat("============================================================\n")
cat("END OF DEMO — All rubric items covered.\n")
cat("============================================================\n")
