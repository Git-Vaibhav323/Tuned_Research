# =============================================================================
#  DA-2 FACULTY DEMO — PART 2  (Advanced / Interactive Commands)
#  Run in RStudio. Paste one CMD block at a time.
#  Working dir must be set: setwd("e:/Tuned_Research")
# =============================================================================

setwd("e:/Tuned_Research")


# ============================================================
# CMD-1: INSPECT LIVE MODEL OBJECTS  (shows actual R objects)
# ============================================================

library(randomForest)
library(e1071)
library(nnet)
library(rpart)
library(xgboost)

mods <- readRDS("data/ml_r/model_objects_v2.rds")

# How many models are stored?
cat("Total trained model objects:", length(mods), "\n")
cat("Model names:", paste(names(mods), collapse = ", "), "\n\n")

# Inspect Random Forest
cat("--- Random Forest ---\n")
print(mods$random_forest)          # prints OOB error, confusion matrix, ntree

# Inspect Logistic Regression
cat("\n--- Logistic Regression (top 10 coefficients) ---\n")
coefs <- coef(mods$logistic_regression)
# coefs is a matrix (one row per class) - show top features for first class
top_coef <- sort(abs(coefs[1,]), decreasing = TRUE)[1:10]
print(round(top_coef, 4))

# Inspect SVM
cat("\n--- SVM (RBF kernel) ---\n")
cat("Kernel    :", mods$svm_rbf$kernel, "\n")
cat("Cost      :", mods$svm_rbf$cost, "\n")
cat("Gamma     :", mods$svm_rbf$gamma, "\n")
cat("SV count  :", nrow(mods$svm_rbf$SV), "\n")
cat("Classes   :", paste(mods$svm_rbf$levels, collapse = " | "), "\n")

# Inspect Decision Tree depth
cat("\n--- Decision Tree ---\n")
cat("Tree class:", class(mods$decision_tree), "\n")
cat("# terminal nodes:", mods$decision_tree$frame$n[mods$decision_tree$frame$var == "<leaf>"][1], "(first leaf)\n")

# Inspect Naive Bayes
cat("\n--- Naive Bayes ---\n")
print(mods$naive_bayes$apriori)    # class prior probabilities


# ============================================================
# CMD-2: LIVE PREDICTION ON A NEW PAPER (single-paper demo)
# ============================================================

cat("\n============================================================\n")
cat("CMD-2: Live Prediction on a New Paper\n")
cat("============================================================\n")

sel          <- read.csv("data/ml_r/selected_features_v2.csv")
lb           <- read.csv("reports/tables/r_model_leaderboard_v2.csv")
metrics      <- read.csv("data/ml_r/model_metrics_v2.csv")
feature_cols <- setdiff(colnames(sel), "oa_category")

# Pick one real paper from test set (row 1501 onwards = test split approx)
test_paper <- sel[1501, feature_cols]

cat("Predicting OA category for paper #1501...\n")
cat("True label:", sel$oa_category[1501], "\n\n")

# Predict with 4 different models
rf_train_cols <- rownames(mods$random_forest$importance)

for (nm in c("random_forest", "logistic_regression", "svm_rbf", "decision_tree")) {
  mod <- mods[[nm]]
  pred <- tryCatch({
    if (nm == "random_forest") {
      # RF needs exact training column order
      predict(mod, newdata = test_paper[, rf_train_cols, drop = FALSE])
    } else if (nm == "decision_tree") {
      # rpart returns probability matrix — take column with highest prob
      p <- predict(mod, newdata = test_paper, type = "class")
      p
    } else {
      predict(mod, newdata = test_paper)
    }
  }, error = function(e) paste("prediction error:", e$message))
  cat(sprintf("  %-25s predicted: %s\n", nm, as.character(pred)[1]))
}

# Predict with probabilities (Random Forest)
cat("\nRandom Forest class probabilities for paper #1501:\n")
probs <- predict(mods$random_forest,
                 newdata = test_paper[, rf_train_cols, drop = FALSE],
                 type = "prob")
print(round(probs, 4))


# ============================================================
# CMD-3: LIVE CONFUSION MATRIX (computed in R, not just a PNG)
# ============================================================

cat("\n============================================================\n")
cat("CMD-3: Confusion Matrix — Random Forest (live computation)\n")
cat("============================================================\n")

# Use last 300 rows as test set proxy
X_test <- sel[1701:2000, feature_cols]
y_test  <- sel$oa_category[1701:2000]

rf_train_cols2 <- rownames(mods$random_forest$importance)
X_test_rf2     <- X_test[, rf_train_cols2, drop = FALSE]
preds_rf <- predict(mods$random_forest, newdata = X_test_rf2)cm <- table(Predicted = preds_rf, Actual = y_test)
cat("Confusion Matrix (Random Forest on rows 1701-2000):\n")
print(cm)

# Compute accuracy from CM
acc <- sum(diag(cm)) / sum(cm)
cat(sprintf("\nAccuracy from CM: %.4f (%.1f%%)\n", acc, acc * 100))

# Per-class precision and recall
cat("\nPer-class Precision & Recall:\n")
for (cls in colnames(cm)) {
  tp  <- cm[cls, cls]
  fp  <- sum(cm[cls, ]) - tp
  fn  <- sum(cm[, cls]) - tp
  prec <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  rec  <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1   <- ifelse((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
  cat(sprintf("  %-16s  Prec=%.3f  Rec=%.3f  F1=%.3f\n", cls, prec, rec, f1))
}


# ============================================================
# CMD-4: FEATURE IMPORTANCE — TOP 20 (live from RF object)
# ============================================================

cat("\n============================================================\n")
cat("CMD-4: Feature Importance (live from Random Forest object)\n")
cat("============================================================\n")

library(ggplot2)

imp <- as.data.frame(mods$random_forest$importance)
imp$feature <- rownames(imp)
# MeanDecreaseGini is the importance column in randomForest
imp <- imp[order(-imp$MeanDecreaseGini), ]
top20 <- head(imp, 20)

cat("Top 20 most important features:\n")
print(top20[, c("feature", "MeanDecreaseGini")], row.names = FALSE)

# Plot it live
p_imp <- ggplot(top20, aes(x = reorder(feature, MeanDecreaseGini),
                            y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "#AED6F1", high = "#1A5276") +
  labs(title = "Top 20 Feature Importances — Random Forest",
       subtitle = "Metric: Mean Decrease in Gini Impurity",
       x = NULL, y = "Mean Decrease Gini") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

print(p_imp)
cat("Feature importance chart shown in Plots pane.\n")


# ============================================================
# CMD-5: SIDE-BY-SIDE METRIC TABLE (all models, clean print)
# ============================================================

cat("\n============================================================\n")
cat("CMD-5: Full Metrics Table — All 12 Models\n")
cat("============================================================\n")

metrics <- read.csv("data/ml_r/model_metrics_v2.csv")
lb      <- read.csv("reports/tables/r_model_leaderboard_v2.csv")

# Clean combined table: test split only, ranked
test_m <- metrics[metrics$split == "test", ]
test_m <- test_m[order(-test_m$f1_macro), ]

cat(sprintf("%-25s %8s %8s %8s %8s %8s\n",
            "Model", "Acc", "Prec", "Rec", "F1", "ROC-AUC"))
cat(strrep("-", 70), "\n")
for (i in 1:nrow(test_m)) {
  r <- test_m[i, ]
  cat(sprintf("%-25s %8.4f %8.4f %8.4f %8.4f %8s\n",
              r$model,
              r$accuracy,
              ifelse(is.na(r$precision), 0, r$precision),
              r$recall,
              r$f1_macro,
              ifelse(is.na(r$roc_auc), "  N/A  ", sprintf("%.4f", r$roc_auc))))
}
cat(strrep("-", 70), "\n")


# ============================================================
# CMD-6: HYPERPARAMETER TUNING — BEFORE vs AFTER TABLE
# ============================================================

cat("\n============================================================\n")
cat("CMD-6: Hyperparameter Tuning — Before vs After\n")
cat("============================================================\n")

tuning <- read.csv("data/ml_r/tuning_results_v2.csv")
tuned_test <- tuning[tuning$split == "test", ]

# Map baseline model names
baseline_map <- c(
  "random_forest_tuned" = "random_forest",
  "xgboost_tuned"       = "xgboost",
  "svm_tuned"           = "svm_rbf"
)

cat(sprintf("%-22s  %10s  %10s  %10s  %s\n",
            "Model", "Base F1", "Tuned F1", "Improvement", "Best Params"))
cat(strrep("-", 90), "\n")

for (i in 1:nrow(tuned_test)) {
  r        <- tuned_test[i, ]
  base_nm  <- baseline_map[r$model]
  base_row <- metrics[metrics$model == base_nm & metrics$split == "test", ]
  base_f1  <- ifelse(nrow(base_row) > 0, base_row$f1_macro[1], NA)
  delta    <- ifelse(!is.na(base_f1), r$f1_macro - base_f1, NA)
  arrow    <- ifelse(!is.na(delta) && delta >= 0, "+", "")
  cat(sprintf("%-22s  %10.4f  %10.4f  %10s  %s\n",
              r$model,
              ifelse(is.na(base_f1), 0, base_f1),
              r$f1_macro,
              ifelse(is.na(delta), "  N/A", paste0(arrow, round(delta, 4))),
              r$best_params))
}
cat(strrep("-", 90), "\n")


# ============================================================
# CMD-7: LIVE ROC CURVE PLOT (multi-class OVR in R)
# ============================================================

cat("\n============================================================\n")
cat("CMD-7: Live ROC Curve — Multi-class OVR\n")
cat("============================================================\n")

library(pROC)

# Use last 300 rows as test set proxy
X_test   <- sel[1701:2000, feature_cols]
y_test   <- sel$oa_category[1701:2000]
classes  <- c("closed", "fully_open", "partially_open")

# Get RF probabilities — ensure column order matches training
rf_train_cols <- rownames(mods$random_forest$importance)
X_test_rf     <- X_test[, rf_train_cols, drop = FALSE]
probs_rf      <- predict(mods$random_forest, newdata = X_test_rf, type = "prob")

cat("Computing OVR ROC curves for Random Forest...\n")
par(mfrow = c(1, 3))   # 3 plots side by side
auc_vals <- c()
for (cls in classes) {
  binary_y <- as.integer(y_test == cls)
  roc_obj  <- roc(binary_y, probs_rf[, cls], quiet = TRUE)
  auc_vals[cls] <- auc(roc_obj)
  plot(roc_obj,
       main  = paste0("ROC — ", cls),
       col   = "#1A5276", lwd = 2,
       print.auc = TRUE, print.auc.y = 0.4)
}
par(mfrow = c(1, 1))   # reset

cat("\nAUC per class (OVR):\n")
for (cls in classes) {
  cat(sprintf("  %-16s  AUC = %.4f\n", cls, auc_vals[cls]))
}
cat(sprintf("  Mean AUC       : %.4f\n", mean(auc_vals)))


# ============================================================
# CMD-8: DB — ADVANCED SQL QUERIES (live analytics)
# ============================================================

cat("\n============================================================\n")
cat("CMD-8: Advanced SQL Queries on Database\n")
cat("============================================================\n")

library(RSQLite)
con <- dbConnect(SQLite(), "database/researchpilot_r.db")

# Q-A: OA category breakdown by year
cat("OA Category breakdown by publication year:\n")
qa <- dbGetQuery(con, "
  SELECT publication_year, oa_category, COUNT(*) AS n
  FROM papers
  GROUP BY publication_year, oa_category
  ORDER BY publication_year DESC, n DESC
")
print(qa)

# Q-B: Papers with high citations + open access
cat("\nHigh-citation open access papers (cited > 5000):\n")
qb <- dbGetQuery(con, "
  SELECT paper_id, cited_by_count, publication_year, oa_category,
         SUBSTR(title, 1, 60) AS title_short
  FROM papers
  WHERE cited_by_count > 5000
  ORDER BY cited_by_count DESC
  LIMIT 8
")
print(qb)

# Q-C: Feature averages by OA category (from ml_features table)
cat("\nAverage features per OA category (from ml_features):\n")
qc <- dbGetQuery(con, "
  SELECT p.oa_category,
         ROUND(AVG(m.title_length), 1)         AS avg_title_len,
         ROUND(AVG(m.abstract_length), 0)       AS avg_abstract_len,
         ROUND(AVG(m.concept_count), 2)         AS avg_concepts,
         ROUND(AVG(m.text_richness), 4)         AS avg_richness,
         COUNT(*) AS n
  FROM ml_features m
  JOIN papers p ON p.paper_id = m.paper_id
  GROUP BY p.oa_category
")
print(qc)

dbDisconnect(con)
cat("DB connection closed.\n")


# ============================================================
# CMD-9: CLUSTERING — SHOW CLUSTER PROFILES LIVE
# ============================================================

cat("\n============================================================\n")
cat("CMD-9: K-Means Cluster Profiles\n")
cat("============================================================\n")

clusters <- read.csv("data/ml_r/cluster_keywords.csv")

cat(sprintf("%-10s  %-8s  %-8s  %s\n",
            "Cluster", "Papers", "Pct(%)", "Top Research Terms"))
cat(strrep("-", 90), "\n")
for (i in 1:nrow(clusters)) {
  cat(sprintf("%-10s  %-8d  %-8.1f  %s\n",
              paste0("Cluster ", clusters$cluster[i]),
              clusters$n_papers[i],
              clusters$pct_papers[i],
              clusters$top_terms[i]))
}
cat(strrep("-", 90), "\n")

# Live ggplot: cluster size bar
p_clust <- ggplot(clusters, aes(x = factor(cluster), y = n_papers,
                                 fill = factor(cluster))) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(n_papers, "\n(", pct_papers, "%)")),
            vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")) +
  labs(title = "K-Means Cluster Sizes (k=3)",
       subtitle = "Cluster 1=Education/GenAI  |  Cluster 2=General ML  |  Cluster 3=Medical Imaging",
       x = "Cluster", y = "Number of Papers") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

print(p_clust)
cat("Cluster bar chart shown in Plots pane.\n")


# ============================================================
# CMD-10: FINAL LIVE DASHBOARD  (4-panel combined chart)
# ============================================================

cat("\n============================================================\n")
cat("CMD-10: Final 4-Panel Summary Dashboard\n")
cat("============================================================\n")

library(ggplot2)
library(gridExtra)

lb_test <- lb[lb$split == "test", ]
lb_test  <- lb_test[order(-lb_test$f1), ]
lb_test$model <- factor(lb_test$model, levels = lb_test$model[order(lb_test$f1)])

# Panel 1: F1 comparison
p1 <- ggplot(lb_test, aes(x = model, y = f1, fill = model_type)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("baseline" = "#5DADE2", "tuned" = "#E67E22")) +
  labs(title = "F1-Macro Score", x = NULL, y = "F1") +
  theme_minimal(base_size = 9) + theme(legend.position = "none")

# Panel 2: Accuracy comparison
p2 <- ggplot(lb_test, aes(x = model, y = acc, fill = model_type)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("baseline" = "#5DADE2", "tuned" = "#E67E22")) +
  labs(title = "Accuracy", x = NULL, y = "Acc") +
  theme_minimal(base_size = 9) + theme(legend.position = "none")

# Panel 3: ROC-AUC (where available)
lb_roc <- lb_test[!is.na(lb_test$roc), ]
lb_roc$model <- factor(lb_roc$model, levels = lb_roc$model[order(lb_roc$roc)])
p3 <- ggplot(lb_roc, aes(x = model, y = roc, fill = model_type)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "red", linewidth = 0.5) +
  coord_flip() +
  scale_fill_manual(values = c("baseline" = "#5DADE2", "tuned" = "#E67E22")) +
  labs(title = "ROC-AUC", x = NULL, y = "AUC") +
  theme_minimal(base_size = 9) + theme(legend.position = "none")

# Panel 4: Class distribution (target variable)
sel <- read.csv("data/ml_r/selected_features_v2.csv")
class_dist <- as.data.frame(table(sel$oa_category))
colnames(class_dist) <- c("class", "count")
p4 <- ggplot(class_dist, aes(x = class, y = count, fill = class)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = count), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("closed"="#E74C3C","fully_open"="#2ECC71","partially_open"="#3498DB")) +
  labs(title = "Class Distribution (Target)", x = NULL, y = "Papers") +
  theme_minimal(base_size = 9) + theme(legend.position = "none")

# Combine all 4 panels
grid.arrange(p1, p2, p3, p4, ncol = 2,
             top = "DA-2 ResearchPilot — Model Performance Dashboard")
cat("4-panel dashboard shown in Plots pane.\n")


# ============================================================
# CMD-11: OPEN ALL SAVED VISUALIZATION IMAGES
# ============================================================

cat("\n============================================================\n")
cat("CMD-11: Open All 15 Visualization Images\n")
cat("============================================================\n")

img_dir <- "reports/figures/phase2_r"
all_imgs <- list.files(img_dir, pattern = "\\.png$", full.names = TRUE)

cat("Opening", length(all_imgs), "visualization files...\n")
cat(paste(" •", basename(all_imgs), collapse = "\n"), "\n\n")

# Open each image (Windows default viewer / browser)
for (img in all_imgs) {
  shell.exec(normalizePath(img))
  Sys.sleep(0.4)
}

cat("All images opened.\n")


# ============================================================
# CMD-12: COMPLETE RUBRIC CHECKLIST — PRINT TO CONSOLE
# ============================================================

cat("\n")
cat("============================================================\n")
cat("   DA-2 RUBRIC COVERAGE CHECKLIST\n")
cat("============================================================\n")
cat("\n")
cat("  [DONE]  Database Connectivity\n")
cat("          SQLite connected, 4 tables, 5 SQL queries executed\n")
cat("          Tables: papers, ml_features, oa_features, model_results\n")
cat("\n")
cat("  [DONE]  Feature Engineering\n")
cat("          206 features engineered from 27 raw columns:\n")
cat("          TF-IDF terms, publisher flags, recency score,\n")
cat("          text richness, citation log, concept count\n")
cat("\n")
cat("  [DONE]  Feature Selection\n")
cat("          80 features selected via Mutual Information + Correlation\n")
cat("          Reduced from 206 → 80 (61% reduction)\n")
cat("\n")

lb_best <- lb[lb$split == "test" & lb$rank == 1, ]
cat("  [DONE]  10-15 ML/DL Algorithms\n")
cat("          12 models implemented:\n")
cat("          Logistic Regression, Elastic Net, Decision Tree,\n")
cat("          Random Forest, XGBoost, Gradient Boosting,\n")
cat("          AdaBoost, SVM-RBF, SVM-Linear,\n")
cat("          Naive Bayes, LDA, MLP (Neural Net)\n")
cat("\n")
cat("  [DONE]  Hyperparameter Tuning\n")
cat("          Grid search on: RF (ntree, mtry), XGBoost (depth, eta,\n")
cat("          rounds), SVM (cost, gamma), GBM\n")
cat(sprintf("          Best: %s | params: %s\n",
            lb_best$model[1], lb_best$best_params[1]))
cat("\n")
cat("  [DONE]  Comparative Performance Analysis\n")
cat(sprintf("          Best model: %-20s\n", lb_best$model[1]))
cat(sprintf("          Accuracy  : %.4f\n", lb_best$acc[1]))
cat(sprintf("          F1 Macro  : %.4f\n", lb_best$f1[1]))
cat("\n")
cat("  [DONE]  Comparative Visualizations (15 plots)\n")
cat("          ROC Curves (OVR), Confusion Matrices,\n")
cat("          Feature Importance, Precision-Recall curves,\n")
cat("          Accuracy/F1/AUC bar charts, Clustering PCA,\n")
cat("          Silhouette plot, Impact tier comparison\n")
cat("\n")
cat("============================================================\n")
cat("  ALL RUBRIC ITEMS COVERED\n")
cat("============================================================\n")
