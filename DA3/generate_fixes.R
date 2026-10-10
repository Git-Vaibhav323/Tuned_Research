# =============================================================================
# ResearchPilot DA3 — Audit Fix Script
# Generates:
#   1. Actual LDA confusion matrix (for dashboard replacement)
#   2. Actual ROC curves from RF + LDA posterior probabilities
#   3. Critical Difference diagram from saved Nemenyi results
#   4. Verified RF feature importance from saved RDS
#   Outputs saved to DA3/figures/ and DA3/tables/
# =============================================================================
setwd("e:/Tuned_Research")
set.seed(42)

cat("============================================================\n")
cat("  DA3 Audit Fix Script\n")
cat("============================================================\n\n")

fig_dir <- "DA3/figures"
tab_dir <- "DA3/tables"
dir.create(fig_dir, showWarnings=FALSE, recursive=TRUE)
dir.create(tab_dir, showWarnings=FALSE, recursive=TRUE)

# ── Load packages ─────────────────────────────────────────────────────────────
for (p in c("MASS","randomForest","ggplot2","dplyr","tidyr","readr","scales")) {
  if (!requireNamespace(p, quietly=TRUE))
    install.packages(p, repos="https://cloud.r-project.org")
  suppressPackageStartupMessages(library(p, character.only=TRUE))
}

# ── Load saved objects ────────────────────────────────────────────────────────
cat("Loading saved objects...\n")
splits <- readRDS("data/ml_r/ml_data_splits_v2.rds")
mo     <- readRDS("data/ml_r/model_objects_v2.rds")
sr     <- readRDS("data/ml_r/statistical_test_results.rds")
rfi    <- readRDS("data/ml_r/rf_importance_v2.rds")

X_test   <- splits$X_test
X_test_s <- splits$X_test_s
y_test   <- splits$y_test
CLASSES  <- levels(y_test)
N_TEST   <- length(y_test)

cat(sprintf("  Test set: %d rows | Classes: %s\n\n", N_TEST,
            paste(CLASSES, collapse=", ")))

# =============================================================================
# FIX 1 — Actual LDA Confusion Matrix
# =============================================================================
cat("--- FIX 1: LDA Confusion Matrix ---\n")

lda_model <- mo[["lda"]]
cat(sprintf("  LDA model class: %s\n", paste(class(lda_model), collapse="/")))

lda_pred <- predict(lda_model, data.frame(X_test_s))
pred_class_lda <- factor(as.character(lda_pred$class), levels=CLASSES)

lda_cm_actual <- table(Predicted=pred_class_lda, Actual=y_test)
cat("  Actual LDA Confusion Matrix (Predicted rows x Actual cols):\n")
print(lda_cm_actual)
cat(sprintf("  Total = %d (expected 301)\n", sum(lda_cm_actual)))

acc_lda <- mean(pred_class_lda == y_test)
cat(sprintf("  Accuracy = %.4f (expected 0.5748)\n\n", acc_lda))

# Verify accuracy matches saved result
if (abs(acc_lda - 0.5748) > 0.001) {
  cat("  WARNING: Accuracy mismatch — check test split or model object\n\n")
} else {
  cat("  PASS: Accuracy matches saved result\n\n")
}

# Save the actual CM to CSV for reference
cm_df <- as.data.frame(lda_cm_actual)
write_csv(cm_df, file.path(tab_dir, "lda_cm_actual.csv"))
cat("  Saved: DA3/tables/lda_cm_actual.csv\n\n")

# Save the raw matrix values for dashboard patching
# Matrix cell order (row=Predicted, col=Actual):
# [closed,closed] [fully_open,closed] [partially_open,closed]
# [closed,fully_open] [fully_open,fully_open] [partially_open,fully_open]
# [closed,p_open] [fully_open,p_open] [partially_open,p_open]
lda_matrix_vals <- as.integer(lda_cm_actual)
cat("  LDA CM values (matrix fill order, col-major):\n")
cat(" ", paste(lda_matrix_vals, collapse=", "), "\n")
cat("  Dimensions:", paste(dim(lda_cm_actual), collapse=" x "), "\n")
cat("  Row/col names:", paste(rownames(lda_cm_actual), collapse=", "), "/",
    paste(colnames(lda_cm_actual), collapse=", "), "\n\n")

# =============================================================================
# FIX 2 — Actual ROC Curves from RF and LDA posterior probabilities
# =============================================================================
cat("--- FIX 2: Actual ROC Curves ---\n")

# LDA posterior probabilities
lda_posterior <- lda_pred$posterior   # n_test x 3 matrix
cat(sprintf("  LDA posterior dim: %s\n", paste(dim(lda_posterior), collapse=" x ")))
cat(sprintf("  LDA posterior colnames: %s\n", paste(colnames(lda_posterior), collapse=", ")))

# RF class probabilities
rf_model <- mo[["random_forest"]]
rf_probs <- predict(rf_model, X_test, type="prob")
cat(sprintf("  RF prob dim: %s\n", paste(dim(rf_probs), collapse=" x ")))
cat(sprintf("  RF prob colnames: %s\n", paste(colnames(rf_probs), collapse=", ")))

# Manual AUC calculation (one-vs-rest, no external package needed)
calc_auc <- function(scores, binary_labels) {
  # Mann-Whitney U statistic = AUC
  n_pos <- sum(binary_labels == 1)
  n_neg <- sum(binary_labels == 0)
  if (n_pos == 0 || n_neg == 0) return(NA_real_)
  # Sort by score descending
  ord  <- order(scores, decreasing=TRUE)
  lab  <- binary_labels[ord]
  # Compute TPR/FPR at each threshold
  tpr <- c(0, cumsum(lab) / n_pos)
  fpr <- c(0, cumsum(1 - lab) / n_neg)
  # Trapezoidal AUC
  sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
}

# Build ROC data frames for ggplot
build_roc_df <- function(probs, y_true, model_name) {
  cls <- colnames(probs)
  roc_rows <- list()
  for (cl in cls) {
    binary <- as.integer(y_true == cl)
    scores <- probs[, cl]
    auc    <- calc_auc(scores, binary)
    n_pts  <- 200
    thresholds <- seq(1, 0, length.out=n_pts)
    tpr <- fpr <- numeric(n_pts)
    n_pos <- sum(binary)
    n_neg <- length(binary) - n_pos
    for (i in seq_along(thresholds)) {
      pred_pos <- as.integer(scores >= thresholds[i])
      tpr[i] <- if (n_pos > 0) sum(pred_pos & binary) / n_pos else 0
      fpr[i] <- if (n_neg > 0) sum(pred_pos & !binary) / n_neg else 0
    }
    roc_rows[[cl]] <- data.frame(
      FPR   = fpr,
      TPR   = tpr,
      Class = cl,
      Model = model_name,
      AUC   = round(auc, 4),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, roc_rows)
}

# Calculate actual AUCs
cat("\n  Calculating actual AUCs...\n")
lda_roc_df <- build_roc_df(lda_posterior, y_test, "LDA")
rf_roc_df  <- build_roc_df(rf_probs,      y_test, "Random Forest")

# Print actual AUCs vs saved values
cat("\n  LDA AUCs (actual from posterior probabilities):\n")
for (cl in CLASSES) {
  auc_val <- unique(lda_roc_df$AUC[lda_roc_df$Class == cl])
  cat(sprintf("    %-20s AUC = %.4f\n", cl, auc_val))
}

cat("\n  RF AUCs (actual from predict probabilities):\n")
rf_aucs <- numeric(length(CLASSES))
names(rf_aucs) <- CLASSES
for (cl in CLASSES) {
  auc_val <- unique(rf_roc_df$AUC[rf_roc_df$Class == cl])
  rf_aucs[cl] <- auc_val
  cat(sprintf("    %-20s AUC = %.4f (saved: closed=0.8256, fully_open=0.7701, partial=0.6759)\n",
              cl, auc_val))
}
cat(sprintf("\n  RF Macro AUC (actual) = %.4f (saved: 0.7572)\n",
            mean(rf_aucs, na.rm=TRUE)))

# Save AUC results to CSV
auc_df <- data.frame(
  model = c(rep("LDA", 3), rep("Random Forest", 3)),
  class = rep(CLASSES, 2),
  actual_auc = c(
    sapply(CLASSES, function(cl) unique(lda_roc_df$AUC[lda_roc_df$Class==cl])),
    rf_aucs
  )
)
write_csv(auc_df, file.path(tab_dir, "actual_auc_values.csv"))
cat("\n  Saved: DA3/tables/actual_auc_values.csv\n\n")

# Save ROC data for dashboard use
roc_data_all <- rbind(lda_roc_df, rf_roc_df)
write_csv(roc_data_all, file.path(tab_dir, "roc_curve_data.csv"))
cat("  Saved: DA3/tables/roc_curve_data.csv\n\n")

# Plot: RF ROC curves (actual)
rf_roc_df$Label <- sprintf("%s (AUC=%.4f)", rf_roc_df$Class, rf_roc_df$AUC)
rf_colors <- c("closed"="#e74c3c", "fully_open"="#2980b9", "partially_open"="#27ae60")

p_roc_rf <- ggplot(rf_roc_df, aes(x=FPR, y=TPR, colour=Class)) +
  geom_line(linewidth=1.2) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey60", linewidth=0.8) +
  scale_colour_manual(
    values = rf_colors,
    labels = with(rf_roc_df[!duplicated(rf_roc_df$Class),],
                  sprintf("%s (AUC=%.4f)", Class, AUC)),
    name   = "Class"
  ) +
  scale_x_continuous(limits=c(0,1), labels=percent_format(accuracy=1)) +
  scale_y_continuous(limits=c(0,1), labels=percent_format(accuracy=1)) +
  annotate("text", x=0.65, y=0.08,
           label=sprintf("Macro AUC = %.4f", mean(rf_aucs, na.rm=TRUE)),
           size=3.5, colour="grey30") +
  labs(
    title    = "ROC Curves — Random Forest (One-vs-Rest, Test Set n=301)",
    subtitle = "ResearchPilot DA3 | Curves from actual test-set predictions",
    x        = "False Positive Rate",
    y        = "True Positive Rate"
  ) +
  theme_bw(base_size=12) +
  theme(
    plot.title       = element_text(face="bold", size=13),
    plot.subtitle    = element_text(colour="grey50", size=10),
    panel.grid.minor = element_blank(),
    legend.position  = c(0.72, 0.25),
    legend.background= element_rect(fill="white", colour="grey80")
  )

ggsave(file.path(fig_dir, "actual_roc_rf.png"), p_roc_rf,
       width=8, height=6, dpi=150)
cat("  Saved: DA3/figures/actual_roc_rf.png\n")

# Plot: LDA ROC curves (actual)
lda_roc_df$Label <- sprintf("%s (AUC=%.4f)", lda_roc_df$Class, lda_roc_df$AUC)
lda_aucs <- sapply(CLASSES, function(cl) unique(lda_roc_df$AUC[lda_roc_df$Class==cl]))

p_roc_lda <- ggplot(lda_roc_df, aes(x=FPR, y=TPR, colour=Class)) +
  geom_line(linewidth=1.2) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey60", linewidth=0.8) +
  scale_colour_manual(
    values = rf_colors,
    labels = with(lda_roc_df[!duplicated(lda_roc_df$Class),],
                  sprintf("%s (AUC=%.4f)", Class, AUC)),
    name   = "Class"
  ) +
  scale_x_continuous(limits=c(0,1), labels=percent_format(accuracy=1)) +
  scale_y_continuous(limits=c(0,1), labels=percent_format(accuracy=1)) +
  annotate("text", x=0.65, y=0.08,
           label=sprintf("Macro AUC = %.4f", mean(lda_aucs, na.rm=TRUE)),
           size=3.5, colour="grey30") +
  labs(
    title    = "ROC Curves — LDA (One-vs-Rest, Test Set n=301)",
    subtitle = "ResearchPilot DA3 | Curves from actual posterior probabilities",
    x        = "False Positive Rate",
    y        = "True Positive Rate"
  ) +
  theme_bw(base_size=12) +
  theme(
    plot.title       = element_text(face="bold", size=13),
    plot.subtitle    = element_text(colour="grey50", size=10),
    panel.grid.minor = element_blank(),
    legend.position  = c(0.72, 0.25),
    legend.background= element_rect(fill="white", colour="grey80")
  )

ggsave(file.path(fig_dir, "actual_roc_lda.png"), p_roc_lda,
       width=8, height=6, dpi=150)
cat("  Saved: DA3/figures/actual_roc_lda.png\n\n")

# =============================================================================
# FIX 3 — Critical Difference Diagram from saved Nemenyi results
# =============================================================================
cat("--- FIX 3: Critical Difference Diagram ---\n")

# Use the saved posthoc_df and cv_matrix from statistical_test_results.rds
posthoc_df <- sr$posthoc_df
cv_df      <- sr$cv_df        # model, Fold1..Fold5, mean_f1, sd_f1
cv_matrix  <- sr$cv_matrix    # 12 x 5 matrix

cat(sprintf("  posthoc_df: %d rows, %d significant pairs\n",
            nrow(posthoc_df), sum(posthoc_df$significant, na.rm=TRUE)))

# Compute average ranks per model across folds
# Rank 1 = best (highest F1) within each fold
K       <- ncol(cv_matrix)
n_models<- nrow(cv_matrix)

# Rank models within each fold (1 = best)
rank_matrix <- apply(cv_matrix, 2, function(col) rank(-col, ties.method="average"))
avg_ranks   <- rowMeans(rank_matrix)
cat("  Average ranks:\n")
for (i in order(avg_ranks)) {
  cat(sprintf("    %-25s  avg_rank=%.3f  mean_f1=%.4f\n",
              names(avg_ranks)[i], avg_ranks[i], cv_df$mean_f1[cv_df$model == names(avg_ranks)[i]]))
}

# Critical Difference (Nemenyi formula)
# CD = q_alpha * sqrt(k(k+1) / (6*n))
# q_alpha at alpha=0.05, k=12 models: q_0.05 ≈ 3.268 (from Demsar 2006 table)
# For k=12, the critical q value from the Studentized range distribution / sqrt(2):
q_alpha <- 3.268   # Nemenyi CD critical value for k=12, alpha=0.05
CD <- q_alpha * sqrt(n_models * (n_models + 1) / (6 * K))
cat(sprintf("\n  Critical Difference (CD) = %.4f\n", CD))
cat(sprintf("  (q_alpha=%.3f, k=%d, n=%d)\n\n", q_alpha, n_models, K))

# Build CD diagram using ggplot2
# Show models ordered by average rank with CD bar
model_ranks_df <- data.frame(
  model      = names(avg_ranks),
  avg_rank   = avg_ranks,
  mean_f1    = cv_df$mean_f1[match(names(avg_ranks), cv_df$model)],
  stringsAsFactors = FALSE
) %>%
  arrange(avg_rank) %>%
  mutate(
    label = dplyr::recode(model,
      logistic_regression = "Logistic Reg.",
      elastic_net         = "Elastic Net",
      decision_tree       = "Decision Tree",
      random_forest       = "Random Forest",
      xgboost             = "XGBoost",
      gradient_boosting   = "Grad. Boost.",
      adaboost            = "AdaBoost",
      svm_rbf             = "SVM-RBF",
      svm_linear          = "SVM-Linear",
      naive_bayes         = "Naive Bayes",
      lda                 = "LDA",
      mlp                 = "MLP"
    ),
    label = factor(label, levels=rev(label))
  )

# Mark models that are significantly worse than the best (rank 1)
best_model   <- model_ranks_df$model[1]
best_rank    <- model_ranks_df$avg_rank[1]
model_ranks_df$sig_worse_than_best <-
  (model_ranks_df$avg_rank - best_rank) > CD

cat("  Models significantly worse than best model:\n")
for (i in which(model_ranks_df$sig_worse_than_best)) {
  cat(sprintf("    %s (rank=%.2f, diff=%.2f > CD=%.2f)\n",
              model_ranks_df$model[i],
              model_ranks_df$avg_rank[i],
              model_ranks_df$avg_rank[i] - best_rank,
              CD))
}

p_cd <- ggplot(model_ranks_df, aes(x=avg_rank, y=label)) +
  # CD bar at top showing width = CD
  annotate("segment",
           x    = model_ranks_df$avg_rank[1],
           xend = model_ranks_df$avg_rank[1] + CD,
           y    = nrow(model_ranks_df) + 0.6,
           yend = nrow(model_ranks_df) + 0.6,
           colour="black", linewidth=1.5) +
  annotate("text",
           x     = model_ranks_df$avg_rank[1] + CD/2,
           y     = nrow(model_ranks_df) + 1.0,
           label = sprintf("CD = %.2f", CD),
           size  = 3.5, colour="black") +
  # Model points
  geom_point(aes(colour=sig_worse_than_best), size=4) +
  geom_segment(aes(x=avg_rank, xend=avg_rank,
                   y=as.numeric(label)-0.35, yend=as.numeric(label)+0.35,
                   colour=sig_worse_than_best), linewidth=0.7) +
  # Mean F1 annotation
  geom_text(aes(label=sprintf("%.3f", mean_f1)),
            hjust=-0.25, size=2.8, colour="grey40") +
  scale_colour_manual(
    values = c("FALSE"="#2ecc71", "TRUE"="#e74c3c"),
    labels = c("FALSE"="Not significantly worse", "TRUE"="Significantly worse"),
    name   = "vs. best model:"
  ) +
  scale_x_continuous(
    limits = c(0.5, n_models + 1),
    breaks = 1:n_models,
    name   = "Average Rank (1 = best)"
  ) +
  labs(
    title    = "Critical Difference Diagram — Nemenyi Post-Hoc Test",
    subtitle = sprintf("ResearchPilot DA3 | Friedman χ²(11)=%.2f, p<0.001 | CD=%.2f (α=0.05, k=12, n=5)",
                       sr$friedman$statistic, CD),
    y        = NULL
  ) +
  theme_bw(base_size=12) +
  theme(
    plot.title       = element_text(face="bold", size=12),
    plot.subtitle    = element_text(colour="grey50", size=9),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom",
    legend.text      = element_text(size=9)
  )

ggsave(file.path(fig_dir, "stat_cd_diagram.png"), p_cd,
       width=10, height=7, dpi=150)
cat("\n  Saved: DA3/figures/stat_cd_diagram.png\n\n")

# =============================================================================
# FIX 4 — Verified RF Feature Importance from saved RDS
# =============================================================================
cat("--- FIX 4: RF Feature Importance from rds ---\n")

cat(sprintf("  rfi class: %s | dim: %s\n",
            paste(class(rfi), collapse="/"),
            paste(dim(rfi), collapse=" x ")))
cat(sprintf("  Columns: %s\n", paste(colnames(rfi), collapse=", ")))

# MeanDecreaseGini is the standard RF importance metric
# Sort by MeanDecreaseGini descending, take top 20
rfi_df <- data.frame(
  Feature    = rownames(rfi),
  MeanDecreaseGini = rfi[, "MeanDecreaseGini"],
  stringsAsFactors = FALSE
) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(20)

cat("\n  Top 20 features by MeanDecreaseGini (from actual rfi RDS):\n")
for (i in 1:nrow(rfi_df)) {
  cat(sprintf("    %2d. %-35s  %.4f\n", i, rfi_df$Feature[i], rfi_df$MeanDecreaseGini[i]))
}

# Save to CSV for dashboard loading
write_csv(rfi_df, file.path(tab_dir, "rf_feature_importance.csv"))
cat(sprintf("\n  Saved: DA3/tables/rf_feature_importance.csv (%d features)\n\n", nrow(rfi_df)))

# Also note what values were previously hardcoded in dashboard vs actual
cat("  Hardcoded dashboard vs actual (top 5):\n")
hardcoded_fi <- data.frame(
  Feature    = c("pub_ieee","pub_mdpi","title_to_abstract_ratio","abstract_length","abstract_word_count"),
  Hardcoded  = c(0.0312, 0.0289, 0.0271, 0.0248, 0.0241)
)
for (i in 1:5) {
  feat <- hardcoded_fi$Feature[i]
  actual_val <- rfi_df$MeanDecreaseGini[rfi_df$Feature == feat]
  if (length(actual_val) == 0) actual_val <- NA
  cat(sprintf("    %-35s  hardcoded=%.4f  actual=%.4f\n",
              feat, hardcoded_fi$Hardcoded[i],
              ifelse(is.na(actual_val), NA, actual_val)))
}

cat("\n============================================================\n")
cat("  generate_fixes.R COMPLETE\n")
cat("  Files created:\n")
cat("    DA3/tables/lda_cm_actual.csv\n")
cat("    DA3/tables/actual_auc_values.csv\n")
cat("    DA3/tables/roc_curve_data.csv\n")
cat("    DA3/tables/rf_feature_importance.csv\n")
cat("    DA3/figures/actual_roc_rf.png\n")
cat("    DA3/figures/actual_roc_lda.png\n")
cat("    DA3/figures/stat_cd_diagram.png\n")
cat("============================================================\n")
