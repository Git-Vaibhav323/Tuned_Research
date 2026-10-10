# =============================================================================
# DA3 Fix 1 — Critical Difference Diagram
# Uses saved statistical_test_results.rds and cv_fold_matrix.csv
# The previous crash: scale_y_continuous() was given a factor column.
# Fix: use numeric y-axis with manual label annotations.
# =============================================================================
setwd("e:/Tuned_Research")
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

cat("--- Fix 1: Critical Difference Diagram ---\n")

fig_dir <- "DA3/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Load saved results
sr         <- readRDS("data/ml_r/statistical_test_results.rds")
cv_matrix  <- sr$cv_matrix      # 12 x 5, rows=models, cols=folds
posthoc_df <- sr$posthoc_df
K          <- ncol(cv_matrix)
n_models   <- nrow(cv_matrix)

# Friendly display names
label_map <- c(
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
)

# Average ranks: rank 1 = best (highest F1) within each fold
rank_matrix <- apply(cv_matrix, 2, function(col) rank(-col, ties.method = "average"))
avg_ranks   <- sort(rowMeans(rank_matrix))   # sorted ascending (1 = best first)

# Build data frame — all numeric, no factors
model_df <- data.frame(
  model      = names(avg_ranks),
  avg_rank   = as.numeric(avg_ranks),
  mean_f1    = as.numeric(rowMeans(cv_matrix[names(avg_ranks), , drop = FALSE],
                                   na.rm = TRUE)),
  stringsAsFactors = FALSE
)
model_df$display <- dplyr::recode(model_df$model, !!!label_map)

# y-position: numeric 1..12 (best at top = n_models)
model_df$y_pos <- rev(seq_len(nrow(model_df)))  # best model gets highest y

# Critical Difference (Nemenyi, alpha=0.05)
# q critical value for k=12 from Demsar 2006 table: 3.268
q_alpha <- 3.268
CD       <- q_alpha * sqrt(n_models * (n_models + 1) / (6 * K))
cat(sprintf("  CD = %.4f  (q=%.3f, k=%d, n=%d)\n", CD, q_alpha, n_models, K))

# Which models are significantly worse than the best?
best_rank <- model_df$avg_rank[1]
model_df$sig_worse <- (model_df$avg_rank - best_rank) > CD

cat("  Models significantly worse than best:\n")
sig_w <- model_df[model_df$sig_worse, ]
for (i in seq_len(nrow(sig_w))) {
  cat(sprintf("    %s (rank=%.2f, diff=%.2f > CD=%.2f)\n",
              sig_w$display[i], sig_w$avg_rank[i],
              sig_w$avg_rank[i] - best_rank, CD))
}

# ── Draw the CD diagram using only numeric y-axis ───────────────────────────
# Style: horizontal rank axis, models on y (numeric), CD bar at top right
top_y    <- max(model_df$y_pos) + 1.2   # position for CD bar annotation
tick_h   <- 0.25                          # tick height at rank endpoints

p <- ggplot(model_df) +

  # Horizontal line connecting each model to its rank dot
  geom_segment(aes(x = avg_rank, xend = avg_rank,
                   y = y_pos - 0.28, yend = y_pos + 0.28,
                   colour = sig_worse),
               linewidth = 0.7, show.legend = FALSE) +

  # Rank dot
  geom_point(aes(x = avg_rank, y = y_pos, colour = sig_worse),
             size = 4) +

  # Model name label — left of dot
  geom_text(aes(x = avg_rank - 0.15, y = y_pos,
                label = display, colour = sig_worse),
            hjust = 1, size = 3.5, show.legend = FALSE) +

  # Mean F1 label — right of dot
  geom_text(aes(x = avg_rank + 0.15, y = y_pos,
                label = sprintf("F1=%.3f", mean_f1)),
            hjust = 0, size = 3.0, colour = "grey50",
            show.legend = FALSE) +

  # CD bar: from best rank to best+CD, at top
  annotate("segment",
           x = best_rank, xend = best_rank + CD,
           y = top_y,     yend = top_y,
           colour = "black", linewidth = 1.8) +
  annotate("segment",  # left tick
           x = best_rank, xend = best_rank,
           y = top_y - tick_h, yend = top_y + tick_h,
           colour = "black", linewidth = 1.2) +
  annotate("segment",  # right tick
           x = best_rank + CD, xend = best_rank + CD,
           y = top_y - tick_h, yend = top_y + tick_h,
           colour = "black", linewidth = 1.2) +
  annotate("text",
           x = best_rank + CD / 2, y = top_y + 0.55,
           label = sprintf("CD = %.2f", CD),
           size = 3.5, colour = "black", fontface = "bold") +

  # Friedman subtitle as annotation box
  annotate("label",
           x = (1 + n_models) / 2, y = 0.25,
           label = sprintf(
             "Friedman: χ²(11) = %.2f,  p = %.2e  →  H₀ REJECTED\n7 significant pairs (Nemenyi, α = 0.05)",
             sr$friedman$statistic, sr$friedman$p.value),
           size = 3.0, colour = "grey20", fill = "#FFF9C4",
           label.size = 0.4) +

  scale_colour_manual(
    values = c("FALSE" = "#2196F3", "TRUE" = "#F44336"),
    labels = c("FALSE" = "Not significantly worse than best",
               "TRUE"  = "Significantly worse than best"),
    name = NULL
  ) +
  scale_x_continuous(
    limits = c(-1.5, n_models + 0.8),
    breaks = 1:n_models,
    name   = "Average Rank across 5 folds  (1 = best)"
  ) +
  scale_y_continuous(
    limits = c(-0.3, top_y + 1.0),
    breaks = NULL,
    name   = NULL,
    expand = c(0, 0)
  ) +
  labs(
    title    = "Critical Difference Diagram — Nemenyi Post-Hoc Test",
    subtitle = "ResearchPilot DA3  |  12 baseline models  |  5-fold CV Macro-F1"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid       = element_blank(),
    axis.ticks.y     = element_blank(),
    legend.position  = "bottom",
    legend.text      = element_text(size = 10),
    panel.border     = element_rect(colour = "grey80")
  )

out_path <- file.path(fig_dir, "stat_cd_diagram.png")
ggsave(out_path, p, width = 11, height = 7.5, dpi = 150)
cat(sprintf("  Saved: %s\n", out_path))

# Verify file exists and has reasonable size
sz <- file.info(out_path)$size
cat(sprintf("  File size: %d bytes  %s\n", sz,
            ifelse(sz > 10000, "[OK]", "[WARNING: suspiciously small]")))
cat("--- Fix 1 COMPLETE ---\n")
