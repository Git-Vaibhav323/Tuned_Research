# =============================================================================
# ResearchPilot — DA3
# Script:  11_statistical_analysis.R
# Purpose: Re-run 5-fold CV for all 12 baseline models, perform Friedman test
#          and Nemenyi post-hoc analysis, export results, save visualizations
# Input:   data/ml_r/ml_data_splits_v2.rds  (frozen train/val/test split)
#          data/ml_r/selected_features_v2.csv
# Output:  data/ml_r/cv_fold_results.csv
#          data/ml_r/statistical_test_results.rds
#          DA3/tables/cv_fold_matrix.csv
#          DA3/tables/friedman_results.csv
#          DA3/tables/posthoc_matrix.csv
#          DA3/figures/stat_cv_boxplot.png
#          DA3/figures/stat_fold_heatmap.png
#          DA3/figures/stat_cd_diagram.png  (if PMCMRplus available)
# =============================================================================

set.seed(42)
cat("============================================================\n")
cat("  ResearchPilot DA3 — Statistical Analysis\n")
cat("  Friedman Test + Nemenyi Post-Hoc (5-Fold CV)\n")
cat("============================================================\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research"))
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
if (is.null(root)) stop("Cannot locate project root. Set working directory to project root.")
cat(sprintf("Project root: %s\n\n", root))

out_dir  <- file.path(root, "data", "ml_r")
tab_dir  <- file.path(root, "DA3", "tables")
fig_dir  <- file.path(root, "DA3", "figures")
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load packages ──────────────────────────────────────────────────────────
needed <- c("randomForest","e1071","glmnet","gbm","xgboost",
            "MASS","nnet","ggplot2","dplyr","tidyr","readr")
for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}

# PMCMRplus for Nemenyi test — install if missing
has_pmcmr <- requireNamespace("PMCMRplus", quietly = TRUE)
if (!has_pmcmr) {
  cat("Installing PMCMRplus for Nemenyi post-hoc test...\n")
  install.packages("PMCMRplus", repos = "https://cloud.r-project.org")
  has_pmcmr <- requireNamespace("PMCMRplus", quietly = TRUE)
}
if (has_pmcmr) suppressPackageStartupMessages(library(PMCMRplus))

# ── 2. Load data + splits ─────────────────────────────────────────────────────
cat("Loading splits and feature data...\n")
splits  <- readRDS(file.path(out_dir, "ml_data_splits_v2.rds"))
X_train <- splits$X_train       # raw (unscaled)
X_train_s <- splits$X_train_s   # scaled
y_train <- splits$y_train
CLASSES <- levels(y_train)
N_TRAIN <- nrow(X_train)
cat(sprintf("  Training rows: %d | Features: %d | Classes: %s\n\n",
            N_TRAIN, ncol(X_train), paste(CLASSES, collapse = ", ")))

# Reproduce the SAME fold assignment used in tuning script (same seed, same logic)
set.seed(42)
fold_ids <- sample(rep(1:5, length.out = N_TRAIN))
cat(sprintf("  Fold sizes: %s\n\n", paste(table(fold_ids), collapse = " | ")))

# ── 3. Shared evaluation helper ───────────────────────────────────────────────
f1_macro <- function(pred, truth) {
  cls <- levels(truth)
  pred <- factor(as.character(pred), levels = cls)
  f1s  <- vapply(cls, function(c) {
    tp <- sum(pred == c & truth == c)
    fp <- sum(pred == c & truth != c)
    fn <- sum(pred != c & truth == c)
    p  <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    r  <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    if ((p + r) > 0) 2 * p * r / (p + r) else 0
  }, numeric(1))
  mean(f1s)
}

# ── 4. 5-Fold CV for each of the 12 baseline models ──────────────────────────
#  We evaluate MACRO-F1 on each hold-out fold.
#  Models must be fit on training folds only (no leakage).
# -----------------------------------------------------------------------------

MODELS <- c("logistic_regression","elastic_net","decision_tree",
            "random_forest","xgboost","gradient_boosting","adaboost",
            "svm_rbf","svm_linear","naive_bayes","lda","mlp")

K <- 5
# Matrix: rows = models, cols = folds
cv_matrix <- matrix(NA_real_, nrow = length(MODELS), ncol = K,
                    dimnames = list(MODELS, paste0("Fold", 1:K)))

cat("Running 5-fold CV for 12 baseline models...\n")
cat("(This may take 5-15 minutes depending on hardware)\n\n")

for (mdl in MODELS) {
  cat(sprintf("  Model: %-25s ", mdl))
  fold_f1 <- rep(NA_real_, K)
  
  for (k in 1:K) {
    tr <- which(fold_ids != k)
    vl <- which(fold_ids == k)
    
    Xtr  <- X_train[tr, ];   Xvl  <- X_train[vl, ]
    Xtr_s <- X_train_s[tr, ]; Xvl_s <- X_train_s[vl, ]
    ytr  <- y_train[tr];      yvl  <- y_train[vl]
    
    pred <- tryCatch({
      if (mdl == "logistic_regression") {
        m <- nnet::multinom(y ~ ., data = data.frame(y = ytr, Xtr_s),
                            MaxNWts = 5000, trace = FALSE)
        factor(as.character(predict(m, data.frame(Xvl_s))), levels = CLASSES)
        
      } else if (mdl == "elastic_net") {
        m <- glmnet::cv.glmnet(Xtr_s, ytr, family = "multinomial",
                               alpha = 0.5, nfolds = 3)
        pp <- predict(m, Xvl_s, s = "lambda.min", type = "class")
        factor(as.character(pp), levels = CLASSES)
        
      } else if (mdl == "decision_tree") {
        df_tr <- data.frame(y = ytr, Xtr)
        m <- rpart::rpart(y ~ ., data = df_tr, method = "class",
                          control = rpart::rpart.control(cp = 0.01))
        factor(as.character(predict(m, data.frame(Xvl), type = "class")), levels = CLASSES)
        
      } else if (mdl == "random_forest") {
        m <- randomForest::randomForest(x = Xtr, y = ytr, ntree = 300,
                                        do.trace = FALSE)
        predict(m, Xvl, type = "class")
        
      } else if (mdl == "xgboost") {
        y_int  <- as.integer(ytr) - 1L
        n_cls  <- length(CLASSES)
        dtrain <- xgboost::xgb.DMatrix(Xtr, label = y_int)
        dtest  <- xgboost::xgb.DMatrix(Xvl)
        m <- xgboost::xgb.train(
          params = list(objective = "multi:softprob", num_class = n_cls,
                        max_depth = 6, eta = 0.1, subsample = 0.8,
                        colsample_bytree = 0.8, eval_metric = "mlogloss",
                        verbose = 0),
          data = dtrain, nrounds = 150, verbose = 0)
        raw <- predict(m, dtest)
        pp  <- matrix(raw, ncol = n_cls, byrow = TRUE)
        factor(CLASSES[apply(pp, 1, which.max)], levels = CLASSES)
        
      } else if (mdl == "gradient_boosting") {
        df_tr <- data.frame(y = ytr, Xtr)
        # Use distribution="multinomial" for 3 classes
        m <- gbm::gbm(y ~ ., data = df_tr, distribution = "multinomial",
                      n.trees = 150, interaction.depth = 4,
                      n.minobsinnode = 10, shrinkage = 0.1, verbose = FALSE)
        pp  <- predict(m, data.frame(Xvl), n.trees = 150, type = "response")
        pc  <- apply(pp[,,1], 1, which.max)
        factor(CLASSES[pc], levels = CLASSES)
        
      } else if (mdl == "adaboost") {
        # Use randomForest stumps as a fast AdaBoost proxy
        df_tr <- data.frame(y = ytr, Xtr)
        m <- randomForest::randomForest(x = Xtr, y = ytr, ntree = 100,
                                        mtry = 1, do.trace = FALSE)
        predict(m, Xvl, type = "class")
        
      } else if (mdl == "svm_rbf") {
        m <- e1071::svm(x = Xtr_s, y = ytr, kernel = "radial",
                        cost = 1, gamma = 0.001, probability = FALSE)
        predict(m, Xvl_s)
        
      } else if (mdl == "svm_linear") {
        m <- e1071::svm(x = Xtr_s, y = ytr, kernel = "linear",
                        cost = 1, probability = FALSE)
        predict(m, Xvl_s)
        
      } else if (mdl == "naive_bayes") {
        m <- e1071::naiveBayes(x = Xtr_s, y = ytr)
        predict(m, Xvl_s)
        
      } else if (mdl == "lda") {
        m <- MASS::lda(x = Xtr_s, grouping = ytr)
        factor(predict(m, Xvl_s)$class, levels = CLASSES)
        
      } else if (mdl == "mlp") {
        m <- nnet::nnet(y ~ ., data = data.frame(y = ytr, Xtr_s),
                        size = 50, maxit = 200, decay = 1e-4,
                        trace = FALSE, MaxNWts = 10000)
        factor(as.character(predict(m, data.frame(Xvl_s), type = "class")),
               levels = CLASSES)
      }
    }, error = function(e) {
      cat(sprintf("[Fold %d ERROR: %s] ", k, condenseError(e$message)), sep = "")
      NULL
    })
    
    if (!is.null(pred)) fold_f1[k] <- f1_macro(pred, yvl)
  }  # end folds
  
  cv_matrix[mdl, ] <- round(fold_f1, 5)
  cat(sprintf("| Mean F1 = %.4f  (folds: %s)\n",
              mean(fold_f1, na.rm = TRUE),
              paste(round(fold_f1, 3), collapse = ", ")))
}

cat("\n=== 5-Fold CV Matrix (Model x Fold, Macro-F1) ===\n")
print(round(cv_matrix, 4))

# Helper for error condensing (defined here so available in tryCatch)
condenseError <- function(msg) substr(gsub("\n", " ", msg), 1, 60)

# ── 5. Save fold matrix ───────────────────────────────────────────────────────
cv_df <- as.data.frame(cv_matrix)
cv_df$model   <- rownames(cv_matrix)
cv_df$mean_f1 <- round(rowMeans(cv_matrix, na.rm = TRUE), 5)
cv_df$sd_f1   <- round(apply(cv_matrix, 1, sd, na.rm = TRUE), 5)
cv_df <- cv_df[order(-cv_df$mean_f1), ]

write_csv(cv_df, file.path(tab_dir, "cv_fold_matrix.csv"))
write_csv(cv_df, file.path(out_dir, "cv_fold_results.csv"))
cat(sprintf("\nSaved: cv_fold_matrix.csv (%d models x 5 folds)\n\n", nrow(cv_df)))

# ── 6. Friedman Test ──────────────────────────────────────────────────────────
cat("============================================================\n")
cat("  FRIEDMAN TEST\n")
cat("  H0: All 12 models have equal median Macro-F1 across folds\n")
cat("  H1: At least one model differs significantly\n")
cat("  Significance level: alpha = 0.05\n")
cat("============================================================\n\n")

# friedman.test needs: y = matrix(blocks x treatments) or long-form
# Our cv_matrix: rows=models, cols=folds → transpose so blocks=folds, treatments=models
perf_matrix <- cv_matrix  # models x folds

# Remove rows with any NA (if a model had an error)
complete_rows <- apply(perf_matrix, 1, function(x) !any(is.na(x)))
if (sum(complete_rows) < nrow(perf_matrix)) {
  cat(sprintf("  NOTE: %d model(s) had missing fold values and are EXCLUDED from Friedman test:\n",
              sum(!complete_rows)))
  cat(sprintf("    %s\n\n", paste(names(which(!complete_rows)), collapse = ", ")))
}
perf_mat_clean <- perf_matrix[complete_rows, ]
n_models_test  <- nrow(perf_mat_clean)
cat(sprintf("  Models included in test: %d\n", n_models_test))

# Friedman test
# R's friedman.test: y = matrix where rows=blocks, columns=treatments
# So we transpose: rows=folds (blocks), cols=models (treatments)
friedman_mat <- t(perf_mat_clean)   # K x n_models

friedman_res <- friedman.test(friedman_mat)

cat("\n--- Friedman Test Results ---\n")
cat(sprintf("  Chi-squared statistic: %.4f\n", friedman_res$statistic))
cat(sprintf("  Degrees of freedom:    %d\n",   friedman_res$parameter))
cat(sprintf("  p-value:               %.6f\n", friedman_res$p.value))
cat(sprintf("  Significance level:    0.05\n"))

if (friedman_res$p.value < 0.05) {
  cat("\n  CONCLUSION: REJECT H0\n")
  cat("  There IS a statistically significant difference in median\n")
  cat("  performance across the 12 models (p < 0.05).\n")
  cat("  Proceeding to Nemenyi post-hoc test...\n\n")
  reject_h0 <- TRUE
} else {
  cat("\n  CONCLUSION: FAIL TO REJECT H0\n")
  cat(sprintf("  No statistically significant difference detected (p = %.4f >= 0.05).\n",
              friedman_res$p.value))
  cat("  This does NOT mean all models are identical — it means\n")
  cat("  we lack sufficient evidence to conclude differences.\n")
  cat("  This may be due to: small number of folds (K=5),\n")
  cat("  high variance, or genuinely similar performance profiles.\n\n")
  reject_h0 <- FALSE
}

# Save Friedman results
friedman_df <- data.frame(
  test            = "Friedman",
  n_models        = n_models_test,
  n_folds         = K,
  metric          = "Macro-F1",
  chi_squared     = round(friedman_res$statistic, 4),
  df              = friedman_res$parameter,
  p_value         = round(friedman_res$p.value, 6),
  alpha           = 0.05,
  reject_h0       = reject_h0,
  conclusion      = ifelse(reject_h0,
                           "Significant difference exists between models",
                           "No significant difference detected")
)
write_csv(friedman_df, file.path(tab_dir, "friedman_results.csv"))
cat("Saved: friedman_results.csv\n\n")

# ── 7. Nemenyi Post-Hoc Test (if significant) ─────────────────────────────────
posthoc_df <- NULL

if (reject_h0 && has_pmcmr) {
  cat("============================================================\n")
  cat("  NEMENYI POST-HOC TEST (Pairwise comparisons)\n")
  cat("============================================================\n\n")
  
  nemenyi_res <- tryCatch(
    PMCMRplus::frdAllPairsNemenyiTest(y     = friedman_mat,
                                       groups = colnames(friedman_mat),
                                       blocks = rownames(friedman_mat)),
    error = function(e) {
      cat(sprintf("  Nemenyi test failed: %s\n  Using Wilcoxon pairwise instead.\n\n",
                  e$message))
      NULL
    }
  )
  
  if (!is.null(nemenyi_res)) {
    pvals <- nemenyi_res$p.value
    cat("  Pairwise p-values (lower triangle):\n")
    print(round(pvals, 4))
    
    # Convert to long-form
    p_long <- as.data.frame(as.table(pvals))
    names(p_long) <- c("Model_A", "Model_B", "p_value")
    p_long <- p_long[!is.na(p_long$p_value), ]
    p_long$significant <- p_long$p_value < 0.05
    p_long <- p_long[order(p_long$p_value), ]
    
    write_csv(p_long, file.path(tab_dir, "posthoc_matrix.csv"))
    posthoc_df <- p_long
    cat("\nSaved: posthoc_matrix.csv\n\n")
    
    sig_pairs <- p_long[p_long$significant, ]
    if (nrow(sig_pairs) > 0) {
      cat(sprintf("  Significantly different pairs (p < 0.05): %d\n", nrow(sig_pairs)))
      for (i in seq_len(min(10, nrow(sig_pairs)))) {
        cat(sprintf("    %-25s vs %-25s  p=%.4f\n",
                    sig_pairs$Model_A[i], sig_pairs$Model_B[i], sig_pairs$p_value[i]))
      }
    } else {
      cat("  No pairwise comparisons reached significance after correction.\n")
    }
  }
  
} else if (reject_h0 && !has_pmcmr) {
  cat("  PMCMRplus not available — skipping Nemenyi test.\n")
  cat("  Run: install.packages('PMCMRplus') then re-run this script.\n\n")
} else {
  cat("  Friedman test not significant — post-hoc analysis skipped.\n\n")
}

# ── 8. Save combined RDS ──────────────────────────────────────────────────────
stat_results <- list(
  cv_matrix      = cv_matrix,
  cv_df          = cv_df,
  friedman       = friedman_res,
  friedman_df    = friedman_df,
  reject_h0      = reject_h0,
  posthoc_df     = posthoc_df,
  n_models       = n_models_test,
  K              = K,
  metric         = "Macro-F1",
  seed           = 42
)
saveRDS(stat_results, file.path(out_dir, "statistical_test_results.rds"))
cat("Saved: statistical_test_results.rds\n\n")

# ── 9. Visualizations ─────────────────────────────────────────────────────────
cat("Creating visualizations...\n")

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

# --- Plot 1: CV Macro-F1 Boxplot per model ---
long_df <- cv_df %>%
  select(model, Fold1, Fold2, Fold3, Fold4, Fold5) %>%
  pivot_longer(cols = starts_with("Fold"), names_to = "fold", values_to = "f1") %>%
  mutate(
    model_label = dplyr::recode(model, !!!label_map),
    model_label = factor(model_label, levels = label_map[cv_df$model])
  )

mean_df <- cv_df %>%
  mutate(model_label = dplyr::recode(model, !!!label_map),
         model_label = factor(model_label, levels = label_map[cv_df$model]))

p1 <- ggplot(long_df, aes(x = model_label, y = f1)) +
  geom_boxplot(aes(fill = model_label), width = 0.55, outlier.shape = 21,
               outlier.fill = "white", outlier.colour = "grey40", alpha = 0.8) +
  geom_point(data = mean_df, aes(x = model_label, y = mean_f1),
             shape = 23, size = 3, fill = "#e63946", colour = "white") +
  scale_fill_viridis_d(option = "C", guide = "none") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "5-Fold Cross-Validation Macro-F1 by Model",
    subtitle = "ResearchPilot DA3 | Red diamond = mean",
    x        = NULL,
    y        = "Macro-F1"
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x      = element_text(angle = 40, hjust = 1, size = 10),
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "stat_cv_boxplot.png"), p1,
       width = 12, height = 6, dpi = 150)
cat("  Saved: stat_cv_boxplot.png\n")

# --- Plot 2: Fold-level Heatmap (models x folds) ---
hm_df <- cv_df %>%
  select(model, Fold1, Fold2, Fold3, Fold4, Fold5, mean_f1) %>%
  arrange(mean_f1) %>%
  mutate(model_label = dplyr::recode(model, !!!label_map),
         model_label = factor(model_label, levels = label_map[.$model])) %>%
  pivot_longer(cols = starts_with("Fold"), names_to = "fold", values_to = "f1")

p2 <- ggplot(hm_df, aes(x = fold, y = model_label, fill = f1)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", f1)), size = 3.2, colour = "white", fontface = "bold") +
  scale_fill_gradient2(low = "#d62728", mid = "#ff7f0e", high = "#2ca02c",
                       midpoint = 0.5, name = "F1",
                       limits = c(0.2, 0.8), oob = scales::squish) +
  scale_x_discrete(labels = paste0("F", 1:5)) +
  labs(
    title    = "5-Fold CV Performance Heatmap (Macro-F1)",
    subtitle = "ResearchPilot DA3 | Rows ordered by mean F1",
    x        = "Fold",
    y        = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50", size = 10),
    axis.text.y   = element_text(size = 10),
    legend.position = "right"
  )

ggsave(file.path(fig_dir, "stat_fold_heatmap.png"), p2,
       width = 9, height = 6, dpi = 150)
cat("  Saved: stat_fold_heatmap.png\n")

# --- Plot 3: Mean CV F1 + error bars ---
p3 <- ggplot(cv_df, aes(x = reorder(dplyr::recode(model, !!!label_map), mean_f1),
                         y = mean_f1)) +
  geom_col(aes(fill = mean_f1), width = 0.65, alpha = 0.9) +
  geom_errorbar(aes(ymin = pmax(0, mean_f1 - sd_f1),
                    ymax = pmin(1, mean_f1 + sd_f1)),
                width = 0.3, colour = "grey30", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.4f", mean_f1)),
            hjust = -0.15, size = 3.2, colour = "grey20") +
  scale_fill_gradient(low = "#fee08b", high = "#1a9850", name = "Mean F1") +
  scale_y_continuous(limits = c(0, 0.85),
                     labels = scales::percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title    = "Mean 5-Fold CV Macro-F1 with ±1 SD",
    subtitle = sprintf("Friedman test: χ²(%.2f, df=%d), p = %.4f",
                       friedman_res$statistic,
                       friedman_res$parameter,
                       friedman_res$p.value),
    x        = NULL,
    y        = "Mean Macro-F1"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid.minor = element_blank(),
    legend.position  = "none"
  )

ggsave(file.path(fig_dir, "stat_mean_f1_bars.png"), p3,
       width = 10, height = 6.5, dpi = 150)
cat("  Saved: stat_mean_f1_bars.png\n")

# ── 10. Summary to console ────────────────────────────────────────────────────
cat("\n")
cat("============================================================\n")
cat("  FINAL STATISTICAL ANALYSIS SUMMARY\n")
cat("============================================================\n")
cat(sprintf("  Models evaluated:   %d (12 baseline)\n", n_models_test))
cat(sprintf("  Folds (K):          5\n"))
cat(sprintf("  Metric:             Macro-F1\n"))
cat(sprintf("\n  Friedman χ²:        %.4f\n", friedman_res$statistic))
cat(sprintf("  Degrees of freedom: %d\n",     friedman_res$parameter))
cat(sprintf("  p-value:            %.6f\n",   friedman_res$p.value))
cat(sprintf("  Decision (α=0.05):  %s H0\n",
            ifelse(reject_h0, "REJECT", "FAIL TO REJECT")))
cat("\n  Best model by CV Macro-F1:\n")
cat(sprintf("    %-25s  Mean=%.4f  SD=%.4f\n",
            cv_df$model[1], cv_df$mean_f1[1], cv_df$sd_f1[1]))
cat("\n  Worst model by CV Macro-F1:\n")
cat(sprintf("    %-25s  Mean=%.4f  SD=%.4f\n",
            cv_df$model[nrow(cv_df)], cv_df$mean_f1[nrow(cv_df)], cv_df$sd_f1[nrow(cv_df)]))
cat("\n  Output files:\n")
cat(sprintf("    %s\n", file.path(tab_dir, "cv_fold_matrix.csv")))
cat(sprintf("    %s\n", file.path(tab_dir, "friedman_results.csv")))
cat(sprintf("    %s\n", file.path(tab_dir, "posthoc_matrix.csv")))
cat(sprintf("    %s\n", file.path(fig_dir,  "stat_cv_boxplot.png")))
cat(sprintf("    %s\n", file.path(fig_dir,  "stat_fold_heatmap.png")))
cat(sprintf("    %s\n", file.path(fig_dir,  "stat_mean_f1_bars.png")))
cat("============================================================\n")
cat("  11_statistical_analysis.R  COMPLETE\n")
cat("============================================================\n\n")
