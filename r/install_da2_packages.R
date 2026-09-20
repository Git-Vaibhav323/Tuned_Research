# Install all DA2-required packages
pkgs_needed <- c(
  "RSQLite",       # SQLite database driver
  "ranger",        # Fast Random Forest
  "e1071",         # SVM + Naive Bayes
  "xgboost",       # XGBoost
  "glmnet",        # Elastic Net / Lasso
  "gbm",           # Gradient Boosting
  "kknn",          # K-Nearest Neighbours
  "naivebayes",    # Naive Bayes
  "tidymodels",    # ML framework (recipes, parsnip, workflows, tune, yardstick)
  "caret",         # Feature selection utilities
  "tidytext",      # TF-IDF, text mining
  "irlba",         # Truncated SVD for sparse matrices
  "patchwork",     # Combine ggplot panels
  "viridis",       # Color scales
  "ggrepel",       # Non-overlapping text labels
  "factoextra",    # Cluster visualizations
  "klaR",          # Naive Bayes (klaR engine)
  "adabag",        # AdaBoost (boosting)
  "kernlab"        # SVM (ksvm engine)
)

ip <- rownames(installed.packages())
to_install <- pkgs_needed[!pkgs_needed %in% ip]

if (length(to_install) == 0) {
  writeLines("All packages already installed!")
} else {
  writeLines(paste("Installing", length(to_install), "packages:"))
  writeLines(paste(" -", to_install))
  install.packages(to_install,
                   repos   = "https://cloud.r-project.org",
                   quiet   = FALSE,
                   Ncpus   = 2)
}

# Verify
writeLines("\n=== Installation verification ===")
for (p in pkgs_needed) {
  ok <- p %in% rownames(installed.packages())
  writeLines(sprintf("  %-20s %s", p, ifelse(ok, "OK", "FAILED")))
}
