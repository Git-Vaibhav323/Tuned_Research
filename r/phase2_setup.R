# =============================================================================
# ResearchPilot — Phase 2 DA2 — Package Setup
# =============================================================================
# Run this script ONCE before executing any Phase 2 R scripts.
# It installs all required packages and verifies the environment.
# =============================================================================

cat("=== ResearchPilot Phase 2 — R Environment Setup ===\n\n")

# ── Core packages ─────────────────────────────────────────────────────────────
packages <- c(
  # Data wrangling
  "dplyr",         # data manipulation
  "tidyr",         # reshaping
  "readr",         # fast CSV I/O
  "stringr",       # string operations
  "forcats",       # factor handling
  "purrr",         # functional programming

  # Machine Learning — tidymodels ecosystem
  "tidymodels",    # meta-package: recipes, parsnip, workflows, yardstick, rsample, tune, dials
  "recipes",       # feature preprocessing
  "parsnip",       # unified model interface
  "workflows",     # model + recipe bundling
  "yardstick",     # model evaluation metrics
  "rsample",       # data splitting / resampling
  "tune",          # hyperparameter tuning
  "dials",         # tuning parameter grids

  # Additional ML algorithm engines
  "randomForest",  # Random Forest (randomForest::randomForest)
  "ranger",        # Fast Random Forest (ranger engine for parsnip)
  "e1071",         # SVM (svm), Naive Bayes (naiveBayes)
  "kernlab",       # SVM via ksvm (parsnip svm_rbf engine)
  "xgboost",       # XGBoost
  "gbm",           # Gradient Boosting Machine
  "C50",           # C5.0 decision tree
  "rpart",         # CART decision tree
  "MASS",          # LDA (lda function)
  "klaR",          # Naive Bayes (NaiveBayes)
  "nnet",          # MLP neural network (nnet engine)
  "adabag",        # AdaBoost (boosting function)
  "glmnet",        # Elastic Net / Lasso / Ridge Logistic Regression
  "kknn",          # K-Nearest Neighbours (kknn engine)
  "naivebayes",    # Naive Bayes (naive_bayes engine for parsnip)

  # Feature selection
  "caret",         # Near-zero variance, correlation filtering, RFE
  "Boruta",        # Boruta feature importance
  "infotheo",      # Mutual information / entropy

  # Database
  "DBI",           # Database Interface
  "RSQLite",       # SQLite driver

  # Text / NLP
  "tidytext",      # Tidy text mining
  "SnowballC",     # Stemmer for tidytext
  "tm",            # Text Mining (for TF-IDF)
  "quanteda",      # Document-feature matrix

  # Clustering
  "cluster",       # silhouette, pam, clara
  "factoextra",    # cluster visualization (fviz_cluster)

  # Dimensionality reduction
  "irlba",         # Truncated SVD / PCA on sparse matrices

  # Visualization
  "ggplot2",       # Grammar of graphics
  "ggrepel",       # Non-overlapping text labels
  "patchwork",     # Combine ggplot panels
  "scales",        # Axis formatting
  "viridis",       # Color scales
  "RColorBrewer",  # Color palettes
  "pheatmap",      # Heatmaps
  "GGally",        # ggpairs, correlation matrix plots
  "cowplot",       # plot_grid for multi-panel figures

  # Reporting / output
  "knitr",         # Markdown table formatting
  "kableExtra",    # Enhanced kable tables
  "glue",          # String interpolation

  # Utilities
  "tictoc",        # Timing
  "progress",      # Progress bars
  "janitor",       # clean_names, tabyl
  "skimr"          # Summary statistics
)

# ── Install missing packages ──────────────────────────────────────────────────
installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]

if (length(to_install) > 0) {
  cat(sprintf("Installing %d missing packages:\n", length(to_install)))
  cat(paste(" -", to_install, collapse = "\n"), "\n\n")
  install.packages(to_install, repos = "https://cloud.r-project.org", quiet = FALSE)
} else {
  cat("All required packages are already installed.\n\n")
}

# ── Verify critical packages load ────────────────────────────────────────────
critical <- c("tidymodels", "DBI", "RSQLite", "ggplot2", "dplyr",
              "randomForest", "xgboost", "glmnet", "e1071", "caret",
              "tidytext", "cluster")

cat("Verifying critical packages load correctly...\n")
failures <- character(0)
for (pkg in critical) {
  tryCatch({
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
    cat(sprintf("  [OK] %s\n", pkg))
  }, error = function(e) {
    cat(sprintf("  [FAIL] %s — %s\n", pkg, conditionMessage(e)))
    failures <<- c(failures, pkg)
  })
}

cat("\n")
if (length(failures) == 0) {
  cat("=== Setup complete. All critical packages loaded successfully. ===\n")
  cat("You can now run r/scripts/phase2/01_feature_engineering.R\n")
} else {
  cat(sprintf("WARNING: %d package(s) failed to load: %s\n",
              length(failures), paste(failures, collapse = ", ")))
  cat("Please install them manually and re-run this setup script.\n")
}

cat(sprintf("\nR version: %s\n", R.version.string))
cat(sprintf("Working directory: %s\n", getwd()))
