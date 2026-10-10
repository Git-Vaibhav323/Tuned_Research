# =============================================================================
# DA3 Fix 2 — Generate rf_feature_importance.csv from saved RDS
#              then verify all dashboard data files are correct
# =============================================================================
setwd("e:/Tuned_Research")
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))

cat("--- Fix 2a: RF Feature Importance CSV ---\n")

rfi <- readRDS("data/ml_r/rf_importance_v2.rds")
cat("  Loaded rf_importance_v2.rds:", paste(dim(rfi), collapse=" x "), "matrix\n")
cat("  Columns:", paste(colnames(rfi), collapse=", "), "\n")

# MeanDecreaseGini is the standard importance metric — confirmed present
rfi_df <- data.frame(
  Feature          = rownames(rfi),
  MeanDecreaseGini = as.numeric(rfi[, "MeanDecreaseGini"]),
  stringsAsFactors = FALSE
) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(20)

cat("\n  Top 20 features by MeanDecreaseGini (actual from rfi RDS):\n")
for (i in 1:nrow(rfi_df)) {
  cat(sprintf("    %2d. %-35s %.4f\n", i, rfi_df$Feature[i], rfi_df$MeanDecreaseGini[i]))
}

write_csv(rfi_df, "DA3/tables/rf_feature_importance.csv")
cat("\n  Saved: DA3/tables/rf_feature_importance.csv\n")
cat(sprintf("  File size: %d bytes\n\n", file.info("DA3/tables/rf_feature_importance.csv")$size))

# ── Verify all dashboard data files exist ───────────────────────────────────
cat("--- Fix 2b: Verify all dashboard data sources ---\n")
files_check <- list(
  "data/ml_r/model_metrics_v2.csv"      = "baseline model metrics",
  "data/ml_r/all_results_v2.csv"        = "all model results",
  "data/ml_r/cluster_keywords.csv"      = "cluster keywords",
  "data/ml_r/cluster_assignments.csv"   = "cluster assignments",
  "data/final/final_dataset.csv"        = "full dataset (OA distribution)",
  "DA3/tables/lda_cm_actual.csv"        = "actual LDA confusion matrix",
  "DA3/tables/roc_curve_data.csv"       = "actual ROC curve data",
  "DA3/tables/rf_feature_importance.csv"= "actual RF importance"
)
all_ok <- TRUE
for (f in names(files_check)) {
  ok <- file.exists(f)
  cat(sprintf("  %-45s [%s]\n", files_check[[f]], ifelse(ok, "OK", "MISSING")))
  if (!ok) all_ok <- FALSE
}
cat(sprintf("\n  All data files present: %s\n\n", ifelse(all_ok, "YES", "NO — CHECK ABOVE")))

# ── Print actual LDA CM for verification ────────────────────────────────────
cat("--- Fix 2c: Verify LDA CM CSV ---\n")
cm_df <- read_csv("DA3/tables/lda_cm_actual.csv", show_col_types=FALSE)
cat("  lda_cm_actual.csv contents:\n")
print(cm_df)
cat(sprintf("  Total predictions: %d (expected 301)\n", sum(cm_df$Freq)))
cat("\n")

# ── Print actual AUC values for verification ────────────────────────────────
cat("--- Fix 2d: Verify actual AUC values ---\n")
auc_df <- read_csv("DA3/tables/actual_auc_values.csv", show_col_types=FALSE)
cat("  actual_auc_values.csv contents:\n")
print(auc_df)

cat("\n--- Fix 2 COMPLETE ---\n")
