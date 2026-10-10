setwd("e:/Tuned_Research")

# ── model_objects_v2.rds ────────────────────────────────────────────────────
cat("--- model_objects_v2.rds ---\n")
mo <- readRDS("data/ml_r/model_objects_v2.rds")
cat("Class:", class(mo), "\n")
cat("Names:", paste(names(mo), collapse="\n       "), "\n\n")

for (nm in names(mo)) {
  obj <- mo[[nm]]
  cat(sprintf("  %-25s  class=%-30s\n", nm, paste(class(obj), collapse="/")))
}

# ── rf_importance_v2.rds ────────────────────────────────────────────────────
cat("\n--- rf_importance_v2.rds ---\n")
rfi <- readRDS("data/ml_r/rf_importance_v2.rds")
cat("Class:", paste(class(rfi), collapse="/"), "\n")
cat("Type:", typeof(rfi), "\n")
if (is.matrix(rfi) || is.data.frame(rfi)) {
  cat("Dim:", paste(dim(rfi), collapse=" x "), "\n")
  cat("Colnames:", paste(colnames(rfi), collapse=", "), "\n")
  cat("First 20 rows:\n")
  print(round(head(rfi, 20), 6))
} else if (is.numeric(rfi) || is.character(rfi)) {
  cat("Length:", length(rfi), "\n")
  print(head(rfi, 25))
} else {
  cat("str:\n")
  str(rfi)
}

# ── statistical_test_results.rds ────────────────────────────────────────────
cat("\n--- statistical_test_results.rds ---\n")
sr <- readRDS("data/ml_r/statistical_test_results.rds")
cat("Names:", paste(names(sr), collapse=", "), "\n")
cat("reject_h0:", sr$reject_h0, "\n")
cat("Friedman p:", sr$friedman$p.value, "\n")
cat("Friedman chi-sq:", sr$friedman$statistic, "\n")
if (!is.null(sr$posthoc_df)) {
  cat("posthoc_df nrow:", nrow(sr$posthoc_df), "\n")
  cat("posthoc_df significant:", sum(sr$posthoc_df$significant, na.rm=TRUE), "\n")
  cat("posthoc_df head:\n")
  print(head(sr$posthoc_df, 10))
}
