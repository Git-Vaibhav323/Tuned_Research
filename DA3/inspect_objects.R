setwd("e:/Tuned_Research")
cat("============================================================\n")
cat("  ResearchPilot DA3 — Saved Object Inspection\n")
cat("============================================================\n\n")

# ── 1. ml_data_splits_v2.rds ────────────────────────────────────────────────
cat("--- ml_data_splits_v2.rds ---\n")
splits <- readRDS("data/ml_r/ml_data_splits_v2.rds")
cat("Names:", paste(names(splits), collapse=", "), "\n")
cat("X_train dim:", paste(dim(splits$X_train), collapse=" x "), "\n")
cat("X_val   dim:", paste(dim(splits$X_val),   collapse=" x "), "\n")
cat("X_test  dim:", paste(dim(splits$X_test),  collapse=" x "), "\n")
cat("y_train table:\n"); print(table(splits$y_train))
cat("y_val   table:\n"); print(table(splits$y_val))
cat("y_test  table:\n"); print(table(splits$y_test))
cat("scale_params present:", "scale_params" %in% names(splits), "\n\n")

# ── 2. model_objects_v2.rds ─────────────────────────────────────────────────
cat("--- model_objects_v2.rds ---\n")
mo <- readRDS("data/ml_r/model_objects_v2.rds")
cat("Class of object:", class(mo), "\n")
if (is.list(mo)) {
  cat("Names:", paste(names(mo), collapse=", "), "\n")
  for (nm in names(mo)) {
    obj <- mo[[nm]]
    cat(sprintf("  %-25s class=%s\n", nm, paste(class(obj), collapse="/")))
  }
}
cat("\n")

# ── 3. Check for LDA model specifically ─────────────────────────────────────
cat("--- LDA model check ---\n")
lda_candidates <- names(mo)[grepl("lda|LDA", names(mo))]
cat("LDA candidate names:", paste(lda_candidates, collapse=", "), "\n")
if (length(lda_candidates) > 0) {
  lda_obj <- mo[[lda_candidates[1]]]
  cat("LDA class:", paste(class(lda_obj), collapse="/"), "\n")
  cat("LDA names:", paste(names(lda_obj), collapse=", "), "\n")
}
cat("\n")

# ── 4. rf_importance_v2.rds ─────────────────────────────────────────────────
cat("--- rf_importance_v2.rds ---\n")
rfi <- readRDS("data/ml_r/rf_importance_v2.rds")
cat("Class:", class(rfi), "\n")
if (is.data.frame(rfi)) {
  cat("Dimensions:", nrow(rfi), "x", ncol(rfi), "\n")
  cat("Columns:", paste(names(rfi), collapse=", "), "\n")
  cat("First 20 rows:\n")
  print(head(rfi, 20))
} else if (is.matrix(rfi)) {
  cat("Dimensions:", nrow(rfi), "x", ncol(rfi), "\n")
  cat("Column names:", paste(colnames(rfi), collapse=", "), "\n")
  cat("First 20 rows:\n")
  print(head(rfi, 20))
} else {
  cat("Type:", typeof(rfi), "\n")
  cat("Length:", length(rfi), "\n")
  print(head(rfi, 20))
}
cat("\n")

# ── 5. statistical_test_results.rds ─────────────────────────────────────────
cat("--- statistical_test_results.rds ---\n")
str_res <- readRDS("data/ml_r/statistical_test_results.rds")
cat("Names:", paste(names(str_res), collapse=", "), "\n")
cat("Friedman chi-sq:", str_res$friedman$statistic, "\n")
cat("Friedman p-val:",  str_res$friedman$p.value, "\n")
cat("reject_h0:", str_res$reject_h0, "\n")
cat("posthoc_df class:", class(str_res$posthoc_df), "\n")
if (!is.null(str_res$posthoc_df)) {
  cat("posthoc_df dims:", nrow(str_res$posthoc_df), "x", ncol(str_res$posthoc_df), "\n")
  cat("significant pairs:", sum(str_res$posthoc_df$significant, na.rm=TRUE), "\n")
}
cat("\n")

# ── 6. Check for any saved prediction probability objects ────────────────────
cat("--- Checking for saved prediction/probability objects ---\n")
all_ml_files <- list.files("data/ml_r", pattern="\\.rds$|\\.csv$", full.names=FALSE)
cat("All files in data/ml_r:\n")
for (f in sort(all_ml_files)) cat(sprintf("  %s\n", f))
cat("\n")

# ── 7. Check model_objects_v2.rds for prediction capability ─────────────────
cat("--- Test: LDA predict on test set ---\n")
if (length(lda_candidates) > 0) {
  library(MASS)
  lda_obj <- mo[[lda_candidates[1]]]
  X_test_s <- splits$X_test_s
  y_test   <- splits$y_test
  
  pred <- tryCatch({
    predict(lda_obj, X_test_s)
  }, error = function(e) {
    cat("  predict() error:", e$message, "\n")
    NULL
  })
  
  if (!is.null(pred)) {
    cat("  predict() succeeded\n")
    cat("  pred names:", paste(names(pred), collapse=", "), "\n")
    pred_class <- pred$class
    cat("  Predicted class table:\n"); print(table(pred_class))
    cat("  Actual class table:\n");    print(table(y_test))
    cm <- table(Predicted=pred_class, Actual=y_test)
    cat("  Confusion matrix (Predicted rows x Actual cols):\n")
    print(cm)
    cat("  Total predicted:", sum(cm), "\n")
    acc <- mean(pred_class == y_test)
    cat("  Accuracy:", round(acc, 4), "(expected 0.5748)\n")
    
    # Class probabilities for ROC
    if ("posterior" %in% names(pred)) {
      cat("  Posterior probabilities available: YES\n")
      cat("  Posterior dim:", paste(dim(pred$posterior), collapse=" x "), "\n")
      cat("  Posterior colnames:", paste(colnames(pred$posterior), collapse=", "), "\n")
    } else {
      cat("  Posterior probabilities: NOT in pred object\n")
    }
  }
}
cat("\n")

# ── 8. Check RF model for prediction capability ──────────────────────────────
cat("--- Test: RF predict on test set (for ROC) ---\n")
rf_candidates <- names(mo)[grepl("random_forest|rf$|^rf", names(mo))]
cat("RF candidate names:", paste(rf_candidates, collapse=", "), "\n")
if (length(rf_candidates) > 0) {
  library(randomForest)
  rf_obj <- mo[[rf_candidates[1]]]
  X_test  <- splits$X_test
  y_test  <- splits$y_test
  
  pred_rf <- tryCatch({
    predict(rf_obj, X_test, type="prob")
  }, error = function(e) {
    cat("  RF prob predict() error:", e$message, "\n")
    NULL
  })
  
  if (!is.null(pred_rf)) {
    cat("  RF prob predict() succeeded\n")
    cat("  Prob matrix dim:", paste(dim(pred_rf), collapse=" x "), "\n")
    cat("  Prob colnames:", paste(colnames(pred_rf), collapse=", "), "\n")
    cat("  First 3 rows:\n"); print(head(pred_rf, 3))
  }
}
cat("\n")

cat("============================================================\n")
cat("  Inspection complete\n")
cat("============================================================\n")
