# =============================================================================
# ResearchPilot DA2 — ENHANCED Pipeline Runner
# Runs all enriched feature pipeline scripts in sequence
# =============================================================================
setwd("E:/Tuned_Research")

run_step <- function(label, script) {
  cat(sprintf("\n%s\n%s\n%s\n", strrep("=",65), label, strrep("=",65)))
  t0 <- proc.time()["elapsed"]
  tryCatch(
    source(script, local = FALSE),
    error = function(e) cat(sprintf("\n[ERROR in %s]: %s\n", script, e$message))
  )
  cat(sprintf("\n[Done in %.1f s]\n", proc.time()["elapsed"] - t0))
}

run_step("M2v2: Feature Selection (Enriched)",
         "r/scripts/phase2/02_feature_selection_v2.R")

run_step("M4v2: ML Models (Enriched — 13 algorithms)",
         "r/scripts/phase2/05_ml_models_v2.R")

run_step("M5v2: Hyperparameter Tuning (RF + XGBoost + SVM)",
         "r/scripts/phase2/06_hyperparameter_tuning_v2.R")

run_step("M6av2: Model Evaluation & Leaderboard",
         "r/scripts/phase2/07_model_evaluation_v2.R")

run_step("M6bv2: Comparative Visualizations",
         "r/scripts/phase2/08_visualizations_v2.R")

cat(sprintf("\n%s\n", strrep("=",65)))
cat("ENHANCED PIPELINE COMPLETE\n")
cat(sprintf("%s\n\n", strrep("=",65)))

# Print final summary
lb_path <- "reports/tables/r_model_leaderboard_v2.csv"
if (file.exists(lb_path)) {
  lb <- read.csv(lb_path)
  cat("FINAL LEADERBOARD (top 5 by accuracy):\n")
  top5 <- head(lb[order(-lb$acc), c("rank","model","model_type","acc","f1","roc")], 5)
  print(top5, row.names=FALSE)
  cat(sprintf("\nBest Accuracy : %.4f (%s)\n", max(lb$acc,na.rm=TRUE),
              lb$model[which.max(lb$acc)]))
  cat(sprintf("Best Macro-F1 : %.4f (%s)\n", max(lb$f1,na.rm=TRUE),
              lb$model[which.max(lb$f1)]))
  cat(sprintf("Improvement   : +%.1f%% accuracy over original pipeline\n",
              (max(lb$acc,na.rm=TRUE)-0.4452)*100))
}
