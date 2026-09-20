# Master runner — executes all Phase 2 scripts in order
# Run from project root: Rscript r/run_pipeline.R

cat("\n", paste(rep("=",60),collapse=""), "\n")
cat("ResearchPilot DA2 — R Pipeline Runner\n")
cat(paste(rep("=",60),collapse=""), "\n\n")

root <- getwd()
scripts <- c(
  "r/scripts/phase2/05_ml_models.R",
  "r/scripts/phase2/06_hyperparameter_tuning.R",
  "r/scripts/phase2/07_model_evaluation.R",
  "r/scripts/phase2/08_comparative_visualizations.R",
  "r/scripts/phase2/09_impact_tier.R",
  "r/scripts/phase2/10_clustering.R"
)

for (s in scripts) {
  cat(sprintf("\n%s\nRunning: %s\n%s\n", 
              paste(rep("-",60),collapse=""), s,
              paste(rep("-",60),collapse="")))
  t_start <- proc.time()["elapsed"]
  tryCatch(
    source(s, local = FALSE),
    error = function(e) {
      cat(sprintf("\n[ERROR in %s]: %s\n", s, conditionMessage(e)))
    }
  )
  elapsed <- proc.time()["elapsed"] - t_start
  cat(sprintf("\n[Done in %.1f s]\n", elapsed))
}

cat("\n", paste(rep("=",60),collapse=""), "\n")
cat("Pipeline complete.\n")
cat(paste(rep("=",60),collapse=""), "\n\n")
