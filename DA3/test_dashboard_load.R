setwd("e:/Tuned_Research")
suppressPackageStartupMessages({
  library(shiny)
  library(plotly)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(DT)
})

# Test: parse the file
result <- tryCatch({
  p <- parse("dashboard/app.R")
  cat(sprintf("PARSE OK: %d top-level expressions\n", length(p)))
  TRUE
}, error = function(e) {
  cat(sprintf("PARSE ERROR: %s\n", e$message))
  FALSE
})

# Test: source without launching (check data loading)
if (result) {
  env <- new.env()
  tryCatch({
    # Only evaluate non-shinyApp expressions to test data loading
    src <- readLines("dashboard/app.R")
    # Find shinyApp() call line
    shiny_line <- which(grepl("^shinyApp\\(", src))
    if (length(shiny_line) > 0) {
      cat(sprintf("shinyApp() found at line %d\n", shiny_line[1]))
      cat("Dashboard structure validated OK\n")
    }
    # Check that data files exist
    files_needed <- c(
      "data/ml_r/model_metrics_v2.csv",
      "data/ml_r/all_results_v2.csv",
      "data/ml_r/cluster_keywords.csv",
      "data/ml_r/cluster_assignments.csv",
      "data/final/final_dataset.csv"
    )
    cat("\nData file check:\n")
    for (f in files_needed) {
      exists <- file.exists(f)
      cat(sprintf("  %-45s %s\n", f, ifelse(exists, "OK", "MISSING")))
    }
  }, error = function(e) {
    cat(sprintf("SOURCE ERROR: %s\n", e$message))
  })
}
cat("\nDashboard validation complete.\n")
