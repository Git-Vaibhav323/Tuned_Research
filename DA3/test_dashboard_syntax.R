suppressPackageStartupMessages({
  library(shiny)
  library(plotly)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(DT)
})
tryCatch({
  p <- parse("dashboard/app.R")
  cat("SYNTAX OK — dashboard/app.R parsed successfully\n")
  cat(sprintf("Expressions parsed: %d\n", length(p)))
}, error = function(e) {
  cat(sprintf("SYNTAX ERROR: %s\n", e$message))
})
