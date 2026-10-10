setwd("e:/Tuned_Research")
result <- tryCatch({
  p <- parse("dashboard/app.R")
  cat(sprintf("PARSE OK: %d expressions\n", length(p)))
}, error = function(e) {
  cat(sprintf("PARSE ERROR: %s\n", e$message))
})
