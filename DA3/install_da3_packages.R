pkgs_needed <- c("shiny", "plotly", "PMCMRplus")
for (p in pkgs_needed) {
  if (!p %in% rownames(installed.packages())) {
    cat(sprintf("Installing %s ...\n", p))
    install.packages(p, repos = "https://cloud.r-project.org", quiet = FALSE)
  } else {
    cat(sprintf("  %s already installed\n", p))
  }
}
cat("\nVerification:\n")
for (p in pkgs_needed) {
  ok <- p %in% rownames(installed.packages())
  cat(sprintf("  %-15s %s\n", p, ifelse(ok, "OK", "FAILED")))
}
