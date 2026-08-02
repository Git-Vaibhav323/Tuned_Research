# =============================================================================
# ResearchPilot — reliable package installer (Windows-friendly)
# =============================================================================
# Why the last install failed
# ---------------------------
# Many "download of package X failed" warnings = network / CRAN timeout while
# downloading a huge batch at once. That is NOT an Rtools problem.
# Rtools warning can be ignored for binary installs.
#
# What this script does
# ---------------------
# Installs only the packages needed for EDA, ONE AT A TIME, with retries.
# =============================================================================

options(install.packages.check.source = "no")
options(timeout = 600)

repos <- "https://cloud.r-project.org"

# Minimal set (no tidyverse meta-package — fewer downloads, more reliable)
pkgs <- c(
  # core tidyverse pieces used by the EDA script
  "cli", "rlang", "lifecycle", "vctrs", "glue", "withr",
  "pillar", "tibble", "purrr", "magrittr",
  "dplyr", "tidyr", "stringr", "forcats", "readr",
  "ggplot2", "scales",
  # EDA helpers
  "skimr",
  "janitor"
)

install_one <- function(pkg, repos, max_tries = 3) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message("[OK] already installed: ", pkg)
    return(TRUE)
  }

  for (attempt in seq_len(max_tries)) {
    message(sprintf("[TRY %d/%d] installing %s ...", attempt, max_tries, pkg))
    status <- tryCatch(
      {
        install.packages(
          pkg,
          repos = repos,
          dependencies = c("Depends", "Imports", "LinkingTo"),
          quiet = FALSE
        )
        TRUE
      },
      error = function(e) {
        message("  error: ", conditionMessage(e))
        FALSE
      },
      warning = function(w) {
        message("  warning: ", conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )

    if (requireNamespace(pkg, quietly = TRUE)) {
      message("[OK] installed: ", pkg)
      return(TRUE)
    }

    Sys.sleep(2)
  }

  message("[FAIL] could not install: ", pkg)
  FALSE
}

message("Installing ", length(pkgs), " packages one-by-one from ", repos)
message("This can take several minutes. Let it finish.\n")

results <- vapply(pkgs, install_one, logical(1), repos = repos)

message("\n========== SUMMARY ==========")
message("Installed OK : ", sum(results))
message("Failed       : ", sum(!results))
if (any(!results)) {
  message("Failed pkgs  : ", paste(pkgs[!results], collapse = ", "))
  message("Re-run this script to retry only the missing ones.")
} else {
  message("All required packages are ready.")
  message("Next: open and run r/scripts/01_eda_final_dataset.R")
}

# Quick load test
message("\nLoad test:")
for (pkg in c("dplyr", "ggplot2", "readr", "tidyr", "stringr", "scales", "skimr", "janitor")) {
  ok <- require(pkg, character.only = TRUE, quietly = TRUE)
  message(if (ok) "  loaded: " else "  MISSING: ", pkg)
}
