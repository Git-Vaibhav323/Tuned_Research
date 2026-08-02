# =============================================================================
# Quick fix: repair broken tidyverse/dplyr install
# =============================================================================
# Your error means dplyr exists, but dependency 'rlang' was never downloaded.
# Run this WHOLE script once in RStudio (Source).
# =============================================================================

options(install.packages.check.source = "no")
options(timeout = 600)
repos <- "https://cloud.r-project.org"

# Install these first (dependencies dplyr needs)
critical <- c(
  "rlang", "cli", "glue", "lifecycle", "vctrs", "withr",
  "pillar", "tibble", "magrittr", "tidyselect", "R6"
)

# Then the packages used by the EDA script
eda_pkgs <- c(
  "dplyr", "tidyr", "stringr", "forcats", "readr",
  "ggplot2", "scales", "skimr", "janitor"
)

install_one <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    message("[OK] ", pkg)
    return(invisible(TRUE))
  }
  message("[INSTALL] ", pkg)
  try(install.packages(pkg, repos = repos, dependencies = FALSE), silent = TRUE)
  # if still missing, try with Imports
  if (!requireNamespace(pkg, quietly = TRUE)) {
    try(
      install.packages(
        pkg,
        repos = repos,
        dependencies = c("Depends", "Imports", "LinkingTo")
      ),
      silent = TRUE
    )
  }
  if (requireNamespace(pkg, quietly = TRUE)) {
    message("[OK] ", pkg)
  } else {
    message("[FAIL] ", pkg, " — re-run this script")
  }
}

message("Step A: critical dependencies")
invisible(lapply(critical, install_one))

message("\nStep B: EDA packages")
invisible(lapply(eda_pkgs, install_one))

message("\nStep C: load test")
pkgs_to_test <- c("rlang", "dplyr", "ggplot2", "readr", "tidyr", "stringr", "scales", "skimr", "janitor")
for (pkg in pkgs_to_test) {
  ok <- require(pkg, character.only = TRUE, quietly = TRUE)
  message(if (ok) "  loaded: " else "  STILL MISSING: ", pkg)
}

message("\nIf all say 'loaded', open r/scripts/01_eda_final_dataset.R and run it.")
