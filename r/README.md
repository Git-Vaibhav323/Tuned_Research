# R Analysis and Visualization

Phase 1 exploratory analysis for ResearchPilot using `data/final/final_dataset.csv`.

## Contents

| File | Purpose |
|------|---------|
| `scripts/00_install_packages.R` | Install/load required R packages (run once) |
| `scripts/01_eda_final_dataset.R` | Step-by-step EDA script (recommended starting point) |
| `scripts/eda_final_dataset.Rmd` | Same EDA as an R Markdown report |
| `visualizations/` | Exported PNG figures from the EDA script |

## Required packages

- `tidyverse` (dplyr, ggplot2, readr, tidyr, stringr, …)
- `skimr`
- `janitor`
- `scales`
- `GGally`

## How to run (RStudio)

1. Open the ResearchPilot project folder.
2. Run `r/scripts/00_install_packages.R` once.
3. Open `r/scripts/01_eda_final_dataset.R`.
4. Execute **Step 0 → Step 15** in order.
5. Review plots in `r/visualizations/` and `reports/figures/`.
6. Optional: knit `eda_final_dataset.Rmd` to HTML.

## Notes

- The script auto-detects the project root by looking for `data/final/final_dataset.csv`.
- Summary tables are written to `reports/tables/`.
- Fill the interpretation checklist at the end of the `.R` script in your Phase 1 write-up.
