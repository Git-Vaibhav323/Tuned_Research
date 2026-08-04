# R Analysis and Visualization

Phase 1 scripts for ResearchPilot — including faculty presentation materials.

## Scripts (run in this order for viva)

| # | File | Rubric focus |
|---|------|----------------|
| 0 | `scripts/00_install_packages.R` | Setup (once) |
| 1 | `scripts/01_eda_final_dataset.R` | EDA visuals (**done**) |
| 2 | `scripts/02_dataset_understanding.R` | Feature description |
| 3 | `scripts/03_preprocessing_cleaning_demo.R` | Cleaning / missing / encoding / normalization |
| 4 | `scripts/04_faculty_phase1_report.Rmd` | Full faculty walkthrough (all rubric items) |

Also see: `FACULTY_PRESENTATION_GUIDE.md`

## Quick start for faculty demo

```r
setwd("E:/Tuned_Research")
source("r/scripts/02_dataset_understanding.R")
source("r/scripts/03_preprocessing_cleaning_demo.R")
rmarkdown::render("r/scripts/04_faculty_phase1_report.Rmd")
```

## Outputs

- Figures: `r/visualizations/`, `reports/figures/`
- Tables: `reports/tables/`
- Reports: `reports/preprocessing_report_r.md`, HTML from the Rmd
