# Faculty Presentation Guide — Phase 1

Use this order when showing your teacher. Total demo time: ~10–15 minutes.

## Rubric → what to open

| Marks | Rubric item | What to show |
|------:|-------------|--------------|
| 1 | Problem statement & objectives | Knit/open `04_faculty_phase1_report.Rmd` → Section 1 |
| 2 | Dataset identification & justification | Same report → Section 2 + `data/raw/raw_papers.csv` |
| 1 | Literature review (≥5 papers) | Same report → Section 3 |
| 2 | Dataset understanding & features | Run `02_dataset_understanding.R` |
| 2 | Data preprocessing | Run `03_preprocessing_cleaning_demo.R` |
| 1 | EDA (≥5 R visuals) | Already done: `r/visualizations/` (19 plots) |
| 1 | Documentation | `docs/summary.md` + `reports/*.md` |

## Exact run commands (RStudio console)

```r
setwd("E:/Tuned_Research")

# 1) Feature dictionary / understanding
source("r/scripts/02_dataset_understanding.R")

# 2) Preprocessing + cleaning demo (missing values, encoding, normalization)
source("r/scripts/03_preprocessing_cleaning_demo.R")

# 3) Optional: re-run EDA if teacher asks
# source("r/scripts/01_eda_final_dataset.R")

# 4) Knit full faculty report to HTML
rmarkdown::render("r/scripts/04_faculty_phase1_report.Rmd")
```

## What to say while demoing preprocessing

1. “We collected 2000 AI/ML papers from OpenAlex with API filters.”  
2. “In R we re-check missing values before cleaning.”  
3. “We clean title/abstract whitespace and drop empty abstracts.”  
4. “We convert year and citations to integers.”  
5. “We keep English only and remove duplicate DOIs.”  
6. “For the rubric, we also show encoding and min-max normalization examples.”  
7. “Final analysis uses engineered features in `final_dataset.csv`.”  
8. “EDA produced 19 R visualizations; here are the key ones.”

## Files created for this presentation pack

- `r/scripts/02_dataset_understanding.R`
- `r/scripts/03_preprocessing_cleaning_demo.R`
- `r/scripts/04_faculty_phase1_report.Rmd`
- `r/FACULTY_PRESENTATION_GUIDE.md` (this file)
