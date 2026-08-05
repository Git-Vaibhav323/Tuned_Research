# ResearchPilot R Analysis

## Scripts

| Script | Purpose |
|--------|---------|
| `02_data_understanding.R` | Dataset overview, feature dictionary, schema, sample rows |
| `03_data_preprocessing.R` | Missing values, cleaning, encoding, normalization |
| `04_eda_visualizations.R` | EDA graphs and summary tables |

## How to run

```r
setwd("E:/Tuned_Research")

source("r/scripts/02_data_understanding.R")
source("r/scripts/03_data_preprocessing.R")
source("r/scripts/04_eda_visualizations.R")
```

## Outputs

- Figures: `r/visualizations/`
- Tables: `reports/tables/`
- Preprocessing report: `reports/preprocessing_report_r.md`
