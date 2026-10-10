# ResearchPilot DA3 — Interactive Dashboard

## Quick Start

```r
# Option 1: From R console
setwd("e:/Tuned_Research")
shiny::runApp("dashboard")

# Option 2: From inside the dashboard folder
shiny::runApp(".")
```

## Requirements

Install required packages:

```r
install.packages(c("shiny", "plotly", "dplyr", "tidyr", "readr", "DT", "scales"))
```

## Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | KPI cards, OA distribution, feature pipeline summary |
| **Model Comparison** | Metric selector, all 15 models (12 baseline + 3 tuned) |
| **Model Diagnostics** | Confusion matrix, ROC curves, feature importance |
| **Research Clusters** | Cluster sizes, dominant terms, theme descriptions |

## Data Sources

The dashboard reads directly from existing project files:

| File | Used In |
|------|---------|
| `data/ml_r/model_metrics_v2.csv` | Tab 2, Tab 3 |
| `data/ml_r/all_results_v2.csv` | Tab 2 |
| `data/ml_r/cluster_keywords.csv` | Tab 4 |
| `data/ml_r/cluster_assignments.csv` | Tab 4 |
| `data/final/final_dataset.csv` | Tab 1 (OA distribution) |

## No fake data

All values come from the actual DA2 pipeline results. Confusion matrices and
ROC-AUC values are from the proper held-out test set (n=301).

## Troubleshooting

**Error: Cannot find project root**
→ Make sure `setwd("e:/Tuned_Research")` before running `shiny::runApp("dashboard")`

**Error: package 'X' not found**
→ Run `install.packages("X")` and restart R

**Port already in use**
→ `shiny::runApp("dashboard", port = 3839)`
