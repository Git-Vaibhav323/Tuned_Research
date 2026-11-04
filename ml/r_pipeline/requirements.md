# ResearchPilot DA2 — R Package Requirements

**R Version**: 4.6.1  
**Platform**: Windows  
**Last Updated**: 2026-09-19

---

## Installation

Run this once before executing any Phase 2 scripts:

```r
setwd("E:/Tuned_Research")
source("r/install_da2_packages.R")
```

Or install individual packages:

```r
install.packages("PACKAGE_NAME", repos = "https://cloud.r-project.org")
```

---

## Core Data Packages (base R — pre-installed)

| Package | Purpose | Notes |
|---------|---------|-------|
| `rpart` | CART Decision Tree | base R |
| `MASS` | LDA, utility functions | base R |
| `nnet` | Multinomial Logistic Regression, MLP | base R |
| `cluster` | Silhouette scoring | base R |

---

## Data Wrangling

| Package | Purpose | Install |
|---------|---------|---------|
| `dplyr` | Data manipulation | `install.packages("dplyr")` |
| `tidyr` | Reshaping data | `install.packages("tidyr")` |
| `readr` | Fast CSV read/write | `install.packages("readr")` |
| `stringr` | String operations | `install.packages("stringr")` |
| `forcats` | Factor handling | `install.packages("forcats")` |
| `purrr` | Functional programming | `install.packages("purrr")` |

---

## Database

| Package | Purpose | Install |
|---------|---------|---------|
| `DBI` | Database Interface (generic) | `install.packages("DBI")` |
| `RSQLite` | SQLite driver for DBI | `install.packages("RSQLite")` |

**Usage example:**
```r
library(DBI); library(RSQLite)
con <- dbConnect(RSQLite::SQLite(), "database/researchpilot_r.db")
result <- dbGetQuery(con, "SELECT * FROM papers LIMIT 5")
dbDisconnect(con)
```

---

## Machine Learning

| Package | Algorithm | Install |
|---------|-----------|---------|
| `randomForest` | Random Forest (Breiman) | `install.packages("randomForest")` |
| `ranger` | Fast Random Forest | `install.packages("ranger")` |
| `e1071` | SVM + Naive Bayes | `install.packages("e1071")` |
| `xgboost` | XGBoost | `install.packages("xgboost")` |
| `glmnet` | Elastic Net / Lasso / Ridge | `install.packages("glmnet")` |
| `gbm` | Gradient Boosting Machine | `install.packages("gbm")` |
| `adabag` | AdaBoost | `install.packages("adabag")` |
| `kknn` | K-Nearest Neighbours | `install.packages("kknn")` |
| `naivebayes` | Naive Bayes | `install.packages("naivebayes")` |

---

## Text Mining / Clustering

| Package | Purpose | Install |
|---------|---------|---------|
| `tidytext` | TF-IDF, tokenisation | `install.packages("tidytext")` |
| `SnowballC` | Word stemming | `install.packages("SnowballC")` |
| `irlba` | Truncated SVD / fast PCA | `install.packages("irlba")` |

---

## Visualisation

| Package | Purpose | Install |
|---------|---------|---------|
| `ggplot2` | Grammar of graphics | `install.packages("ggplot2")` |
| `patchwork` | Combine ggplot panels | `install.packages("patchwork")` |
| `scales` | Axis/colour scale helpers | `install.packages("scales")` |
| `viridis` | Perceptually uniform colour palettes | `install.packages("viridis")` |
| `ggrepel` | Non-overlapping text labels | `install.packages("ggrepel")` |
| `factoextra` | Cluster visualisation | `install.packages("factoextra")` |

---

## Utilities

| Package | Purpose | Install |
|---------|---------|---------|
| `tidymodels` | ML meta-package (parsnip, recipes, tune…) | `install.packages("tidymodels")` |
| `caret` | Feature selection utilities | `install.packages("caret")` |
| `janitor` | `clean_names`, `tabyl` | `install.packages("janitor")` |
| `skimr` | Summary statistics | `install.packages("skimr")` |
| `knitr` | Markdown table rendering | `install.packages("knitr")` |

---

## Script → Package Mapping

| Script | Key Packages Required |
|--------|----------------------|
| `01_feature_engineering.R` | dplyr, readr, stringr |
| `02_feature_selection.R` | dplyr, readr, stringr, ggplot2, tidyr, randomForest |
| `03_database_setup.R` | DBI, RSQLite, dplyr, readr |
| `04_database_queries.R` | DBI, RSQLite, dplyr, readr, ggplot2, tidyr |
| `05_ml_models.R` | dplyr, readr, nnet, rpart, randomForest, gbm, xgboost, adabag, e1071, MASS, kknn |
| `06_hyperparameter_tuning.R` | dplyr, readr, randomForest, xgboost, e1071 |
| `07_model_evaluation.R` | dplyr, readr, DBI, RSQLite |
| `08_comparative_visualizations.R` | dplyr, ggplot2, tidyr, stringr, scales, e1071, MASS, randomForest, xgboost, gbm, nnet |
| `09_impact_tier.R` | dplyr, readr, ggplot2, nnet, rpart, randomForest, gbm, xgboost, adabag, e1071, MASS |
| `10_clustering.R` | dplyr, readr, ggplot2, tidyr, tidytext, irlba, cluster, viridis, SnowballC |

---

## Quick Environment Check

```r
setwd("E:/Tuned_Research")
source("r/check_packages.R")
```

---

## Reproducibility Notes

- Set `set.seed(42)` at the start of every script
- All train/test splits use stratified sampling with `set.seed(42)`
- Impact-tier thresholds computed on training data only (saved in `data/ml_r/impact_tier_thresholds.csv`)
- No `renv` used — package versions not locked. For full reproducibility, note:
  - R version: 4.6.1
  - randomForest: 4.7+
  - xgboost: 1.7+
  - glmnet: 4.1+
  - tidytext: 0.4+
