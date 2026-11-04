# ResearchPilot — R Shiny Dashboard

Interactive analytics dashboard for exploring all ML pipeline results.

---

## Launch

```r
setwd("E:/Tuned_Research")
shiny::runApp("dashboard")
```

Required R packages: `shiny`, `plotly`, `dplyr`, `tidyr`, `readr`, `DT`, `scales`

---

## Tabs

### Tab 1 — Overview
- KPI cards: 2,000 papers · 27→206→80 features · 12 models · 301 test papers
- Publication year distribution (2022–2025)
- Open-access category breakdown (fully open / partially open / closed)

### Tab 2 — Model Comparison
- Interactive bar chart of all 15 models (12 baseline + 3 tuned)
- Metric selector: Accuracy, Macro-F1, ROC-AUC
- Full results table with sortable columns

### Tab 3 — Model Diagnostics
- Model selector: LDA or Random Forest
- Confusion matrix heatmap (column-normalised, actual counts in cells)
- ROC curves per class (one-vs-rest) with AUC values
- Random Forest feature importance — top 15 by MeanDecreaseGini

### Tab 4 — Research Clusters
- K-Means topic clustering (k=8, silhouette=0.0645)
- Cluster size pie chart
- Dominant TF-IDF terms per cluster
- Interpretation note: clusters are exploratory due to low silhouette score

---

## Key numbers shown

| Metric | Value |
|--------|-------|
| Best model | SVM-RBF (tuned) — Macro-F1 = 0.5701 |
| Best baseline | LDA — Macro-F1 = 0.5696, Accuracy = 57.48% |
| Friedman test | χ²(11) = 43.46, p = 9×10⁻⁶ |
| Top feature | pub_ieee (Gini = 43.78) |
| Cluster silhouette | 0.0645 (exploratory) |

---

## Data sources

All charts are generated from saved CSV outputs in `reports/tables/` and `data/ml_r/`.
No models are re-trained when the dashboard loads.
