# DA3 Final Verification Checklist
## ResearchPilot | Submission: October 16, 2026

Run this checklist before submission to verify all DA3 criteria are satisfied.

---

## CRITERION 1: Performance Comparison + Statistical Analysis (2 Marks)

### Performance Comparison
- [ ] `DA3/tables/final_model_comparison.csv` exists (15 rows: 12 baseline + 3 tuned)
- [ ] Table contains: Model, Type, Accuracy, Precision, Recall, Macro-F1, ROC-AUC
- [ ] Best baseline identified: LDA (F1=0.5696)
- [ ] Best tuned identified: SVM-RBF (F1=0.5701)
- [ ] `DA3/figures/final_f1_comparison.png` exists
- [ ] `DA3/figures/final_accuracy_comparison.png` exists
- [ ] `DA3/figures/final_roc_comparison.png` exists
- [ ] `DA3/figures/final_baseline_vs_tuned.png` exists
- [ ] `DA3/figures/final_all_metrics_heatmap.png` exists

**Generate with:** `source("r/scripts/phase3/12_final_comparison.R")`

### Statistical Analysis
- [ ] `DA3/tables/cv_fold_matrix.csv` exists (12 models × 5 folds)
- [ ] `DA3/tables/friedman_results.csv` exists (chi-squared, df, p-value, decision)
- [ ] `DA3/tables/posthoc_matrix.csv` exists (if H0 rejected)
- [ ] `data/ml_r/statistical_test_results.rds` exists
- [ ] `DA3/figures/stat_cv_boxplot.png` exists
- [ ] `DA3/figures/stat_fold_heatmap.png` exists
- [ ] `DA3/figures/stat_mean_f1_bars.png` exists
- [ ] `DA3/DA3_Statistical_Analysis.md` contains: H0, H1, test statistic, p-value, conclusion

**Generate with:** `source("r/scripts/phase3/11_statistical_analysis.R")`

---

## CRITERION 2: Interactive Dashboard (2 Marks)

- [ ] `dashboard/app.R` runs without errors
- [ ] `shiny::runApp("dashboard")` launches the app from project root
- [ ] Tab 1 (Overview): KPI cards visible, OA distribution chart interactive
- [ ] Tab 2 (Model Comparison): Metric selector dropdown changes chart dynamically
- [ ] Tab 2: Baseline (blue) vs Tuned (red) colours visible
- [ ] Tab 2: Hover tooltips show exact values
- [ ] Tab 3 (Diagnostics): Model selector changes confusion matrix and ROC curves
- [ ] Tab 3: Feature importance chart shows top 15 features for Random Forest
- [ ] Tab 4 (Clusters): Pie chart shows 3 clusters with correct percentages
- [ ] Tab 4: Dominant terms chart shows terms per cluster
- [ ] `dashboard/README.md` has run instructions

---

## CRITERION 3: Result Interpretation and Discussion (2 Marks)

- [ ] `DA3/DA3_Results_Discussion.md` exists
- [ ] Section 1: Dataset distribution discussion (imbalance, class definitions)
- [ ] Section 2: Feature engineering impact (27→206→80, rationale)
- [ ] Section 3: Model comparison discussion (why LDA wins, why XGBoost fails)
- [ ] Section 4: Hyperparameter tuning discussion (modest gains explained)
- [ ] Section 5: Statistical analysis interpretation (practical vs statistical)
- [ ] Section 6: Confusion matrix analysis (partially_open hardest class)
- [ ] Section 7: ROC-AUC interpretation (discrimination ability, not accuracy)
- [ ] Section 8: Feature importance discussion (with causation warning)
- [ ] Section 9: Clustering interpretation (human labelling caveat)
- [ ] Section 10: Overall synthesis

---

## CRITERION 4: Conclusion, Limitations, Future Scope (1 Mark)

- [ ] `DA3/DA3_Conclusion_Limitations_Future.md` exists
- [ ] Conclusion section summarises DA3 achievements with actual metrics
- [ ] At least 8 distinct limitations documented with impact and justification
- [ ] L4: Partially_open ambiguity limitation included
- [ ] L5: TF-IDF (not embeddings) limitation included
- [ ] Future scope clearly marked: ❌ NOT IMPLEMENTED for all future items
- [ ] RAG, Ollama, fine-tuning all marked as NOT IMPLEMENTED
- [ ] Future scope organized by priority tiers
- [ ] Section 4 (clear table) showing IMPLEMENTED vs NOT IMPLEMENTED

---

## CRITERION 5: Final Report + Code Organization (2 Marks)

### Final Report
- [ ] `DA3/DA3_Final_Report.md` exists
- [ ] Section 5 (Performance Comparison) includes complete table with all 15 models
- [ ] Section 6 (Statistical Analysis) includes methodology, hypotheses, results
- [ ] Section 7 (Dashboard) describes all 4 tabs and how to run
- [ ] Section 8 (Discussion) summarises key findings
- [ ] Section 12 (Code Organization) shows directory tree
- [ ] References section included (≥5 references)
- [ ] Appendices included (hyperparameter grids, top features, run instructions)

### Code Organization
- [ ] `DA3/` directory exists with all deliverable documents
- [ ] `dashboard/` directory exists with `app.R` and `README.md`
- [ ] `r/scripts/phase3/` directory exists with `11_` and `12_` scripts
- [ ] DA2 scripts in `r/scripts/phase2/` unchanged
- [ ] `README.md` updated with DA3 status and quick start commands
- [ ] All new R scripts have header comments (project, purpose, inputs, outputs)
- [ ] `set.seed(42)` present in statistical analysis script

---

## CRITERION 6: Presentation + Viva (1 Mark)

- [ ] `DA3/DA3_Viva_QA.md` exists with ≥50 Q&A pairs
- [ ] Questions cover all 8 categories (dataset, features, models, tuning, stats, dashboard, interpretation, technical)
- [ ] All answers are accurate and grounded in actual results
- [ ] No future features claimed as implemented in viva answers

**For presentation:**
- [ ] Slides created (minimum: PDF or PPTX with 12 slides covering all DA3 criteria)
- [ ] Slide 5–6 covers final performance comparison with chart
- [ ] Slide 7 covers statistical analysis results
- [ ] Slide 8 covers dashboard (screenshot or live demo plan)
- [ ] Backup screenshots prepared for dashboard demo

---

## REPRODUCIBILITY CHECK

Run these in order and verify no errors:

```r
setwd("e:/Tuned_Research")

# DA3 Phase 1: Statistical analysis
source("r/scripts/phase3/11_statistical_analysis.R")
# Expected: cv_fold_matrix.csv, friedman_results.csv, 3 PNG figures

# DA3 Phase 2: Final comparison
source("r/scripts/phase3/12_final_comparison.R")
# Expected: final_model_comparison.csv, 5 PNG figures

# DA3 Phase 3: Dashboard
shiny::runApp("dashboard")
# Expected: App launches at localhost:XXXX, all 4 tabs functional
```

---

## FINAL STATUS

| Criterion | Status | Mark (self-assessment) |
|-----------|--------|------------------------|
| 1. Performance + Statistics | [ ] | /2 |
| 2. Interactive Dashboard | [ ] | /2 |
| 3. Result Interpretation | [ ] | /2 |
| 4. Conclusion/Limits/Future | [ ] | /1 |
| 5. Report + Organization | [ ] | /2 |
| 6. Presentation + Viva | [ ] | /1 |
| **TOTAL** | | **/10** |

---

## POST-DA3: OLLAMA/LLM LAYER

Only begin this after DA3 submission is confirmed complete.

- [ ] Hardware inspection (GPU, VRAM, RAM, disk)
- [ ] Ollama model selection based on hardware
- [ ] Ollama installation
- [ ] Base model testing
- [ ] Instruction dataset preparation
- [ ] Fine-tuning decision
- [ ] Integration with ResearchPilot
- [ ] End-to-end demonstration

---

*ResearchPilot DA3 Checklist | October 2026*
