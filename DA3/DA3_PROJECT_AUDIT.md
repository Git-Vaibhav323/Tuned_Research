# DA3 PROJECT AUDIT REPORT

**Date:** October 6, 2026  
**Project:** ResearchPilot — Human-Centered AI Research Assistant  
**Purpose:** Map existing DA2 implementation against DA3 rubric requirements

---

## EXECUTIVE SUMMARY

The ResearchPilot project has **completed Phase 1 and Phase 2 (DA2)** with comprehensive implementation:
- 2,000 AI/ML research papers from OpenAlex
- 27 → 206 → 80 feature engineering pipeline
- SQLite database with 4 tables
- 12 ML algorithms trained and evaluated
- 3 models hyperparameter tuned
- Proper train/val/test splits with no leakage
- Impact-tier classification (secondary task)
- Topic clustering (K-Means, k=3)
- Complete DA2 documentation and viva materials

**DA3 STATUS:** Not yet implemented. This audit identifies what needs to be built.

---

## 1. EXISTING PROJECT STRUCTURE

### 1.1 Data Assets

| Asset | Location | Size | Status |
|-------|----------|------|--------|
| Final dataset | `data/final/final_dataset.csv` | 2000×27 | ✅ Locked |
| Engineered features | `data/ml_r/enriched_features_v2.csv` | 2000×206 | ✅ Available |
| Selected features | `data/ml_r/selected_features_v2.csv` | 2000×80 | ✅ Available |
| Train/val/test splits | `data/ml_r/ml_data_splits_v2.rds` | — | ✅ Available |
| Model objects | `data/ml_r/model_objects_v2.rds` | — | ✅ Available |
| All results | `data/ml_r/all_results_v2.csv` | 30 rows | ✅ Available |

### 1.2 Database

| Item | Value |
|------|-------|
| Engine | SQLite |
| File | `database/researchpilot_r.db` |
| Tables | papers, ml_features, oa_features, model_results |
| Size | 2000 papers |
| Status | ✅ Operational |

### 1.3 R Scripts (Phase 2)

| Script | Purpose | Status |
|--------|---------|--------|
| `00_enriched_features.R` | Feature engineering (27→206) | ✅ Complete |
| `02_feature_selection_v2.R` | Feature selection (206→80) | ✅ Complete |
| `03_database_setup.R` | SQLite database creation | ✅ Complete |
| `04_database_queries.R` | SQL query demonstrations | ✅ Complete |
| `05_ml_models_v2.R` | 12 ML algorithms training | ✅ Complete |
| `06_hyperparameter_tuning_v2.R` | RF/SVM/XGBoost tuning | ✅ Complete |
| `07_model_evaluation_v2.R` | Leaderboard & metrics | ✅ Complete |
| `08_visualizations_v2.R` | 15+ visualizations | ✅ Complete |
| `09_impact_tier_v2.R` | Impact-tier classification | ✅ Complete |
| `10_clustering.R` | Topic clustering (k=3) | ✅ Complete |

### 1.4 Existing Visualizations

**Location:** `reports/figures/phase2_r/` and `DA2/figures/`

| Category | Files | Status |
|----------|-------|--------|
| Feature selection | m3_feature_selection_agreement.png | ✅ |
| Model comparison | m4_model_comparison_f1.png | ✅ |
| Tuning comparison | m5_vs_m4_test_f1.png | ✅ |
| M6 comprehensive | 8 plots (CM, ROC, PR, metrics, FI) | ✅ |
| M7 clustering | 7 plots (clusters, silhouette, impact) | ✅ |

### 1.5 Documentation

| Document | Purpose | Status |
|----------|---------|--------|
| `README.md` | Main project overview | ✅ Complete |
| `DA2/ResearchPilot_DA2_Final_Report.md` | DA2 final report | ✅ Complete |
| `DA2/DA2_Demo_Script.md` | Demo walkthrough | ✅ Complete |
| `DA2/DA2_Viva_QA.md` | Viva preparation | ✅ Complete |
| `DA2/REPRODUCIBILITY.md` | Reproducibility guide | ✅ Complete |
| M1-M7 milestone reports | Individual milestone docs | ✅ Complete |

---

## 2. EXISTING MODEL RESULTS (VERIFIED)

### 2.1 Baseline Models (Test Set, n=301)

| Rank | Model | Accuracy | Macro-F1 | ROC-AUC | Type |
|------|-------|----------|----------|---------|------|
| 1 | LDA | 0.5748 | **0.5696** | 0.7577 | Baseline |
| 2 | MLP | 0.5681 | 0.5657 | 0.7659 | Baseline |
| 3 | Logistic Regression | 0.5648 | 0.5631 | 0.7565 | Baseline |
| 4 | SVM-RBF | 0.5681 | 0.5559 | 0.7574 | Baseline |
| 5 | Elastic Net | 0.5581 | 0.5539 | 0.7584 | Baseline |
| 6 | SVM-Linear | 0.5515 | 0.5441 | 0.7587 | Baseline |
| 7 | Gradient Boosting | 0.5382 | 0.5384 | 0.7402 | Baseline |
| 8 | Random Forest | 0.5482 | 0.5317 | 0.7572 | Baseline |
| 9 | AdaBoost | 0.5150 | 0.5123 | 0.6963 | Baseline |
| 10 | Decision Tree | 0.5083 | 0.5017 | 0.6924 | Baseline |
| 11 | Naive Bayes | 0.4186 | 0.4081 | 0.6840 | Baseline |
| 12 | XGBoost | 0.3455 | 0.3428 | 0.5243 | Baseline |

### 2.2 Tuned Models (Test Set, n=301)

| Model | Best Params | Base F1 | Tuned F1 | Improvement | Accuracy |
|-------|-------------|---------|----------|-------------|----------|
| **SVM-RBF** | cost=10, gamma=0.001 | 0.5559 | **0.5701** | +0.0142 | 0.5748 |
| Random Forest | ntree=500, mtry=8 | 0.5317 | 0.5475 | +0.0158 | 0.5615 |
| XGBoost | depth=6, eta=0.10, rounds=300 | 0.3428 | 0.3537 | +0.0109 | 0.3555 |

### 2.3 Key Findings

- **Best baseline model:** LDA (F1=0.5696)
- **Best tuned model:** SVM-RBF (F1=0.5701)
- **Overall best improvement:** Random Forest (+0.0158 F1)
- **Margin between best baseline and best tuned:** +0.0005 F1
- **Test set distribution:** closed=67, fully_open=125, partially_open=109

### 2.4 Cross-Validation Data Availability

From inspection of `data/ml_r/all_results_v2.csv`:
- ✅ CV F1 scores available for tuned models
- ✅ Test metrics available for all 12 baseline + 3 tuned models
- ✅ Validation metrics available
- ❌ **MISSING:** Fold-level performance data for statistical testing

**IMPLICATION:** Need to extract or re-compute fold-level CV results to enable proper statistical analysis (Friedman test or similar).

---

## 3. DA3 RUBRIC MAPPING

### DA3 Official Criteria (10 Marks Total)

| # | Criterion | Marks | Status | Implementation Needed |
|---|-----------|-------|--------|----------------------|
| 1 | Final performance comparison + statistical analysis | 2 | ⚠️ Partial | Statistical test missing |
| 2 | Interactive visualization/dashboard (Plotly/Shiny) | 2 | ❌ Missing | Build R Shiny app |
| 3 | Result interpretation and discussion | 2 | ⚠️ Partial | Expand interpretation |
| 4 | Conclusion, limitations, future scope | 1 | ⚠️ Partial | Formalize sections |
| 5 | Final report quality + code organization | 2 | ⚠️ Partial | DA3-specific report needed |
| 6 | Final presentation + viva | 1 | ❌ Missing | Create presentation |

---

## 4. DETAILED GAP ANALYSIS

### 4.1 Criterion 1: Performance Comparison + Statistical Analysis

**EXISTING:**
✅ Complete model comparison table (12 baseline + 3 tuned)  
✅ All metrics: Accuracy, Precision, Recall, F1, ROC-AUC  
✅ Baseline vs tuned comparison  
✅ Proper test set evaluation  
✅ CV F1 scores for tuned models

**MISSING:**
❌ **Statistical significance testing**  
❌ Fold-level performance data for statistical test  
❌ Friedman test or equivalent multi-model comparison  
❌ Post-hoc analysis if significant differences found  
❌ Formal hypothesis testing documentation

**REQUIRED IMPLEMENTATION:**
1. Extract or recompute fold-level CV performance for comparable models
2. Perform Friedman test (or appropriate alternative)
3. If significant: perform Nemenyi or Wilcoxon post-hoc tests
4. Document: null/alternative hypothesis, test statistic, p-value, conclusion
5. Create publication-ready statistical results table

**ESTIMATED EFFORT:** 4-6 hours

---

### 4.2 Criterion 2: Interactive Dashboard

**EXISTING:**
✅ Static visualizations (15+ plots)  
✅ All data available in CSV/RDS format  
✅ Database operational

**MISSING:**
❌ **Entire interactive dashboard**  
❌ R Shiny application  
❌ Interactive Plotly charts  
❌ User controls (metric selector, model selector)  
❌ Multiple tabs/pages  
❌ KPI cards  
❌ Deployment capability

**REQUIRED IMPLEMENTATION:**

**Tab 1: Overview**
- KPI cards: total papers, features, models, test size
- Interactive OA category distribution (Plotly bar chart)
- Paper distribution by year
- Summary statistics

**Tab 2: Model Comparison**
- Metric selector dropdown (Accuracy/Precision/Recall/F1/AUC)
- Dynamic Plotly bar chart
- Baseline vs tuned distinction
- Hover tooltips with exact values
- Export capability

**Tab 3: Model Diagnostics**
- Model selector dropdown
- Confusion matrix heatmap (Plotly)
- ROC curves (Plotly)
- Precision-Recall curves
- Feature importance (for tree-based models)
- Class-level metrics table

**Tab 4: Research Clusters**
- Interactive cluster size visualization
- Dominant terms word cloud or bar chart
- Research theme labels
- Cluster profiles table

**ESTIMATED EFFORT:** 12-16 hours

---

### 4.3 Criterion 3: Result Interpretation & Discussion

**EXISTING:**
✅ Individual milestone reports (M1-M7)  
✅ Technical documentation  
✅ DA2 final report with results

**MISSING:**
❌ **Comprehensive DA3 results discussion section**  
❌ Integrated interpretation across all components  
❌ Discussion of statistical analysis results  
❌ Feature importance interpretation  
❌ Confusion matrix deep-dive  
❌ ROC-AUC interpretation by class  
❌ Clustering theme discussion  
❌ Model failure analysis

**REQUIRED IMPLEMENTATION:**

1. **Dataset & Distribution Analysis**
   - Class imbalance impact
   - Feature distribution insights

2. **Feature Engineering Impact**
   - Which feature groups were most predictive
   - Feature selection rationale and impact

3. **Model Comparison Discussion**
   - Why LDA performed well (linear separability)
   - Why XGBoost struggled (overfitting, data size)
   - Ensemble vs linear model tradeoffs

4. **Hyperparameter Tuning Discussion**
   - Modest gains explained
   - Diminishing returns

5. **Statistical Analysis Interpretation**
   - What the statistical test reveals
   - Practical vs statistical significance

6. **Confusion Matrix Analysis**
   - Which classes are confused
   - Possible reasons (metadata overlap)

7. **Feature Importance Discussion**
   - Top features and their meaning
   - Domain insights

8. **Clustering Interpretation**
   - Research themes identified
   - Cluster characteristics
   - Relationship to OA categories (if any)

**ESTIMATED EFFORT:** 6-8 hours

---

### 4.4 Criterion 4: Conclusion, Limitations, Future Scope

**EXISTING:**
✅ README mentions limitations  
✅ Future roadmap section exists  
✅ DA2 report has conclusions

**MISSING:**
❌ **Formal DA3 conclusion section**  
❌ Structured limitations discussion  
❌ Detailed future scope with priorities

**REQUIRED IMPLEMENTATION:**

**Conclusion:**
- Summary of DA3 achievements
- Key findings (best model, statistical results)
- Contribution statement

**Limitations:**
1. Dataset size (2000 papers)
2. Domain scope (AI/ML only)
3. Moderate classification performance (~57% accuracy)
4. Three-class overlap (partially_open ambiguity)
5. TF-IDF features (not semantic embeddings)
6. Static dataset (not continuously updated)
7. Metadata-dependent (missing DOI/publisher affects features)
8. Cluster interpretation subjectivity

**Future Scope:**
1. RAG-based research assistant (high priority)
2. Semantic embeddings (Sentence-BERT, SciBERT)
3. Research gap detection
4. Paper similarity search
5. Journal recommendation system
6. Larger dataset (10k+ papers)
7. Multi-domain expansion
8. Real-time data pipeline
9. **Local LLM integration (Ollama) — POST-DA3**
10. Fine-tuning for research tasks — POST-DA3

**ESTIMATED EFFORT:** 3-4 hours

---

### 4.5 Criterion 5: Final Report + Code Organization

**EXISTING:**
✅ DA2 final report  
✅ R scripts organized in `r/scripts/phase2/`  
✅ Data files organized  
✅ Documentation exists

**MISSING:**
❌ **DA3-specific final report**  
❌ Consolidated report covering DA3 criteria  
❌ Code organization audit for DA3  
❌ Clear file naming for DA3 deliverables

**REQUIRED IMPLEMENTATION:**

**DA3 Final Report Structure:**
1. Title & Abstract
2. Executive Summary
3. Project Context (brief DA2 recap)
4. DA3 Objectives
5. Final Model Comparison (Criterion 1)
6. Statistical Analysis (Criterion 1)
7. Interactive Dashboard (Criterion 2)
8. Results & Discussion (Criterion 3)
9. Conclusion (Criterion 4)
10. Limitations (Criterion 4)
11. Future Scope (Criterion 4)
12. References
13. Appendices

**Code Organization:**
```
Tuned_Research/
├── DA3/                          # DA3-specific deliverables
│   ├── DA3_Final_Report.md
│   ├── DA3_Statistical_Analysis.md
│   ├── DA3_Results_Discussion.md
│   ├── DA3_Presentation.pdf
│   ├── figures/                  # DA3-specific figures
│   └── tables/                   # DA3-specific tables
├── dashboard/
│   └── app.R                     # R Shiny dashboard
├── r/scripts/phase2/             # Existing (no change needed)
├── data/ml_r/                    # Existing (no change needed)
├── database/                     # Existing (no change needed)
└── README.md                     # Update with DA3 status
```

**ESTIMATED EFFORT:** 8-10 hours

---

### 4.6 Criterion 6: Final Presentation + Viva

**EXISTING:**
✅ DA2 viva preparation  
✅ DA2 demo script  
✅ Existing visualizations

**MISSING:**
❌ **DA3 presentation slides**  
❌ DA3 viva Q&A preparation  
❌ Live demo script for dashboard  
❌ Statistical results presentation

**REQUIRED IMPLEMENTATION:**

**Presentation (12 slides):**
1. Title: ResearchPilot DA3
2. Problem & Objectives
3. Overall Pipeline (Phase 1 → Phase 2 → DA3)
4. Dataset & Features (27→206→80)
5. 12 ML Models Overview
6. **Final Performance Comparison** (DA3 focus)
7. **Statistical Analysis Results** (DA3 focus)
8. **Interactive Dashboard Demo** (DA3 focus)
9. Key Findings & Interpretation
10. Clustering & Research Themes
11. Limitations & Future Scope
12. Conclusion

**Viva Preparation:**
- Expected questions based on DA3 criteria
- Statistical test explanation
- Dashboard technical choices
- Interpretation justifications
- Limitation acknowledgments

**Live Demo Script:**
1. Show dashboard overview tab
2. Demonstrate metric selector interactivity
3. Show model diagnostics for best model
4. Display clustering visualization
5. Answer "what if" questions using dashboard

**ESTIMATED EFFORT:** 6-8 hours

---

## 5. CRITICAL FINDINGS

### 5.1 Data Integrity ✅

All DA2 results have been verified:
- ✅ Proper train/val/test splits (no leakage)
- ✅ Feature engineering fitted on training only
- ✅ Consistent test set used across all evaluations
- ✅ Results match across CSV files and reports
- ✅ Model objects saved and reproducible

### 5.2 Missing Statistical Analysis ⚠️

**CRITICAL:** The DA3 rubric explicitly requires statistical analysis (2 marks), but:
- No statistical significance testing has been performed
- Fold-level CV data may need to be extracted/recomputed
- This is the HIGHEST PRIORITY gap for DA3

### 5.3 No Interactive Dashboard ❌

**CRITICAL:** Interactive dashboard is worth 2 marks but is completely missing:
- No R Shiny app exists
- No interactive Plotly charts
- This is the SECOND HIGHEST PRIORITY gap for DA3

### 5.4 Fragmented Documentation ⚠️

Existing documentation is comprehensive for DA2 but needs DA3-specific integration:
- No single DA3 final report
- Interpretation scattered across milestone reports
- Needs consolidation

---

## 6. EXISTING RESOURCES THAT CAN BE REUSED

### 6.1 For Statistical Analysis
- ✅ `data/ml_r/all_results_v2.csv` — test results
- ✅ `data/ml_r/model_objects_v2.rds` — trained models
- ✅ `data/ml_r/ml_data_splits_v2.rds` — consistent splits
- ⚠️ May need: fold-level CV results extraction

### 6.2 For Dashboard
- ✅ `data/ml_r/all_results_v2.csv` — metrics
- ✅ `data/ml_r/enriched_features_v2.csv` — feature data
- ✅ `data/ml_r/cluster_assignments.csv` — clustering
- ✅ `data/ml_r/rf_importance_v2.rds` — feature importance
- ✅ `database/researchpilot_r.db` — SQLite database
- ✅ All existing visualizations as reference

### 6.3 For Results Discussion
- ✅ All milestone reports (M1-M7)
- ✅ `DA2/ResearchPilot_DA2_Final_Report.md`
- ✅ `DA2/M6_Comparative_Analysis_Report.md`
- ✅ Confusion matrices, ROC curves, feature importance plots

---

## 7. IMPLEMENTATION PRIORITY QUEUE

### Phase 1: Core DA3 Requirements (HIGH PRIORITY)

**Priority 1.1: Statistical Analysis (HIGHEST)**
- Task: Implement Friedman test or equivalent
- Deliverable: Statistical test results with p-values
- Estimated effort: 4-6 hours
- **MUST BE DONE FIRST**

**Priority 1.2: Interactive Dashboard (HIGH)**
- Task: Build R Shiny + Plotly dashboard
- Deliverable: Functional app.R with 4 tabs
- Estimated effort: 12-16 hours
- **SECOND PRIORITY**

**Priority 1.3: Results Discussion (HIGH)**
- Task: Write comprehensive interpretation
- Deliverable: DA3_Results_Discussion.md
- Estimated effort: 6-8 hours
- **THIRD PRIORITY**

### Phase 2: Documentation & Presentation (MEDIUM PRIORITY)

**Priority 2.1: DA3 Final Report**
- Task: Consolidate all DA3 content
- Deliverable: DA3_Final_Report.md
- Estimated effort: 8-10 hours

**Priority 2.2: Presentation Slides**
- Task: Create 12-slide presentation
- Deliverable: DA3_Presentation.pdf
- Estimated effort: 4-6 hours

**Priority 2.3: Viva Preparation**
- Task: Prepare Q&A document
- Deliverable: DA3_Viva_QA.md
- Estimated effort: 2-4 hours

### Phase 3: Polish & Verification (LOW PRIORITY)

**Priority 3.1: Code Organization**
- Task: Create DA3 directory structure
- Estimated effort: 2-3 hours

**Priority 3.2: README Update**
- Task: Update main README with DA3 status
- Estimated effort: 1 hour

---

## 8. POST-DA3: OLLAMA/LLM LAYER

**IMPORTANT:** This should ONLY begin after all 6 DA3 criteria are satisfied.

### Current Status
❌ Not implemented  
❌ No Ollama installation  
❌ No LLM model selected  
❌ No fine-tuning dataset  
❌ No RAG implementation

### Required Steps (POST-DA3)
1. Hardware inspection (GPU, VRAM, RAM)
2. Model selection (Qwen/Llama/Gemma based on hardware)
3. Ollama installation and configuration
4. Instruction dataset preparation (paper summarization, Q&A)
5. Base model evaluation
6. Fine-tuning decision (LoRA/QLoRA if beneficial)
7. RAG/retrieval layer implementation
8. Integration with ResearchPilot
9. Evaluation (base vs fine-tuned)
10. End-to-end demonstration

**ESTIMATED EFFORT:** 20-30 hours (AFTER DA3 completion)

---

## 9. RISK ASSESSMENT

### High Risk Items

1. **Statistical Analysis Fold Data Availability**
   - Risk: Fold-level CV data may not exist in saved format
   - Mitigation: Extract from model_objects_v2.rds or recompute CV
   - Impact: Could delay statistical analysis by 2-4 hours

2. **Dashboard Performance with Large Data**
   - Risk: 2000 rows × 206 features may cause Shiny slowdown
   - Mitigation: Use reactive programming, data caching, plotly optimization
   - Impact: May need performance tuning

### Medium Risk Items

3. **Time Constraint**
   - Risk: DA3 submission is October 16, 2026 (10 days)
   - Mitigation: Prioritize core criteria (1, 2, 3) first
   - Impact: May need to defer polish/documentation to later iterations

4. **R Package Dependencies**
   - Risk: Dashboard may require additional packages
   - Mitigation: Test installation early, document requirements
   - Impact: Could add 1-2 hours setup time

### Low Risk Items

5. **Interpretation Subjectivity**
   - Risk: Results discussion may be challenged in viva
   - Mitigation: Ground all interpretations in actual data, avoid overclaiming
   - Impact: Minor, can be addressed in viva Q&A

---

## 10. RECOMMENDATIONS

### Immediate Actions (Next 24 Hours)

1. ✅ **DONE:** Complete project audit (this document)
2. ⏭️ **NEXT:** Extract/verify fold-level CV data availability
3. ⏭️ **NEXT:** Design statistical analysis approach
4. ⏭️ **NEXT:** Create dashboard wireframe/mockup

### Week 1 Focus (Oct 6-10)

- Statistical analysis implementation and verification
- Dashboard development (all 4 tabs)
- Dashboard testing and refinement

### Week 2 Focus (Oct 11-16)

- Results discussion writing
- DA3 final report consolidation
- Presentation slide creation
- Viva preparation
- Final testing and submission preparation

### Post-DA3 (After Oct 16)

- Begin Ollama/LLM layer ONLY after DA3 submission
- Hardware inspection first
- Incremental implementation with testing

---

## 11. DELIVERABLE CHECKLIST (DA3)

### Must-Have (Required for 10 Marks)

- [ ] Statistical significance test results (Criterion 1)
- [ ] Final model comparison table (Criterion 1)
- [ ] Interactive R Shiny dashboard with 4 tabs (Criterion 2)
- [ ] Comprehensive results discussion document (Criterion 3)
- [ ] Formal conclusion section (Criterion 4)
- [ ] Documented limitations (Criterion 4)
- [ ] Future scope section (Criterion 4)
- [ ] DA3 final report (Criterion 5)
- [ ] Organized DA3 code structure (Criterion 5)
- [ ] Presentation slides (12 slides) (Criterion 6)
- [ ] Viva Q&A preparation (Criterion 6)
- [ ] Live dashboard demo script (Criterion 6)

### Nice-to-Have (Enhances Quality)

- [ ] Additional statistical tests (effect size, confidence intervals)
- [ ] Dashboard deployment instructions
- [ ] Video recording of dashboard demo
- [ ] Extended interpretation with literature references
- [ ] Code comments and documentation
- [ ] Unit tests for dashboard components

---

## 12. CONCLUSION

The ResearchPilot project has a **strong DA2 foundation** with:
- Complete and reproducible ML pipeline
- Verified results with proper data splits
- Comprehensive documentation
- All required data artifacts available

**DA3 implementation is feasible** within the given timeframe but requires focused effort on:
1. **Statistical analysis** (highest priority, missing component)
2. **Interactive dashboard** (second priority, completely missing)
3. **Integrated results discussion** (third priority, needs consolidation)

The project is well-positioned for DA3 completion and subsequent Ollama/LLM layer addition.

**Estimated Total DA3 Effort:** 50-60 hours  
**Available Time:** 10 days  
**Recommendation:** Focus on must-have deliverables first, add enhancements as time permits.

---

**Audit Completed:** October 6, 2026  
**Next Step:** Begin statistical analysis implementation  
**Status:** Ready to proceed with DA3 implementation

