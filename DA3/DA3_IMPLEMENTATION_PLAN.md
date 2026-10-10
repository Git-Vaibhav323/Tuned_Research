# DA3 IMPLEMENTATION PLAN

**Project:** ResearchPilot DA3  
**Date:** October 6, 2026  
**Submission Deadline:** October 16, 2026 (10 days)  
**Based on:** DA3 Project Audit

---

## IMPLEMENTATION STRATEGY

This plan follows the exact DA3 rubric requirements and implements components in priority order to ensure all 10 marks are achievable within the deadline.

---

## PHASE 1: STATISTICAL ANALYSIS (CRITERION 1, Part 2)

**Priority:** HIGHEST  
**Marks:** 1 out of 2 (Criterion 1)  
**Status:** ❌ Not started  
**Estimated Effort:** 6-8 hours

### Objective

Perform rigorous statistical analysis to compare model performance across multiple algorithms and demonstrate statistically significant differences (or lack thereof).

### Tasks

#### Task 1.1: Data Preparation for Statistical Testing
**File:** `r/scripts/phase2/11_statistical_analysis.R`

```r
# Extract fold-level performance from CV results
# OR: Recompute 5-fold CV for comparable models
# Create matrix: models × folds × metric
```

**Deliverables:**
- Fold-level performance matrix (models × folds)
- Save as `data/ml_r/cv_fold_results.csv`
- At minimum: top 6-8 models with CV results

**Time:** 2-3 hours

#### Task 1.2: Friedman Test Implementation

**Statistical Approach:**
- **Null Hypothesis (H₀):** All models have equal median performance
- **Alternative Hypothesis (H₁):** At least one model has different median performance
- **Significance Level:** α = 0.05
- **Test:** Friedman test (non-parametric, repeated measures)

```r
# Friedman test for multiple model comparison
friedman_result <- friedman.test(y = performance, 
                                  groups = model, 
                                  blocks = fold)
```

**Deliverables:**
- Test statistic (Chi-square)
- Degrees of freedom
- p-value
- Decision (reject/fail to reject H₀)

**Time:** 1-2 hours

#### Task 1.3: Post-Hoc Analysis (If Significant)

If p < 0.05:
- **Nemenyi post-hoc test** for pairwise comparisons
- Critical difference (CD) calculation
- Pairwise significance matrix

```r
# Nemenyi post-hoc test
library(PMCMRplus)
nemenyi_result <- frdAllPairsNemenyiTest(y = performance,
                                          groups = model,
                                          blocks = fold)
```

**Deliverables:**
- Pairwise comparison matrix
- Critical difference diagram
- Significantly different pairs identified

**Time:** 2 hours

#### Task 1.4: Visualization

**Figures to create:**
1. **Critical Difference Diagram** (Nemenyi plot)
2. **Model Performance Distribution** (boxplot by model)
3. **Fold-wise Performance** (heatmap: models × folds)

**Deliverables:**
- `DA3/figures/statistical_analysis_cd_diagram.png`
- `DA3/figures/statistical_analysis_distribution.png`
- `DA3/figures/statistical_analysis_fold_heatmap.png`

**Time:** 1-2 hours

#### Task 1.5: Documentation

**File:** `DA3/DA3_Statistical_Analysis.md`

**Content:**
1. Research Question
2. Methodology
3. Null and Alternative Hypotheses
4. Test Selection Rationale (why Friedman)
5. Results
   - Test statistic
   - p-value
   - Interpretation
6. Post-hoc Results (if applicable)
7. Practical Significance Discussion
8. Conclusion

**Time:** 1 hour

### Acceptance Criteria
- [ ] Fold-level CV data extracted/computed for ≥6 models
- [ ] Friedman test executed successfully
- [ ] p-value calculated and interpreted
- [ ] Post-hoc analysis completed if p < 0.05
- [ ] 3 statistical visualizations created
- [ ] Statistical analysis document completed
- [ ] Results are reproducible with set.seed()

---

## PHASE 2: INTERACTIVE DASHBOARD (CRITERION 2)

**Priority:** HIGH  
**Marks:** 2 out of 10  
**Status:** ❌ Not started  
**Estimated Effort:** 14-18 hours

### Objective

Build a fully interactive R Shiny dashboard with Plotly visualizations that allows dynamic exploration of ResearchPilot results.

### Technology Stack
- **Framework:** R Shiny
- **Plotting:** Plotly (interactive)
- **UI:** shinydashboard or bslib
- **Data:** Load from CSV/RDS files

### Dashboard Structure

#### Tab 1: Overview

**KPI Cards (top row):**
- Total Papers: 2000
- Raw Features: 27
- Engineered Features: 206
- Selected Features: 80
- ML Models: 12
- Test Set Size: 301

**Visualizations:**
1. **OA Category Distribution** (interactive Plotly bar chart)
   - fully_open: 834
   - partially_open: 723
   - closed: 443
   - Click to filter (optional)

2. **Papers by Publication Year** (Plotly line chart)

3. **Feature Engineering Pipeline** (Sankey diagram)
   - 27 → 206 → 80

**Time:** 3-4 hours

#### Tab 2: Model Comparison

**Controls:**
- **Metric Selector:** Dropdown (Accuracy / Precision / Recall / F1 / ROC-AUC)
- **Model Type Filter:** Checkboxes (Baseline / Tuned / All)

**Main Visualization:**
- **Dynamic Plotly Bar Chart**
  - X-axis: Model names
  - Y-axis: Selected metric
  - Color: Baseline (blue) vs Tuned (orange)
  - Hover: Exact values, model type, params (if tuned)
  - Sorted by performance (descending)

**Additional Elements:**
- **Summary Table** below chart
  - Top 5 models for selected metric
  - All metric values visible

**Interactivity:**
- Change metric → chart updates
- Change filter → models update
- Click bar → show model details in sidebar

**Time:** 4-5 hours

#### Tab 3: Model Diagnostics

**Controls:**
- **Model Selector:** Dropdown (all 15 models: 12 baseline + 3 tuned)

**Visualizations (based on selected model):**

1. **Confusion Matrix** (Plotly heatmap)
   - Rows: Predicted
   - Columns: Actual
   - Color: Count
   - Annotated with numbers

2. **ROC Curves** (Plotly line chart)
   - One-vs-Rest for each class
   - Show AUC in legend
   - Diagonal reference line

3. **Precision-Recall Curves** (Plotly line chart)
   - Per-class curves
   - Show AP in legend

4. **Feature Importance** (for tree-based models only)
   - Plotly horizontal bar chart
   - Top 20 features
   - Sorted by importance

5. **Class-Level Metrics Table**
   - Precision, Recall, F1 per class

**Conditional Display:**
- Feature importance only for: Random Forest, XGBoost, AdaBoost, Decision Tree, Gradient Boosting
- Show message for others: "Feature importance not available for this model type"

**Time:** 5-6 hours

#### Tab 4: Research Clusters

**Visualizations:**

1. **Cluster Size Distribution** (Plotly pie/donut chart)
   - Cluster 1: 10% (Education/GenAI/ChatGPT)
   - Cluster 2: 71.1% (General ML/Robotics)
   - Cluster 3: 18.9% (Medical Imaging/Deep Learning)
   - Click to filter

2. **Cluster Profiles** (Plotly stacked bar or grouped bar)
   - Show OA category distribution within each cluster
   - X: Cluster, Y: Count, Color: OA category

3. **Dominant Terms** (Plotly bar chart or word cloud alternative)
   - Top 10 terms per cluster
   - Interactive: click term to highlight papers

4. **Cluster Summary Table**
   - Cluster ID
   - Size (n and %)
   - Interpreted Theme
   - Top 3 Terms
   - Avg Citation Count
   - Most common OA category

**Time:** 2-3 hours

### Additional Dashboard Features

**Global:**
- Sidebar navigation
- Project logo/title
- "About" info box
- Data last updated timestamp
- Download button for displayed data (CSV)

**Responsiveness:**
- Mobile-friendly layout (optional)
- Collapsible sidebar

**Performance:**
- Cache loaded data with `reactive()`
- Debounce user inputs for expensive operations

### File Structure

```
dashboard/
├── app.R                    # Main Shiny app
├── global.R                 # Data loading, global objects
├── ui.R                     # UI definition (optional separate file)
├── server.R                 # Server logic (optional separate file)
├── modules/                 # Modular components (optional)
│   ├── mod_overview.R
│   ├── mod_model_comparison.R
│   ├── mod_diagnostics.R
│   └── mod_clusters.R
├── utils/                   # Helper functions
│   └── plotting_utils.R
└── www/                     # Static assets
    ├── custom.css
    └── logo.png
```

**Recommended:** Start with single `app.R`, refactor to modules if needed.

### Testing Checklist

- [ ] App launches without errors
- [ ] All tabs render correctly
- [ ] Metric selector updates chart in Tab 2
- [ ] Model selector updates diagnostics in Tab 3
- [ ] Hover tooltips show correct information
- [ ] All Plotly charts are interactive (zoom, pan, hover)
- [ ] No console errors in R
- [ ] Data loads correctly from files
- [ ] Performance is acceptable (<2s for tab switches)

### Deployment (Optional)

If time permits:
- Deploy to shinyapps.io (free tier)
- Document deployment steps
- Include URL in report

**Time for deployment:** +2 hours (OPTIONAL)

### Acceptance Criteria

- [ ] Dashboard app runs locally
- [ ] 4 tabs implemented and functional
- [ ] All interactive elements work
- [ ] Plotly charts are genuinely interactive
- [ ] Metric/model selectors update displays correctly
- [ ] Data loaded from existing project files (no hardcoding)
- [ ] Code is organized and commented
- [ ] README with run instructions created

---

## PHASE 3: RESULTS INTERPRETATION & DISCUSSION (CRITERION 3)

**Priority:** HIGH  
**Marks:** 2 out of 10  
**Status:** ⚠️ Partial (DA2 reports exist)  
**Estimated Effort:** 8-10 hours

### Objective

Create a comprehensive, integrated discussion of all ResearchPilot results with deep interpretation and domain insights.

### Structure

**File:** `DA3/DA3_Results_Discussion.md`

### Section 1: Dataset & Distribution Analysis (1 hour)

**Content:**
- 2000 papers from OpenAlex API (AI/ML domain)
- OA category distribution: fully_open (41.7%) > partially_open (36.2%) > closed (22.2%)
- Class imbalance impact on model performance
- Temporal distribution (publication years)
- Citation patterns and relationship to OA status

**Key Interpretations:**
- Why fully_open is the majority class
- How imbalance affects precision/recall tradeoffs
- What "partially_open" ambiguity means for classification

### Section 2: Feature Engineering Impact (1 hour)

**Content:**
- 27 raw columns → 206 engineered features
- Feature categories: structural, domain, publisher, TF-IDF
- 206 → 80 feature selection (NZV, correlation, RF importance)
- Rationale for each feature group

**Key Interpretations:**
- Which feature groups contributed most to predictive power
- Why TF-IDF terms were important
- Why publisher features (DOI prefix) matter
- Impact of feature selection on model complexity

### Section 3: Model Comparison Discussion (1.5 hours)

**Content:**
- 12 baseline models evaluated
- LDA best baseline (F1=0.5696): Why?
  - Linear separability in feature space
  - Dimensionality reduction effect
  - Robustness to multicollinearity
- MLP second (F1=0.5657): Why?
  - Non-linear patterns captured
  - Hidden layer representation learning
- XGBoost struggled (F1=0.3428): Why?
  - Possible overfitting
  - Small dataset relative to complexity
  - TF-IDF sparsity issues

**Key Interpretations:**
- Linear models (LDA, LR, Elastic Net) performed well → suggests linear separability
- Tree ensembles mid-range → feature interactions exist but not dominant
- SVM-RBF competitive → kernel captures some non-linearity
- Naive Bayes weak → violated independence assumption (TF-IDF terms correlated)

### Section 4: Hyperparameter Tuning Discussion (1 hour)

**Content:**
- 3 models tuned: RF, SVM-RBF, XGBoost
- Best tuned: SVM-RBF (F1=0.5701)
- Improvement over best baseline: only +0.0005 F1

**Key Interpretations:**
- Why tuning gains were modest:
  - LDA already found near-optimal linear boundary
  - Dataset size limits tuning benefit
  - Feature engineering already captured key patterns
  - Diminishing returns
- Individual model improvements:
  - RF: +0.0158 F1 (worthwhile)
  - SVM-RBF: +0.0142 F1 (worthwhile)
  - XGBoost: +0.0109 F1 (still weak)
- Practical significance vs statistical significance

### Section 5: Statistical Analysis Interpretation (1.5 hours)

**Content:**
- Friedman test results
- p-value and its meaning
- Post-hoc pairwise comparisons (if significant)
- Critical difference (CD) interpretation
- Effect size considerations

**Key Interpretations:**
- If p < 0.05:
  - Models do have statistically significant performance differences
  - Which pairs are significantly different
  - Practical vs statistical significance discussion
- If p ≥ 0.05:
  - No statistically significant difference detected
  - Does NOT mean models are identical
  - Power analysis and sample size discussion
  - Practical differences may still exist
- Relationship between CV performance and test performance
- Variance across folds and what it reveals

### Section 6: Confusion Matrix & Error Analysis (1 hour)

**Content:**
- Best model (LDA) confusion matrix deep dive
- Which classes are most confused with each other
- Precision vs recall tradeoffs per class

**Example Analysis:**
- fully_open: highest recall (0.704) but precision varies
  - Model captures most fully_open papers
  - But also misclassifies some closed/partial as fully_open
- partially_open: hardest to classify
  - Overlaps with both other categories
  - Metadata ambiguity
- closed: often confused with partially_open
  - Similar publisher patterns?
  - Temporal effects?

**Key Interpretations:**
- Error patterns reveal metadata limitations
- Some papers genuinely ambiguous
- Classification difficulty is domain-inherent, not just model weakness

### Section 7: ROC-AUC Interpretation (30 min)

**Content:**
- ROC-AUC as discrimination ability (not classification accuracy)
- One-vs-Rest macro AUC = 0.7577 (LDA)
- Class-level AUC breakdown:
  - closed: 0.8256 (best discrimination)
  - fully_open: 0.7701
  - partially_open: 0.6759 (worst discrimination)

**Key Interpretations:**
- AUC > 0.75 is reasonably good for this task
- closed is most separable → distinctive features
- partially_open overlaps significantly → inherent ambiguity
- Relationship between AUC and F1 per class

### Section 8: Feature Importance Discussion (1 hour)

**Content:**
- Random Forest feature importance (Gini-based)
- Top features:
  - pub_ieee, pub_mdpi (publisher effects)
  - title_to_abstract_ratio (structural)
  - abstract_length, title_length (text length)
  - text_richness (vocabulary diversity)
  - concept_count, keyword_count (metadata richness)

**Key Interpretations:**
- Publisher features dominate → policy-driven OA patterns
- Text length features → longer abstracts correlate with OA status (why?)
- Domain features less important than expected (why?)
- TF-IDF terms individually weak but collectively useful
- **CRITICAL:** Feature importance ≠ causation
  - Correlation vs causation warning
  - Confounding variables discussion

### Section 9: Clustering Interpretation (1 hour)

**Content:**
- K-Means clustering (k=3) on TF-IDF features
- Silhouette score ~0.06 (weak separation expected)
- Cluster themes:
  1. Education / GenAI / ChatGPT (10%)
  2. General ML / Prediction / Robotics (71.1%)
  3. Medical Imaging / Deep Learning (18.9%)

**Key Interpretations:**
- Clustering captures research themes, not OA categories
- Themes align with AI/ML subfields
- Weak separation reflects domain overlap (AI/ML is interconnected)
- Cluster labels are human-interpreted from dominant terms (not algorithmic labels)
- Relationship (or lack thereof) between clusters and OA status
- Why K=3 was chosen (silhouette scores, interpretability)

### Section 10: Overall Discussion (1 hour)

**Content:**
- Synthesis of all findings
- What works: feature engineering, database integration, reproducibility
- What's challenging: moderate predictive performance, partially_open ambiguity
- Domain insights: OA publishing patterns in AI/ML
- Comparison to prior work (if any literature available)

**Key Interpretations:**
- 57% accuracy is reasonable given:
  - Three-class problem with overlap
  - Metadata-only features (no full text)
  - Domain ambiguity
- Project demonstrates solid data science process
- Limitations acknowledged

### Acceptance Criteria

- [ ] All 10 sections completed
- [ ] Every interpretation grounded in actual data/results
- [ ] No overclaiming or exaggeration
- [ ] Statistical analysis discussed
- [ ] Feature importance interpretation includes causation warning
- [ ] Clustering interpretation clarifies human labeling
- [ ] Confusion matrix error patterns explained
- [ ] ROC-AUC interpreted correctly (discrimination, not accuracy)
- [ ] 8000-10000 words total
- [ ] Clear, well-structured, B.Tech-appropriate language

---

## PHASE 4: CONCLUSION, LIMITATIONS, FUTURE SCOPE (CRITERION 4)

**Priority:** MEDIUM  
**Marks:** 1 out of 10  
**Status:** ⚠️ Partial  
**Estimated Effort:** 4-5 hours

### Objective

Formalize project conclusion, honestly document limitations, and clearly separate implemented features from future scope.

### File

`DA3/DA3_Conclusion_Limitations_Future.md`

### Section 1: Conclusion (1 hour)

**Content:**
- Summary of DA3 achievements
- Key findings:
  - Best model: SVM-RBF tuned (F1=0.5701)
  - Best baseline: LDA (F1=0.5696)
  - Statistical analysis results (include p-value)
  - Research themes identified via clustering
- Contribution statement:
  - Complete reproducible ML pipeline
  - Database-backed research data management
  - Interactive exploration via dashboard
  - Statistical validation
  - Foundation for future research assistant features

**Length:** 500-800 words

### Section 2: Limitations (1.5 hours)

**Format:** Structured list with explanation for each

1. **Dataset Size**
   - Only 2000 papers
   - Impact: Limited generalization, small tuning gains
   - Justification: Sufficient for B.Tech project, API constraints

2. **Domain Scope**
   - AI/ML only
   - Impact: Findings not generalizable to other fields
   - Justification: Focused domain enables better feature engineering

3. **Classification Performance**
   - ~57% accuracy (moderate)
   - Impact: Not production-ready
   - Justification: Three-class overlap, metadata-only features

4. **Partially_Open Ambiguity**
   - Class definition varies by publisher
   - Impact: Inherent noise in target variable
   - Justification: OpenAlex categorization used as-is

5. **Feature Type: TF-IDF**
   - Not semantic embeddings
   - Impact: Misses semantic similarity
   - Justification: Interpretable, computationally efficient for project scope

6. **Static Dataset**
   - Not continuously updated
   - Impact: Model trained on 2024-2025 snapshot
   - Justification: Project timeline constraint

7. **Metadata Dependency**
   - Relies on OpenAlex metadata completeness
   - Impact: Missing DOI/publisher reduces feature quality
   - Justification: Real-world data challenge

8. **Clustering Subjectivity**
   - Cluster labels human-interpreted
   - Impact: Another researcher might label differently
   - Justification: Unsupervised method inherent characteristic

9. **No Full-Text Analysis**
   - Only titles and abstracts
   - Impact: Misses methodology details in paper body
   - Justification: OpenAlex API limitation, copyright constraints

10. **Cross-Validation Folds**
    - Only 5 folds for computational efficiency
    - Impact: Higher variance in CV estimates
    - Justification: Balanced accuracy and computation

**Length:** 1500-2000 words

### Section 3: Future Scope (1.5 hours)

**CRITICAL:** Clearly mark what is NOT implemented vs what could be implemented.

#### Future Scope Tier 1: High Priority (Next Implementation)

1. **RAG-Based Research Assistant** ❌ NOT IMPLEMENTED
   - Description: Retrieval-Augmented Generation for Q&A
   - Benefit: Grounded responses from paper corpus
   - Approach: Vector DB + LLM
   - Estimated effort: 30-40 hours

2. **Local LLM Integration (Ollama)** ❌ NOT IMPLEMENTED
   - Description: Privacy-preserving local inference
   - Benefit: No data sent to external APIs
   - Models: Qwen, Llama, Gemma (based on hardware)
   - Estimated effort: 20-30 hours

3. **Fine-Tuning for Research Tasks** ❌ NOT IMPLEMENTED
   - Description: LoRA/QLoRA fine-tuning on paper summarization
   - Benefit: Improved research-specific responses
   - Dataset: Instruction-response pairs from 2000 papers
   - Estimated effort: 25-35 hours

#### Future Scope Tier 2: Medium Priority

4. **Semantic Embeddings (Sentence-BERT/SciBERT)**
   - Replace TF-IDF with contextual embeddings
   - Expected improvement: +5-10% F1

5. **Research Gap Detection**
   - Identify underexplored topics
   - Cross-reference citations and keywords

6. **Paper Similarity Search**
   - Find related papers via embeddings
   - "More like this" functionality

7. **Journal Recommendation System**
   - Predict suitable journals for a given paper
   - Multi-label classification

#### Future Scope Tier 3: Long-Term

8. **Larger Dataset (10k+ papers)**
   - Expand beyond AI/ML
   - Multi-domain corpus

9. **Real-Time Data Pipeline**
   - Continuous OpenAlex ingestion
   - Incremental model updates

10. **Multi-Modal Features**
    - Extract features from figures/tables
    - CV on paper diagrams

11. **Causal Inference**
    - Move beyond correlation
    - Instrumental variables or causal DAGs

12. **Explainable AI (XAI)**
    - LIME/SHAP for model explanations
    - Per-prediction feature contributions

#### Format

For each item:
- Name
- Description (2-3 sentences)
- Expected benefit
- Estimated effort (if known)
- Status: ❌ NOT IMPLEMENTED (be explicit)

**Length:** 1000-1500 words

### Acceptance Criteria

- [ ] Conclusion summarizes DA3 achievements clearly
- [ ] Key findings included with metrics
- [ ] Contribution statement articulates value
- [ ] At least 8-10 limitations documented
- [ ] Each limitation includes impact and justification
- [ ] Future scope clearly marked as NOT IMPLEMENTED
- [ ] Future scope organized by priority
- [ ] Future scope items are realistic and scoped
- [ ] No feature claimed as implemented when it's future work
- [ ] Total 3000-4000 words

---

## PHASE 5: FINAL REPORT & CODE ORGANIZATION (CRITERION 5)

**Priority:** MEDIUM  
**Marks:** 2 out of 10  
**Status:** ⚠️ Partial  
**Estimated Effort:** 10-12 hours

### Objective

Create a comprehensive DA3 final report and ensure code is well-organized, documented, and reproducible.

### Task 5.1: DA3 Final Report (6-8 hours)

**File:** `DA3/DA3_Final_Report.md`

**Structure:**

```markdown
# ResearchPilot DA3 Final Report
## Human-Centered AI Research Assistant — Final Evaluation & Project Demonstration

### Table of Contents

1. Executive Summary
2. Project Context
   2.1. Recap: DA1 & DA2
   2.2. DA3 Objectives
3. Dataset & Methodology
   3.1. Data Source
   3.2. Feature Engineering (27→206→80)
   3.3. Database Integration
4. Machine Learning Models
   4.1. Twelve Algorithms Overview
   4.2. Training & Evaluation Methodology
   4.3. Held-Out Test Set
5. Final Performance Comparison ⭐ DA3 CRITERION 1
   5.1. Model Comparison Table
   5.2. Baseline vs Tuned Models
   5.3. Performance Metrics Breakdown
6. Statistical Analysis ⭐ DA3 CRITERION 1
   6.1. Research Question
   6.2. Methodology (Friedman Test)
   6.3. Results
   6.4. Post-Hoc Analysis
   6.5. Interpretation
7. Interactive Dashboard ⭐ DA3 CRITERION 2
   7.1. Technology Stack
   7.2. Dashboard Structure
   7.3. Tab 1: Overview
   7.4. Tab 2: Model Comparison
   7.5. Tab 3: Model Diagnostics
   7.6. Tab 4: Research Clusters
   7.7. Interactivity Features
   7.8. Usage Instructions
8. Results & Discussion ⭐ DA3 CRITERION 3
   8.1. Dataset Distribution
   8.2. Feature Engineering Impact
   8.3. Model Comparison Discussion
   8.4. Hyperparameter Tuning Discussion
   8.5. Statistical Analysis Interpretation
   8.6. Confusion Matrix Analysis
   8.7. ROC-AUC Interpretation
   8.8. Feature Importance Discussion
   8.9. Clustering Interpretation
   8.10. Overall Discussion
9. Conclusion ⭐ DA3 CRITERION 4
10. Limitations ⭐ DA3 CRITERION 4
11. Future Scope ⭐ DA3 CRITERION 4
12. References
13. Appendices
    A. Feature List
    B. Hyperparameter Grids
    C. Confusion Matrices
    D. Statistical Test Details
```

**Content Guidelines:**
- Sections 1-4: Concise recap (reuse DA2 content, ~2000 words)
- Sections 5-11: Detailed DA3 content (~8000-10000 words)
- Total length: 12000-15000 words
- Include figures inline with captions
- Include tables with clear formatting
- Use formal academic writing style
- Cite all data sources, packages, methods

**Figures to Include:**
- Final model comparison charts (3-4 figures)
- Statistical analysis figures (CD diagram, distributions)
- Dashboard screenshots (4-6 figures)
- Confusion matrices (best models)
- ROC curves
- Feature importance
- Clustering visualizations

**Tables to Include:**
- Model performance comparison (baseline & tuned)
- Statistical test results
- Feature engineering stages
- Cluster profiles
- Hyperparameter configurations

**Time:** 6-8 hours

### Task 5.2: Code Organization (2-3 hours)

**Current Structure:**
```
Tuned_Research/
├── r/scripts/phase2/         # DA2 scripts (KEEP AS-IS)
├── data/ml_r/                # DA2 data artifacts (KEEP AS-IS)
├── database/                 # SQLite database (KEEP AS-IS)
├── DA2/                      # DA2 documentation (KEEP AS-IS)
```

**NEW DA3 Structure to Create:**

```
Tuned_Research/
├── DA3/                                    # 🆕 NEW DA3 DELIVERABLES
│   ├── DA3_Final_Report.md                # Main DA3 report
│   ├── DA3_Statistical_Analysis.md        # Statistical analysis detail
│   ├── DA3_Results_Discussion.md          # Results interpretation
│   ├── DA3_Conclusion_Limitations_Future.md
│   ├── DA3_Project_Audit.md               # This audit document
│   ├── DA3_Implementation_Plan.md         # This plan
│   ├── DA3_Presentation.pdf               # Presentation slides
│   ├── DA3_Viva_QA.md                     # Viva preparation
│   ├── figures/                           # DA3-specific figures
│   │   ├── statistical_*.png
│   │   ├── dashboard_*.png
│   │   └── final_comparison_*.png
│   ├── tables/                            # DA3-specific tables
│   │   ├── model_comparison_final.csv
│   │   ├── statistical_test_results.csv
│   │   └── fold_performance_matrix.csv
│   └── README.md                          # DA3 overview
├── dashboard/                              # 🆕 INTERACTIVE DASHBOARD
│   ├── app.R                              # Main Shiny app
│   ├── global.R                           # Data loading
│   ├── utils/
│   │   └── plotting_utils.R
│   ├── www/
│   │   └── custom.css
│   └── README.md                          # Dashboard instructions
├── r/scripts/
│   ├── phase2/                            # DA2 scripts (NO CHANGES)
│   └── phase3/                            # 🆕 DA3 scripts
│       ├── 11_statistical_analysis.R     # Friedman test, post-hoc
│       ├── 12_final_comparison.R         # Consolidated comparison
│       └── 13_export_for_dashboard.R     # Data prep for Shiny
├── data/ml_r/                             # Existing (add new files)
│   ├── cv_fold_results.csv               # 🆕 Fold-level performance
│   ├── statistical_test_results.rds      # 🆕 Friedman results
│   └── dashboard_data.rds                # 🆕 Dashboard data bundle
├── DA2/                                   # DA2 documentation (NO CHANGES)
├── database/                              # SQLite (NO CHANGES)
├── README.md                              # 🔄 UPDATE with DA3 status
└── DA3_CHECKLIST.md                       # 🆕 Final verification checklist
```

**Actions:**
1. Create `DA3/` directory
2. Create `dashboard/` directory
3. Create `r/scripts/phase3/` directory
4. Move statistical analysis script to phase3/
5. Update main `README.md` with DA3 status
6. Create `DA3/README.md` as index
7. Create `dashboard/README.md` with run instructions

**Time:** 2-3 hours

### Task 5.3: Code Documentation (1 hour)

**Standards:**
- All R scripts have header comments:
  ```r
  # ResearchPilot DA3
  # Script: 11_statistical_analysis.R
  # Purpose: Perform Friedman test and post-hoc analysis
  # Input: data/ml_r/cv_fold_results.csv
  # Output: data/ml_r/statistical_test_results.rds
  # Author: [Your Name]
  # Date: 2026-10-XX
  ```
- Key functions have Roxygen-style comments
- Inline comments for non-obvious logic
- Set seed for reproducibility
- Print progress messages

**Time:** 1 hour

### Task 5.4: Reproducibility Documentation (30 min)

**File:** `DA3/DA3_REPRODUCIBILITY.md`

**Content:**
- How to run statistical analysis
- How to run dashboard locally
- Required R packages (with versions)
- Expected run times
- Troubleshooting common issues

**Time:** 30 min

### Acceptance Criteria

- [ ] DA3_Final_Report.md completed (12000-15000 words)
- [ ] All figures embedded with captions
- [ ] All tables formatted clearly
- [ ] DA3 directory structure created
- [ ] Code organized by phase (phase2 = DA2, phase3 = DA3)
- [ ] All new scripts have header comments
- [ ] README.md updated with DA3 status
- [ ] Reproducibility instructions provided
- [ ] No unnecessary file duplication
- [ ] Clear separation: DA2 (frozen) vs DA3 (new)

---

## PHASE 6: PRESENTATION & VIVA PREPARATION (CRITERION 6)

**Priority:** MEDIUM  
**Marks:** 1 out of 10  
**Status:** ❌ Not started  
**Estimated Effort:** 6-8 hours

### Objective

Create presentation slides and prepare for live demonstration and viva voce examination.

### Task 6.1: Presentation Slides (4-5 hours)

**File:** `DA3/DA3_Presentation.pdf` (or .pptx)

**Slide Structure (12 slides max):**

1. **Title Slide**
   - ResearchPilot: A Human-Centered AI Research Assistant
   - DA3 Final Evaluation & Project Demonstration
   - Your Name, Roll Number
   - October 16, 2026

2. **Problem Statement & Objectives**
   - Challenge: Structured exploration of research literature
   - Objective: ML pipeline + dashboard for OA category prediction
   - DA3 Focus: Statistical validation + interactive visualization

3. **Project Pipeline Overview**
   - Diagram: Phase 1 (data) → Phase 2 (ML) → DA3 (evaluation/dashboard)
   - 2000 papers, 12 models, statistical testing, interactive dashboard

4. **Dataset & Feature Engineering**
   - OpenAlex API → 2000 AI/ML papers
   - 27 raw → 206 engineered → 80 selected features
   - Database: SQLite (4 tables)
   - Visual: Feature engineering pipeline

5. **Machine Learning Models (12 Algorithms)**
   - List all 12 with brief description
   - Evaluation: proper train/val/test split
   - Target: 3-class OA category classification

6. **Final Performance Comparison ⭐**
   - Chart: Model vs Macro-F1 (all 15 models: 12 baseline + 3 tuned)
   - Highlight: Best baseline (LDA, 0.5696) vs Best tuned (SVM-RBF, 0.5701)
   - Table: Top 5 models with all metrics

7. **Statistical Analysis Results ⭐**
   - Friedman test results
   - Null hypothesis: All models equal performance
   - Result: p-value = X.XXX
   - Conclusion: [Reject/Fail to reject] H₀
   - Visual: Critical difference diagram (if post-hoc performed)

8. **Interactive Dashboard Demo ⭐**
   - Screenshot: Dashboard overview with 4 tabs
   - Key features: metric selector, model selector, interactive Plotly charts
   - Technology: R Shiny + Plotly
   - Live demo during presentation

9. **Key Findings & Interpretation**
   - LDA performed best: linear separability insight
   - Tuning gains modest: dataset size limitation
   - Feature importance: publisher features dominate
   - Confusion: partially_open is hardest class

10. **Research Clusters & Themes**
    - K-Means clustering (k=3)
    - Themes: Education/GenAI, General ML, Medical Imaging
    - Visual: Cluster size distribution + dominant terms

11. **Limitations & Future Scope**
    - Limitations: dataset size, moderate performance, TF-IDF features
    - Future: RAG, local LLM (Ollama), semantic embeddings, larger dataset
    - Clear distinction: current implementation vs future plans

12. **Conclusion & Thank You**
    - DA3 achievements: statistical validation, interactive dashboard, complete pipeline
    - Contribution: foundation for research assistant features
    - Thank you + Questions?

**Design Guidelines:**
- Clean, professional template
- Consistent colors (ResearchPilot branding)
- Minimal text per slide (bullet points, not paragraphs)
- High-quality figures (300 DPI)
- Readable fonts (18pt minimum body text)
- Animations: minimal (slide transitions only)

**Time:** 4-5 hours

### Task 6.2: Live Demo Script (1 hour)

**File:** `DA3/DA3_Live_Demo_Script.md`

**Structure:**

```markdown
# ResearchPilot DA3 Live Demo Script

## Setup (Before Presentation)
- [ ] Launch R Studio
- [ ] Open dashboard/ project
- [ ] Run app.R
- [ ] Verify app loads correctly
- [ ] Open browser to localhost:XXXX
- [ ] Test all tabs and interactions
- [ ] Prepare backup screenshots (if app fails)

## Demo Flow (5 minutes during presentation)

### Part 1: Overview (1 min)
1. Show Tab 1: Overview
2. Point out KPI cards (2000 papers, 80 features, 12 models)
3. Highlight OA category distribution chart
4. Mention: "This is all interactive — let me show you"

### Part 2: Model Comparison (2 min)
1. Navigate to Tab 2: Model Comparison
2. Current metric: Macro-F1
3. Point out: LDA at top (baseline), SVM-tuned second
4. **INTERACT:** Change metric dropdown to "Accuracy"
   - Chart updates dynamically
   - Point out: "Notice how rankings shift slightly"
5. **INTERACT:** Hover over a bar
   - Tooltip shows exact values
   - Demonstrate: "All charts are interactive — hover, zoom, pan"
6. Return to Macro-F1

### Part 3: Model Diagnostics (1.5 min)
1. Navigate to Tab 3: Model Diagnostics
2. Current model: LDA (or SVM-tuned)
3. Show confusion matrix heatmap
   - Point out: partially_open is hardest class
4. Show ROC curves
   - Point out: closed has highest AUC
5. **INTERACT:** Change model selector to "Random Forest"
   - Charts update
   - Feature importance appears (tree-based model)
   - Point out top features: pub_ieee, title_to_abstract_ratio

### Part 4: Clusters (0.5 min)
1. Navigate to Tab 4: Research Clusters
2. Show cluster size distribution (pie chart)
3. Point out research themes (Education/GenAI, Medical Imaging, etc.)

### Closing (0.5 min)
- "This dashboard allows exploratory data analysis of all ResearchPilot results"
- "Built with R Shiny and Plotly for full interactivity"
- "All data loaded directly from project database and result files"
- Return to presentation slides

## Backup Plan
If app fails to load:
- Use pre-recorded screen capture video (prepare in advance)
- OR: Use static screenshots with explanation
- Acknowledge: "Live demo unavailable, but here are screenshots showing..."

## Common Demo Pitfalls to Avoid
- Don't scroll too fast
- Don't change too many things at once (audience gets lost)
- Don't spend time troubleshooting live (use backup)
- Don't apologize for UI imperfections (just demo functionality)
```

**Time:** 1 hour

### Task 6.3: Viva Q&A Preparation (2 hours)

**File:** `DA3/DA3_Viva_QA.md`

**Format:** Question → Answer structure

**Categories:**

#### A. Dataset & Preprocessing (5-7 questions)
Example:
- Q: Why only 2000 papers?
- A: Balanced project scope with computational constraints. OpenAlex API rate limits + sufficient size for B.Tech project. Larger dataset is future scope.

- Q: How did you handle missing values?
- A: [Refer to DA2 preprocessing methodology]

#### B. Feature Engineering (8-10 questions)
- Q: Why 206 features? Isn't that high-dimensional?
- A: Started with 27 raw, engineered 4 groups (structural, domain, publisher, TF-IDF). Feature selection reduced to 80 to avoid curse of dimensionality while retaining predictive power.

- Q: How did you avoid data leakage?
- A: All feature engineering steps (scaling, NZV, correlation filter, RF importance) fitted on training split only. Test set untouched until final evaluation.

- Q: Why use TF-IDF instead of word embeddings?
- A: TF-IDF is interpretable, computationally efficient, and sufficient for metadata-based classification. Semantic embeddings are future scope.

#### C. Machine Learning Models (10-12 questions)
- Q: Why did LDA perform best?
- A: LDA captures linear separability in feature space. Our engineered features likely exhibit linear relationships with OA categories. Also, LDA's dimensionality reduction helps with high-dimensional TF-IDF features.

- Q: Why did XGBoost perform poorly?
- A: Possible overfitting despite regularization. Dataset size (1400 training samples) may be insufficient for XGBoost's complexity. TF-IDF sparsity could also be a factor. Tuning improved it slightly (+0.0109 F1) but remained weak.

- Q: Why is 57% accuracy acceptable?
- A: Three-class classification with inherent overlap (especially partially_open). Metadata-only features (no full text). Baseline (random guessing) would be 33%, so 57% is meaningful improvement. Also, ROC-AUC of 0.76 shows good discrimination ability.

- Q: What's the difference between validation and test set?
- A: Validation (15%) used during hyperparameter tuning to select best params. Test (15%) held out completely until final evaluation. This prevents overfitting to validation set.

#### D. Statistical Analysis (6-8 questions)
- Q: Why Friedman test?
- A: Non-parametric test for comparing multiple models across repeated measures (folds). Doesn't assume normality. Appropriate for CV performance data.

- Q: What if p-value > 0.05?
- A: Cannot reject null hypothesis. Doesn't mean models are identical, just that we lack evidence of significant difference. Could be due to sample size (folds), or genuinely similar performance. Practical differences may still exist.

- Q: What's the difference between statistical and practical significance?
- A: Statistical significance: p < 0.05 means unlikely due to chance. Practical significance: is the difference large enough to matter? SVM-tuned vs LDA: +0.0005 F1 is statistically insignificant AND practically negligible.

- Q: What's a post-hoc test?
- A: After Friedman test rejects H₀, post-hoc tests (e.g., Nemenyi) identify which specific model pairs differ significantly. Controls for multiple comparisons.

#### E. Dashboard (5-7 questions)
- Q: Why R Shiny instead of Python Dash?
- A: Project primarily in R (DA2 pipeline). Shiny integrates seamlessly with existing R objects. Plotly available in both. Personal preference: R ecosystem.

- Q: How is interactivity achieved?
- A: Shiny's reactive programming: user input (metric selector) triggers reactive expression, updates plot automatically. Plotly provides hover, zoom, pan out-of-the-box.

- Q: Can this dashboard be deployed?
- A: Yes, can deploy to shinyapps.io (free tier) or Shiny Server. For this project, local deployment demonstrated. Production deployment is future scope.

#### F. Results Interpretation (8-10 questions)
- Q: What does feature importance tell us?
- A: Which features the model uses most for decisions. NOT causation. Publisher features (pub_ieee, pub_mdpi) dominate, suggesting OA policy is publisher-driven. Text length features also important.

- Q: Why is partially_open the hardest class?
- A: Overlaps with both fully_open and closed. Definition varies by publisher. Some "partially_open" papers have embargoes, some have paywalled versions — metadata may not capture this nuance.

- Q: What do the clusters represent?
- A: Research themes/topics, NOT OA categories. Unsupervised clustering on TF-IDF identified 3 subfields: Education/GenAI, General ML, Medical Imaging. Labels are human-interpreted from dominant terms.

- Q: What's the relationship between clusters and OA status?
- A: Minimal direct relationship. Clustering is topic-based, OA status is publisher/policy-based. Some fields (medical) may have different OA patterns, but not strongly separated in our data.

#### G. Limitations & Future Scope (5-7 questions)
- Q: What's the biggest limitation?
- A: Moderate classification performance (~57% accuracy). Root cause: partially_open ambiguity + metadata-only features. Would need full-text analysis or better OA status definitions to improve significantly.

- Q: What would improve performance most?
- A: (1) Semantic embeddings (SciBERT) instead of TF-IDF, (2) Larger dataset (10k+), (3) Full-text features, (4) Better target variable definition (collapse partially_open into binary?)

- Q: Is RAG implemented?
- A: No. RAG (Retrieval-Augmented Generation) is future scope. Would involve vector database + LLM for Q&A over paper corpus.

- Q: Is the local LLM (Ollama) implemented?
- A: No. Ollama integration is future scope, planned AFTER DA3. Would enable privacy-preserving local inference for research assistant features.

#### H. Technical Details (5-7 questions)
- Q: How did you ensure reproducibility?
- A: set.seed() in all R scripts. Saved train/val/test split indices. Documented package versions. Saved model objects. Provided run instructions.

- Q: What database did you use?
- A: SQLite (researchpilot_r.db). 4 tables: papers, ml_features, oa_features, model_results. Queried via RSQLite package in R.

- Q: How long does the pipeline take to run?
- A: Full pipeline (feature engineering → training → tuning → evaluation): ~30-45 minutes on standard laptop. Dashboard launches in <10 seconds.

**Total:** 50-60 Q&A pairs

**Time:** 2 hours

### Acceptance Criteria

- [ ] Presentation has exactly 12 slides
- [ ] All DA3 criteria covered in slides
- [ ] Slides are visually clean and professional
- [ ] Live demo script is detailed and timed
- [ ] Backup plan prepared (screenshots or video)
- [ ] Viva Q&A has 50+ questions with complete answers
- [ ] All answers are accurate and grounded in actual implementation
- [ ] No feature claimed as implemented when it's not

---

## TIMELINE & MILESTONES

**Total Available Time:** 10 days (Oct 6-16, 2026)

### Week 1: Oct 6-10 (Core Implementation)

**Day 1 (Oct 6): Audit & Planning** ✅ DONE
- [x] Project audit completed
- [x] Implementation plan created
- [x] Priority queue established

**Day 2 (Oct 7): Statistical Analysis START**
- [ ] Extract/verify CV fold data
- [ ] Implement Friedman test
- [ ] Create statistical visualizations (partial)
- **Deliverable:** `r/scripts/phase3/11_statistical_analysis.R` (50% complete)

**Day 3 (Oct 8): Statistical Analysis FINISH**
- [ ] Post-hoc analysis (if significant)
- [ ] Complete statistical visualizations
- [ ] Write statistical analysis document
- **Deliverable:** Statistical analysis complete and documented

**Day 4 (Oct 9): Dashboard START**
- [ ] Set up Shiny project structure
- [ ] Implement Tab 1: Overview (KPI cards, OA distribution)
- [ ] Implement Tab 2: Model Comparison (50%)
- **Deliverable:** Dashboard skeleton running

**Day 5 (Oct 10): Dashboard CONTINUE**
- [ ] Complete Tab 2: Model Comparison
- [ ] Implement Tab 3: Model Diagnostics (70%)
- [ ] Test interactivity and reactivity
- **Deliverable:** Dashboard 60% complete

### Week 2: Oct 11-16 (Documentation & Polish)

**Day 6 (Oct 11): Dashboard FINISH**
- [ ] Complete Tab 3: Model Diagnostics
- [ ] Implement Tab 4: Research Clusters
- [ ] Polish UI and test all interactions
- [ ] Create dashboard README
- **Deliverable:** Dashboard 100% complete

**Day 7 (Oct 12): Results Discussion**
- [ ] Write all 10 sections of results discussion
- [ ] Integrate with statistical analysis findings
- [ ] Review for accuracy and clarity
- **Deliverable:** DA3_Results_Discussion.md complete

**Day 8 (Oct 13): Conclusion & Report**
- [ ] Write conclusion, limitations, future scope
- [ ] Compile DA3 final report (integrate all sections)
- [ ] Add all figures and tables to report
- [ ] Proofread and format
- **Deliverable:** DA3_Final_Report.md complete

**Day 9 (Oct 14): Presentation & Viva Prep**
- [ ] Create presentation slides (12 slides)
- [ ] Write live demo script
- [ ] Prepare viva Q&A (50+ questions)
- [ ] Record backup demo video (optional)
- **Deliverable:** Presentation and viva materials complete

**Day 10 (Oct 15): Final Review & Testing**
- [ ] Test dashboard end-to-end
- [ ] Verify all statistical results reproducible
- [ ] Review all documents for errors
- [ ] Practice presentation and demo (3-4 run-throughs)
- [ ] Finalize submission package
- **Deliverable:** Ready for submission

**Oct 16: SUBMISSION DAY** 🎯
- [ ] Submit DA3 deliverables
- [ ] Prepare for live demonstration
- [ ] Prepare for viva voce

---

## RISK MITIGATION

### Risk 1: Statistical Analysis Data Unavailable
**Mitigation:**
- Day 2 morning: Verify fold data immediately
- If missing: Recompute 5-fold CV for top 8 models (4-hour task)
- Adjust timeline: compress dashboard Day 4 if needed

### Risk 2: Dashboard Performance Issues
**Mitigation:**
- Use reactive programming best practices
- Cache data loading with `reactive()`
- Test with full dataset early (Day 4)
- If slow: optimize queries, reduce data points, add loading spinners

### Risk 3: Time Overrun
**Mitigation:**
- Daily progress check against timeline
- If behind: prioritize must-haves, defer nice-to-haves
- Focus on functional completeness over polish
- Presentation/viva can be compressed if needed (still deliver core content)

### Risk 4: Software/Package Issues
**Mitigation:**
- Day 2: Install all required packages early
- Document versions: `sessionInfo()`
- Keep fallback: static plots if Plotly fails

---

## SUCCESS CRITERIA

### Minimum Viable DA3 (Required for 7/10 marks)
- [ ] Statistical analysis complete with p-value
- [ ] Dashboard functional (all 4 tabs working)
- [ ] Results discussion document exists
- [ ] Conclusion, limitations, future scope documented
- [ ] Basic presentation slides created
- [ ] Core viva Q&A prepared

### Complete DA3 (Required for 10/10 marks)
- [ ] All minimum viable items above
- [ ] Statistical analysis with post-hoc (if applicable)
- [ ] Dashboard polished with excellent UX
- [ ] Comprehensive results discussion (8000+ words)
- [ ] DA3 final report complete (12000+ words)
- [ ] Professional presentation (12 slides)
- [ ] Live demo practiced and smooth
- [ ] Comprehensive viva preparation (50+ Q&A)
- [ ] Code well-organized and documented
- [ ] All deliverables reproducible

---

## POST-DA3: OLLAMA/LLM LAYER

**START DATE:** October 17, 2026 (DAY AFTER DA3 SUBMISSION)

### Phase A: Hardware & Model Selection (Oct 17)
- [ ] Inspect system: GPU, VRAM, RAM, disk space
- [ ] Research compatible models for hardware
- [ ] Select model (Qwen/Llama/Gemma)
- [ ] Install Ollama
- [ ] Test model inference

### Phase B: Instruction Dataset Preparation (Oct 18-19)
- [ ] Design instruction-response schema
- [ ] Generate paper summarization examples (100-200)
- [ ] Generate Q&A examples (100-200)
- [ ] Generate research question examples (50-100)
- [ ] Validate dataset quality

### Phase C: Base Model Evaluation (Oct 20)
- [ ] Test base model on sample tasks
- [ ] Measure: factual accuracy, hallucination rate, response structure
- [ ] Document baseline performance

### Phase D: Fine-Tuning Decision (Oct 21)
- [ ] Evaluate: is fine-tuning beneficial for this dataset size?
- [ ] Consider: LoRA vs full fine-tuning vs prompt engineering only
- [ ] Decision: proceed with fine-tuning OR use base model + good prompts

### Phase E: Fine-Tuning (If Decided) (Oct 22-24)
- [ ] Set up fine-tuning environment
- [ ] Implement LoRA/QLoRA
- [ ] Train on instruction dataset
- [ ] Validate on hold-out set
- [ ] Save adapters

### Phase F: RAG Layer (Optional) (Oct 25-26)
- [ ] Set up vector database (Chroma/FAISS)
- [ ] Generate embeddings for papers
- [ ] Implement retrieval logic
- [ ] Test retrieval quality

### Phase G: Integration (Oct 27-28)
- [ ] Integrate LLM with ResearchPilot
- [ ] Create API/interface layer
- [ ] Test end-to-end workflows
- [ ] Handle errors and edge cases

### Phase H: Evaluation (Oct 29)
- [ ] Base vs fine-tuned comparison
- [ ] Task-specific evaluation (summarization, Q&A)
- [ ] User testing (if applicable)

### Phase I: Documentation (Oct 30)
- [ ] Document Ollama architecture
- [ ] Document fine-tuning process
- [ ] Document API usage
- [ ] Update README

**TOTAL ESTIMATED TIME:** 15-20 hours spread over 2 weeks

---

## FINAL CHECKLIST (Before Submission)

### Code
- [ ] All R scripts run without errors
- [ ] Statistical analysis script produces correct output
- [ ] Dashboard launches successfully
- [ ] No hardcoded paths (use relative paths)
- [ ] All scripts have header comments
- [ ] set.seed() used for reproducibility

### Data
- [ ] All required CSV/RDS files present
- [ ] Fold-level CV data available
- [ ] Statistical results saved
- [ ] Dashboard data exported

### Documentation
- [ ] DA3_Final_Report.md complete (12000+ words)
- [ ] DA3_Statistical_Analysis.md complete
- [ ] DA3_Results_Discussion.md complete
- [ ] DA3_Conclusion_Limitations_Future.md complete
- [ ] All figures have captions
- [ ] All tables have headers
- [ ] References cited

### Presentation
- [ ] 12 slides created
- [ ] All DA3 criteria covered
- [ ] Figures are high quality
- [ ] Live demo script prepared
- [ ] Backup screenshots/video ready

### Viva
- [ ] 50+ Q&A prepared
- [ ] All answers accurate
- [ ] No feature overclaimed
- [ ] Limitations acknowledged

### Organization
- [ ] DA3/ directory created
- [ ] dashboard/ directory created
- [ ] r/scripts/phase3/ created
- [ ] README.md updated
- [ ] No duplicate files

### Verification
- [ ] All statistical results reproduce
- [ ] Dashboard works on fresh R session
- [ ] All links in documents are valid
- [ ] Spelling and grammar checked
- [ ] No placeholder text ("[TODO]") remaining

---

## NOTES FOR IMPLEMENTER

1. **DO NOT SKIP STATISTICAL ANALYSIS** — It's explicitly required and worth 1/10 marks. Even if simple, it must be done.

2. **DASHBOARD MUST BE INTERACTIVE** — A collection of static plots is NOT a dashboard. Use Shiny reactive programming and Plotly.

3. **GROUND ALL CLAIMS IN DATA** — Every interpretation must be supported by actual results. No fabrication.

4. **SEPARATE IMPLEMENTED FROM FUTURE** — Be explicit: RAG, Ollama, fine-tuning are NOT implemented in DA3. Future scope only.

5. **TEST EARLY AND OFTEN** — Don't wait until Day 10 to test the dashboard. Test on Day 4-5.

6. **PRIORITIZE RUTHLESSLY** — If time is short, focus on: statistical analysis, functional dashboard, results discussion. Defer: report polish, fancy visualizations, nice-to-haves.

7. **PRACTICE PRESENTATION** — A smooth 10-minute presentation with live demo is worth more than a 30-slide deck.

8. **PREPARE FOR TECHNICAL FAILURES** — Dashboard might not load during demo. Have backup screenshots.

9. **REPRODUCIBILITY MATTERS** — Evaluator should be able to run your code and get same results.

10. **HONEST LIMITATIONS** — Acknowledging limitations shows maturity. Don't oversell results.

---

**IMPLEMENTATION STARTS NOW**

Next immediate action: Begin Task 1.1 (Statistical Analysis Data Preparation)

