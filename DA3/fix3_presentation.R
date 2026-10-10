# =============================================================================
# DA3 Fix 3 — Create DA3_Presentation.pptx using officer package
# =============================================================================
setwd("e:/Tuned_Research")

# Install officer if needed
if (!requireNamespace("officer", quietly = TRUE)) {
  cat("Installing officer...\n")
  install.packages("officer", repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(library(officer))

cat("officer version:", as.character(packageVersion("officer")), "\n\n")

out_pptx <- "DA3/DA3_Presentation.pptx"

# ── Helper functions ──────────────────────────────────────────────────────────
add_slide_title_content <- function(prs, title, content_lines) {
  prs <- add_slide(prs, layout = "Title and Content", master = "Office Theme")
  prs <- ph_with(prs, value = title,
                 location = ph_location_type(type = "title"))
  body_text <- paste(content_lines, collapse = "\n")
  prs <- ph_with(prs, value = body_text,
                 location = ph_location_type(type = "body"))
  prs
}

add_title_slide <- function(prs, title, subtitle) {
  prs <- add_slide(prs, layout = "Title Slide", master = "Office Theme")
  prs <- ph_with(prs, value = title,
                 location = ph_location_type(type = "ctrTitle"))
  prs <- ph_with(prs, value = subtitle,
                 location = ph_location_type(type = "subTitle"))
  prs
}

# ── Create presentation ───────────────────────────────────────────────────────
prs <- read_pptx()   # blank presentation

# ── SLIDE 1 — Title ───────────────────────────────────────────────────────────
prs <- add_title_slide(prs,
  title    = "ResearchPilot\nA Human-Centered AI Research Assistant",
  subtitle = paste0(
    "DA3 — Final Evaluation & Project Demonstration\n",
    "October 16, 2026\n\n",
    "Dataset: OpenAlex API  |  2,000 papers  |  12 ML models\n",
    "Best model: SVM-RBF Tuned (Macro-F1 = 0.5701)\n",
    "Friedman p = 0.000009  |  R Shiny + Plotly Dashboard"
  )
)

# ── SLIDE 2 — Problem and Objectives ─────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Problem Statement and DA3 Objectives",
  content_lines = c(
    "PROBLEM",
    "Can we automatically predict whether an AI/ML research paper is freely",
    "accessible — using only its metadata (title, abstract, publisher, citations)?",
    "",
    "TARGET VARIABLE: OA Category (3 classes)",
    "  fully_open      834 papers  (41.7%)",
    "  partially_open  723 papers  (36.2%)",
    "  closed          443 papers  (22.2%)",
    "",
    "DA3 OBJECTIVES (Official Rubric — 10 Marks)",
    "  1. Final performance comparison + statistical analysis  [2 marks]",
    "  2. Interactive R Shiny + Plotly dashboard               [2 marks]",
    "  3. Result interpretation and discussion                 [2 marks]",
    "  4. Conclusion, limitations, future scope                [1 mark]",
    "  5. Final report quality + code organization             [2 marks]",
    "  6. Presentation + viva                                  [1 mark]"
  )
)

# ── SLIDE 3 — Pipeline ────────────────────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "ResearchPilot — Complete Project Pipeline",
  content_lines = c(
    "DA1 — DATA FOUNDATION",
    "  OpenAlex API  ->  2,000 AI/ML papers  ->  EDA  ->  final_dataset.csv",
    "",
    "DA2 — INTELLIGENCE LAYER",
    "  Feature Engineering:  27 raw  ->  206 engineered  ->  80 selected",
    "  SQLite Database:  4 tables  (papers, ml_features, oa_features, model_results)",
    "  12 ML Models trained + 3 hyperparameter-tuned",
    "  Impact-tier classification  +  K-Means topic clustering (k=3)",
    "",
    "DA3 — FINAL EVALUATION  [TODAY]",
    "  Friedman non-parametric test  (chi-sq=43.46, p=0.000009)",
    "  Final model comparison  (15 models: 12 baseline + 3 tuned)",
    "  R Shiny + Plotly interactive dashboard  (4 tabs)",
    "  Complete report + viva preparation",
    "",
    "FUTURE SCOPE  [NOT IMPLEMENTED]",
    "  RAG research assistant  |  Ollama local LLM  |  Semantic embeddings"
  )
)

# ── SLIDE 4 — Dataset and Features ───────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Dataset and Feature Engineering",
  content_lines = c(
    "DATASET",
    "  Source:   OpenAlex API  (open scholarly database)",
    "  Size:     2,000 AI/ML research papers  x  27 raw columns",
    "  Target:   oa_category  (3 classes)",
    "  Splits:   70% train (1400)  |  15% val (299)  |  15% test (301)",
    "  Seed:     set.seed(42)  —  fully reproducible",
    "",
    "FEATURE ENGINEERING PIPELINE",
    "  27 raw  ->  206 engineered features:",
    "    Structural/text-length  (12):  abstract_length, text_richness, ...",
    "    Domain binary flags     (25):  dom_medical, dom_llm, dom_education, ...",
    "    Publisher one-hot       (18):  pub_ieee, pub_mdpi, pub_springer, ...",
    "    TF-IDF terms           (~150):  tf_model, tf_deep, tf_learning, ...",
    "",
    "  206  ->  80 selected (all steps fitted on TRAINING data only):",
    "    Step 1: Near-Zero Variance filter",
    "    Step 2: Pearson correlation filter  (TF-IDF group, |r| > 0.95)",
    "    Step 3: Random Forest Gini importance  ->  top 80 retained",
    "",
    "  KEY: No leakage — test set untouched until final evaluation"
  )
)

# ── SLIDE 5 — 12 ML Models ────────────────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Machine Learning Models — 12 Algorithms",
  content_lines = c(
    "LINEAR / DISCRIMINANT         TREE-BASED / ENSEMBLE",
    "  LDA   (best baseline)         Decision Tree",
    "  Logistic Regression           Random Forest",
    "  Elastic Net                   Gradient Boosting",
    "  SVM-Linear                    AdaBoost",
    "  SVM-RBF                       XGBoost  (worst)",
    "",
    "PROBABILISTIC / NEURAL",
    "  Naive Bayes                   MLP  (neural network, 50 units)",
    "",
    "EVALUATION",
    "  Same stratified test set for ALL models  (n=301, no leakage)",
    "  Primary metric:  Macro-F1  (equal weight to all 3 classes)",
    "  Also reported:   Accuracy, Precision, Recall, ROC-AUC",
    "",
    "HYPERPARAMETER TUNING  (5-fold CV grid search on training only)",
    "  Random Forest   ntree=500, mtry=8    ->  F1: 0.5317  ->  0.5475  (+0.0158)",
    "  SVM-RBF         cost=10, gamma=0.001 ->  F1: 0.5559  ->  0.5701  (+0.0142)",
    "  XGBoost         depth=6, eta=0.10    ->  F1: 0.3428  ->  0.3537  (+0.0109)"
  )
)

# ── SLIDE 6 — Final Performance Comparison ───────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Final Performance Comparison  (Test Set, n=301)",
  content_lines = c(
    "MODEL                     ACCURACY  MACRO-F1  ROC-AUC  TYPE",
    "SVM-RBF (Tuned)  [BEST]    57.48%   0.5701      —      Tuned",
    "LDA              [TOP BASE] 57.48%   0.5696    0.7577   Baseline",
    "MLP                         56.81%   0.5657    0.7659   Baseline",
    "Logistic Regression         56.48%   0.5631    0.7565   Baseline",
    "Elastic Net                 55.81%   0.5539    0.7584   Baseline",
    "RF (Tuned)                  56.15%   0.5475      —      Tuned",
    "SVM-Linear                  55.15%   0.5441    0.7587   Baseline",
    "Gradient Boosting           53.82%   0.5384    0.7402   Baseline",
    "Random Forest               54.82%   0.5317    0.7572   Baseline",
    "AdaBoost                    51.50%   0.5123    0.6963   Baseline",
    "Decision Tree               50.83%   0.5017    0.6924   Baseline",
    "Naive Bayes                 41.86%   0.4081    0.6840   Baseline",
    "XGBoost (Tuned)             35.55%   0.3537      —      Tuned",
    "XGBoost         [WORST]     34.55%   0.3428    0.5243   Baseline",
    "",
    "Tuning margin over best baseline: +0.0005 F1  (practically negligible)",
    "See: DA3/figures/final_f1_comparison.png"
  )
)

# ── SLIDE 7 — Statistical Analysis ───────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Statistical Analysis — Friedman Test + Nemenyi Post-Hoc",
  content_lines = c(
    "QUESTION",
    "  Are performance differences real, or random fold variation?",
    "",
    "METHOD: Friedman non-parametric test",
    "  Why?  Repeated measures (same 5 folds for all models)",
    "        Non-parametric (cannot assume normality at K=5)",
    "",
    "HYPOTHESES",
    "  H0: All 12 models have equal median Macro-F1  (differences = chance)",
    "  H1: At least one model differs significantly",
    "  Significance level: alpha = 0.05",
    "",
    "RESULTS",
    "  Chi-squared(11) = 43.4615",
    "  p-value         = 0.000009   ->   REJECT H0",
    "  Conclusion: Differences are STATISTICALLY REAL",
    "",
    "POST-HOC: Nemenyi test  ->  7 significant pairs",
    "  XGBoost significantly worse than: RF, LDA, Elastic Net, Logistic Reg.",
    "  Naive Bayes significantly worse than: RF, LDA, Elastic Net",
    "  CD diagram: DA3/figures/stat_cd_diagram.png",
    "",
    "Top models (LDA, RF, Elastic Net, LR) do NOT significantly differ"
  )
)

# ── SLIDE 8 — Interactive Dashboard ──────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Interactive Dashboard — R Shiny + Plotly",
  content_lines = c(
    "Launch:  setwd('e:/Tuned_Research')  then  shiny::runApp('dashboard')",
    "OR:      Double-click  LAUNCH_DASHBOARD.bat",
    "",
    "TAB 1 — OVERVIEW",
    "  KPI cards: 2000 papers, 27->206->80 features, 12 models, 301 test",
    "  Interactive Plotly bar chart: OA category distribution",
    "",
    "TAB 2 — MODEL COMPARISON  [LIVE DEMO]",
    "  Metric selector dropdown:  Macro-F1 / Accuracy / ROC-AUC / Precision / Recall",
    "  Chart updates instantly when metric is changed",
    "  Baseline (blue) vs Tuned (red)  |  hover for exact values",
    "",
    "TAB 3 — MODEL DIAGNOSTICS  [LIVE DEMO]",
    "  Model selector: LDA or Random Forest",
    "  Confusion matrix heatmap  (ACTUAL test-set predictions)",
    "  ROC curves  (ACTUAL from LDA posterior / RF probabilities)",
    "  Feature importance top-20  (from saved rf_importance_v2.rds)",
    "",
    "TAB 4 — RESEARCH CLUSTERS",
    "  Cluster size pie chart  |  Dominant TF-IDF terms per cluster",
    "",
    "All data loaded from project files — no fabricated values"
  )
)

# ── SLIDE 9 — Key Findings ────────────────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Key Findings and Interpretation",
  content_lines = c(
    "1. PUBLISHER FEATURES DOMINATE OA PREDICTION",
    "   pub_ieee (43.78), pub_mdpi (38.48) are top RF importance features",
    "   OA status = publisher POLICY decision, not paper content",
    "   IMPORTANT: Importance = predictive correlation, NOT causation",
    "",
    "2. LINEAR MODELS OUTPERFORM COMPLEX ENSEMBLES",
    "   LDA > MLP > Logistic Reg > SVM-RBF > Elastic Net > ... > XGBoost",
    "   Feature space is approximately linearly separable",
    "   ~1,400 training samples insufficient for XGBoost's complexity",
    "",
    "3. PARTIALLY_OPEN IS THE HARDEST CLASS",
    "   Lowest class ROC-AUC: partially_open = 0.6649 (LDA)",
    "   vs closed = 0.8379  |  Definitional ambiguity -> label noise",
    "",
    "4. TUNING GIVES DIMINISHING RETURNS",
    "   Best individual gain: RF +0.0158 F1",
    "   Overall best-vs-best margin: +0.0005 F1  (LDA already near-optimal)",
    "",
    "5. STATISTICALLY VALIDATED",
    "   XGBoost and Naive Bayes provably inferior (Nemenyi p < 0.05)",
    "   Top-tier models are statistically interchangeable at K=5 folds"
  )
)

# ── SLIDE 10 — Clustering ─────────────────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Topic Clustering — K-Means (k=3) on TF-IDF",
  content_lines = c(
    "METHOD",
    "  K-Means on L2-normalised TF-IDF (200 terms)",
    "  k=3 chosen by silhouette analysis (k=3 to 10)",
    "  Silhouette ~0.06  (expected weak separation for overlapping AI/ML topics)",
    "  Fully independent of supervised ML pipeline",
    "",
    "CLUSTER 1  (200 papers, 10.0%)",
    "  Theme:  Education / GenAI / ChatGPT",
    "  Terms:  educ, chatgpt, student, generat, tool, artifici",
    "",
    "CLUSTER 2  (1422 papers, 71.1%)",
    "  Theme:  General ML / Prediction / Robotics",
    "  Terms:  model, learn, data, predict, system, robot, review",
    "",
    "CLUSTER 3  (378 papers, 18.9%)",
    "  Theme:  Medical Imaging / Deep Learning",
    "  Terms:  imag, detect, segment, medic, network, diseas, convolut",
    "",
    "IMPORTANT: Labels are HUMAN-INTERPRETED from top TF-IDF terms",
    "K-Means produces numbers 1/2/3 — NOT semantic labels",
    "See dashboard Tab 4 for interactive exploration"
  )
)

# ── SLIDE 11 — Limitations and Future Scope ───────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Limitations and Future Scope",
  content_lines = c(
    "GENUINE LIMITATIONS (10 verified)",
    "  L1. Dataset: 2,000 papers, AI/ML only, static 2024-25 snapshot",
    "  L2. Performance: ~57% accuracy — 3-class overlap limits ceiling",
    "  L3. partially_open: ambiguous definition -> inherent label noise",
    "  L4. Features: TF-IDF only, no semantic embeddings",
    "  L5. No full-text: titles + abstracts only",
    "  L6. Metadata dependency: missing DOIs degrade predictions",
    "  L7. Static dataset: no continuous update pipeline",
    "  L8. Cluster labels: human-interpreted, not algorithmic",
    "  L9. CV folds K=5: limited statistical power for small gaps",
    " L10. No full-text: methodology details in paper body unused",
    "",
    "FUTURE SCOPE  [ALL MARKED: NOT IMPLEMENTED]",
    "  Tier 1 (High priority):",
    "    RAG-based research assistant       [NOT IMPLEMENTED]",
    "    Local LLM via Ollama               [NOT IMPLEMENTED]",
    "    LoRA/QLoRA instruction fine-tuning [NOT IMPLEMENTED]",
    "  Tier 2:",
    "    SciBERT/Sentence-BERT embeddings   [NOT IMPLEMENTED]",
    "    Paper similarity search            [NOT IMPLEMENTED]",
    "    Research gap detection             [NOT IMPLEMENTED]"
  )
)

# ── SLIDE 12 — Conclusion ─────────────────────────────────────────────────────
prs <- add_slide_title_content(prs,
  title = "Conclusion",
  content_lines = c(
    "DA3 ACHIEVEMENTS",
    "  12 ML models trained + 3 tuned + all evaluated on proper test set (n=301)",
    "  Best model: SVM-RBF Tuned  (Macro-F1=0.5701, Accuracy=57.48%)",
    "  Statistical validation: chi-sq=43.46, p=0.000009 -> REJECT H0",
    "  XGBoost and Naive Bayes provably inferior (Nemenyi, p<0.05)",
    "  R Shiny + Plotly dashboard: 4 interactive tabs, actual data throughout",
    "  3 research themes identified via K-Means clustering",
    "  Complete documentation: report, viva prep, presentation",
    "",
    "KEY INSIGHT",
    "  OA status is a PUBLISHER POLICY decision, not a paper content property",
    "  57.5% accuracy = 73% lift over random baseline (33%)",
    "  Top models are statistically interchangeable — LDA is recommended",
    "  (simple, interpretable, near-identical performance to SVM-RBF tuned)",
    "",
    "FOUNDATION LAID",
    "  Clean reproducible pipeline ready for Phase 3 extensions",
    "  RAG + Ollama LLM integration planned POST-DA3 submission",
    "",
    "Thank you  —  Questions?"
  )
)

# ── Save ──────────────────────────────────────────────────────────────────────
print(prs, target = out_pptx)
sz <- file.info(out_pptx)$size
cat(sprintf("\nSaved: %s  (%d KB)\n", out_pptx, as.integer(sz/1024)))
cat(sprintf("Slides: %d\n", length(prs$slide$get_metadata())))
cat("Fix 3 COMPLETE\n")
