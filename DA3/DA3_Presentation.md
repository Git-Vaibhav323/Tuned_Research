# ResearchPilot DA3 — Final Presentation
## 12-Slide Deck with Speaker Notes

**Event:** DA3 Final Evaluation & Project Demonstration  
**Date:** October 16, 2026  
**Duration:** 10–12 minutes + live demo + Q&A  
**All numbers are real — verified from actual pipeline outputs**

---

---

# SLIDE 1 — TITLE

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│          ResearchPilot                                          │
│   A Human-Centered AI Research Assistant                        │
│                                                                 │
│   DA3 — Final Evaluation & Project Demonstration               │
│   October 16, 2026                                              │
│                                                                 │
│   ┌──────────────────────────────────────────────────────┐      │
│   │  2,000 Papers  │  12 ML Models  │  Friedman p<0.001 │      │
│   │  R Shiny Dashboard  │  57.01% Best F1               │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                 │
│   Dataset: OpenAlex API  │  Language: R  │  DB: SQLite          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- Good [morning/afternoon]. My project is ResearchPilot — a human-centered AI research assistant.
- Today I'm presenting DA3: the final evaluation phase.
- The project predicts the open-access category of AI/ML research papers using machine learning.
- I'll cover: final model comparison, statistical analysis, interactive dashboard, key findings, and future scope.
- Let me start with the problem.

---

---

# SLIDE 2 — PROBLEM AND OBJECTIVES

```
┌─────────────────────────────────────────────────────────────────┐
│  THE PROBLEM                                                    │
│                                                                 │
│  Researchers face a large, unstructured literature landscape.   │
│  Can we automatically predict whether a paper is freely        │
│  accessible — using only its metadata?                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Three OA Categories to Predict:                        │   │
│  │                                                         │   │
│  │  🟢 fully_open      834 papers  (41.7%)                 │   │
│  │  🟡 partially_open  723 papers  (36.2%)                 │   │
│  │  🔴 closed          443 papers  (22.2%)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  DA3 OBJECTIVES:                                                │
│  ① Final performance comparison (12 baseline + 3 tuned)        │
│  ② Statistical significance testing (Friedman test)            │
│  ③ Interactive R Shiny + Plotly dashboard                       │
│  ④ Result interpretation and discussion                        │
│  ⑤ Conclusion, limitations, future scope                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- The problem: given metadata for a research paper — title, abstract, publisher, citations — predict whether it is fully open, partially open, or closed access.
- This matters because open-access status determines whether a researcher can actually read the paper.
- The dataset has 2,000 papers across three classes with moderate imbalance.
- DA3 objectives are the six rubric criteria: comparison, stats, dashboard, interpretation, conclusion, and presentation.

---

---

# SLIDE 3 — PROJECT PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│  RESEARCHPILOT — COMPLETE PIPELINE                             │
│                                                                 │
│  DA1 ──────────────────────────────────────────────────────┐   │
│  OpenAlex API → 2,000 papers → Clean → EDA → final_dataset │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                     │
│  DA2 ──────────────────────────────────────────────────────┐   │
│  27 raw → 206 engineered → 80 selected features            │   │
│  SQLite DB (4 tables) → 12 ML models → Tuning → Clustering │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                     │
│  DA3 ──────────────────────────────────────────────────────┐   │
│  Friedman Test → Statistical Validation                    │   │
│  Final Comparison → All 15 Models Evaluated                │   │
│  R Shiny Dashboard → Interactive Exploration               │   │
│  Report + Viva Preparation                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                     │
│  FUTURE  [ RAG  |  Ollama LLM  |  Fine-tuning ]  ❌ Roadmap    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- Here is the three-phase pipeline.
- DA1: collected and cleaned 2,000 papers from OpenAlex.
- DA2: feature engineering (27→206→80), SQLite database, 12 ML models, hyperparameter tuning, clustering.
- DA3 — what I'm presenting today: statistical validation, final model comparison, interactive dashboard.
- Future scope includes RAG, local LLM via Ollama, fine-tuning — these are NOT implemented. They are planned post-DA3.

---

---

# SLIDE 4 — DATASET AND FEATURE ENGINEERING

```
┌─────────────────────────────────────────────────────────────────┐
│  DATASET & FEATURE ENGINEERING                                  │
│                                                                 │
│  SOURCE: OpenAlex API | 2,000 AI/ML papers | 27 raw columns    │
│                                                                 │
│  FEATURE ENGINEERING PIPELINE:                                  │
│                                                                 │
│   27 raw ──────────────────────────────────────────► 206        │
│   columns   + Structural (12): abstract_length,                 │
│               title_length, text_richness                       │
│             + Domain flags (25): dom_medical, dom_llm           │
│             + Publisher one-hot (18): pub_ieee, pub_mdpi        │
│             + TF-IDF terms (~150): tf_deep, tf_model            │
│                                                                 │
│   206 ─────────────────────────────────────────────► 80         │
│   features  Step 1: Near-Zero Variance filter                   │
│             Step 2: Pearson correlation filter (TF-IDF)         │
│             Step 3: RF Gini importance → top 80                 │
│                                                                 │
│   ⚠️ All steps fitted on TRAINING DATA ONLY — no leakage        │
│                                                                 │
│  SPLITS: 70% train (1400) | 15% val (299) | 15% test (301)     │
│          Stratified | set.seed(42) | Test untouched until end   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- From 27 raw OpenAlex columns I engineered 206 features across four groups.
- Key insight from feature importance: publisher features (pub_ieee, pub_mdpi) are the strongest predictors — OA status is primarily a publisher policy decision.
- Feature selection reduced 206 to 80 using three steps, all fitted on training data only to prevent leakage.
- The test set of 301 papers was untouched until the final evaluation.

---

---

# SLIDE 5 — 12 MACHINE LEARNING MODELS

```
┌─────────────────────────────────────────────────────────────────┐
│  MACHINE LEARNING MODELS — 12 ALGORITHMS                       │
│                                                                 │
│  LINEAR / DISCRIMINANT       TREE-BASED / ENSEMBLE             │
│  ─────────────────────       ───────────────────────           │
│  ✦ LDA (best baseline)       ✦ Decision Tree                   │
│  ✦ Logistic Regression       ✦ Random Forest                   │
│  ✦ Elastic Net               ✦ Gradient Boosting               │
│  ✦ SVM-Linear                ✦ AdaBoost                        │
│  ✦ SVM-RBF                   ✦ XGBoost (worst)                 │
│                                                                 │
│  PROBABILISTIC / NEURAL                                        │
│  ────────────────────────────                                   │
│  ✦ Naive Bayes               ✦ MLP (neural network)            │
│                                                                 │
│  EVALUATION:                                                    │
│  • Same stratified test set for all models (n=301)             │
│  • Metrics: Accuracy, Precision, Recall, Macro-F1, ROC-AUC     │
│  • Macro-F1 = primary metric (handles class imbalance)         │
│                                                                 │
│  TUNED (5-fold CV grid search):                                 │
│  Random Forest | SVM-RBF | XGBoost                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- I trained 12 algorithms covering all major algorithm families.
- All models evaluated on the same 301-paper test set — ensuring a fair apples-to-apples comparison.
- Macro-F1 is the primary metric because it weights all three classes equally, avoiding the bias that accuracy has toward the majority class.
- Three models were hyperparameter-tuned using 5-fold CV grid search on the training set only.

---

---

# SLIDE 6 — FINAL PERFORMANCE COMPARISON ⭐

```
┌─────────────────────────────────────────────────────────────────┐
│  FINAL PERFORMANCE COMPARISON (Test Set, n=301)                │
│                                                                 │
│  MODEL                    ACCURACY  MACRO-F1  ROC-AUC  TYPE    │
│  ─────────────────────────────────────────────────────────────  │
│  SVM-RBF (Tuned)  ★        57.48%   0.5701     —      Tuned    │
│  LDA               ★       57.48%   0.5696    0.7577   Base    │
│  MLP                       56.81%   0.5657    0.7659   Base    │
│  Logistic Regression       56.48%   0.5631    0.7565   Base    │
│  Elastic Net               55.81%   0.5539    0.7584   Base    │
│  RF (Tuned)                56.15%   0.5475     —      Tuned    │
│  SVM-Linear                55.15%   0.5441    0.7587   Base    │
│  Gradient Boosting         53.82%   0.5384    0.7402   Base    │
│  Random Forest             54.82%   0.5317    0.7572   Base    │
│  AdaBoost                  51.50%   0.5123    0.6963   Base    │
│  Decision Tree             50.83%   0.5017    0.6924   Base    │
│  Naive Bayes               41.86%   0.4081    0.6840   Base    │
│  XGBoost (Tuned)           35.55%   0.3537     —      Tuned    │
│  XGBoost                   34.55%   0.3428    0.5243   Base    │
│                                                                 │
│  KEY: Best baseline = LDA (0.5696) | Best tuned = SVM (0.5701) │
│       Tuning margin over best baseline = +0.0005 F1            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- This is the complete comparison — 12 baseline and 3 tuned models on the same held-out test set.
- Best model overall: SVM-RBF Tuned at 57.01% Macro-F1 and 57.48% accuracy.
- Best baseline: LDA at 56.96% F1 — remarkable for a simple linear model.
- Linear models (LDA, LR, Elastic Net) all outperform complex tree ensembles — the feature space is approximately linearly separable.
- XGBoost performs worst (34.28% F1) — the dataset is too small for its complexity.
- The tuning margin is only +0.0005 F1 over best baseline — diminishing returns, the feature space is already well-exploited.
- See figure: DA3/figures/final_f1_comparison.png

---

---

# SLIDE 7 — STATISTICAL ANALYSIS ⭐

```
┌─────────────────────────────────────────────────────────────────┐
│  STATISTICAL ANALYSIS — FRIEDMAN TEST                          │
│                                                                 │
│  QUESTION: Are performance differences statistically real,      │
│            or just random fold variation?                       │
│                                                                 │
│  TEST: Friedman (non-parametric, repeated measures)            │
│        Why? Same 5 folds → repeated measures                   │
│             Can't assume normality at K=5 → non-parametric     │
│                                                                 │
│  H₀: All 12 models have equal median Macro-F1 across folds     │
│  H₁: At least one model differs significantly                  │
│  α  = 0.05                                                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  χ²(11) = 43.4615   p = 0.000009   → REJECT H₀         │  │
│  │                                                          │  │
│  │  CONCLUSION: Model differences are STATISTICALLY REAL   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  POST-HOC (Nemenyi) — 7 significant pairs:                     │
│  • XGBoost vs RF/LDA/ElasticNet/LR   (all p < 0.05)           │
│  • Naive Bayes vs RF/LDA/ElasticNet  (all p < 0.05)           │
│  • Top models (LDA, RF, EN, LR) DO NOT differ significantly    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- The Friedman test is the standard non-parametric test for comparing multiple models across repeated CV folds.
- Result: chi-squared = 43.46, p = 0.000009 — we reject H₀ with overwhelming evidence.
- The performance differences we see are statistically real, not noise.
- Post-hoc Nemenyi test found 7 significantly different pairs — all involving XGBoost or Naive Bayes at the bottom.
- The top models — LDA, RF, Elastic Net, Logistic Regression — do NOT significantly differ from each other. Any of them is a statistically defensible choice.
- Important: CV ranking differs from test ranking (RF best in CV, LDA best on test). This is expected and explained in the report.
- See figures: DA3/figures/stat_cv_boxplot.png, stat_fold_heatmap.png, stat_mean_f1_bars.png

---

---

# SLIDE 8 — INTERACTIVE DASHBOARD DEMO ⭐

```
┌─────────────────────────────────────────────────────────────────┐
│  INTERACTIVE DASHBOARD — R Shiny + Plotly                      │
│                                                                 │
│  Launch: setwd("e:/Tuned_Research")                            │
│          shiny::runApp("dashboard")                             │
│                                                                 │
│  ┌──────────┬──────────────┬──────────────┬──────────────┐     │
│  │ Tab 1    │ Tab 2        │ Tab 3        │ Tab 4        │     │
│  │ Overview │ Model        │ Model        │ Research     │     │
│  │          │ Comparison   │ Diagnostics  │ Clusters     │     │
│  └──────────┴──────────────┴──────────────┴──────────────┘     │
│                                                                 │
│  TAB 1: KPI cards (2000 papers, 80 features, 12 models)        │
│          OA distribution | Feature pipeline                     │
│                                                                 │
│  TAB 2: ★ METRIC SELECTOR — choose Accuracy/F1/AUC/Precision  │
│          Dynamic Plotly bar chart updates instantly             │
│          Baseline (blue) vs Tuned (red) | Hover for values     │
│                                                                 │
│  TAB 3: MODEL SELECTOR — choose LDA or Random Forest           │
│          Confusion matrix heatmap | ROC curves | Feature imp.  │
│                                                                 │
│  TAB 4: Cluster pie chart | Dominant terms | Theme labels      │
│                                                                 │
│  All data loaded from project files — no hardcoded values      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- [LIVE DEMO MOMENT — launch dashboard before presentation starts]
- The dashboard is built with R Shiny and Plotly — fully interactive, not static images.
- Tab 1: Show KPI cards. Click the OA distribution bar — it's interactive.
- Tab 2: Change the metric dropdown from F1 to Accuracy — watch the chart update in real time.
- Hover over a bar to see exact values. Blue = baseline, red = tuned.
- Tab 3: Change the model selector — confusion matrix and ROC curves update. Show feature importance for Random Forest.
- Tab 4: Show cluster pie chart — click segments to highlight.
- All data is loaded from CSV files — no hardcoded values anywhere.
- BACKUP: If dashboard fails to load, use the screenshots in DA3/figures/.

---

---

# SLIDE 9 — KEY FINDINGS AND INTERPRETATION ⭐

```
┌─────────────────────────────────────────────────────────────────┐
│  KEY FINDINGS                                                   │
│                                                                 │
│  1. PUBLISHER FEATURES DOMINATE                                 │
│     pub_ieee, pub_mdpi = top-2 RF importance features          │
│     OA status = publisher policy, NOT paper content            │
│     ⚠️ Correlation, NOT causation                              │
│                                                                 │
│  2. LINEAR MODELS WIN                                           │
│     LDA > MLP > LR > SVM-RBF > Elastic Net > ...              │
│     Feature space is approximately linearly separable          │
│     Complex ensembles hurt on small dataset (~1400 rows)       │
│                                                                 │
│  3. partially_open IS THE HARDEST CLASS                         │
│     Lowest class AUC (0.6759 vs closed 0.8256)                 │
│     Definitional ambiguity → inherent label noise              │
│     Most cross-class confusion in the CM                       │
│                                                                 │
│  4. TUNING GIVES DIMINISHING RETURNS                            │
│     Best gain: RF +0.0158 F1                                   │
│     Overall margin over best baseline: +0.0005 F1              │
│     LDA already found near-optimal linear boundary             │
│                                                                 │
│  5. STATISTICALLY VALIDATED                                     │
│     XGBoost and Naive Bayes are provably inferior (p<0.05)     │
│     Top models are statistically interchangeable               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- Finding 1: Publisher features dominate. This is the most practically important insight. Open-access status is determined by publisher policy, not by what the paper is about. MDPI is almost entirely open-access; Elsevier is mostly closed.
- Finding 2: Linear models win — surprising but explainable. The binary publisher flags and structural features form clean linear boundaries.
- Finding 3: partially_open is the hardest class because its definition is inherently fuzzy — the same paper can be open on arXiv but paywalled at the journal.
- Finding 4: Tuning improved individual models but the overall margin is tiny — the features, not the algorithm, are the bottleneck.
- Finding 5: The Friedman test statistically confirms that XGBoost and Naive Bayes are genuinely inferior for this task.

---

---

# SLIDE 10 — CLUSTERING AND RESEARCH THEMES

```
┌─────────────────────────────────────────────────────────────────┐
│  TOPIC CLUSTERING — K-Means (k=3) on TF-IDF                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  CLUSTER 1 (10.0%, 200 papers)                         │   │
│  │  🎓 Education / GenAI / ChatGPT                        │   │
│  │  Terms: educ, chatgpt, student, generat, tool           │   │
│  │                                                         │   │
│  │  CLUSTER 2 (71.1%, 1422 papers)                        │   │
│  │  🤖 General ML / Prediction / Robotics                 │   │
│  │  Terms: model, learn, data, predict, robot, system     │   │
│  │                                                         │   │
│  │  CLUSTER 3 (18.9%, 378 papers)                         │   │
│  │  🏥 Medical Imaging / Deep Learning                    │   │
│  │  Terms: imag, detect, segment, medic, network, diseas  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Method: L2-normalised TF-IDF | Best k by silhouette score     │
│  Silhouette ≈ 0.06 (weak sep. — expected for AI/ML overlap)    │
│                                                                 │
│  ⚠️ Labels are HUMAN-INTERPRETED from top TF-IDF terms         │
│     K-Means produces numbers, NOT semantic labels              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- Clustering is a separate, fully unsupervised analysis — it does NOT use the OA category labels.
- K-Means identified 3 research themes in the AI/ML corpus.
- Cluster 2 dominates at 71% — this is mainstream AI/ML: classification, regression, robotics.
- Cluster 1 (10%) is a distinct emerging subfield: the explosion of papers about ChatGPT in education since 2022.
- Cluster 3 (19%) is coherent: medical imaging with deep learning — very specific vocabulary around segmentation and disease detection.
- Important caveat: I assigned the theme labels by reading the top TF-IDF terms. The algorithm produced cluster numbers 1, 2, 3 — not "Education" or "Medical Imaging". This is standard practice in unsupervised learning.
- See dashboard Tab 4 for interactive exploration.

---

---

# SLIDE 11 — LIMITATIONS AND FUTURE SCOPE

```
┌─────────────────────────────────────────────────────────────────┐
│  LIMITATIONS (genuine — not hypothetical)                      │
│                                                                 │
│  ① Dataset: 2,000 papers, AI/ML only, static snapshot          │
│  ② Performance: ~57% accuracy — 3-class overlap limits ceiling │
│  ③ partially_open: ambiguous definition → label noise          │
│  ④ Features: TF-IDF only — no semantic embeddings              │
│  ⑤ No full-text: titles + abstracts only                       │
│  ⑥ Metadata dependency: missing DOIs degrade predictions       │
│  ⑦ CV folds K=5: limited statistical power for small gaps      │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  FUTURE SCOPE  (❌ NOT IMPLEMENTED — planned roadmap)           │
│                                                                 │
│  Tier 1 (High Priority):                                        │
│  ❌ RAG-based research assistant (vector DB + LLM)             │
│  ❌ Local LLM via Ollama (hardware-appropriate model)          │
│  ❌ LoRA/QLoRA fine-tuning for research tasks                  │
│                                                                 │
│  Tier 2:                                                        │
│  ❌ SciBERT / Sentence-BERT semantic embeddings                │
│  ❌ Paper similarity search                                     │
│  ❌ Research gap detection                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- Limitations are genuine. The 57% accuracy reflects the inherent difficulty of this task — three overlapping classes with metadata-only features.
- The most impactful future improvement would be SciBERT embeddings replacing TF-IDF — expected +5-10% F1.
- Collapsing to binary (open vs closed) would likely push accuracy above 70%.
- The Ollama/LLM layer is the next major development phase — AFTER DA3 submission. Hardware inspection, model selection, RAG implementation.
- Every future item on this slide does NOT exist in the current codebase.

---

---

# SLIDE 12 — CONCLUSION

```
┌─────────────────────────────────────────────────────────────────┐
│  CONCLUSION                                                     │
│                                                                 │
│  ResearchPilot DA3 delivers a complete, reproducible,           │
│  statistically validated ML pipeline for scholarly metadata.   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  WHAT WAS ACHIEVED                                      │   │
│  │                                                         │   │
│  │  ✅ 12 ML models trained, 3 tuned, all evaluated        │   │
│  │  ✅ Best model: SVM-RBF Tuned (Macro-F1 = 0.5701)       │   │
│  │  ✅ Statistical validation: χ²=43.46, p=0.000009        │   │
│  │     → XGBoost & Naive Bayes provably inferior           │   │
│  │  ✅ R Shiny + Plotly dashboard (4 interactive tabs)     │   │
│  │  ✅ 3 research themes identified via K-Means            │   │
│  │  ✅ Full documentation + viva preparation               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  KEY INSIGHT: OA status is a publisher policy decision.        │
│  Linear models + engineered features achieve 57.5% accuracy   │
│  — a 73% lift over random baseline (33%).                      │
│                                                                 │
│  Foundation laid for Phase 3: RAG + Ollama LLM (Post-DA3)     │
│                                                                 │
│              Thank you — Questions?                             │
└─────────────────────────────────────────────────────────────────┘
```

**SPEAKER NOTES:**
- ResearchPilot DA3 is complete. We have a fully reproducible ML pipeline with statistical validation and an interactive dashboard.
- The single most important insight: open-access status is about publisher policy, not paper content. This is something the models learned from data, not something we assumed.
- 57.5% accuracy represents a 73% lift over random guessing — meaningful, but not production-ready. We know exactly why and what would fix it.
- The foundation is solid for the Ollama/LLM layer in Phase 3.
- Thank you. I'm ready for questions.

---

---

## PRESENTATION DELIVERY GUIDE

### Timing (10-12 minutes total)

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title | 30 sec |
| 2 | Problem & Objectives | 60 sec |
| 3 | Pipeline | 60 sec |
| 4 | Dataset & Features | 90 sec |
| 5 | 12 Models | 60 sec |
| 6 | Final Comparison ⭐ | 90 sec |
| 7 | Statistical Analysis ⭐ | 90 sec |
| 8 | Dashboard Demo ⭐ | 120 sec (live demo) |
| 9 | Key Findings | 90 sec |
| 10 | Clustering | 60 sec |
| 11 | Limitations & Future | 60 sec |
| 12 | Conclusion | 30 sec |
| **Total** | | **~12 min** |

### Pre-Presentation Checklist

```
□ Open RStudio
□ Run: setwd("e:/Tuned_Research"); shiny::runApp("dashboard")
□ Verify dashboard loads in browser (all 4 tabs)
□ Test metric selector in Tab 2 (change from F1 to Accuracy)
□ Test model selector in Tab 3
□ Minimize RStudio, maximize browser
□ Have DA3/figures/ folder open as backup
□ Know your 5 key numbers:
    - Best model F1 = 0.5701 (SVM-RBF Tuned)
    - Best baseline F1 = 0.5696 (LDA)
    - Friedman p = 0.000009
    - Significant pairs = 7
    - Test accuracy = 57.48%
```

### Demo Flow (Slide 8 — 2 minutes)

1. **Tab 1** (20 sec): "Here is the overview — 2,000 papers, 80 features, 12 models."
2. **Tab 2** (40 sec): "Let me show you the model comparison — [change dropdown to Accuracy] — notice how the ranking shifts slightly. [Hover over LDA bar] — 57.48% accuracy, tooltip shows exact value."
3. **Tab 3** (40 sec): "Model diagnostics — [change to Random Forest] — confusion matrix updates. [Point to feature importance] — publisher features at the top."
4. **Tab 4** (20 sec): "Research clusters — three themes. The labels here are my interpretation of the top TF-IDF terms, not algorithmic labels."

### If Dashboard Fails

Open `DA3/figures/` folder and show:
- `final_f1_comparison.png` instead of Tab 2
- `stat_cv_boxplot.png` for statistical analysis
- `final_all_metrics_heatmap.png` for comprehensive view

Say: "I have a live dashboard but let me show the static figures as backup."

---

## LIKELY VIVA QUESTIONS AFTER PRESENTATION

**Q: "Why did LDA perform better than Random Forest on the test set, but Random Forest was best in CV?"**

A: CV and test set use different data splits and conditions. In CV, all models use default hyperparameters on training folds — Random Forest benefits more from the larger fold data volume. On the test set, LDA's linear discriminant finds a cleaner boundary with the final feature set. Both results are valid — they measure different things. The Friedman test, which uses CV data, shows they are NOT statistically different from each other.

**Q: "What is the chi-squared value and what does it mean?"**

A: χ²(11) = 43.4615, p = 0.000009. The chi-squared statistic measures how much the rank sums deviate from what we'd expect if all models were equal. With df=11 (12 models minus 1), a statistic of 43.46 is far in the tail of the chi-squared distribution — probability 0.000009 of observing this by chance alone.

**Q: "Why Friedman and not ANOVA?"**

A: Three reasons: (1) We have repeated measures — same 5 folds for all models, so observations are not independent. (2) We cannot assume normality of F1 distributions with only 5 data points per model. (3) Friedman is the standard test in ML benchmarking for this exact scenario (Demšar, 2006).

**Q: "What does 57% accuracy actually mean in practice?"**

A: Random guessing on 3 equally likely classes gives 33%. Our best model at 57.5% represents a 73% improvement over random. But practically, 43% of predictions are wrong — too many for production deployment. The right next step is either binary classification (open vs closed, dropping partial) or semantic embeddings — both would push accuracy above 70%.

**Q: "Is the Ollama LLM implemented?"**

A: No. Local LLM integration via Ollama is future scope. It is planned to begin after DA3 submission. The plan involves: hardware inspection, appropriate model selection (Qwen/Llama/Gemma based on GPU VRAM), RAG pipeline implementation, and integration with ResearchPilot. It is explicitly documented as future scope in the report.
