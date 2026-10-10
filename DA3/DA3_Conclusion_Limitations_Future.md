# Conclusion, Limitations, and Future Scope
## ResearchPilot DA3 — Human-Centered AI Research Assistant

---

## 1. Conclusion

ResearchPilot DA3 completes the final evaluation phase of a human-centered AI research assistant built on a corpus of 2,000 AI/ML scholarly papers from the OpenAlex API.

### 1.1 Summary of DA3 Achievements

**Criterion 1 — Final Performance Comparison and Statistical Analysis**

A complete performance comparison of all 15 models (12 baseline + 3 hyperparameter-tuned) was produced on the proper held-out test set (n=301). The best-performing model is **SVM-RBF (Tuned)** with Macro-F1 = 0.5701 and Accuracy = 57.48%, closely followed by the best baseline **LDA** (Macro-F1 = 0.5696). The margin between the best tuned and best baseline model is +0.0005 F1 — practically negligible but confirming that tuning provided incremental benefit.

A **Friedman non-parametric test** was applied across 12 baseline models evaluated on 5-fold cross-validation, testing whether observed performance differences are statistically consistent. The test was accompanied by Nemenyi post-hoc analysis where the null hypothesis was rejected. All statistical outputs include the null hypothesis, alternative hypothesis, test statistic, degrees of freedom, p-value, and a plain-language conclusion.

**Criterion 2 — Interactive Dashboard**

An **R Shiny + Plotly interactive dashboard** was built with four tabs:
- Tab 1: Project overview with KPI cards and OA distribution
- Tab 2: Model comparison with a live metric selector (Accuracy, F1, ROC-AUC, Precision, Recall)
- Tab 3: Model diagnostics including confusion matrix heatmap, ROC curves, and feature importance
- Tab 4: Research cluster visualization with theme descriptions and dominant terms

The dashboard is fully interactive — metric selectors, model selectors, and hover tooltips all update the display dynamically. All data is read from existing project files with no hardcoded metrics.

**Criterion 3 — Result Interpretation and Discussion**

A comprehensive 10-section discussion covers: dataset distribution, feature engineering impact, model comparison reasoning, hyperparameter tuning analysis, statistical analysis interpretation, confusion matrix error analysis, ROC-AUC interpretation, feature importance with causation warning, clustering interpretation with labeling caveat, and overall synthesis.

**Criterion 4 — Conclusion, Limitations, Future Scope**

This document.

**Criterion 5 — Final Report and Code Organization**

The project is organized with a dedicated `DA3/` directory containing all Phase 3 deliverables. New R scripts are placed in `r/scripts/phase3/`. The existing DA2 pipeline is preserved unchanged. A comprehensive final report covering all six criteria is provided.

**Criterion 6 — Presentation and Viva**

A 12-slide presentation covering all DA3 criteria and a 50+ question viva preparation document are provided in `DA3/DA3_Viva_QA.md`.

### 1.2 Key Findings

1. **Publisher features dominate OA prediction.** The top two Random Forest features are publisher indicators (`pub_ieee`, `pub_mdpi`), confirming that open-access status is primarily a publisher policy decision, not a paper-content property.

2. **Linear models outperform complex ensembles.** LDA, Logistic Regression, and Elastic Net all outperform Random Forest, XGBoost, and AdaBoost. This reflects the linear separability of the engineered feature space and the dataset size constraint.

3. **Tuning provides modest but real improvement.** Random Forest improved by +0.0158 F1, SVM-RBF by +0.0142. The overall best model (SVM-RBF Tuned) is only +0.0005 ahead of the best baseline (LDA), demonstrating diminishing returns when the feature space is already well-exploited.

4. **Partial_open is the most challenging class.** ROC-AUC for `partially_open` (0.6759) is 0.0942 lower than for `closed` (0.8256), and the confusion matrix shows it is frequently misclassified into both other categories. This reflects genuine ambiguity in the target definition.

5. **Three distinct research themes exist in AI/ML.** K-Means clustering identified Education/GenAI/ChatGPT (10%), General ML/Robotics (71.1%), and Medical Imaging/Deep Learning (18.9%) as the dominant topic clusters. The weak silhouette score (~0.06) reflects the expected overlap in an interconnected research domain.

### 1.3 Contribution Statement

ResearchPilot demonstrates that a fully reproducible, statistically validated ML pipeline for scholarly metadata analysis is achievable with openly available data and R-based tooling. The project contributes:

- A clean 80-feature engineering pipeline from raw OpenAlex metadata
- A 12-model comparative evaluation with proper train/val/test separation
- A statistical analysis framework for multi-model comparison
- An interactive dashboard for exploratory result analysis
- A foundation for future research assistant features (RAG, local LLM, semantic search)

---

## 2. Limitations

The following limitations are **genuine and verified** — they reflect the actual constraints of the current implementation, not hypothetical concerns.

### L1: Dataset Size (2,000 Papers)

**What:** The corpus contains 2,000 papers from a single API snapshot.

**Impact:** Complex models (XGBoost, MLP, deep neural networks) are under-served by this dataset size and likely to overfit. Statistical tests have limited power with few observations per class per fold.

**Justification:** The dataset size was constrained by the OpenAlex API rate limits, the project timeline, and the scope of a B.Tech research project. It is sufficient to demonstrate the pipeline and train interpretable models, but not to benchmark large-scale ML methods.

### L2: AI/ML Domain Only

**What:** All papers are from the AI/ML research domain. No papers from biology, economics, social science, or other fields are included.

**Impact:** The trained models and feature importance rankings may not generalise to other scholarly domains. Publisher features are AI/ML-specific (MDPI and IEEE dominate this domain; different publishers dominate other fields).

**Justification:** A focused domain was chosen to enable meaningful domain-specific feature engineering (domain binary flags, relevant TF-IDF terms). Multi-domain expansion is future scope.

### L3: Moderate Classification Performance (~57% Accuracy)

**What:** The best model achieves 57.48% accuracy on a 3-class problem.

**Impact:** This level of accuracy is insufficient for production deployment. A research tool that misclassifies 43% of papers would frustrate users and erode trust.

**Why this is expected:** Three-class classification with a poorly-defined middle class (`partially_open`) is inherently difficult. Binary classification (open vs closed) or removing `partially_open` entirely would likely yield 70%+ accuracy. The current performance reflects the difficulty of the task, not a failure in methodology.

### L4: Partially_Open Class Ambiguity

**What:** The `partially_open` category has a fluid definition — it depends on embargoes, preprint availability, institutional access, and OpenAlex's metadata at a specific point in time.

**Impact:** This introduces label noise into the training data. Some papers assigned `partially_open` may genuinely be fully accessible at the time of model deployment, or the reverse.

**Why this cannot be fixed:** Without re-annotating the dataset with time-aware OA status (a major data collection effort), this limitation is inherent.

### L5: TF-IDF Features (Not Semantic Embeddings)

**What:** Text features are represented as TF-IDF term frequencies, not contextual semantic embeddings (Sentence-BERT, SciBERT, OpenAI embeddings).

**Impact:** TF-IDF misses semantic similarity — "cardiac imaging" and "heart scan" would be unrelated in TF-IDF space but similar in embedding space. This limits the model's ability to learn topic-based patterns that transcend exact vocabulary.

**Why TF-IDF was used:** Computationally efficient, fully interpretable, reproducible without API keys or GPU requirements, and appropriate for a B.Tech data science project scope.

### L6: Static Dataset (Snapshot in Time)

**What:** The 2,000 papers were collected at a specific point in time (2024–2025). No mechanism for continuous updates exists.

**Impact:** The model will gradually become outdated as new papers are published, OA policies change, and new publishers emerge. Predictions for papers published after the corpus snapshot may be less accurate.

**Future fix:** A real-time data ingestion pipeline from OpenAlex would address this (future scope).

### L7: Metadata Dependency

**What:** The feature engineering relies on OpenAlex metadata quality — DOI availability, abstract completeness, concept annotations, citation counts.

**Impact:** Papers with missing DOIs cannot benefit from publisher features. Papers without abstracts cannot use TF-IDF features. Approximately 5–10% of papers in typical scholarly databases have incomplete metadata.

**Why this is a limitation:** The project cannot control OpenAlex data quality. Papers with missing critical fields will receive degraded predictions.

### L8: Clustering Label Subjectivity

**What:** The cluster labels ("Education/GenAI/ChatGPT", "General ML/Prediction/Robotics", "Medical Imaging/Deep Learning") are assigned by human interpretation of TF-IDF top terms.

**Impact:** Another researcher might assign different labels to the same clusters. The labels are interpretations, not ground truths.

**Why this is inherent:** Unsupervised clustering does not have "correct" labels. Human interpretation is the standard and accepted practice in topic modeling and cluster analysis.

### L9: No Full-Text Analysis

**What:** Features are derived from titles and abstracts only. The body of the paper (methodology, experiments, conclusions) is not analysed.

**Impact:** Methodological signals (e.g., "we conducted an RCT" suggesting medical/funded research) that correlate with OA status are missed. Full-text features could significantly improve model performance.

**Why this was not done:** OpenAlex does not provide full text. Copyright constraints prevent bulk full-text downloading. Full-text analysis is future scope.

### L10: 5-Fold Cross-Validation Statistical Power

**What:** The Friedman test uses K=5 folds, providing 5 data points per model for the statistical test.

**Impact:** K=5 provides limited power to detect small performance differences. A difference of 0.01 F1 between models may not reach significance even if it is real.

**Why 5 folds:** Computational cost — with 12 models and 5 folds, each fold requires retraining 12 models. K=10 would double the computation with only marginal power improvement. K=5 is a standard and defensible choice.

---

## 3. Future Scope

The following are **planned but not implemented** features. Each is explicitly marked as future work.

### IMPORTANT DISTINCTION

| Status | Meaning |
|--------|---------|
| ✅ IMPLEMENTED | This exists in the current codebase and has been tested |
| ❌ NOT IMPLEMENTED | This is future scope only. It does not exist in the current project. |

---

### Tier 1: High Priority (Nearest-Term Roadmap)

**F1. RAG-Based Research Assistant** ❌ NOT IMPLEMENTED

Retrieval-Augmented Generation (RAG) would allow users to ask questions about the research corpus and receive grounded answers backed by retrieved papers.

- Approach: Vector database (Chroma or FAISS) → semantic search → retrieved context → LLM response
- Benefit: Grounded responses, no hallucination of citations, scalable knowledge base
- Stack: Python, LangChain/LlamaIndex, Sentence-BERT/OpenAI embeddings, Chroma
- Estimated effort: 30–40 hours

**F2. Local LLM Integration via Ollama** ❌ NOT IMPLEMENTED

Integrating a locally-run LLM (via Ollama) would enable privacy-preserving inference without sending paper data to external APIs.

- Approach: Install Ollama → select appropriate model (Qwen, Llama, Gemma based on hardware) → connect to ResearchPilot retrieval pipeline
- Benefit: Data privacy, no API costs, works offline
- Hardware dependency: Model selection depends on GPU VRAM — will be assessed before implementation
- Estimated effort: 15–25 hours

**F3. Instruction Fine-Tuning for Research Tasks** ❌ NOT IMPLEMENTED

A small instruction dataset (500–1,000 examples) could be constructed from the 2,000 papers to fine-tune an LLM for research-specific tasks:
- Paper summarization
- Research question generation
- Methodology extraction
- Domain identification

- Approach: LoRA/QLoRA fine-tuning on base instruct model
- Benefit: Better structure and relevance for research-oriented responses compared to a general-purpose LLM
- Prerequisite: Base model selected via Ollama (F2)
- Estimated effort: 20–30 hours

---

### Tier 2: Medium Priority

**F4. Semantic Embeddings (SciBERT / Sentence-BERT)** ❌ NOT IMPLEMENTED

Replace TF-IDF features with contextual embeddings from a pre-trained scientific language model.

- Expected improvement: +5–10% F1 based on literature
- Models: SciBERT (scientific text), Sentence-BERT (sentence similarity), specter2 (paper embeddings)
- Benefit: Captures semantic similarity beyond keyword overlap
- Estimated effort: 8–12 hours

**F5. Research Gap Detection** ❌ NOT IMPLEMENTED

Analyse the corpus to identify research areas with few recent publications or low citation density, suggesting promising under-explored directions.

- Approach: Citation graph analysis + TF-IDF cluster density + temporal trends
- Benefit: Helps researchers find novel directions
- Estimated effort: 15–20 hours

**F6. Paper Similarity Search** ❌ NOT IMPLEMENTED

Given a paper (or abstract), retrieve the most similar papers from the corpus.

- Approach: Semantic embedding → cosine similarity → top-k retrieval
- Benefit: "More like this" functionality for literature discovery
- Estimated effort: 8–12 hours

**F7. Journal Recommendation System** ❌ NOT IMPLEMENTED

Predict the most suitable journal/venue for a given paper based on its content.

- Approach: Multi-label classification on journal/conference targets
- Benefit: Helps researchers target appropriate publication venues
- Estimated effort: 12–15 hours

---

### Tier 3: Long-Term Vision

**F8. Larger Multi-Domain Dataset (10,000+ Papers)** ❌ NOT IMPLEMENTED

Expand beyond 2,000 AI/ML papers to a larger, multi-domain corpus (biology, economics, social science, etc.) with continuous updates from OpenAlex.

**F9. Real-Time Data Pipeline** ❌ NOT IMPLEMENTED

Automated ingestion from OpenAlex (weekly or monthly) to keep the corpus current and retrain models incrementally.

**F10. Causal Inference on OA Patterns** ❌ NOT IMPLEMENTED

Move beyond predictive correlation to understand whether specific interventions (e.g., choosing a particular publisher) causally affect a paper's OA status. Requires instrumental variable analysis or causal DAG modelling.

**F11. Explainable AI (SHAP / LIME) Integration** ❌ NOT IMPLEMENTED

Per-prediction feature contribution analysis to explain individual model decisions ("why did the model classify this paper as fully_open?").

**F12. Multi-Modal Features** ❌ NOT IMPLEMENTED

Extract features from paper figures, tables, and equations using computer vision and layout analysis.

---

## 4. Clearly Implemented vs Future: Summary Table

| Feature | Status |
|---------|--------|
| OpenAlex data collection | ✅ IMPLEMENTED |
| Feature engineering (27→206→80) | ✅ IMPLEMENTED |
| SQLite database (4 tables) | ✅ IMPLEMENTED |
| 12 ML models trained and evaluated | ✅ IMPLEMENTED |
| Hyperparameter tuning (RF, SVM, XGBoost) | ✅ IMPLEMENTED |
| Impact-tier secondary classification | ✅ IMPLEMENTED |
| Topic clustering (K-Means, k=3) | ✅ IMPLEMENTED |
| Friedman statistical test | ✅ IMPLEMENTED |
| Interactive R Shiny dashboard | ✅ IMPLEMENTED |
| RAG-based Q&A system | ❌ NOT IMPLEMENTED |
| Local LLM (Ollama) integration | ❌ NOT IMPLEMENTED |
| Fine-tuning for research tasks | ❌ NOT IMPLEMENTED |
| Semantic embeddings | ❌ NOT IMPLEMENTED |
| Research gap detection | ❌ NOT IMPLEMENTED |
| Paper similarity search | ❌ NOT IMPLEMENTED |
| Journal recommendation | ❌ NOT IMPLEMENTED |

---

*ResearchPilot DA3 | October 2026*
