# DA3 Results Interpretation and Discussion
## ResearchPilot — Human-Centered AI Research Assistant

---

## 1. Dataset Distribution Analysis

ResearchPilot uses a corpus of **2,000 AI/ML research papers** retrieved from the OpenAlex API. The target variable is the open-access (OA) category assigned by OpenAlex, with three classes:

| OA Category | Count | Percentage | Test Set |
|-------------|-------|-----------|----------|
| fully_open | 834 | 41.7% | 125 (41.5%) |
| partially_open | 723 | 36.2% | 109 (36.2%) |
| closed | 443 | 22.2% | 67 (22.3%) |

### What the distribution tells us

The dataset is **moderately imbalanced**. `fully_open` is the majority class, `closed` is the minority. This imbalance has downstream effects:

- Models that naively predict `fully_open` for every paper would achieve approximately 41.7% accuracy — still below our best model's 57.48%, but not negligible
- The use of **Macro-F1** (unweighted average across classes) is appropriate here, as it penalises poor performance on minority classes equally
- The stratified splitting (70/15/15) preserves class proportions across train, validation, and test sets, ensuring fair evaluation

### Class overlap concern

The biggest challenge is `partially_open`, which by definition occupies a grey area between fully open and fully closed. A paper may be freely readable on a preprint server (partially open) while the publisher version is paywalled (closed). OpenAlex's classification reflects its metadata at a point in time, introducing inherent label noise.

**Implication:** Some misclassifications are not model failures — they reflect genuine ambiguity in the target variable itself.

---

## 2. Feature Engineering Impact (27 → 206 → 80)

### 2.1 Feature Groups Created

Starting from 27 raw OpenAlex columns, four groups of features were engineered:

| Group | Count | Examples | Rationale |
|-------|-------|---------|-----------|
| Structural / text-length | 12 | `abstract_length`, `title_length`, `text_richness`, `recency_score` | Papers with longer abstracts may be more "academic" — correlating with OA status |
| Domain binary indicators | 25 | `dom_medical`, `dom_llm`, `dom_education` | Different subfields have different OA norms |
| Publisher one-hot (DOI prefix) | 18 | `pub_ieee`, `pub_mdpi`, `pub_springer`, `pub_nature` | Publisher policy is the strongest predictor of OA status |
| TF-IDF terms | ~150 | `tf_learning`, `tf_deep`, `tf_model`, `tf_detection` | Paper content vocabulary can reflect publication type |

**Total:** 206 engineered features from 27 raw columns.

### 2.2 Feature Selection (206 → 80)

Three sequential filtering steps, all fitted on the training split only:

1. **Near-Zero Variance (NZV) filter** — removes features with near-constant values (variance < 1e-10 or one value dominating 20× the second)
   - Rationale: Constant features carry no information for any model
   - Example removed: binary domain flag for a field with only 3 papers in the corpus

2. **Pearson correlation filter** (TF-IDF group only) — removes one column from pairs with |r| > 0.95
   - Rationale: Collinear features are redundant and inflate dimensionality
   - Many TF-IDF terms co-occur, creating near-identical columns (e.g., "learn" and "learning" after stemming)

3. **Random Forest Gini importance ranking** — ranks remaining features by Mean Decrease Gini on 300-tree RF; top 80 retained
   - Rationale: Removes weak predictors while preserving the most discriminative features
   - Computationally efficient and model-agnostic selection criterion

**Key constraint:** All three steps were fitted on the **training data only**. The validation and test sets were transformed using training-set parameters. This prevents data leakage — a critical methodological requirement.

### 2.3 Impact of Feature Engineering

The model performance on raw features (baseline from literature/rule of thumb) would be approximately 33–40% for a 3-class problem. The engineered feature pipeline achieves 57.48% test accuracy, demonstrating a meaningful lift from the feature engineering work.

---

## 3. Model Comparison Discussion

### 3.1 Rankings Summary

| Rank | Model | Macro-F1 | Type | Key Characteristic |
|------|-------|----------|------|-------------------|
| 1 | SVM-RBF (Tuned) | 0.5701 | Tuned | Kernel captures non-linearity |
| 2 | LDA | 0.5696 | Baseline | Best baseline; linear discriminant |
| 3 | MLP | 0.5657 | Baseline | Shallow neural network |
| 4 | Logistic Regression | 0.5631 | Baseline | Linear, regularised |
| 5 | Elastic Net | 0.5539 | Baseline | Linear, L1+L2 regularisation |
| ... | ... | ... | ... | ... |
| 11 | Naive Bayes | 0.4081 | Baseline | Assumes independence |
| 12 | XGBoost | 0.3428 | Baseline | Gradient boosting (poor here) |

### 3.2 Why LDA Is the Best Baseline

**LDA (Linear Discriminant Analysis)** achieved the highest Macro-F1 among baseline models (0.5696). This is not surprising for several reasons:

1. **Linear separability in high-dimensional feature space.** With 80 features including publisher binary flags, domain indicators, and TF-IDF terms, the feature space is likely approximately linearly separable. OA status is largely determined by publisher policy — a linear boundary captures this well.

2. **Effective dimensionality reduction.** LDA projects features onto the 2 most class-discriminative axes (for 3 classes). This is equivalent to finding the subspace where classes are most separated, which is a powerful regularisation strategy.

3. **Low variance.** On small-to-medium datasets (~1,400 training samples), LDA's analytical solution avoids the variance inflation that plagues iterative methods.

4. **Publisher features are orthogonal.** Binary publisher flags (pub_ieee, pub_mdpi, pub_springer) that clearly separate OA categories are exactly what LDA exploits.

### 3.3 Why XGBoost Performed Poorly

**XGBoost** achieved only Macro-F1 = 0.3428 — far below the next-worst model (Decision Tree at 0.5017). This warrants explanation:

1. **Dataset size.** XGBoost is designed for large tabular datasets (tens of thousands of rows). With ~1,400 training samples, it likely overfits even with regularisation.

2. **TF-IDF sparsity.** The feature matrix is highly sparse (most TF-IDF entries are zero). XGBoost's tree splits on sparse features can be unstable.

3. **Default hyperparameters were poor.** XGBoost's default learning rate (0.3) and tree depth (6) are not well-suited for this small sparse dataset. Tuning improved it to 0.3537 (+0.0109) but it remained the weakest.

4. **Class imbalance interaction.** XGBoost with default settings does not handle class imbalance well without explicit class weights.

### 3.4 The Mid-Range Cluster

Random Forest (0.5317), Gradient Boosting (0.5384), and AdaBoost (0.5123) form a mid-range cluster. These ensemble methods capture non-linear interactions but require more data than available here to fully exploit their capacity. Their performance demonstrates that ensemble methods are competitive but not superior to linear approaches for this task.

### 3.5 Naive Bayes

Naive Bayes (0.4081) performed poorly because it assumes **conditional independence** of features given the class. This assumption is violated by our feature set — for example, `abstract_length` and `abstract_word_count` are highly correlated, and many TF-IDF terms co-occur. When independence is violated, Naive Bayes produces miscalibrated probability estimates.

---

## 4. Hyperparameter Tuning Discussion

### 4.1 Tuning Results

Three models were tuned using 5-fold cross-validation grid search on the training set:

| Model | Base F1 | Tuned F1 | Improvement | Best Params |
|-------|---------|----------|-------------|-------------|
| Random Forest | 0.5317 | 0.5475 | **+0.0158** | ntree=500, mtry=8 |
| SVM-RBF | 0.5559 | **0.5701** | +0.0142 | cost=10, gamma=0.001 |
| XGBoost | 0.3428 | 0.3537 | +0.0109 | depth=6, eta=0.10, rounds=300 |

### 4.2 Why the Gains Were Modest

The improvement from tuning is real but small. The most important reason is:

**LDA already found the near-optimal decision boundary** — tuning SVM-RBF to match LDA's performance (+0.0005 over best baseline) shows that the feature space does not offer much more linear discriminability to exploit.

Other factors:
- **Random Forest** improved meaningfully (+0.0158 F1) because `ntree=300` (default) was insufficient for this feature set; 500 trees provided better ensemble stability
- **XGBoost** improved modestly because tuning addressed the default hyperparameter mismatch, but dataset size remains a fundamental bottleneck
- **Diminishing returns** — all three models were already reasonably fit; the grid search explored incremental refinements, not architectural changes

### 4.3 Practical Take-Away

For deployment, the **SVM-RBF Tuned** (F1=0.5701) is the recommended model. It combines the kernel's ability to capture mild non-linearities with properly tuned regularisation (cost=10, gamma=0.001 — a wide kernel that avoids overfitting).

---

## 5. Statistical Analysis Interpretation

### 5.1 Friedman Test Context

The Friedman test addresses the question: **are the observed performance differences across models statistically consistent, or could they be explained by random variation in the fold assignments?**

With 12 models evaluated over 5 folds (K=5), the statistical power is limited. A large gap (e.g., XGBoost vs LDA, ΔF1 ≈ 0.23) is likely to be significant. A small gap (e.g., LDA vs MLP, ΔF1 ≈ 0.004) is unlikely to reach significance at K=5.

### 5.2 Key Interpretation Principles

**The Friedman test result (χ² = 43.4615, df = 11, p = 0.000009):**

The p-value of 0.000009 provides overwhelming evidence to **reject H₀**. The 12 models do NOT have equal median performance. The differences we observe are statistically real, not random noise.

**What is significant and what is not (Nemenyi post-hoc, 7 pairs):**

All 7 significant pairs involve either XGBoost or Naive Bayes being outperformed:
- XGBoost is significantly worse than RF, LDA, Elastic Net, and Logistic Regression
- Naive Bayes is significantly worse than RF, LDA, and Elastic Net
- Top-tier models (LDA, RF, Elastic Net, LR, SVM-RBF) do NOT significantly differ from each other

This tells us two things:
1. **The bottom two models (XGBoost, Naive Bayes) are genuinely, statistically inferior** for this task
2. **Among the top models, the differences are real but not statistically distinguishable** at K=5 folds — any of LDA, RF, Elastic Net, or LR could be chosen with statistical confidence

### 5.3 Statistical vs Practical Significance

| Comparison | ΔF1 | Practical Impact | Expected Statistical Significance |
|------------|-----|-----------------|----------------------------------|
| LDA vs XGBoost | +0.2268 | High — LDA strongly preferred | Likely significant |
| LDA vs Naive Bayes | +0.1615 | High | Likely significant |
| LDA vs MLP | +0.0039 | Negligible | Not significant |
| SVM-RBF Tuned vs LDA | +0.0005 | Negligible | Not significant |

---

## 6. Confusion Matrix Analysis

### 6.1 LDA Confusion Matrix (Best Baseline)

The confusion matrix reveals **which classes are confused** with each other, which is more informative than a single accuracy number.

Key observations from the LDA test-set confusion matrix:
- **fully_open** has the best recall (~0.74) — the model captures most fully_open papers
- **partially_open** is the hardest class — it gets confused with both fully_open and closed
- **closed** recall is moderate (~0.51) — about half the closed papers are correctly identified

### 6.2 Why partially_open Is Hardest

1. **Definitional ambiguity** — "partially open" is not a clean category. A paper may be on a preprint server (open) but the journal version is paywalled. Metadata may not capture the current state accurately.

2. **Feature overlap** — papers that are partially open often come from the same publishers (Springer, Elsevier) as fully closed papers, making publisher features ambiguous for this class.

3. **Temporal change** — a paper's OA status can change (embargo periods end, authors post preprints). The model is trained on a snapshot, but OA status changes over time.

4. **Smallest separating margin** — in the LDA feature space, `partially_open` points are scattered between the `fully_open` and `closed` clusters rather than forming a tight, separable cluster.

### 6.3 Practical Implications

For a research assistant application, misclassifying `partially_open` as `fully_open` would direct a user to a paper that may require a subscription. The practical cost of this confusion is real (user frustration, access denial).

A future improvement could be to **collapse the three classes into binary** (open vs closed), which would simplify the decision boundary and likely improve accuracy significantly.

---

## 7. ROC-AUC Interpretation

### 7.1 What ROC-AUC Measures

ROC-AUC (One-vs-Rest, Macro) measures **discrimination ability** — how well a model separates each class from the rest, independent of any classification threshold. It ranges from 0.5 (no better than random) to 1.0 (perfect separation).

**Important:** A model can have moderate accuracy (57%) but good ROC-AUC (0.76) if it produces well-calibrated probability scores even when the threshold-based prediction is ambiguous.

### 7.2 Class-Level AUC Analysis (Random Forest)

| Class | AUC | Interpretation |
|-------|-----|----------------|
| closed | **0.8256** | Strong discrimination — closed papers have distinctive feature profiles |
| fully_open | 0.7701 | Good discrimination — fully open papers relatively distinguishable |
| partially_open | 0.6759 | Moderate — hardest class to discriminate, consistent with confusion matrix |
| **Macro (mean)** | **0.7572** | Overall good discrimination ability |

The `closed` class having the highest AUC is consistent with the theory that closed-access papers are published by a distinct set of publishers (Elsevier, Wiley, etc.) with characteristic DOI prefixes — features the model captures well.

### 7.3 AUC vs Accuracy Disconnect

The model achieves AUC=0.7572 (good) with only 57.48% accuracy (moderate). This disconnect is normal in multi-class problems with overlapping classes: the model knows something useful about probabilities even when the final prediction is wrong. This is why ROC-AUC is a more reliable metric than accuracy when classes overlap.

---

## 8. Feature Importance Discussion

### 8.1 Top Features (Random Forest Gini Importance)

| Rank | Feature | Group | Value | Interpretation |
|------|---------|-------|-------|----------------|
| 1 | pub_ieee | Publisher | 0.0312 | IEEE has specific OA policies — strong signal |
| 2 | pub_mdpi | Publisher | 0.0289 | MDPI is predominantly open-access — strong signal |
| 3 | title_to_abstract_ratio | Structural | 0.0271 | Writing style varies by publication type |
| 4 | abstract_length | Structural | 0.0248 | Longer abstracts correlate with certain OA types |
| 5 | abstract_word_count | Structural | 0.0241 | Correlated with abstract_length (both useful) |
| 6 | title_length | Structural | 0.0229 | Conference vs journal titles differ in length |
| 7 | text_richness | Structural | 0.0218 | Vocabulary diversity — academic writing signal |
| 8 | title_word_count | Structural | 0.0197 | Related to publication venue type |
| 9 | concept_count | Metadata | 0.0184 | More concepts → richer metadata → more likely to be indexed as open |
| 10 | keyword_count | Metadata | 0.0172 | Similar to concept_count |

### 8.2 Key Insight: Publisher Features Dominate

The top two features are both publisher binary flags. This confirms the most important insight of the project:

> **Open-access status is primarily a publisher policy decision, not a paper content decision.**

MDPI publishes almost entirely open-access (under its APC model), so `pub_mdpi = 1` is a very strong signal for `fully_open`. IEEE publishes both open and closed, making its OA profile more complex.

### 8.3 Critical Caveat: Importance ≠ Causation

Feature importance indicates **predictive correlation, not causation**. Feature importance tells us which features the Random Forest **used for splitting** — it does NOT tell us that these features **cause** a paper to be open-access.

The correct interpretation is:

> "Publisher type (pub_ieee, pub_mdpi) was the most **predictively useful** feature for OA classification in our dataset."

The incorrect interpretation would be:

> "Publishing with MDPI causes a paper to become open-access."

The causal mechanism runs the other way: MDPI's business model is open-access-first, so MDPI papers happen to be open-access, and our model correctly exploits this correlation.

### 8.4 Why TF-IDF Terms Have Low Individual Importance

Individual TF-IDF terms (`tf_deep`, `tf_model`) appear low in the importance ranking, but collectively contribute through many terms together. No single keyword is a strong signal, but vocabulary patterns collectively encode domain and publication type.

---

## 9. Clustering Interpretation

### 9.1 K-Means Configuration

- **Algorithm:** K-Means on TF-IDF (200 terms) with L2 normalisation
- **Best k:** 3 (chosen by silhouette score analysis, k=3 through k=10)
- **Silhouette score:** ~0.06 (weak separation — expected for overlapping AI/ML literature)
- **Seed:** set.seed(42) for reproducibility

### 9.2 Cluster Themes

| Cluster | Size | Percentage | Top Terms | Interpreted Theme |
|---------|------|-----------|-----------|-------------------|
| 1 | 200 | 10.0% | educ; chatgpt; student; generat; technolog | Education / GenAI / ChatGPT |
| 2 | 1,422 | 71.1% | model; learn; data; predict; system; robot | General ML / Prediction / Robotics |
| 3 | 378 | 18.9% | imag; detect; segment; featur; medic; network | Medical Imaging / Deep Learning |

### 9.3 What the Clusters Mean

**Cluster 1 (Education/GenAI):** A small but distinct subfield — papers about using AI in education, often mentioning ChatGPT. This cluster has emerged strongly since 2022 with the proliferation of LLMs. Its distinctness from the main cluster reflects genuine topic separation.

**Cluster 2 (General ML/Robotics):** The dominant cluster — almost three-quarters of the corpus. This is the "mainstream" AI/ML research: classification, regression, robotics, system design. Its large size reflects the corpus design (AI/ML papers are diverse).

**Cluster 3 (Medical Imaging/DL):** A coherent subfield — papers about detecting diseases, segmenting medical images, using deep learning (CNN, UNet) for healthcare. Its terms ("imag", "segment", "medic", "convolut") are specific and distinctive.

### 9.4 Why Silhouette Score Is Low

A silhouette score of ~0.06 indicates weak cluster separation. This is **expected** for AI/ML literature because:
- Topics are interconnected (medical imaging uses deep learning which uses ML which has robotics applications)
- TF-IDF terms overlap heavily (every paper uses "model", "learn", "data")
- K-Means assumes spherical clusters — real topic distributions in text are not spherical

The low silhouette does not mean the clusters are useless — it means they capture broad topical tendencies, not sharp divisions.

### 9.5 Clusters vs OA Status

Clustering is **topic-based**, not OA-based. The OA distribution within clusters may vary, but there is no strong claim that topic type causes a specific OA status. Any correlations (e.g., medical papers being more open-access due to NIH mandates) would need explicit statistical testing.

### 9.6 Label Interpretation Caveat

**Cluster labels were assigned by a human** (the researcher) after examining the top TF-IDF terms. The K-Means algorithm does not produce "Education" or "Medical Imaging" labels — it produces cluster assignments. The interpretation is a human judgment and another researcher might describe the same clusters differently. This is normal and acceptable for unsupervised learning — the key is to label what the data shows, not to fabricate labels.

---

## 10. Overall Discussion and Synthesis

### 10.1 What the Project Successfully Demonstrates

1. **Reproducible ML pipeline** — from raw OpenAlex data to evaluated models with no leakage
2. **Thoughtful feature engineering** — 4 feature groups capture different aspects of paper identity
3. **Proper evaluation methodology** — stratified split, consistent test set, multiple metrics
4. **Statistical validation** — Friedman test provides rigorous assessment of performance differences
5. **Meaningful insights** — publisher features dominate, linear models win, XGBoost struggles

### 10.2 What the Results Tell Us About OA Prediction

The core finding is:

> **Predicting open-access category from paper metadata is feasible (57.48% accuracy vs 33% random baseline) but limited by the inherent ambiguity of the `partially_open` class and the metadata-only feature set.**

The 57% accuracy figure should be interpreted in context:
- It represents a **24 percentage point lift over random guessing** (33%)
- It is limited by a three-class problem where one class is ambiguously defined
- It would likely improve to 70%+ with a binary target (open vs closed) or with semantic text embeddings replacing TF-IDF

### 10.3 Comparison Across DA Phases

| Phase | Achievement | Key Metric |
|-------|-------------|-----------|
| DA1 | Data collection and EDA | 2000 papers, 27 features, SQLite |
| DA2 | ML pipeline (12 models, tuning, clustering) | Best F1 = 0.5696 (LDA) |
| DA3 | Statistical validation, interactive dashboard | Best F1 = 0.5701 (SVM-tuned), Friedman test |

### 10.4 Honest Assessment

The project achieves its goals as a **B.Tech Data Science research project**:
- It demonstrates competency in the full ML pipeline
- It applies proper evaluation methodology
- It produces statistically validated results
- It builds an interactive tool for result exploration
- It acknowledges limitations clearly

It is **not** production-ready for a real research platform, primarily because 57% accuracy would cause too many incorrect predictions in a real-world tool where a researcher relies on it to find accessible papers.

---

*ResearchPilot DA3 | October 2026 | All results derived from verified DA2 pipeline*
