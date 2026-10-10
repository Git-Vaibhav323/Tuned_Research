# DA3 Viva Q&A Preparation
## ResearchPilot — Final Evaluation

**50+ questions across 8 categories**  
All answers are grounded in the actual implementation. Nothing is fabricated.

---

## CATEGORY A: DATASET AND PREPROCESSING

**Q1. Where does your data come from?**  
The OpenAlex API — a free, open scholarly database. We collected 2,000 AI/ML research papers with 27 metadata fields including titles, abstracts, DOIs, citation counts, publication years, and open-access status.

**Q2. Why only 2,000 papers?**  
Balanced project scope. The OpenAlex API has rate limits, and 2,000 papers is sufficient to demonstrate a full ML pipeline for a B.Tech project. A larger dataset (10,000+) is listed as future scope.

**Q3. What are the three target classes?**  
`fully_open` (freely accessible everywhere), `partially_open` (free on some platforms but not all), and `closed` (paywalled). These come directly from OpenAlex's OA status categorisation.

**Q4. What is the class distribution?**  
fully_open: 834 (41.7%), partially_open: 723 (36.2%), closed: 443 (22.2%). The dataset is moderately imbalanced with `closed` being the minority class.

**Q5. How was the data split?**  
Stratified 70/15/15 split: ~1,400 training, ~299 validation, 301 test. Stratified means each split preserves the class proportions. `set.seed(42)` ensures reproducibility.

**Q6. How did you handle missing values?**  
Features with near-zero variance (constant or near-constant) were removed in the NZV filtering step. Papers with missing abstracts would have zero TF-IDF values — their structural features remain valid.

**Q7. Why stratified splitting specifically?**  
With imbalanced classes, random splitting might put all `closed` papers in training and none in test (or vice versa). Stratified splitting ensures each split has proportional representation, making test evaluation fair and reproducible.

---

## CATEGORY B: FEATURE ENGINEERING

**Q8. How did you go from 27 to 206 features?**  
Four feature groups were engineered:
1. Structural/text-length (12): `abstract_length`, `title_length`, `text_richness`, `recency_score`, etc.
2. Domain binary indicators (25): `dom_medical`, `dom_llm`, `dom_education`, etc.
3. Publisher one-hot encoding (18): `pub_ieee`, `pub_mdpi`, `pub_springer`, etc.
4. TF-IDF terms (~150): Top-200 TF-IDF terms from abstracts after stemming and stopword removal

Total: 27 raw → 206 engineered.

**Q9. How did you go from 206 to 80 features?**  
Three sequential steps, all fitted on training data only:
1. Near-Zero Variance filter — removed constant/near-constant columns
2. Pearson correlation filter — within TF-IDF group, removed one column from pairs with |r| > 0.95
3. Random Forest Gini importance ranking — ranked remaining features; top 80 retained

**Q10. What is near-zero variance and why is it a problem?**  
A feature with near-zero variance has almost the same value for every sample. It provides no discriminative information and can cause numerical instability in some algorithms. Example: a domain flag for a topic with only 2 papers in the corpus.

**Q11. Why TF-IDF and not word embeddings?**  
TF-IDF is computationally efficient, fully interpretable (each feature is a specific word), reproducible without API keys or GPU, and appropriate for this project's scope. Semantic embeddings (SciBERT, Sentence-BERT) would likely improve performance and are listed as future scope.

**Q12. What is data leakage and how did you prevent it?**  
Data leakage is when information from the test or validation set influences model training or feature selection, producing artificially inflated performance. We prevented it by fitting all transformations (scaling, NZV, correlation filter, RF importance) on the training split only. The test set was never seen until the final evaluation.

**Q13. What does `text_richness` measure?**  
The ratio of unique words to total words in the abstract — a measure of vocabulary diversity. Papers with high text richness tend to be more specialized academic writing, which may correlate with publication type.

**Q14. Why one-hot encode publishers instead of using raw publisher names?**  
Most ML algorithms require numerical input. One-hot encoding converts categorical publisher names into binary features, allowing the model to learn OA-specific effects for each publisher. For example, MDPI has a very different OA policy from Elsevier, and one-hot encoding lets the model learn this.

---

## CATEGORY C: MACHINE LEARNING MODELS

**Q15. Why 12 models?**  
To compare a broad range of algorithm types: linear (LDA, LR, Elastic Net), tree-based (Decision Tree, RF, XGBoost, GB, AdaBoost), kernel-based (SVM-RBF, SVM-Linear), probabilistic (Naive Bayes), and neural (MLP). Different algorithm families have different inductive biases, and comparing them reveals which biases are most appropriate for this task.

**Q16. What was the best model and why?**  
Best tuned: SVM-RBF (F1=0.5701). Best baseline: LDA (F1=0.5696). LDA works because the engineered feature space is approximately linearly separable — publisher binary flags and structural features form relatively clean linear boundaries. SVM-RBF adds a small kernel-based non-linearity advantage.

**Q17. Why did XGBoost perform so poorly (F1=0.3428)?**  
Three reasons: (1) Dataset size — XGBoost is designed for large tabular datasets; ~1,400 training samples is insufficient. (2) TF-IDF sparsity — sparse features can cause unstable tree splits. (3) Default hyperparameters were poorly matched — tuning improved it to 0.3537 but it remained the weakest.

**Q18. What is Macro-F1 and why use it?**  
Macro-F1 is the unweighted average of per-class F1 scores. It treats all classes equally regardless of size — the `closed` class (22%) is weighted the same as `fully_open` (42%). This is appropriate when we care about performance on minority classes, not just overall accuracy. Accuracy can be misleading in imbalanced settings.

**Q19. What is the difference between precision, recall, and F1?**  
Precision = of all predictions of class X, how many were actually class X (correctness of positive predictions). Recall = of all actual class X instances, how many did we correctly predict (completeness). F1 = harmonic mean of precision and recall — balances both. F1 is preferred when false positives and false negatives are both costly.

**Q20. What is ROC-AUC?**  
Area Under the Receiver Operating Characteristic curve. For multi-class problems, we use One-vs-Rest macro AUC. It measures discrimination ability — how well the model separates each class from the rest — at all possible thresholds. AUC=0.5 means no better than random; AUC=1.0 is perfect. Our best model (LDA) has AUC=0.7577, indicating good discrimination.

**Q21. Why is accuracy not the primary metric?**  
With imbalanced classes, a model that always predicts `fully_open` (42% of papers) would get 41.7% accuracy with zero practical utility. Macro-F1 penalises poor performance on minority classes and gives a more honest picture of model quality.

**Q22. What is the confusion matrix for the best model?**  
For LDA: The model performs well on `fully_open` (highest recall ~0.74) but struggles with `partially_open` — it gets confused with both other classes. `closed` has moderate recall (~0.51). `partially_open` is the hardest class due to its ambiguous definition.

---

## CATEGORY D: HYPERPARAMETER TUNING

**Q23. Which models were tuned and why those three?**  
Random Forest, SVM-RBF, and XGBoost. These three are the most hyperparameter-sensitive models in our set. LDA has no meaningful hyperparameters to tune. Logistic Regression was already well-performing with default regularisation. Decision Tree, AdaBoost, etc. had weaker baseline performance not worth tuning.

**Q24. What tuning method was used?**  
5-fold cross-validation grid search on the training set only. For each hyperparameter combination, we compute CV Macro-F1 and select the configuration with the best CV score. The selected model is then evaluated once on the untouched test set.

**Q25. Why not use the validation set for tuning?**  
We do use the validation set for SVM and RF as a proxy check, but the primary tuning criterion is CV F1 on training folds. Using the test set for tuning would constitute data leakage — the model would be selected to fit the test set, making the final evaluation misleading.

**Q26. What were the best hyperparameters for SVM-RBF?**  
cost=10, gamma=0.001. `cost=10` means a higher penalty for misclassification (tighter margin). `gamma=0.001` is a wide kernel (small gamma = wide influence = smoother boundary). This combination avoids overfitting while capturing the mild non-linearity in the data.

**Q27. Why was the tuning improvement so small (+0.0005 over best baseline)?**  
LDA already found the near-optimal linear decision boundary. The feature space is approximately linearly separable, so a kernel SVM with optimal parameters matches but cannot substantially exceed LDA's performance. The diminishing returns reflect that the features, not the algorithm choice, are the primary performance bottleneck.

**Q28. Which model had the largest tuning improvement?**  
Random Forest: +0.0158 F1 (from 0.5317 baseline to 0.5475 tuned). The improvement came from `ntree=500` (vs default 300 trees) and `mtry=8` (vs default ~9 for 80 features, which was close). More trees reduced variance in the ensemble.

---

## CATEGORY E: STATISTICAL ANALYSIS

**Q29. What statistical test did you use and why?**  
The Friedman test — a non-parametric test for comparing multiple treatments (models) across repeated blocks (folds). It is the correct choice because: (1) we have repeated measures (same folds for all models), (2) we cannot assume normality of F1 distributions with K=5, (3) it is widely accepted for ML model comparison in the literature (Demšar, 2006).

**Q30. What are your null and alternative hypotheses?**  
H₀: All 12 models have equal median Macro-F1 across folds — any differences are due to random variation.  
H₁: At least one model has a different median Macro-F1.  
Significance level α = 0.05.

**Q31. What does p-value mean here?**  
The probability of observing the test statistic (or more extreme) if H₀ were true. If p < 0.05, we have strong evidence against H₀ and conclude at least one model differs significantly.

**Q32. What is the post-hoc test and when is it needed?**  
The Nemenyi test, applied only if the Friedman test rejects H₀. The Friedman test only tells us "at least one model differs" — it doesn't say which ones. Nemenyi performs all pairwise comparisons and controls for multiple comparison errors (Type I error inflation).

**Q33. What if the p-value is > 0.05 — does that mean all models are equal?**  
No. Failing to reject H₀ means we lack sufficient statistical evidence to conclude differences, NOT that models are identical. With K=5 folds, statistical power is limited. The practical differences (e.g., LDA F1=0.5696 vs XGBoost F1=0.3428) are still real and meaningful even if not statistically significant at K=5.

**Q34. Why K=5 folds instead of K=10?**  
Computational cost. With 12 models × 5 folds, every model must be retrained 5 times. K=10 would double the compute time with only marginal power improvement. K=5 is standard and defensible.

**Q35. What is the Friedman test statistic?**  
The Friedman statistic follows a chi-squared distribution with degrees of freedom = k-1 = 11 (for 12 models). A larger statistic means more evidence against H₀. The exact value is in `DA3/tables/friedman_results.csv` after running `11_statistical_analysis.R`.

**Q36. What is the difference between statistical significance and practical significance?**  
Statistical significance (p < 0.05) means the result is unlikely due to chance. Practical significance means the difference is large enough to matter in real applications. Example: SVM-RBF Tuned vs LDA — ΔF1 = +0.0005. Even if this were statistically significant, it is practically negligible. We always discuss both.

---

## CATEGORY F: DASHBOARD

**Q37. What technology did you use for the dashboard?**  
R Shiny for the reactive web application framework and Plotly for interactive charts. The combination provides hover tooltips, zoom/pan, and dynamic chart updates without any custom JavaScript.

**Q38. Why R Shiny instead of Python Dash or Streamlit?**  
The entire DA2 pipeline is in R. Shiny integrates seamlessly with existing R objects (RDS files, data frames). Using Shiny avoids introducing a new language and its dependency chain.

**Q39. How is the dashboard interactive (not just static)?**  
Shiny's reactive programming model: when the user changes the metric selector in Tab 2, a reactive expression automatically recomputes and re-renders the Plotly chart. The chart updates in real time without page reload. Every Plotly chart also has built-in hover tooltips, zoom, and pan.

**Q40. What are the four dashboard tabs?**  
Tab 1: Overview — KPI cards (2000 papers, 80 features, 12 models, 301 test size), OA distribution chart, feature engineering pipeline visual.  
Tab 2: Model Comparison — metric selector dropdown updates bar chart dynamically; table below.  
Tab 3: Model Diagnostics — model selector shows confusion matrix, ROC curves, feature importance.  
Tab 4: Research Clusters — pie chart of cluster sizes, dominant terms bar chart, cluster summary table.

**Q41. Where does the dashboard data come from?**  
Directly from existing project files in `data/ml_r/`: `model_metrics_v2.csv`, `all_results_v2.csv`, `cluster_keywords.csv`, `cluster_assignments.csv`, and `data/final/final_dataset.csv`. No data is hardcoded — everything is loaded from the actual pipeline outputs.

**Q42. How do you run the dashboard?**  
`setwd("e:/Tuned_Research"); shiny::runApp("dashboard")` in R. Required packages: shiny, plotly, dplyr, tidyr, readr, DT.

**Q43. Can the dashboard be deployed online?**  
Yes — it could be deployed to shinyapps.io (free tier) with one command: `rsconnect::deployApp("dashboard")`. Deployment requires an account and the app code uploaded. This is not done in DA3 but is straightforward.

---

## CATEGORY G: RESULTS INTERPRETATION

**Q44. Why do publisher features dominate the feature importance ranking?**  
Open-access status is primarily a publisher policy decision. MDPI publishes almost entirely open-access under its APC model. IEEE publishes both open and closed. The model correctly exploits these publisher-policy correlations.

**Q45. Does feature importance mean causation?**  
NO. Feature importance (Random Forest Gini) indicates which features were most predictively useful for splitting — it does not imply causal relationships. The correct statement is: "publisher type is the most predictively useful feature." The incorrect statement would be: "being published by MDPI causes a paper to be open-access." The causation runs the other way.

**Q46. What do the research clusters represent?**  
Topic/thematic groupings within AI/ML research, identified by K-Means on TF-IDF. Cluster 1: Education/GenAI (10%), Cluster 2: General ML/Robotics (71%), Cluster 3: Medical Imaging/DL (19%). These reflect the three major subfields present in the corpus.

**Q47. Why is the silhouette score only ~0.06?**  
This is expected for AI/ML literature. Research topics are interconnected — medical imaging uses deep learning, robotics uses ML, and every paper uses "model" and "learn." K-Means assumes spherical clusters, but real topic distributions overlap significantly. The low silhouette means weak separation, not that the clusters are meaningless.

**Q48. Are the cluster labels generated by the algorithm?**  
No. K-Means generates numeric cluster assignments (1, 2, 3). A human (the researcher) looked at the top TF-IDF terms and assigned interpretive labels. This is standard practice in unsupervised learning and is explicitly disclosed in the report.

**Q49. Why is 57% accuracy acceptable for a B.Tech project?**  
Context: (1) Random baseline is 33% (3 equally probable classes), so 57% represents a 73% lift over random. (2) The task involves a genuinely ambiguous middle class (`partially_open`). (3) The 0.7577 ROC-AUC shows good probability calibration even when threshold-based accuracy is moderate. (4) The project demonstrates a complete, reproducible, statistically validated pipeline — accuracy is one metric, not the only measure of success.

**Q50. What is the best next step to improve performance?**  
Two options with high expected impact: (1) Collapse to binary classification (open vs closed) — removes the ambiguous middle class; expected accuracy 70%+. (2) Replace TF-IDF with SciBERT semantic embeddings — captures semantic similarity; expected F1 improvement +5–10%. Both are listed as future scope.

---

## CATEGORY H: TECHNICAL AND REPRODUCIBILITY

**Q51. How do you ensure reproducibility?**  
`set.seed(42)` in all scripts. Saved train/val/test split indices in `ml_data_splits_v2.rds`. Saved model objects in `model_objects_v2.rds`. Documented package versions via `sessionInfo()`. Provided `REPRODUCIBILITY.md` with step-by-step run instructions. Any researcher with the same software versions can reproduce all results to 4 decimal places.

**Q52. What database do you use and what tables does it have?**  
SQLite database at `database/researchpilot_r.db`. Four tables: `papers` (raw paper metadata), `ml_features` (80 selected features), `oa_features` (OA-specific features), `model_results` (stored model metrics). Queried in R using the RSQLite package.

**Q53. What is the difference between validation and test set?**  
Validation set (15%, n≈299): used during hyperparameter tuning to evaluate candidate parameter combinations. Test set (15%, n=301): used ONCE at the very end for final reported performance. The validation set helps select hyperparameters without touching the test set. The test set gives the unbiased final estimate of performance.

**Q54. What R packages does the project rely on?**  
Core: randomForest, e1071 (SVM + Naive Bayes), glmnet (Elastic Net), xgboost, gbm (Gradient Boosting), MASS (LDA), nnet (MLP, Logistic Regression), RSQLite, ggplot2, dplyr, readr. Dashboard: shiny, plotly, DT. Statistical: PMCMRplus (Nemenyi test). All available on CRAN.

**Q55. How long does the full pipeline take to run?**  
Full retraining (feature engineering + all models + tuning + evaluation): approximately 30–45 minutes on a standard laptop. Dashboard launch: under 10 seconds. Statistical analysis (11_statistical_analysis.R): 10–20 minutes (due to 5-fold CV × 12 models).

**Q56. Is the LLM/Ollama feature implemented?**  
No. Local LLM integration via Ollama is future scope. It is planned after DA3 completion. Hardware inspection, model selection, and integration design will happen post-submission. This is explicitly documented in the limitations and future scope sections.

**Q57. What would you change if you had more time?**  
In priority order: (1) Use SciBERT embeddings instead of TF-IDF, (2) Add binary OA prediction (open vs closed) as a simpler parallel task, (3) Implement RAG pipeline for paper Q&A, (4) Expand corpus to 10,000+ multi-domain papers, (5) Integrate Ollama for local LLM inference.

---

## VIVA PREPARATION TIPS

1. **Know your numbers.** LDA F1=0.5696, SVM-tuned F1=0.5701, test n=301, classes: 67/125/109, clusters: 200/1422/378.

2. **Know the distinction between CV and test set.** CV F1 was used for hyperparameter selection. Test F1 is the final reported score. These are different numbers and that is expected.

3. **Be honest about limitations.** The 57% accuracy is moderate and you know why. The p-value result (whatever it is) is correct. Don't oversell.

4. **Feature importance ≠ causation.** Repeat this if asked anything about feature importance interpretation.

5. **Cluster labels are human-interpreted.** The algorithm produced numbers; you assigned meaning.

6. **Dashboard interactivity.** Be ready to demonstrate live. Know which tab to open first and what to click to show interactivity.

7. **Statistical test rationale.** Explain: repeated measures + non-parametric → Friedman. Not ANOVA (normality assumption). Not Kruskal-Wallis (independent samples assumption).

8. **Future scope is NOT implemented.** RAG, Ollama, fine-tuning — if asked, say clearly: "This is future scope. It is not in the current codebase."

---

*ResearchPilot DA3 | October 2026 | 57 Q&A pairs*
