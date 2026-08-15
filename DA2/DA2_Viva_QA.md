# DA2 Viva Q&A (≥40 questions)

**Project-specific answers for ResearchPilot.** Keep responses short in the viva; expand only if asked.

---

### Dataset & Phase 1

**1. What is ResearchPilot?**  
A Phase 1–2 foundation that collects OpenAlex AI/ML papers, stores them in SQLite, and builds evaluated ML models for OA category and relative impact tiers—plus exploratory topic clusters. Chat/RAG is future work.

**2. How many papers are in the final dataset?**  
2000 papers with 27 columns in `data/final/final_dataset.csv`.

**3. Why OpenAlex?**  
Open scholarly metadata with citation and OA fields, suitable for reproducible API collection.

**4. What is `oa_category`?**  
A three-class label: `fully_open`, `partially_open`, `closed`, derived from OpenAlex OA metadata in Phase 1.

**5. Did you modify Phase 1 files in Phase 2?**  
No. Phase 1 outputs are read-only inputs to Phase 2.

---

### Feature engineering & selection

**6. What engineered features did you create?**  
Examples: `paper_age`, lengths, keyword/concept counts, `citation_per_year`, `citation_log`, `recent_paper`, OA flags/category.

**7. How is text represented?**  
Title + abstract → TF-IDF (up to 100 features, uni/bi-grams, English stop words).

**8. What is feature selection in M3?**  
Consensus ranking (e.g., mutual information / model votes), drop low-variance/redundant features, scale using train-fit transformers.

**9. Why drop `has_doi`?**  
Near-constant (~99.9% present); almost no predictive signal.

**10. What features are excluded for OA prediction to prevent leakage?**  
OA-defining fields such as `is_open_access`, `oa_status`, `oa_url`, `oa_category`, `open_access`, `has_fulltext`.

**11. What must not be used as impact-tier input features?**  
`cited_by_count`, `citation_per_year`, `citation_log`, and `impact_tier` itself.

**12. How are impact tiers created?**  
Train-only tertiles of `citation_per_year`; thresholds ≈ 87.33 and 134.0 applied to all splits.

**13. Why fit thresholds on train only?**  
So validation/test labels are not used to choose cutpoints (avoids label-construction leakage).

**14. Are classes balanced for impact tier?**  
Approximately yes (~33% each: 669 / 657 / 674).

---

### Database & SQL

**15. Which database technology?**  
SQLite file `database/researchpilot.db`.

**16. What is the primary key of `papers`?**  
`id` (OpenAlex work ID string).

**17. How do you connect from Python?**  
`sqlite3` / `pandas.read_sql_query` after `01_load_to_db.py`.

**18. Can R query the same database?**  
Yes, via `RSQLite` against the same file path.

**19. Give one useful SQL query you ran.**  
Group by `oa_category` for counts/percentages; or top-N by `cited_by_count`; or recent papers with `publication_year >= 2023`.

**20. How many papers are in the DB after load?**  
2000 (matches the final CSV).

---

### Splits, CV, leakage

**21. What is the train/val/test split?**  
1399 / 300 / 301, stratified by `oa_category`.

**22. What is used for model selection?**  
Validation macro-F1 (not test).

**23. When is the test set used?**  
Once, for final reporting after selection.

**24. What is cross-validation used for here?**  
Reporting/train-side estimation (M4 CV; M5 RandomizedSearchCV), not for peeking at test.

**25. Define data leakage in one sentence.**  
Using information that improperly reveals the target or future knowledge, producing over-optimistic scores.

---

### Algorithms

**26. How many distinct ML algorithms in M4?**  
Fourteen (Dummy through MLP, including XGBoost and LightGBM).

**27. Why include a Dummy classifier?**  
Baseline sanity check; shows whether real models beat naive majority/most-frequent behavior.

**28. Why AdaBoost did well on OA?**  
It combined weak learners effectively on this noisy multiclass problem; empirically best test Acc/F1 among M4 defaults.

**29. What is the difference between Random Forest and Extra Trees?**  
Extra Trees uses stronger randomization in splits; both are bagged trees—Extra Trees was strong on ROC after tuning.

**30. Why keep XGBoost if it failed once on CUDA?**  
It is a required modern booster in the inventory; we forced a CPU-safe config so it runs reliably on Windows.

**31. Is MLP deep learning?**  
It is a neural net (multi-layer perceptron). We treat it as the DL-style model in the comparative set; we did not train large transformers.

---

### Hyperparameter tuning

**32. How did you tune models?**  
`RandomizedSearchCV`, 20 iterations, 3-fold CV on train, scoring macro-F1.

**33. Which models were tuned?**  
AdaBoost, Extra Trees, Gradient Boosting, Logistic Regression, Random Forest.

**34. Did tuning always improve test F1?**  
No. M5 Extra Trees won validation but did not beat M4 AdaBoost on test macro-F1.

**35. Why still report M5?**  
Shows proper optimization protocol and that validation leaders can differ from test leaders.

---

### Metrics & evaluation

**36. Why not trust accuracy alone?**  
Multiclass imbalance/difficulty; accuracy can hide weak minority-class performance. We also use macro-F1, ROC-AUC, AP.

**37. What is macro-F1?**  
Unweighted mean of per-class F1; treats each OA/impact class equally.

**38. What is ROC-AUC (OvR) here?**  
One-vs-rest ranking quality averaged across classes using predicted probabilities.

**39. What is Average Precision?**  
Area-style summary of the precision–recall curve; useful when positive class ranking matters.

**40. How do you read a confusion matrix?**  
Rows true classes, columns predictions; adjacent-tier confusions are expected for ordinal impact labels.

**41. Best OA accuracy / macro-F1?**  
m4_adaboost ≈ **0.482** / **0.452**.

**42. Best OA ROC-AUC?**  
m5_extra_trees_tuned ≈ **0.628**.

**43. Why can different metrics pick different winners?**  
Accuracy/F1 emphasize hard class decisions; ROC/AP emphasize ranking/calibration of probabilities.

**44. Is 0.45 F1 “highly accurate”?**  
No. We call it moderate performance for a difficult three-class task.

---

### Feature importance

**45. Does high feature importance mean causation?**  
No. It means the feature contributed strongly to the model’s predictions under this fit.

**46. What dominated AdaBoost impact importance?**  
Engineered temporal/numerical features such as `paper_age`, `recent_paper`, `publication_year`, with additional TF-IDF terms.

---

### Impact tier & clustering

**47. M7 impact champion and test metrics?**  
AdaBoost; test Acc ≈ 0.551, macro-F1 ≈ 0.543, ROC-AUC ≈ 0.680.

**48. Why is impact F1 higher than OA F1?**  
Relative citation tiers may align more with age/text patterns in this corpus than OA status does—still only partially separable.

**49. What clustering method and representation?**  
KMeans on TF-IDF features.

**50. Best k and silhouette?**  
k=8, silhouette≈0.064.

**51. Is clustering successful?**  
It is exploratory: low silhouette means substantial overlap; useful as supporting neighborhood structure, not crisp topics.

**52. How could clusters help ResearchPilot later?**  
Similar-paper discovery, topic browsing, and research-gap ideation on top of better embeddings later.

---

### Limitations & process

**53. Main limitations?**  
2k AI/ML sample; moderate OA predictability; relative impact labels; time-dependent citations; weak clusters; TF-IDF limits; no chat assistant yet.

**54. What would you upgrade first in Phase 3?**  
Semantic embeddings + retrieval (RAG), keeping the same data and evaluation discipline.

**55. How do you reproduce M7?**  
`python scripts/phase2/07_impact_and_clustering.py` with M3 impact matrices present.

**56. Where are faculty docs?**  
`DA2/` pack—final report, demo script, this viva file, checklist, reproducibility guide.
