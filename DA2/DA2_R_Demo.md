# DA2 Live Faculty Demo — ResearchPilot (R Implementation)
**Duration**: 7–10 minutes | **Language**: R | **Database**: SQLite

---

## Pre-Demo Setup (do before faculty arrives)
```r
# In RStudio console or terminal:
setwd("E:/Tuned_Research")
source("r/scripts/phase2/03_database_setup.R")   # builds the DB once
# Then run all pipeline scripts in order (already run — outputs exist)
```
Open RStudio → File → Open Project → navigate to `E:/Tuned_Research`

---

## STEP 1 — Load the Dataset (30 seconds)

**WHAT I RUN:**
```r
library(readr)
df <- read_csv("data/final/final_dataset.csv")
dim(df)        # 2000 rows, 27 columns
head(df, 3)
table(df$oa_category)
```

**WHAT I SHOW:** 2000 rows, 27 columns. Three OA classes: fully_open (834), partially_open (723), closed (443).

**WHAT I SAY:**
> "This is our ResearchPilot corpus — 2000 AI and machine learning papers collected from OpenAlex. Each paper has metadata like publication year, citation count, keywords, and open-access status. The primary classification target is oa_category with three classes."

**LIKELY FACULTY QUESTION:** Where did this data come from?

**SHORT ANSWER:** OpenAlex is a free scholarly metadata API. We collected papers via keyword search for AI/ML topics, published 2022–2025, and preprocessed them in Phase 1.

---

## STEP 2 — Feature Engineering (45 seconds)

**WHAT I RUN:**
```r
eng <- read_csv("data/ml_r/engineered_features.csv")
dim(eng)  # 2000 x 38 — 11 new features added

# Show the new features
new_feats <- c("title_word_count","abstract_word_count","title_to_abstract_ratio",
               "text_richness","recency_score","text_length_category",
               "abstract_keyword_overlap","publication_year_norm")
eng[1:5, new_feats]
```

**WHAT I SHOW:** The 11 new engineered features. Point at `text_richness` and `recency_score` as good examples of purpose-built features.

**WHAT I SAY:**
> "Phase 1 gave us 27 columns. For DA2 we engineered 11 additional features. For example, text_richness measures how many keywords and concepts a paper has per word of abstract — a proxy for how well-structured the paper's metadata is. recency_score normalises paper age to a 0–1 scale."

**LIKELY FACULTY QUESTION:** Why not just use the original features?

**SHORT ANSWER:** Some original features are character counts. Word counts are more semantically meaningful. We also needed scale-normalised versions for distance-based models like KNN and SVM.

---

## STEP 3 — Feature Selection (30 seconds)

**WHAT I RUN:**
```r
sel_meta <- read_csv("data/ml_r/feature_selection_metadata.csv")
sel_meta[, c("feature","pass_mi","pass_rf","votes","mi_score","selected")]
```

**WHAT I SHOW:** The feature selection table. Point at the `mi_score` column and which features were selected.

**WHAT I SAY:**
> "We applied four selection methods: near-zero variance filter, correlation filter, mutual information, and random forest importance. Three features were removed by NZV — has_doi was nearly constant, and keyword_diversity was always 1.0 because OpenAlex deduplicates assigned terms. Five more were removed by correlation. Final selected set: 9 features."

**LIKELY FACULTY QUESTION:** Why remove correlated features?

**SHORT ANSWER:** High correlation causes multicollinearity in linear models like logistic regression, and it inflates KNN distances. It also makes the model less interpretable. We keep the more informative of any correlated pair.

---

## STEP 4 — Connect to SQLite Database (60 seconds)

**WHAT I RUN:**
```r
library(DBI)
library(RSQLite)

# Connect
con <- dbConnect(RSQLite::SQLite(), "database/researchpilot_r.db")
cat("Connected!\n")

# Show tables
dbListTables(con)

# Row counts
dbGetQuery(con, "SELECT 'papers' AS tbl, COUNT(*) AS n FROM papers
           UNION ALL SELECT 'ml_features', COUNT(*) FROM ml_features
           UNION ALL SELECT 'oa_features', COUNT(*) FROM oa_features")
```

**WHAT I SHOW:** Connection success, table names (papers, ml_features, oa_features, model_results), row counts.

**WHAT I SAY:**
> "This is a SQLite database created entirely in R using the DBI and RSQLite packages. It has four tables — papers holds the core metadata, ml_features holds the engineered numeric features, oa_features holds the selected feature set for ML, and model_results stores our evaluation metrics. Everything goes through R's DBI interface."

**LIKELY FACULTY QUESTION:** Why SQLite and not PostgreSQL?

**SHORT ANSWER:** SQLite is serverless, portable, and fully reproducible — the entire database is a single file. For a research demonstration it's ideal because anyone can run it without installing a database server.

---

## STEP 5 — Run SQL Queries (60 seconds)

**WHAT I RUN:**
```r
# Query 1: OA category distribution
dbGetQuery(con, "
  SELECT oa_category, COUNT(*) AS n,
         ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct,
         ROUND(AVG(cited_by_count), 1) AS avg_citations
  FROM papers
  GROUP BY oa_category ORDER BY n DESC
")

# Query 2: Papers per year
dbGetQuery(con, "
  SELECT publication_year, COUNT(*) AS papers,
         SUM(is_open_access) AS open_access_papers
  FROM papers GROUP BY publication_year ORDER BY publication_year
")

# Query 3: JOIN — average ML features by OA category
dbGetQuery(con, "
  SELECT p.oa_category, COUNT(*) AS n,
         ROUND(AVG(m.title_word_count), 1) AS avg_title_words,
         ROUND(AVG(m.text_richness), 4) AS avg_richness
  FROM papers p
  JOIN ml_features m ON p.paper_id = m.paper_id
  GROUP BY p.oa_category
")

dbDisconnect(con)
```

**WHAT I SHOW:** Three query results printed as R data frames. The JOIN query is the most impressive.

**WHAT I SAY:**
> "This is the R to DBI to SQLite to SQL to data frame pipeline. The result comes back as a native R data frame which we can immediately use for analysis or visualisation. The JOIN query combines paper metadata with ML features in a single SQL statement."

**LIKELY FACULTY QUESTION:** How is this different from just using the CSV?

**SHORT ANSWER:** With a database we can express complex multi-table queries in SQL, add new results tables without reloading everything, and it scales to much larger datasets. It also demonstrates proper data engineering practice — separating storage from computation.

---

## STEP 6 — Show Model List (30 seconds)

**WHAT I RUN:**
```r
lb <- read_csv("reports/tables/r_model_leaderboard.csv")
cat(sprintf("Models trained: %d\n", nrow(lb)))
lb[, c("rank","model","model_type","acc","f1","roc")]
```

**WHAT I SHOW:** The full leaderboard. Count the models — 13+ algorithms.

**WHAT I SAY:**
> "We trained 13 genuinely different algorithm families: Logistic Regression, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost, SVM with RBF kernel, SVM with linear kernel, Naive Bayes, LDA, KNN, and MLP Neural Network. These cover linear, tree-based, kernel-based, probabilistic, and neural network approaches."

---

## STEP 7 — Hyperparameter Tuning (30 seconds)

**WHAT I RUN:**
```r
tuning <- read_csv("data/ml_r/tuning_results.csv")
tuning %>% filter(!is.na(accuracy)) %>%
  select(model, best_params, cv_f1_macro, accuracy, f1_macro)
```

**WHAT I SHOW:** Tuning results comparing baseline vs tuned.

**WHAT I SAY:**
> "We tuned three strong candidates — Random Forest, XGBoost, and SVM — using 5-fold cross-validation on the training set only. The test set was never touched during tuning. Grid search explored combinations of ntree/mtry for RF, max_depth/eta/nrounds for XGBoost, and cost/gamma for SVM."

**LIKELY FACULTY QUESTION:** Did tuning improve results?

**SHORT ANSWER:** Results are shown in the leaderboard. Tuning typically gives modest improvement on a 9-feature set. The comparison is honest — we don't claim tuning always helps.

---

## STEP 8 — Show Model Leaderboard (30 seconds)

**WHAT I RUN:**
```r
lb <- read_csv("reports/tables/r_model_leaderboard.csv")
# Top 5 by F1
head(lb[order(-lb$f1),], 5)
```

**WHAT I SHOW:** Top 5 models. Point out that different metrics have different leaders.

**WHAT I SAY:**
> "Our leaderboard reports Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and ROC-AUC. These often have different leaders. Accuracy alone is misleading for imbalanced classes — we have 834 fully_open vs 443 closed. The Python M4 baseline was AdaBoost with accuracy 0.482 and macro-F1 0.452. Our R results are in the same range."

---

## STEP 9 — Show Confusion Matrix (30 seconds)

**WHAT I SHOW:** Open `reports/figures/phase2_r/06a_confusion_matrix_best_model.png`

**WHAT I SAY:**
> "The confusion matrix reveals where the model struggles. The biggest confusion is between fully_open and partially_open. This makes sense — both are open-access papers, and the textual and metadata features don't cleanly distinguish them. This was also the dominant error pattern in the Python baseline."

**LIKELY FACULTY QUESTION:** Why is partially_open vs fully_open confused?

**SHORT ANSWER:** Our features are title length, word count, text richness, recency, keyword count — none of these directly capture the legal/licensing differences that distinguish fully open from partially open access. We deliberately excluded the OA metadata columns to avoid leakage.

---

## STEP 10 — Show ROC Curve (30 seconds)

**WHAT I SHOW:** Open `reports/figures/phase2_r/03_roc_curves_ovr.png`

**WHAT I SAY:**
> "These are one-vs-rest ROC curves for the top 5 models. The closed class has the cleanest separation — AUC around 0.65–0.70 — because closed papers have distinct structural characteristics. The partially_open vs rest is the hardest boundary."

---

## STEP 11 — Show Feature Importance (30 seconds)

**WHAT I SHOW:** Open `reports/figures/phase2_r/07_feature_importance_rf.png`

**WHAT I SAY:**
> "Random Forest feature importance shows title_word_count, text_richness, and abstract_word_count as the top contributors to OA category predictions. Importantly, we say 'contributed to model predictions' — not 'caused' the OA status. These are correlation patterns, not causal mechanisms."

---

## STEP 12 — Impact-Tier Result (30 seconds)

**WHAT I RUN:**
```r
impact <- read_csv("reports/tables/r_impact_leaderboard.csv")
impact[, c("model","accuracy","f1_macro","roc_auc")]
```

**WHAT I SAY:**
> "The secondary task is impact-tier classification: low, medium, high citation impact. Thresholds are calculated from training data only — no leakage. Our R AdaBoost result should be close to the Python baseline of test-F1=0.543, test-accuracy=0.551. Any difference is due to random seed and split implementation differences between R and Python."

---

## STEP 13 — Topic Clustering (30 seconds)

**WHAT I SHOW:** Open `reports/figures/phase2_r/clustering_silhouette_vs_k.png` and `clustering_pca_scatter.png`

**WHAT I SAY:**
> "For topic clustering we used TF-IDF on title plus abstract, then KMeans. We tested k=3 to k=10 using silhouette score. The best k is around 8, silhouette around 0.064, consistent with the Python baseline. This is a low silhouette — deliberately not over-claimed. AI/ML papers share a lot of vocabulary and concepts overlap substantially across clusters."

**LIKELY FACULTY QUESTION:** Why is the silhouette so low?

**SHORT ANSWER:** Because this is a semantically rich corpus where most papers discuss similar concepts — neural networks, deep learning, machine learning. Clear separation would only emerge if we had papers from completely different fields. The clustering is exploratory, not definitional.

---

## STEP 14 — Show Architecture / Explain ResearchPilot Direction (60 seconds)

**WHAT I SHOW:** Open `DA2/figures/final/researchpilot_final_architecture.png` (if available), or the DA2 final report.

**WHAT I SAY:**
> "ResearchPilot is building toward an AI research assistant. Phase 1 collected and cleaned 2000 papers. Phase 2 — which is DA2 — built the classification and analysis layer: we can predict a new paper's open-access status, estimate its impact tier, and cluster it into a research topic group. Phase 3 will add language model capabilities — RAG for question answering over the corpus and fine-tuning for research-specific tasks. The SQLite database built in DA2 provides the structured data store for the Phase 3 retrieval system."

---

## Key Numbers to Remember
| Metric | Value |
|--------|-------|
| Dataset size | 2000 papers |
| Original features | 27 |
| Engineered features | +11 (38 total) |
| Selected OA features | 9 |
| ML algorithms | 13 |
| Python baseline accuracy | 0.482 (AdaBoost) |
| Python baseline Macro-F1 | 0.452 (AdaBoost) |
| Python baseline ROC-AUC | 0.628 (Extra Trees tuned) |
| Impact-tier test F1 | 0.543 (AdaBoost, Python) |
| Clustering best k | 8 |
| Clustering silhouette | 0.064 |
| Database tables | 4 |
| SQL queries demonstrated | 7 |
