# DA2 Live Faculty Demo — ResearchPilot (R Enhanced Pipeline v2)
**Duration**: 8–10 minutes | **Language**: R | **Database**: SQLite
**Best Result**: 57.5% accuracy (+13% over original) | **Features**: 80 enriched

---

## Pre-Demo Checklist (do BEFORE faculty arrives)
```r
setwd("E:/Tuned_Research")
# Verify all outputs exist
file.exists("database/researchpilot_r.db")          # TRUE
file.exists("data/ml_r/enriched_features_v2.csv")   # TRUE
file.exists("data/ml_r/model_objects_v2.rds")       # TRUE
file.exists("reports/tables/r_model_leaderboard_v2.csv") # TRUE
```
Open RStudio → set working directory to `E:/Tuned_Research`

---

## STEP 1 — Load the Dataset (30 sec)

**WHAT I RUN:**
```r
library(readr)
df <- read_csv("data/final/final_dataset.csv")
dim(df)                         # 2000 rows, 27 columns
table(df$oa_category)           # closed:443  fully_open:834  partially_open:723
```

**WHAT I SAY:**
> "This is our ResearchPilot corpus — 2000 AI and ML papers from OpenAlex.
> The primary classification target is oa_category: is a paper fully open,
> partially open, or closed access? We have 834 fully open, 723 partially open,
> and 443 closed papers."

**LIKELY FACULTY QUESTION:** What is the baseline accuracy?

**SHORT ANSWER:** A naive classifier always predicting the majority class (fully_open) would get 41.7%. Our v1 model got 44.5%. After adding enriched features, we reach 57.5% — a 13% absolute improvement.

---

## STEP 2 — Feature Engineering (45 sec)

**WHAT I RUN:**
```r
eng <- read_csv("data/ml_r/enriched_features_v2.csv")
dim(eng)    # 2000 × 206
names(eng)[1:30]   # show first 30 feature names

# Show the key insight — publisher signals
library(dplyr)
eng %>% group_by(oa_category) %>%
  summarise(pct_ieee = mean(pub_ieee),
            pct_mdpi = mean(pub_mdpi),
            pct_medical = mean(dom_medical))
```

**WHAT I SHOW:** The table reveals that 56.9% of closed papers are IEEE-published vs only 7.8% of fully open papers. MDPI is 23.7% fully open vs 0% closed.

**WHAT I SAY:**
> "The key insight driving our v2 pipeline is that OA status is primarily
> determined by publisher policy, not paper content. IEEE papers are mostly
> closed-access. MDPI and BioMed Central are gold open-access publishers.
> We encode this through the DOI prefix — the first 7 characters of any DOI
> identify the publisher. This is NOT leakage because publisher identity is
> causally prior to the OA decision."

**LIKELY FACULTY QUESTION:** Isn't using DOI to predict OA kind of obvious?

**SHORT ANSWER:** Yes — and that's the scientific finding. The model is doing what an expert librarian would do: look up who published it. The contribution is showing quantitatively how much signal the publisher carries vs. content features.

---

## STEP 3 — Feature Selection (30 sec)

**WHAT I RUN:**
```r
meta <- read_csv("data/ml_r/enriched_feature_metadata.csv")
table(meta$group)   # tfidf=150, domain=25, publisher=18, structural=12

sel <- readRDS("data/ml_r/selected_feature_names.rds")
length(sel)    # 80 features selected
head(sel, 10)  # top features — pub_ieee, pub_mdpi, title_to_abstract_ratio...
```

**WHAT I SAY:**
> "We started with 205 engineered features across 4 groups — 150 TF-IDF terms,
> 25 domain indicators, 18 publisher signals, and 12 structural features.
> Near-zero variance filtering removed 41 (mainly zero-count features).
> Random Forest importance ranking selected the top 80.
> The top two are pub_ieee and pub_mdpi — confirming publisher is the key signal."

---

## STEP 4 — Connect to SQLite Database (45 sec)

**WHAT I RUN:**
```r
library(DBI); library(RSQLite)

con <- dbConnect(RSQLite::SQLite(), "database/researchpilot_r.db")
cat("Connected!\n")
dbListTables(con)
# [1] "ml_features" "model_results" "oa_features" "papers"

# Row counts
dbGetQuery(con, "
  SELECT 'papers' tbl, COUNT(*) n FROM papers
  UNION ALL SELECT 'ml_features', COUNT(*) FROM ml_features
  UNION ALL SELECT 'model_results', COUNT(*) FROM model_results
")
```

**WHAT I SAY:**
> "This SQLite database was created entirely in R using DBI and RSQLite.
> It has 4 tables — papers holds core metadata, ml_features holds our
> engineered features, oa_features holds the selected feature set,
> and model_results stores evaluation metrics that get updated every time
> we run the evaluation script. The whole database is a single 4.2 MB file."

---

## STEP 5 — Run SQL Queries (45 sec)

**WHAT I RUN:**
```r
# Query 1: OA distribution with stats
dbGetQuery(con, "
  SELECT oa_category,
         COUNT(*) AS papers,
         ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct,
         ROUND(AVG(cited_by_count), 0) AS avg_citations
  FROM papers GROUP BY oa_category ORDER BY papers DESC
")

# Query 2: JOIN - avg features by OA class
dbGetQuery(con, "
  SELECT p.oa_category,
         COUNT(*) AS n,
         ROUND(AVG(m.keyword_count), 1) AS avg_keywords,
         ROUND(AVG(m.text_richness), 4) AS avg_richness
  FROM papers p
  JOIN ml_features m ON p.paper_id = m.paper_id
  GROUP BY p.oa_category
")

dbDisconnect(con)
```

**WHAT I SAY:**
> "The JOIN query combines metadata from two tables using SQL. The result
> comes back as a native R data frame, ready for analysis or visualisation.
> This is the R → DBI → SQLite → SQL → data.frame pipeline."

---

## STEP 6 — Show Model List (30 sec)

**WHAT I RUN:**
```r
lb <- read_csv("reports/tables/r_model_leaderboard_v2.csv")
cat("Models:", nrow(lb), "\n")
lb[, c("rank", "model", "model_type", "acc", "f1")]
```

**WHAT I SAY:**
> "We trained 13 genuinely different algorithms across 7 families —
> linear models, tree models, boosting, SVMs, probabilistic, discriminant analysis,
> and neural networks. That gives us a comprehensive comparison. Then we tuned
> the top 3 using grid search with 5-fold cross-validation."

---

## STEP 7 — Model Leaderboard (45 sec)

**WHAT I RUN:**
```r
lb <- read_csv("reports/tables/r_model_leaderboard_v2.csv")
# Top 5 by accuracy
head(lb[order(-lb$acc), c("rank","model","model_type","acc","prec","rec","f1","roc")], 5)
```

**WHAT I SHOW:** The leaderboard. Point out:
- SVM Tuned tops with **57.5% accuracy** and **F1=0.570**
- MLP has the best **ROC-AUC=0.766**
- Different metrics have different leaders

**WHAT I SAY:**
> "Our best model is the tuned SVM with 57.5% accuracy. The original 9-feature
> pipeline only reached 44.5%. That's a +13 percentage point improvement
> purely from better feature engineering — same algorithms, same splits,
> same evaluation methodology. The best ROC-AUC is 0.766 from MLP,
> which means the model can correctly rank papers by OA probability
> in 76.6% of comparisons."

**LIKELY FACULTY QUESTION:** Why didn't you reach 70%?

**SHORT ANSWER:** The remaining ~30% error reflects genuinely ambiguous cases — papers from hybrid publishers like Springer or Elsevier where OA status depends on individual journal policies and author choices, not just publisher name. Without journal-level OA policy data (from DOAJ), this ceiling is hard to break. I documented this as a limitation and proposed it as future work.

---

## STEP 8 — Show Confusion Matrix (30 sec)

**WHAT I SHOW:** Open `reports/figures/phase2_r/v2_cm_random_forest.png`

**WHAT I SAY:**
> "The confusion matrix shows that closed papers are predicted most accurately —
> the model has learned that IEEE and ACM papers are typically closed.
> The main remaining confusion is between fully_open and partially_open.
> Both involve open-access papers but different licensing types —
> gold OA vs green/hybrid OA — and these can come from the same publishers."

---

## STEP 9 — Show ROC Curve & Feature Importance (30 sec)

**WHAT I SHOW:**
1. `reports/figures/phase2_r/v2_04_roc_auc_comparison.png`
2. `reports/figures/phase2_r/v2_07_feature_importance_enriched.png`

**WHAT I SAY on ROC:**
> "ROC-AUC above 0.75 for most models shows strong ranking ability.
> The horizontal dotted line at 0.5 represents random guessing —
> all our models significantly exceed it."

**WHAT I SAY on Importance:**
> "The two most important features are pub_ieee and pub_mdpi.
> The model has essentially learned the OA policy of major publishers.
> TF-IDF terms like 'abstract', 'model', 'data' also contribute —
> these capture domain vocabulary correlated with certain venues."

---

## STEP 10 — Old vs New Comparison Figure (20 sec)

**WHAT I SHOW:** `reports/figures/phase2_r/v2_03_old_vs_new_comparison.png`

**WHAT I SAY:**
> "This plot summarises the entire improvement. Original 9 features gave
> 44.5% accuracy and 37.8% F1. The enriched 80-feature pipeline gives
> 57.5% accuracy and 57.0% F1. The red dashed line is the 70% target —
> we're at 82% of the way there. To close that gap would require
> journal-level OA policy data, which I've proposed as Phase 3 future work."

---

## STEP 11 — Impact-Tier Result (30 sec)

**WHAT I RUN:**
```r
impact <- read_csv("reports/tables/r_impact_leaderboard_v2.csv")
impact[, c("model", "accuracy", "f1_macro", "roc_auc")]
```

**WHAT I SAY:**
> "The secondary task is predicting whether a paper is low, medium, or high
> citation impact — determined by citation_per_year tertiles calculated
> on training data only. Our thresholds match the Python baseline
> (q_low=87.28 ≈ 87.33, q_high=137.22 ≈ 134.0).
> For this task, publisher signals are less useful since citation count
> is determined after publication, not by venue alone."

---

## STEP 12 — Clustering (20 sec)

**WHAT I SHOW:** `reports/figures/phase2_r/clustering_pca_scatter.png`

**WHAT I SAY:**
> "For topic discovery we used TF-IDF on title plus abstract, then KMeans.
> We found 3 broad clusters — education/ChatGPT papers at 10%,
> general ML/AI at 71%, and computer vision/medical at 19%.
> The silhouette score of 0.42 indicates clear separation on the PCA
> projection — these clusters are meaningfully different in vocabulary."

---

## STEP 13 — ResearchPilot Direction (45 sec)

**WHAT I SAY:**
> "ResearchPilot is building toward an AI research assistant.
> Phase 1 collected and processed 2000 papers.
> Phase 2 — this DA2 — built the classification and analysis layer.
> The SQLite database stores structured metadata and model results.
> Phase 3 will add language model capabilities — RAG for question answering
> and fine-tuning for domain-specific generation.
> The feature engineering we did here — especially TF-IDF and domain indicators —
> directly feeds into the Phase 3 embedding and retrieval pipeline."

---

## Key Numbers to Have Ready

| Metric | Original (9 features) | Enhanced (80 features) |
|--------|----------------------|----------------------|
| Best Accuracy | 0.4452 | **0.5748 (+13%)** |
| Best Macro-F1 | 0.3783 | **0.5701 (+19%)** |
| Best ROC-AUC | 0.5714 | **0.7659 (+19%)** |
| Features | 9 | 80 (from 205 engineered) |
| Models trained | 11 | 13 + 3 tuned |
| Database size | 4.2 MB | 4.2 MB (same) |
| SQL queries | 7 | 7 |
| Figures | 20 | 29 (20 + 9 v2) |

---

## One-Sentence Answers to Tough Questions

**"Isn't using DOI to predict OA cheating?"**
No — publisher identity is in the original data and is causally prior to the OA decision; using it is the same as an expert knowing that IEEE charges for access.

**"Why 57% not 70%?"**
Hybrid publishers give ~30% of papers unknown OA status depending on individual journal and author choices — journal-level data would close this gap.

**"Why R instead of Python?"**
R is the language of statistical computing and academic data analysis; DBI/RSQLite, tidytext, and ggplot2 give a clean, reproducible pipeline with excellent visualization.

**"How did you prevent data leakage?"**
Excluded: is_open_access, oa_status, oa_url, open_access (direct OA metadata). Impact tier excludes cited_by_count and citation_per_year. All thresholds and scaling parameters computed on training set only.
