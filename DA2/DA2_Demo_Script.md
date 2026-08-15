# DA2 Live Demonstration Script (7–10 minutes)

**Project:** ResearchPilot  
**Goal:** Show a coherent Phase 1 + Phase 2 story without claiming Phase 3 features.

---

## 1. Project introduction — 30 sec

**WHAT TO SHOW:** Title slide / `README.md` status line / architecture PNG  
`reports/figures/final/researchpilot_final_architecture.png`

**WHAT TO SAY:**  
“ResearchPilot is a human-centered research intelligence foundation. For DA2 we completed Phase 1 data work and Phase 2 ML: database, features, fourteen algorithms for open-access prediction, tuning, comparison, plus impact-tier prediction and exploratory clustering.”

**LIKELY FACULTY QUESTION:** Is this a chatbot already?  
**ANSWER:** No. Chat/RAG is Phase 3 roadmap. What exists is the data-to-models pipeline and analysis.

---

## 2. Dataset and Phase 1 — 45 sec

**WHAT TO SHOW:** `data/final/final_dataset.csv` row count (2000×27) and one EDA figure (e.g. OA category).

**WHAT TO SAY:**  
“We collected AI/ML papers from OpenAlex, cleaned them, engineered citation and OA fields, and locked a final CSV used read-only in Phase 2.”

**LIKELY FACULTY QUESTION:** Why OpenAlex?  
**ANSWER:** Open scholarly metadata API with OA and citation fields suitable for reproducible collection without paywalled scrapes.

---

## 3. Database connectivity — 1 min

**WHAT TO SHOW:** Run live:

```powershell
python scripts/phase2/08_da2_database_demo.py
```

**WHAT TO SAY:**  
“Python connects to SQLite, runs SQL, and returns DataFrames—recent papers, highly cited papers, and OA category counts. Same DB file can be queried from R.”

**LIKELY FACULTY QUESTION:** Why SQLite not PostgreSQL?  
**ANSWER:** Local, zero-server demo suitable for coursework; schema is portable if we later move to Postgres.

---

## 4. Feature engineering / selection — 1 min

**WHAT TO SHOW:** `data/metadata/phase2_feature_dictionary.md` + M3 agreement figure.

**WHAT TO SAY:**  
“We built structural features and TF-IDF from title/abstract. For OA prediction we exclude OA flags to avoid leakage. For impact tier we exclude citation counts because they define the label. Thresholds for impact tiers come from training tertiles only.”

**LIKELY FACULTY QUESTION:** What is leakage?  
**ANSWER:** Using information that would not be available—or that directly encodes the label—at prediction time, which inflates metrics unrealistically.

---

## 5. ML models — 1 min

**WHAT TO SHOW:** `DA2/m4_leaderboard.csv` / M4 F1 bar chart.

**WHAT TO SAY:**  
“We trained fourteen algorithms—from logistic regression and SVM to random forests, boosting, XGBoost, LightGBM, and an MLP. AdaBoost leads test accuracy and macro-F1 for open-access prediction at about 0.48 accuracy and 0.45 F1—moderate, not perfect.”

**LIKELY FACULTY QUESTION:** Why so many models?  
**ANSWER:** DA2 requires a broad comparative study; diverse inductive biases help show which families work on this multiclass OA task.

---

## 6. Hyperparameter tuning — 45 sec

**WHAT TO SHOW:** `DA2/m5_tuning_leaderboard.csv` and M5 vs M4 plot.

**WHAT TO SAY:**  
“We tuned five strong models with RandomizedSearchCV on train CV, picked champions on validation, and scored test once. Extra Trees won validation, but did not beat AdaBoost’s test F1—so we report that honestly.”

**LIKELY FACULTY QUESTION:** Did tuning always help?  
**ANSWER:** No. Tuning improved some validation scores; test F1 did not uniformly improve over M4 AdaBoost.

---

## 7. Comparative results — 1 min

**WHAT TO SHOW:** M6 ROC/PR/CM grid + metrics table.

**WHAT TO SAY:**  
“Different metrics pick different leaders: AdaBoost for accuracy/F1, tuned Extra Trees for ROC-AUC around 0.63. We do not force a single universal winner.”

**LIKELY FACULTY QUESTION:** Why is accuracy only ~48%?  
**ANSWER:** Three-class OA labels with leakage-safe features are hard; chance baseline is not 50%. Macro-F1 and ROC give a fuller picture than accuracy alone.

---

## 8. Impact-tier prediction — 1 min

**WHAT TO SHOW:** M7 confusion matrix + leaderboard; mention thresholds 87.33 / 134.0.

**WHAT TO SAY:**  
“Secondary task: relative impact tiers from citation-per-year tertiles. Twelve models ran successfully after a CPU fix for XGBoost. AdaBoost is champion with test F1 about 0.54 and ROC-AUC about 0.68—moderate, useful signal.”

**LIKELY FACULTY QUESTION:** Isn’t using citations circular?  
**ANSWER:** Citations build the *label*; they are excluded from *features*. Predictors are text/metadata such as age and TF-IDF.

---

## 9. Topic clustering — 45 sec

**WHAT TO SHOW:** PCA cluster plot + silhouette curve; k=8, silhouette≈0.064.

**WHAT TO SAY:**  
“KMeans on TF-IDF found eight groups, but silhouette is low, so clusters overlap. We treat this as exploratory support for future similar-paper discovery—not strong topic separation.”

**LIKELY FACULTY QUESTION:** Why keep clustering if silhouette is low?  
**ANSWER:** It documents an honest baseline and still yields interpretable keyword neighborhoods for qualitative exploration.

---

## 10. Final ResearchPilot vision — 30 sec

**WHAT TO SHOW:** README “Current vs Future” / final report capabilities section.

**WHAT TO SAY:**  
“Today we have a solid data-to-intelligence foundation. Next phases can add embeddings, RAG, and a conversational interface—built on these evaluated components, not instead of them.”

**LIKELY FACULTY QUESTION:** What would you improve first?  
**ANSWER:** Stronger text representations (embeddings) and possibly a larger, more balanced corpus—while keeping the same leakage and split discipline.

---

## Timing buffer

If short on time: skip detailed M5 params; keep DB demo + M6 leaders + M7 honesty + clustering caveat.  
If ahead: show one SQL file from `database/queries/` and feature importance plot.
