# ResearchPilot

**A Human-Centered AI Research Assistant — Data Science Foundation (Phase 1 + Phase 2)**

ResearchPilot builds a reproducible pipeline over OpenAlex AI/ML scholarly metadata: collect and clean papers, store them in SQLite, engineer leakage-safe features, train and compare multiple ML models for open-access category prediction, tune strong models, and add secondary impact-tier prediction with exploratory topic clustering.

> **Status:** Phase 1 complete · Phase 2 (M1–M7) complete and frozen · M8 documentation/demo pack complete  
> Phase 3 (RAG / chat assistant / dashboard) is **roadmap only** — not implemented.

---

## Problem

Researchers need structured ways to explore large literature collections. Before conversational AI features, the project establishes trustworthy data, database access, and evaluated predictive models for:

1. **Open-access category** (`fully_open` / `partially_open` / `closed`)  
2. **Relative impact tier** (`low` / `medium` / `high`)  
3. **Exploratory topic neighborhoods** (KMeans on TF-IDF)

---

## Dataset

| Item | Value |
|------|--------|
| Source | OpenAlex API |
| Final table | `data/final/final_dataset.csv` |
| Size | **2000** papers × **27** columns |
| Domain | AI / ML focused corpus |

---

## Phase 1 — Data Foundation

OpenAlex → raw CSV → preprocessing → feature extraction → feature engineering → Python/R EDA → locked `final_dataset.csv`.

Scripts: `scripts/phase1/` · R: `r/scripts/` · Figures: `reports/figures/01_*.png` …

---

## Phase 2 — Intelligence Layer

| Milestone | Focus |
|-----------|--------|
| M1 | SQLite load + SQL retrieval |
| M2 | Splits, TF-IDF, impact_tier labels |
| M3 | Feature selection + scaling |
| M4 | 14 ML algorithms (OA task) |
| M5 | Hyperparameter tuning (5 models) |
| M6 | Comparative metrics + visualizations |
| M7 | Impact-tier (12 models) + clustering |
| M8 | Integration, docs, demo, viva pack |

---

## Architecture

See `reports/figures/final/researchpilot_final_architecture.png` and `DA2/ResearchPilot_DA2_Final_Report.md`.

```text
OpenAlex → Dataset → Features → SQLite → Selection → M4 → M5 → M6
                                              ├─ OA Classification
                                              └─ Impact Tier → Clustering → Insights
```

---

## Database

- Engine: **SQLite** (`database/researchpilot.db`, local / gitignored)  
- Schemas: `database/schemas/`  
- Demo: `python scripts/phase2/08_da2_database_demo.py`

---

## ML models (verified)

### Open-access (M4/M6)

| Leader | Model | Metric |
|--------|-------|--------|
| Best Accuracy / Macro-F1 | **m4_adaboost** | Acc **0.482** · F1 **0.452** |
| Best ROC-AUC | **m5_extra_trees_tuned** | **0.628** |

14 distinct algorithms in M4 (Dummy through MLP, including XGBoost/LightGBM).

### Impact-tier (M7, frozen)

| Item | Value |
|------|--------|
| Champion | **AdaBoost** |
| Val / Test macro-F1 | **0.600** / **0.543** |
| Test Acc / ROC-AUC | **0.551** / **0.680** |
| Models | **12/12** successful |

### Clustering

Best **k=8**, silhouette **0.064** (exploratory; substantial overlap).

---

## How to run (high level)

```powershell
cd E:\Tuned_Research
pip install -r requirements.txt   # then ensure Phase 2 packages installed

# Phase 1 (if rebuilding dataset)
python scripts/phase1/collect_openalex.py
python scripts/phase1/preprocess_papers.py
python scripts/phase1/feature_extraction.py
python scripts/phase1/feature_engineering.py

# Phase 2
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py
python scripts/phase2/04_train_models.py
python scripts/phase2/05_tune_hyperparameters.py
python scripts/phase2/06_comparative_visualizations.py
python scripts/phase2/07_impact_and_clustering.py
python scripts/phase2/08_da2_database_demo.py
```

Full detail: [`DA2/REPRODUCIBILITY.md`](DA2/REPRODUCIBILITY.md).  
**Do not retrain** solely for documentation; M4–M7 metrics are frozen unless a deliberate rebuild is needed.

OpenAlex requests may need an email/polite pool key via environment variables — never commit secrets (see `.env.example` if present; keep real keys out of git).

---

## Folder structure (essentials)

```text
Tuned_Research/
├── data/final/              # Phase 1 locked CSV
├── data/ml/                 # Phase 2 matrices
├── database/                # schemas, queries, local .db
├── scripts/phase1|phase2/   # pipelines
├── src/researchpilot/       # library code
├── configs/                 # YAML configs
├── evaluation/reports/      # local metrics (gitignored)
├── models/checkpoints/      # local joblibs (gitignored)
├── reports/                 # figures + reports
├── DA2/                     # faculty documentation pack
├── r/                       # R EDA
└── docs/                    # architecture notes
```

---

## Limitations

Moderate OA predictability; relative impact tiers; weak cluster separation; 2k-paper AI/ML sample; TF-IDF (not deep semantics); no chat/RAG yet.

---

## Future roadmap

Embeddings, RAG, summarization, similar-paper UI, gap analysis, conversational assistant — **not implemented** in this repository state.

---

## Documentation pack

| Doc | Path |
|-----|------|
| Final DA2 report | [`DA2/ResearchPilot_DA2_Final_Report.md`](DA2/ResearchPilot_DA2_Final_Report.md) |
| Demo script | [`DA2/DA2_Demo_Script.md`](DA2/DA2_Demo_Script.md) |
| Viva Q&A | [`DA2/DA2_Viva_QA.md`](DA2/DA2_Viva_QA.md) |
| Checklist | [`DA2/DA2_FINAL_CHECKLIST.md`](DA2/DA2_FINAL_CHECKLIST.md) |
| Reproducibility | [`DA2/REPRODUCIBILITY.md`](DA2/REPRODUCIBILITY.md) |
| Cleanup (await approval) | [`DA2/M8_Cleanup_Recommendations.md`](DA2/M8_Cleanup_Recommendations.md) |
