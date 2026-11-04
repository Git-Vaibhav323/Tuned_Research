# ResearchPilot — Complete Project Summary

**Project:** ResearchPilot — A Human-Centered Research Intelligence System  
**Course framing:** Programming for Data Science (three-phase delivery)  
**Document type:** End-to-end progress summary  
**Current status:** Phase 1 data pipeline completed through feature engineering  
**Last updated:** August 2026

---

## 1. Executive Summary

ResearchPilot is being built as a **data-first** research intelligence system. It is intentionally **not LLM-only**. The repository first establishes a complete data science workflow—collection, cleaning, feature extraction, and feature engineering—before later phases add machine learning comparison, fine-tuning, RAG, and a dashboard.

To date, the team has:

1. Designed and scaffolded a professional multi-phase repository  
2. Collected **2,000** English research articles from OpenAlex (2022–2025, AI / Machine Learning)  
3. Built a reproducible preprocessing pipeline  
4. Parsed nested JSON metadata into readable fields  
5. Engineered **10** modeling-ready features  
6. Produced a final analysis dataset with **27** columns and supporting documentation/reports  

The primary analysis-ready artifact is:

```text
data/final/final_dataset.csv
```

---

## 2. Project Vision and Scope

### 2.1 Problem

Researchers often struggle with:

- Dense academic language that is hard to understand  
- Uncertainty about appropriate statistical methods  
- Weak abstract writing (grammar, tone, clarity, novelty)  
- Difficulty identifying research gaps across multiple papers  

Generic chatbots can help casually, but they lack a reproducible data science backbone, documented datasets, comparative modeling, and grounded evaluation.

### 2.2 Final product modules (planned for Phase 3)

| Module | Purpose |
|--------|---------|
| Adaptive Research Translator | Simplify academic text; key takeaways; examples |
| Statistical Advisor | Recommend tests; rationale; assumptions; alternatives |
| Abstract Improver | Improve grammar, tone, clarity, novelty framing |
| Research Gap Finder | Compare papers; limitations; future work; gap ideas |

### 2.3 Delivery phases

```text
Phase 1 → Dataset Collection → Preprocessing → EDA → Documentation → Features
Phase 2 → Machine Learning → Comparative Analysis
Phase 3 → Fine-tuning → RAG → Dashboard
```

**Completed so far:** the core Phase 1 data-engineering track (collection → cleaning → extraction → feature engineering). Formal EDA notebooks and R visualizations remain next within Phase 1.

---

## 3. Repository Foundation

Before any data was collected, the project was organized as a production-ready research repository at the workspace root (`Tuned_Research/`).

### 3.1 Major folders

| Path | Role |
|------|------|
| `data/raw/` | Immutable collected data |
| `data/cleaned/` | Cleaned paper metadata |
| `data/processed/` | Feature-extracted tables |
| `data/final/` | Final engineered dataset |
| `scripts/phase1|2|3/` | Phase-aligned automation scripts |
| `configs/phase1|2|3/` | YAML configuration placeholders |
| `notebooks/phase1_eda|phase2_ml|phase3_llm/` | Notebook scaffolds |
| `src/researchpilot/` | Reusable package layout (`data`, `features`, `models`, `visualization`, `rag`, `assistant`) |
| `docs/` | Architecture, roadmap, dataset plan, scope, benchmarks |
| `reports/` | Generated reports, figures, tables |
| `backend/`, `frontend/` | Phase 3 API / dashboard scaffolds |
| `models/`, `evaluation/`, `database/`, `r/` | Later-phase and analytics support |

### 3.2 Foundation documents created

- `README.md` — project overview and phase plan  
- `docs/architecture.md` — system architecture  
- `docs/project_architecture.md` — detailed flow diagrams  
- `docs/dataset_plan.md` — Phase 1 dataset strategy  
- `docs/development_roadmap.md` — Phase 1–3 roadmap  
- `docs/project_scope.md` — in-scope / out-of-scope boundaries  
- `docs/benchmark_plan.md` — evaluation design across phases  
- `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE` (MIT), `.gitignore`  
- `requirements.txt` — planned dependencies by phase (not all installed)  
- `.env` / `.env.example` — OpenAlex API key configuration (secrets not committed)

### 3.3 Design principle

> **Data before AI.**  
> Phase 3 fine-tuning / RAG / dashboard must consume validated Phase 1 data and Phase 2 models. The repository must first satisfy a complete data science workflow.

---

## 4. Phase 1 Pipeline Overview

The implemented Phase 1 data flow is:

```text
OpenAlex API
    ↓
scripts/phase1/collect_openalex.py
    ↓
data/raw/raw_papers.csv                     (2,000 × 11)
    ↓
scripts/phase1/preprocess_papers.py
    ↓
data/cleaned/cleaned_papers.csv             (2,000 × 11)
reports/preprocessing_report.md
    ↓
scripts/phase1/feature_extraction.py
    ↓
data/processed/feature_extracted.csv        (2,000 × 17)
    ↓
scripts/phase1/feature_engineering.py
    ↓
data/final/final_dataset.csv                (2,000 × 27)
reports/feature_engineering_report.md
```

---

## 5. Dataset Collection (OpenAlex)

### 5.1 Script

`scripts/phase1/collect_openalex.py`

### 5.2 Authentication

- API key loaded from project-root `.env`  
- Expected variable: `OpenAlex_API_KEY`  
- Key is never hard-coded in source  

### 5.3 Final collection filters (API-side)

All filters are applied **in the OpenAlex request**, not after download:

| Filter | Value |
|--------|--------|
| Topics | Artificial Intelligence **OR** Machine Learning (concept IDs) |
| Publication year | `2022–2025` |
| Language | English (`en`) |
| Type | `article` |
| Has abstract | `true` |

Concept IDs used:

- Artificial Intelligence → `C154945302`  
- Machine Learning → `C119857082`  

### 5.4 Collection mechanics

- Endpoint: `https://api.openalex.org/works`  
- Cursor pagination (`cursor=*`, then `meta.next_cursor`)  
- ~100 results per page  
- Target: **~2,000** unique papers  
- Retry with backoff on failures / HTTP 429  
- Progress logging per page  
- Incremental checkpoint save every 100 papers  
- Abstracts reconstructed from OpenAlex `abstract_inverted_index`  

### 5.5 Fields saved at collection time

| Field | Description |
|-------|-------------|
| `id` | OpenAlex work ID |
| `title` | Paper title |
| `abstract` | Reconstructed plain-text abstract |
| `publication_year` | Year of publication |
| `cited_by_count` | Citation count |
| `language` | Language code |
| `type` | Work type |
| `concepts` | JSON list of concepts |
| `keywords` | JSON list of keywords |
| `doi` | DOI URL / identifier |
| `open_access` | JSON open-access object |

### 5.6 Collection result

| Metric | Value |
|--------|------:|
| Output file | `data/raw/raw_papers.csv` |
| Rows collected | 2,000 |
| Columns | 11 |
| Years present | 2022, 2023, 2024, 2025 |
| Language | 100% English |

### 5.7 Earlier collection note

An earlier broader collection (`data/raw/papers.csv`) used more topics and no year restriction. The **canonical raw dataset** for the current Phase 1 pipeline is `raw_papers.csv` (2022–2025, AI/ML only).

---

## 6. Preprocessing / Cleaning

### 6.1 Script

`scripts/phase1/preprocess_papers.py`  
(also callable via `scripts/phase1/preprocess_data.py`)

### 6.2 Input / output

| | Path |
|--|--|
| Input | `data/raw/raw_papers.csv` |
| Cleaned output | `data/cleaned/cleaned_papers.csv` |
| Report | `reports/preprocessing_report.md` |

### 6.3 Cleaning operations

1. **Title cleaning** — remove newlines; collapse extra whitespace; strip  
2. **Abstract cleaning** — normalize whitespace; drop null/empty abstracts  
3. **`publication_year`** — convert to integer  
4. **`cited_by_count`** — convert to integer  
5. **Language filter** — keep only `en` / `english`  
6. **DOI deduplication** — drop duplicate DOIs, keeping the highest `cited_by_count`; retain rows with missing DOI  

### 6.4 Preprocessing outcomes

| Metric | Before | After |
|--------|-------:|------:|
| Rows | 2,000 | 2,000 |
| Columns | 11 | 11 |
| Fully duplicate rows | 0 | 0 |
| Duplicate DOIs | 0 | 0 |
| Rows removed | — | 0 |

**Interpretation:** Because strong filters were already applied during OpenAlex collection, cleaning mainly validated and hardened the dataset. Only **2** DOIs were missing; all other core fields were complete.

### 6.5 Missing values after cleaning

| Column | Missing |
|--------|--------:|
| Most fields | 0 |
| `doi` | 2 |

---

## 7. Feature Extraction (JSON parsing)

### 7.1 Script

`scripts/phase1/feature_extraction.py`

### 7.2 Input / output

| | Path |
|--|--|
| Input | `data/cleaned/cleaned_papers.csv` |
| Output | `data/processed/feature_extracted.csv` |

### 7.3 Parsed source columns

- `concepts` (JSON array)  
- `keywords` (JSON array)  
- `open_access` (JSON object)  

### 7.4 New readable columns created

| New column | Source / meaning |
|------------|------------------|
| `concepts_clean` | Concept `display_name`s joined by `"; "` |
| `keywords_clean` | Keyword `display_name`s joined by `"; "` |
| `is_open_access` | `open_access.is_oa` |
| `oa_status` | `open_access.oa_status` (gold, hybrid, green, diamond, bronze, closed, …) |
| `oa_url` | `open_access.oa_url` |
| `has_fulltext` | `open_access.any_repository_has_fulltext` |

### 7.5 Extraction result

| Metric | Value |
|--------|------:|
| Rows | 2,000 |
| Columns | 17 (11 original + 6 new) |
| Open-access rate | ~77.8% |
| Fulltext available (repository flag) | ~63.9% |

Original JSON columns were **kept unchanged**.

---

## 8. Feature Engineering

### 8.1 Script

`scripts/phase1/feature_engineering.py`

### 8.2 Input / output

| | Path |
|--|--|
| Input | `data/processed/feature_extracted.csv` |
| Final dataset | `data/final/final_dataset.csv` |
| Report | `reports/feature_engineering_report.md` |

### 8.3 Engineered features (formulas and purpose)

| Feature | Formula | Why it was created |
|---------|---------|--------------------|
| `paper_age` | `max(2026 − publication_year, 0)` | Controls citation exposure time; enables temporal analysis |
| `title_length` | character length of stripped title | Lightweight text-complexity / verbosity proxy |
| `abstract_length` | character length of stripped abstract | Proxy for abstract richness / completeness |
| `keyword_count` | count of tokens in `keywords_clean` | Topical tagging density |
| `concept_count` | count of tokens in `concepts_clean` | Conceptual / topical breadth |
| `citation_per_year` | `cited_by_count / max(paper_age, 1)` | Age-normalized impact (fairer across years) |
| `citation_log` | `ln(1 + cited_by_count)` | Reduces right-skew of citation counts |
| `recent_paper` | `1` if `paper_age ≤ 2`, else `0` | Flags newest cohort for stratified analysis |
| `has_doi` | `1` if DOI present, else `0` | Metadata quality / citability signal |
| `oa_category` | mapped from `oa_status` | Collapses OA statuses into ML-friendly categories |

#### `oa_category` mapping

| OpenAlex status | Engineered category |
|-----------------|---------------------|
| `gold`, `diamond` | `fully_open` |
| `hybrid`, `green`, `bronze` | `partially_open` |
| `closed` | `closed` |
| other / ambiguous | `unknown` (or `closed` if `is_open_access` is false) |

### 8.4 Engineered feature summary statistics

| Feature | Mean | Median | Min | Max |
|---------|-----:|-------:|----:|----:|
| `paper_age` | 3.38 | 4 | 1 | 4 |
| `title_length` | 83.88 | 82 | 6 | 232 |
| `abstract_length` | 1393.89 | 1347.50 | 83 | 10719 |
| `keyword_count` | 12.10 | 12 | 2 | 29 |
| `concept_count` | 15.26 | 15 | 2 | 36 |
| `citation_per_year` | 177.78 | 105 | 59.75 | 25760.67 |
| `citation_log` | 6.00 | 5.83 | 5.48 | 11.26 |

| Feature | Result |
|---------|--------|
| `recent_paper` | 265 papers (13.2%) |
| `has_doi` | 1,998 papers (99.9%) |

| `oa_category` | Count | Share |
|---------------|------:|------:|
| `fully_open` | 834 | 41.7% |
| `partially_open` | 723 | 36.1% |
| `closed` | 443 | 22.1% |

### 8.5 Final dataset shape

| Metric | Value |
|--------|------:|
| File | `data/final/final_dataset.csv` |
| Rows | 2,000 |
| Columns | 27 |

All earlier columns are retained; engineered features are appended.

---

## 9. Final Dataset Schema (27 columns)

### Identity and bibliographic fields

`id`, `title`, `abstract`, `publication_year`, `cited_by_count`, `language`, `type`, `doi`

### Nested / raw metadata

`concepts`, `keywords`, `open_access`

### Extracted readable fields

`concepts_clean`, `keywords_clean`, `is_open_access`, `oa_status`, `oa_url`, `has_fulltext`

### Engineered analytical features

`paper_age`, `title_length`, `abstract_length`, `keyword_count`, `concept_count`, `citation_per_year`, `citation_log`, `recent_paper`, `has_doi`, `oa_category`

---

## 10. Scripts Inventory (Phase 1)

| Script | Purpose | Primary output |
|--------|---------|----------------|
| `collect_openalex.py` | Fetch AI/ML papers from OpenAlex | `data/raw/raw_papers.csv` |
| `preprocess_papers.py` | Clean and validate metadata | `data/cleaned/cleaned_papers.csv` + preprocessing report |
| `preprocess_data.py` | Entrypoint alias for preprocessing | same as above |
| `feature_extraction.py` | Parse JSON into readable columns | `data/processed/feature_extracted.csv` |
| `feature_engineering.py` | Create modeling features + report | `data/final/final_dataset.csv` + FE report |
| `collect_data.py` | Generic collection scaffold | reserved |

### How to reproduce Phase 1 data artifacts

```bash
# from project root, with OpenAlex_API_KEY set in .env
python scripts/phase1/collect_openalex.py
python scripts/phase1/preprocess_papers.py
python scripts/phase1/feature_extraction.py
python scripts/phase1/feature_engineering.py
```

Supporting packages used in these scripts include `pandas`, `numpy`, `requests`, and `python-dotenv`.

---

## 11. Reports and Documentation Artifacts

| Artifact | Contents |
|----------|----------|
| `reports/preprocessing_report.md` | Before/after cleaning summary, missingness, rows removed |
| `reports/feature_engineering_report.md` | Feature definitions, formulas, rationale, summary stats |
| `docs/architecture.md` | Logical architecture across phases |
| `docs/project_architecture.md` | Detailed flow diagrams |
| `docs/dataset_plan.md` | Dataset strategy and quality plan |
| `docs/development_roadmap.md` | Phase 1–3 roadmap and exit criteria |
| `docs/project_scope.md` | Scope boundaries |
| `docs/benchmark_plan.md` | Evaluation plan for data, ML, and assistant quality |
| `docs/summary.md` | This complete progress summary |

---

## 12. Key Findings So Far

1. **Collection quality is high.** API-side filters produced a coherent English AI/ML article corpus for 2022–2025 with abstracts present.  
2. **Cleaning removed 0 rows**, confirming that collection filters already enforced most quality constraints; preprocessing still remains necessary for reproducibility and documentation.  
3. **Open access is common in this slice** (~78% OA; ~42% fully open).  
4. **DOI coverage is nearly complete** (99.9%).  
5. **Citation distribution is skewed**, which justifies `citation_log` and `citation_per_year`.  
6. **Topical metadata is rich** (median ~12 keywords and ~15 concepts per paper).  
7. **The final table is Phase-2-ready** for classical ML / comparative analysis once a prediction target is chosen.

---

## 13. What Is Intentionally Not Done Yet

These items are planned but **not implemented** yet:

### Remaining Phase 1

- Full EDA notebooks with interpreted charts  
- R visualizations (`r/` scripts / ggplot outputs)  
- Formal data dictionary under `data/metadata/`  
- Deeper bias / coverage analysis across venues and topics  

### Phase 2

- Database schema / SQL integration  
- Train/validation/test modeling pipeline  
- Baseline + multiple ML/DL algorithms  
- Comparative evaluation metrics and visualizations  

### Phase 3

- Instruction dataset construction  
- Fine-tuning / PEFT  
- RAG index (chunking, embeddings, ChromaDB)  
- Research assistant modules  
- Interactive dashboard (Streamlit / FastAPI)  

---

## 14. Immediate Next Steps

Recommended next work (still Phase 1):

1. Create a formal **data dictionary** and provenance note in `data/metadata/`  
2. Complete **EDA** in `notebooks/phase1_eda/` using `data/final/final_dataset.csv`  
3. Produce Python (and optional R) visualizations into `reports/figures/`  
4. Decide the Phase 2 **prediction target** (examples: open-access category, citation tier, recent vs established impact)  
5. Freeze a small held-out benchmark slice before modeling begins  

---

## 15. One-Page Snapshot

| Item | Status / value |
|------|----------------|
| Project name | ResearchPilot |
| Architecture style | Data-first, three-phase |
| Raw papers | 2,000 |
| Years | 2022–2025 |
| Topics | AI OR Machine Learning |
| Language | English |
| Cleaned rows | 2,000 |
| Feature-extracted columns | 17 |
| Final columns | 27 |
| Engineered features | 10 |
| Open-access rate | ~77.8% |
| DOI coverage | ~99.9% |
| Canonical final file | `data/final/final_dataset.csv` |
| Next major milestone | Phase 1 EDA + documentation polish → Phase 2 ML |

---

## 16. Conclusion

ResearchPilot now has a **complete, reproducible Phase 1 data foundation**: from OpenAlex collection through cleaning, JSON feature extraction, and engineered analytical features. The repository structure, scripts, configs, notebooks, and docs are aligned to a three-phase course roadmap that keeps classical data science first and places fine-tuning, RAG, and dashboard work in Phase 3.

The project is ready to proceed into deeper exploratory analysis and then Phase 2 machine learning / comparative modeling using `data/final/final_dataset.csv` as the primary dataset.

---

*End of summary.*
