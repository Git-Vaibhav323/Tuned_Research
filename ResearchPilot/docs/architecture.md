# ResearchPilot — System Architecture

**Document type:** Design  
**Phase:** 1 — Foundation  
**Status:** Draft (implementation deferred)

---

## 1. Purpose

This document defines ResearchPilot as a data-first research intelligence system for a Programming for Data Science course. The architecture is intentionally cumulative: DA1 produces trustworthy analytical data, DA2 turns that data into database-backed predictive models, and DA3 adds the AI research assistant and interactive product.

---

## 2. Architectural Principles

| Principle | Implication |
|-----------|-------------|
| **Human-centered** | Outputs include explanations, assumptions, alternatives, and uncertainty—not bare answers. |
| **Data before AI** | LLM features consume validated data and evaluated models; they do not substitute for data science. |
| **Modular skills** | Each MVP module is an independent skill with a clear contract; shared infra is reused. |
| **Reproducibility** | Datasets, configs, and eval protocols are versioned and documented. |
| **Separation of concerns** | Data, training, serving, UI, and evaluation are isolated packages/directories. |
| **Incremental delivery** | DA1, DA2, and DA3 each produce independently assessable artifacts. |
| **Traceability** | Where retrieval is used, answers cite sources and retain provenance. |

---

## 3. Three-Phase Logical Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│ DA1 — DATA FOUNDATION                                               │
│ Sources → Collection → Documentation → Cleaning → Preprocessing     │
│                          ↓                                         │
│              EDA + Statistical Analysis + Python/R Visualizations   │
│ Outputs: processed data, metadata, notebooks, figures, DA1 report   │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ validated datasets
┌──────────────────────────────▼─────────────────────────────────────┐
│ DA2 — DATABASE & MODELING                                           │
│ Database → SQL Analysis → Feature Engineering → Train/Val/Test       │
│                          ↓                                         │
│        Multiple ML/DL Algorithms → Metrics → Model Comparison       │
│ Outputs: schema, features, model artifacts, comparison report       │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ evaluated data/model services
┌──────────────────────────────▼─────────────────────────────────────┐
│ DA3 — AI APPLICATION                                                │
│ LLM/Retrieval Integration → Research Assistant Modules              │
│                          ↓                                         │
│ Interactive Dashboard + Visual Analytics + Final Evaluation         │
│ Outputs: assistant, dashboard, end-to-end evaluation, final report  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase Boundaries

### 4.1 DA1 — Data Foundation

`src/researchpilot/data/`, `notebooks/`, `r/`, and `data/` own acquisition, data dictionaries, cleaning rules, preprocessing, EDA, and visual outputs. Raw data is immutable. Every processed dataset must be reproducible from documented sources.

### 4.2 DA2 — Database and Modeling

`database/`, `src/researchpilot/features/`, `src/researchpilot/models/`, `evaluation/`, and `reports/` own persistent storage, SQL, features, conventional ML/DL experiments, metrics, and comparative visualizations. LLMs are not required to satisfy DA2.

### 4.3 DA3 — AI Assistant and Dashboard

`src/researchpilot/assistant/`, `backend/`, and `frontend/` own LLM/retrieval integration, the four assistant modules, interactive analytics, and final user-facing evaluation. DA3 reuses DA1 data assets and DA2 models/metrics.

---

## 5. Final MVP Module Contracts (Conceptual)

### 5.1 Adaptive Research Translator

| Field | Description |
|-------|-------------|
| **Input** | Academic passage or paper section; optional audience level |
| **Output** | Simplified text, key takeaways, concept explanations with examples |
| **Side inputs** | Optional glossary / domain hints (future) |

### 5.2 Statistical Advisor

| Field | Description |
|-------|-------------|
| **Input** | Study design description, variable types, hypothesis / goal |
| **Output** | Recommended test(s), rationale, assumptions, alternatives |
| **Constraints** | Must not fabricate p-values or analyze raw data unless provided (future) |

### 5.3 Abstract Improver

| Field | Description |
|-------|-------------|
| **Input** | Draft abstract; optional target venue / word limit |
| **Output** | Revised abstract + notes on grammar, tone, clarity, novelty |

### 5.4 Research Gap Finder

| Field | Description |
|-------|-------------|
| **Input** | Two or more paper texts / abstracts |
| **Output** | Limitations, future work suggestions, research gap ideas |
| **Side inputs** | Optional retrieval over related literature (DA3) |

---

## 6. Data Architecture

```text
data/
├── raw_papers/            # Immutable source artifacts
├── external/              # Immutable third-party datasets
├── interim/               # Intermediate cleaning outputs
├── processed/             # Analysis-ready data
├── instruction_dataset/   # Train / val instruction pairs
├── benchmark/             # Frozen evaluation sets
└── metadata/              # Licenses, DOI, source URLs, splits
```

**Rules (planned):**

- Raw data is never overwritten; processing always writes to `processed/`.
- Interim transformations are reproducible and may be regenerated.
- DA1 data dictionaries and cleaning decisions are stored in metadata/docs.
- Instruction and benchmark splits must not leak (documented in `dataset_plan.md`).
- Metadata must record license and redistribution constraints.

---

## 7. DA2 Modeling Architecture (Planned)

```text
Processed data / database views
        ↓
Feature engineering pipeline
        ↓
Train / validation / test split
        ↓
Baseline + multiple ML/DL algorithms
        ↓
Common evaluation protocol
        ↓
Comparison tables and visualizations
        ↓
Selected model artifact + model card
```

Candidate families include linear baselines, tree ensembles, boosting, clustering/topic methods where appropriate, and a small neural baseline. Algorithms will be chosen based on the final dataset and prediction task rather than included merely for quantity.

---

## 8. DA3 LLM Architecture (Planned)

```text
Base LLM
   ↓
Parameter-efficient fine-tuning (PEFT / LoRA via peft + trl / unsloth)
   ↓
Task adapters (optional per module)  →  models/adapters/
   ↓
Merged or adapter-loaded checkpoint  →  models/checkpoints/ | models/exports/
```

Retrieval (when enabled) uses embedding models (`sentence-transformers`) and a vector index (`chromadb`), orchestrated via LangChain / LangGraph patterns.

---

## 9. Evaluation Architecture

```text
DA1 data-quality checks + EDA findings
        ↓
DA2 held-out predictive metrics + cross-model comparison
        ↓
DA3 benchmark set (data/benchmark/)
        ↓
Inference runner (scripts/ + evaluation/)
        ↓
Automatic metrics (BERTScore, task-specific scores via evaluate)
        ↓
Human rubric sampling (reports/)
        ↓
evaluation/reports/
```

See [benchmark_plan.md](benchmark_plan.md).

---

## 10. Deployment View (Future)

| Environment | Role |
|-------------|------|
| **Local / research** | Notebooks, training, offline eval |
| **Dev API** | FastAPI + Uvicorn for integration testing |
| **Demo UI** | Streamlit for stakeholder demos |
| **Production** (later) | Containerized API, model serving, monitoring |

The foundation does not define cloud vendor lock-in; configs under `configs/` will parameterize environments later.

---

## 11. Security & Ethics (Baseline)

- Respect paper licenses and copyright when ingesting corpora.  
- Prefer citation and quotation over wholesale republication.  
- Surface model uncertainty for high-stakes advice (especially statistics).  
- Avoid storing private user documents without consent (policy TBD in later phases).  

---

## 12. Related Documents

- [project_architecture.md](project_architecture.md) — detailed flow diagrams  
- [dataset_plan.md](dataset_plan.md) — data engineering plan  
- [development_roadmap.md](development_roadmap.md) — phased delivery  
- [project_scope.md](project_scope.md) — boundaries  
- [benchmark_plan.md](benchmark_plan.md) — evaluation design  
