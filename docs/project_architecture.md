# ResearchPilot — Detailed Project Architecture

**Document type:** Architecture diagrams & end-to-end flows  
**Phase:** 1 — Foundation  
**Status:** Planning artifact (no code implementation)

This document expands [architecture.md](architecture.md) with Markdown diagrams suitable for onboarding and design reviews.

---

## 1. Course-Phase Evolution

```text
DA1 — DATA SCIENCE FOUNDATION
Research Sources
  ↓
Collection + Dataset Documentation
  ↓
Raw Data → Cleaning → Preprocessing → Processed Data
  ↓
Exploratory Data Analysis
  ↓
Python Visualizations + R Visualizations
  ↓
DA1 Dataset, Figures, and Findings

                  ↓ reusable validated data

DA2 — DATABASE AND MODELING
Database Schema + Ingestion + SQL Analysis
  ↓
Feature Engineering
  ↓
Baseline + Multiple ML/DL Algorithms
  ↓
Evaluation Metrics + Error Analysis
  ↓
Model Comparison + Comparative Visualizations
  ↓
DA2 Selected Model and Report

                  ↓ evaluated models and analytical assets

DA3 — RESEARCH INTELLIGENCE APPLICATION
LLM + Optional Retrieval + DA2 Model Services
  ↓
AI Research Assistant Modules
  ↓
Dashboard + Interactive Visualizations
  ↓
Final Technical and User-Centered Evaluation
```

## 2. DA3 End-to-End User Flow

```text
User
  │
  ▼
Upload PDF / Ask Question
  │
  ▼
Document Processing
  │  (PDF parse · text clean · section detect)
  ▼
Chunking
  │  (semantic / fixed-size chunks · metadata attach)
  ▼
Retrieval Layer
  │  (embed query · similarity search · rerank optional)
  ▼
Fine-Tuned Language Model
  │  (module-specific system prompt · context window packing)
  ▼
Response Generator
  │  (format · cite · calibrate confidence)
  ▼
Answer + Sources + Confidence
```

---

## 3. DA3 Request Lifecycle (Sequence)

```text
┌──────┐   ┌────────┐   ┌─────────────┐   ┌──────────┐   ┌─────┐   ┌──────────┐
│ User │   │   UI   │   │  FastAPI    │   │Orchestr. │   │ RAG │   │   LLM    │
└──┬───┘   └───┬────┘   └──────┬──────┘   └────┬─────┘   └──┬──┘   └────┬─────┘
   │  submit   │               │               │            │           │
   │──────────►│  HTTP JSON    │               │            │           │
   │           │──────────────►│  route module │            │           │
   │           │               │──────────────►│  retrieve  │           │
   │           │               │               │───────────►│           │
   │           │               │               │◄───────────│ contexts  │
   │           │               │               │  generate  │           │
   │           │               │               │───────────────────────►│
   │           │               │               │◄───────────────────────│
   │           │               │◄──────────────│  structured reply      │
   │           │◄──────────────│               │            │           │
   │◄──────────│  answer+src   │               │            │           │
```

---

## 4. Full Component Diagram

```text
┌───────────────────────────────────────────────────────────────┐
│ DA1: data/ + src/researchpilot/data/ + notebooks/ + r/         │
│ Collection · cleaning · preprocessing · EDA · visualization    │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ DA2: database/ + features/ + models/ + evaluation/ + reports/  │
│ SQL · feature pipelines · ML/DL · metrics · comparisons        │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ DA3: assistant/ + backend/ + frontend/                         │
│ LLM/retrieval · research modules · dashboard · interaction     │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ Shared artifacts: processed data · database · evaluated models │
│ benchmark sets · figures · reports · model/LLM artifacts       │
└───────────────────────────────────────────────────────────────┘
```

---

## 5. DA3 Module Router

```text
                    Incoming request
                           │
                           ▼
                 ┌─────────────────────┐
                 │   Module classifier │
                 │  (explicit module   │
                 │   id from client)   │
                 └──────────┬──────────┘
      ┌───────────┬─────────┼─────────┬───────────┐
      ▼           ▼         ▼         ▼           ▼
 Translator   Stats Adv.  Abstract   Gap Finder  (Future)
      │           │         │         │
      └───────────┴────┬────┴─────────┘
                       ▼
              Shared LLM + tools
                       │
                       ▼
              Normalized response schema
              { answer, sources[], confidence, meta }
```

MVP uses **explicit module selection** from the UI (no automatic intent classifier required for v1).

---

## 6. DA1 Document Processing Pipeline

```text
PDF / text upload
      │
      ▼
┌─────────────┐
│  Ingest     │  → data/raw_papers/ (+ metadata/)
└──────┬──────┘
       ▼
┌─────────────┐
│  Extract    │  pymupdf → plain text / pages
└──────┬──────┘
       ▼
┌─────────────┐
│  Normalize  │  whitespace, encoding, boilerplate removal
└──────┬──────┘
       ▼
┌─────────────┐
│  Sectionize │  title, abstract, intro, method, ... (heuristic)
└──────┬──────┘
       ▼
┌─────────────┐
│  Chunk      │  size + overlap; store ids & offsets
└──────┬──────┘
       ▼
 data/interim/ → data/processed/
       │
       ├── Python EDA / figures
       └── R analysis / visualizations
```

---

## 7. DA2 Database and Modeling Flow

```text
data/processed/
      │
      ▼
Database ingestion → schemas + constraints + SQL queries
      │
      ▼
Feature engineering pipeline
      │
      ▼
Consistent train / validation / test split
      │
      ├── Baseline
      ├── Classical ML models
      ├── Ensemble / boosting models
      └── Compact DL model (when justified)
      │
      ▼
Metrics + cross-validation + error analysis
      │
      ▼
Comparison tables + comparative visualizations
```

## 8. DA3 Retrieval Layer (Planned)

```text
User query / module context
        │
        ▼
   Query embedding (sentence-transformers)
        │
        ▼
   Vector search (chromadb)
        │
        ▼
   Top-k chunks (+ optional rerank)
        │
        ▼
   Context packer (token budget)
        │
        ▼
   Prompt → Fine-tuned LLM
```

**Module-specific retrieval policy (planned):**

| Module | Retrieval |
|--------|-----------|
| Translator | Optional (glossary / similar explanations) |
| Statistical Advisor | Strongly recommended (stats knowledge snippets) |
| Abstract Improver | Optional (style exemplars) |
| Gap Finder | Strongly recommended (multi-paper chunks) |

---

## 9. DA3 LLM Training / Artifact Flow

```text
data/instruction_dataset/
          │
          ▼
   configs/*.yaml  (hyperparams, paths)
          │
          ▼
   scripts/ (LLM train entrypoint — DA3, only if required)
          │
          ▼
   models/adapters/  or  models/checkpoints/
          │
          ▼
   models/exports/  (quantized / merged for serve)
          │
          ▼
   backend model loader
```

---

## 10. Cross-Phase Evaluation Flow

```text
DA1: data quality + EDA validation
       │
       ▼
DA2: held-out predictive metrics + model comparison
       │
       ▼
DA3: data/benchmark/
       │
       ▼
evaluation/ + scripts/
       │
       ├── automatic metrics → evaluation/metrics/
       └── human review notes → evaluation/reports/
```

---

## 11. Repository Mapping

| Architectural concern | Repository location |
|-----------------------|---------------------|
| Raw & processed data | `data/` |
| Data cleaning/preprocessing | `src/researchpilot/data/` |
| R analysis and visualization | `r/` |
| Database schemas and SQL | `database/` |
| Feature engineering | `src/researchpilot/features/` |
| Conventional ML/DL workflows | `src/researchpilot/models/` |
| Training configs | `configs/` |
| API & services | `backend/app/` |
| UI | `frontend/` |
| Model weights | `models/` |
| Metrics & reports | `evaluation/` |
| Course figures and tables | `reports/` |
| Automation scripts | `scripts/` |
| Specs & design | `docs/` |
| Tests | `tests/` |
| Exploration | `notebooks/` |

---

## 12. Foundation Non-Goals

These diagrams **do not** imply that the foundation includes:

- Running APIs or UI  
- Implemented RAG or embedding pipelines  
- Trained model weights  
- Live vector databases  

They define the target architecture for **DA1, DA2, and DA3**. The presence of DA3 components does not imply that LLM work begins before the DA1 and DA2 quality gates are met.
