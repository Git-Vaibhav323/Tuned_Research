# ResearchPilot — System Architecture

**Status:** Foundation design (implementation deferred)

---

## 1. Purpose

ResearchPilot is a data-first research intelligence system. Architecture evolves in three phases: data foundation, predictive modeling, then fine-tuning / RAG / dashboard.

---

## 2. Principles

| Principle | Meaning |
|-----------|---------|
| Data before AI | Phase 3 consumes Phase 1–2 artifacts |
| Reproducibility | Configs, metadata, and notebooks track every run |
| Separation of concerns | Data, ML, RAG, UI live in dedicated packages |
| Traceability | Answers cite sources when retrieval is used |
| Human-centered | Explanations, assumptions, alternatives, uncertainty |

---

## 3. Three-Phase Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ Phase 1 — DATA FOUNDATION                                    │
│ Collection → Cleaning → Preprocessing → EDA → Documentation │
│ Outputs: processed data, metadata, notebooks, figures        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Phase 2 — MACHINE LEARNING                                   │
│ Features → Baseline + Models → Metrics → Comparison          │
│ Outputs: model artifacts, comparison report                  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Phase 3 — AI APPLICATION                                     │
│ Fine-tuning → RAG → Research Modules → Dashboard             │
│ Outputs: adapters, index, API/UI, final evaluation           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Final Product Modules (Phase 3)

1. **Adaptive Research Translator** — simplify text, takeaways, examples  
2. **Statistical Advisor** — test recommendation, rationale, assumptions  
3. **Abstract Improver** — grammar, tone, clarity, novelty  
4. **Research Gap Finder** — multi-paper gaps and future work  

---

## 5. Data Layout

```text
data/
├── raw/                   # Immutable structured sources
├── raw_papers/            # PDFs / documents
├── external/              # Third-party datasets
├── interim/               # Intermediate files
├── processed/             # Analysis-ready data
├── instruction_dataset/   # Phase 3 instruction pairs
├── benchmark/             # Frozen eval sets
└── metadata/              # Dictionary, licenses, manifests
```

Rules:

- Never overwrite raw data.  
- Write transforms to `interim/` then `processed/`.  
- Record licenses and provenance in `metadata/`.  

---

## 6. Phase 2 Modeling View

```text
Processed data → Feature engineering → Train/Val/Test
    → Baseline + ML/DL models → Shared metrics → Comparison report
```

---

## 7. Phase 3 Intelligence View

```text
User query / PDF
    → Document processing → Chunking
    → Retrieval (embeddings + vector store)
    → Fine-tuned LLM + optional Phase 2 model services
    → Response generator
    → Answer + Sources + Confidence
```

---

## 8. Repository Mapping

| Concern | Location |
|---------|----------|
| Data pipelines | `src/researchpilot/data/` |
| Features / ML | `src/researchpilot/features/`, `models/` |
| RAG | `src/researchpilot/rag/` |
| Assistant modules | `src/researchpilot/assistant/` |
| Configs / scripts / notebooks | `configs/`, `scripts/`, `notebooks/` |
| Dashboard / API | `frontend/`, `backend/` |

---

## 9. Related Docs

- [project_architecture.md](project_architecture.md)  
- [dataset_plan.md](dataset_plan.md)  
- [development_roadmap.md](development_roadmap.md)  
- [project_scope.md](project_scope.md)  
- [benchmark_plan.md](benchmark_plan.md)  
