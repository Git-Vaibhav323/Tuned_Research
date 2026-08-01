# ResearchPilot — Development Roadmap

**Status:** Foundation scaffold complete  
**Delivery model:** Three cumulative phases

---

## Overview

```text
Phase 1: Dataset Collection → Preprocessing → EDA → Documentation
Phase 2: Machine Learning → Comparative Analysis
Phase 3: Fine-tuning → RAG → Dashboard
```

Each phase produces assessable artifacts used by the next phase.

---

## Foundation (complete)

- [x] Repository layout  
- [x] README and docs  
- [x] Phase configs, scripts, notebook folders  
- [x] Planned `requirements.txt`  
- [x] No premature ML/RAG/dashboard implementation  

---

## Phase 1 — Dataset, EDA, Preprocessing, Documentation

### Goals

- Collect a licensed, research-relevant dataset  
- Document provenance, schema, and data dictionary  
- Clean and preprocess into `data/processed/`  
- Perform exploratory data analysis  
- Publish figures, tables, and findings  

### Locations

| Work | Path |
|------|------|
| Raw / external data | `data/raw/`, `data/external/`, `data/raw_papers/` |
| Processed data | `data/processed/` |
| Metadata | `data/metadata/` |
| Scripts | `scripts/phase1/` |
| Configs | `configs/phase1/` |
| Notebooks | `notebooks/phase1_eda/` |
| Code | `src/researchpilot/data/`, `visualization/` |
| Reports | `reports/` |

### Exit criteria

- [ ] Sources and licenses documented  
- [ ] Data dictionary complete  
- [ ] Cleaning decisions reproducible  
- [ ] EDA notebooks interpreted  
- [ ] Processed dataset validated  
- [ ] Phase 2 modeling target proposed  

---

## Phase 2 — Machine Learning and Comparative Analysis

### Goals

- Engineer features without leakage  
- Train a baseline and multiple ML/DL models  
- Evaluate with task-appropriate metrics  
- Compare models with tables and visualizations  
- Select and justify a best model  

### Locations

| Work | Path |
|------|------|
| Features | `src/researchpilot/features/` |
| Models | `src/researchpilot/models/` |
| Scripts | `scripts/phase2/` |
| Configs | `configs/phase2/` |
| Notebooks | `notebooks/phase2_ml/` |
| Artifacts | `models/` |
| Evaluation | `evaluation/`, `reports/` |

### Exit criteria

- [ ] Baseline + multiple algorithms trained  
- [ ] Fair split and metrics documented  
- [ ] Comparative analysis completed  
- [ ] Selected model justified  

---

## Phase 3 — Fine-tuning, RAG, Dashboard

### Goals

- Prepare instruction / retrieval corpora from Phase 1 data  
- Fine-tune (or PEFT) a language model for research modules  
- Build a RAG pipeline (chunk → embed → retrieve → generate)  
- Ship an interactive dashboard exposing assistant + analytics  
- Run final evaluation  

### Locations

| Work | Path |
|------|------|
| Fine-tune / RAG code | `src/researchpilot/assistant/`, `rag/` |
| Scripts | `scripts/phase3/` |
| Configs | `configs/phase3/` |
| Notebooks | `notebooks/phase3_llm/` |
| API / UI | `backend/`, `frontend/` |
| Benchmarks | `data/benchmark/`, `evaluation/` |

### Exit criteria

- [ ] Fine-tuned or adapter-based model available  
- [ ] RAG returns grounded passages  
- [ ] Four research modules exposed in dashboard  
- [ ] Final evaluation and limitations documented  

---

## Immediate Next Step

Start **Phase 1**: finalize dataset choice, write the data dictionary template, and begin collection into `data/raw/` / `data/external/`.
