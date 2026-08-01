# ResearchPilot — Project Architecture Diagrams

**Status:** Planning artifact

---

## 1. Phase Evolution

```text
Phase 1 — DATA SCIENCE FOUNDATION
Research Sources
  ↓
Dataset Collection + Documentation
  ↓
Cleaning → Preprocessing → Processed Data
  ↓
Exploratory Data Analysis + Visualizations
  ↓
Phase 1 Findings and Data Dictionary

Phase 2 — MACHINE LEARNING
Feature Engineering
  ↓
Baseline + Multiple Algorithms
  ↓
Evaluation Metrics
  ↓
Comparative Analysis + Selected Model

Phase 3 — AI APPLICATION
Fine-tuning + RAG Index
  ↓
Research Assistant Modules
  ↓
Interactive Dashboard
  ↓
Final Evaluation
```

---

## 2. Phase 3 End-to-End User Flow

```text
User
  ↓
Upload PDF / Ask Question
  ↓
Document Processing
  ↓
Chunking
  ↓
Retrieval Layer
  ↓
Fine-Tuned Language Model
  ↓
Response Generator
  ↓
Answer + Sources + Confidence
```

---

## 3. Component Map

```text
Phase 1: data/ + src/.../data/ + notebooks/phase1_eda/ + scripts/phase1/
Phase 2: features/ + models/ + notebooks/phase2_ml/ + scripts/phase2/
Phase 3: rag/ + assistant/ + backend/ + frontend/ + notebooks/phase3_llm/
Shared:  configs/ + evaluation/ + reports/ + models/ + docs/
```

---

## 4. RAG Pipeline (Phase 3)

```text
Processed papers/text
  ↓
Chunking (size + overlap)
  ↓
Embeddings (sentence-transformers)
  ↓
Vector store (chromadb)
  ↓
Query → Top-k contexts → Prompt packing → LLM → Grounded answer
```

---

## 5. Fine-tuning Flow (Phase 3)

```text
data/instruction_dataset/
  ↓
configs/phase3/finetune.yaml
  ↓
scripts/phase3/finetune_model.py
  ↓
models/adapters/ or models/checkpoints/
  ↓
backend / dashboard model loader
```

---

## 6. Evaluation Across Phases

```text
Phase 1: data quality + EDA validation
Phase 2: held-out ML metrics + model comparison
Phase 3: assistant benchmarks + RAG grounding + usability
```

---

## 7. Foundation Non-Goals

These diagrams do **not** mean the foundation includes:

- Implemented collection or EDA code  
- Trained ML models  
- Live RAG indexes  
- Running dashboard  

They define the target structure for Phase 1–3 implementation.
