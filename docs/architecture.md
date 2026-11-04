# ResearchPilot — System Architecture

## Overview

ResearchPilot is a four-layer system where each layer builds on verified outputs of the previous one.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Data Foundation                                      │
│  OpenAlex API → JSON ingestion → preprocessing → SQLite         │
│  Output: final_dataset.csv (2,000 × 27)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Layer 2 — Intelligence                                         │
│  Feature engineering (27→206→80) → 12 ML models → tuning       │
│  Output: model objects, leaderboard, evaluation metrics         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Layer 3 — Evaluation                                           │
│  Friedman test → Nemenyi post-hoc → R Shiny dashboard          │
│  Output: statistical report, interactive visualisations         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  Layer 4 — Research Assistant                                   │
│  ChromaDB indexing → RAG pipeline → fine-tuned LLM → Streamlit │
│  Output: local Q&A system with citations                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline

```
User question
      │
      ▼
  Embedder ──── all-MiniLM-L6-v2 (384-dim, fully local, ~22 MB)
      │
      ▼
  ChromaDB ──── 5,796 chunks from 2,000 papers
      │          L2 distance, top-5 retrieval, ~41 ms
      │
      ▼
  Context assembly ──── 5 evidence passages with metadata
      │
      ▼
  Ollama ──── Qwen2.5-1.5B (fine-tuned via QLoRA, GGUF Q4_K_M)
      │        first-token latency ~10 s on CPU
      │
      ▼
  Citations ──── built from stored chunk metadata, never from LLM
      │
      ▼
  Answer + source list
```

---

## ML Feature Engineering

```
27 raw columns (OpenAlex metadata)
      │
      ├── Structural features (12)     — text length, paper age, keyword count
      ├── Domain indicators (25)        — medical imaging, NLP, robotics, etc.
      ├── Publisher one-hot (18)        — IEEE, MDPI, Springer, arXiv, etc.
      └── TF-IDF terms (~151)           — title + abstract unigrams
      │
      ▼
206 engineered features
      │
      ├── Near-zero variance filter
      ├── Pearson correlation filter
      ├── Mutual information scoring (top-30)
      ├── ANOVA F-score (top-30)
      ├── Random Forest Gini importance (top-30)
      └── RFE + Logistic Regression (top-30)
      │
      Consensus: feature retained if ≥ 2 of 4 methods select it
      │
      ▼
80 selected features  →  ML models
```

---

## Database Schema

Four-table SQLite database (`database/researchpilot_r.db`):

```
papers          — 2,000 rows, 27 columns (core corpus)
ml_features     — per-paper feature vectors + split assignments
experiment_runs — model training metadata and metrics
load_manifest   — data ingestion log with checksums
```

---

## Key Design Decisions

**Why local-only?**
Data privacy, reproducibility, and no ongoing API costs. The embedding model and LLM run entirely on-device.

**Why LDA performs best?**
The 80-feature space is approximately linearly separable. Publisher binary flags (top-2 features) create clear linear boundaries between OA categories.

**Why ChromaDB?**
Persistent, zero-config, runs embedded in the Python process. No separate vector DB server needed.

**Why Qwen2.5-1.5B?**
Compact enough to run on CPU with acceptable latency (~10 s TTFT), fine-tunable on consumer hardware, and sufficient for grounded summarisation tasks.
