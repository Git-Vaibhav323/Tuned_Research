# ResearchPilot

A locally-running AI research assistant built over 2,000 OpenAlex AI/ML papers.
Combines a machine learning classification pipeline, an R Shiny analytics dashboard,
and a fully local RAG-powered chat interface — no cloud APIs, no data leaves your machine.

---

## What it does

- **Open-access prediction** — classifies whether a paper is fully open, partially open,
  or closed, using 80 engineered metadata features and 12 ML models
- **Research chat** — ask natural language questions about the 2,000-paper corpus;
  answers are grounded in retrieved evidence with traceable citations
- **Analytics dashboard** — interactive R Shiny app for exploring model performance,
  confusion matrices, feature importance, and topic clusters
- **Paper analysis** — upload a PDF or paste text to summarise, compare, or find research gaps

---

## Quick start

```powershell
# 1. Make sure Ollama is running with the model loaded
ollama list   # should show researchpilot:latest

# 2. Launch the Streamlit app
cd E:\Tuned_Research
py -3.12 -m streamlit run streamlit_app/app.py
```

Or double-click **`LAUNCH_CHATBOT.bat`**.

First launch: go to **⚙️ Index Management** → click **Index full corpus** (~3 min).
The index persists at `data/rag_vector_store/` so subsequent launches are instant.

---

## Launch the R Shiny dashboard

```r
setwd("E:/Tuned_Research")
shiny::runApp("dashboard")
```

---

## Project structure

```
ResearchPilot/
│
├── streamlit_app/          # Main Streamlit UI (8 pages)
├── rag/                    # RAG pipeline — retriever, generator, citations, indexer
├── dashboard/              # R Shiny + Plotly interactive dashboard
│
├── data/
│   ├── final/              # Locked corpus — 2,000 papers × 27 columns
│   ├── ml/                 # Feature matrices and splits
│   ├── ml_r/               # R-side model objects and results
│   └── rag_vector_store/   # ChromaDB persistent index (gitignored)
│
├── ml/
│   ├── r_pipeline/         # R scripts — feature engineering through statistical analysis
│   └── python_pipeline/    # Python scripts — data collection through clustering
│
├── models/
│   ├── checkpoints/        # Saved joblib model checkpoints
│   └── researchpilot-lora/ # Fine-tuned LoRA adapter
│
├── configs/
│   ├── data_collection/    # OpenAlex collection settings
│   ├── ml_pipeline/        # ML experiment, feature, tuning configs
│   └── rag_pipeline/       # RAG, fine-tuning, dashboard configs
│
├── docs/                   # Setup guides and architecture docs
├── reports/                # ML pipeline reports and analysis markdown
│   ├── figures/
│   │   ├── eda/            # Exploratory data analysis charts
│   │   ├── models/         # Model comparison and evaluation charts
│   │   └── publication/    # High-resolution publication figures
│   └── tables/             # CSV result tables
│
├── training/               # LLM fine-tuning scripts (QLoRA / GGUF export)
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Reusable Python package (researchpilot)
├── evaluation/             # RAG evaluation harness and results
├── database/               # SQLite database files
├── ollama/                 # Ollama Modelfile
│
├── run_app.py              # Launcher (pre-loads embedding model)
├── rag_evaluation.py       # Run all RAG evaluation tests
├── test_rag_cli.py         # Quick CLI test for a single RAG query
├── LAUNCH_CHATBOT.bat      # One-click Streamlit launcher
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not committed)
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Data source | OpenAlex API |
| Database | SQLite |
| ML pipeline | R / caret + Python / scikit-learn |
| Vector store | ChromaDB 1.5.9 |
| Embeddings | all-MiniLM-L6-v2 (384-dim, local) |
| LLM | Qwen2.5-1.5B fine-tuned via QLoRA, served by Ollama |
| Dashboard | R Shiny + Plotly |
| UI | Streamlit |

---

## Corpus

- **2,000 AI/ML papers** from OpenAlex (2022–2025)
- **27 raw metadata columns** → **206 engineered features** → **80 selected features**
- **3-class target** — fully open / partially open / closed
- **Best model** — SVM-RBF (tuned): Macro-F1 = 0.5701, Accuracy = 57.48%
- **RAG index** — 5,796 chunks, ~41 ms retrieval latency

---

## ML pipeline results

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| SVM-RBF (tuned) | **0.5701** | **57.48%** |
| LDA | 0.5696 | 57.48% |
| MLP | 0.5657 | 56.81% |
| Logistic Regression | 0.5631 | 56.48% |
| XGBoost | 0.3428 | 34.55% |

Statistical validation: Friedman χ²(11) = 43.46, p = 9×10⁻⁶ — performance differences are not due to chance.

---

## RAG evaluation

39/40 automated tests pass. Covers index integrity, retrieval quality,
citation correctness, latency, and generation grounding.

```powershell
py -3.12 rag_evaluation.py
```

---

## Docs

| Document | Location |
|----------|----------|
| RAG setup guide | `docs/RAG_SETUP_WINDOWS.md` |
| RAG implementation report | `docs/RAG_IMPLEMENTATION_REPORT.md` |
| Architecture overview | `docs/architecture.md` |
| LLM training report | `docs/LLM_TRAINING_REPORT.md` |

---

## Requirements

- Python 3.12
- R 4.6+
- Ollama (with `researchpilot:latest` model)
- ~4 GB disk space for models and vector store

```powershell
py -3.12 -m pip install -r requirements.txt
```
