# ResearchPilot

**A locally-running AI research assistant for scholarly literature.**

ResearchPilot indexes 2,000 OpenAlex AI/ML papers into a persistent vector store and answers natural language questions with grounded, citation-backed responses — all running on your own machine with no cloud API calls.

It also ships a full ML classification pipeline (open-access prediction across 12 models) and an interactive R Shiny analytics dashboard.

---

## Features

| Feature | Description |
|---------|-------------|
| RAG Research Chat | Ask questions about 2,000 indexed papers — answers cite the source |
| Open-Access Prediction | 12 ML models classify papers as fully open / partially open / closed |
| R Shiny Dashboard | Interactive model comparison, confusion matrix, ROC curves, feature importance |
| Paper Analysis | Upload a PDF or paste text — summarise, compare, or find research gaps |
| Source Inspector | See retrieved passages and relevance scores for any query |
| 100% Local | No OpenAI, no cloud APIs, no data leaves your machine |

---

## Quick Start

### Prerequisites

- Python 3.12
- R 4.6+
- [Ollama](https://ollama.com) with the model loaded

```powershell
# Check Ollama has the model
ollama list   # must show researchpilot:latest
```

### Install dependencies

```powershell
cd E:\Tuned_Research
py -3.12 -m pip install -r requirements.txt
```

### Launch the Streamlit app

```powershell
py -3.12 -m streamlit run streamlit_app/app.py
```

Or double-click **`LAUNCH_CHATBOT.bat`**.

Open `http://localhost:8501` — on first launch go to **⚙️ Index Management** and click **Index full corpus** (~3 min). The index persists so subsequent launches are instant.

### Launch the R Shiny dashboard

```r
setwd("E:/Tuned_Research")
shiny::runApp("dashboard")
```

---

## Project Structure

```
ResearchPilot/
│
├── streamlit_app/          # 8-page Streamlit UI
│   └── app.py
│
├── rag/                    # RAG pipeline
│   ├── config.py           # All settings — paths, model names, timeouts
│   ├── pipeline.py         # Facade: rag_chat, rag_summarise, rag_compare
│   ├── retriever.py        # Semantic search over ChromaDB
│   ├── generator.py        # Ollama streaming generation
│   ├── vector_store.py     # ChromaDB persistent collection
│   ├── embeddings.py       # all-MiniLM-L6-v2 singleton
│   ├── indexer.py          # Bulk and incremental indexing
│   ├── chunker.py          # Overlapping character chunking
│   ├── citations.py        # Citation formatting from metadata
│   └── document_loader.py  # CSV / PDF / SQLite loaders
│
├── dashboard/              # R Shiny + Plotly dashboard
│   └── app.R               # 4-tab interactive app
│
├── ml/
│   ├── r_pipeline/         # R scripts — feature engineering through stats
│   └── python_pipeline/    # Python scripts — collection through clustering
│
├── data/
│   ├── final/              # Locked corpus — 2,000 papers × 27 columns
│   ├── ml/                 # Feature matrices and train/val/test splits
│   ├── ml_r/               # R-side model objects (.rds) and result CSVs
│   └── rag_vector_store/   # ChromaDB persistent index (not in git)
│
├── models/
│   ├── checkpoints/        # Saved .joblib model checkpoints
│   └── researchpilot-lora/ # Fine-tuned LoRA adapter weights
│
├── configs/
│   ├── data_collection/    # OpenAlex collection settings
│   ├── ml_pipeline/        # ML experiment, feature, tuning configs
│   └── rag_pipeline/       # RAG, fine-tuning, dashboard configs
│
├── docs/                   # Setup guides and architecture docs
├── reports/                # Pipeline analysis reports and result tables
├── training/               # LLM fine-tuning scripts (QLoRA / GGUF export)
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Reusable Python package (researchpilot)
├── evaluation/             # RAG evaluation harness and results
├── database/               # SQLite schema, queries, migrations
│
├── run_app.py              # Pre-loads embedding model then starts Streamlit
├── rag_evaluation.py       # Run all 39+ RAG evaluation tests
├── test_rag_cli.py         # Quick CLI test for a single RAG query
├── LAUNCH_CHATBOT.bat      # One-click Windows launcher
├── launch_streamlit.ps1    # PowerShell launcher with port cleanup
└── requirements.txt        # Python dependencies
```

---

## ML Pipeline Results

Trained and evaluated on 301 held-out test papers (70/15/15 stratified split).

| Rank | Model | Macro-F1 | Accuracy | ROC-AUC |
|------|-------|----------|----------|---------|
| 1 | SVM-RBF (tuned) | **0.5701** | **57.48%** | — |
| 2 | LDA | 0.5696 | 57.48% | 0.7577 |
| 3 | MLP | 0.5657 | 56.81% | 0.7659 |
| 4 | Logistic Regression | 0.5631 | 56.48% | 0.7565 |
| 5 | SVM-RBF (baseline) | 0.5559 | 56.81% | 0.7574 |
| 12 | XGBoost | 0.3428 | 34.55% | 0.5243 |

Statistical validation: Friedman χ²(11) = 43.46, p = 9×10⁻⁶ — performance differences are not due to chance.

Top features: `pub_ieee` (Gini 43.78), `pub_mdpi` (38.48) — publisher flags dominate, confirming open-access status is driven by publisher policy, not paper content.

---

## RAG System

| Metric | Value |
|--------|-------|
| Papers indexed | 2,000 |
| Chunks | 5,796 |
| Embedding model | all-MiniLM-L6-v2 (384-dim, ~22 MB) |
| Vector store | ChromaDB 1.5.9 |
| Retrieval latency | ~41 ms (warm) |
| First-token latency | ~10 s (Qwen2.5-1.5B) |
| Eval pass rate | 39/40 (97.5%) |

Run the evaluation suite:

```powershell
py -3.12 rag_evaluation.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data source | OpenAlex API |
| Database | SQLite |
| ML (Python) | scikit-learn, XGBoost, LightGBM |
| ML (R) | caret, randomForest, e1071 |
| Statistical testing | Friedman + Nemenyi (R / NSM3) |
| Vector store | ChromaDB 1.5.9 |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| LLM | Qwen2.5-1.5B fine-tuned via QLoRA, served by Ollama |
| Dashboard | R Shiny + Plotly |
| UI | Streamlit |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```powershell
Copy-Item .env.example .env
```

Key settings in `rag/config.py`:

```python
OLLAMA_MODEL   = "researchpilot"      # must match `ollama list`
OLLAMA_TIMEOUT = 180                  # seconds
DEFAULT_TOP_K  = 5                    # chunks retrieved per query
CHUNK_SIZE     = 700                  # characters per chunk
```

---

## Docs

| Document | Location |
|----------|----------|
| Full setup guide | `docs/RAG_SETUP_WINDOWS.md` |
| RAG implementation details | `docs/RAG_IMPLEMENTATION_REPORT.md` |
| LLM training guide | `docs/LLM_SETUP_WINDOWS.md` |
| System architecture | `docs/architecture.md` |
| Development roadmap | `docs/development_roadmap.md` |

---

## Running the ML Pipeline

The pipeline is frozen — saved model objects are in `data/ml_r/`. To re-run from scratch:

```r
# In RStudio, set working directory first
setwd("E:/Tuned_Research")

# Run in order
source("ml/r_pipeline/00_enriched_features.R")
source("ml/r_pipeline/02_feature_selection_v2.R")
source("ml/r_pipeline/03_database_setup.R")
source("ml/r_pipeline/05_ml_models_v2.R")
source("ml/r_pipeline/06_hyperparameter_tuning_v2.R")
source("ml/r_pipeline/07_model_evaluation_v2.R")
source("ml/r_pipeline/11_statistical_analysis.R")
```

---

## License

MIT — see `LICENSE`.
