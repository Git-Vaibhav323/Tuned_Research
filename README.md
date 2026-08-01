# ResearchPilot

**A Human-Centered Research Intelligence System**

ResearchPilot helps researchers understand papers, choose statistical methods, improve abstracts, and find research gaps. The project is built as a complete data science workflow first, then extended with fine-tuning, RAG, and an interactive dashboard.

> **Status:** Foundation scaffold  
> No datasets, ML training, RAG, or dashboard runtime are implemented yet.

---

## Development Phases

```text
Phase 1 → Dataset Collection → Preprocessing → EDA → Documentation
                ↓
Phase 2 → Machine Learning → Comparative Analysis
                ↓
Phase 3 → Fine-tuning → RAG → Dashboard
```

| Phase | Focus | Key outputs |
|-------|--------|-------------|
| **1** | Dataset collection, preprocessing, EDA, documentation | `data/processed/`, notebooks, reports, docs |
| **2** | Machine learning and comparative analysis | trained models, metrics, comparison figures |
| **3** | Fine-tuning, RAG, dashboard | adapters/checkpoints, vector index, UI |

Phase 3 does **not** replace Phase 1 or Phase 2. AI features consume validated data and evaluated models.

---

## Core Features (Final Product)

| Module | Purpose |
|--------|---------|
| Adaptive Research Translator | Simplify academic text; key takeaways; examples |
| Statistical Advisor | Recommend tests; rationale; assumptions; alternatives |
| Abstract Improver | Grammar, tone, clarity, novelty framing |
| Research Gap Finder | Compare papers; limitations; future work; gap ideas |

---

## Folder Structure

```text
Tuned_Research/
├── data/
│   ├── raw/                  # Immutable structured source data
│   ├── raw_papers/           # Source PDFs / documents
│   ├── external/             # Third-party reference datasets
│   ├── interim/              # Intermediate transforms
│   ├── processed/            # Analysis-ready datasets
│   ├── instruction_dataset/  # Phase 3 instruction pairs
│   ├── benchmark/            # Held-out evaluation sets
│   └── metadata/             # Data dictionary, licenses, provenance
├── notebooks/
│   ├── phase1_eda/
│   ├── phase2_ml/
│   └── phase3_llm/
├── scripts/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── configs/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── src/researchpilot/
│   ├── data/                 # Phase 1 data utilities
│   ├── features/             # Phase 2 feature engineering
│   ├── models/               # Phase 2 ML/DL workflows
│   ├── visualization/        # Plotting helpers
│   ├── rag/                  # Phase 3 retrieval
│   └── assistant/            # Phase 3 research modules
├── database/                 # Optional SQL schemas / queries
├── r/                        # Optional R visualizations
├── models/                   # Checkpoints, adapters, exports
├── evaluation/               # Metrics and reports
├── reports/                  # Figures and tables for write-ups
├── backend/                  # Phase 3 API scaffold
├── frontend/                 # Phase 3 dashboard scaffold
├── docs/                     # Architecture and phase plans
├── tests/
├── README.md
└── requirements.txt
```

---

## Planned Architecture

```text
Phase 1
Sources → Collection → Cleaning → Preprocessing → EDA → Documentation

Phase 2
Processed Data → Features → ML Models → Metrics → Comparative Analysis

Phase 3
Fine-tuned Model + RAG Index → Research Assistant → Dashboard
```

Phase 3 user flow (product):

```text
User → Upload / Question → Document Processing → Chunking
    → Retrieval → Fine-Tuned LLM → Answer + Sources + Confidence
```

Details: [docs/architecture.md](docs/architecture.md), [docs/project_architecture.md](docs/project_architecture.md)

---

## Technology Stack

| Phase | Technologies |
|-------|----------------|
| **1** | `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `jupyter`, `pymupdf` |
| **2** | `scikit-learn`, `xgboost`, `lightgbm`, `torch`, SQLAlchemy |
| **3** | `transformers`, `peft`, `trl`, `sentence-transformers`, `chromadb`, `langchain`, `fastapi`, `streamlit` |

See [`requirements.txt`](requirements.txt). Packages are listed only—not installed.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture |
| [docs/project_architecture.md](docs/project_architecture.md) | Flow diagrams |
| [docs/dataset_plan.md](docs/dataset_plan.md) | Phase 1 dataset plan |
| [docs/development_roadmap.md](docs/development_roadmap.md) | Phase 1–3 roadmap |
| [docs/project_scope.md](docs/project_scope.md) | Scope and boundaries |
| [docs/benchmark_plan.md](docs/benchmark_plan.md) | Evaluation plan |

---

## Getting Started (later)

1. Clone the repository  
2. Create a virtual environment  
3. Install dependencies from `requirements.txt` when a phase starts  
4. Follow the active phase plan under `docs/development_roadmap.md`  

Foundation work does not require installing packages yet.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep work aligned with the active phase.

## License

[MIT License](LICENSE)

## Changelog

See [CHANGELOG.md](CHANGELOG.md)
