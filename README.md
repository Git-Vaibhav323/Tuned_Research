# ResearchPilot

**A Human-Centered Research Intelligence System**

ResearchPilot is a research-grade **data science and AI platform** that assists researchers across the research lifecycle. It begins with a reproducible data workflow—collection, cleaning, analysis, visualization, databases, feature engineering, and predictive modeling—and culminates in a human-centered AI research assistant.

> **Current status:** Foundation for the Programming for Data Science course  
> The repository is organized around three assessed development phases: **DA1 (Data Foundation), DA2 (Database & Modeling), and DA3 (AI Application)**. No runtime logic is implemented yet.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Course Development Phases](#course-development-phases)
- [Core Features (Final MVP)](#core-features-final-mvp)
- [Folder Structure](#folder-structure)
- [Planned Architecture](#planned-architecture)
- [Technology Stack](#technology-stack)
- [Development Roadmap](#development-roadmap)
- [Future Scope](#future-scope)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)

---

## Project Overview

Academic research is information-dense, methodologically complex, and often inaccessible to early-career researchers, interdisciplinary collaborators, and students. Existing tools typically address isolated tasks (e.g., grammar checking or paper search) without offering an integrated, human-centered intelligence layer.

**ResearchPilot** aims to close that gap by providing four focused, high-value modules in Version 1 (MVP):

| Module | Purpose |
|--------|---------|
| **Adaptive Research Translator** | Simplify academic language and explain concepts |
| **Statistical Advisor** | Recommend and justify statistical tests |
| **Abstract Improver** | Refine grammar, tone, clarity, and novelty framing |
| **Research Gap Finder** | Compare papers and surface limitations / future work |

The system is not LLM-only. Its AI features are the final layer of a complete data science workflow. Datasets, data-quality artifacts, SQL assets, EDA notebooks, R visualizations, engineered features, conventional ML/DL experiments, benchmarks, and application components are first-class repository concerns.

---

## Problem Statement

Researchers frequently struggle with:

1. **Comprehension barriers** — Dense jargon and long-form papers slow understanding, especially for newcomers and cross-domain readers.
2. **Methodological uncertainty** — Choosing the correct statistical test (and knowing its assumptions) remains error-prone without expert guidance.
3. **Writing quality gaps** — Abstracts often lack clarity, academic tone, or a clear novelty signal, reducing paper impact.
4. **Gap discovery friction** — Identifying limitations and open questions across multiple papers is manual, slow, and inconsistent.

Commercial LLM chat interfaces can help ad hoc, but they do not replace a reproducible data science process. ResearchPilot addresses the problem through documented datasets, statistical exploration, comparative predictive modeling, and only then an AI-assisted interface with traceable sources and formal evaluation.

---

## Objectives

### Primary objectives

- Build and document a reproducible research dataset.
- Perform cleaning, preprocessing, EDA, statistical analysis, and Python/R visualization.
- Integrate a database and build reusable feature-engineering pipelines.
- Compare multiple ML/DL algorithms with appropriate evaluation metrics.
- Deliver four reliable, research-oriented AI modules with clear input/output contracts in DA3.
- Establish a scalable layout that connects data, modeling, evaluation, serving, and UI without rework.
- Keep the system **human-centered**: explanations, assumptions, alternatives, and confidence—not opaque answers.

### Secondary objectives

- Enable modular extension (new research tasks without rewriting the core).
- Support open-source contribution with clear docs and contribution guidelines.
- Prepare for responsible AI practices: citation, uncertainty, and evaluation transparency.

---

## Course Development Phases

### DA1 — Data Foundation and Exploration

- Dataset collection, provenance, licensing, and documentation
- Data cleaning, validation, and preprocessing
- Exploratory data analysis in reproducible notebooks
- Statistical summaries and visualizations in Python
- Dedicated **R visualizations** with exported figures

### DA2 — Database and Predictive Modeling

- Relational database integration and documented schemas
- Reusable feature engineering
- Multiple classical ML and deep-learning algorithms
- Consistent train/validation/test methodology
- Model comparison using task-appropriate metrics
- Comparative charts and experiment reports

### DA3 — AI Research Assistant and Product

- LLM integration and the four research-assistant modules
- Optional retrieval and source grounding
- Interactive dashboard and visual analytics
- End-to-end integration and final evaluation

Each phase consumes the validated artifacts of the previous phase. DA3 does not bypass or replace DA1 and DA2.

---

## Core Features (Final MVP)

### 1. Adaptive Research Translator

- Convert academic prose into beginner-friendly language  
- Explain research ideas in plain words  
- Generate key takeaways  
- Illustrate technical concepts with concrete examples  

### 2. Statistical Advisor

- Recommend appropriate statistical tests from study design / data description  
- Explain why a test was selected  
- List assumptions and when they may be violated  
- Suggest alternative tests when assumptions fail  

### 3. Abstract Improver

- Improve grammar and fluency  
- Strengthen academic tone  
- Improve clarity and structure  
- Suggest novelty / contribution framing improvements  

### 4. Research Gap Finder

- Compare multiple papers (abstracts / sections)  
- Identify limitations stated or implied  
- Suggest future work directions  
- Generate research gap ideas for follow-on studies  

---

## Folder Structure

```text
ResearchPilot/
├── data/
│   ├── raw_papers/           # Original PDFs and source documents
│   ├── external/             # Third-party reference datasets
│   ├── interim/              # Intermediate cleaning outputs
│   ├── processed/            # Cleaned / structured text extracts
│   ├── instruction_dataset/  # Instruction–response pairs for fine-tuning
│   ├── benchmark/            # Held-out evaluation sets
│   └── metadata/             # Paper metadata, licenses, provenance
├── notebooks/                # DA1 EDA and DA2 modeling notebooks
├── r/                        # DA1 R scripts and visualizations
├── database/                 # DA2 schemas, migrations, and SQL queries
├── src/researchpilot/        # Reusable data-science source package
│   ├── data/                 # Collection, cleaning, preprocessing
│   ├── features/             # Feature engineering
│   ├── models/               # Conventional ML/DL workflows
│   ├── visualization/        # Python visualization utilities
│   └── assistant/            # DA3 AI orchestration
├── backend/                  # DA3 FastAPI application
│   └── app/
│       ├── api/
│       ├── core/
│       ├── services/
│       ├── models/
│       ├── schemas/
│       └── utils/
├── frontend/                 # DA3 dashboard / user interface
├── models/                   # Checkpoints, adapters, exports
│   ├── checkpoints/
│   ├── adapters/
│   └── exports/
├── evaluation/               # Metrics, runners, reports
│   ├── metrics/
│   └── reports/
├── scripts/                  # Data prep, training, eval entrypoints
├── reports/                  # Course reports, figures, and tables
├── docs/                     # Architecture and planning documents
├── tests/                    # Unit and integration tests
│   ├── unit/
│   └── integration/
├── configs/                  # YAML/JSON configs for train/serve/eval
├── README.md
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── CHANGELOG.md
```

See [docs/](docs/) for detailed design documents.

---

## Planned Architecture

The complete system grows through a data-first pipeline:

```text
DA1: Sources → Raw Data → Cleaning → Preprocessing → EDA → Python/R Visualizations
  ↓
DA2: Database → Feature Engineering → ML/DL Models → Comparison → Evaluation
  ↓
DA3: LLM Integration → Research Assistant → Dashboard → Final Evaluation
```

Module requests are expected to route through a shared orchestration layer (e.g., LangGraph-style workflows) that selects the appropriate skill prompt, retrieval policy, and post-processing rules.

For diagrams and component boundaries, see:

- [docs/architecture.md](docs/architecture.md)
- [docs/project_architecture.md](docs/project_architecture.md)

---

## Technology Stack

| Layer | Planned technologies |
|-------|----------------------|
| **Data collection / processing** | `pandas`, `numpy`, `pymupdf`, `pyarrow` |
| **EDA / statistics** | `scipy`, `statsmodels`, `jupyter`, `seaborn` |
| **R analytics** | R, `tidyverse`, `ggplot2`, `RMarkdown` |
| **Database** | SQLite/PostgreSQL, SQLAlchemy |
| **ML / DL** | `scikit-learn`, `xgboost`, `lightgbm`, `torch` |
| **LLM / Fine-tuning** | `transformers`, `torch`, `datasets`, `accelerate`, `peft`, `trl`, `unsloth` |
| **Retrieval / RAG** | `sentence-transformers`, `chromadb`, `langchain`, `langgraph` |
| **Backend API** | `fastapi`, `uvicorn` |
| **Frontend (MVP UI)** | `streamlit` (later: richer web UI if needed) |
| **Document processing** | `pymupdf` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly`, `ggplot2` |
| **Evaluation** | `bert-score`, `evaluate` |

Dependencies are listed in [`requirements.txt`](requirements.txt). **Do not assume packages are installed**—the foundation only declares planned dependencies.

---

## Development Roadmap

| Course phase | Focus | Status |
|--------------|-------|--------|
| **Foundation** | Repository and architecture | **Complete** |
| **DA1** | Dataset, cleaning, preprocessing, EDA, R visualization | Planned |
| **DA2** | Database, features, ML/DL comparison, evaluation | Planned |
| **DA3** | AI assistant, LLM, dashboard, final evaluation | Planned |

Detailed milestones: [docs/development_roadmap.md](docs/development_roadmap.md)

---

## Future Scope

Beyond MVP, ResearchPilot may expand to:

- Literature review assistants and citation graph exploration  
- Experiment design helpers and power analysis guidance  
- Multimodal paper understanding (figures / tables)  
- Collaborative workspaces and shared project memory  
- Domain packs (biomedicine, HCI, ML, social science)  
- Plugin APIs for institutional research portals  

See [docs/project_scope.md](docs/project_scope.md) for in-scope / out-of-scope boundaries.

---

## Documentation

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | System architecture and component design |
| [project_architecture.md](docs/project_architecture.md) | Detailed architecture diagrams (Markdown) |
| [dataset_plan.md](docs/dataset_plan.md) | Dataset collection and labeling strategy |
| [development_roadmap.md](docs/development_roadmap.md) | Phased delivery plan |
| [project_scope.md](docs/project_scope.md) | Scope, assumptions, non-goals |
| [benchmark_plan.md](docs/benchmark_plan.md) | Evaluation protocol and metrics |

---

## Contributing

Contributions should align with the active DA1, DA2, or DA3 deliverable. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening issues or pull requests.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

**ResearchPilot** — Human-centered research intelligence, built for researchers.
