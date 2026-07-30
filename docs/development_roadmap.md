# ResearchPilot — DA1/DA2/DA3 Development Roadmap

**Course:** Programming for Data Science  
**Status:** Foundation complete; implementation deferred

## 1. Delivery Strategy

ResearchPilot is developed as one cumulative data science project. Each assessment phase has a complete, demonstrable outcome and creates stable inputs for the next phase.

```text
DA1: trustworthy data and analysis
  ↓
DA2: database-backed features and evaluated predictive models
  ↓
DA3: AI assistant, dashboard, and end-to-end evaluation
```

The LLM is a DA3 integration component, not the foundation of the repository.

## 2. Foundation — Repository and Architecture

### Deliverables

- [x] Professional repository structure
- [x] Documentation, scope, and contribution guidance
- [x] Data, database, reusable source, R, modeling, reporting, and app boundaries
- [x] Planned dependencies only
- [x] No premature API, UI, ML, or LLM implementation

### Exit criterion

The repository can accept DA1 artifacts without structural redesign.

## 3. DA1 — Dataset and Exploratory Analysis

### Objectives

- Collect a relevant research dataset from legal, documented sources
- Create a data dictionary and provenance/license record
- Assess missing values, duplicates, outliers, consistency, and bias
- Clean and preprocess data through reproducible Python workflows
- Perform univariate, bivariate, and multivariate EDA
- Produce meaningful visualizations in Python and R
- Interpret findings in the context of the research problem

### Repository outputs

| Output | Location |
|--------|----------|
| Source data pointers / allowed raw files | `data/raw_papers/`, `data/external/` |
| Provenance and data dictionary | `data/metadata/`, `docs/dataset_plan.md` |
| Intermediate and clean datasets | `data/interim/`, `data/processed/` |
| Reusable data code | `src/researchpilot/data/` |
| EDA notebooks | `notebooks/` |
| R scripts and plots | `r/scripts/`, `r/visualizations/` |
| Figures and summary tables | `reports/figures/`, `reports/tables/` |

### Exit criteria

- [ ] Data source and license documented
- [ ] Data dictionary completed
- [ ] Cleaning decisions justified and reproducible
- [ ] Processed dataset validated
- [ ] EDA answers defined research questions
- [ ] R visualizations included and interpreted
- [ ] DA1 findings documented

## 4. DA2 — Database, Features, and ML/DL

### Objectives

- Design and integrate an appropriate database
- Load validated DA1 data and demonstrate meaningful SQL queries
- Engineer reproducible features without leakage
- Define a prediction, classification, clustering, or ranking task supported by the data
- Implement a baseline and multiple suitable ML/DL algorithms
- Evaluate all candidate models under a consistent protocol
- Compare performance, complexity, interpretability, and limitations

### Repository outputs

| Output | Location |
|--------|----------|
| Schema and migration assets | `database/schemas/`, `database/migrations/` |
| Analytical SQL | `database/queries/` |
| Feature pipelines | `src/researchpilot/features/` |
| ML/DL workflows | `src/researchpilot/models/` |
| Experiment configs | `configs/` |
| Metrics and evaluation | `evaluation/metrics/` |
| Comparative figures/tables | `reports/figures/`, `reports/tables/` |
| Model artifacts | `models/` |

### Candidate comparisons

At minimum, compare a simple baseline with several task-appropriate families, such as linear models, tree ensembles, gradient boosting, and a compact neural model. Final choices must follow the dataset and target; unsuitable algorithms should not be added solely to inflate model count.

### Exit criteria

- [ ] Database schema and ingestion documented
- [ ] SQL queries demonstrate integration
- [ ] Features are reproducible and leakage-safe
- [ ] Baseline plus multiple ML/DL algorithms evaluated
- [ ] Metrics match the task and class/data characteristics
- [ ] Comparative visualizations and error analysis completed
- [ ] Selected model justified

## 5. DA3 — AI Research Assistant and Dashboard

### Objectives

- Integrate an LLM without discarding DA1/DA2 artifacts
- Implement the four final assistant modules
- Add optional retrieval and source attribution
- Build an interactive dashboard
- Surface DA1 insights and DA2 model comparisons interactively
- Conduct final technical and user-centered evaluation

### Final modules

1. Adaptive Research Translator
2. Statistical Advisor
3. Abstract Improver
4. Research Gap Finder

### Repository outputs

| Output | Location |
|--------|----------|
| Assistant orchestration | `src/researchpilot/assistant/` |
| API/service layer | `backend/` |
| Dashboard | `frontend/` |
| LLM artifacts/configs | `models/`, `configs/` |
| Final benchmarks | `data/benchmark/`, `evaluation/` |
| Final visualizations and report assets | `reports/` |

### Exit criteria

- [ ] Assistant modules work through documented interfaces
- [ ] Sources and uncertainty are presented where applicable
- [ ] Dashboard exposes data, model, and assistant views
- [ ] Interactive visualizations support user exploration
- [ ] DA2 model and DA3 assistant are evaluated separately
- [ ] End-to-end limitations and ethical considerations documented

## 6. Cross-Phase Quality Gates

- Raw data remains immutable.
- Every derived artifact is traceable to inputs and configuration.
- Training and test data remain separated.
- Database, ML, and LLM evaluation results are not conflated.
- No private papers, credentials, or large generated artifacts are committed.
- Documentation is updated with every assessed deliverable.

## 7. Immediate Next Step

Begin DA1 by finalizing the dataset choice, research questions, data dictionary template, collection policy, and EDA plan before writing processing code.
