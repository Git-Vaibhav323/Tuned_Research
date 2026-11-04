# Notebooks

Exploratory and experimental notebooks are organized by project phase.

| Folder | Phase | Focus |
|--------|-------|--------|
| `phase1_eda/` | Phase 1 | Dataset profiling, cleaning checks, EDA, visualization |
| `phase2_ml/` | Phase 2 | Feature exploration, model experiments, comparative analysis |
| `phase3_llm/` | Phase 3 | Fine-tuning prototypes, RAG experiments, dashboard UX notes |

## Conventions

- Prefer calling reusable code from `src/researchpilot/` instead of burying logic only in notebooks.
- Export figures/tables to `reports/figures/` and `reports/tables/`.
- Document dataset version and config path at the top of each notebook.
- Do not commit large intermediate outputs or secrets.
