# Scripts

Command-line entrypoints by phase. Phase 1–2 pipelines are implemented; Phase 3 remains scaffold.

## Phase 1

| Script | Purpose |
|--------|---------|
| `phase1/collect_openalex.py` | Fetch OpenAlex papers → `data/raw/` |
| `phase1/preprocess_papers.py` | Clean → `data/cleaned/` |
| `phase1/feature_extraction.py` | Parse fields → `data/processed/` |
| `phase1/feature_engineering.py` | Engineer → `data/final/final_dataset.csv` |

## Phase 2

| Script | Purpose |
|--------|---------|
| `phase2/01_load_to_db.py` | Load CSV → SQLite |
| `phase2/02_build_ml_features.py` | M2 features / splits / impact_tier |
| `phase2/03_feature_selection.py` | M3 selection + scaling |
| `phase2/04_train_models.py` | M4 OA models (14 algorithms) |
| `phase2/05_tune_hyperparameters.py` | M5 RandomizedSearchCV |
| `phase2/06_comparative_visualizations.py` | M6 plots + tables |
| `phase2/07_impact_and_clustering.py` | M7 impact + KMeans |
| `phase2/08_da2_database_demo.py` | Faculty DB demo (3 SQL queries) |
| `phase2/08_generate_architecture_diagram.py` | Final architecture PNG |
| `phase2/run_db_queries.py` | Run all `database/queries/*.sql` |

Aliases: `train_models.py`, `compare_models.py` (legacy names).

## Phase 3 (scaffold only)

`build_index.py`, `finetune_model.py`, `run_dashboard.py` — not implemented for DA2.

See `DA2/REPRODUCIBILITY.md` for the full command sequence.
