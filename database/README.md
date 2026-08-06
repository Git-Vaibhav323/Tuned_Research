# Database Assets — ResearchPilot Phase 2

**Engine chosen:** SQLite  
**Database file:** `database/researchpilot.db` (generated locally; gitignored)  
**Source of truth:** `data/final/final_dataset.csv` (Phase 1 — never modified by DB scripts)

## Layout

| Path | Role |
|------|------|
| `schemas/` | DDL for papers, ml_features, experiments, manifest |
| `queries/` | Documented analytical SQL for demos and reports |
| `migrations/` | Reserved for future schema versioning |
| `researchpilot.db` | Local SQLite file created by the load script |

## Create / refresh the database (M1)

From the project root:

```bash
python scripts/phase2/01_load_to_db.py
```

Config: `configs/phase2/db.yaml`

## Design rules

1. Cleaning and feature engineering stay in Python (`scripts/phase1/`, later `scripts/phase2/02_*`).  
2. SQL is for storage, analytical queries, and experiment logging — not a second preprocessing pipeline.  
3. Phase 1 CSV remains read-only; reloading the DB never writes back to `data/`.

## Tables (M1)

| Table | Status |
|-------|--------|
| `papers` | Populated from Phase 1 final dataset |
| `load_manifest` | Populated on each successful load |
| `ml_features` | Schema ready — filled in M2 |
| `experiment_runs` / `predictions` / `feature_importance` | Schema ready — filled in M4+ |
