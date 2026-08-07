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

If `ml_features` was already filled (M2), a papers reload clears dependent tables first
(`ml_features`, `predictions`) so foreign keys do not block `DELETE FROM papers`.
After reloading papers, rebuild features:

```bash
python scripts/phase2/02_build_ml_features.py
```

## Design rules

1. Cleaning and feature engineering stay in Python (`scripts/phase1/`, later `scripts/phase2/02_*`).  
2. SQL is for storage, analytical queries, and experiment logging — not a second preprocessing pipeline.  
3. Phase 1 CSV remains read-only; reloading the DB never writes back to `data/`.

## Tables

| Table | Status |
|-------|--------|
| `papers` | Populated from Phase 1 final dataset (M1) |
| `load_manifest` | Populated on each successful load (M1) |
| `ml_features` | Populated by `scripts/phase2/02_build_ml_features.py` (M2) |
| `experiment_runs` / `predictions` / `feature_importance` | Schema ready — filled in M4+ |

## Refresh ML features (M2)

```bash
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
```

Reports: `reports/phase2_m1_database_report.md`, `reports/phase2_m2_feature_engineering_report.md`
