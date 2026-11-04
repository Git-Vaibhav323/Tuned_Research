# Phase 2 / M1 — Database Connectivity Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M1 — SQLite schema, Phase 1 ingest, analytical queries  
**Date:** 2026-08-06  
**Status:** Complete  

---

## 1. Objective

Connect the validated Phase 1 dataset to a relational database so that Phase 2 machine learning can:

- read papers from a durable store (not only CSV),
- log experiments and predictions later (M4+),
- demonstrate SQL analytics for DA2 database connectivity marks.

Phase 1 artifacts were **not** modified. The database is a **read ingest** of `data/final/final_dataset.csv`.

---

## 2. Technology choice

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Engine | **SQLite** | Portable, zero server setup, suitable for course demos, already reserved in `database/` |
| Driver | Python `sqlite3` (stdlib) | No extra install required for M1 |
| Config | `configs/phase2/db.yaml` | Paths and schema file list in one place |
| DB file | `database/researchpilot.db` | Local; gitignored (`*.db`) — recreate with the load script |

PostgreSQL/MySQL remain possible later; schema is standard SQL and can be ported if required.

---

## 3. What was delivered

| Artifact | Path |
|----------|------|
| Papers DDL | `database/schemas/01_papers.sql` |
| ML features DDL (empty until M2) | `database/schemas/02_ml_features.sql` |
| Experiments / predictions DDL | `database/schemas/03_experiments.sql` |
| Load manifest DDL | `database/schemas/04_manifest.sql` |
| DB config | `configs/phase2/db.yaml` |
| Connectivity helpers | `src/researchpilot/db/__init__.py` |
| Load script | `scripts/phase2/01_load_to_db.py` |
| Demo queries | `database/queries/01_*.sql` … `07_*.sql` |
| Database README | `database/README.md` |
| Generated DB | `database/researchpilot.db` (~12.9 MB) |

---

## 4. Schema overview

### 4.1 `papers` (populated)

Mirrors all **27** Phase 1 columns. Boolean-like fields stored as SQLite `INTEGER` (0/1); nullable fields (`doi`, `oa_url`, `has_fulltext`) allow `NULL`.

Indexes: `publication_year`, `oa_category`, `oa_status`, `cited_by_count`, `doi`.

### 4.2 Tables reserved for later milestones

| Table | Filled in |
|-------|-----------|
| `ml_features` | M2 — train/val/test features + `impact_tier` |
| `experiment_runs` | M4+ — model runs and metrics |
| `predictions` | M4+ — per-paper predictions |
| `feature_importance` | M4+ — importance scores |
| `load_manifest` | M1 — every successful ingest |

Foreign keys reference `papers(id)` where applicable (`PRAGMA foreign_keys = ON`).

---

## 5. Load procedure

```bash
# from project root E:/Tuned_Research
python scripts/phase2/01_load_to_db.py
```

Steps performed by the script:

1. Resolve project root and read `configs/phase2/db.yaml`
2. Read Phase 1 CSV (**read-only**)
3. Apply schema SQL files
4. Replace rows in `papers` and bulk insert 2,000 records
5. Append a row to `load_manifest`
6. Print verification summary

### 5.1 Load verification (this run)

| Check | Result |
|-------|--------|
| Rows loaded | **2000** |
| Columns | **27** |
| Distinct `oa_category` | **3** |
| Year range | **2022–2025** |
| Phase 1 CSV modified | **No** |
| CSV size | 11,814,018 bytes |
| CSV MD5 (integrity reference) | `1932a02a88c81d3ec3ac87325ac6e8bd` |
| Load timestamp (UTC-ish SQLite `now`) | 2026-08-06 15:48:50 |

Missing values preserved after ingest:

| Column | NULLs in DB |
|--------|-------------|
| `doi` | 2 |
| `oa_url` | 444 |
| `has_fulltext` | 9 |

---

## 6. Analytical query results

Queries live under `database/queries/`. Results below were executed against `researchpilot.db` after the successful load.

### Q1 — Dataset overview

| n_papers | n_years | year_min | year_max | avg_citations | avg_citation_log |
|----------|---------|----------|----------|---------------|------------------|
| 2000 | 4 | 2022 | 2025 | 538.9 | 5.999 |

### Q2 — Open-access category distribution (Phase 2 primary target)

| oa_category | n_papers | pct |
|-------------|----------|-----|
| fully_open | 834 | 41.70% |
| partially_open | 723 | 36.15% |
| closed | 443 | 22.15% |

Classes are usable for multiclass classification without extreme imbalance.

### Q3 — Papers by year

| year | n_papers | avg_citations | avg_paper_age | n_open_access |
|------|----------|---------------|---------------|---------------|
| 2022 | 1043 | 444.1 | 4.0 | 785 |
| 2023 | 692 | 631.8 | 3.0 | 548 |
| 2024 | 242 | 690.0 | 2.0 | 205 |
| 2025 | 23 | 458.9 | 1.0 | 19 |

### Q4 — `oa_status` → `oa_category` mapping

| oa_status | oa_category | n |
|-----------|-------------|---|
| closed | closed | 443 |
| gold | fully_open | 711 |
| diamond | fully_open | 123 |
| hybrid | partially_open | 497 |
| green | partially_open | 175 |
| bronze | partially_open | 51 |

Confirms the Phase 1 engineered target is a coherent collapse of OpenAlex OA status.

### Q5 — Feature means by OA category (signal check for M2)

| oa_category | avg_citations | avg_cite_per_year | avg_title_len | avg_abstract_len | frac_fulltext |
|-------------|---------------|-------------------|---------------|------------------|---------------|
| closed | 418.8 | 127.0 | 80.3 | 1351.1 | 0.000 |
| fully_open | 450.8 | 146.2 | 88.0 | 1425.2 | 0.888 |
| partially_open | 714.2 | 245.4 | 81.3 | 1383.9 | 0.756 |

Closed papers show **no** fulltext flag on average; lengths and keyword/concept counts are similar across classes — useful for leakage-aware feature design in M2.

### Q6 — Load manifest

Recorded source path, DB path, 2000 rows, 27 columns, `phase1_locked = 1`.

### Q7 — Model leaderboard

Empty (0 rows) — expected until models are trained in M4+.

---

## 7. How to re-run demo queries

**Python:**

```python
import sqlite3
from pathlib import Path

db = Path("database/researchpilot.db")
sql = Path("database/queries/02_oa_category_distribution.sql").read_text()
conn = sqlite3.connect(db)
print(conn.execute(sql).fetchall())
conn.close()
```

**CLI (if `sqlite3` is on PATH):**

```bash
sqlite3 database/researchpilot.db < database/queries/01_dataset_overview.sql
```

---

## 8. DA2 rubric mapping (M1)

| Requirement | M1 contribution |
|-------------|-----------------|
| Database connectivity (2 marks) | SQLite DB + Python load + SQL queries + documented schema |
| Progress / documentation | This report + `database/README.md` |
| Does not replace | Feature selection, 10–15 models, tuning, ROC/CM (M2–M6) |

---

## 9. Phase 1 integrity

- Input path only: `data/final/final_dataset.csv`
- Load script does **not** write to `data/`
- Reloading DB is idempotent (`DELETE` + re-insert papers) and safe for demos

---

## 10. Next milestone (M2)

1. Build `ml_features` from `papers` (encode, scale, optional TF-IDF)
2. Derive `impact_tier` for the secondary track
3. Assign stratified `train` / `val` / `test` splits
4. Document leakage exclusions for `oa_category` prediction

---

## 11. Conclusion

M1 is complete: ResearchPilot now has a working SQLite layer with **2,000** Phase 1 papers, provenance logging, and seven analytical queries ready for demonstration. The Phase 1 pipeline remains unchanged; Phase 2 modeling can proceed on top of this database.
