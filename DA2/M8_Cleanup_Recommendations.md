# M8 Cleanup Recommendations

**Project:** ResearchPilot  
**Date:** 2026-08-15  
**Rule:** No automatic deletion. Await explicit approval before destructive cleanup.

---

## KEEP

| Path | Reason |
|------|--------|
| `data/final/final_dataset.csv` | Phase 1 source of truth |
| `data/ml/` + `data/ml/m3/` | Feature matrices / label maps |
| `database/schemas/`, `database/queries/` | DB definition + demo SQL |
| `models/checkpoints/m4|m5|m7/` | Trained artifacts (local) |
| `evaluation/reports/m4|m5|m6|m7/` | Verified metrics (local; gitignored) |
| `DA2/M*_*.md`, `DA2/figures/` | Faculty pack |
| `reports/phase2_m*.md`, `reports/figures/` | Canonical reports + figures |
| `configs/phase1/`, `configs/phase2/` | Reproducible configs |
| `src/researchpilot/` | Implementation |
| `scripts/phase1/`, `scripts/phase2/01–08_*.py` | Entrypoints |
| `r/scripts/`, `r/visualizations/` | R EDA |

---

## ARCHIVE (optional later)

| Path | Note |
|------|------|
| `DA2/M7_Impact_Clustering_Report.md` | Stub → prefer `M7_Impact_and_Clustering_Report.md` |
| `reports/phase2_m7_impact_clustering_report.md` | Same stub |
| Older progress-only packs once final report exists | Keep until viva ends |
| Duplicate Phase 1 figures if disk pressure | Prefer one canonical tree |

---

## REMOVE (only after approval)

| Path | Reason |
|------|--------|
| `scripts/_tmp_*.py` | Temporary QA helpers (if any remain) |
| `__pycache__/`, `.ipynb_checkpoints/` | Cache |
| `*.pyc` | Bytecode |
| Orphan empty scaffold dirs with only `.gitkeep` if unused | Optional tidy |

---

## DO NOT COMMIT

| Pattern | Why |
|---------|-----|
| `.env`, `.env.*`, API keys | Secrets |
| `database/researchpilot.db` | Large local DB (`*.db` gitignored) |
| `data/raw/**`, `data/final/**`, `data/ml/**` (except approved JSON) | Large / regenerable |
| `models/checkpoints/**`, `*.joblib` | Binaries |
| `evaluation/reports/**` | Local metrics dumps |
| Credential / secrets YAML | Security |

Current `.gitignore` already covers most of these. Verify before any commit that no keys or DB dumps are staged.

---

## Large local files (do not force into git)

- `database/researchpilot.db` (~15 MB)
- `data/raw/raw_papers.csv`, `data/final/final_dataset.csv`
- Joblib checkpoints under `models/checkpoints/`

---

## Approval gate

Destructive cleanup requires an explicit user message such as:

> “Approve M8 cleanup: remove temporary scripts and caches listed under REMOVE.”

Until then: **no deletes**.
