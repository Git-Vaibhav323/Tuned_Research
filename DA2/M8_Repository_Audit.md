# M8 — Repository Audit Summary

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Date:** 2026-08-15  
**Scope:** Read-only audit before M8 documentation work  

---

## Pipeline (as implemented)

```text
DATA → PREPROCESSING → FEATURE ENGINEERING → DATABASE
  → FEATURE SELECTION → ML (M4) → TUNING (M5) → COMPARISON (M6)
  → IMPACT TIER + CLUSTERING (M7)
```

---

## What is complete

| Area | Status | Evidence |
|------|--------|----------|
| Phase 1 OpenAlex collection | Done | `scripts/phase1/collect_openalex.py`, `data/raw/` |
| Preprocessing / extraction / engineering | Done | Phase 1 scripts → `data/final/final_dataset.csv` (2000×27) |
| R + Python EDA | Done | `r/scripts/`, `reports/figures/01_*.png` … |
| M1 SQLite DB | Done | `database/researchpilot.db`, schemas, queries |
| M2 features + splits + impact_tier | Done | `data/ml/`, `label_maps.json` |
| M3 feature selection | Done | `data/ml/m3/` |
| M4 14 OA models | Done | `evaluation/reports/m4/`, checkpoints |
| M5 tuning (5 models) | Done | `evaluation/reports/m5/` |
| M6 comparative viz | Done | `reports/figures/m6/` |
| M7 impact + clustering | Done + **FROZEN** | `evaluation/reports/m7/`, `reports/m7_final_qa.md` |
| Milestone reports M1–M7 | Done | `DA2/`, `reports/phase2_m*.md` |

---

## What was missing (M8 gaps at audit time)

- Root `README.md` still described a scaffold (stale)
- Final DA2 report pack
- Architecture figure under `reports/figures/final/`
- Live demo script + viva Q&A
- Reproducibility guide
- DA2 completion checklist
- Cleanup recommendations (non-destructive)
- Dedicated database demo script for faculty live run

---

## What is duplicated (intentional mirrors)

- Milestone reports mirrored in `reports/` and `DA2/`
- M7 figures mirrored in `reports/figures/m7/` and `DA2/figures/m7/`
- Leaderboard CSVs copied into `DA2/` for pack visibility (`evaluation/reports/` is gitignored)
- M7 report naming stubs (`…_impact_clustering…` → canonical `…_impact_and_clustering…`)
- Phase 1 EDA figures in both `r/visualizations/` and `reports/figures/`

---

## What should be documented in M8

- End-to-end architecture (Phase 1 vs Phase 2)
- Feature engineering + leakage prevention
- Database connectivity (Python → SQL → DataFrame)
- Full ML inventory (14 OA algorithms; 12 impact models; KMeans)
- Honest M5 before/after + M6 metric leaders
- Frozen M7 impact + exploratory clustering
- Demo script, viva pack, reproducibility, checklist

---

## What should NOT be changed

- Verified M4 / M5 / M6 / M7 metrics and champion narratives
- `data/final/final_dataset.csv` (Phase 1 read-only)
- Train-only impact thresholds (`q_low≈87.33`, `q_high≈134.0`)
- Existing model checkpoints unless a full intentional rebuild is requested
- Phase 3 product claims (not implemented)

### Frozen baselines

| Track | Best by | Value |
|-------|---------|------:|
| OA | Accuracy / Macro-F1 | m4_adaboost **0.482** / **0.452** |
| OA | ROC-AUC | m5_extra_trees_tuned **0.628** |
| Impact | Champion | AdaBoost (val F1 **0.600**; test F1 **0.543**; Acc **0.551**; ROC **0.680**) |
| Clustering | k / silhouette | **8** / **0.064** |
