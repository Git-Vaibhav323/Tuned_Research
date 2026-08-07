# ResearchPilot — DA2 Progress Report (M1–M3)

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Course deliverable:** Digital Assignment 2 (Phase 2)  
**Coverage:** Milestones **M1**, **M2**, **M3** (completed)  
**Date:** 2026-08-07  
**Phase 1 status:** Complete (locked — not modified by Phase 2)

---

## 1. Executive summary

Phase 2 continues directly from the Phase 1 OpenAlex corpus (`data/final/final_dataset.csv`, **2,000** AI/ML papers, **27** columns).

| Milestone | Theme | Status |
|-----------|--------|--------|
| **M1** | Database connectivity (SQLite) | Done |
| **M2** | Feature engineering, splits, impact tiers | Done |
| **M3** | Multi-method feature selection + preprocessing | Done |
| **M4** | Train 10–15 ML algorithms (OA category) | Done — see [M4_Model_Training_Report.md](M4_Model_Training_Report.md) |
| M5–M8 | Tuning, comparison viz, impact track, final docs | Pending |

**Primary ML task (locked):** multiclass prediction of `oa_category`  
(`fully_open` / `partially_open` / `closed`)

**Secondary task (locked):** multiclass `impact_tier`  
(`low` / `medium` / `high`) from train-only citation tertiles

This folder packages the DA2 progress documentation for work completed through M3.

---

## 2. DA2 rubric mapping (progress so far)

| DA2 requirement | Marks | Progress after M1–M3 |
|-----------------|-------|----------------------|
| 1. Feature engineering & feature selection | 1 | **Satisfied** (M2 + M3) |
| 2. Database connectivity | 2 | **Satisfied** (M1 + M2 `ml_features`) |
| 3. 10–15 ML/DL algorithms | 3 | **Satisfied (M4)** — 14 models trained |
| 4. Hyperparameter tuning | 1 | Pending (M5) |
| 5. Comparative performance analysis | 1 | Partial (M4 leaderboard); full pack in M6 |
| 6. Comparative visualizations | 1 | Partial (M3 + M4 plots); ROC/CM suite in M6 |
| 7. Progress demo & documentation (75%) | 1 | **In progress** — DA2 pack documents M1–M4 |

---

## 3. Continuity from Phase 1

| Phase 1 asset | Role in DA2 |
|---------------|-------------|
| `data/final/final_dataset.csv` | Read-only source of truth |
| Engineered features (`paper_age`, lengths, citation transforms, `oa_category`, …) | Reused as structural signals |
| R EDA / reports | Motivation for targets and class balance |
| Cleaning pipeline | Not re-implemented in SQL |

**Rule:** Phase 2 never overwrites Phase 1 CSVs.

---

## 4. Milestone M1 — Database connectivity

**Report:** [M1_Database_Connectivity_Report.md](M1_Database_Connectivity_Report.md)

### Delivered
- Engine: **SQLite** → `database/researchpilot.db`
- Config: `configs/phase2/db.yaml`
- Schemas: `papers`, `ml_features`, `experiment_runs`, `predictions`, `feature_importance`, `load_manifest`
- Load script: `scripts/phase2/01_load_to_db.py`
- Demo SQL: `database/queries/01`–`09`
- FK-safe reload (clears dependents before replacing `papers`)

### Verified results
| Check | Result |
|-------|--------|
| Papers loaded | **2000** |
| Columns | **27** |
| Years | **2022–2025** |
| `oa_category` | fully_open 834 (41.7%), partially_open 723 (36.2%), closed 443 (22.2%) |

### Reproduce
```bash
python scripts/phase2/01_load_to_db.py
python scripts/phase2/run_db_queries.py
```

---

## 5. Milestone M2 — Feature engineering

**Report:** [M2_Feature_Engineering_Report.md](M2_Feature_Engineering_Report.md)

### Delivered
- Stratified splits by `oa_category` (70% / 15% / 15%, `random_state=42`)
- Derived `impact_tier` using **train-only** tertiles of `citation_per_year`
- TF-IDF on title+abstract (fit on train; max 100 features)
- Track-specific leakage exclusions
- Initial MI selection (k=30)
- SQLite `ml_features` populated (2000 rows)
- Dictionary: `data/metadata/phase2_feature_dictionary.md`

### Verified results
| Item | Value |
|------|--------|
| Train / val / test | **1399 / 300 / 301** |
| Impact thresholds (train) | q_low **87.33**, q_high **134.0** |
| Impact counts | low 669, medium 657, high 674 |
| OA full features | **107** |
| Impact full features | **112** |

### Leakage highlights
- **OA track:** excludes `is_open_access`, `oa_status`, `oa_url`, `oa_category`, `open_access`, `has_fulltext`
- **Impact track:** excludes `cited_by_count`, `citation_per_year`, `citation_log`, `impact_tier`

### Reproduce
```bash
python scripts/phase2/02_build_ml_features.py
```

---

## 6. Milestone M3 — Feature selection & preprocessing

**Report:** [M3_Feature_Selection_Report.md](M3_Feature_Selection_Report.md)  
**Figure:** [figures/m3_feature_selection_agreement.png](figures/m3_feature_selection_agreement.png)

### Delivered
- Four selection methods (train only): Mutual Information, ANOVA F, Random Forest importance, RFE + Logistic Regression
- Consensus final set (k=30, ≥2 method votes)
- `StandardScaler` on structural columns (train-fit)
- `LabelEncoder` for both targets
- Preprocessor bundle for M4: `data/ml/m3/preprocessor_bundle.joblib`
- Split-ready files: `data/ml/m3/oa_train.csv`, `oa_val.csv`, `oa_test.csv`

### Verified results
| Item | Value |
|------|--------|
| Input features | 107 |
| Final selected | **30** |
| OA classes (encoder) | closed, fully_open, partially_open |
| Impact classes (encoder) | high, low, medium |

### Method agreement with final top-30
| Method | Overlap | % |
|--------|---------|---|
| mutual_info | 20 / 30 | 66.67% |
| anova_f | 22 / 30 | 73.33% |
| random_forest | 15 / 30 | 50.00% |
| rfe_logreg | 19 / 30 | 63.33% |

### Reproduce
```bash
python scripts/phase2/03_feature_selection.py
```

---

## 7. End-to-end pipeline (completed so far)

```text
Phase 1 final_dataset.csv  (LOCKED)
        │
        ▼
M1  SQLite researchpilot.db  ← papers (2000)
        │
        ▼
M2  Feature matrices + splits + impact_tier + TF-IDF
        │     data/ml/*  +  ml_features table
        ▼
M3  Consensus selection + scaling + label encoding
        │     data/ml/m3/*
        ▼
M4  Train 14 models (champion: AdaBoost)
        │     evaluation/reports/m4/  +  models/checkpoints/m4/
        ▼
M5  Hyperparameter tuning   ← NEXT
```

---

## 8. How to rebuild M1–M4 from scratch

From project root `E:\Tuned_Research`:

```powershell
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py
python scripts/phase2/04_train_models.py
python scripts/phase2/run_db_queries.py
```

**Prerequisites**
- `data/final/final_dataset.csv` must exist locally (gitignored)
- Python packages used: `pandas`, `pyyaml`, `scikit-learn`, `joblib`, `pyarrow`, `matplotlib`

---

## 9. Key artifact index

| Area | Path |
|------|------|
| DB config | `configs/phase2/db.yaml` |
| Feature config | `configs/phase2/features.yaml` |
| Selection config | `configs/phase2/feature_selection.yaml` |
| Experiment config (M4 paths) | `configs/phase2/ml_experiment.yaml` |
| Database | `database/researchpilot.db` |
| M2 matrices | `data/ml/` |
| M3 train-ready | `data/ml/m3/` |
| Feature dictionary | `data/metadata/phase2_feature_dictionary.md` |
| Detailed reports (repo) | `reports/phase2_m1_*.md`, `phase2_m2_*.md`, `phase2_m3_*.md` |

### M4 outputs (complete)
- Leaderboard: `evaluation/reports/m4/leaderboard.csv`
- Checkpoints: `models/checkpoints/m4/` (AdaBoost, Extra Trees, Gradient Boosting)
- Report: [M4_Model_Training_Report.md](M4_Model_Training_Report.md)
- Champion: **AdaBoost** (test macro-F1 ≈ 0.452, accuracy ≈ 0.482)

---

## 10. What remains for full DA2

| Milestone | Work |
|-----------|------|
| **M4** | Done — 14 algorithms trained |
| **M5** | Hyperparameter tuning on top models |
| **M6** | Comparative metrics + ROC / PR / confusion matrix / feature importance |
| **M7** | Impact-tier track + light clustering (research-intelligence narrative) |
| **M8** | Final DA2 demonstration pack and 75% completion documentation |

---

## 11. Folder contents (this `DA2/` pack)

| File | Description |
|------|-------------|
| `README.md` | Index of this folder |
| `DA2_Progress_Report_M1_M2_M3.md` | Consolidated report (updated for M4) |
| `M1_Database_Connectivity_Report.md` | Full M1 report |
| `M2_Feature_Engineering_Report.md` | Full M2 report |
| `M3_Feature_Selection_Report.md` | Full M3 report |
| `M4_Model_Training_Report.md` | Full M4 report |
| `m4_leaderboard.csv` / `m4_leaderboard.md` | Model comparison table |
| `figures/m3_feature_selection_agreement.png` | M3 method-agreement chart |
| `figures/m4_model_comparison_f1.png` | M4 macro-F1 comparison |

---

## 12. Conclusion

Through **M1–M4**, ResearchPilot has SQLite connectivity, leakage-aware features, consensus selection, and **14** trained classifiers for open-access category prediction (**AdaBoost** champion). Next step: **M5 hyperparameter tuning**.
