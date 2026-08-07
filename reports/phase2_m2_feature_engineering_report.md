# Phase 2 / M2 — Feature Engineering & Feature Selection Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M2 — ML features, impact tiers, stratified splits, selection  
**Date:** 2026-08-06  
**Status:** Complete  
**Depends on:** M1 (`database/researchpilot.db` with 2,000 papers)

---

## 1. Objective

Prepare a leakage-safe modeling layer on top of Phase 1 so that DA2 can train 10–15 algorithms without touching Phase 1 files.

M2 delivers:

1. Stratified train / val / test splits  
2. Derived secondary target `impact_tier`  
3. Track-specific feature matrices (OA + impact)  
4. TF-IDF text features (fit on train only)  
5. Mutual-information feature selection (primary track)  
6. Population of SQLite `ml_features`  
7. Documented feature dictionary and JSON manifests  

---

## 2. Inputs / outputs

| Direction | Path | Role |
|-----------|------|------|
| Read-only | `data/final/final_dataset.csv` | Phase 1 final table |
| Read | `database/researchpilot.db` | Must contain 2,000 `papers` rows (M1) |
| Config | `configs/phase2/features.yaml` | Features, leakage, splits |
| Config | `configs/phase2/ml_experiment.yaml` | Locked primary/secondary tasks |
| Write | `data/ml/*` | Matrices, labels, splits, manifests |
| Write | SQLite `ml_features` | Per-paper summary store |
| Docs | `data/metadata/phase2_feature_dictionary.md` | Dictionary |
| Code | `src/researchpilot/features/m2_pipeline.py` | Reusable builder |
| Script | `scripts/phase2/02_build_ml_features.py` | Entrypoint |

Phase 1 CSV is never modified.

---

## 3. Modeling tracks (locked)

### Track 1 — Primary (DA2 graded spine)

| Item | Value |
|------|--------|
| Target | `oa_category` |
| Classes | fully_open / partially_open / closed |
| Counts | 834 / 723 / 443 |
| Problem | Multiclass classification |

### Track 2 — Secondary (research-intelligence narrative)

| Item | Value |
|------|--------|
| Target | `impact_tier` (derived) |
| Classes | low / medium / high |
| Counts | 669 / 657 / 674 |
| Source | `citation_per_year` |
| Thresholds (train only) | q_low = **87.3333**, q_high = **134.0** |

---

## 4. Split strategy

| Split | Fraction | Rows | Stratify by |
|-------|----------|------|-------------|
| train | 70% | 1399 | `oa_category` |
| val | 15% | 300 | `oa_category` |
| test | 15% | 301 | `oa_category` |

`random_state = 42` (reproducible).

Impact-tier thresholds are computed on the **train** subset only, then applied to val/test. This avoids constructing labels with full-dataset quantiles.

---

## 5. Feature engineering

### 5.1 Structural features (from Phase 1)

Used for OA track:  
`publication_year`, `paper_age`, `title_length`, `abstract_length`, `keyword_count`, `concept_count`, `recent_paper`

Also for impact track:  
`is_open_access`, `has_fulltext`, plus one-hot `oa_cat_fully_open|partially_open|closed`

Dropped as near-constant: `has_doi` (~99.9% = 1).

### 5.2 Text features

| Setting | Value |
|---------|--------|
| Fields | `title` + `abstract` |
| Method | TF-IDF |
| Fit on | train only |
| max_features | 100 |
| ngram_range | (1, 2) |
| stop_words | english |
| Artifact | `data/ml/tfidf_vectorizer.joblib` |

### 5.3 Matrix sizes (this run)

| Matrix | Features |
|--------|----------|
| OA full | **107** (7 structural + 100 TF-IDF) |
| OA selected | **30** |
| Impact full | **112** (9 structural + 3 OA one-hots + 100 TF-IDF) |

---

## 6. Leakage controls

### OA track — excluded from inputs

`is_open_access`, `oa_status`, `oa_url`, `oa_category`, `open_access`, `has_fulltext`, raw IDs/JSON/text columns (text enters only via TF-IDF).

Citations are also **not** used on the OA track so the model predicts accessibility from content/metadata-style signals.

### Impact track — excluded from inputs

`cited_by_count`, `citation_per_year`, `citation_log`, `impact_tier`, raw OA JSON/status/URL, `has_doi`.

OA category one-hots **are** allowed (metadata available independently of citation outcomes).

---

## 7. Feature selection (primary track)

| Item | Value |
|------|--------|
| Method | Mutual information + `SelectKBest` |
| Fit on | train only |
| Target | `oa_category` |
| k | 30 |

Top mutual-information signals (illustrative): TF-IDF terms such as `machine learning`, `intelligence`, `artificial intelligence`, `proposed`, `knowledge`, plus structural cues like `publication_year` / `abstract_length`.

Full ranking: `data/ml/feature_selection_scores_oa.csv`  
Selected list: `data/ml/feature_lists.json` → `oa_category.selected_features`

---

## 8. Database update

`ml_features` populated with **2000** rows:

| Check | Result |
|-------|--------|
| Row count | 2000 |
| FK orphans vs `papers` | 0 |
| Splits | train 1399 / val 300 / test 301 |

New demo queries:

- `database/queries/08_ml_split_summary.sql`
- `database/queries/09_impact_tier_by_split.sql`

---

## 9. How to reproduce

```bash
# prerequisite
python scripts/phase2/01_load_to_db.py

# M2
python scripts/phase2/02_build_ml_features.py

# optional SQL check
python scripts/phase2/run_db_queries.py
```

---

## 10. Artifact checklist

| Artifact | Purpose |
|----------|---------|
| `data/ml/train.csv`, `val.csv`, `test.csv` | Ready-to-train splits |
| `data/ml/X_oa_category_selected.*` | Primary feature matrix |
| `data/ml/X_impact_tier_full.*` | Secondary feature matrix |
| `data/ml/y_oa_category.csv`, `y_impact_tier.csv` | Labels |
| `data/ml/label_maps.json` | Class maps + counts |
| `data/ml/feature_lists.json` | Leakage + selected features |
| `data/ml/m2_build_summary.json` | Machine-readable summary |
| `data/metadata/phase2_feature_dictionary.md` | Human-readable dictionary |

---

## 11. DA2 rubric mapping (M2)

| Requirement | M2 contribution |
|-------------|-----------------|
| Feature engineering & selection (1 mark) | Structural + TF-IDF + MI selection + dictionary + leakage docs |
| Database connectivity | `ml_features` filled; split/tier SQL queries |
| Progress toward 75% | Modeling-ready matrices for M4 training |

Not in M2 (next): train 10–15 models (M4), hyperparameter tuning (M5), comparative plots (M6).

---

## 12. Ready for next milestones

| Next | Uses M2 outputs |
|------|-----------------|
| **M3** | Optional deeper selection / encoding polish (can merge into M4 if time-boxed) |
| **M4** | Load `train/val/test` + selected OA matrix → 10–15 algorithms |
| **M5** | Tune top models on val / CV |
| **M6** | ROC, PR, confusion matrix, feature importance from champion models |

### Recommended M4 entrypoint files

- Features: `data/ml/X_oa_category_selected.csv`
- Labels: `data/ml/y_oa_category.csv`
- Splits column: `split`
- Config: `configs/phase2/ml_experiment.yaml`

---

## 13. Conclusion

M2 is complete. ResearchPilot now has a reproducible, leakage-aware feature store for open-access classification and relative impact tiers, wired into SQLite and exported under `data/ml/`. Phase 1 remains unchanged. The project is ready to train the Phase 2 model suite.
