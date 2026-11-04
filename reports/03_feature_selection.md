# Phase 2 / M3 — Feature Selection & Preprocessing Report

**Project:** ResearchPilot – A Human-Centered AI Research Assistant  
**Milestone:** M3 — Multi-method feature selection, scaling, label encoding  
**Date:** 2026-08-07  
**Status:** Complete  
**Depends on:** M2 (`data/ml/X_oa_category_full.csv` and related artifacts)

---

## 1. Objective

M2 built full feature matrices and a first mutual-information shortlist.  
M3 hardens that into a **train-ready modeling package** for M4 by:

1. Comparing **four** selection methods on the OA track (train only)
2. Building a **consensus** final feature set (k = 30)
3. Fitting **StandardScaler** on structural columns (train only)
4. Fitting **LabelEncoder** for `oa_category` and `impact_tier`
5. Saving a reusable **preprocessor bundle** for training scripts

Phase 1 files and M2 split assignments are not changed.

---

## 2. Inputs / outputs

| Direction | Path |
|-----------|------|
| Config | `configs/phase2/feature_selection.yaml` |
| Code | `src/researchpilot/features/m3_selection.py` |
| Script | `scripts/phase2/03_feature_selection.py` |
| Input (M2) | `data/ml/X_oa_category_full.csv`, `y_oa_category.csv`, impact counterparts |
| Output dir | `data/ml/m3/` |
| Plot | `reports/figures/m3_feature_selection_agreement.png` |
| M4 pointers | updated in `configs/phase2/ml_experiment.yaml` |

---

## 3. Selection protocol (no leakage)

| Step | Detail |
|------|--------|
| Data | OA full matrix from M2 (107 features) |
| Fit scope | **train** rows only (`split == train`) |
| Variance filter | `VarianceThreshold(1e-8)` — none dropped in this run |
| k | 30 |
| Consensus | Feature must appear in ≥ **2** method top-30 lists; pad to 30 by MI rank if needed |

### Methods compared

| Method | Role |
|--------|------|
| Mutual information | Non-linear dependency with label |
| ANOVA F (`f_classif`) | Linear class-separation score |
| Random Forest importance | Embedded tree-based ranking |
| RFE + Logistic Regression | Wrapper selection (features scaled inside RFE) |

---

## 4. Results (this run)

| Metric | Value |
|--------|--------|
| OA input features | 107 |
| After variance filter | 107 |
| Final selected | **30** |
| Train / val / test | 1399 / 300 / 301 |

### Method agreement with final set

| Method | Overlap with final top-30 | % |
|--------|---------------------------|---|
| mutual_info | 20 / 30 | 66.67% |
| anova_f | 22 / 30 | 73.33% |
| random_forest | 15 / 30 | 50.00% |
| rfe_logreg | 19 / 30 | 63.33% |

Plot: `reports/figures/m3_feature_selection_agreement.png`  
CSV: `data/ml/m3/method_agreement_oa.csv`
### Final OA features (examples)

Structural signals retained in the consensus set include **`abstract_length`** and **`keyword_count`**.  
Most selected columns are TF-IDF terms (e.g. domain / method language such as `machine learning`, `artificial intelligence`, `chatgpt`, `medical`, `detection`), which is expected for an accessibility/content prediction task.

Full list: `data/ml/m3/final_feature_list_oa.json`

---

## 5. Scaling & encoding

### StandardScaler (train-fit)

Applied only to structural columns (TF-IDF left as produced by M2 vectorizer):

- OA: `publication_year`, `paper_age`, `title_length`, `abstract_length`, `keyword_count`, `concept_count`, `recent_paper`
- Impact: same plus `is_open_access`, `has_fulltext`

### LabelEncoder (train-fit)

| Target | Classes (encoder order) |
|--------|-------------------------|
| `oa_category` | closed, fully_open, partially_open |
| `impact_tier` | high, low, medium |

---

## 6. Artifacts for M4

| File | Use |
|------|-----|
| `data/ml/m3/X_oa_final.csv` | Full selected + scaled OA matrix + id/split |
| `data/ml/m3/y_oa_final.csv` | Labels + encoded labels |
| `data/ml/m3/oa_train.csv` / `oa_val.csv` / `oa_test.csv` | Split-ready X |
| `data/ml/m3/oa_train_y.csv` / `oa_val_y.csv` / `oa_test_y.csv` | Split-ready y |
| `data/ml/m3/X_impact_scaled.csv` | Impact track matrix (scaled) |
| `data/ml/m3/y_impact_final.csv` | Impact labels |
| `data/ml/m3/preprocessor_bundle.joblib` | Scalers, encoders, final feature list |
| `data/ml/m3/selection_scoreboard_oa.csv` | Per-feature scores across methods |
| `data/ml/m3/m3_selection_summary.json` | Machine-readable summary |

`configs/phase2/ml_experiment.yaml` now points M4 paths at these `data/ml/m3/oa_*` files.

---

## 7. How to reproduce

```bash
# prerequisites
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py

# M3
python scripts/phase2/03_feature_selection.py
```

---

## 8. DA2 rubric mapping (M3)

| Requirement | M3 contribution |
|-------------|-----------------|
| Feature engineering & selection (1 mark) | Completes FE mark with multi-method comparison, consensus set, scaling/encoding docs |
| Progress / documentation | This report + agreement figure + JSON manifests |

M3 does **not** train the 10–15 algorithms (that is **M4**).

---

## 9. Ready for M4

Train classifiers on:

- `data/ml/m3/oa_train.csv` + `oa_train_y.csv`
- Validate on `oa_val.*`
- Final score on `oa_test.*`
- Target column: `oa_category` (or `oa_category_encoded`)
- Feature columns: all except `id`, `split`

Preprocessor for inverse transforms / class names:  
`data/ml/m3/preprocessor_bundle.joblib`

---

## 10. Conclusion

M3 is complete. ResearchPilot now has a documented, leakage-aware, consensus-selected feature set with persisted scaling and label encodings — the correct handoff into multi-model training (M4).
