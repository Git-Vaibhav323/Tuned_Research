# Phase 2 DA2 — M2: Feature Selection Report (R)

> **Language**: R  |  **Script**: `r/scripts/phase2/02_feature_selection.R`  |  **Date**: 2026-09-20
> **Input**: `data/ml_r/engineered_features.csv`
> **Outputs**: `selected_features_oa.csv`, `selected_features_impact.csv`, `feature_selection_metadata.csv`

---

## 1. Methodology

Feature selection was performed on the **OA category classification** task (primary task).
A four-step pipeline was applied, with all thresholds computed on the **training set only**.

| Step | Method | Type | Threshold |
|------|--------|------|-----------|
| 1 | Near-Zero Variance (NZV) | Filter | variance < 1×10⁻¹⁰ OR freq_ratio > 19 |
| 2 | Pairwise Correlation | Filter | Pearson |r| > 0.90 |
| 3 | Mutual Information | Ranking | MI ≥ 0.001 |
| 4 | Random Forest Importance | Embedded | Score ≥ median |

**Consensus rule**: Feature must survive Steps 1+2 AND be selected by ≥ 1 of Steps 3+4.

---

## 2. Candidate Features

- Total engineered columns in input: **38**
- Always-excluded (identifiers, raw text): **11**
- OA-task leakage exclusions: **9**
- Candidate features entering selection: **17**

---

## 3. Filter Results

### Step 1 — NZV Filter
- **Removed** (3): `has_doi`, `keyword_diversity`, `concept_diversity`
- **Retained**: 14
- `keyword_diversity` and `concept_diversity` removed: both are exactly 1.0 for all
  records (OpenAlex does not assign duplicate terms to a single paper).
- `has_doi` removed: 99.9% of records have a DOI — near-constant, no predictive signal.

### Step 2 — Correlation Filter (|r| > 0.90)
- **Removed** (5): `paper_age`, `publication_year`, `title_length`, `abstract_length`, `publication_year_norm`
- **Retained**: 9

---

## 4. Ranking Results and Consensus

| Feature | MI | RF | Votes | MI Score | RF Score | Selected |
|---------|----|----|-------|----------|----------|----------|
| `title_word_count` | YES | YES | 2 | 0.02077 | 98.4533 | **YES** |
| `text_richness` | YES | YES | 2 | 0.01301 | 170.3062 | **YES** |
| `title_to_abstract_ratio` | YES | YES | 2 | 0.01284 | 182.5294 | **YES** |
| `abstract_word_count` | YES | YES | 2 | 0.01103 | 164.7396 | **YES** |
| `concept_count` | YES | YES | 2 | 0.00484 | 108.0478 | **YES** |
| `keyword_count` | YES | — | 1 | 0.00991 | 95.5018 | **YES** |
| `recency_score` | YES | — | 1 | 0.00889 | 45.0452 | **YES** |
| `text_length_category` | YES | — | 1 | 0.00685 | 12.2108 | **YES** |
| `abstract_keyword_overlap` | YES | — | 1 | 0.00310 | 22.2950 | **YES** |

---

## 5. Final Selected Features

### OA Classification Task
- **Before selection**: 17 candidate features
- **After selection**: **9 features**

| # | Feature |
|---|---------|
| 1 | `title_word_count` |
| 2 | `text_richness` |
| 3 | `title_to_abstract_ratio` |
| 4 | `abstract_word_count` |
| 5 | `concept_count` |
| 6 | `keyword_count` |
| 7 | `recency_score` |
| 8 | `text_length_category` |
| 9 | `abstract_keyword_overlap` |

### Impact-Tier Task
- **Feature count**: 19 features + 1 target
- `publication_year`
- `paper_age`
- `title_length`
- `abstract_length`
- `keyword_count`
- `concept_count`
- `has_doi`
- `title_word_count`
- `abstract_word_count`
- `title_to_abstract_ratio`
- `text_richness`
- `recency_score`
- `abstract_keyword_overlap`
- `publication_year_norm`
- `is_open_access_int`
- `has_fulltext_int`
- `oa_cat_fully_open`
- `oa_cat_partially_open`
- `oa_cat_closed`

### Impact-Tier Thresholds (training data only)

| Threshold | R Computed | Python Baseline | Match? |
|-----------|------------|-----------------|--------|
| q_low (33rd pct) | **87.2778** | 87.33 | ~Yes |
| q_high (67th pct) | **137.2222** | 134.0 | ~Yes |

Minor differences from the Python baseline are expected due to random seed
implementation differences between scikit-learn and R's `sample()` for stratification.

---

## 6. Visualisations

- `reports/figures/phase2_r/feature_selection_agreement.png` — method agreement heatmap
- `reports/figures/phase2_r/feature_mi_scores.png` — mutual information bar chart

---

## 7. Method Justification

| Method | Why appropriate here |
|--------|---------------------|
| NZV | Removes zero-variance features that carry no predictive information |
| Correlation | Avoids multicollinearity that would inflate certain models (LR, KNN, SVM) |
| Mutual Information | Model-free, captures non-linear dependencies, appropriate for multiclass |
| RF Importance | Captures interaction effects; consistent with tree-based M4 models |

**Methods NOT used** (and why):
- RFE was omitted because it is computationally expensive on a 17-feature set and
  would provide minimal additional benefit over the four methods above.
- ANOVA-F was omitted because MI captures both linear and non-linear dependencies.

---

*Generated: 2026-09-20 13:59:40.591766 | R R version 4.6.1 (2026-06-24 ucrt)*
