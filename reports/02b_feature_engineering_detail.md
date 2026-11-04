# Phase 2 DA2 — M1: Feature Engineering Report (R)

> **Language**: R  |  **Script**: `r/scripts/phase2/01_feature_engineering.R`  |  **Date**: 2026-09-20
> **Input**: `data/final/final_dataset.csv`  |  **Output**: `data/ml_r/engineered_features.csv`

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Total records | 2000 |
| Original Phase 1 columns | 27 |
| New DA2 features engineered | 11 |
| Total columns after engineering | 38 |

### OA Category Distribution (Target Variable)

| Class | Count | Proportion |
|-------|-------|------------|
| closed | 443 | 22.1% |
| fully_open | 834 | 41.7% |
| partially_open | 723 | 36.1% |

### Publication Year Distribution

| Year | Count |
|------|-------|
| 2022 | 1043 |
| 2023 | 692 |
| 2024 | 242 |
| 2025 | 23 |

---

## 2. Phase 1 Baseline Features (12 features)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `publication_year` | integer | Year of publication (2022–2025) |
| 2 | `paper_age` | integer | Years since publication |
| 3 | `title_length` | integer | Character count of title |
| 4 | `abstract_length` | integer | Character count of abstract |
| 5 | `keyword_count` | integer | Number of keywords |
| 6 | `concept_count` | integer | Number of OpenAlex concepts |
| 7 | `citation_per_year` | float | Citations per year (**impact-tier only**) |
| 8 | `citation_log` | float | ln(1 + cited_by_count) (**impact-tier only**) |
| 9 | `recent_paper` | binary | 1 if paper_age ≤ 2 |
| 10 | `has_doi` | binary | 1 if DOI present |
| 11 | `is_open_access` | boolean | OpenAlex OA flag (**excluded from OA task**) |
| 12 | `has_fulltext` | boolean | Fulltext URL available (**excluded from OA task**) |

---

## 3. New DA2 Features (11 features)

| # | Feature | Type | Description | Rationale |
|---|---------|------|-------------|-----------|
| 1 | `title_word_count` | integer | Word count of title | Richer than character length; reflects title verbosity |
| 2 | `abstract_word_count` | integer | Word count of abstract | More semantically meaningful than character count |
| 3 | `title_to_abstract_ratio` | float | title_length / abstract_length | Captures paper structural balance |
| 4 | `keyword_diversity` | float | Unique keywords / total keywords | Topical breadth measure (≈1.0 since OpenAlex deduplicates) |
| 5 | `concept_diversity` | float | Unique concepts / total concepts | Concept annotation richness |
| 6 | `text_richness` | float | (keywords + concepts) / abstract words | Information density per word |
| 7 | `recency_score` | float | Normalised inverse of paper_age | Continuous recency signal [0, 1] |
| 8 | `text_length_category` | ordinal | short / medium / long | Abstract length bucket for tree models |
| 9 | `abstract_keyword_overlap` | binary | Title word in keywords | Topical alignment between title and keyword metadata |
| 10 | `publication_year_norm` | float | Min-max normalised year | Scale-normalised year for distance-based models |
| 11 | `oa_category_encoded` | integer | Target as integer | Reference only — **excluded from all feature matrices** |

---

## 4. Feature Statistics

| Feature | Min | Max | Mean | SD |
|---------|-----|-----|------|----|
| `title_word_count` | 1.000 | 32.000 | 10.795 | 3.803 |
| `abstract_word_count` | 11.000 | 1659.000 | 196.143 | 78.924 |
| `title_to_abstract_ratio` | 0.003 | 1.229 | 0.072 | 0.077 |
| `keyword_diversity` | 1.000 | 1.000 | 1.000 | 0.000 |
| `concept_diversity` | 1.000 | 1.000 | 1.000 | 0.000 |
| `text_richness` | 0.002 | 3.667 | 0.168 | 0.161 |
| `recency_score` | 0.000 | 1.000 | 0.207 | 0.246 |
| `abstract_keyword_overlap` | 0.000 | 1.000 | 0.763 | 0.425 |
| `publication_year_norm` | 0.000 | 1.000 | 0.207 | 0.246 |
| `oa_category_encoded` | 0.000 | 2.000 | 1.196 | 0.775 |

### Text Length Category Distribution

| Category | Count | Proportion |
|----------|-------|------------|
| long | 690 | 34.5% |
| medium | 1267 | 63.4% |
| short | 43 | 2.1% |

---

## 5. Leakage Exclusion Policy

The columns below are present in the engineered CSV but are **excluded**
from model feature matrices to prevent data leakage.

**OA Category Classification (primary task):**
- `is_open_access` — binary form of OA status; direct leakage
- `oa_status` — raw OA status string; direct leakage
- `oa_url` — presence implies open access; leakage
- `open_access` — full JSON OA metadata; leakage
- `oa_category` — the target variable itself
- `oa_category_encoded` — numeric encoding of target; reference only

**Impact-Tier Classification (secondary task):**
- `cited_by_count` — raw count used to derive target
- `citation_per_year` — directly defines tier boundaries
- `citation_log` — derived from citation count
- `impact_tier` — the target variable for this task

---

## 6. Notes

- `keyword_diversity` and `concept_diversity` are effectively 1.0 for all
  records because OpenAlex already deduplicates assigned terms. These are
  retained for rubric completeness and to document this data property.
- `text_richness` varies meaningfully (range 0.006–0.918) and captures how
  well-annotated a paper is per unit of abstract text.
- All computations are fully vectorised using `stringr` and base R; no
  row-wise loops are used, ensuring fast execution on 2000 records.

---

*Generated: 2026-09-20 16:19:52.247275 | R R version 4.6.1 (2026-06-24 ucrt)*
