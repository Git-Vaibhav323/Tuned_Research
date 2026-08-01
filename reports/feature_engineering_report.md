# Feature Engineering Report

**Project:** ResearchPilot  
**Input:** `data/processed/feature_extracted.csv`  
**Output:** `data/final/final_dataset.csv`  
**Rows:** 2,000  
**Reference year (for age features):** 2026

---

## 1. Purpose

This step converts cleaned / feature-extracted OpenAlex metadata into modeling-ready
numeric and categorical signals for Phase 1 analysis and Phase 2 machine learning.
All original columns are retained; engineered fields are appended.

---

## 2. Engineered features

### 2.1 `paper_age`

| | |
|--|--|
| **Formula** | `paper_age = max(2026 - publication_year, 0)` |
| **Type** | Integer (years) |
| **Why** | Age controls exposure time for citations and supports temporal analysis. Newer papers have had less time to accumulate citations, so raw `cited_by_count` alone can mislead. |

### 2.2 `title_length`

| | |
|--|--|
| **Formula** | `title_length = len(strip(title))` (character count) |
| **Type** | Integer |
| **Why** | Title verbosity can relate to clarity, SEO/indexing behavior, and topic specificity. Useful as a lightweight text-complexity proxy without NLP models. |

### 2.3 `abstract_length`

| | |
|--|--|
| **Formula** | `abstract_length = len(strip(abstract))` (character count) |
| **Type** | Integer |
| **Why** | Abstract length is a proxy for document richness and completeness. Extremely short abstracts may indicate lower-quality metadata or incomplete records. |

### 2.4 `keyword_count`

| | |
|--|--|
| **Formula** | Count of non-empty tokens in `keywords_clean` split on `";"` |
| **Type** | Integer |
| **Why** | Measures topical tagging density from OpenAlex keywords. Higher counts can indicate broader topical coverage or richer indexing. |

### 2.5 `concept_count`

| | |
|--|--|
| **Formula** | Count of non-empty tokens in `concepts_clean` split on `";"` |
| **Type** | Integer |
| **Why** | Captures how many OpenAlex concepts are attached to a paper. Useful for multi-disciplinarity and topical breadth features. |

### 2.6 `citation_per_year`

| | |
|--|--|
| **Formula** | `citation_per_year = cited_by_count / max(paper_age, 1)` |
| **Type** | Float |
| **Why** | Age-normalized impact metric. Dividing by `max(paper_age, 1)` avoids division by zero for papers published in the reference year and makes citations comparable across cohorts. |

### 2.7 `citation_log`

| | |
|--|--|
| **Formula** | `citation_log = ln(1 + cited_by_count)` (`numpy.log1p`) |
| **Type** | Float |
| **Why** | Citation counts are heavily right-skewed. A log transform stabilizes variance and reduces the dominance of a few ultra-high-citation papers in models and plots. |

### 2.8 `recent_paper`

| | |
|--|--|
| **Formula** | `recent_paper = 1 if paper_age ≤ 2 else 0` |
| **Type** | Binary integer (0/1) |
| **Why** | Flags papers from the most recent cohort (age 0–2 years). Useful for stratified EDA and for models that behave differently on emerging vs established literature. |

### 2.9 `has_doi`

| | |
|--|--|
| **Formula** | `has_doi = 1 if strip(doi) is non-empty else 0` |
| **Type** | Binary integer (0/1) |
| **Why** | DOI presence is a metadata-quality and citability signal. Papers without DOIs may be harder to link across scholarly graphs. |

### 2.10 `oa_category`

| | |
|--|--|
| **Formula** | Mapped from `oa_status` (+ `is_open_access` fallback): |
| | `gold`, `diamond` → `fully_open` |
| | `hybrid`, `green`, `bronze` → `partially_open` |
| | `closed` → `closed` |
| | otherwise → `unknown` (or `closed` if `is_open_access` is false) |
| **Type** | Categorical string |
| **Why** | OpenAlex OA statuses are granular; collapsing them into a few categories improves interpretability and reduces sparsity for classical ML models. |

---

## 3. Summary statistics (engineered numeric features)

| Feature | Count | Mean | Std | Min | Median | Max |
|---------|------:|-----:|----:|----:|-------:|----:|
| `paper_age` | 2000 | 3.38 | 0.74 | 1 | 4 | 4 |
| `title_length` | 2000 | 83.88 | 29.00 | 6 | 82 | 232 |
| `abstract_length` | 2000 | 1393.89 | 541.78 | 83 | 1347.50 | 10719 |
| `keyword_count` | 2000 | 12.10 | 3.77 | 2 | 12 | 29 |
| `concept_count` | 2000 | 15.26 | 5.20 | 2 | 15 | 36 |
| `citation_per_year` | 2000 | 177.78 | 686.45 | 59.75 | 105 | 25760.67 |
| `citation_log` | 2000 | 6.00 | 0.53 | 5.48 | 5.83 | 11.26 |

### Binary / categorical features

| Feature | Summary |
|---------|---------|
| `recent_paper` | 265 recent (13.2%) |
| `has_doi` | 1998 with DOI (99.9%) |

#### `oa_category` distribution

| Category | Count | Share |
|----------|------:|------:|
| `fully_open` | 834 | 41.7% |
| `partially_open` | 723 | 36.1% |
| `closed` | 443 | 22.1% |

---

## 4. Data quality notes

- Features were derived only from existing columns; no external joins were performed.
- `citation_per_year` uses `max(paper_age, 1)` so current-year papers are not dropped or set to infinity.
- Empty `keywords_clean` / `concepts_clean` values contribute a count of 0.
- Original JSON and cleaned text columns remain available for Phase 1 EDA and Phase 3 RAG / LLM work.

---

## 5. Output schema (engineered columns only)

```text
paper_age
title_length
abstract_length
keyword_count
concept_count
citation_per_year
citation_log
recent_paper
has_doi
oa_category
```

Full output path: `data/final/final_dataset.csv`
