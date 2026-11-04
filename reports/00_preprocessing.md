# Preprocessing Report

**Project:** ResearchPilot  
**Input:** `data/raw/raw_papers.csv`  
**Output:** `data/cleaned/cleaned_papers.csv`

---

## 1. Dataset summary (before cleaning)

| Metric | Value |
|--------|------:|
| Rows | 2000 |
| Columns | 11 |
| Fully duplicate rows | 0 |
| Duplicate DOIs (non-null) | 0 |

### Column data types

| Column | dtype |
|--------|-------|
| `id` | `object` |
| `title` | `object` |
| `abstract` | `object` |
| `publication_year` | `int64` |
| `cited_by_count` | `int64` |
| `language` | `object` |
| `type` | `object` |
| `concepts` | `object` |
| `keywords` | `object` |
| `doi` | `object` |
| `open_access` | `object` |

### Missing values

| Column | Missing count |
|--------|--------------:|
| `id` | 0 |
| `title` | 0 |
| `abstract` | 0 |
| `publication_year` | 0 |
| `cited_by_count` | 0 |
| `language` | 0 |
| `type` | 0 |
| `concepts` | 0 |
| `keywords` | 0 |
| `doi` | 2 |
| `open_access` | 0 |

---

## 2. Cleaning steps applied

1. **Title cleaning** — removed newline characters and collapsed extra whitespace.  
2. **Abstract cleaning** — normalized whitespace; dropped null/empty abstracts.  
3. **`publication_year`** — coerced to integer.  
4. **`cited_by_count`** — coerced to integer.  
5. **Language filter** — kept only English records (`en` / `english`).  
6. **DOI deduplication** — removed duplicate DOIs, keeping the highest `cited_by_count`. Rows without DOI were retained.

---

## 3. Rows removed by step

| Step | Rows removed |
|------|-------------:|
| Null / empty abstracts | 0 |
| Invalid year / citation values | 0 |
| Non-English language | 0 |
| Duplicate DOIs | 0 |
| **Total removed** | **0** |

---

## 4. Dataset summary (after cleaning)

| Metric | Value |
|--------|------:|
| Rows | 2000 |
| Columns | 11 |
| Fully duplicate rows | 0 |
| Duplicate DOIs (non-null) | 0 |
| Empty titles | 0 |

### Column data types

| Column | dtype |
|--------|-------|
| `id` | `object` |
| `title` | `object` |
| `abstract` | `string` |
| `publication_year` | `int32` |
| `cited_by_count` | `int32` |
| `language` | `object` |
| `type` | `object` |
| `concepts` | `object` |
| `keywords` | `object` |
| `doi` | `object` |
| `open_access` | `object` |

### Missing values

| Column | Missing count |
|--------|--------------:|
| `id` | 0 |
| `title` | 0 |
| `abstract` | 0 |
| `publication_year` | 0 |
| `cited_by_count` | 0 |
| `language` | 0 |
| `type` | 0 |
| `concepts` | 0 |
| `keywords` | 0 |
| `doi` | 2 |
| `open_access` | 0 |

---

## 5. Notes

- Filters for year, language, type, and abstract presence were already applied at collection time via the OpenAlex API; this pipeline re-validates and hardens the dataset for Phase 1 analysis.
- `concepts`, `keywords`, and `open_access` remain JSON-encoded strings for downstream parsing.
- Papers with missing DOI are kept if they otherwise pass quality checks.
