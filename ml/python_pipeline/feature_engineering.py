"""
Phase 1 feature engineering for ResearchPilot paper metadata.

Reads:  data/processed/feature_extracted.csv
Writes: data/final/final_dataset.csv
        reports/feature_engineering_report.md

Engineered features
-------------------
paper_age          : years since publication
title_length       : character length of title
abstract_length    : character length of abstract
keyword_count      : number of parsed keywords
concept_count      : number of parsed concepts
citation_per_year  : citations normalized by paper age
citation_log       : log1p(cited_by_count)
recent_paper       : 1 if published within the last 2 years
has_doi            : 1 if a DOI is present
oa_category        : closed | partially_open | fully_open | unknown
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "feature_extracted.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "final" / "final_dataset.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "feature_engineering_report.md"

REFERENCE_YEAR = datetime.now().year
RECENT_PAPER_MAX_AGE = 2  # inclusive: age 0, 1, 2 → recent

FULLY_OPEN_STATUSES = frozenset({"gold", "diamond"})
PARTIALLY_OPEN_STATUSES = frozenset({"hybrid", "green", "bronze"})
CLOSED_STATUSES = frozenset({"closed"})

ENGINEERED_COLUMNS = [
    "paper_age",
    "title_length",
    "abstract_length",
    "keyword_count",
    "concept_count",
    "citation_per_year",
    "citation_log",
    "recent_paper",
    "has_doi",
    "oa_category",
]

REQUIRED_COLUMNS = [
    "publication_year",
    "cited_by_count",
    "title",
    "abstract",
    "keywords_clean",
    "concepts_clean",
    "doi",
    "is_open_access",
    "oa_status",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_engineering")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _as_text(series: pd.Series) -> pd.Series:
    """Normalize a column to nullable string (NaN → empty for length/count ops)."""
    return series.astype("string").fillna("").str.strip()


def count_semicolon_list(series: pd.Series) -> pd.Series:
    """
    Count items in a '; '-delimited list.

    Empty / whitespace-only cells → 0.
    """
    text = _as_text(series)
    counts = text.map(
        lambda value: 0
        if value == ""
        else len([part for part in value.split(";") if part.strip()])
    )
    return counts.astype(int)


def map_oa_category(is_open_access: Any, oa_status: Any) -> str:
    """
    Collapse OpenAlex OA status into modeling-friendly categories.

    fully_open     : gold, diamond
    partially_open : hybrid, green, bronze
    closed         : closed, or is_open_access is False
    unknown        : missing / unrecognized status
    """
    status = "" if pd.isna(oa_status) else str(oa_status).strip().lower()

    if status in CLOSED_STATUSES:
        return "closed"
    if status in FULLY_OPEN_STATUSES:
        return "fully_open"
    if status in PARTIALLY_OPEN_STATUSES:
        return "partially_open"

    # Fall back to boolean OA flag when status is missing/unexpected.
    if pd.isna(is_open_access):
        return "unknown"
    return "closed" if not bool(is_open_access) else "unknown"


def engineer_features(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """
    Append engineered features; leave all original columns unchanged.

    Parameters
    ----------
    df:
        Feature-extracted paper table.
    reference_year:
        Year used to compute paper_age and recent_paper.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out = df.copy()

    year = pd.to_numeric(out["publication_year"], errors="coerce")
    citations = pd.to_numeric(out["cited_by_count"], errors="coerce").fillna(0)

    # paper_age = reference_year - publication_year  (clip at 0 for safety)
    out["paper_age"] = (reference_year - year).clip(lower=0).astype("Int64")

    # title_length / abstract_length — character counts after strip
    out["title_length"] = _as_text(out["title"]).str.len().astype(int)
    out["abstract_length"] = _as_text(out["abstract"]).str.len().astype(int)

    # keyword_count / concept_count — from cleaned list strings
    out["keyword_count"] = count_semicolon_list(out["keywords_clean"])
    out["concept_count"] = count_semicolon_list(out["concepts_clean"])

    # citation_per_year = cited_by_count / max(paper_age, 1)
    age_for_rate = out["paper_age"].fillna(0).astype(float).clip(lower=1)
    out["citation_per_year"] = (citations.astype(float) / age_for_rate).round(4)

    # citation_log = ln(1 + cited_by_count)
    out["citation_log"] = np.log1p(citations.astype(float))

    # recent_paper — published within the last RECENT_PAPER_MAX_AGE years
    out["recent_paper"] = (
        out["paper_age"].fillna(RECENT_PAPER_MAX_AGE + 1) <= RECENT_PAPER_MAX_AGE
    ).astype(int)

    # has_doi — non-empty DOI string
    doi_text = _as_text(out["doi"])
    out["has_doi"] = (doi_text != "").astype(int)

    # oa_category — coarse open-access grouping
    out["oa_category"] = [
        map_oa_category(is_oa, status)
        for is_oa, status in zip(out["is_open_access"], out["oa_status"], strict=True)
    ]

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }


def _fmt_num(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.{digits}f}"


def build_report(
    df: pd.DataFrame,
    reference_year: int,
    input_path: Path,
    output_path: Path,
) -> str:
    """Render Markdown documentation for engineered features."""
    numeric_features = [
        "paper_age",
        "title_length",
        "abstract_length",
        "keyword_count",
        "concept_count",
        "citation_per_year",
        "citation_log",
    ]
    rows = []
    for col in numeric_features:
        stats = _numeric_summary(df[col])
        rows.append(
            "| `{name}` | {count} | {mean} | {std} | {min} | {median} | {max} |".format(
                name=col,
                count=stats["count"],
                mean=_fmt_num(stats["mean"]),
                std=_fmt_num(stats["std"]),
                min=_fmt_num(stats["min"]),
                median=_fmt_num(stats["median"]),
                max=_fmt_num(stats["max"]),
            )
        )

    recent_rate = 100 * df["recent_paper"].mean()
    doi_rate = 100 * df["has_doi"].mean()
    oa_counts = df["oa_category"].value_counts(dropna=False)
    oa_table = "\n".join(
        f"| `{idx}` | {count} | {100 * count / len(df):.1f}% |"
        for idx, count in oa_counts.items()
    )

    return f"""# Feature Engineering Report

**Project:** ResearchPilot  
**Input:** `{input_path.as_posix()}`  
**Output:** `{output_path.as_posix()}`  
**Rows:** {len(df):,}  
**Reference year (for age features):** {reference_year}

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
| **Formula** | `paper_age = max({reference_year} - publication_year, 0)` |
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
| **Formula** | `recent_paper = 1 if paper_age ≤ {RECENT_PAPER_MAX_AGE} else 0` |
| **Type** | Binary integer (0/1) |
| **Why** | Flags papers from the most recent cohort (age 0–{RECENT_PAPER_MAX_AGE} years). Useful for stratified EDA and for models that behave differently on emerging vs established literature. |

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
{chr(10).join(rows)}

### Binary / categorical features

| Feature | Summary |
|---------|---------|
| `recent_paper` | {df["recent_paper"].sum()} recent ({recent_rate:.1f}%) |
| `has_doi` | {df["has_doi"].sum()} with DOI ({doi_rate:.1f}%) |

#### `oa_category` distribution

| Category | Count | Share |
|----------|------:|------:|
{oa_table}

---

## 4. Data quality notes

- Features were derived only from existing columns; no external joins were performed.
- `citation_per_year` uses `max(paper_age, 1)` so current-year papers are not dropped or set to infinity.
- Empty `keywords_clean` / `concepts_clean` values contribute a count of 0.
- Original JSON and cleaned text columns remain available for Phase 1 EDA and Phase 3 RAG / LLM work.

---

## 5. Output schema (engineered columns only)

```text
{chr(10).join(ENGINEERED_COLUMNS)}
```

Full output path: `{output_path.as_posix()}`
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    if not INPUT_CSV.exists():
        logger.error("Input file not found: %s", INPUT_CSV)
        return 1

    logger.info("Reading %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d rows × %d columns", df.shape[0], df.shape[1])

    try:
        featured = engineer_features(df, reference_year=REFERENCE_YEAR)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    featured.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(
        "Saved %d rows × %d columns → %s",
        featured.shape[0],
        featured.shape[1],
        OUTPUT_CSV,
    )

    report = build_report(
        featured,
        reference_year=REFERENCE_YEAR,
        input_path=INPUT_CSV.relative_to(PROJECT_ROOT),
        output_path=OUTPUT_CSV.relative_to(PROJECT_ROOT),
    )
    REPORT_MD.write_text(report, encoding="utf-8")
    logger.info("Wrote report → %s", REPORT_MD)

    logger.info(
        "Engineered features: %s",
        ", ".join(ENGINEERED_COLUMNS),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
