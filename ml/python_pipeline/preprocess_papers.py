"""
Phase 1 preprocessing pipeline for OpenAlex paper metadata.

Reads:  data/raw/raw_papers.csv
Writes: data/cleaned/cleaned_papers.csv
        reports/preprocessing_report.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "raw_papers.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "cleaned" / "cleaned_papers.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "preprocessing_report.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_papers")


def clean_title(series: pd.Series) -> pd.Series:
    """Remove newlines and collapse extra whitespace in titles."""
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.str.replace(r"[\r\n]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.str.strip()


def clean_abstract(series: pd.Series) -> pd.Series:
    """Normalize whitespace in abstracts (null/empty handled separately)."""
    cleaned = series.astype("string")
    cleaned = cleaned.str.replace(r"[\r\n]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.str.strip()


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Build a dataset summary dictionary."""
    missing = df.isna().sum()
    # Treat empty strings as missing for text fields in the summary.
    empty_strings = {
        col: int((df[col].astype("string").fillna("").str.len() == 0).sum())
        for col in df.select_dtypes(include=["object", "string"]).columns
    }
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(v) for col, v in missing.items()},
        "empty_strings": empty_strings,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_dois": int(
            df["doi"].dropna().astype(str).str.strip().str.lower().duplicated().sum()
        )
        if "doi" in df.columns
        else 0,
    }


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply cleaning steps and return cleaned frame plus step counters.
    """
    stats: dict[str, int] = {"input_rows": len(df)}
    out = df.copy()

    # 3. Clean title
    out["title"] = clean_title(out["title"])
    stats["empty_titles_after_clean"] = int((out["title"].str.len() == 0).sum())

    # 4. Clean abstract + drop null/empty abstracts
    out["abstract"] = clean_abstract(out["abstract"])
    before_abstract = len(out)
    out = out[out["abstract"].notna() & (out["abstract"].str.len() > 0)].copy()
    stats["removed_null_or_empty_abstracts"] = before_abstract - len(out)

    # 5–6. Convert numeric fields to integer
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    out["cited_by_count"] = pd.to_numeric(out["cited_by_count"], errors="coerce")
    before_numeric = len(out)
    out = out.dropna(subset=["publication_year", "cited_by_count"]).copy()
    out["publication_year"] = out["publication_year"].astype(int)
    out["cited_by_count"] = out["cited_by_count"].astype(int)
    stats["removed_invalid_numeric"] = before_numeric - len(out)

    # 7. Keep only English records
    before_lang = len(out)
    lang = out["language"].astype("string").str.strip().str.lower()
    out = out[lang.isin(["en", "english"])].copy()
    stats["removed_non_english"] = before_lang - len(out)

    # 8. Remove duplicate papers using DOI
    # Normalize DOI for comparison; keep the row with highest citation count.
    before_doi = len(out)
    out["_doi_norm"] = (
        out["doi"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )
    with_doi = out[out["_doi_norm"].notna()].copy()
    without_doi = out[out["_doi_norm"].isna()].copy()

    with_doi = with_doi.sort_values(
        by=["cited_by_count", "publication_year"],
        ascending=[False, False],
    )
    with_doi = with_doi.drop_duplicates(subset=["_doi_norm"], keep="first")

    out = pd.concat([with_doi, without_doi], ignore_index=True)
    out = out.drop(columns=["_doi_norm"])
    stats["removed_duplicate_dois"] = before_doi - len(out)

    # Stable column order matching the raw schema
    preferred = [
        "id",
        "title",
        "abstract",
        "publication_year",
        "cited_by_count",
        "language",
        "type",
        "concepts",
        "keywords",
        "doi",
        "open_access",
    ]
    cols = [c for c in preferred if c in out.columns] + [
        c for c in out.columns if c not in preferred
    ]
    out = out[cols].reset_index(drop=True)
    stats["output_rows"] = len(out)
    return out, stats


def render_report(
    before: dict,
    after: dict,
    step_stats: dict,
    input_path: Path,
    output_path: Path,
) -> str:
    def fmt_missing(missing: dict) -> str:
        lines = [f"| `{col}` | {count} |" for col, count in missing.items()]
        return "\n".join(lines) if lines else "| _(none)_ | 0 |"

    def fmt_dtypes(dtypes: dict) -> str:
        lines = [f"| `{col}` | `{dtype}` |" for col, dtype in dtypes.items()]
        return "\n".join(lines)

    removed_total = step_stats["input_rows"] - step_stats["output_rows"]

    return f"""# Preprocessing Report

**Project:** ResearchPilot  
**Input:** `{input_path.as_posix()}`  
**Output:** `{output_path.as_posix()}`

---

## 1. Dataset summary (before cleaning)

| Metric | Value |
|--------|------:|
| Rows | {before["n_rows"]} |
| Columns | {before["n_columns"]} |
| Fully duplicate rows | {before["duplicate_rows"]} |
| Duplicate DOIs (non-null) | {before["duplicate_dois"]} |

### Column data types

| Column | dtype |
|--------|-------|
{fmt_dtypes(before["dtypes"])}

### Missing values

| Column | Missing count |
|--------|--------------:|
{fmt_missing(before["missing_values"])}

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
| Null / empty abstracts | {step_stats["removed_null_or_empty_abstracts"]} |
| Invalid year / citation values | {step_stats["removed_invalid_numeric"]} |
| Non-English language | {step_stats["removed_non_english"]} |
| Duplicate DOIs | {step_stats["removed_duplicate_dois"]} |
| **Total removed** | **{removed_total}** |

---

## 4. Dataset summary (after cleaning)

| Metric | Value |
|--------|------:|
| Rows | {after["n_rows"]} |
| Columns | {after["n_columns"]} |
| Fully duplicate rows | {after["duplicate_rows"]} |
| Duplicate DOIs (non-null) | {after["duplicate_dois"]} |
| Empty titles | {step_stats["empty_titles_after_clean"]} |

### Column data types

| Column | dtype |
|--------|-------|
{fmt_dtypes(after["dtypes"])}

### Missing values

| Column | Missing count |
|--------|--------------:|
{fmt_missing(after["missing_values"])}

---

## 5. Notes

- Filters for year, language, type, and abstract presence were already applied at collection time via the OpenAlex API; this pipeline re-validates and hardens the dataset for Phase 1 analysis.
- `concepts`, `keywords`, and `open_access` remain JSON-encoded strings for downstream parsing.
- Papers with missing DOI are kept if they otherwise pass quality checks.
"""


def main() -> int:
    if not INPUT_CSV.exists():
        logger.error("Input file not found: %s", INPUT_CSV)
        return 1

    logger.info("Reading %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    before = summarize_dataframe(df)
    logger.info(
        "Raw summary: %d rows, %d cols, %d duplicate rows, %d duplicate DOIs",
        before["n_rows"],
        before["n_columns"],
        before["duplicate_rows"],
        before["duplicate_dois"],
    )

    cleaned, step_stats = preprocess(df)
    after = summarize_dataframe(cleaned)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info("Saved cleaned dataset: %d rows → %s", len(cleaned), OUTPUT_CSV)

    report = render_report(before, after, step_stats, INPUT_CSV.relative_to(PROJECT_ROOT), OUTPUT_CSV.relative_to(PROJECT_ROOT))
    REPORT_MD.write_text(report, encoding="utf-8")
    logger.info("Wrote report → %s", REPORT_MD)

    logger.info(
        "Done. %d → %d rows (%d removed).",
        step_stats["input_rows"],
        step_stats["output_rows"],
        step_stats["input_rows"] - step_stats["output_rows"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
