"""
Phase 1 feature extraction from cleaned OpenAlex paper metadata.

Reads:  data/cleaned/cleaned_papers.csv
Writes: data/processed/feature_extracted.csv

Parses JSON columns (concepts, keywords, open_access) into readable fields
while keeping all original columns unchanged.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "cleaned" / "cleaned_papers.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "feature_extracted.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_extraction")


def parse_json_cell(value: Any) -> Any:
    """Parse a JSON string cell; return None on empty/invalid values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def concepts_to_clean(value: Any) -> str:
    """Join concept display names into a readable semicolon-separated string."""
    parsed = parse_json_cell(value)
    if not isinstance(parsed, list):
        return ""
    names = [
        str(item.get("display_name")).strip()
        for item in parsed
        if isinstance(item, dict) and item.get("display_name")
    ]
    return "; ".join(names)


def keywords_to_clean(value: Any) -> str:
    """Join keyword display names into a readable semicolon-separated string."""
    parsed = parse_json_cell(value)
    if not isinstance(parsed, list):
        return ""
    names = [
        str(item.get("display_name")).strip()
        for item in parsed
        if isinstance(item, dict) and item.get("display_name")
    ]
    return "; ".join(names)


def extract_open_access_fields(value: Any) -> dict[str, Any]:
    """Extract open-access fields from the JSON open_access object."""
    parsed = parse_json_cell(value)
    if not isinstance(parsed, dict):
        return {
            "is_open_access": pd.NA,
            "oa_status": pd.NA,
            "oa_url": pd.NA,
            "has_fulltext": pd.NA,
        }
    return {
        "is_open_access": parsed.get("is_oa"),
        "oa_status": parsed.get("oa_status"),
        "oa_url": parsed.get("oa_url"),
        "has_fulltext": parsed.get("any_repository_has_fulltext"),
    }


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable feature columns without modifying original columns."""
    out = df.copy()

    out["concepts_clean"] = out["concepts"].map(concepts_to_clean)
    out["keywords_clean"] = out["keywords"].map(keywords_to_clean)

    oa_fields = out["open_access"].map(extract_open_access_fields)
    oa_df = pd.DataFrame(list(oa_fields))
    out["is_open_access"] = oa_df["is_open_access"].astype("boolean")
    out["oa_status"] = oa_df["oa_status"]
    out["oa_url"] = oa_df["oa_url"]
    out["has_fulltext"] = oa_df["has_fulltext"].astype("boolean")

    return out


def main() -> int:
    if not INPUT_CSV.exists():
        logger.error("Input file not found: %s", INPUT_CSV)
        return 1

    logger.info("Reading %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d rows × %d columns", df.shape[0], df.shape[1])

    required = {"concepts", "keywords", "open_access"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", ", ".join(sorted(missing)))
        return 1

    featured = extract_features(df)
    new_cols = [
        "concepts_clean",
        "keywords_clean",
        "is_open_access",
        "oa_status",
        "oa_url",
        "has_fulltext",
    ]
    logger.info("Added columns: %s", ", ".join(new_cols))
    logger.info(
        "Open access rate: %.1f%% | Fulltext available: %.1f%%",
        100 * featured["is_open_access"].fillna(False).mean(),
        100 * featured["has_fulltext"].fillna(False).mean(),
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(
        "Saved %d rows × %d columns → %s",
        featured.shape[0],
        featured.shape[1],
        OUTPUT_CSV,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
