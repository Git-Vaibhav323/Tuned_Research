"""
Collect research papers from the OpenAlex API into data/raw/raw_papers.csv.

API-side filters:
- publication_year: 2022–2025
- language: English
- type: article
- has_abstract: true
- topics: Artificial Intelligence OR Machine Learning

Requires OpenAlex_API_KEY in a project-root .env file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "raw_papers.csv"
ENV_PATH = PROJECT_ROOT / ".env"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
TARGET_COUNT = 2000
PER_PAGE = 100
SAVE_EVERY = 100
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 60

# OpenAlex concept IDs — OR-combined (Artificial Intelligence | Machine Learning)
CONCEPT_IDS = {
    "Artificial Intelligence": "C154945302",
    "Machine Learning": "C119857082",
}

COLUMNS = [
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_openalex")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OpenAlex_API_KEY") or os.getenv("OPENALEX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OpenAlex API key. Set OpenAlex_API_KEY in the project .env file."
        )
    return api_key.strip()


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain-text abstract from OpenAlex abstract_inverted_index."""
    if not inverted_index:
        return None
    position_to_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word
    if not position_to_word:
        return None
    return " ".join(position_to_word[i] for i in sorted(position_to_word))


def serialize_concepts(concepts: list[dict[str, Any]] | None) -> str:
    if not concepts:
        return "[]"
    simplified = [
        {
            "id": c.get("id"),
            "display_name": c.get("display_name"),
            "level": c.get("level"),
            "score": c.get("score"),
        }
        for c in concepts
    ]
    return json.dumps(simplified, ensure_ascii=False)


def serialize_keywords(keywords: list[dict[str, Any]] | None) -> str:
    if not keywords:
        return "[]"
    simplified = [
        {
            "id": k.get("id"),
            "display_name": k.get("display_name"),
            "score": k.get("score"),
        }
        for k in keywords
    ]
    return json.dumps(simplified, ensure_ascii=False)


def serialize_open_access(open_access: dict[str, Any] | None) -> str:
    if not open_access:
        return "{}"
    return json.dumps(open_access, ensure_ascii=False)


def extract_row(work: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": work.get("id"),
        "title": work.get("title") or work.get("display_name"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "language": work.get("language"),
        "type": work.get("type"),
        "concepts": serialize_concepts(work.get("concepts")),
        "keywords": serialize_keywords(work.get("keywords")),
        "doi": work.get("doi"),
        "open_access": serialize_open_access(work.get("open_access")),
    }


def build_filter() -> str:
    """Build OpenAlex filter string — all constraints applied server-side."""
    concept_or = "|".join(CONCEPT_IDS.values())
    return (
        f"concepts.id:{concept_or},"
        "publication_year:2022-2025,"
        "type:article,"
        "has_abstract:true,"
        "language:en"
    )


def request_with_retries(
    session: requests.Session,
    params: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                OPENALEX_WORKS_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                retry_after = float(
                    response.headers.get("Retry-After", RETRY_BACKOFF_SECONDS * attempt)
                )
                logger.warning(
                    "Rate limited (429). Sleeping %.1fs (attempt %d/%d).",
                    retry_after,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                raise requests.HTTPError(
                    f"Server error {response.status_code}: {response.text[:200]}",
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            sleep_for = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Request failed (%s). Retrying in %.1fs (attempt %d/%d).",
                exc,
                sleep_for,
                attempt,
                MAX_RETRIES,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"OpenAlex request failed after {MAX_RETRIES} retries: {last_error}")


def save_checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Checkpoint saved: %d papers → %s", len(df), path)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def collect_papers(api_key: str, target: int = TARGET_COUNT) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ResearchPilot/0.1 (OpenAlex collector; educational project)",
            "Accept": "application/json",
        }
    )

    params: dict[str, Any] = {
        "filter": build_filter(),
        "per_page": PER_PAGE,
        "cursor": "*",
        "api_key": api_key,
        "select": (
            "id,title,display_name,abstract_inverted_index,publication_year,"
            "cited_by_count,language,type,concepts,keywords,doi,open_access"
        ),
        "sort": "cited_by_count:desc",
    }

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page = 0
    last_saved = 0

    logger.info("Topics: %s", " OR ".join(CONCEPT_IDS))
    logger.info("Target: %d unique papers", target)
    logger.info("Filter: %s", params["filter"])

    while len(rows) < target:
        page += 1
        payload = request_with_retries(session, params)
        results = payload.get("results") or []
        if not results:
            logger.info("No more results from OpenAlex (page %d).", page)
            break

        new_on_page = 0
        for work in results:
            work_id = work.get("id")
            if not work_id or work_id in seen_ids:
                continue
            seen_ids.add(work_id)
            rows.append(extract_row(work))
            new_on_page += 1

            if len(rows) - last_saved >= SAVE_EVERY:
                save_checkpoint(rows, OUTPUT_CSV)
                last_saved = len(rows)

            if len(rows) >= target:
                break

        meta = payload.get("meta") or {}
        total_available = meta.get("count")
        logger.info(
            "Progress: %d / %d | page %d | +%d new | catalog matches≈%s",
            len(rows),
            target,
            page,
            new_on_page,
            total_available,
        )

        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            logger.info("Reached end of cursor pages.")
            break
        params["cursor"] = next_cursor

        time.sleep(0.1)

    if rows:
        save_checkpoint(rows[:target], OUTPUT_CSV)

    return rows[:target]


def main() -> int:
    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Starting OpenAlex collection → %s", OUTPUT_CSV)
    try:
        rows = collect_papers(api_key, TARGET_COUNT)
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        logger.exception("Collection failed: %s", exc)
        return 1

    logger.info("Done. Collected %d papers.", len(rows))
    if len(rows) < TARGET_COUNT:
        logger.warning(
            "Collected fewer than %d papers (%d). Catalog filters may be too strict.",
            TARGET_COUNT,
            len(rows),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
