"""
rag/retriever.py — Semantic retrieval of relevant paper passages.

Wraps vector_store.query_chunks with:
  - Configurable top-k and distance threshold.
  - Optional metadata filtering (by year range, source type).
  - Duplicate paper deduplication (already handled in vector_store).
  - "No results" handling with a clear signal.
  - Distance-to-relevance score conversion for display.
"""

from __future__ import annotations

import logging

from rag.config import DEFAULT_TOP_K, MAX_DISTANCE
from rag.vector_store import query_chunks

log = logging.getLogger(__name__)


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = MAX_DISTANCE,
    year_from: int | None = None,
    year_to:   int | None = None,
    source:    str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant paper passages for *question*.

    Args:
        question:     The user's research question.
        top_k:        Maximum number of passages to return.
        max_distance: L2 distance ceiling (lower = more restrictive).
        year_from:    Optional minimum publication year filter.
        year_to:      Optional maximum publication year filter.
        source:       Optional source filter ("openalex_csv", "pdf_upload", etc.).

    Returns a list of hit dicts, each containing:
        chunk_id, doc_id, text, distance, relevance_score,
        title, authors, year, doi, bare_doi, openalex_id,
        oa_category, source, filename, chunk_index, is_fulltext, page_hint.

    Returns an empty list when the index is empty or nothing matches.
    """
    if not question or not question.strip():
        log.debug("retrieve() called with empty question.")
        return []

    # Build optional ChromaDB 'where' filter
    where = _build_filter(year_from, year_to, source)

    hits = query_chunks(
        query_text=question,
        top_k=top_k,
        max_distance=max_distance,
        filter_meta=where,
    )

    # Add a 0–1 relevance score (purely for display; not a probability)
    for hit in hits:
        hit["relevance_score"] = _distance_to_score(hit["distance"])

    if not hits:
        log.info("No relevant chunks found for: %s", question[:80])

    return hits


def retrieve_for_doc(doc_id: str, top_k: int = 8) -> list[dict]:
    """
    Retrieve all indexed chunks for a specific document.
    Useful for summarisation — returns up to top_k chunks from that paper.
    """
    from rag.vector_store import get_collection

    col = get_collection()
    if col.count() == 0:
        return []

    try:
        result = col.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["documents", "metadatas"],
            limit=top_k,
        )
    except Exception as exc:
        log.error("retrieve_for_doc failed for doc_id=%s: %s", doc_id, exc)
        return []

    ids   = result.get("ids",       [])
    docs  = result.get("documents", [])
    metas = result.get("metadatas", [])

    hits = []
    for cid, text, meta in zip(ids, docs, metas):
        hits.append({
            "chunk_id":       cid,
            "text":           text,
            "distance":       0.0,
            "relevance_score": 1.0,
            **meta,
        })

    # Sort by chunk_index for reading order
    hits.sort(key=lambda h: h.get("chunk_index", 0))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter(
    year_from: int | None,
    year_to:   int | None,
    source:    str | None,
) -> dict | None:
    """
    Build a ChromaDB 'where' clause dict.
    Returns None when no filters are needed.
    """
    conditions: list[dict] = []

    if year_from is not None:
        conditions.append({"year": {"$gte": str(year_from)}})
    if year_to is not None:
        conditions.append({"year": {"$lte": str(year_to)}})
    if source is not None:
        conditions.append({"source": {"$eq": source}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _distance_to_score(distance: float) -> float:
    """
    Convert an L2 distance to a 0–1 display score.

    This is a heuristic for UI display only — not a probability.
    distance = 0   → score = 1.0 (perfect match)
    distance = MAX → score ≈ 0.0
    """
    score = max(0.0, 1.0 - distance / MAX_DISTANCE)
    return round(score, 3)
