"""
rag/vector_store.py — Persistent ChromaDB vector store for ResearchPilot.

Responsibilities:
  - Open / create the persistent ChromaDB collection.
  - Add chunks with deduplication (idempotent upsert by chunk_id).
  - Query by semantic similarity.
  - Delete all chunks belonging to a document.
  - Report collection statistics.
  - Persist across application restarts (data/rag_vector_store/).

ChromaDB 1.5.9 notes:
  - PersistentClient auto-persists; no explicit .persist() call needed.
  - Must supply an EmbeddingFunction with name() method.
  - get_or_create_collection is idempotent and safe.
  - Metadata values must be str | int | float | bool (no None / lists).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

from rag.config import (
    VECTOR_STORE_DIR,
    CHROMA_COLLECTION,
    DEFAULT_TOP_K,
    MAX_DISTANCE,
)
from rag.embeddings import ResearchPilotEmbeddingFunction

log = logging.getLogger(__name__)

# Module-level client singleton
_client: chromadb.PersistentClient | None = None
_collection = None


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def get_collection():
    """
    Return (and lazily create) the persistent ChromaDB collection.

    Safe to call multiple times — reuses the existing client and collection.
    """
    global _client, _collection
    if _collection is not None:
        return _collection

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Opening ChromaDB at %s", VECTOR_STORE_DIR)

    _client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    ef = ResearchPilotEmbeddingFunction()

    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "l2"},   # L2 distance (lower = more similar)
    )
    count = _collection.count()
    log.info("Collection '%s' ready — %d chunks indexed.", CHROMA_COLLECTION, count)
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# Add / upsert chunks
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_metadata(meta: dict) -> dict:
    """
    ChromaDB requires metadata values to be str | int | float | bool.
    Convert everything else to str; replace None with empty string.
    """
    clean: dict = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def add_chunks(chunks: list[dict], batch_size: int = 200) -> int:
    """
    Upsert a list of chunk dicts into the vector store.

    Uses chunk_id as the stable document ID — re-adding the same chunk
    is idempotent.  Returns the number of new chunks actually added.
    """
    if not chunks:
        return 0

    col = get_collection()

    # Check which chunk_ids already exist to avoid redundant encoding
    ids_to_add = [c["chunk_id"] for c in chunks]
    existing = set()
    try:
        # ChromaDB .get() accepts up to ~5000 ids at once safely
        result = col.get(ids=ids_to_add, include=[])
        existing = set(result.get("ids", []))
    except Exception:
        existing = set()

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]
    if not new_chunks:
        log.debug("All %d chunks already indexed — nothing to add.", len(chunks))
        return 0

    added = 0
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        ids       = [c["chunk_id"] for c in batch]
        documents = [c["text"]     for c in batch]
        metadatas = [
            _sanitise_metadata({
                "doc_id":      c.get("doc_id",      ""),
                "title":       c.get("title",        "")[:500],   # cap length
                "authors":     c.get("authors",      "")[:300],
                "year":        c.get("year",          ""),
                "doi":         c.get("doi",           "")[:300],
                "bare_doi":    c.get("bare_doi",      "")[:200],
                "openalex_id": c.get("openalex_id",   "")[:200],
                "oa_category": c.get("oa_category",   ""),
                "oa_status":   c.get("oa_status",     ""),
                "cited_by":    c.get("cited_by",      ""),
                "source":      c.get("source",        ""),
                "filename":    c.get("filename",      ""),
                "chunk_index": int(c.get("chunk_index",  0)),
                "total_chunks": int(c.get("total_chunks", 1)),
                "is_fulltext": bool(c.get("is_fulltext",  False)),
                "page_hint":   int(c.get("page_hint",    0)),
            })
            for c in batch
        ]

        try:
            col.add(ids=ids, documents=documents, metadatas=metadatas)
            added += len(batch)
            log.debug("Added batch %d–%d (%d chunks)", i, i + len(batch), len(batch))
        except Exception as exc:
            log.error("Error adding batch starting at %d: %s", i, exc)

    log.info("add_chunks: added %d / %d new chunks.", added, len(new_chunks))
    return added


# ─────────────────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────────────────

def query_chunks(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = MAX_DISTANCE,
    filter_meta: dict | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for *query_text*.

    Returns a list of result dicts, each containing:
        text, distance, chunk_id, and all stored metadata fields.

    Results with L2 distance > max_distance are filtered out.
    Deduplicates by doc_id keeping only the closest chunk per paper.
    """
    col = get_collection()
    if col.count() == 0:
        return []

    where = filter_meta if filter_meta else None

    try:
        results = col.query(
            query_texts=[query_text],
            n_results=min(top_k * 3, col.count()),   # over-fetch then filter
            include=["documents", "metadatas", "distances"],
            where=where,
        )
    except Exception as exc:
        log.error("ChromaDB query failed: %s", exc)
        return []

    docs      = results.get("documents",  [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]
    ids       = results.get("ids",        [[]])[0]

    hits: list[dict] = []
    seen_docs: set[str] = set()   # dedup by doc_id

    for doc_text, meta, dist, cid in zip(docs, metas, distances, ids):
        if dist > max_distance:
            continue
        doc_id = meta.get("doc_id", cid)
        if doc_id in seen_docs:
            continue   # keep only the closest chunk per paper
        seen_docs.add(doc_id)

        hits.append({
            "chunk_id":    cid,
            "text":        doc_text,
            "distance":    round(dist, 4),
            **meta,
        })

        if len(hits) >= top_k:
            break

    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Deletion
# ─────────────────────────────────────────────────────────────────────────────

def delete_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to *doc_id*.
    Returns the number of chunks removed.
    """
    col = get_collection()
    try:
        existing = col.get(where={"doc_id": {"$eq": doc_id}}, include=[])
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            log.info("Deleted %d chunks for doc_id=%s", len(ids_to_delete), doc_id)
            return len(ids_to_delete)
    except Exception as exc:
        log.error("Error deleting doc_id=%s: %s", doc_id, exc)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """
    Return a dict with collection statistics.

    {
        "total_chunks": int,
        "unique_docs":  int,
        "sources":      dict[str, int],   # source → chunk count
        "store_path":   str,
    }
    """
    col = get_collection()
    total = col.count()

    stats: dict[str, Any] = {
        "total_chunks": total,
        "unique_docs":  0,
        "sources":      {},
        "store_path":   str(VECTOR_STORE_DIR),
    }

    if total == 0:
        return stats

    try:
        # Pull all metadata to compute unique docs and source breakdown
        # For large collections, sample; for ≤ 10k chunks, fetch all.
        fetch_limit = min(total, 10_000)
        result = col.get(limit=fetch_limit, include=["metadatas"])
        metas = result.get("metadatas", [])

        doc_ids: set[str] = set()
        sources: dict[str, int] = {}
        for m in metas:
            doc_ids.add(m.get("doc_id", ""))
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        stats["unique_docs"] = len(doc_ids)
        stats["sources"]     = sources
    except Exception as exc:
        log.warning("Could not compute detailed stats: %s", exc)

    return stats


def reset_collection() -> None:
    """
    DANGER: delete the entire collection and recreate it empty.
    Used for testing and re-indexing from scratch.
    """
    global _collection
    if _client is not None:
        try:
            _client.delete_collection(CHROMA_COLLECTION)
            log.warning("Collection '%s' deleted.", CHROMA_COLLECTION)
        except Exception:
            pass
    _collection = None
    get_collection()   # recreate
    log.info("Collection '%s' recreated (empty).", CHROMA_COLLECTION)
