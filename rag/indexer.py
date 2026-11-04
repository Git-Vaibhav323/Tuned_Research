"""
rag/indexer.py — Orchestrates bulk and incremental indexing.

Usage:
    from rag.indexer import index_openalex_csv, index_document, index_uploaded_file

All public functions are safe to call multiple times — already-indexed
chunks are detected and skipped (idempotent via chunk_id dedup in vector_store).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from rag.config import FINAL_CSV, RAW_PAPERS_DIR
from rag.document_loader import (
    load_openalex_csv,
    load_pdf_bytes,
    load_pdf_file,
    load_uploaded_csv,
    load_sqlite_papers,
)
from rag.chunker import chunk_document, chunk_documents
from rag.vector_store import add_chunks, get_stats

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bulk indexing — OpenAlex CSV (main corpus)
# ─────────────────────────────────────────────────────────────────────────────

def index_openalex_csv(
    csv_path: str | Path = FINAL_CSV,
    limit: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Load and index papers from the OpenAlex CSV.

    Args:
        csv_path:    Path to final_dataset.csv (defaults to config value).
        limit:       Index only the first N papers (useful for testing).
        progress_cb: Optional callback(current, total) called every 100 papers.

    Returns a summary dict with counts.
    """
    docs = load_openalex_csv(csv_path)
    if limit:
        docs = docs[:limit]

    total = len(docs)
    log.info("Indexing %d papers from OpenAlex CSV...", total)

    all_chunks: list[dict] = []
    for i, doc in enumerate(docs):
        all_chunks.extend(chunk_document(doc))
        if progress_cb and (i + 1) % 100 == 0:
            progress_cb(i + 1, total)

    added = add_chunks(all_chunks)

    summary = {
        "docs_loaded":   total,
        "chunks_created": len(all_chunks),
        "chunks_added":   added,
        "chunks_skipped": len(all_chunks) - added,
    }
    log.info("index_openalex_csv done: %s", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Incremental — single document
# ─────────────────────────────────────────────────────────────────────────────

def index_document(doc: dict) -> dict:
    """
    Index a single pre-loaded document dict.

    Returns {"chunks_created": N, "chunks_added": M}.
    """
    chunks = chunk_document(doc)
    added  = add_chunks(chunks)
    return {"chunks_created": len(chunks), "chunks_added": added}


# ─────────────────────────────────────────────────────────────────────────────
# Uploaded file indexing (called from Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def index_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
    extra_meta: dict | None = None,
) -> dict:
    """
    Index an uploaded PDF.

    Args:
        pdf_bytes:  Raw bytes from st.file_uploader.
        filename:   Original file name for provenance.
        extra_meta: Optional dict with keys title/authors/year/doi to
                    override auto-extracted values.

    Returns {"status": "ok"|"error", "message": str, "chunks_added": int}.
    """
    doc = load_pdf_bytes(pdf_bytes, filename)
    if doc is None:
        return {
            "status":  "error",
            "message": (
                f"Could not extract text from '{filename}'. "
                "The file may be a scanned image (OCR required) or corrupt."
            ),
            "chunks_added": 0,
        }

    if extra_meta:
        for key in ("title", "authors", "year", "doi"):
            if extra_meta.get(key):
                doc[key] = extra_meta[key]

    result = index_document(doc)
    return {
        "status":       "ok",
        "message":      f"Indexed {result['chunks_added']} new chunks from '{filename}'.",
        "chunks_added": result["chunks_added"],
        "doc_id":       doc["doc_id"],
    }


def index_uploaded_csv(
    csv_bytes: bytes,
    filename: str,
) -> dict:
    """
    Index papers from an uploaded CSV file.

    Returns {"status": "ok"|"error", "message": str,
             "docs_loaded": int, "chunks_added": int, "warnings": list[str]}.
    """
    docs, warnings = load_uploaded_csv(csv_bytes, filename)

    if not docs:
        msg = warnings[0] if warnings else f"No valid papers found in '{filename}'."
        return {
            "status":     "error",
            "message":    msg,
            "docs_loaded": 0,
            "chunks_added": 0,
            "warnings":   warnings,
        }

    all_chunks = chunk_documents(docs)
    added = add_chunks(all_chunks)

    return {
        "status":       "ok",
        "message":      f"Indexed {added} new chunks from {len(docs)} papers in '{filename}'.",
        "docs_loaded":  len(docs),
        "chunks_added": added,
        "warnings":     warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SQLite supplementary indexing
# ─────────────────────────────────────────────────────────────────────────────

def index_sqlite(db_path: str | Path) -> dict:
    """Index papers from the ResearchPilot SQLite database."""
    docs = load_sqlite_papers(db_path)
    if not docs:
        return {"docs_loaded": 0, "chunks_added": 0}
    all_chunks = chunk_documents(docs)
    added = add_chunks(all_chunks)
    return {"docs_loaded": len(docs), "chunks_added": added}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: index all available sources
# ─────────────────────────────────────────────────────────────────────────────

def index_all_sources(
    limit: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Index the OpenAlex CSV and any PDFs in raw_papers/.

    Returns a combined summary dict.
    """
    result = index_openalex_csv(limit=limit, progress_cb=progress_cb)

    # Index any PDFs that have been placed in raw_papers/
    pdf_added = 0
    for pdf_path in RAW_PAPERS_DIR.glob("*.pdf"):
        doc = load_pdf_file(pdf_path)
        if doc:
            r = index_document(doc)
            pdf_added += r["chunks_added"]

    result["pdf_chunks_added"] = pdf_added
    result["final_stats"] = get_stats()
    return result
