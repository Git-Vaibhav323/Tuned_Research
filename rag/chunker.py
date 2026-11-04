"""
rag/chunker.py — Split document text into overlapping passages.

Design decisions:
- Character-based splitting (not token-based) for simplicity and speed.
  all-MiniLM-L6-v2 has a 256-token limit; 700 chars ≈ 140–180 tokens, safe.
- Overlap preserves sentence context across chunk boundaries.
- Every chunk carries full provenance: doc_id, title, source, page estimate.
- Chunks shorter than MIN_CHUNK_CHARS are silently discarded.
- The chunk_id is a deterministic hash so re-indexing is idempotent.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    doc: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    """
    Split a standard document dict into a list of chunk dicts.

    Each chunk dict has:
        chunk_id   str   — deterministic SHA-1 id
        doc_id     str   — parent document id
        text       str   — the passage text
        title      str   — parent paper title
        authors    str
        year       str
        doi        str
        oa_category str
        source     str   — "openalex_csv" | "pdf_upload" | "csv_upload" | "sqlite"
        filename   str
        chunk_index int  — 0-based position within the document
        is_fulltext bool — inherited from parent document
        page_hint   int  — rough page estimate for PDF sources (0 otherwise)
    """
    text = doc.get("text") or doc.get("abstract") or ""
    text = text.strip()

    if len(text) < min_chars:
        log.debug("Document '%s' text too short (%d chars) — skipped.", doc.get("doc_id"), len(text))
        return []

    raw_chunks = _split_text(text, chunk_size, chunk_overlap)
    chunks: list[dict] = []

    for idx, passage in enumerate(raw_chunks):
        passage = passage.strip()
        if len(passage) < min_chars:
            continue

        chunk_id = _make_chunk_id(doc["doc_id"], idx, passage)
        page_hint = _estimate_page(idx, len(raw_chunks), doc.get("page_count", 0))

        chunks.append({
            "chunk_id":    chunk_id,
            "doc_id":      doc["doc_id"],
            "text":        passage,
            "title":       doc.get("title", ""),
            "authors":     doc.get("authors", ""),
            "year":        doc.get("year", ""),
            "doi":         doc.get("doi", ""),
            "bare_doi":    doc.get("bare_doi", ""),
            "openalex_id": doc.get("openalex_id", ""),
            "oa_category": doc.get("oa_category", ""),
            "oa_status":   doc.get("oa_status", ""),
            "cited_by":    doc.get("cited_by", ""),
            "source":      doc.get("source", ""),
            "filename":    doc.get("filename", ""),
            "chunk_index": idx,
            "total_chunks": len(raw_chunks),
            "is_fulltext": doc.get("is_fulltext", False),
            "page_hint":   page_hint,
        })

    return chunks


def chunk_documents(
    docs: Sequence[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    """Chunk a list of documents. Returns all chunks in a flat list."""
    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap, min_chars))
    log.info("Chunked %d docs → %d chunks", len(docs), len(all_chunks))
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping windows of *size* characters,
    trying to break at sentence boundaries ('. ') where possible.
    """
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + size

        if end >= len(text):
            # Last chunk — take the remainder
            chunks.append(text[start:])
            break

        # Try to find the last sentence boundary within the window
        window = text[start:end]
        best_break = _last_sentence_break(window)

        if best_break > size // 2:
            # Good sentence boundary found in the second half
            end = start + best_break
        # else: hard cut at `end`

        chunks.append(text[start:end].strip())
        # Advance with overlap — but never go backwards
        advance = max(end - start - overlap, 1)
        start += advance

    return [c for c in chunks if c]


def _last_sentence_break(text: str) -> int:
    """
    Return the character index of the last sentence-ending boundary
    in *text*, or 0 if none found.

    Looks for '. ', '? ', '! ', '\n\n'.
    """
    best = 0
    for marker in (". ", "? ", "! ", "\n\n"):
        pos = text.rfind(marker)
        if pos > best:
            best = pos + len(marker)
    return best


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID from doc_id + index + first 64 chars of text."""
    seed = f"{doc_id}::{chunk_index}::{text[:64]}"
    return hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest()[:20]


def _estimate_page(chunk_index: int, total_chunks: int, page_count: int) -> int:
    """
    Rough page estimate for PDF sources.
    Returns 0 for non-PDF docs (page_count == 0).
    """
    if page_count <= 0 or total_chunks <= 0:
        return 0
    # Linear interpolation: chunk position → page number
    ratio = chunk_index / max(total_chunks - 1, 1)
    return max(1, round(ratio * page_count))
