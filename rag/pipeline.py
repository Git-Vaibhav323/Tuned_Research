"""
rag/pipeline.py — End-to-end RAG pipeline facade.

This module is the single import point for the Streamlit UI.
It wires together retriever → citations → generator and
exposes simple, high-level functions.

Usage (from Streamlit):
    from rag.pipeline import rag_chat, rag_summarise, rag_compare, rag_gap

All functions that generate text return Iterators[str] for streaming.
"""

from __future__ import annotations

from typing import Iterator

from rag.config import DEFAULT_TOP_K, MAX_DISTANCE
from rag.retriever import retrieve, retrieve_for_doc
from rag.citations import format_citations, format_citations_markdown
from rag.generator import (
    generate_rag_answer,
    generate_summary,
    generate_comparison,
    generate_gap_analysis,
    ollama_available,
)
from rag.vector_store import get_stats
from rag.indexer import (
    index_openalex_csv,
    index_uploaded_pdf,
    index_uploaded_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Research chat — main RAG Q&A
# ─────────────────────────────────────────────────────────────────────────────

def rag_chat(
    question: str,
    history: list[dict] | None = None,
    top_k: int = DEFAULT_TOP_K,
    stream: bool = True,
) -> tuple[Iterator[str], list[dict]]:
    """
    Answer *question* with evidence from the indexed literature.

    Returns:
        (token_iterator, citations_list)

    The caller should iterate the token_iterator to stream the answer,
    then display citations_list below the response.
    """
    hits      = retrieve(question, top_k=top_k)
    citations = format_citations(hits)
    tokens    = generate_rag_answer(question, hits, history=history, stream=stream)
    return tokens, citations


# ─────────────────────────────────────────────────────────────────────────────
# Paper summarisation
# ─────────────────────────────────────────────────────────────────────────────

def rag_summarise(
    doc_id: str,
    title: str,
    stream: bool = True,
) -> tuple[Iterator[str], list[dict]]:
    """
    Summarise a paper by its doc_id using its indexed chunks.

    Returns (token_iterator, citations_list).
    """
    hits      = retrieve_for_doc(doc_id, top_k=8)
    citations = format_citations(hits)
    tokens    = generate_summary(title, hits, stream=stream)
    return tokens, citations


# ─────────────────────────────────────────────────────────────────────────────
# Multi-paper comparison
# ─────────────────────────────────────────────────────────────────────────────

def rag_compare(
    doc_ids: list[str],
    titles: list[str],
    dimensions: list[str] | None = None,
    stream: bool = True,
) -> tuple[Iterator[str], list[dict]]:
    """
    Compare papers by doc_id.

    Returns (token_iterator, combined_citations_list).
    """
    hits_per_doc: dict[str, list[dict]] = {}
    all_hits: list[dict] = []
    for doc_id, title in zip(doc_ids, titles):
        doc_hits = retrieve_for_doc(doc_id, top_k=6)
        hits_per_doc[doc_id] = doc_hits
        all_hits.extend(doc_hits)

    citations = format_citations(all_hits)
    tokens    = generate_comparison(titles, hits_per_doc, dimensions=dimensions, stream=stream)
    return tokens, citations


# ─────────────────────────────────────────────────────────────────────────────
# Research gap analysis
# ─────────────────────────────────────────────────────────────────────────────

def rag_gap(
    query: str,
    top_k: int = 8,
    stream: bool = True,
) -> tuple[Iterator[str], list[dict]]:
    """
    Suggest research gaps based on papers related to *query*.

    Returns (token_iterator, citations_list).
    """
    hits      = retrieve(query, top_k=top_k)
    citations = format_citations(hits)
    tokens    = generate_gap_analysis(hits, stream=stream)
    return tokens, citations


# ─────────────────────────────────────────────────────────────────────────────
# Index management helpers (thin wrappers for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def get_index_stats() -> dict:
    """Return current vector store statistics."""
    return get_stats()


def index_corpus(
    limit: int | None = None,
    progress_cb=None,
) -> dict:
    """Index the main OpenAlex CSV corpus. Idempotent."""
    return index_openalex_csv(limit=limit, progress_cb=progress_cb)


def index_pdf_upload(pdf_bytes: bytes, filename: str, extra_meta: dict | None = None) -> dict:
    """Index an uploaded PDF. Returns status dict."""
    return index_uploaded_pdf(pdf_bytes, filename, extra_meta)


def index_csv_upload(csv_bytes: bytes, filename: str) -> dict:
    """Index an uploaded CSV. Returns status dict."""
    return index_uploaded_csv(csv_bytes, filename)


def is_ollama_ready() -> bool:
    """Check Ollama availability."""
    return ollama_available()


def format_citations_md(citations: list[dict]) -> str:
    """Render a citations list as Markdown."""
    return format_citations_markdown(citations)
