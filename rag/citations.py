"""
rag/citations.py — Format source citations from retrieved chunk metadata.

Design principles:
  - Every citation is built from stored metadata, never from LLM output.
  - Only fields that are genuinely known are included.
  - DOI links are only generated when the DOI looks like a real URL.
  - Page numbers are shown only for PDF sources with a real page_hint.
  - Abstract-only vs full-text evidence is clearly labelled.
"""

from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def format_citations(hits: list[dict]) -> list[dict]:
    """
    Build a list of citation dicts from retrieved chunks.

    Each citation dict:
        number       int    — 1-based display number
        label        str    — short inline label, e.g. "[1]"
        title        str
        authors      str    — or empty string
        year         str    — or empty string
        doi_url      str    — full https://doi.org/... URL or empty string
        openalex_url str    — OpenAlex paper URL or empty string
        page_note    str    — e.g. "p. 3" for PDFs, "" otherwise
        evidence_type str   — "Full text" | "Abstract only"
        source       str    — human-readable source label
        chunk_id     str    — for traceability
        doc_id       str
    """
    citations: list[dict] = []
    seen_docs: set[str] = set()

    n = 1
    for hit in hits:
        doc_id = hit.get("doc_id", hit.get("chunk_id", ""))
        # One citation per document even if multiple chunks from same paper were retrieved
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        title   = hit.get("title",   "Untitled") or "Untitled"
        authors = hit.get("authors", "") or ""
        year    = hit.get("year",    "") or ""
        doi     = hit.get("doi",     "") or ""
        openalex_id = hit.get("openalex_id", "") or ""
        is_fulltext = hit.get("is_fulltext", False)
        source      = hit.get("source",      "")
        page_hint   = int(hit.get("page_hint", 0) or 0)

        doi_url      = _build_doi_url(doi)
        openalex_url = _build_openalex_url(openalex_id)

        # Page note only for PDF sources with a real page estimate
        if source == "pdf_upload" and page_hint > 0:
            page_note = f"p. {page_hint}"
        else:
            page_note = ""

        evidence_type = "Full text" if is_fulltext else "Abstract only"
        source_label  = _source_label(source)

        citations.append({
            "number":        n,
            "label":         f"[{n}]",
            "title":         title,
            "authors":       authors,
            "year":          year,
            "doi_url":       doi_url,
            "openalex_url":  openalex_url,
            "page_note":     page_note,
            "evidence_type": evidence_type,
            "source":        source_label,
            "chunk_id":      hit.get("chunk_id", ""),
            "doc_id":        doc_id,
        })
        n += 1

    return citations


def format_citations_markdown(citations: list[dict]) -> str:
    """
    Render citations as a Markdown reference list.

    Example:
        **[1]** Attention Is All You Need · 2017
        DOI: https://doi.org/10.48550/arXiv.1706.03762 · Abstract only
    """
    if not citations:
        return "*No sources retrieved.*"

    lines: list[str] = ["**Sources:**", ""]
    for c in citations:
        line = f"**{c['label']}** {c['title']}"
        if c["authors"]:
            line += f" · {c['authors']}"
        if c["year"]:
            line += f" · {c['year']}"
        if c["page_note"]:
            line += f" · {c['page_note']}"
        lines.append(line)

        meta_parts: list[str] = []
        if c["doi_url"]:
            meta_parts.append(f"[DOI]({c['doi_url']})")
        elif c["openalex_url"]:
            meta_parts.append(f"[OpenAlex]({c['openalex_url']})")
        meta_parts.append(c["evidence_type"])
        meta_parts.append(c["source"])
        lines.append("  " + " · ".join(meta_parts))
        lines.append("")

    return "\n".join(lines)


def build_inline_refs(citations: list[dict]) -> str:
    """
    Build an inline reference string for the RAG prompt, e.g.:
        [1] "Attention Is All You Need" (2017) | [2] "BERT" (2019)
    """
    parts: list[str] = []
    for c in citations:
        part = f"{c['label']} \"{c['title']}\""
        if c["year"]:
            part += f" ({c['year']})"
        parts.append(part)
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_DOI_RE = re.compile(r"^10\.\d{4,}/\S+$")


def _build_doi_url(doi: str) -> str:
    """
    Return a valid https://doi.org/... URL, or empty string.

    Accepts:
      - Full URL already: "https://doi.org/10.1234/..."
      - Bare DOI:         "10.1234/..."
    """
    if not doi:
        return ""
    doi = doi.strip()
    if doi.startswith("https://doi.org/") or doi.startswith("http://doi.org/"):
        return doi.replace("http://doi.org/", "https://doi.org/")
    if _DOI_RE.match(doi):
        return f"https://doi.org/{doi}"
    return ""


def _build_openalex_url(openalex_id: str) -> str:
    """Return the OpenAlex web URL, or empty string."""
    if not openalex_id:
        return ""
    if openalex_id.startswith("https://openalex.org/"):
        return openalex_id
    if openalex_id.startswith("W"):
        return f"https://openalex.org/{openalex_id}"
    return ""


def _source_label(source: str) -> str:
    labels = {
        "openalex_csv": "OpenAlex corpus",
        "pdf_upload":   "Uploaded PDF",
        "csv_upload":   "Uploaded CSV",
        "sqlite":       "ResearchPilot DB",
    }
    return labels.get(source, source or "Unknown source")
