"""
rag/document_loader.py — Load research papers from multiple sources.

Supports:
  - OpenAlex CSV  (data/final/final_dataset.csv)
  - Uploaded PDFs (via PyMuPDF / fitz)
  - Uploaded CSVs (research metadata with column validation)
  - SQLite        (researchpilot_r.db — fallback / supplementary)

Each returned document is a dict with a standard schema so downstream
chunking and indexing code never needs to branch on source type.

Standard document schema
------------------------
{
    "doc_id":    str,   # stable unique id (openalex ID, DOI hash, or filename-based)
    "title":     str,
    "abstract":  str,
    "text":      str,   # full text if available, otherwise == abstract
    "authors":   str,   # comma-separated or empty string
    "year":      str,
    "doi":       str,   # canonical DOI URL or empty string
    "oa_category": str, # open-access category or empty string
    "source":    str,   # "openalex_csv" | "pdf_upload" | "csv_upload" | "sqlite"
    "filename":  str,   # original file name or empty string
    "page_count": int,  # number of PDF pages, 0 for non-PDF sources
    "is_fulltext": bool,# True only when full text beyond abstract is available
}
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_doc_id(value: str) -> str:
    """Return a short stable hex id from any string."""
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return str(text).strip()


def _empty_doc() -> dict:
    return {
        "doc_id":      "",
        "title":       "",
        "abstract":    "",
        "text":        "",
        "authors":     "",
        "year":        "",
        "doi":         "",
        "oa_category": "",
        "source":      "",
        "filename":    "",
        "page_count":  0,
        "is_fulltext": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpenAlex CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_openalex_csv(csv_path: str | Path) -> list[dict]:
    """
    Load all papers from the OpenAlex final_dataset.csv.

    Returns a list of standard document dicts.
    Papers with empty titles AND empty abstracts are skipped.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return []

    docs: list[dict] = []
    skipped = 0

    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title    = _clean(row.get("title", ""))
            abstract = _clean(row.get("abstract", ""))

            if not title and not abstract:
                skipped += 1
                continue

            raw_id = _clean(row.get("id", "")) or _clean(row.get("doi", "")) or title
            doc_id = _make_doc_id(raw_id)

            doi = _clean(row.get("doi", ""))
            # Normalise DOI — strip leading "https://doi.org/" to get bare DOI
            bare_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

            doc = _empty_doc()
            doc.update({
                "doc_id":      doc_id,
                "title":       title,
                "abstract":    abstract,
                "text":        abstract,          # abstract is the only text for this corpus
                "authors":     "",                # not present in this dataset
                "year":        _clean(row.get("publication_year", "")),
                "doi":         doi,
                "bare_doi":    bare_doi,
                "openalex_id": _clean(row.get("id", "")),
                "oa_category": _clean(row.get("oa_category", "")),
                "oa_status":   _clean(row.get("oa_status", "")),
                "cited_by":    _clean(row.get("cited_by_count", "")),
                "source":      "openalex_csv",
                "filename":    csv_path.name,
                "page_count":  0,
                "is_fulltext": False,
            })
            docs.append(doc)

    log.info("Loaded %d docs from %s (%d skipped)", len(docs), csv_path.name, skipped)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# PDF loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> dict | None:
    """
    Extract text from a PDF supplied as raw bytes (e.g. from st.file_uploader).

    Returns a standard document dict, or None if extraction yields no text.
    The caller is responsible for adding title/abstract/metadata after loading
    if they are known separately.
    """
    try:
        import pymupdf  # PyMuPDF (replaces fitz)
    except ImportError:
        log.error("PyMuPDF not installed — cannot extract PDF text.")
        return None

    TESSDATA = r"C:\Program Files\Tesseract-OCR\tessdata"

    try:
        doc_fitz = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        log.warning("Could not open PDF '%s': %s", filename, exc)
        return None

    pages_text: list[str] = []
    for page_num, page in enumerate(doc_fitz, start=1):
        page_text = page.get_text("text")
        if page_text and page_text.strip():
            pages_text.append(page_text.strip())

    full_text = "\n\n".join(pages_text).strip()

    # OCR fallback for scanned PDFs
    if not full_text:
        log.info("PDF '%s' has no text layer — attempting OCR.", filename)
        ocr_pages: list[str] = []
        import os
        for page in doc_fitz:
            try:
                tp = page.get_textpage_ocr(
                    flags=pymupdf.TEXT_PRESERVE_WHITESPACE,
                    dpi=200,
                    full=True,
                    tessdata=TESSDATA,
                )
                ocr_pages.append(tp.extractText())
            except Exception as exc:
                log.warning("OCR failed on page: %s", exc)
                ocr_pages.append("")
        full_text = "\n\n".join(p for p in ocr_pages if p.strip())

    doc_fitz.close()

    if not full_text:
        log.warning("PDF '%s' produced no extractable text (tried OCR).", filename)
        return None

    doc_id = _make_doc_id(filename + str(len(full_text)))

    doc = _empty_doc()
    doc.update({
        "doc_id":      doc_id,
        "title":       Path(filename).stem.replace("_", " ").replace("-", " "),
        "abstract":    full_text[:500],   # first 500 chars as pseudo-abstract
        "text":        full_text,
        "source":      "pdf_upload",
        "filename":    filename,
        "page_count":  len(pages_text),
        "is_fulltext": True,
    })
    return doc


def load_pdf_file(pdf_path: str | Path) -> dict | None:
    """Load a PDF from disk. Convenience wrapper around load_pdf_bytes."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        log.error("PDF file not found: %s", pdf_path)
        return None
    return load_pdf_bytes(pdf_path.read_bytes(), filename=pdf_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CSV upload loader
# ─────────────────────────────────────────────────────────────────────────────

# Accepted column name aliases (case-insensitive)
_COL_ALIASES: dict[str, list[str]] = {
    "title":    ["title", "paper_title", "name"],
    "abstract": ["abstract", "summary", "description", "text", "content"],
    "year":     ["year", "publication_year", "pub_year", "date"],
    "doi":      ["doi", "url", "link"],
    "authors":  ["authors", "author", "author_names"],
}


def _resolve_col(header: list[str], field: str) -> str | None:
    """Return the first header column that matches a known alias for *field*."""
    lower_header = {h.lower(): h for h in header}
    for alias in _COL_ALIASES.get(field, []):
        if alias in lower_header:
            return lower_header[alias]
    return None


def load_uploaded_csv(
    csv_bytes: bytes,
    filename: str = "upload.csv",
) -> tuple[list[dict], list[str]]:
    """
    Parse an uploaded CSV containing paper metadata.

    Returns (documents, warnings).
    Requires at least a 'title' or 'abstract' column (case-insensitive aliases).
    """
    warnings: list[str] = []
    docs: list[dict] = []

    try:
        text = csv_bytes.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames or []
    except Exception as exc:
        return [], [f"Could not parse CSV: {exc}"]

    # Resolve columns
    col_map = {field: _resolve_col(list(header), field) for field in _COL_ALIASES}

    if not col_map["title"] and not col_map["abstract"]:
        return [], [
            f"CSV '{filename}' has no recognised title or abstract column. "
            f"Found columns: {', '.join(header[:10])}"
        ]

    skipped = 0
    for i, row in enumerate(reader, start=2):
        title    = _clean(row.get(col_map["title"]    or "", ""))
        abstract = _clean(row.get(col_map["abstract"] or "", ""))
        year     = _clean(row.get(col_map["year"]     or "", ""))
        doi      = _clean(row.get(col_map["doi"]      or "", ""))
        authors  = _clean(row.get(col_map["authors"]  or "", ""))

        if not title and not abstract:
            skipped += 1
            continue

        raw_id = doi or title or f"{filename}_{i}"
        doc_id = _make_doc_id(raw_id)

        doc = _empty_doc()
        doc.update({
            "doc_id":      doc_id,
            "title":       title,
            "abstract":    abstract,
            "text":        abstract or title,
            "authors":     authors,
            "year":        year,
            "doi":         doi,
            "source":      "csv_upload",
            "filename":    filename,
            "page_count":  0,
            "is_fulltext": False,
        })
        docs.append(doc)

    if skipped:
        warnings.append(f"Skipped {skipped} rows with no title or abstract.")

    log.info("Loaded %d docs from uploaded CSV '%s'", len(docs), filename)
    return docs, warnings


# ─────────────────────────────────────────────────────────────────────────────
# SQLite loader (supplementary)
# ─────────────────────────────────────────────────────────────────────────────

def load_sqlite_papers(db_path: str | Path) -> list[dict]:
    """
    Load papers from the ResearchPilot SQLite database (papers table).

    Returns an empty list gracefully if the DB or table does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.warning("SQLite DB not found: %s", db_path)
        return []

    docs: list[dict] = []
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Discover available columns first
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if "papers" not in tables:
            log.warning("No 'papers' table in %s", db_path)
            con.close()
            return []

        cur.execute("PRAGMA table_info(papers)")
        cols = {r[1] for r in cur.fetchall()}

        select_cols = ["id"]
        for c in ["title", "abstract", "publication_year", "doi", "oa_category"]:
            if c in cols:
                select_cols.append(c)

        query = f"SELECT {', '.join(select_cols)} FROM papers"  # noqa: S608
        cur.execute(query)
        rows = cur.fetchall()
        con.close()

        for row in rows:
            r = dict(row)
            title    = _clean(r.get("title", ""))
            abstract = _clean(r.get("abstract", ""))
            if not title and not abstract:
                continue
            raw_id = _clean(r.get("doi", "")) or _clean(r.get("id", "")) or title
            doc_id = _make_doc_id(raw_id)
            doc = _empty_doc()
            doc.update({
                "doc_id":      doc_id,
                "title":       title,
                "abstract":    abstract,
                "text":        abstract,
                "year":        str(r.get("publication_year", "")),
                "doi":         _clean(r.get("doi", "")),
                "oa_category": _clean(r.get("oa_category", "")),
                "source":      "sqlite",
                "filename":    db_path.name,
            })
            docs.append(doc)

    except Exception as exc:
        log.error("Error reading SQLite DB %s: %s", db_path, exc)
        return []

    log.info("Loaded %d docs from SQLite %s", len(docs), db_path.name)
    return docs
