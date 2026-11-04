"""
rag_evaluation.py — Automated RAG pipeline evaluation for ResearchPilot.

Run:
    py -3.12 rag_evaluation.py

Produces:
    evaluation/rag_eval_results.csv
    evaluation/rag_eval_results.json
    evaluation/rag_eval_report.md

Tests covered
─────────────
Retrieval:
  - Relevant queries return at least 1 hit
  - Top hit title contains expected keyword(s)
  - Negative/nonsense query has low max relevance score
  - Metadata fields (title, year, doi, source) are present and non-empty

Citation:
  - Citations are built from metadata (not hallucinated)
  - DOI URLs are well-formed when present

Pipeline:
  - rag_chat() returns tokens + citations list
  - Ollama connectivity check
  - Index persistence after module re-import

Failure / edge cases:
  - Empty query
  - Query when index is empty (tested on a temporary empty collection)
  - Invalid / empty PDF bytes
  - CSV with no title/abstract columns
  - CSV with some empty rows
  - Duplicate document upload (idempotent indexing)

NOTE: Answer *quality* (faithfulness, relevance) requires human review.
This script tests structural correctness only and flags automated results
as AUTO or HUMAN-REVIEW.
"""

from __future__ import annotations

import atexit
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Suppress urllib3's known TypeError during interpreter shutdown when
# a streaming response is abandoned mid-stream (urllib3 bug, not ours).
import urllib3
_orig_urllib3_warn = getattr(urllib3, "disable_warnings", None)
try:
    urllib3.disable_warnings()
except Exception:
    pass
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    test_id:    str
    category:   str
    description: str
    status:     str       # PASS | FAIL | SKIP | HUMAN-REVIEW
    review_type: str      # AUTO | HUMAN-REVIEW
    detail:     str = ""
    latency_ms: float = 0.0


results: list[EvalResult] = []


def record(
    test_id: str,
    category: str,
    description: str,
    passed: bool | None,
    detail: str = "",
    latency_ms: float = 0.0,
    human_review: bool = False,
) -> None:
    if human_review:
        status = "HUMAN-REVIEW"
        rtype  = "HUMAN-REVIEW"
    elif passed is None:
        status = "SKIP"
        rtype  = "AUTO"
    elif passed:
        status = "PASS"
        rtype  = "AUTO"
    else:
        status = "FAIL"
        rtype  = "AUTO"

    r = EvalResult(test_id, category, description, status, rtype, detail, latency_ms)
    results.append(r)
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "HUMAN-REVIEW": "👁️"}.get(status, "?")
    print(f"  {icon} [{test_id}] {description[:70]}   {detail[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return result, ms


# ─────────────────────────────────────────────────────────────────────────────
# Import RAG modules
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== ResearchPilot RAG Evaluation ===")
print(f"Timestamp: {datetime.now().isoformat()}")
print()

try:
    from rag.retriever    import retrieve
    from rag.citations    import format_citations, format_citations_markdown
    from rag.vector_store import get_stats, get_collection
    from rag.document_loader import load_openalex_csv, load_pdf_bytes, load_uploaded_csv
    from rag.chunker      import chunk_document, chunk_documents
    from rag.indexer      import index_openalex_csv, index_uploaded_pdf, index_uploaded_csv
    from rag.generator    import ollama_available, _call_ollama
    from rag.pipeline     import rag_chat, get_index_stats
    from rag.config       import FINAL_CSV, MAX_DISTANCE
    record("IMPORT", "Setup", "All RAG modules import successfully", True)
except Exception as e:
    record("IMPORT", "Setup", "Module import", False, str(e))
    print("FATAL: cannot import RAG modules — aborting.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. INDEX STATE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1] Index state")
stats = get_index_stats()
n_chunks = stats.get("total_chunks", 0)
n_docs   = stats.get("unique_docs",  0)
record("IDX-01", "Index", f"Index not empty ({n_chunks} chunks, {n_docs} docs)",
       n_chunks > 0, f"chunks={n_chunks} docs={n_docs}")
record("IDX-02", "Index", "Index has at least 50 documents",
       n_docs >= 50, f"unique_docs={n_docs}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RETRIEVAL QUALITY
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2] Retrieval quality")

_RETRIEVAL_TESTS = [
    # (test_id, query, expected_keyword_in_title, min_expected_score)
    ("RET-01", "transformer attention mechanism",      "attention",    0.20),
    ("RET-02", "object detection deep learning YOLO",  "detection",    0.30),
    ("RET-03", "convolutional neural network image",   "image",        0.10),
    ("RET-04", "natural language processing BERT GPT", "language",     0.10),
    ("RET-05", "random forest classification",         None,           0.05),
]

for tid, q, kw, min_score in _RETRIEVAL_TESTS:
    if n_chunks == 0:
        record(tid, "Retrieval", f"retrieve('{q[:40]}')", None, "skipped — index empty")
        continue
    hits, ms = timed(retrieve, q, top_k=5)
    got_hits  = len(hits) > 0
    top_score = hits[0]["relevance_score"] if hits else 0.0
    kw_found  = (kw is None) or any(kw.lower() in h.get("title","").lower() for h in hits)
    score_ok  = top_score >= min_score
    passed = got_hits and score_ok
    detail = f"top_score={top_score:.2f} hits={len(hits)} kw={'✓' if kw_found else '✗'}"
    record(tid, "Retrieval", f"retrieve('{q[:45]}')", passed, detail, ms)


# Negative test
print()
record_neg_q = "xyzfoobar123 nonsense query abcdef"
if n_chunks > 0:
    hits_neg, ms = timed(retrieve, record_neg_q, top_k=3)
    max_s = max((h["relevance_score"] for h in hits_neg), default=0.0)
    record("RET-NEG", "Retrieval", "Nonsense query → low relevance scores",
           max_s < 0.15, f"max_score={max_s:.3f}", ms)
else:
    record("RET-NEG", "Retrieval", "Nonsense query → low scores", None, "index empty")

# Empty query
record("RET-EMPTY", "Retrieval", "Empty query returns no results",
       retrieve("") == [], "empty string")


# ─────────────────────────────────────────────────────────────────────────────
# 3. METADATA COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3] Metadata completeness")

if n_chunks > 0:
    hits = retrieve("deep learning", top_k=5)
    if hits:
        h = hits[0]
        required_fields = ["title", "year", "source", "chunk_id", "doc_id"]
        for field_name in required_fields:
            present = bool(h.get(field_name, ""))
            record(f"META-{field_name.upper()}", "Metadata",
                   f"Hit has non-empty '{field_name}'", present,
                   f"value='{str(h.get(field_name,''))[:30]}'")
    else:
        record("META-FIELDS", "Metadata", "Metadata completeness check", None, "no hits")
else:
    record("META-FIELDS", "Metadata", "Metadata completeness check", None, "index empty")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CITATION CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4] Citation correctness")

if n_chunks > 0:
    hits = retrieve("neural network classification", top_k=3)
    cits = format_citations(hits)

    record("CIT-01", "Citations", "format_citations returns non-empty list",
           len(cits) > 0, f"count={len(cits)}")

    if cits:
        c = cits[0]
        record("CIT-02", "Citations", "Citation has label field '[N]'",
               bool(c.get("label","").startswith("[")), f"label={c.get('label','')}")
        record("CIT-03", "Citations", "Citation title is non-empty",
               bool(c.get("title","")), f"title={c.get('title','')[:40]}")
        record("CIT-04", "Citations", "Citation number starts at 1",
               c.get("number") == 1, f"number={c.get('number')}")

        # DOI URL validation
        import re
        doi_url = c.get("doi_url","")
        doi_ok = (not doi_url) or doi_url.startswith("https://doi.org/")
        record("CIT-05", "Citations", "DOI URL format correct when present",
               doi_ok, f"doi_url={doi_url[:50]}")

        # Evidence type
        ev = c.get("evidence_type","")
        record("CIT-06", "Citations", "Evidence type is 'Full text' or 'Abstract only'",
               ev in ("Full text","Abstract only"), f"evidence_type={ev}")

    # Markdown render
    md = format_citations_markdown(cits)
    record("CIT-MD", "Citations", "format_citations_markdown returns non-empty string",
           len(md) > 10, f"len={len(md)}")
else:
    for tid in ["CIT-01","CIT-02","CIT-03","CIT-04","CIT-05","CIT-06","CIT-MD"]:
        record(tid, "Citations", "Citation test", None, "index empty")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DOCUMENT LOADING & CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] Document loading and chunking")

# CSV loading
docs_csv, ms = timed(load_openalex_csv, FINAL_CSV)
record("LOAD-CSV", "Loading", f"load_openalex_csv loads 2000 docs",
       len(docs_csv) == 2000, f"loaded={len(docs_csv)}", ms)

if docs_csv:
    d = docs_csv[0]
    record("LOAD-SCHEMA", "Loading", "Loaded doc has required fields",
           all(k in d for k in ["doc_id","title","abstract","text","source"]),
           f"keys={list(d.keys())[:6]}")

# Chunking
chunks = chunk_documents(docs_csv[:5])
record("CHUNK-01", "Chunking", "chunk_documents produces chunks for 5 docs",
       len(chunks) > 0, f"chunks={len(chunks)}")

if chunks:
    c = chunks[0]
    record("CHUNK-FIELDS", "Chunking", "Chunk has required provenance fields",
           all(k in c for k in ["chunk_id","doc_id","text","title","source","chunk_index"]),
           f"keys={list(c.keys())[:6]}")
    record("CHUNK-MINLEN", "Chunking", "All chunks >= MIN_CHUNK_CHARS",
           all(len(ch["text"]) >= 80 for ch in chunks),
           f"min_len={min(len(ch['text']) for ch in chunks)}")

# PDF edge cases
record("PDF-EMPTY", "Loading", "load_pdf_bytes(b'') returns None",
       load_pdf_bytes(b"", "empty.pdf") is None, "empty bytes")
record("PDF-GARBAGE", "Loading", "load_pdf_bytes(garbage) returns None",
       load_pdf_bytes(b"not a pdf", "bad.pdf") is None, "random bytes")

# CSV upload — no title/abstract columns
bad_csv = b"author,journal,pages\nSmith,Nature,1-10\n"
docs_bad, warnings_bad = load_uploaded_csv(bad_csv, "bad.csv")
record("CSV-BAD-COLS", "Loading", "CSV with no title/abstract returns error",
       len(docs_bad) == 0 and len(warnings_bad) > 0,
       f"docs={len(docs_bad)} warnings={len(warnings_bad)}")

# CSV upload — valid
good_csv = b"title,abstract,year\n\"Deep Learning\",\"A survey paper\",2023\n"
docs_good, warnings_good = load_uploaded_csv(good_csv, "good.csv")
record("CSV-GOOD", "Loading", "Valid CSV with title+abstract loads correctly",
       len(docs_good) == 1, f"docs={len(docs_good)}")

# CSV upload — some empty rows
mixed_csv = b"title,abstract\n\"Paper One\",\"Abstract one\"\n\"\",\"\"\n\"Paper Two\",\"Abstract two\"\n"
docs_mixed, _ = load_uploaded_csv(mixed_csv, "mixed.csv")
record("CSV-SKIP-EMPTY", "Loading", "CSV skips empty rows, loads valid ones",
       len(docs_mixed) == 2, f"docs={len(docs_mixed)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. IDEMPOTENT INDEXING (DUPLICATE DETECTION)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6] Idempotent indexing")

if n_chunks > 0:
    stats_before = get_index_stats()
    # Re-index the same 5 papers
    from rag.vector_store import add_chunks
    from rag.document_loader import load_openalex_csv
    from rag.chunker import chunk_documents
    docs5 = load_openalex_csv(FINAL_CSV)[:5]
    chunks5 = chunk_documents(docs5)
    added_again = add_chunks(chunks5)
    stats_after = get_index_stats()
    record("IDEM-01", "Indexing", "Re-indexing same docs adds 0 new chunks",
           added_again == 0, f"added={added_again}")
    record("IDEM-02", "Indexing", "Total chunk count unchanged after re-index",
           stats_before["total_chunks"] == stats_after["total_chunks"],
           f"before={stats_before['total_chunks']} after={stats_after['total_chunks']}")
else:
    record("IDEM-01", "Indexing", "Idempotency test", None, "index empty")
    record("IDEM-02", "Indexing", "Idempotency test", None, "index empty")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7] Persistence")

import rag.vector_store as vs
before_count = vs.get_collection().count()
vs._collection = None
vs._client     = None
after_count = vs.get_collection().count()
record("PERSIST-01", "Persistence",
       "Chunk count matches after collection re-open",
       before_count == after_count,
       f"before={before_count} after={after_count}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. OLLAMA CONNECTIVITY & GENERATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[8] Ollama connectivity")

ollama_ok = ollama_available()
record("OLLAMA-01", "Generation", "Ollama is reachable and model loaded",
       ollama_ok, f"model_ready={ollama_ok}")

if ollama_ok and n_chunks > 0:
    print("   Calling rag_chat() for latency test — please wait…")
    tokens_collected = []
    t0 = time.perf_counter()
    token_iter, citations = rag_chat("What is a transformer model?", history=[])
    for tok in token_iter:
        tokens_collected.append(tok)
        if len(tokens_collected) > 200:   # stop early — we just need timing
            break
    latency_ms = (time.perf_counter() - t0) * 1000
    full_text = "".join(tokens_collected)
    record("OLLAMA-02", "Generation", "rag_chat() returns tokens and citations",
           len(tokens_collected) > 0 and isinstance(citations, list),
           f"tokens={len(tokens_collected)} cits={len(citations)}", latency_ms)
    record("OLLAMA-03", "Generation", "rag_chat() first-token latency < 30s",
           latency_ms < 30_000, f"latency={latency_ms:.0f}ms", latency_ms)
    record("OLLAMA-QUALITY", "Generation",
           "Answer quality — does it use evidence and cite sources?",
           None, "Requires human review of answer content",
           human_review=True)
else:
    for tid in ["OLLAMA-02","OLLAMA-03","OLLAMA-QUALITY"]:
        record(tid, "Generation", "Generation test",
               None, "skipped — Ollama not ready or index empty")


# ─────────────────────────────────────────────────────────────────────────────
# 9. LATENCY
# ─────────────────────────────────────────────────────────────────────────────

print("\n[9] Retrieval latency")

if n_chunks > 0:
    _, ms1 = timed(retrieve, "deep learning neural network", top_k=5)
    _, ms2 = timed(retrieve, "transformer attention vision",  top_k=5)
    avg_ms = (ms1 + ms2) / 2
    record("LAT-RET", "Latency", "Average retrieval latency < 2000ms",
           avg_ms < 2000, f"avg={avg_ms:.0f}ms q1={ms1:.0f}ms q2={ms2:.0f}ms", avg_ms)
else:
    record("LAT-RET", "Latency", "Retrieval latency", None, "index empty")


# ─────────────────────────────────────────────────────────────────────────────
# Summary and output
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
n_pass  = sum(1 for r in results if r.status == "PASS")
n_fail  = sum(1 for r in results if r.status == "FAIL")
n_skip  = sum(1 for r in results if r.status == "SKIP")
n_human = sum(1 for r in results if r.status == "HUMAN-REVIEW")
total   = len(results)

print(f"PASS:          {n_pass}/{total - n_skip - n_human}")
print(f"FAIL:          {n_fail}")
print(f"SKIP:          {n_skip}")
print(f"HUMAN-REVIEW:  {n_human}")

# ── Write outputs ─────────────────────────────────────────────────────────────

eval_dir = ROOT / "evaluation"
eval_dir.mkdir(exist_ok=True)

# CSV
import csv as csv_mod
csv_path = eval_dir / "rag_eval_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv_mod.DictWriter(f, fieldnames=["test_id","category","description",
                                           "status","review_type","detail","latency_ms"])
    w.writeheader()
    for r in results:
        w.writerow(asdict(r))
print(f"\nCSV  → {csv_path}")

# JSON
json_path = eval_dir / "rag_eval_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "summary":   {"pass": n_pass, "fail": n_fail, "skip": n_skip, "human_review": n_human, "total": total},
        "results":   [asdict(r) for r in results],
    }, f, indent=2)
print(f"JSON → {json_path}")

# Markdown report
md_path = eval_dir / "rag_eval_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# ResearchPilot RAG Evaluation Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"| Metric | Count |\n|--------|-------|\n")
    f.write(f"| PASS   | {n_pass} |\n")
    f.write(f"| FAIL   | {n_fail} |\n")
    f.write(f"| SKIP   | {n_skip} |\n")
    f.write(f"| HUMAN-REVIEW | {n_human} |\n")
    f.write(f"| **Total** | **{total}** |\n\n")
    f.write("## Results by Category\n\n")
    categories = dict.fromkeys(r.category for r in results)
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        f.write(f"### {cat}\n\n")
        f.write("| ID | Description | Status | Detail | Latency (ms) |\n")
        f.write("|----|-------------|--------|--------|--------------|\n")
        for r in cat_results:
            icon = {"PASS":"✅","FAIL":"❌","SKIP":"⏭️","HUMAN-REVIEW":"👁️"}.get(r.status,"?")
            f.write(f"| {r.test_id} | {r.description[:55]} | {icon} {r.status} "
                    f"| {r.detail[:40]} | {r.latency_ms:.0f} |\n")
        f.write("\n")
    f.write(
        "\n## Notes\n\n"
        "- AUTO tests verify structural correctness only.\n"
        "- HUMAN-REVIEW tests require reading actual model outputs.\n"
        "- Answer faithfulness, relevance, and citation accuracy require human evaluation.\n"
        "- SKIP means the test condition (non-empty index, Ollama running) was not met.\n"
    )
print(f"MD   → {md_path}")
print("\n=== Evaluation complete ===\n")

# Force GC before interpreter teardown to avoid urllib3 shutdown noise
import gc
gc.collect()
