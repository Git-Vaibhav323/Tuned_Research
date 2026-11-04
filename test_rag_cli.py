"""
test_rag_cli.py — Interactive CLI test for ResearchPilot RAG pipeline + Ollama model.

Tests everything without needing Streamlit:
  1. System check   — Python, packages, Ollama, index
  2. Retrieval      — semantic search, scores, metadata
  3. Citations      — formatting, DOI validation
  4. Generation     — streaming tokens from Ollama
  5. RAG pipeline   — end-to-end rag_chat()
  6. Summarise      — rag_summarise() on a real paper
  7. Compare        — rag_compare() on 2 papers
  8. Gap analysis   — rag_gap()
  9. Ollama direct  — raw model call (no RAG context)
  10. Interactive    — REPL: type questions, get streamed answers

Run:
    py -3.12 test_rag_cli.py              # full test suite then interactive REPL
    py -3.12 test_rag_cli.py --quick      # system checks + one end-to-end call only
    py -3.12 test_rag_cli.py --repl       # skip tests, go straight to interactive REPL
    py -3.12 test_rag_cli.py --section 5  # run only section 5 (RAG pipeline)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def hdr(title, n):
    bar = "─" * (56 - len(title))
    print(f"\n{BOLD}[{n}] {title} {DIM}{bar}{RESET}")


_pass = _fail = 0

def check(label: str, condition: bool, detail: str = "") -> bool:
    global _pass, _fail
    if condition:
        _pass += 1
        ok(f"{label}  {DIM}{detail}{RESET}" if detail else label)
    else:
        _fail += 1
        fail(f"{label}  {DIM}{detail}{RESET}" if detail else label)
    return condition


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — System check
# ═════════════════════════════════════════════════════════════════════════════

def section_1_system():
    hdr("System Check", 1)

    # Python version
    v = sys.version_info
    check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 10),
          "need 3.10+" if v < (3, 10) else "")

    # Key packages
    packages = [
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb",              "chromadb"),
        ("requests",              "requests"),
        ("pandas",                "pandas"),
    ]
    for mod, pkg in packages:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            check(f"Package: {pkg}", True, f"v{ver}")
        except ImportError:
            check(f"Package: {pkg}", False, "not installed — pip install " + pkg)

    # Optional: fitz / pymupdf for PDF support
    try:
        import fitz
        check("Package: pymupdf (PDF support)", True, f"v{fitz.__version__}")
    except ImportError:
        warn("pymupdf not installed — PDF upload will be disabled")

    # RAG module imports
    try:
        from rag.config       import FINAL_CSV, OLLAMA_MODEL, EMBED_MODEL_NAME
        from rag.retriever    import retrieve
        from rag.citations    import format_citations
        from rag.vector_store import get_stats
        from rag.pipeline     import rag_chat, rag_summarise, rag_compare, rag_gap, get_index_stats
        from rag.generator    import ollama_available
        check("RAG modules import", True)
        info(f"Model: {OLLAMA_MODEL}  |  Embeddings: {EMBED_MODEL_NAME}")
        info(f"Dataset: {FINAL_CSV}")
        return True
    except Exception as e:
        check("RAG modules import", False, str(e))
        return False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Index + embedding model
# ═════════════════════════════════════════════════════════════════════════════

def section_2_index():
    hdr("Index & Embedding Model", 2)
    from rag.pipeline import get_index_stats
    from rag.embeddings import get_model

    # Index stats
    stats = get_index_stats()
    n_chunks = stats.get("total_chunks", 0)
    n_docs   = stats.get("unique_docs",  0)
    check(f"Index non-empty", n_chunks > 0,   f"{n_chunks} chunks, {n_docs} papers")
    check(f"At least 1,000 papers indexed", n_docs >= 1000, f"unique_docs={n_docs}")

    # Embedding model loads
    t0 = time.perf_counter()
    try:
        model = get_model()
        ms = (time.perf_counter() - t0) * 1000
        check("Embedding model loaded", True, f"{ms:.0f} ms")

        # Smoke-test encode
        vecs = model.encode(["test sentence"], convert_to_numpy=True)
        check("Embedding model encodes", vecs.shape == (1, 384), f"shape={vecs.shape}")
    except Exception as e:
        check("Embedding model", False, str(e))

    return n_chunks > 0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Retrieval
# ═════════════════════════════════════════════════════════════════════════════

def section_3_retrieval(index_ready: bool):
    hdr("Semantic Retrieval", 3)
    if not index_ready:
        warn("Skipped — index is empty")
        return []

    from rag.retriever import retrieve

    queries = [
        ("transformer attention mechanism",       0.20),
        ("object detection YOLO deep learning",   0.30),
        ("federated learning privacy",            0.05),
        ("random forest feature importance",      0.10),
        ("BERT fine-tuning NLP classification",   0.15),
    ]

    all_hits = []
    for q, min_score in queries:
        t0 = time.perf_counter()
        hits = retrieve(q, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        top = hits[0]["relevance_score"] if hits else 0.0
        passed = len(hits) > 0 and top >= min_score
        check(f'retrieve("{q[:45]}")',
              passed,
              f"hits={len(hits)} top_score={top:.2f} ({ms:.0f}ms)")
        if hits:
            all_hits.extend(hits[:2])
            info(f"  Top result: \"{hits[0].get('title','')[:65]}\"")

    # Negative query — should return noticeably lower scores than real queries
    bad_hits = retrieve("zxqfoobar999 nonsense abcdef", top_k=3)
    max_bad  = max((h["relevance_score"] for h in bad_hits), default=0.0)
    # Real queries score 0.5+; nonsense should be well below that
    check("Nonsense query → low scores (< 0.30)", max_bad < 0.30, f"max_score={max_bad:.3f}")

    # Empty query — should return []
    check("Empty query → []", retrieve("") == [], "empty string guard")

    # Latency: two queries averaged
    _, ms1 = _timed(retrieve, "deep learning classification", top_k=5)
    _, ms2 = _timed(retrieve, "transformer vision attention",  top_k=5)
    avg = (ms1 + ms2) / 2
    check("Retrieval latency < 500 ms avg", avg < 500, f"avg={avg:.0f}ms")

    return all_hits


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Citations
# ═════════════════════════════════════════════════════════════════════════════

def section_4_citations(hits: list[dict]):
    hdr("Citation Formatting", 4)
    if not hits:
        warn("Skipped — no hits from retrieval")
        return

    from rag.citations import format_citations, format_citations_markdown

    cits = format_citations(hits[:5])
    check("format_citations returns list",  isinstance(cits, list) and len(cits) > 0,
          f"count={len(cits)}")

    if cits:
        c0 = cits[0]
        check("Citation has [N] label",    c0.get("label","").startswith("["),
              f"label={c0.get('label')}")
        check("Citation has title",        bool(c0.get("title","")),
              f"\"{c0.get('title','')[:50]}\"")
        check("Citation number = 1",       c0.get("number") == 1,
              f"number={c0.get('number')}")
        doi = c0.get("doi_url","")
        check("DOI URL well-formed",
              not doi or doi.startswith("https://doi.org/"),
              f"doi_url={doi[:50] or 'none'}")
        ev = c0.get("evidence_type","")
        check("Evidence type valid",
              ev in ("Full text","Abstract only"),
              f"evidence_type={ev}")

        # Markdown render
        md = format_citations_markdown(cits)
        check("Markdown render non-empty", len(md) > 20, f"len={len(md)}")
        info("Sample citation:")
        info(f"  {cits[0]['label']} {cits[0]['title'][:60]} ({cits[0]['year']})")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Ollama connectivity + raw generation
# ═════════════════════════════════════════════════════════════════════════════

def section_5_ollama():
    hdr("Ollama Model — Raw Generation", 5)
    from rag.generator import ollama_available, _call_ollama
    from rag.config    import OLLAMA_MODEL

    is_up = ollama_available()
    check(f"Ollama reachable & model '{OLLAMA_MODEL}' loaded", is_up)
    if not is_up:
        warn("Start Ollama:  ollama serve   (then ensure model exists: ollama list)")
        return False

    # Direct call — no RAG context, just a simple question
    print(f"\n  {DIM}Calling model directly (no RAG context)…{RESET}")
    messages = [
        {"role": "system",  "content": "You are a helpful assistant. Be very brief."},
        {"role": "user",    "content": "In one sentence, what is a transformer model?"},
    ]
    tokens = []
    t0 = time.perf_counter()
    print(f"  {CYAN}Model:{RESET} ", end="", flush=True)
    for tok in _call_ollama(messages, stream=True):
        print(tok, end="", flush=True)
        tokens.append(tok)
        if len(tokens) > 300:   # cap at 300 tokens for the test
            break
    ms = (time.perf_counter() - t0) * 1000
    print()

    check("Model returned tokens",          len(tokens) > 0,   f"tokens={len(tokens)}")
    check("First-token latency < 30 s",     ms < 30_000,       f"{ms:.0f} ms")
    check("No error string in response",
          not any("Error:" in t for t in tokens),
          "clean response")

    return True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — End-to-end RAG chat
# ═════════════════════════════════════════════════════════════════════════════

def section_6_rag_chat(index_ready: bool, ollama_ready: bool):
    hdr("End-to-End RAG Chat", 6)
    if not index_ready or not ollama_ready:
        warn("Skipped — need both index and Ollama")
        return

    from rag.pipeline import rag_chat

    question = "What transformer architectures are used for object detection?"
    print(f"\n  {CYAN}Question:{RESET} {question}")
    print(f"  {DIM}Retrieving evidence and generating answer…{RESET}\n")

    tokens = []
    citations = []
    t0 = time.perf_counter()

    print(f"  {CYAN}Answer:{RESET} ", end="", flush=True)
    token_iter, citations = rag_chat(question, history=[])
    for tok in token_iter:
        print(tok, end="", flush=True)
        tokens.append(tok)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n")

    full_answer = "".join(tokens)
    check("rag_chat returns tokens",     len(tokens) > 0,       f"tokens={len(tokens)}")
    check("rag_chat returns citations",  isinstance(citations, list), f"citations={len(citations)}")
    check("Answer not empty",            len(full_answer.strip()) > 20)
    check("No error in answer",          "Error:" not in full_answer)
    check("Citations present",           len(citations) > 0,    f"count={len(citations)}")
    check("Total latency < 60 s",        ms < 60_000,           f"{ms:.0f} ms")

    if citations:
        print(f"\n  {CYAN}Citations used:{RESET}")
        for c in citations[:3]:
            print(f"    {c['label']} {c['title'][:65]} ({c['year']})")

    # Multi-turn: follow-up question
    print(f"\n  {DIM}Follow-up question (multi-turn history test)…{RESET}")
    history = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": full_answer},
    ]
    followup = "What datasets do those papers use for evaluation?"
    print(f"  {CYAN}Follow-up:{RESET} {followup}")
    print(f"  {CYAN}Answer:{RESET} ", end="", flush=True)
    fu_tokens = []
    fu_iter, _ = rag_chat(followup, history=history)
    for tok in fu_iter:
        print(tok, end="", flush=True)
        fu_tokens.append(tok)
    print()
    check("Multi-turn follow-up works", len(fu_tokens) > 0, f"tokens={len(fu_tokens)}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RAG summarise
# ═════════════════════════════════════════════════════════════════════════════

def section_7_summarise(index_ready: bool, ollama_ready: bool):
    hdr("RAG Summarise", 7)
    if not index_ready or not ollama_ready:
        warn("Skipped — need both index and Ollama")
        return

    from rag.retriever import retrieve
    from rag.pipeline  import rag_summarise

    # Find a real indexed doc_id
    hits = retrieve("deep learning image classification", top_k=1)
    if not hits:
        warn("No hits returned — skipping summarise test")
        return

    doc_id    = hits[0]["doc_id"]
    doc_title = hits[0].get("title", "Unknown")
    print(f"\n  {CYAN}Summarising:{RESET} \"{doc_title[:70]}\"")
    print(f"  {DIM}doc_id={doc_id}{RESET}\n")

    tokens = []
    print(f"  {CYAN}Summary:{RESET} ", end="", flush=True)
    t0 = time.perf_counter()
    token_iter, citations = rag_summarise(doc_id, doc_title)
    for tok in token_iter:
        print(tok, end="", flush=True)
        tokens.append(tok)
    ms = (time.perf_counter() - t0) * 1000
    print()

    check("rag_summarise returns tokens",    len(tokens) > 0,      f"tokens={len(tokens)}")
    check("Summary non-empty",               len("".join(tokens).strip()) > 20)
    check("Summary latency < 60 s",          ms < 60_000,          f"{ms:.0f} ms")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RAG compare
# ═════════════════════════════════════════════════════════════════════════════

def section_8_compare(index_ready: bool, ollama_ready: bool):
    hdr("RAG Compare Papers", 8)
    if not index_ready or not ollama_ready:
        warn("Skipped — need both index and Ollama")
        return

    from rag.retriever import retrieve
    from rag.pipeline  import rag_compare

    # Find 2 distinct papers
    hits_a = retrieve("object detection convolutional",   top_k=3)
    hits_b = retrieve("natural language processing BERT", top_k=3)

    if not hits_a or not hits_b:
        warn("Not enough hits for comparison test — skipping")
        return

    doc_id_a = hits_a[0]["doc_id"]
    doc_id_b = hits_b[0]["doc_id"]
    title_a  = hits_a[0].get("title","Paper A")[:60]
    title_b  = hits_b[0].get("title","Paper B")[:60]

    print(f"\n  {CYAN}Paper A:{RESET} \"{title_a}\"")
    print(f"  {CYAN}Paper B:{RESET} \"{title_b}\"")
    print(f"  {DIM}Comparing on: Methodology, Key findings{RESET}\n")

    tokens = []
    print(f"  {CYAN}Comparison:{RESET} ", end="", flush=True)
    t0 = time.perf_counter()
    token_iter, citations = rag_compare(
        [doc_id_a, doc_id_b],
        [title_a,  title_b],
        dimensions=["Methodology", "Key findings"],
    )
    for tok in token_iter:
        print(tok, end="", flush=True)
        tokens.append(tok)
    ms = (time.perf_counter() - t0) * 1000
    print()

    check("rag_compare returns tokens",  len(tokens) > 0, f"tokens={len(tokens)}")
    check("Compare latency < 90 s",      ms < 90_000,     f"{ms:.0f} ms")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RAG gap analysis
# ═════════════════════════════════════════════════════════════════════════════

def section_9_gaps(index_ready: bool, ollama_ready: bool):
    hdr("RAG Gap Analysis", 9)
    if not index_ready or not ollama_ready:
        warn("Skipped — need both index and Ollama")
        return

    from rag.pipeline import rag_gap

    topic = "federated learning for medical image segmentation"
    print(f"\n  {CYAN}Topic:{RESET} {topic}\n")

    tokens = []
    print(f"  {CYAN}Gaps:{RESET} ", end="", flush=True)
    t0 = time.perf_counter()
    token_iter, citations = rag_gap(topic, top_k=5)
    for tok in token_iter:
        print(tok, end="", flush=True)
        tokens.append(tok)
    ms = (time.perf_counter() - t0) * 1000
    print()

    check("rag_gap returns tokens",  len(tokens) > 0, f"tokens={len(tokens)}")
    check("Gap analysis < 90 s",     ms < 90_000,     f"{ms:.0f} ms")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Interactive REPL
# ═════════════════════════════════════════════════════════════════════════════

def section_10_repl(index_ready: bool, ollama_ready: bool):
    hdr("Interactive RAG REPL", 10)

    if not index_ready:
        warn("Index is empty — retrieval will not work. Run indexing first.")
    if not ollama_ready:
        warn("Ollama not ready — generation will not work. Run: ollama serve")
    if not index_ready or not ollama_ready:
        print(f"\n  {DIM}Fix the above issues then re-run to use the REPL.{RESET}")
        return

    from rag.pipeline  import rag_chat
    from rag.retriever import retrieve

    print(f"""
  {BOLD}Commands:{RESET}
    {CYAN}/retrieve <query>{RESET}  — show top 5 retrieved passages (no generation)
    {CYAN}/clear{RESET}             — clear conversation history
    {CYAN}/history{RESET}           — show current conversation history
    {CYAN}/quit{RESET} or Ctrl+C   — exit

  Type any question to get a full RAG answer with citations.
""")

    history: list[dict] = []

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Exiting REPL.{RESET}")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"{DIM}Exiting REPL.{RESET}")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print(f"  {DIM}History cleared.{RESET}")
            continue

        if user_input.lower() == "/history":
            if not history:
                print(f"  {DIM}No history yet.{RESET}")
            for i, m in enumerate(history):
                role = m["role"].upper()
                snip = m["content"][:100].replace("\n", " ")
                print(f"  {DIM}[{i}] {role}: {snip}…{RESET}")
            continue

        if user_input.lower().startswith("/retrieve "):
            q = user_input[10:].strip()
            if not q:
                print(f"  {DIM}Usage: /retrieve <query>{RESET}")
                continue
            hits = retrieve(q, top_k=5)
            if not hits:
                print(f"  {DIM}No results.{RESET}")
            else:
                print(f"\n  {CYAN}Retrieved {len(hits)} passages:{RESET}")
                for i, h in enumerate(hits, 1):
                    score = h.get("relevance_score", 0)
                    title = h.get("title","")[:70]
                    year  = h.get("year","")
                    text  = h.get("text","")[:150]
                    print(f"\n  {BOLD}[{i}]{RESET} {title} ({year})")
                    print(f"      {GREEN}Score: {score:.3f}{RESET}")
                    print(f"      {DIM}{text}…{RESET}")
            continue

        # ── RAG answer ────────────────────────────────────────────────────────

        print(f"\n{BOLD}{CYAN}ResearchPilot:{RESET} ", end="", flush=True)
        tokens = []
        t0 = time.perf_counter()

        try:
            token_iter, citations = rag_chat(user_input, history=history)
            for tok in token_iter:
                print(tok, end="", flush=True)
                tokens.append(tok)
        except Exception as e:
            print(f"\n  {RED}Error: {e}{RESET}")
            continue

        ms = (time.perf_counter() - t0) * 1000
        full_answer = "".join(tokens)
        print(f"\n")

        # Show citations
        if citations:
            print(f"  {CYAN}Sources ({len(citations)}):{RESET}")
            for c in citations:
                doi_part = f"  {DIM}{c.get('doi_url','')}{RESET}" if c.get("doi_url") else ""
                print(f"    {c['label']} {c['title'][:65]} ({c['year']}){doi_part}")

        print(f"  {DIM}[{ms:.0f} ms · {len(tokens)} tokens]{RESET}\n")

        # Update history
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": full_answer})


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def _print_summary():
    total = _pass + _fail
    colour = GREEN if _fail == 0 else RED
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}Results:{RESET} "
          f"{colour}{_pass}/{total} PASS{RESET}"
          + (f"  {RED}{_fail} FAIL{RESET}" if _fail else ""))
    print(f"{'═'*60}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ResearchPilot RAG + Model CLI tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick",   action="store_true",
                        help="Run section 1 + one end-to-end call only")
    parser.add_argument("--repl",    action="store_true",
                        help="Skip all tests, go straight to interactive REPL")
    parser.add_argument("--section", type=int, choices=range(1, 11), metavar="N",
                        help="Run only section N (1-10)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*60}")
    print("  ResearchPilot — RAG + Model CLI Test")
    print(f"{'═'*60}{RESET}")

    # ── REPL-only mode ────────────────────────────────────────────────────────
    if args.repl:
        from rag.pipeline  import get_index_stats
        from rag.generator import ollama_available
        stats = get_index_stats()
        section_10_repl(stats.get("total_chunks", 0) > 0, ollama_available())
        return

    # ── Single section mode ───────────────────────────────────────────────────
    if args.section:
        # Still need system + index state for context
        ok_import = section_1_system()
        index_ok  = section_2_index() if ok_import else False
        from rag.generator import ollama_available
        ollama_ok = ollama_available()

        dispatch = {
            1: lambda: None,
            2: lambda: None,
            3: lambda: section_3_retrieval(index_ok),
            4: lambda: section_4_citations(section_3_retrieval(index_ok)),
            5: lambda: section_5_ollama(),
            6: lambda: section_6_rag_chat(index_ok, ollama_ok),
            7: lambda: section_7_summarise(index_ok, ollama_ok),
            8: lambda: section_8_compare(index_ok, ollama_ok),
            9: lambda: section_9_gaps(index_ok, ollama_ok),
            10: lambda: section_10_repl(index_ok, ollama_ok),
        }
        if args.section not in (1, 2):
            dispatch[args.section]()
        _print_summary()
        return

    # ── Full suite ────────────────────────────────────────────────────────────
    ok_import = section_1_system()
    if not ok_import:
        print(f"\n{RED}Fatal: RAG modules could not be imported. Fix errors above.{RESET}")
        sys.exit(1)

    index_ok = section_2_index()
    hits     = section_3_retrieval(index_ok)
    section_4_citations(hits)
    ollama_ok = section_5_ollama()

    if args.quick:
        if index_ok and ollama_ok:
            section_6_rag_chat(index_ok, ollama_ok)
        _print_summary()
        return

    section_6_rag_chat(index_ok, ollama_ok)
    section_7_summarise(index_ok, ollama_ok)
    section_8_compare(index_ok, ollama_ok)
    section_9_gaps(index_ok, ollama_ok)

    _print_summary()

    # Ask user if they want the REPL after the full suite
    try:
        resp = input(f"\n{BOLD}Launch interactive REPL? [y/N]{RESET} ").strip().lower()
        if resp in ("y", "yes"):
            section_10_repl(index_ok, ollama_ok)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
