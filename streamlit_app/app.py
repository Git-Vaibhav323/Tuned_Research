"""
ResearchPilot — Streamlit UI  (Phase 4 RAG Edition)
File: streamlit_app/app.py

Run:  py -3.12 -m streamlit run streamlit_app/app.py
      OR double-click LAUNCH_CHATBOT.bat
"""

from __future__ import annotations

import csv
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

# ── path so `rag` package resolves from repo root ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from rag.config import FINAL_CSV, OLLAMA_MODEL
from rag.pipeline import (
    rag_chat,
    rag_summarise,
    rag_compare,
    rag_gap,
    get_index_stats,
    index_corpus,
    index_pdf_upload,
    index_csv_upload,
    is_ollama_ready,
    format_citations_md,
)
from rag.retriever import retrieve, retrieve_for_doc
from rag.citations import format_citations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchPilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #0f1117; color: #e0e0e0; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d2f3e; }
.user-msg  { background: #1e3a5f; border-left: 4px solid #4a9eff;  padding: 12px 16px;
             border-radius: 8px; margin: 8px 0; color: #e8f4fd; }
.bot-msg   { background: #1a2438; border-left: 4px solid #7c4dff;  padding: 12px 16px;
             border-radius: 8px; margin: 8px 0; color: #e0e0e0; }
.rp-header { background: linear-gradient(135deg,#1a1d27 0%,#0f1117 100%);
             border-bottom: 2px solid #4a9eff; padding: 16px 24px;
             margin-bottom: 20px; border-radius: 10px; }
.rp-title  { font-size: 2em; font-weight: 800; color: #4a9eff; margin: 0; }
.rp-sub    { font-size: 0.9em; color: #888; margin: 4px 0 0 0; }
.metric-card { background: #1a1d27; border: 1px solid #2d2f3e; border-radius: 8px;
               padding: 12px; text-align: center; margin: 4px; }
.metric-val  { font-size: 1.6em; font-weight: 700; color: #4a9eff; }
.metric-lbl  { font-size: 0.75em; color: #888; text-transform: uppercase; }
.paper-card  { background: #1a1d27; border: 1px solid #2d2f3e; border-radius: 8px;
               padding: 14px; margin: 6px 0; border-left: 3px solid #7c4dff; }
.paper-title { font-weight: 600; color: #e0e0e0; font-size: 0.95em; }
.paper-meta  { color: #888; font-size: 0.8em; margin-top: 4px; }
.source-box  { background: #12161f; border: 1px solid #2d2f3e; border-radius: 6px;
               padding: 10px 14px; margin: 6px 0; font-size: 0.85em; color: #ccc;
               border-left: 3px solid #4a9eff; }
.chip        { display:inline-block; padding:2px 10px; border-radius:12px;
               font-size:0.75em; font-weight:600; margin:2px; }
.chip-open   { background:#1a3a1a; color:#4caf50; border:1px solid #4caf50; }
.chip-closed { background:#3a1a1a; color:#f44336; border:1px solid #f44336; }
.chip-hybrid { background:#3a3a1a; color:#ff9800; border:1px solid #ff9800; }
.stTextArea textarea { background:#1a1d27 !important; color:#e0e0e0 !important;
                       border:1px solid #2d2f3e !important; border-radius:8px !important; }
.stButton > button   { background:#4a9eff !important; color:white !important;
                       border:none !important; border-radius:8px !important;
                       font-weight:600 !important; width:100% !important; }
.stButton > button:hover { background:#2979ff !important; }
#MainMenu { visibility:hidden; } footer { visibility:hidden; } header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_papers() -> list[dict]:
    if not FINAL_CSV.exists():
        return []
    with open(FINAL_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_resource(show_spinner=False)
def get_rag_ready() -> bool:
    """Load the embedding model once per process. Cached by st.cache_resource."""
    from rag.embeddings import get_model
    get_model()
    return True


@st.cache_data(ttl=30, show_spinner=False)
def cached_index_stats() -> dict:
    """Cache index stats for 30 s to avoid hitting ChromaDB on every rerun."""
    return get_index_stats()


# ── Helpers ───────────────────────────────────────────────────────────────────

def oa_chip(status: str) -> str:
    s = status.lower()
    if s in ("gold", "diamond", "green"):
        return f'<span class="chip chip-open">{status}</span>'
    elif s == "closed":
        return f'<span class="chip chip-closed">closed</span>'
    return f'<span class="chip chip-hybrid">{status}</span>'


def keyword_search(query: str, papers: list[dict], n: int = 10) -> list[dict]:
    q = query.lower()
    scored = []
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        score = sum(text.count(w) for w in q.split() if len(w) > 3)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:n]]


def stream_to_placeholder(token_iter, placeholder) -> str:
    """Stream tokens into a Streamlit placeholder. Returns full text."""
    full = ""
    for tok in token_iter:
        full += tok
        placeholder.markdown(full + "▌")
    placeholder.markdown(full)
    return full


def render_citations(citations: list[dict]) -> None:
    """Render a citation list in a styled expander."""
    if not citations:
        return
    with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
        for c in citations:
            doi_part  = f" · [DOI]({c['doi_url']})"       if c.get("doi_url")      else ""
            opx_part  = f" · [OpenAlex]({c['openalex_url']})" if c.get("openalex_url") else ""
            page_part = f" · {c['page_note']}"              if c.get("page_note")   else ""
            st.markdown(
                f"**{c['label']}** {c['title']}  \n"
                f"<span style='color:#888;font-size:0.85em'>"
                f"{c.get('year','')}{page_part} · {c['evidence_type']} · {c['source']}"
                f"{doi_part}{opx_part}"
                f"</span>",
                unsafe_allow_html=True,
            )


def render_source_passages(hits: list[dict]) -> None:
    """Show retrieved passages with scores — Source Inspect panel."""
    if not hits:
        st.info("No passages retrieved.")
        return
    for i, h in enumerate(hits, 1):
        score = h.get("relevance_score", 0)
        bar_w = int(score * 100)
        bar_color = "#4caf50" if score >= 0.4 else "#ff9800" if score >= 0.15 else "#888"
        st.markdown(
            f"<div class='source-box'>"
            f"<b>[{i}] {h.get('title','')[:80]}</b> "
            f"<span style='color:#888'>({h.get('year','')})</span><br>"
            f"<div style='margin:4px 0'>"
            f"<span style='color:#888;font-size:0.8em'>Relevance: </span>"
            f"<span style='background:{bar_color};border-radius:4px;"
            f"padding:1px 8px;font-size:0.8em;color:#fff'>{score:.2f}</span> "
            f"<span style='color:#666;font-size:0.75em'>"
            f"(L2 dist {h.get('distance',0):.3f})</span>"
            f"</div>"
            f"<div style='color:#aaa;margin-top:6px;font-size:0.85em'>{h.get('text','')[:400]}…</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Session state init ────────────────────────────────────────────────────────

def _init():
    defaults = {
        "page":               "rag_chat",
        "rag_messages":       [],       # RAG chat history
        "plain_messages":     [],       # plain Ollama chat (legacy tab)
        "paper_chat_history": [],
        "uploaded_paper_text": "",
        "uploaded_paper_name": "",
        "uploaded_doc_id":    "",
        "compare_doc_ids":    [],
        "compare_titles":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 ResearchPilot")
    st.markdown("*AI Research Assistant · Phase 4 RAG*")
    st.markdown("---")

    # Navigation
    nav_items = [
        ("🔍 RAG Research Chat",    "rag_chat"),
        ("📄 Analyse / Summarise",  "upload"),
        ("⚖️ Compare Papers",       "compare"),
        ("💡 Research Gaps",        "gaps"),
        ("🔎 Source Inspector",     "sources"),
        ("📚 Browse Papers",        "papers"),
        ("📊 Project Stats",        "stats"),
        ("⚙️ Index Management",    "index"),
    ]
    for label, key in nav_items:
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(label, key=f"nav_{key}", type=btn_type):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")

    # Status indicators
    ollama_ok = is_ollama_ready()
    idx_stats = cached_index_stats()
    n_chunks  = idx_stats.get("total_chunks", 0)
    n_docs    = idx_stats.get("unique_docs",  0)

    st.markdown(f"**Ollama:** {'🟢 ' + OLLAMA_MODEL if ollama_ok else '🔴 Not running'}")
    st.markdown(f"**Index:** {'🟢' if n_chunks > 0 else '🟡'} {n_docs} papers · {n_chunks} chunks")

    if n_chunks == 0:
        st.warning("Index is empty — go to ⚙️ Index Management to build it.")
    if not ollama_ok:
        st.warning("Start Ollama before using chat features.")

    st.markdown("---")

    # Quick clear buttons
    if st.button("🗑️ Clear RAG chat"):
        st.session_state.rag_messages = []
        st.rerun()
    if st.button("🗑️ Clear plain chat"):
        st.session_state.plain_messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.75em;color:#666;'>
<b>ResearchPilot v2.0</b><br>
Phase 4 — RAG Pipeline<br>
Embed: all-MiniLM-L6-v2<br>
Vector DB: ChromaDB 1.5.9<br>
LLM: Qwen2.5-1.5B (Ollama)<br>
Runs 100% locally
</div>
""", unsafe_allow_html=True)


# Load papers lazily — cached after first run, safe to call here
papers = load_papers()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: RAG RESEARCH CHAT
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "rag_chat":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">🔍 RAG Research Chat</div>
  <div class="rp-sub">Ask questions about the indexed literature — every answer is grounded in retrieved evidence with citations</div>
</div>
""", unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("The index is empty. Go to **⚙️ Index Management** and click 'Index full corpus' first.")
        st.stop()

    # Display history
    for msg in st.session_state.rag_messages:
        avatar = "👤" if msg["role"] == "user" else "🔬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # Suggested questions (empty state)
    if not st.session_state.rag_messages:
        st.markdown("#### Try asking:")
        cols = st.columns(2)
        suggestions = [
            "What transformer architectures are used for object detection?",
            "Compare attention mechanisms across different vision papers",
            "What are the main limitations of current deep learning approaches?",
            "Which papers discuss federated learning or privacy-preserving ML?",
            "What datasets are commonly used for image classification?",
            "What evaluation metrics are used for object detection models?",
        ]
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {sug[:55]}", key=f"rag_sug_{i}"):
                    st.session_state.rag_messages.append({"role": "user", "content": sug})
                    st.rerun()

    # Chat input
    user_q = st.chat_input("Ask a research question…", key="rag_input")

    if user_q and user_q.strip():
        q = user_q.strip()
        st.session_state.rag_messages.append({"role": "user", "content": q})

        with st.chat_message("user", avatar="👤"):
            st.markdown(q)

        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            history = [
                m for m in st.session_state.rag_messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            with st.spinner("Retrieving evidence…"):
                token_iter, citations = rag_chat(q, history=history)
            full_answer = stream_to_placeholder(token_iter, placeholder)
            render_citations(citations)

        st.session_state.rag_messages.append({
            "role":      "assistant",
            "content":   full_answer,
            "citations": citations,
        })
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSE / SUMMARISE PAPER
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "upload":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">📄 Analyse a Paper</div>
  <div class="rp-sub">Upload a PDF or CSV, or paste text — then summarise, ask questions, and add to the index</div>
</div>
""", unsafe_allow_html=True)

    # ── extraction helpers ─────────────────────────────────────────────────────

    def extract_pdf(uploaded_file) -> tuple[str, int]:
        """Return (text, page_count). Falls back to OCR for scanned PDFs."""
        try:
            import pymupdf
            data = uploaded_file.read()
            doc  = pymupdf.open(stream=data, filetype="pdf")
            n_pages = len(doc)

            # First pass: try normal text extraction
            pages = [p.get_text("text") for p in doc]
            text  = "\n".join(p for p in pages if p.strip())

            # If text is too short, the PDF is likely scanned — try OCR
            if len(text.strip()) < 100:
                ocr_pages = []
                tessdata = r"C:\Program Files\Tesseract-OCR\tessdata"
                for page in doc:
                    try:
                        tp = page.get_textpage_ocr(
                            flags=pymupdf.TEXT_PRESERVE_WHITESPACE,
                            dpi=200,
                            full=True,
                            tessdata=tessdata,
                        )
                        ocr_pages.append(tp.extractText())
                    except Exception:
                        ocr_pages.append("")
                ocr_text = "\n".join(p for p in ocr_pages if p.strip())
                doc.close()
                if ocr_text.strip():
                    return ocr_text, n_pages
                return "", n_pages   # OCR also failed

            doc.close()
            return text, n_pages
        except ImportError:
            return "", 0
        except Exception as e:
            return f"PDF extraction error: {e}", 0

    def truncate(text: str, max_chars: int = 3000) -> str:
        return text[:max_chars] + "\n\n[truncated…]" if len(text) > max_chars else text

    # ── upload area ────────────────────────────────────────────────────────────

    col_up, col_info = st.columns([2, 1])

    with col_up:
        st.markdown("#### Step 1 — Provide the paper")
        tab_pdf, tab_text, tab_csv_up = st.tabs(["📎 Upload PDF", "📝 Paste Text", "📊 Upload CSV"])

        with tab_pdf:
            uploaded_pdf = st.file_uploader("Upload a research paper PDF",
                                             type=["pdf"], key="pdf_uploader")
            if uploaded_pdf:
                with st.spinner("Extracting text…"):
                    text, n_pages = extract_pdf(uploaded_pdf)
                if not text or text.startswith("PDF extraction error"):
                    st.error(text or "Could not extract text from this PDF. If it is a scanned document, OCR may not have found readable text.")
                else:
                    st.session_state.uploaded_paper_text = text
                    st.session_state.uploaded_paper_name = uploaded_pdf.name
                    st.session_state.paper_chat_history  = []
                    st.success(f"✅ Extracted {len(text):,} chars from {n_pages} pages")

                    # Offer to index
                    if st.button("➕ Add to RAG index", key="idx_pdf_btn"):
                        uploaded_pdf.seek(0)
                        res = index_pdf_upload(uploaded_pdf.read(), uploaded_pdf.name)
                        if res["status"] == "ok":
                            st.success(res["message"])
                            st.session_state.uploaded_doc_id = res.get("doc_id", "")
                            st.cache_data.clear()
                        else:
                            st.error(res["message"])

        with tab_text:
            pasted = st.text_area("Paste title + abstract (or full text):",
                                   height=200, placeholder="Title: …\nAbstract: …")
            pname  = st.text_input("Paper name:", placeholder="e.g. my_paper.txt")
            if st.button("Use this text ▶", key="use_pasted"):
                if pasted.strip():
                    st.session_state.uploaded_paper_text = pasted.strip()
                    st.session_state.uploaded_paper_name = pname or "Pasted text"
                    st.session_state.paper_chat_history  = []
                    st.success(f"✅ Loaded {len(pasted):,} chars")
                else:
                    st.warning("Paste some text first.")

        with tab_csv_up:
            uploaded_csv = st.file_uploader("Upload a CSV with paper metadata",
                                             type=["csv"], key="csv_uploader")
            if uploaded_csv:
                if st.button("Index this CSV ▶", key="idx_csv_btn"):
                    res = index_csv_upload(uploaded_csv.read(), uploaded_csv.name)
                    if res["status"] == "ok":
                        st.success(res["message"])
                        for w in res.get("warnings", []):
                            st.warning(w)
                        st.cache_data.clear()
                    else:
                        st.error(res["message"])

    with col_info:
        st.markdown("#### What you can do")
        st.markdown("""
<div class="paper-card">
<b>Once a paper is loaded:</b><br><br>
📋 Summarise with RAG citations<br>
🔍 Ask questions about it<br>
➕ Add it to the search index<br>
⚖️ Then compare it with others
</div>
""", unsafe_allow_html=True)

    # ── loaded paper actions ───────────────────────────────────────────────────

    if st.session_state.uploaded_paper_text:
        st.markdown("---")
        paper_text = st.session_state.uploaded_paper_text
        paper_name = st.session_state.uploaded_paper_name
        doc_id     = st.session_state.uploaded_doc_id

        st.markdown(f"#### Loaded: `{paper_name}`")

        with st.expander(f"Preview ({len(paper_text):,} chars)", expanded=False):
            st.text(paper_text[:1500] + ("…" if len(paper_text) > 1500 else ""))

        # Action tabs
        act_tab1, act_tab2, act_tab3 = st.tabs(["📋 RAG Summarise", "❓ Ask Questions", "🔬 Source-Only Q&A"])

        # ── Tab 1: RAG Summarise ──────────────────────────────────────────────
        with act_tab1:
            st.markdown("Generate a structured summary using the indexed literature as additional context.")
            if st.button("Generate RAG Summary ▶", key="gen_sum_btn"):
                if not doc_id:
                    st.warning("Index this paper first (use the '➕ Add to RAG index' button above) to enable RAG summarisation.")
                else:
                    with st.chat_message("assistant", avatar="🔬"):
                        placeholder = st.empty()
                        token_iter, citations = rag_summarise(doc_id, paper_name)
                        answer = stream_to_placeholder(token_iter, placeholder)
                        render_citations(citations)

        # ── Tab 2: Ask with RAG context ───────────────────────────────────────
        with act_tab2:
            st.markdown("Ask questions — the model retrieves relevant literature AND uses the uploaded paper.")

            for msg in st.session_state.paper_chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🔬"):
                    st.markdown(msg.get("content_display", msg.get("content", "")))
                    if msg.get("citations"):
                        render_citations(msg["citations"])

            user_pq = st.chat_input("Ask about this paper…", key="paper_rag_chat")
            if user_pq and user_pq.strip():
                q = user_pq.strip()
                with st.chat_message("user", avatar="👤"):
                    st.markdown(q)

                with st.chat_message("assistant", avatar="🔬"):
                    placeholder = st.empty()
                    token_iter, citations = rag_chat(q, history=st.session_state.paper_chat_history)
                    answer = stream_to_placeholder(token_iter, placeholder)
                    render_citations(citations)

                st.session_state.paper_chat_history.append({"role": "user", "content": q, "content_display": q})
                st.session_state.paper_chat_history.append({"role": "assistant", "content": answer, "citations": citations})
                st.rerun()

        # ── Tab 3: Source-only Q&A (no retrieval) ────────────────────────────
        with act_tab3:
            st.markdown("Answer questions using **only** the text of the uploaded paper (no RAG retrieval).")
            quick_cols = st.columns(4)
            quick_actions = [
                ("📋 Summarise",    "Summarise this paper clearly for a postgraduate student."),
                ("🔍 Methodology",  "What research methodology does this paper use?"),
                ("⚠️ Limitations",  "What are the limitations of this paper?"),
                ("💡 Contributions","What are the main contributions and findings?"),
            ]
            for col, (btn_lbl, action_prompt) in zip(quick_cols, quick_actions):
                with col:
                    if st.button(btn_lbl, key=f"qa_{btn_lbl}"):
                        st.session_state["_paper_action"] = action_prompt

            src_q = st.chat_input("Ask about this paper's content only…", key="paper_src_chat")
            if hasattr(st.session_state, "_paper_action") and not src_q:
                src_q = st.session_state.pop("_paper_action")

            if src_q and src_q.strip():
                from rag.generator import _call_ollama
                trunc = truncate(paper_text, 3000)
                sys_msg = (
                    "The user has uploaded a research paper. Answer using ONLY the content provided. "
                    "Do not fabricate details not in the paper. If the paper lacks the information, say so.\n\n"
                    f"--- PAPER ---\n{trunc}\n--- END ---"
                )
                messages = [
                    {"role": "system",    "content": sys_msg},
                    {"role": "user",      "content": src_q.strip()},
                ]
                with st.chat_message("user", avatar="👤"):
                    st.markdown(src_q.strip())
                with st.chat_message("assistant", avatar="🔬"):
                    placeholder = st.empty()
                    stream_to_placeholder(_call_ollama(messages), placeholder)

        if st.button("🗑️ Clear paper", key="clear_paper"):
            st.session_state.uploaded_paper_text = ""
            st.session_state.uploaded_paper_name = ""
            st.session_state.uploaded_doc_id     = ""
            st.session_state.paper_chat_history   = []
            st.rerun()

    else:
        st.info("Upload a PDF or paste text above.")
        if st.button("📄 Load demo: Attention Is All You Need"):
            st.session_state.uploaded_paper_text = (
                "Title: Attention Is All You Need\n\n"
                "Abstract: The dominant sequence transduction models are based on complex recurrent or "
                "convolutional neural networks. We propose the Transformer, based solely on attention "
                "mechanisms. Experiments on two machine translation tasks show these models superior in "
                "quality while being more parallelizable. Our model achieves 28.4 BLEU on WMT 2014 EN-DE.\n\n"
                "Authors: Vaswani et al., 2017. NeurIPS."
            )
            st.session_state.uploaded_paper_name = "Attention Is All You Need (2017)"
            st.session_state.paper_chat_history  = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE PAPERS
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "compare":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">⚖️ Compare Papers</div>
  <div class="rp-sub">Select 2–4 indexed papers and compare research objectives, methods, findings, and limitations</div>
</div>
""", unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("Index is empty — go to ⚙️ Index Management first.")
        st.stop()

    st.markdown("#### Select papers to compare")
    st.info("Search the index below. Click 'Add to comparison' for 2–4 papers, then click Compare.")

    # Search box
    cmp_q = st.text_input("Search indexed papers:", key="cmp_search",
                           placeholder="e.g. object detection transformer")

    if cmp_q.strip():
        hits = retrieve(cmp_q.strip(), top_k=8)
        if hits:
            for h in hits:
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    st.markdown(
                        f"<div class='source-box'><b>{h.get('title','')[:90]}</b> "
                        f"<span style='color:#888'>({h.get('year','')})</span></div>",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    already = h.get("doc_id", "") in st.session_state.compare_doc_ids
                    btn_label = "✓ Added" if already else "➕ Add"
                    if st.button(btn_label, key=f"cmp_add_{h.get('chunk_id','')}",
                                 disabled=already or len(st.session_state.compare_doc_ids) >= 4):
                        st.session_state.compare_doc_ids.append(h["doc_id"])
                        st.session_state.compare_titles.append(h.get("title", "")[:80])
                        st.rerun()
        else:
            st.info("No results — try different keywords.")

    # Show selection
    if st.session_state.compare_doc_ids:
        st.markdown(f"**Selected ({len(st.session_state.compare_doc_ids)}/4):**")
        for i, (did, title) in enumerate(zip(
                st.session_state.compare_doc_ids, st.session_state.compare_titles)):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f"  {i+1}. {title}")
            with col2:
                if st.button("✖", key=f"cmp_rm_{i}"):
                    st.session_state.compare_doc_ids.pop(i)
                    st.session_state.compare_titles.pop(i)
                    st.rerun()

        col_cmp, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️ Clear all"):
                st.session_state.compare_doc_ids  = []
                st.session_state.compare_titles   = []
                st.rerun()

        with col_cmp:
            dims = st.multiselect(
                "Comparison dimensions:",
                ["Research objective", "Methodology", "Dataset", "Evaluation metrics",
                 "Key findings", "Limitations", "Future directions"],
                default=["Research objective", "Methodology", "Key findings", "Limitations"],
            )

        if st.button("⚖️ Compare selected papers ▶", key="run_compare",
                     disabled=len(st.session_state.compare_doc_ids) < 2):
            with st.chat_message("assistant", avatar="🔬"):
                placeholder = st.empty()
                token_iter, citations = rag_compare(
                    st.session_state.compare_doc_ids,
                    st.session_state.compare_titles,
                    dimensions=dims or None,
                )
                answer = stream_to_placeholder(token_iter, placeholder)
                render_citations(citations)
    else:
        st.info("Search and add 2–4 papers above to compare them.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH GAPS
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "gaps":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">💡 Research Gaps</div>
  <div class="rp-sub">Discover candidate research gaps based on retrieved literature — always verify novelty against broader sources</div>
</div>
""", unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("Index is empty — go to ⚙️ Index Management first.")
        st.stop()

    st.info(
        "ℹ️ Gap suggestions are based on the retrieved passages only. "
        "They are candidate ideas — not verified research gaps. "
        "Always validate against the broader literature."
    )

    gap_q = st.text_input("Enter a research topic or question:",
                           placeholder="e.g. transformer models for medical image segmentation")

    col_k, col_run = st.columns([1, 2])
    with col_k:
        gap_k = st.slider("Retrieve top-k papers:", 3, 12, 6)

    if st.button("💡 Suggest research gaps ▶", key="run_gaps", disabled=not gap_q.strip()):
        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            token_iter, citations = rag_gap(gap_q.strip(), top_k=gap_k)
            answer = stream_to_placeholder(token_iter, placeholder)
            render_citations(citations)

        # Also show which papers were used
        hits = retrieve(gap_q.strip(), top_k=gap_k)
        st.markdown("---")
        st.markdown("**Papers used as evidence:**")
        render_source_passages(hits)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SOURCE INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "sources":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">🔎 Source Inspector</div>
  <div class="rp-sub">See exactly which passages are retrieved for any query — with relevance scores and metadata</div>
</div>
""", unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("Index is empty — go to ⚙️ Index Management first.")
        st.stop()

    src_q = st.text_input("Enter a query to inspect retrieved sources:",
                           placeholder="e.g. attention mechanism in vision transformers")
    col_k2, col_d2 = st.columns(2)
    with col_k2:
        src_k = st.slider("Top-k results:", 1, 15, 5)
    with col_d2:
        src_maxd = st.slider("Max L2 distance:", 0.5, 2.0, 1.6, step=0.1)

    if src_q.strip():
        from rag.retriever import retrieve
        hits = retrieve(src_q.strip(), top_k=src_k, max_distance=src_maxd)

        st.markdown(f"**{len(hits)} passages retrieved** for: *{src_q}*")

        if hits:
            # Summary table
            rows = [{
                "Title":     h.get("title", "")[:60],
                "Year":      h.get("year",  ""),
                "Score":     h.get("relevance_score", 0),
                "Distance":  h.get("distance", 0),
                "Evidence":  "Full text" if h.get("is_fulltext") else "Abstract",
                "Source":    h.get("source", ""),
            } for h in hits]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**Full passage content:**")
            render_source_passages(hits)
        else:
            st.info("No passages match this query within the distance threshold.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BROWSE PAPERS
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "papers":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">📚 Browse Papers</div>
  <div class="rp-sub">Search and explore 2,000 AI/ML papers in the OpenAlex corpus</div>
</div>
""", unsafe_allow_html=True)

    if not papers:
        st.error(f"Dataset not found at {FINAL_CSV}")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("🔍 Search by title, abstract or topic:",
                                   placeholder="e.g. YOLO object detection")
        with col2:
            oa_filter = st.selectbox("OA Status", ["All", "fully_open", "partially_open", "closed"])

        filtered = papers
        if oa_filter != "All":
            filtered = [p for p in filtered if p.get("oa_category") == oa_filter]

        if query.strip():
            filtered = keyword_search(query, filtered, n=20)
            st.markdown(f"**{len(filtered)} results** for *{query}*")
        else:
            filtered = random.sample(filtered, min(20, len(filtered)))
            st.markdown("Showing 20 random papers — search to filter")

        for p in filtered:
            title    = p.get("title", "Untitled")[:120]
            year     = p.get("publication_year", "")
            cited    = p.get("cited_by_count", "0")
            oa_s     = p.get("oa_status", "")
            oa_cat   = p.get("oa_category", "")
            abstract = p.get("abstract", "")[:200]

            chip_html = oa_chip(oa_s)
            st.markdown(f"""
<div class="paper-card">
  <div class="paper-title">{title}</div>
  <div class="paper-meta">
    📅 {year} &nbsp;·&nbsp; 📊 {cited} citations &nbsp;·&nbsp;
    {chip_html} &nbsp; <span style="color:#888;font-size:0.85em">{oa_cat}</span>
  </div>
  <div style="color:#aaa;font-size:0.85em;margin-top:6px">{abstract}…</div>
</div>
""", unsafe_allow_html=True)

            col_a, col_b = st.columns([1, 4])
            with col_a:
                if st.button("Ask RAG 🔬", key=f"ask_{p.get('id',title)[:20]}"):
                    prompt = (f"Summarise this paper:\n\nTitle: {p.get('title','')}\n"
                              f"Year: {year}\nAbstract: {p.get('abstract','')[:400]}")
                    st.session_state.rag_messages.append({"role": "user", "content": prompt})
                    st.session_state.page = "rag_chat"
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PROJECT STATS
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "stats":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">📊 Project Statistics</div>
  <div class="rp-sub">ResearchPilot DA3 — verified ML pipeline results + Phase 4 RAG stats</div>
</div>
""", unsafe_allow_html=True)

    # KPIs
    kpi_cols = st.columns(6)
    kpis = [("2,000","Papers"),("27→80","Features"),("12","ML Models"),
            ("57.5%","Best Acc"),("0.5701","Best F1"),("0.7577","Best AUC")]
    for col, (val, lbl) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
<div class="metric-card"><div class="metric-val">{val}</div>
<div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Model Leaderboard (Test Set, n=301)")
        leaderboard = [
            ("SVM-RBF (Tuned)",    "0.5748","0.5701","—",     "Tuned"),
            ("LDA",                "0.5748","0.5696","0.7577","Baseline"),
            ("MLP",                "0.5681","0.5657","0.7659","Baseline"),
            ("Logistic Regression","0.5648","0.5631","0.7565","Baseline"),
            ("Elastic Net",        "0.5581","0.5539","0.7584","Baseline"),
            ("RF (Tuned)",         "0.5615","0.5475","—",     "Tuned"),
            ("SVM-Linear",         "0.5515","0.5441","0.7587","Baseline"),
            ("Gradient Boosting",  "0.5382","0.5384","0.7402","Baseline"),
            ("Random Forest",      "0.5482","0.5317","0.7572","Baseline"),
            ("AdaBoost",           "0.5150","0.5123","0.6963","Baseline"),
            ("Decision Tree",      "0.5083","0.5017","0.6924","Baseline"),
            ("Naive Bayes",        "0.4186","0.4081","0.6840","Baseline"),
            ("XGBoost (Tuned)",    "0.3555","0.3537","—",     "Tuned"),
            ("XGBoost",            "0.3455","0.3428","0.5243","Baseline"),
        ]
        df_lb = pd.DataFrame(leaderboard, columns=["Model","Accuracy","Macro-F1","ROC-AUC","Type"])
        st.dataframe(df_lb, use_container_width=True, height=450, hide_index=True)

    with col2:
        st.markdown("#### 📈 Statistical Analysis")
        st.markdown("""
<div class="metric-card" style="text-align:left;padding:20px;">
  <div style="font-size:1em;color:#4a9eff;font-weight:700;margin-bottom:12px">Friedman Test</div>
  <div style="margin:6px 0"><span style="color:#888">χ²(11):</span> <b style="color:#4a9eff">43.4615</b></div>
  <div style="margin:6px 0"><span style="color:#888">p-value:</span> <b style="color:#4a9eff">0.000009</b></div>
  <div style="margin:6px 0"><span style="color:#888">Decision:</span> <b style="color:#4caf50">REJECT H₀ ✅</b></div>
  <hr style="border-color:#2d2f3e">
  <div>Nemenyi post-hoc: <b>7 significant pairs</b></div>
  <div style="font-size:0.85em;color:#666;margin-top:8px">XGBoost &amp; Naive Bayes provably inferior (p&lt;0.05)</div>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 🤖 Fine-Tuned Model")
        st.markdown("""
<div class="metric-card" style="text-align:left;padding:16px;">
  <div><span style="color:#888">Base:</span> Qwen2.5-1.5B-Instruct</div>
  <div><span style="color:#888">Method:</span> QLoRA 4-bit NF4</div>
  <div><span style="color:#888">Examples:</span> 8,343</div>
  <div><span style="color:#888">Train loss:</span> 2.84 → 0.49</div>
  <div><span style="color:#888">GGUF size:</span> 934 MB (Q4_K_M)</div>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 🔍 RAG Index")
        rag_cols = st.columns(2)
        with rag_cols[0]:
            st.markdown(f"""<div class="metric-card">
<div class="metric-val">{n_docs}</div><div class="metric-lbl">Indexed Papers</div></div>""",
                        unsafe_allow_html=True)
        with rag_cols[1]:
            st.markdown(f"""<div class="metric-card">
<div class="metric-val">{n_chunks}</div><div class="metric-lbl">Chunks</div></div>""",
                        unsafe_allow_html=True)

    # OA distribution
    if papers:
        st.markdown("---")
        st.markdown("#### 📊 Open-Access Distribution")
        oa_counts = Counter(p.get("oa_category", "unknown") for p in papers)
        oa_df = pd.DataFrame([
            {"Category": k, "Count": v, "Pct": f"{100*v/len(papers):.1f}%"}
            for k, v in oa_counts.most_common()
        ])
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(oa_df, hide_index=True, use_container_width=True)
        with c2:
            st.bar_chart(oa_df.set_index("Category")["Count"])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INDEX MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "index":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">⚙️ Index Management</div>
  <div class="rp-sub">Build and manage the RAG vector index — indexing is idempotent, already-indexed chunks are skipped</div>
</div>
""", unsafe_allow_html=True)

    # Current stats
    st.markdown("#### Current index state")
    live_stats = get_index_stats()
    s_cols = st.columns(4)
    stats_display = [
        ("Total chunks",   live_stats.get("total_chunks", 0)),
        ("Unique papers",  live_stats.get("unique_docs",  0)),
        ("Store path",     str(live_stats.get("store_path", ""))),
        ("Sources",        str(live_stats.get("sources", {}))),
    ]
    for col, (lbl, val) in zip(s_cols, stats_display):
        with col:
            st.metric(lbl, str(val)[:40])

    st.markdown("---")

    # ── Index full corpus ──────────────────────────────────────────────────────
    st.markdown("#### Index OpenAlex corpus (2,000 papers)")
    st.markdown(
        "Indexes `data/final/final_dataset.csv`. "
        "Safe to run multiple times — already-indexed chunks are skipped automatically."
    )

    col_lim, col_btn = st.columns([2, 1])
    with col_lim:
        idx_limit = st.number_input(
            "Limit (0 = all 2,000):", min_value=0, max_value=2000, value=0, step=50
        )
    with col_btn:
        run_index = st.button("▶ Index corpus", key="run_index_btn", type="primary")

    if run_index:
        limit = int(idx_limit) if idx_limit > 0 else None
        label = f"Indexing {limit or 2000} papers…"

        progress_bar = st.progress(0, text=label)
        status_text  = st.empty()

        def update_progress(current: int, total: int) -> None:
            pct = current / max(total, 1)
            progress_bar.progress(pct, text=f"Indexing… {current}/{total}")
            status_text.text(f"Processed {current} papers")

        with st.spinner(label):
            result = index_corpus(limit=limit, progress_cb=update_progress)

        progress_bar.progress(1.0, text="Done!")
        st.success(
            f"✅ Indexed {result['docs_loaded']} papers → "
            f"{result['chunks_created']} chunks created, "
            f"{result['chunks_added']} new chunks added, "
            f"{result['chunks_skipped']} already existed."
        )
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Delete all ─────────────────────────────────────────────────────────────
    st.markdown("#### ⚠️ Reset index")
    st.warning("This deletes all indexed chunks permanently. You will need to re-index afterwards.")
    if st.button("🗑️ Delete entire index", key="del_index_btn"):
        if st.session_state.get("confirm_delete"):
            from rag.vector_store import reset_collection
            reset_collection()
            st.session_state.confirm_delete = False
            st.success("Index cleared. Run 'Index corpus' to rebuild.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.session_state.confirm_delete = True
            st.warning("Click again to confirm deletion.")

    st.markdown("---")
    st.markdown("#### 📖 Notes")
    st.markdown("""
- Vector store persists across restarts at `data/rag_vector_store/`
- Chunk size: 700 chars · Overlap: 100 chars · Min chunk: 80 chars
- Embedding model: `all-MiniLM-L6-v2` (384-dim, local, ~22 MB)
- ChromaDB collection: `researchpilot_papers` (L2 distance)
- Re-indexing the same paper is safe — duplicates are detected by `chunk_id`
""")
