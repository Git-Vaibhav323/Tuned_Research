"""
ResearchPilot — Streamlit UI (v3.0)
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from rag.config import FINAL_CSV, OLLAMA_MODEL
from rag.pipeline import (
    rag_chat, rag_summarise, rag_compare, rag_gap,
    get_index_stats, index_corpus, index_pdf_upload,
    index_csv_upload, is_ollama_ready, format_citations_md,
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
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0a0c14; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
    border-right: 1px solid #1f2937;
    padding-top: 0;
}

/* ── Sidebar logo area ── */
.rp-logo {
    background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
    padding: 20px 16px;
    margin: -1rem -1rem 1rem -1rem;
    text-align: center;
}
.rp-logo-title {
    font-size: 1.4em; font-weight: 800; color: #fff; letter-spacing: -0.5px;
}
.rp-logo-sub {
    font-size: 0.72em; color: rgba(255,255,255,0.7); margin-top: 2px;
}

/* ── Nav buttons ── */
.nav-btn {
    display: block; width: 100%; text-align: left;
    padding: 10px 14px; margin: 2px 0;
    border-radius: 8px; border: none; cursor: pointer;
    font-size: 0.88em; font-weight: 500; transition: all 0.15s;
    background: transparent; color: #94a3b8;
}
.nav-btn:hover { background: #1f2937; color: #e2e8f0; }
.nav-btn.active { background: #1e3a5f; color: #60a5fa; font-weight: 600; }

/* ── Status pills ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 20px; font-size: 0.75em;
    font-weight: 600; margin: 3px 0;
}
.pill-green { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-red   { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
.pill-yellow{ background: #422006; color: #fb923c; border: 1px solid #7c2d12; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #1e293b;
    border-left: 4px solid #3b82f6;
    padding: 20px 24px; margin-bottom: 24px;
    border-radius: 12px;
}
.page-title { font-size: 1.7em; font-weight: 800; color: #f1f5f9; margin: 0; }
.page-sub   { font-size: 0.88em; color: #64748b; margin: 4px 0 0 0; }

/* ── Metric card ── */
.metric-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 16px; text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3b82f6; }
.metric-val { font-size: 1.8em; font-weight: 800; color: #60a5fa; line-height: 1; }
.metric-lbl { font-size: 0.7em; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.08em; margin-top: 4px; }

/* ── Paper card ── */
.paper-card {
    background: #111827; border: 1px solid #1f2937;
    border-left: 3px solid #7c3aed; border-radius: 8px;
    padding: 14px 16px; margin: 6px 0; transition: border-color 0.2s;
}
.paper-card:hover { border-color: #60a5fa; }
.paper-title { font-weight: 600; color: #f1f5f9; font-size: 0.94em; }
.paper-meta  { color: #64748b; font-size: 0.78em; margin-top: 6px; }

/* ── Source box ── */
.source-box {
    background: #0d1117; border: 1px solid #1f2937;
    border-left: 3px solid #2563eb; border-radius: 6px;
    padding: 12px 14px; margin: 6px 0; font-size: 0.84em; color: #cbd5e1;
}

/* ── Chat messages ── */
.chat-user {
    background: #1e3a5f; border-left: 3px solid #3b82f6;
    padding: 12px 16px; border-radius: 8px; margin: 6px 0; color: #e0f2fe;
}
.chat-bot {
    background: #1a1040; border-left: 3px solid #7c3aed;
    padding: 12px 16px; border-radius: 8px; margin: 6px 0; color: #e2e8f0;
}

/* ── Chips ── */
.chip { display:inline-block; padding:2px 9px; border-radius:20px;
        font-size:0.72em; font-weight:600; margin:2px; }
.chip-open   { background:#052e16; color:#4ade80; border:1px solid #166534; }
.chip-closed { background:#450a0a; color:#f87171; border:1px solid #991b1b; }
.chip-hybrid { background:#422006; color:#fb923c; border:1px solid #7c2d12; }

/* ── Suggestion buttons ── */
.sug-btn {
    background: #111827; border: 1px solid #1f2937; border-radius: 8px;
    padding: 10px 14px; margin: 4px 0; font-size: 0.84em; color: #94a3b8;
    cursor: pointer; text-align: left; width: 100%; transition: all 0.15s;
}
.sug-btn:hover { background: #1e3a5f; color: #60a5fa; border-color: #3b82f6; }

/* ── Stat highlight box ── */
.stat-box {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 10px; padding: 20px;
}
.stat-label { color: #64748b; font-size: 0.8em; }
.stat-value { color: #60a5fa; font-size: 1.1em; font-weight: 700; }

/* ── Streamlit overrides ── */
.stTextInput > div > div > input {
    background: #111827 !important; color: #e2e8f0 !important;
    border: 1px solid #1f2937 !important; border-radius: 8px !important;
}
.stTextArea textarea {
    background: #111827 !important; color: #e2e8f0 !important;
    border: 1px solid #1f2937 !important; border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: #111827 !important; color: #e2e8f0 !important;
    border: 1px solid #1f2937 !important;
}
.stButton > button {
    background: #1e40af !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: background 0.15s !important;
}
.stButton > button:hover { background: #2563eb !important; }
.stButton > button[kind="secondary"] {
    background: #111827 !important; color: #94a3b8 !important;
    border: 1px solid #1f2937 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #1f2937 !important; color: #e2e8f0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #111827 !important; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #60a5fa !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #111827 !important; border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* ── Divider ── */
hr { border-color: #1f2937 !important; }

/* ── Hide Streamlit default chrome ── */
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0c14; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }
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
    from rag.embeddings import get_model
    get_model()
    return True


@st.cache_data(ttl=30, show_spinner=False)
def cached_index_stats() -> dict:
    return get_index_stats()


# ── Helpers ───────────────────────────────────────────────────────────────────

def oa_chip(status: str) -> str:
    s = status.lower()
    if s in ("gold", "diamond", "green"):
        return f'<span class="chip chip-open">{status}</span>'
    elif s == "closed":
        return f'<span class="chip chip-closed">closed</span>'
    return f'<span class="chip chip-hybrid">{status}</span>'


def keyword_search(query: str, papers: list[dict], n: int = 15) -> list[dict]:
    q = query.lower()
    scored = []
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        score = sum(text.count(w) for w in q.split() if len(w) > 2)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:n]]


def stream_to_placeholder(token_iter, placeholder) -> str:
    full = ""
    for tok in token_iter:
        full += tok
        placeholder.markdown(full + " ▌")
    placeholder.markdown(full)
    return full


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📚 {len(citations)} source{'s' if len(citations) != 1 else ''}", expanded=False):
        for c in citations:
            doi_part = f" · [DOI ↗]({c['doi_url']})" if c.get("doi_url") else ""
            opx_part = f" · [OpenAlex ↗]({c['openalex_url']})" if c.get("openalex_url") else ""
            st.markdown(
                f"<div class='source-box'>"
                f"<b>{c['label']}</b> {c['title']}<br>"
                f"<span style='color:#64748b;font-size:0.82em'>"
                f"{c.get('year', '')} · {c['evidence_type']} · {c['source']}"
                f"{doi_part}{opx_part}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_source_passages(hits: list[dict]) -> None:
    if not hits:
        st.info("No passages retrieved.")
        return
    for i, h in enumerate(hits, 1):
        score = h.get("relevance_score", 0)
        colour = "#4ade80" if score >= 0.4 else "#fb923c" if score >= 0.15 else "#94a3b8"
        st.markdown(
            f"<div class='source-box'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<b style='color:#f1f5f9'>[{i}] {h.get('title','')[:80]}</b>"
            f"<span style='background:{colour}22;color:{colour};border:1px solid {colour}44;"
            f"border-radius:12px;padding:2px 8px;font-size:0.75em;font-weight:600'>"
            f"{score:.2f}</span>"
            f"</div>"
            f"<div style='color:#64748b;font-size:0.78em;margin:4px 0'>"
            f"{h.get('year','')} · L2 dist {h.get('distance',0):.3f} · "
            f"{'Full text' if h.get('is_fulltext') else 'Abstract'}"
            f"</div>"
            f"<div style='color:#94a3b8;font-size:0.84em;margin-top:8px;line-height:1.5'>"
            f"{h.get('text','')[:420]}…"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "page": "rag_chat",
        "rag_messages": [],
        "paper_chat_history": [],
        "uploaded_paper_text": "",
        "uploaded_paper_name": "",
        "uploaded_doc_id": "",
        "compare_doc_ids": [],
        "compare_titles": [],
        "confirm_delete": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# Preload embedding model
get_rag_ready()

# Live status
ollama_ok  = is_ollama_ready()
idx_stats  = cached_index_stats()
n_chunks   = idx_stats.get("total_chunks", 0)
n_docs     = idx_stats.get("unique_docs", 0)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="rp-logo">
        <div class="rp-logo-title">🔬 ResearchPilot</div>
        <div class="rp-logo-sub">AI Research Assistant · Phase 4 RAG</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status ────────────────────────────────────────────────────────────────
    ollama_pill = (
        f'<span class="status-pill pill-green">● {OLLAMA_MODEL}</span>'
        if ollama_ok else
        '<span class="status-pill pill-red">● Ollama offline</span>'
    )
    index_pill = (
        f'<span class="status-pill pill-green">● {n_docs} papers · {n_chunks} chunks</span>'
        if n_chunks > 0 else
        '<span class="status-pill pill-yellow">● Index empty</span>'
    )
    st.markdown(ollama_pill, unsafe_allow_html=True)
    st.markdown(index_pill,  unsafe_allow_html=True)

    if not ollama_ok:
        st.warning("Run `ollama serve` to enable generation.", icon="⚠️")
    if n_chunks == 0:
        st.info("Go to **⚙️ Index Management** to build the index.", icon="ℹ️")

    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    nav = [
        ("🔍", "RAG Research Chat",   "rag_chat"),
        ("📄", "Analyse Paper",        "upload"),
        ("⚖️", "Compare Papers",      "compare"),
        ("💡", "Research Gaps",        "gaps"),
        ("🔎", "Source Inspector",     "sources"),
        ("📚", "Browse Papers",        "papers"),
        ("📊", "Project Stats",        "stats"),
        ("⚙️", "Index Management",    "index"),
    ]

    for icon, label, key in nav:
        active = st.session_state.page == key
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{key}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.rag_messages = []
            st.rerun()
    with col_b:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("""
    <div style='font-size:0.72em;color:#374151;margin-top:12px;line-height:1.7'>
    <b style='color:#4b5563'>ResearchPilot v3.0</b><br>
    Embed: all-MiniLM-L6-v2<br>
    Vector DB: ChromaDB 1.5.9<br>
    LLM: Qwen2.5-1.5B (Ollama)<br>
    Runs 100% locally
    </div>
    """, unsafe_allow_html=True)

# ── Load papers ───────────────────────────────────────────────────────────────
papers = load_papers()


# =============================================================================
# PAGE: RAG RESEARCH CHAT
# =============================================================================
if st.session_state.page == "rag_chat":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔍 RAG Research Chat</div>
        <div class="page-sub">Every answer is grounded in retrieved evidence from the indexed literature — with traceable citations</div>
    </div>
    """, unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("The index is empty. Go to **⚙️ Index Management** and click **Index corpus** first.", icon="⚠️")
        st.stop()

    # Chat history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🔬"):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # Empty state: suggestions
    if not st.session_state.rag_messages:
        st.markdown("#### Suggested questions")
        suggestions = [
            "What transformer architectures are used for object detection?",
            "Compare attention mechanisms across different vision papers",
            "What are the main limitations of current deep learning approaches?",
            "Which papers discuss federated learning or privacy-preserving ML?",
            "What datasets are commonly used for image classification?",
            "What evaluation metrics are used for object detection models?",
            "How is ChatGPT used in medical education?",
            "What are recent advances in protein structure prediction?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {sug[:60]}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.rag_messages.append({"role": "user", "content": sug})
                    st.rerun()

    # Chat input
    user_q = st.chat_input("Ask a research question…")

    if user_q and user_q.strip():
        q = user_q.strip()
        st.session_state.rag_messages.append({"role": "user", "content": q})

        with st.chat_message("user", avatar="👤"):
            st.markdown(q)

        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            placeholder.markdown("*Retrieving evidence…* ⏳")
            history = [m for m in st.session_state.rag_messages[:-1]
                       if m["role"] in ("user", "assistant")]
            token_iter, citations = rag_chat(q, history=history)
            full_answer = stream_to_placeholder(token_iter, placeholder)
            render_citations(citations)

        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": full_answer,
            "citations": citations,
        })
        st.rerun()


# =============================================================================
# PAGE: ANALYSE PAPER
# =============================================================================
elif st.session_state.page == "upload":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">📄 Analyse a Paper</div>
        <div class="page-sub">Upload a PDF or paste text — summarise, ask questions, or add it to the search index</div>
    </div>
    """, unsafe_allow_html=True)

    def extract_pdf(uploaded_file) -> tuple[str, int]:
        try:
            import pymupdf
            data = uploaded_file.read()
            doc  = pymupdf.open(stream=data, filetype="pdf")
            n    = len(doc)
            pages = [p.get_text("text") for p in doc]
            text  = "\n".join(p for p in pages if p.strip())
            doc.close()
            return text, n
        except ImportError:
            return "[PyMuPDF not installed — run: pip install pymupdf]", 0
        except Exception as e:
            return f"Error: {e}", 0

    def truncate(text: str, n: int = 3000) -> str:
        return text[:n] + "\n\n[truncated…]" if len(text) > n else text

    # Input area
    col_left, col_right = st.columns([3, 1])

    with col_left:
        tab_pdf, tab_text, tab_csv = st.tabs(["📎 PDF Upload", "📝 Paste Text", "📊 CSV Upload"])

        with tab_pdf:
            f = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
            if f:
                with st.spinner("Extracting text…"):
                    text, pages = extract_pdf(f)
                if text and not text.startswith("[") and not text.startswith("Error"):
                    st.session_state.uploaded_paper_text = text
                    st.session_state.uploaded_paper_name = f.name
                    st.session_state.paper_chat_history  = []
                    st.session_state.uploaded_doc_id     = ""
                    st.success(f"✅ {pages} pages · {len(text):,} characters extracted")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("➕ Add to RAG index"):
                            f.seek(0)
                            res = index_pdf_upload(f.read(), f.name)
                            if res["status"] == "ok":
                                st.success(res["message"])
                                st.session_state.uploaded_doc_id = res.get("doc_id", "")
                                st.cache_data.clear()
                            else:
                                st.error(res["message"])
                else:
                    st.error(text or "Could not extract text.")

        with tab_text:
            pasted = st.text_area("Paste title + abstract (or full text):", height=180)
            pname  = st.text_input("Label:", placeholder="e.g. My paper")
            if st.button("▶ Use this text"):
                if pasted.strip():
                    st.session_state.uploaded_paper_text = pasted.strip()
                    st.session_state.uploaded_paper_name = pname or "Pasted text"
                    st.session_state.paper_chat_history  = []
                    st.success(f"✅ Loaded {len(pasted):,} characters")
                else:
                    st.warning("Paste some text first.")

        with tab_csv:
            csv_f = st.file_uploader("Upload CSV with paper metadata", type=["csv"])
            if csv_f and st.button("▶ Index this CSV"):
                res = index_csv_upload(csv_f.read(), csv_f.name)
                if res["status"] == "ok":
                    st.success(res["message"])
                    st.cache_data.clear()
                else:
                    st.error(res["message"])

    with col_right:
        st.markdown("""
        <div class="paper-card">
        <div class="paper-title">What you can do</div>
        <div class="paper-meta" style="margin-top:10px;line-height:2">
        📋 Summarise with citations<br>
        ❓ Ask questions<br>
        🔬 Source-only Q&A<br>
        ➕ Add to index<br>
        ⚖️ Compare with others
        </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📄 Load demo paper"):
            st.session_state.uploaded_paper_text = (
                "Title: Attention Is All You Need\n\n"
                "Abstract: The dominant sequence transduction models are based on complex recurrent "
                "or convolutional neural networks. We propose the Transformer, based solely on "
                "attention mechanisms. Experiments on two machine translation tasks show these models "
                "to be superior in quality while being more parallelizable. Our model achieves 28.4 "
                "BLEU on WMT 2014 EN-DE translation.\n\nAuthors: Vaswani et al., NeurIPS 2017."
            )
            st.session_state.uploaded_paper_name = "Attention Is All You Need (2017)"
            st.session_state.paper_chat_history  = []
            st.rerun()

    # Loaded paper actions
    if st.session_state.uploaded_paper_text:
        st.markdown("---")
        text = st.session_state.uploaded_paper_text
        name = st.session_state.uploaded_paper_name
        doc_id = st.session_state.uploaded_doc_id

        col_name, col_clear = st.columns([5, 1])
        with col_name:
            st.markdown(f"**Loaded:** `{name}` · {len(text):,} chars")
        with col_clear:
            if st.button("✖ Clear"):
                for k in ("uploaded_paper_text","uploaded_paper_name",
                          "uploaded_doc_id","paper_chat_history"):
                    st.session_state[k] = "" if "history" not in k else []
                st.rerun()

        with st.expander("Preview text", expanded=False):
            st.code(text[:1200] + ("…" if len(text) > 1200 else ""), language=None)

        act1, act2, act3 = st.tabs(["📋 RAG Summarise", "❓ Ask Questions", "🔬 Source-Only Q&A"])

        with act1:
            st.markdown("Summarise using both the paper text and retrieved literature.")
            if st.button("▶ Generate RAG summary"):
                if not doc_id:
                    st.info("Add this paper to the index first to enable RAG summarisation.")
                else:
                    with st.chat_message("assistant", avatar="🔬"):
                        ph = st.empty()
                        tok_iter, cits = rag_summarise(doc_id, name)
                        stream_to_placeholder(tok_iter, ph)
                        render_citations(cits)

        with act2:
            st.markdown("Ask questions — retrieves related papers AND uses this paper.")
            for msg in st.session_state.paper_chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🔬"):
                    st.markdown(msg.get("content", ""))
                    if msg.get("citations"):
                        render_citations(msg["citations"])

            pq = st.chat_input("Ask about this paper…", key="paper_chat")
            if pq and pq.strip():
                with st.chat_message("user", avatar="👤"):
                    st.markdown(pq.strip())
                with st.chat_message("assistant", avatar="🔬"):
                    ph = st.empty()
                    tok_iter, cits = rag_chat(pq.strip(), history=st.session_state.paper_chat_history)
                    ans = stream_to_placeholder(tok_iter, ph)
                    render_citations(cits)
                st.session_state.paper_chat_history += [
                    {"role": "user",      "content": pq.strip()},
                    {"role": "assistant", "content": ans, "citations": cits},
                ]
                st.rerun()

        with act3:
            st.markdown("Answer using **only** the uploaded text — no RAG retrieval.")
            q_cols = st.columns(4)
            quick = [
                ("📋 Summarise",    "Summarise this paper clearly."),
                ("🔍 Methodology",  "What research methodology does this paper use?"),
                ("⚠️ Limitations",  "What are the limitations of this paper?"),
                ("💡 Contributions","What are the main contributions?"),
            ]
            chosen_q = None
            for col, (btn, prompt) in zip(q_cols, quick):
                with col:
                    if st.button(btn, use_container_width=True):
                        chosen_q = prompt

            src_q = st.chat_input("Ask about this paper's content only…", key="src_chat")
            final_q = chosen_q or (src_q.strip() if src_q else None)
            if final_q:
                from rag.generator import _call_ollama
                sys_msg = (
                    "Answer using ONLY the paper content below. "
                    "Do not fabricate details not in the text. "
                    "If the information is not present, say so.\n\n"
                    f"--- PAPER ---\n{truncate(text)}\n--- END ---"
                )
                msgs = [{"role": "system", "content": sys_msg},
                        {"role": "user",   "content": final_q}]
                with st.chat_message("user", avatar="👤"):
                    st.markdown(final_q)
                with st.chat_message("assistant", avatar="🔬"):
                    ph = st.empty()
                    stream_to_placeholder(_call_ollama(msgs), ph)


# =============================================================================
# PAGE: COMPARE PAPERS
# =============================================================================
elif st.session_state.page == "compare":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">⚖️ Compare Papers</div>
        <div class="page-sub">Select 2–4 indexed papers and compare objectives, methods, findings, and limitations side-by-side</div>
    </div>
    """, unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("Index is empty — go to ⚙️ Index Management first.", icon="⚠️")
        st.stop()

    col_search, col_sel = st.columns([3, 2])

    with col_search:
        st.markdown("#### Search papers")
        cmp_q = st.text_input("Search:", placeholder="e.g. transformer object detection",
                               label_visibility="collapsed")
        if cmp_q.strip():
            hits = retrieve(cmp_q.strip(), top_k=8)
            for h in hits:
                c1, c2 = st.columns([6, 1])
                with c1:
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<b>{h.get('title','')[:85]}</b> "
                        f"<span style='color:#64748b'>({h.get('year','')})</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    added = h.get("doc_id", "") in st.session_state.compare_doc_ids
                    if st.button(
                        "✓" if added else "➕",
                        key=f"add_{h.get('chunk_id','')}",
                        disabled=added or len(st.session_state.compare_doc_ids) >= 4,
                    ):
                        st.session_state.compare_doc_ids.append(h["doc_id"])
                        st.session_state.compare_titles.append(h.get("title", "")[:80])
                        st.rerun()

    with col_sel:
        st.markdown(f"#### Selected ({len(st.session_state.compare_doc_ids)}/4)")
        if st.session_state.compare_doc_ids:
            for i, (did, title) in enumerate(zip(
                    st.session_state.compare_doc_ids, st.session_state.compare_titles)):
                c1, c2 = st.columns([8, 1])
                with c1:
                    st.markdown(f"<div class='paper-card'><div class='paper-title'>"
                                f"{i+1}. {title[:65]}</div></div>", unsafe_allow_html=True)
                with c2:
                    if st.button("✖", key=f"rm_{i}"):
                        st.session_state.compare_doc_ids.pop(i)
                        st.session_state.compare_titles.pop(i)
                        st.rerun()

            dims = st.multiselect(
                "Compare on:",
                ["Research objective", "Methodology", "Dataset",
                 "Evaluation metrics", "Key findings", "Limitations",
                 "Future directions"],
                default=["Research objective", "Methodology",
                         "Key findings", "Limitations"],
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("⚖️ Compare ▶", type="primary",
                             disabled=len(st.session_state.compare_doc_ids) < 2,
                             use_container_width=True):
                    with st.chat_message("assistant", avatar="🔬"):
                        ph = st.empty()
                        tok_iter, cits = rag_compare(
                            st.session_state.compare_doc_ids,
                            st.session_state.compare_titles,
                            dimensions=dims or None,
                        )
                        stream_to_placeholder(tok_iter, ph)
                        render_citations(cits)
            with c2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.compare_doc_ids = []
                    st.session_state.compare_titles  = []
                    st.rerun()
        else:
            st.info("Search and add 2–4 papers on the left.")


# =============================================================================
# PAGE: RESEARCH GAPS
# =============================================================================
elif st.session_state.page == "gaps":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">💡 Research Gaps</div>
        <div class="page-sub">Discover candidate gaps based on retrieved literature — always verify novelty against broader sources</div>
    </div>
    """, unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("Index is empty — go to ⚙️ Index Management first.", icon="⚠️")
        st.stop()

    st.info(
        "Gap suggestions are derived from retrieved passages only — not the full literature. "
        "Treat them as starting points for your own review.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        gap_q = st.text_input("Research topic:", placeholder="e.g. transformer models for medical image segmentation")
    with col2:
        gap_k = st.slider("Retrieve top-k:", 3, 12, 6)

    # Quick topic suggestions
    st.markdown("**Quick topics:**")
    qcols = st.columns(4)
    topics = [
        "federated learning privacy",
        "LLMs in healthcare",
        "vision transformers efficiency",
        "continual learning catastrophic forgetting",
    ]
    chosen_topic = None
    for col, t in zip(qcols, topics):
        with col:
            if st.button(f"💡 {t[:30]}", use_container_width=True):
                chosen_topic = t

    run_q = chosen_topic or (gap_q.strip() if gap_q.strip() else None)

    if run_q:
        if st.button("🔍 Find research gaps ▶", type="primary") or chosen_topic:
            with st.chat_message("assistant", avatar="🔬"):
                ph = st.empty()
                tok_iter, cits = rag_gap(run_q, top_k=gap_k)
                stream_to_placeholder(tok_iter, ph)
                render_citations(cits)

            st.markdown("---")
            st.markdown("**Evidence passages used:**")
            render_source_passages(retrieve(run_q, top_k=gap_k))


# =============================================================================
# PAGE: SOURCE INSPECTOR
# =============================================================================
elif st.session_state.page == "sources":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔎 Source Inspector</div>
        <div class="page-sub">See exactly which passages are retrieved for any query — with relevance scores and raw text</div>
    </div>
    """, unsafe_allow_html=True)

    if n_chunks == 0:
        st.warning("Index is empty.", icon="⚠️")
        st.stop()

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        src_q = st.text_input("Query:", placeholder="e.g. attention mechanism vision transformers")
    with col2:
        src_k = st.slider("Top-k:", 1, 15, 5)
    with col3:
        src_d = st.slider("Max dist:", 0.5, 2.0, 1.6, 0.1)

    if src_q.strip():
        hits = retrieve(src_q.strip(), top_k=src_k, max_distance=src_d)
        st.markdown(f"**{len(hits)} passage{'s' if len(hits) != 1 else ''} retrieved** for: *{src_q}*")

        if hits:
            df = pd.DataFrame([{
                "Title":    h.get("title","")[:55],
                "Year":     h.get("year",""),
                "Score":    round(h.get("relevance_score",0), 3),
                "Distance": round(h.get("distance",0), 3),
                "Type":     "Full text" if h.get("is_fulltext") else "Abstract",
            } for h in hits])
            st.dataframe(df, use_container_width=True, hide_index=True, height=200)
            st.markdown("---")
            render_source_passages(hits)
        else:
            st.info("No passages matched. Try increasing Max distance or changing the query.")


# =============================================================================
# PAGE: BROWSE PAPERS
# =============================================================================
elif st.session_state.page == "papers":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">📚 Browse Papers</div>
        <div class="page-sub">Explore all 2,000 AI/ML papers in the OpenAlex corpus</div>
    </div>
    """, unsafe_allow_html=True)

    if not papers:
        st.error(f"Dataset not found at {FINAL_CSV}")
        st.stop()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("🔍 Search titles and abstracts:",
                               placeholder="e.g. YOLO object detection")
    with col2:
        oa_filter = st.selectbox("OA category", ["All", "fully_open", "partially_open", "closed"])
    with col3:
        year_filter = st.selectbox("Year", ["All", "2025", "2024", "2023", "2022"])

    filtered = papers
    if oa_filter != "All":
        filtered = [p for p in filtered if p.get("oa_category") == oa_filter]
    if year_filter != "All":
        filtered = [p for p in filtered if str(p.get("publication_year","")) == year_filter]

    if query.strip():
        filtered = keyword_search(query, filtered, n=20)
        st.markdown(f"**{len(filtered)} results** for *{query}*")
    else:
        filtered = random.sample(filtered, min(20, len(filtered)))
        st.caption("Showing 20 random papers — search to filter")

    for p in filtered:
        title    = p.get("title", "Untitled")[:110]
        year     = p.get("publication_year", "")
        cited    = p.get("cited_by_count", "0")
        oa_s     = p.get("oa_status", "")
        oa_cat   = p.get("oa_category", "")
        abstract = p.get("abstract", "")[:180]

        col_card, col_btn = st.columns([8, 1])
        with col_card:
            st.markdown(f"""
            <div class="paper-card">
            <div class="paper-title">{title}</div>
            <div class="paper-meta">
                📅 {year} &nbsp;·&nbsp; 📊 {cited} citations &nbsp;·&nbsp;
                {oa_chip(oa_s)} <span style="color:#64748b;font-size:0.82em">{oa_cat}</span>
            </div>
            <div style="color:#64748b;font-size:0.82em;margin-top:6px;line-height:1.5">
                {abstract}…
            </div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if st.button("🔬", key=f"ask_{p.get('id',title)[:16]}", help="Ask RAG"):
                prompt = (f"Summarise this paper:\n\nTitle: {p.get('title','')}\n"
                          f"Year: {year}\nAbstract: {p.get('abstract','')[:400]}")
                st.session_state.rag_messages.append({"role": "user", "content": prompt})
                st.session_state.page = "rag_chat"
                st.rerun()


# =============================================================================
# PAGE: PROJECT STATS
# =============================================================================
elif st.session_state.page == "stats":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Project Statistics</div>
        <div class="page-sub">ML pipeline results, statistical analysis, RAG metrics — all verified numbers</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPIs
    kpi_data = [
        ("2,000", "Papers Indexed"),
        ("27 → 80", "Feature Pipeline"),
        ("12", "ML Models"),
        ("57.5%", "Best Accuracy"),
        ("0.5701", "Best Macro-F1"),
        ("0.7577", "Best ROC-AUC"),
    ]
    cols = st.columns(6)
    for col, (val, lbl) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    tab_models, tab_stats, tab_rag, tab_oa = st.tabs([
        "🏆 Model Leaderboard",
        "📈 Statistical Analysis",
        "🤖 RAG & LLM",
        "📊 Corpus Stats",
    ])

    with tab_models:
        leaderboard = [
            ("SVM-RBF (Tuned)",    "0.5748", "0.5701", "—",     "🟠 Tuned"),
            ("LDA",                "0.5748", "0.5696", "0.7577","🔵 Baseline"),
            ("MLP",                "0.5681", "0.5657", "0.7659","🔵 Baseline"),
            ("Logistic Regression","0.5648", "0.5631", "0.7565","🔵 Baseline"),
            ("Elastic Net",        "0.5581", "0.5539", "0.7584","🔵 Baseline"),
            ("RF (Tuned)",         "0.5615", "0.5475", "—",     "🟠 Tuned"),
            ("SVM-Linear",         "0.5515", "0.5441", "0.7587","🔵 Baseline"),
            ("Gradient Boosting",  "0.5382", "0.5384", "0.7402","🔵 Baseline"),
            ("Random Forest",      "0.5482", "0.5317", "0.7572","🔵 Baseline"),
            ("AdaBoost",           "0.5150", "0.5123", "0.6963","🔵 Baseline"),
            ("Decision Tree",      "0.5083", "0.5017", "0.6924","🔵 Baseline"),
            ("Naive Bayes",        "0.4186", "0.4081", "0.6840","🔵 Baseline"),
            ("XGBoost (Tuned)",    "0.3555", "0.3537", "—",     "🟠 Tuned"),
            ("XGBoost",            "0.3455", "0.3428", "0.5243","🔵 Baseline"),
        ]
        df = pd.DataFrame(leaderboard,
                          columns=["Model", "Accuracy", "Macro-F1", "ROC-AUC", "Type"])
        st.dataframe(df, use_container_width=True, height=490, hide_index=True)
        st.caption("Test set n=301 · 3-class OA prediction · R/caret pipeline")

    with tab_stats:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="stat-box">
            <div style="font-size:1.1em;font-weight:700;color:#60a5fa;margin-bottom:16px">
                Friedman Non-Parametric Test
            </div>
            <div style="margin:8px 0">
                <span class="stat-label">Null hypothesis:</span><br>
                <span style="color:#e2e8f0;font-size:0.9em">All 12 models share the same median CV F1</span>
            </div>
            <div style="margin:12px 0">
                <span class="stat-label">Chi-squared statistic</span><br>
                <span class="stat-value">43.4615</span>
            </div>
            <div style="margin:8px 0">
                <span class="stat-label">Degrees of freedom</span><br>
                <span class="stat-value">11</span>
            </div>
            <div style="margin:8px 0">
                <span class="stat-label">p-value</span><br>
                <span class="stat-value">9 × 10⁻⁶</span>
            </div>
            <div style="margin:12px 0;padding:10px;background:#052e16;border-radius:6px">
                <span style="color:#4ade80;font-weight:700">✅ REJECT H₀</span>
                <span style="color:#86efac;font-size:0.85em"> — performance differences are real</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="stat-box">
            <div style="font-size:1.1em;font-weight:700;color:#60a5fa;margin-bottom:16px">
                Nemenyi Post-hoc Analysis
            </div>
            <div style="margin:8px 0">
                <span class="stat-label">Total pairs tested:</span>
                <span class="stat-value"> 66</span>
            </div>
            <div style="margin:8px 0">
                <span class="stat-label">Significant pairs (α=0.05):</span>
                <span class="stat-value"> 7</span>
            </div>
            <div style="margin:12px 0">
                <span class="stat-label">Key finding:</span><br>
                <span style="color:#e2e8f0;font-size:0.88em;line-height:1.6">
                XGBoost and Naive Bayes are significantly worse than<br>
                the top group (LDA, RF, Elastic Net). The top models<br>
                are <b>statistically indistinguishable</b> from each other.
                </span>
            </div>
            <div style="margin:8px 0">
                <span class="stat-label">Top feature (Gini):</span><br>
                <span class="stat-value">pub_ieee = 43.78</span>
                <span style="color:#64748b;font-size:0.8em"> (publisher flag)</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_rag:
        c1, c2, c3 = st.columns(3)
        rag_kpis = [
            ("5,796", "Chunks Indexed"),
            ("~41 ms", "Retrieval Latency"),
            ("39/40", "Eval Pass Rate"),
        ]
        for col, (val, lbl) in zip([c1, c2, c3], rag_kpis):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="stat-box">
            <div style="font-size:1em;font-weight:700;color:#60a5fa;margin-bottom:12px">RAG Pipeline</div>
            <div style="line-height:2;font-size:0.88em">
            <span class="stat-label">Embedding model:</span> all-MiniLM-L6-v2<br>
            <span class="stat-label">Dimensions:</span> 384<br>
            <span class="stat-label">Vector store:</span> ChromaDB 1.5.9<br>
            <span class="stat-label">Chunk size:</span> 700 chars · 100 overlap<br>
            <span class="stat-label">Top-k retrieval:</span> 5 chunks per query<br>
            <span class="stat-label">TTFT:</span> ~10 s (Qwen2.5-1.5B)<br>
            </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="stat-box">
            <div style="font-size:1em;font-weight:700;color:#60a5fa;margin-bottom:12px">Fine-tuned LLM</div>
            <div style="line-height:2;font-size:0.88em">
            <span class="stat-label">Base model:</span> Qwen2.5-1.5B-Instruct<br>
            <span class="stat-label">Method:</span> QLoRA 4-bit NF4<br>
            <span class="stat-label">Training examples:</span> 8,343<br>
            <span class="stat-label">Train loss:</span> 2.84 → 0.49<br>
            <span class="stat-label">Export:</span> GGUF Q4_K_M (934 MB)<br>
            <span class="stat-label">Deployment:</span> Ollama (local)<br>
            </div>
            </div>
            """, unsafe_allow_html=True)

        # Live RAG stats
        st.markdown("---")
        st.markdown("**Live index stats:**")
        live = get_index_stats()
        lc = st.columns(4)
        for col, (k, v) in zip(lc, [
            ("Total chunks",  live.get("total_chunks",0)),
            ("Unique papers", live.get("unique_docs",0)),
            ("Sources",       list(live.get("sources",{}).keys())),
            ("Store",         str(live.get("store_path",""))[-30:]),
        ]):
            with col:
                st.metric(k, str(v)[:35])

    with tab_oa:
        if papers:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**OA Category Distribution**")
                oa_c = Counter(p.get("oa_category","") for p in papers)
                df_oa = pd.DataFrame([
                    {"Category": k, "Count": v,
                     "Percent": f"{100*v/len(papers):.1f}%"}
                    for k, v in oa_c.most_common()
                ])
                st.dataframe(df_oa, hide_index=True, use_container_width=True)
                st.bar_chart(df_oa.set_index("Category")["Count"])

            with c2:
                st.markdown("**Papers by Year**")
                yr_c = Counter(p.get("publication_year","") for p in papers)
                df_yr = pd.DataFrame([
                    {"Year": k, "Count": v}
                    for k, v in sorted(yr_c.items())
                ])
                st.dataframe(df_yr, hide_index=True, use_container_width=True)
                st.bar_chart(df_yr.set_index("Year")["Count"])
        else:
            st.info("Dataset not found — check FINAL_CSV path in rag/config.py")


# =============================================================================
# PAGE: INDEX MANAGEMENT
# =============================================================================
elif st.session_state.page == "index":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">⚙️ Index Management</div>
        <div class="page-sub">Build and manage the RAG vector index — indexing is idempotent, already-indexed chunks are skipped</div>
    </div>
    """, unsafe_allow_html=True)

    # Live stats
    live = get_index_stats()
    cols = st.columns(4)
    for col, (lbl, val) in zip(cols, [
        ("Chunks",   live.get("total_chunks",  0)),
        ("Papers",   live.get("unique_docs",   0)),
        ("Sources",  len(live.get("sources",  {}))),
        ("Ollama",   "🟢 Ready" if ollama_ok else "🔴 Offline"),
    ]):
        with col:
            st.metric(lbl, str(val))

    st.markdown("---")

    # Index corpus
    st.markdown("#### ▶ Index OpenAlex Corpus (2,000 papers)")
    st.caption("Source: `data/final/final_dataset.csv` · Safe to run multiple times")

    col1, col2 = st.columns([2, 1])
    with col1:
        limit = st.number_input("Papers to index (0 = all):",
                                 min_value=0, max_value=2000, value=0, step=100)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Start indexing", type="primary", use_container_width=True)

    if run_btn:
        n = int(limit) if limit > 0 else None
        bar = st.progress(0, text=f"Indexing {n or 2000} papers…")
        txt = st.empty()

        def cb(current: int, total: int) -> None:
            bar.progress(current / max(total, 1),
                         text=f"Indexing… {current}/{total}")
            txt.caption(f"Processed {current} papers")

        with st.spinner("Indexing corpus…"):
            res = index_corpus(limit=n, progress_cb=cb)

        bar.progress(1.0, text="Complete!")
        st.success(
            f"✅ {res['docs_loaded']} papers → "
            f"{res['chunks_created']} chunks created · "
            f"{res['chunks_added']} new · "
            f"{res['chunks_skipped']} skipped (already indexed)"
        )
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Reset
    st.markdown("#### ⚠️ Reset Index")
    st.warning("Deletes all indexed chunks. You will need to re-index afterwards.", icon="⚠️")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        del_btn = st.button("🗑️ Delete index", use_container_width=True)
    with col_b:
        confirm = st.checkbox("Yes, I want to delete the entire index")

    if del_btn:
        if confirm:
            from rag.vector_store import reset_collection
            reset_collection()
            st.success("Index cleared.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Check the confirmation box first.")

    st.markdown("---")
    st.markdown("#### Index Configuration")
    st.code(f"""
Embedding model : all-MiniLM-L6-v2  (384-dim, ~22 MB, fully local)
Vector store    : ChromaDB 1.5.9  (L2 distance, persistent)
Collection      : researchpilot_papers
Store path      : data/rag_vector_store/
Chunk size      : 700 characters
Chunk overlap   : 100 characters
Min chunk size  : 80 characters
Deduplication   : SHA-1 chunk_id (idempotent)
""", language=None)
