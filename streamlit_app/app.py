"""
ResearchPilot — Streamlit Chat UI
File: streamlit_app/app.py

Run: py -3.12 -m streamlit run streamlit_app/app.py
     OR: streamlit run streamlit_app/app.py
"""

import streamlit as st
import requests
import json
import time
import csv
import random
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://127.0.0.1:11434"
MODEL        = "researchpilot"
ROOT         = Path(__file__).resolve().parent.parent
DATASET_CSV  = ROOT / "data" / "final" / "final_dataset.csv"

st.set_page_config(
    page_title  = "ResearchPilot",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background: #0f1117; color: #e0e0e0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1d27;
    border-right: 1px solid #2d2f3e;
}

/* Chat messages */
.user-msg {
    background: #1e3a5f;
    border-left: 4px solid #4a9eff;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #e8f4fd;
}
.bot-msg {
    background: #1a2438;
    border-left: 4px solid #7c4dff;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #e0e0e0;
}

/* Header */
.rp-header {
    background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%);
    border-bottom: 2px solid #4a9eff;
    padding: 16px 24px;
    margin-bottom: 20px;
    border-radius: 10px;
}
.rp-title { font-size: 2em; font-weight: 800; color: #4a9eff; margin: 0; }
.rp-sub   { font-size: 0.9em; color: #888; margin: 4px 0 0 0; }

/* Metric cards */
.metric-card {
    background: #1a1d27;
    border: 1px solid #2d2f3e;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin: 4px;
}
.metric-val { font-size: 1.6em; font-weight: 700; color: #4a9eff; }
.metric-lbl { font-size: 0.75em; color: #888; text-transform: uppercase; }

/* Paper card */
.paper-card {
    background: #1a1d27;
    border: 1px solid #2d2f3e;
    border-radius: 8px;
    padding: 14px;
    margin: 6px 0;
    border-left: 3px solid #7c4dff;
}
.paper-title { font-weight: 600; color: #e0e0e0; font-size: 0.95em; }
.paper-meta  { color: #888; font-size: 0.8em; margin-top: 4px; }

/* Chip badges */
.chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 600;
    margin: 2px;
}
.chip-open   { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
.chip-closed { background: #3a1a1a; color: #f44336; border: 1px solid #f44336; }
.chip-hybrid { background: #3a3a1a; color: #ff9800; border: 1px solid #ff9800; }

/* Input */
.stTextArea textarea {
    background: #1a1d27 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2d2f3e !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: #4a9eff !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #2979ff !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

/* Scrollable chat area */
.chat-container { max-height: 60vh; overflow-y: auto; padding-right: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_papers():
    """Load papers from final_dataset.csv."""
    if not DATASET_CSV.exists():
        return []
    papers = []
    with open(DATASET_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            papers.append(row)
    return papers

def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def model_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(MODEL in m for m in models)
    except Exception:
        return False

def ask_ollama(messages: list, stream: bool = True):
    """Call Ollama chat API. Yields tokens if stream=True."""
    payload = {
        "model":    MODEL,
        "messages": messages,
        "stream":   stream,
        "options":  {"temperature": 0.15, "seed": 42, "num_predict": 600},
    }
    try:
        if stream:
            with requests.post(f"{OLLAMA_URL}/api/chat",
                               json=payload, stream=True, timeout=120) as resp:
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        else:
            resp = requests.post(f"{OLLAMA_URL}/api/chat",
                                 json=payload, timeout=120)
            yield resp.json()["message"]["content"]
    except Exception as e:
        yield f"\n\n*Error: {e}*"

def oa_chip(status: str) -> str:
    s = status.lower()
    if s in ("gold", "diamond", "green"):
        return f'<span class="chip chip-open">{status}</span>'
    elif s == "closed":
        return f'<span class="chip chip-closed">closed</span>'
    else:
        return f'<span class="chip chip-hybrid">{status}</span>'

def search_papers(query: str, papers: list, n: int = 5) -> list:
    """Simple keyword search over titles and abstracts."""
    q = query.lower()
    scored = []
    for p in papers:
        score = 0
        text  = (p.get("title","") + " " + p.get("abstract","")).lower()
        for word in q.split():
            if len(word) > 3 and word in text:
                score += text.count(word)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:n]]

# ── Quick prompts ─────────────────────────────────────────────────────────────
QUICK_PROMPTS = [
    ("📄 Summarise a paper",
     "Summarise this paper for a postgraduate student:\n\nTitle: [paste title here]\nAbstract: [paste abstract here]"),
    ("🔍 Explain methodology",
     "What research methodology does this paper use?\n\nTitle: [paste title]\nAbstract: [paste abstract]"),
    ("❓ Research questions",
     "What research questions does this paper address?\n\nTitle: [paste title]\nAbstract: [paste abstract]"),
    ("⚠️ Limitations",
     "What are the likely limitations of a paper on [your topic]?"),
    ("📊 Explain statistics",
     "Explain what Macro-F1 score means and when to use it instead of accuracy."),
    ("🔬 Compare methods",
     "What is the difference between LoRA and full fine-tuning of a language model?"),
    ("📈 Cite count",
     "How influential is a paper with 500 citations published in 2022?"),
    ("🌐 Open access",
     "What does it mean for a paper to be 'gold open access' vs 'green open access'?"),
]

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "paper_search" not in st.session_state:
    st.session_state.paper_search = ""

papers = load_papers()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 ResearchPilot")
    st.markdown("*AI Research Assistant*")
    st.markdown("---")

    # Nav
    pages = {
        "💬 Chat": "chat",
        "📄 Analyse Paper": "upload",
        "📚 Browse Papers": "papers",
        "📊 Project Stats": "stats",
    }
    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}",
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")

    # Model status
    is_up    = ollama_available()
    is_model = model_available()
    st.markdown(f"**Ollama:** {'🟢 Running' if is_up else '🔴 Not running'}")
    st.markdown(f"**Model:** {'🟢 ' + MODEL if is_model else '🔴 Not found'}")

    if not is_up:
        st.warning("Start Ollama from the system tray.")

    st.markdown("---")

    # Quick prompts
    st.markdown("**Quick prompts**")
    for emoji_label, prompt_text in QUICK_PROMPTS:
        if st.button(emoji_label, key=f"qp_{emoji_label}"):
            st.session_state._quick_prompt = prompt_text
            st.session_state.page = "chat"
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.75em;color:#666;'>
<b>ResearchPilot v1.0</b><br>
Fine-tuned Qwen2.5-1.5B<br>
QLoRA · 8,343 examples<br>
Train loss: 2.84 → 0.49<br>
Runs 100% locally
</div>
""", unsafe_allow_html=True)

# ── CHAT PAGE ─────────────────────────────────────────────────────────────────
if st.session_state.page == "chat":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">🔬 ResearchPilot</div>
  <div class="rp-sub">Your AI research assistant — fine-tuned on 8,343 research examples · runs locally</div>
</div>
""", unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        role  = msg["role"]
        content = msg["content"]
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🔬"):
                st.markdown(content)

    # Handle quick prompt injection
    prefill = ""
    if hasattr(st.session_state, "_quick_prompt"):
        prefill = st.session_state._quick_prompt
        del st.session_state._quick_prompt

    # Chat input
    user_input = st.chat_input(
        "Ask about a paper, method, dataset, or concept...",
        key="chat_input"
    )

    # If a quick prompt was selected, show it as a pre-filled text area
    if prefill and not user_input:
        st.info(f"Quick prompt loaded. Edit below and press Enter to send.")
        user_input = st.text_area("Edit your message:", value=prefill, height=120, key="prefill_ta")
        if st.button("Send ▶", key="send_prefill"):
            pass  # user_input already set above
        else:
            user_input = None

    if user_input and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input.strip())

        # Build message history for context (last 10 turns)
        history = st.session_state.messages[-20:]

        # Stream response
        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            full_response = ""
            with st.spinner(""):
                for token in ask_ollama(history, stream=True):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        st.rerun()

    # Suggested questions (show only when chat is empty)
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("#### Suggested questions to get started:")
        cols = st.columns(2)
        suggestions = [
            "What is a transformer model and why is it important?",
            "Explain the difference between precision and recall.",
            "What is RAG and how does it improve language models?",
            "What does 'open access' mean in academic publishing?",
            "Explain what K-Means clustering is used for.",
            "What is the difference between LoRA and full fine-tuning?",
        ]
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {sug[:55]}...", key=f"sug_{i}"):
                    st.session_state.messages.append({"role": "user", "content": sug})
                    st.rerun()

# ── ANALYSE PAPER PAGE ───────────────────────────────────────────────────────
elif st.session_state.page == "upload":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">📄 Analyse a Paper</div>
  <div class="rp-sub">Upload any PDF or paste text — ResearchPilot answers questions about it</div>
</div>
""", unsafe_allow_html=True)

    # ── extraction helpers ─────────────────────────────────────────────────────
    def extract_pdf(uploaded_file) -> str:
        """Extract text from uploaded PDF using pymupdf."""
        try:
            import fitz  # pymupdf
            import io
            data  = uploaded_file.read()
            doc   = fitz.open(stream=data, filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            return "\n".join(pages).strip()
        except ImportError:
            return ""
        except Exception as e:
            return f"PDF extraction error: {e}"

    def truncate_text(text: str, max_chars: int = 3000) -> str:
        """Keep first max_chars characters to stay within context window."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[... text truncated to fit context window ...]"

    # ── state for uploaded paper ───────────────────────────────────────────────
    if "uploaded_paper_text" not in st.session_state:
        st.session_state.uploaded_paper_text  = ""
    if "uploaded_paper_name" not in st.session_state:
        st.session_state.uploaded_paper_name  = ""
    if "paper_chat_history" not in st.session_state:
        st.session_state.paper_chat_history   = []

    # ── upload area ────────────────────────────────────────────────────────────
    col_up, col_info = st.columns([2, 1])

    with col_up:
        st.markdown("#### Step 1 — Provide the paper")
        tab_pdf, tab_text = st.tabs(["📎 Upload PDF", "📝 Paste Text"])

        with tab_pdf:
            uploaded = st.file_uploader(
                "Upload a research paper PDF",
                type=["pdf"],
                help="Max ~20 MB. First 3,000 characters of text are sent to the model."
            )
            if uploaded:
                with st.spinner("Extracting text from PDF..."):
                    extracted = extract_pdf(uploaded)
                if extracted.startswith("PDF extraction error"):
                    st.error(extracted)
                    st.info("pymupdf may still be installing. Try the 'Paste Text' tab instead.")
                elif not extracted:
                    st.warning("Could not extract text. Install pymupdf or use the Paste Text tab.")
                else:
                    st.session_state.uploaded_paper_text = extracted
                    st.session_state.uploaded_paper_name = uploaded.name
                    st.session_state.paper_chat_history  = []
                    st.success(f"✅ Extracted {len(extracted):,} characters from {uploaded.name}")

        with tab_text:
            pasted = st.text_area(
                "Paste the paper title + abstract (or full text):",
                height=220,
                placeholder="Title: ...\n\nAbstract: ...\n\nIntroduction: ..."
            )
            paper_name_input = st.text_input("Paper name (optional):", placeholder="e.g. attention_is_all_you_need.txt")
            if st.button("Use this text ▶", key="use_pasted"):
                if pasted.strip():
                    st.session_state.uploaded_paper_text = pasted.strip()
                    st.session_state.uploaded_paper_name = paper_name_input or "Pasted text"
                    st.session_state.paper_chat_history  = []
                    st.success(f"✅ Loaded {len(pasted):,} characters")
                else:
                    st.warning("Please paste some text first.")

    with col_info:
        st.markdown("#### What can ResearchPilot do?")
        st.markdown("""
<div class="paper-card">
<b>With your paper, you can ask:</b><br><br>
📋 Summarise this paper<br>
🔍 What methodology is used?<br>
❓ What are the research questions?<br>
⚠️ What are the limitations?<br>
📊 Explain the results<br>
🔬 What datasets were used?<br>
💡 What are the contributions?<br>
📖 Explain [specific term] from this paper<br>
🔗 Compare this to [other method]
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="paper-card" style="border-left-color:#4caf50">
<b>✅ Works fully offline</b><br><br>
• No API calls to external services<br>
• PDF extraction: local (pymupdf)<br>
• AI inference: local (Ollama)<br>
• Your paper never leaves your PC
</div>
""", unsafe_allow_html=True)

    # ── show loaded paper ──────────────────────────────────────────────────────
    if st.session_state.uploaded_paper_text:
        st.markdown("---")
        paper_text = st.session_state.uploaded_paper_text
        paper_name = st.session_state.uploaded_paper_name

        st.markdown(f"#### Step 2 — Ask about: `{paper_name}`")

        # Preview
        with st.expander(f"Preview paper text ({len(paper_text):,} chars)", expanded=False):
            st.text(paper_text[:1500] + ("..." if len(paper_text) > 1500 else ""))

        # Quick action buttons
        st.markdown("**Quick actions:**")
        quick_cols = st.columns(4)
        quick_actions = [
            ("📋 Summarise",    "Summarise this paper clearly for a postgraduate student."),
            ("🔍 Methodology",  "What research methodology does this paper use? What kind of study is it?"),
            ("⚠️ Limitations",  "What are the limitations of this paper? What should readers be cautious about?"),
            ("💡 Contributions","What are the main contributions and findings of this paper?"),
        ]
        for col, (btn_lbl, action_prompt) in zip(quick_cols, quick_actions):
            with col:
                if st.button(btn_lbl, key=f"qa_{btn_lbl}"):
                    st.session_state._paper_action = action_prompt

        # Chat history for this paper
        for msg in st.session_state.paper_chat_history:
            role = msg["role"]
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content_display"])
            else:
                with st.chat_message("assistant", avatar="🔬"):
                    st.markdown(msg["content"])

        # Handle quick action
        prefill_paper = ""
        if hasattr(st.session_state, "_paper_action"):
            prefill_paper = st.session_state._paper_action
            del st.session_state._paper_action

        # Input
        user_q = st.chat_input("Ask a question about this paper...", key="paper_chat")

        if prefill_paper and not user_q:
            user_q = prefill_paper

        if user_q and user_q.strip():
            # Build context: paper text + question
            truncated = truncate_text(paper_text, max_chars=3000)
            system_context = (
                "The user has uploaded a research paper. Answer questions based ONLY on the content "
                "provided below. Do not fabricate information not present in the paper. If the paper "
                "does not contain enough information to answer, say so clearly.\n\n"
                f"--- PAPER CONTENT ---\n{truncated}\n--- END OF PAPER ---"
            )

            # Build message list: system + history + new question
            messages = [{"role": "system", "content": system_context}]
            for h in st.session_state.paper_chat_history[-10:]:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_q})

            # Display question (short version for history)
            display_q = user_q[:200] + ("..." if len(user_q) > 200 else "")

            with st.chat_message("user", avatar="👤"):
                st.markdown(display_q)

            # Stream answer
            with st.chat_message("assistant", avatar="🔬"):
                placeholder = st.empty()
                full_answer = ""
                with st.spinner(""):
                    for token in ask_ollama(messages, stream=True):
                        full_answer += token
                        placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)

            # Save to paper chat history
            st.session_state.paper_chat_history.append({
                "role": "user",
                "content": user_q,
                "content_display": display_q
            })
            st.session_state.paper_chat_history.append({
                "role": "assistant",
                "content": full_answer
            })
            st.rerun()

        # Clear paper button
        if st.button("🗑️ Clear paper and start over", key="clear_paper"):
            st.session_state.uploaded_paper_text = ""
            st.session_state.uploaded_paper_name = ""
            st.session_state.paper_chat_history  = []
            st.rerun()

    else:
        st.markdown("---")
        st.info("Upload a PDF or paste text above to get started.")

        # Demo example
        st.markdown("#### Try a demo — click to load example paper:")
        if st.button("📄 Load example: Attention Is All You Need"):
            demo_text = """Title: Attention Is All You Need

Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional 
neural networks that include an encoder and a decoder. The best performing models also connect the 
encoder and decoder through an attention mechanism. We propose a new simple network architecture, 
the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions 
entirely. Experiments on two machine translation tasks show these models to be superior in quality 
while being more parallelizable and requiring significantly less time to train. Our model achieves 
28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best 
results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, 
our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 
3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, 
Lukasz Kaiser, Illia Polosukhin

Year: 2017 | Venue: NeurIPS | Citations: 100,000+

Methodology: The paper introduces the Transformer architecture with multi-head self-attention 
and position-wise feed-forward networks. The model uses scaled dot-product attention. Training 
used the Adam optimizer with a custom learning rate schedule and dropout for regularization.

Results: BLEU score of 28.4 on WMT 2014 EN-DE and 41.0 on WMT 2014 EN-FR translation tasks.
Training time: 3.5 days on 8 GPUs for the large model.

Limitations: The paper focuses on machine translation. Computational cost scales quadratically 
with sequence length due to self-attention. No evaluation on tasks beyond translation in this paper."""

            st.session_state.uploaded_paper_text = demo_text
            st.session_state.uploaded_paper_name = "Attention Is All You Need (Vaswani et al., 2017)"
            st.session_state.paper_chat_history  = []
            st.rerun()


# ── BROWSE PAPERS PAGE ────────────────────────────────────────────────────────
elif st.session_state.page == "papers":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">📚 Browse Papers</div>
  <div class="rp-sub">Search and explore the 2,000 AI/ML papers in ResearchPilot's corpus</div>
</div>
""", unsafe_allow_html=True)

    if not papers:
        st.error(f"Dataset not found at {DATASET_CSV}")
    else:
        # Search
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("🔍 Search papers by title, abstract or topic:",
                                  placeholder="e.g. transformer attention machine translation")
        with col2:
            oa_filter = st.selectbox("OA Status", ["All", "fully_open", "partially_open", "closed"])

        # Filter
        filtered = papers
        if oa_filter != "All":
            filtered = [p for p in filtered if p.get("oa_category") == oa_filter]

        if query.strip():
            filtered = search_papers(query, filtered, n=20)
            st.markdown(f"**{len(filtered)} results** for *{query}*")
        else:
            # Show random sample if no search
            filtered = random.sample(filtered, min(20, len(filtered)))
            st.markdown(f"Showing 20 random papers — search to filter")

        for p in filtered:
            title   = p.get("title", "Untitled")[:120]
            year    = p.get("publication_year", "")
            cited   = p.get("cited_by_count", "0")
            oa_s    = p.get("oa_status", "")
            oa_cat  = p.get("oa_category", "")
            abstract= p.get("abstract", "")[:200]
            doi     = p.get("doi", "")

            chip_html = oa_chip(oa_s)

            st.markdown(f"""
<div class="paper-card">
  <div class="paper-title">{title}</div>
  <div class="paper-meta">
    📅 {year} &nbsp;·&nbsp; 📊 {cited} citations &nbsp;·&nbsp;
    {chip_html} &nbsp; <span style="color:#888;font-size:0.85em">{oa_cat}</span>
  </div>
  <div style="color:#aaa;font-size:0.85em;margin-top:6px">{abstract}...</div>
</div>
""", unsafe_allow_html=True)

            # Ask about this paper button
            col_a, col_b = st.columns([1, 4])
            with col_a:
                if st.button("Ask ResearchPilot 🔬", key=f"ask_{p.get('id',title)[:20]}"):
                    prompt = (f"Summarise this paper for a postgraduate student:\n\n"
                              f"Title: {p.get('title','')}\n"
                              f"Year: {year}\n"
                              f"Citations: {cited}\n"
                              f"Abstract: {p.get('abstract','')[:500]}")
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.page = "chat"
                    st.rerun()

# ── STATS PAGE ────────────────────────────────────────────────────────────────
elif st.session_state.page == "stats":

    st.markdown("""
<div class="rp-header">
  <div class="rp-title">📊 Project Statistics</div>
  <div class="rp-sub">ResearchPilot DA3 — verified results from the ML pipeline</div>
</div>
""", unsafe_allow_html=True)

    # KPI row
    kpi_cols = st.columns(6)
    kpis = [
        ("2,000",  "Papers"),
        ("27→80",  "Features"),
        ("12",     "ML Models"),
        ("57.5%",  "Best Accuracy"),
        ("0.5701", "Best F1"),
        ("0.7577", "Best AUC"),
    ]
    for col, (val, lbl) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-val">{val}</div>
  <div class="metric-lbl">{lbl}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Model Leaderboard (Test Set, n=301)")
        leaderboard = [
            ("SVM-RBF (Tuned)",    "0.5748", "0.5701", "—",     "Tuned"),
            ("LDA",                "0.5748", "0.5696", "0.7577","Baseline"),
            ("MLP",                "0.5681", "0.5657", "0.7659","Baseline"),
            ("Logistic Regression","0.5648", "0.5631", "0.7565","Baseline"),
            ("Elastic Net",        "0.5581", "0.5539", "0.7584","Baseline"),
            ("RF (Tuned)",         "0.5615", "0.5475", "—",     "Tuned"),
            ("SVM-Linear",         "0.5515", "0.5441", "0.7587","Baseline"),
            ("Gradient Boosting",  "0.5382", "0.5384", "0.7402","Baseline"),
            ("Random Forest",      "0.5482", "0.5317", "0.7572","Baseline"),
            ("AdaBoost",           "0.5150", "0.5123", "0.6963","Baseline"),
            ("Decision Tree",      "0.5083", "0.5017", "0.6924","Baseline"),
            ("Naive Bayes",        "0.4186", "0.4081", "0.6840","Baseline"),
            ("XGBoost (Tuned)",    "0.3555", "0.3537", "—",     "Tuned"),
            ("XGBoost",            "0.3455", "0.3428", "0.5243","Baseline"),
        ]
        import pandas as pd
        df = pd.DataFrame(leaderboard, columns=["Model","Accuracy","Macro-F1","ROC-AUC","Type"])
        st.dataframe(df, use_container_width=True, height=450, hide_index=True)

    with col2:
        st.markdown("#### 📈 Statistical Analysis")
        st.markdown("""
<div class="metric-card" style="text-align:left;padding:20px;">
  <div style="font-size:1em;color:#4a9eff;font-weight:700;margin-bottom:12px">Friedman Test Results</div>
  <div style="margin:6px 0"><span style="color:#888">H₀:</span> All 12 models equal performance</div>
  <div style="margin:6px 0"><span style="color:#888">H₁:</span> At least one model differs</div>
  <div style="margin:6px 0"><span style="color:#888">α:</span> 0.05</div>
  <hr style="border-color:#2d2f3e">
  <div style="margin:6px 0"><span style="color:#888">χ²(11):</span> <b style="color:#4a9eff">43.4615</b></div>
  <div style="margin:6px 0"><span style="color:#888">p-value:</span> <b style="color:#4a9eff">0.000009</b></div>
  <div style="margin:6px 0"><span style="color:#888">Decision:</span> <b style="color:#4caf50">REJECT H₀ ✅</b></div>
  <hr style="border-color:#2d2f3e">
  <div style="margin:6px 0;color:#888">Nemenyi post-hoc: <b style="color:#e0e0e0">7 significant pairs</b></div>
  <div style="font-size:0.85em;color:#666;margin-top:8px">XGBoost & Naive Bayes provably inferior (p&lt;0.05)</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 🗂️ Research Clusters (K-Means, k=3)")
        clusters = [
            ("Cluster 1", "10.0%",  "200",  "Education / GenAI / ChatGPT"),
            ("Cluster 2", "71.1%", "1,422", "General ML / Robotics"),
            ("Cluster 3", "18.9%",  "378",  "Medical Imaging / Deep Learning"),
        ]
        for name, pct, n, theme in clusters:
            st.markdown(f"""
<div class="paper-card" style="padding:10px">
  <span style="color:#4a9eff;font-weight:700">{name}</span>
  <span style="color:#888;margin-left:10px">{n} papers ({pct})</span>
  <div style="color:#aaa;font-size:0.9em;margin-top:4px">{theme}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 🤖 Fine-Tuned Model")
        st.markdown("""
<div class="metric-card" style="text-align:left;padding:16px;">
  <div style="margin:4px 0"><span style="color:#888">Base model:</span> Qwen2.5-1.5B-Instruct</div>
  <div style="margin:4px 0"><span style="color:#888">Method:</span> QLoRA 4-bit NF4</div>
  <div style="margin:4px 0"><span style="color:#888">Training examples:</span> 8,343</div>
  <div style="margin:4px 0"><span style="color:#888">Train loss:</span> 2.84 → 0.49</div>
  <div style="margin:4px 0"><span style="color:#888">Duration:</span> 9h 29min</div>
  <div style="margin:4px 0"><span style="color:#888">GGUF size:</span> 934 MB (Q4_K_M)</div>
  <div style="margin:4px 0"><span style="color:#888">Inference:</span> Ollama (local, private)</div>
</div>
""", unsafe_allow_html=True)

    # OA Distribution (using paper data if available)
    if papers:
        st.markdown("---")
        st.markdown("#### 📊 Open-Access Category Distribution")
        from collections import Counter
        oa_counts = Counter(p.get("oa_category","unknown") for p in papers)
        oa_df = pd.DataFrame([
            {"Category": k, "Count": v, "Percentage": f"{100*v/len(papers):.1f}%"}
            for k, v in oa_counts.most_common()
        ])
        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.dataframe(oa_df, hide_index=True, use_container_width=True)
        with col_d:
            st.bar_chart(oa_df.set_index("Category")["Count"])
