# ResearchPilot Phase 4 — RAG Setup Guide (Windows)

This guide covers everything you need to get the RAG pipeline running on a
**native Windows** machine. No WSL, no Linux, no cloud service required.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12.x | `pythoncore-3.12-64` — used by LAUNCH_CHATBOT.bat |
| Ollama | latest | Running locally on port 11434 |
| `researchpilot` model | Q4_K_M GGUF | Must be imported via `ollama/Modelfile` |
| NVIDIA GPU (optional) | RTX 3050+ | CPU inference works but is slower |
| RAM | ≥ 8 GB | 16 GB recommended |
| Disk space | ≥ 2 GB | ~1 GB for model, ~200 MB for vector store |

---

## 1. Verify Ollama and model

Open PowerShell:

```powershell
# Check Ollama is running
Invoke-RestMethod http://127.0.0.1:11434/api/tags

# Should list "researchpilot:latest" in the models array.
# If not, import it:
ollama create researchpilot -f ollama/Modelfile
```

---

## 2. Install Python dependencies

All required packages should already be present. Verify with:

```powershell
py -3.12 -m pip list | Select-String "chromadb|sentence-transformers|streamlit|pymupdf|torch"
```

Expected versions:

| Package | Required version |
|---------|-----------------|
| chromadb | 1.5.9 |
| sentence-transformers | 2.7.0 |
| transformers | 4.40.2 |
| torch | 2.3.1+cu121 |
| streamlit | ≥ 1.40 |
| pymupdf | ≥ 1.20 |

If any are missing:

```powershell
py -3.12 -m pip install chromadb==1.5.9 "sentence-transformers==2.7.0" "transformers==4.40.2"
py -3.12 -m pip install streamlit pymupdf
```

> **Note:** Do NOT upgrade `sentence-transformers` to 6.x — it requires torch ≥ 2.5
> which conflicts with the current CUDA 12.1 torch installation.

---

## 3. Build the vector index

**First-time setup only.** The index persists across restarts at
`data/rag_vector_store/`.

### Option A — via the Streamlit UI (recommended)

1. Launch the app (see step 4).
2. Click **⚙️ Index Management** in the sidebar.
3. Leave "Limit" at 0 (all 2,000 papers).
4. Click **▶ Index corpus**.
5. Wait ~3–5 minutes for the progress bar to complete.

### Option B — via PowerShell

```powershell
cd E:\Tuned_Research
py -3.12 rag/_index_full_corpus.py
```

Expected output:
```
Indexing full corpus (2,000 papers)...
  [####################] 2000/2000 (100%)

Result:
  docs_loaded   : 2000
  chunks_created: ~5400
  chunks_added  : ~5400
  elapsed       : ~180s

Final index state:
  total_chunks : ~5400
  unique_docs  : 2000
```

> Re-running is safe — already-indexed chunks are detected by `chunk_id` and skipped.

---

## 4. Launch the application

### Option A — double-click

```
LAUNCH_CHATBOT.bat
```

### Option B — PowerShell

```powershell
cd E:\Tuned_Research
py -3.12 -m streamlit run streamlit_app/app.py
```

The browser opens automatically at **http://localhost:8501**.

---

## 5. First use

1. The sidebar shows **🟢 researchpilot** (Ollama) and **🟢 2000 papers · ~5400 chunks** (index).
2. Click **🔍 RAG Research Chat** and ask a question.
3. Answers arrive with cited sources in the **📚 Sources (N)** expander below each response.

---

## 6. Uploading your own papers

### PDF
1. Go to **📄 Analyse / Summarise**.
2. Upload a PDF via the **📎 Upload PDF** tab.
3. Click **➕ Add to RAG index** to make it searchable.

### CSV
1. Go to **📄 Analyse / Summarise → 📊 Upload CSV**.
2. Your CSV must have at least one of these column names (case-insensitive):
   - Title: `title`, `paper_title`, `name`
   - Abstract: `abstract`, `summary`, `description`, `text`, `content`
3. Click **Index this CSV ▶**.

---

## 7. Running the evaluation suite

```powershell
cd E:\Tuned_Research
py -3.12 rag_evaluation.py
```

Outputs are written to `evaluation/`:
- `rag_eval_results.csv`
- `rag_eval_results.json`
- `rag_eval_report.md`

---

## 8. Troubleshooting

### "Index is empty"
Run the indexing step (section 3). The vector store directory may have been deleted.

### "Ollama not running" (🔴 in sidebar)
Start Ollama from the system tray or run `ollama serve` in a terminal.

### `sentence_transformers` import error
```powershell
py -3.12 -m pip uninstall sentence-transformers transformers -y
py -3.12 -m pip install "sentence-transformers==2.7.0" "transformers==4.40.2"
```

### `ChromaDB` embedding error ("no attribute embed_query")
This happens if the old ChromaDB 0.4.x is being used instead of 1.5.9.
Check: `py -3.12 -c "import chromadb; print(chromadb.__version__)"` — should print `1.5.9`.

### Streamlit opens on wrong port
Edit `LAUNCH_CHATBOT.bat` and change `--server.port 8501` to any free port.

### Slow first response
The embedding model (`all-MiniLM-L6-v2`) loads on first use (~2 s).
The Ollama model generates at approximately 15–30 tokens/second on an RTX 3050.

---

## 9. Directory layout (RAG-specific)

```
Tuned_Research/
├── rag/
│   ├── config.py          # Central settings (paths, model names, thresholds)
│   ├── document_loader.py # CSV / PDF / CSV-upload / SQLite loaders
│   ├── chunker.py         # Overlapping text chunking
│   ├── embeddings.py      # SentenceTransformer singleton + ChromaDB EF
│   ├── vector_store.py    # ChromaDB persistent store
│   ├── indexer.py         # Bulk + incremental indexing
│   ├── retriever.py       # Semantic search with dedup
│   ├── generator.py       # Ollama prompt construction + streaming
│   ├── citations.py       # Citation formatting from metadata
│   └── pipeline.py        # End-to-end facade used by Streamlit
├── data/
│   └── rag_vector_store/  # ChromaDB files (auto-created, gitignored)
├── evaluation/
│   ├── rag_eval_results.csv
│   ├── rag_eval_results.json
│   └── rag_eval_report.md
├── rag_evaluation.py       # Automated evaluation script
├── streamlit_app/
│   └── app.py             # Streamlit UI (Phase 4 RAG edition)
└── LAUNCH_CHATBOT.bat     # One-click launcher
```

---

## 10. Configuration tuning

Edit `rag/config.py` to adjust:

| Setting | Default | Effect |
|---------|---------|--------|
| `CHUNK_SIZE` | 700 chars | Larger = more context per chunk, fewer chunks |
| `CHUNK_OVERLAP` | 100 chars | Higher = better sentence continuity across chunks |
| `DEFAULT_TOP_K` | 5 | More chunks = richer context, slower generation |
| `MAX_DISTANCE` | 1.6 | Lower = stricter relevance filter |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Swap for a larger model if quality insufficient |
| `OLLAMA_MODEL` | `researchpilot` | Change to use a different local model |
| `GENERATION_OPTS.temperature` | 0.15 | Higher = more creative, less grounded answers |

After changing chunk settings, reset and re-index the collection via
**⚙️ Index Management → Delete entire index**, then re-run indexing.
