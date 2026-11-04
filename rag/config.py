"""
rag/config.py — Central configuration for the ResearchPilot RAG pipeline.

All paths, model names, and tuneable parameters live here.
Import this module everywhere instead of hardcoding values.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # e.g. E:\Tuned_Research

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR          = ROOT / "data"
FINAL_CSV         = DATA_DIR / "final" / "final_dataset.csv"
RAW_PAPERS_DIR    = DATA_DIR / "raw_papers"

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_STORE_DIR  = DATA_DIR / "rag_vector_store"
CHROMA_COLLECTION = "researchpilot_papers"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"   # 384-dim, ~22 MB, fully local
EMBED_BATCH_SIZE  = 64                    # chunks per encoding batch

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 700    # approximate characters (not tokens) per chunk
CHUNK_OVERLAP     = 100    # character overlap between consecutive chunks
MIN_CHUNK_CHARS   = 80     # discard chunks shorter than this

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K     = 5      # number of chunks returned per query
MIN_SIMILARITY    = 0.0    # cosine similarity floor (0 = no filter; Chroma uses distance)
MAX_DISTANCE      = 1.6    # L2 distance ceiling — chunks beyond this are too dissimilar

# ── Generation (Ollama) ───────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",    "researchpilot")
OLLAMA_TIMEOUT    = 180    # seconds
GENERATION_OPTS: dict = {
    "temperature": 0.15,
    "seed":        42,
    "num_predict": 800,
}

# ── Conversation memory ────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 6   # number of prior (user, assistant) pairs kept in context

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO")

# ── Derived convenience ───────────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create any required directories that do not yet exist."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
