"""
rag/embeddings.py — Local embedding model with a module-level singleton.

Uses sentence-transformers/all-MiniLM-L6-v2:
  - 384-dimensional dense vectors
  - ~22 MB on disk, loads in < 2 s
  - Fully offline after first download
  - Compatible with sentence-transformers 2.7.0 + torch 2.3.1

The model is loaded once per Python process and reused.
Streamlit's @st.cache_resource is used in the UI layer; this module
provides the raw singleton for use outside Streamlit.

ChromaDB 1.5.9 requires a proper EmbeddingFunction[Documents] subclass with:
  name(), __call__(), get_config(), build_from_config()
"""

from __future__ import annotations

import logging
from typing import List

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from rag.config import EMBED_MODEL_NAME, EMBED_BATCH_SIZE

log = logging.getLogger(__name__)

# Module-level singleton — None until first call to get_model()
_model = None


def get_model():
    """
    Return the cached SentenceTransformer model instance.
    Loads from disk/HuggingFace cache on first call.
    """
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Embedding model loaded.")
    return _model


def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
    """
    Embed a list of strings.  Returns a list of float lists (one per text).
    Suitable for direct use with ChromaDB's add/query calls.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB 1.5.9 compatible EmbeddingFunction
# ─────────────────────────────────────────────────────────────────────────────

class ResearchPilotEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Wraps the local SentenceTransformer model as a ChromaDB EmbeddingFunction.

    ChromaDB 1.5.9 requires the full Protocol:
      - __call__(self, input: Documents) -> Embeddings
      - name(self) -> str
      - get_config(self) -> dict
      - build_from_config(cls, config) -> instance
    """

    def name(self) -> str:
        return EMBED_MODEL_NAME

    def __call__(self, input: Documents) -> Embeddings:
        return embed_texts(list(input))

    def get_config(self) -> dict:
        return {"model_name": EMBED_MODEL_NAME}

    @classmethod
    def build_from_config(cls, config: dict) -> "ResearchPilotEmbeddingFunction":
        return cls()
