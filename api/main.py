"""
ResearchPilot — FastAPI Backend
Run: py -3.12 -m uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.config import FINAL_CSV, OLLAMA_MODEL
from rag.pipeline import (
    rag_chat, rag_summarise, rag_compare, rag_gap,
    get_index_stats, index_corpus, index_pdf_upload, index_csv_upload,
    is_ollama_ready,
)
from rag.retriever import retrieve
from rag.generator import _call_ollama

app = FastAPI(title="ResearchPilot API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000",
                   "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    top_k: int = 5

class GapRequest(BaseModel):
    topic: str
    top_k: int = 6

class CompareRequest(BaseModel):
    doc_ids: list[str]
    titles: list[str]
    dimensions: list[str] | None = None

class SummariseRequest(BaseModel):
    doc_id: str
    title: str

class SourceRequest(BaseModel):
    query: str
    top_k: int = 5
    max_distance: float = 1.6


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(data: str) -> str:
    """Format a Server-Sent Event chunk."""
    safe = data.replace("\n", "\\n")
    return f"data: {json.dumps({'token': safe})}\n\n"

def _sse_done() -> str:
    return "data: [DONE]\n\n"

def _sse_citations(citations: list[dict]) -> str:
    return f"data: {json.dumps({'citations': citations})}\n\n"


async def _stream_tokens(token_iter, citations: list[dict]) -> AsyncGenerator[str, None]:
    for tok in token_iter:
        yield _sse(tok)
    yield _sse_citations(citations)
    yield _sse_done()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    stats = get_index_stats()
    return {
        "status":    "ok",
        "ollama":    is_ollama_ready(),
        "model":     OLLAMA_MODEL,
        "n_chunks":  stats.get("total_chunks", 0),
        "n_docs":    stats.get("unique_docs",  0),
    }


@app.post("/api/chat")
def chat_stream(req: ChatRequest):
    history = [m.model_dump() for m in req.history]
    token_iter, citations = rag_chat(req.question, history=history, top_k=req.top_k)
    return StreamingResponse(
        _stream_tokens(token_iter, citations),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/gaps")
def gaps_stream(req: GapRequest):
    token_iter, citations = rag_gap(req.topic, top_k=req.top_k)
    return StreamingResponse(
        _stream_tokens(token_iter, citations),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/compare")
def compare_stream(req: CompareRequest):
    token_iter, citations = rag_compare(req.doc_ids, req.titles, dimensions=req.dimensions)
    return StreamingResponse(
        _stream_tokens(token_iter, citations),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/summarise")
def summarise_stream(req: SummariseRequest):
    token_iter, citations = rag_summarise(req.doc_id, req.title)
    return StreamingResponse(
        _stream_tokens(token_iter, citations),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sources")
def sources(req: SourceRequest):
    hits = retrieve(req.query, top_k=req.top_k, max_distance=req.max_distance)
    return {"hits": hits, "count": len(hits)}


@app.get("/api/search")
def search(q: str = Query(...), top_k: int = 8, max_distance: float = 1.6):
    hits = retrieve(q, top_k=top_k, max_distance=max_distance)
    return {"hits": hits, "count": len(hits)}


@app.get("/api/papers")
def papers(
    q: str = Query(""),
    oa: str = Query(""),
    year: str = Query(""),
    limit: int = 20,
):
    if not FINAL_CSV.exists():
        raise HTTPException(404, "Dataset not found")
    with open(FINAL_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if oa:
        rows = [r for r in rows if r.get("oa_category") == oa]
    if year:
        rows = [r for r in rows if str(r.get("publication_year", "")) == year]
    if q.strip():
        ql = q.lower()
        def score(r):
            t = (r.get("title","") + " " + r.get("abstract","")).lower()
            return sum(t.count(w) for w in ql.split() if len(w) > 2)
        rows = [(score(r), r) for r in rows]
        rows = [r for s, r in sorted(rows, key=lambda x: -x[0]) if s > 0]
    else:
        import random
        random.shuffle(rows)

    return {"papers": rows[:limit], "total": len(rows)}


@app.get("/api/stats")
def stats():
    idx = get_index_stats()
    return {
        "index":  idx,
        "ollama": is_ollama_ready(),
        "model":  OLLAMA_MODEL,
        "corpus": {
            "total_papers": 2000,
            "features_raw": 27,
            "features_engineered": 206,
            "features_selected": 80,
            "test_set": 301,
        },
        "ml": {
            "best_model":    "SVM-RBF (Tuned)",
            "best_f1":       0.5701,
            "best_accuracy": 0.5748,
            "best_auc":      None,
            "friedman_chi2": 43.4615,
            "friedman_p":    9e-6,
            "n_models":      12,
        },
        "rag": {
            "chunks":    idx.get("total_chunks", 0),
            "docs":      idx.get("unique_docs",  0),
            "embed_model":   "all-MiniLM-L6-v2",
            "chunk_size":    700,
            "top_k":         5,
            "eval_pass":     "39/40",
        },
    }


@app.post("/api/index/build")
def index_build(limit: int = 0):
    result = index_corpus(limit=limit if limit > 0 else None)
    return result


@app.post("/api/index/reset")
def index_reset():
    from rag.vector_store import reset_collection
    reset_collection()
    return {"status": "ok", "message": "Index cleared"}


@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    data = await file.read()
    res  = index_pdf_upload(data, file.filename or "upload.pdf")
    return res


@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    data = await file.read()
    res  = index_csv_upload(data, file.filename or "upload.csv")
    return res
