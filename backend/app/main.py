import time
import json
import pathlib
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from backend.rerank import rerank
from backend.search import hybrid_search

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

class RAGRequest(BaseModel):
    query: str
    top_k: int = 10

class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: int

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(req: RAGRequest):
    start = time.time()
    try:
        # 1️⃣ Hybrid retrieval (fetch more candidates for reranking)
        candidates = hybrid_search(req.query, top_k=req.top_k * 2)
        # 2️⃣ Rerank and keep top_k
        reranked = rerank(req.query, candidates, top_k=req.top_k)
        # 3️⃣ Build context from top chunks
        context = "\n\n---\n\n".join([c["text"] for c in reranked])
        prompt = (
            "Answer the following question using ONLY the provided context.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.query}"
        )
        # 4️⃣ Call Ollama chat model
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:0.5b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "")
    except Exception as e:
        # Fallback for any failure (embedding, search, chat, etc.)
        answer = f"Error retrieving answer: {e}"
        reranked = []
    # 5️⃣ Prepare sources list
    sources = [
        {"source": c.get("source"), "page": c.get("page"), "chunk_index": c.get("chunk_index")}
        for c in reranked
    ]
    latency = int((time.time() - start) * 1000)
    return RAGResponse(answer=answer, sources=sources, latency_ms=latency)

