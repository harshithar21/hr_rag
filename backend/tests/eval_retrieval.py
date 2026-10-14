"""Evaluation script for RAG retrieval using Recall@5 and MRR.

It creates a temporary SQLite DB with a tiny set of known chunks,
executes `hybrid_search` (pre‑rerank) and then `rerank` (post‑rerank),
and reports Recall@5 and Mean Reciprocal Rank (MRR) for a few queries.

No changes are made to the production pipeline – the script works on
its own temporary DB.
"""
import os
import json
import pathlib
import sqlite3
import tempfile
from typing import List, Dict, Tuple

# Import the functions we want to evaluate
from backend.search import hybrid_search, _connect, _sanitize_fts5, embed_text
from backend.rerank import rerank

# ---------------------------------------------------------------------------
# Helper utilities for metrics
# ---------------------------------------------------------------------------

def recall_at_k(relevant_ids: set, retrieved_ids: List[int], k: int = 5) -> float:
    """Recall@k – 1.0 if any relevant id appears in the top‑k list, else 0.0."""
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0

def reciprocal_rank(relevant_ids: set, retrieved_ids: List[int]) -> float:
    """Reciprocal Rank – 1 / rank of the first relevant result, 0 if none present."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0

# ---------------------------------------------------------------------------
# Create a temporary test DB with a minimal schema and a few chunks
# ---------------------------------------------------------------------------

def create_test_db() -> pathlib.Path:
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="rag_eval_"))
    db_path = tmp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    # Enable extension loading and load sqlite‑vec (the same extension used by the app)
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    cur = conn.cursor()
    # Main table – note that sqlite‑vec expects a column named 'embedding' of type BLOB
    cur.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            text TEXT,
            source TEXT,
            page INTEGER,
            chunk_index INTEGER,
            offset INTEGER,
            embedding BLOB
        )
        """
    )
    # FTS5 virtual table for keyword search (mirrors 'chunks')
    cur.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            id UNINDEXED,
            source UNINDEXED,
            page UNINDEXED,
            chunk_index UNINDEXED,
            offset UNINDEXED,
            content='chunks',
            content_rowid='rowid'
        )
        """
    )

    # Insert a few deterministic chunks. We will embed each chunk using the same function
    chunks = [
        {
            "id": "c1",
            "text": "Machine learning is a field of artificial intelligence.",
            "source": "test_doc.pdf",
            "page": 1,
            "chunk_index": 0,
            "offset": 0,
        },
        {
            "id": "c2",
            "text": "Deep learning is a subset of machine learning focusing on neural networks.",
            "source": "test_doc.pdf",
            "page": 2,
            "chunk_index": 1,
            "offset": 0,
        },
        {
            "id": "c3",
            "text": "Random unrelated text about cooking recipes.",
            "source": "test_doc.pdf",
            "page": 3,
            "chunk_index": 2,
            "offset": 0,
        },
        {
            "id": "c4",
            "text": "Another paragraph about machine learning applications in healthcare.",
            "source": "test_doc.pdf",
            "page": 4,
            "chunk_index": 3,
            "offset": 0,
        },
        {
            "id": "c5",
            "text": "Irrelevant content about sports and football.",
            "source": "test_doc.pdf",
            "page": 5,
            "chunk_index": 4,
            "offset": 0,
        },
    ]

    # Compute embeddings for each chunk using the same embed_text function we use in production
    for ch in chunks:
        vec = embed_text(ch["text"])  # real embedding via Ollama
        # sqlite‑vec expects the vector to be stored as JSON text
        embedding_blob = json.dumps(vec)
        cur.execute(
            "INSERT INTO chunks (id, text, source, page, chunk_index, offset, embedding) VALUES (?,?,?,?,?,?,?)",
            (ch["id"], ch["text"], ch["source"], ch["page"], ch["chunk_index"], ch["offset"], embedding_blob),
        )
        # Populate the FTS5 table – it will automatically pull the rowid
        cur.execute(
            "INSERT INTO chunks_fts (rowid, text, id, source, page, chunk_index, offset) VALUES (last_insert_rowid(), ?, ?, ?, ?, ?, ?)",
            (ch["text"], ch["id"], ch["source"], ch["page"], ch["chunk_index"], ch["offset"]),
        )

    conn.commit()
    conn.close()
    return db_path

# ---------------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------------

def evaluate():
    db_path = create_test_db()
    # Patch the global DB_PATH used by backend.search to point at our test DB
    from backend import search as search_mod
    search_mod.DB_PATH = db_path

    queries = [
        {
            "query": "What is machine learning?",
            "relevant": {"c1", "c2", "c4"},  # three chunks are relevant
        },
        {
            "query": "Explain deep learning.",
            "relevant": {"c2"},
        },
        {
            "query": "Cooking recipes for pasta",
            "relevant": {"c3"},
        },
    ]

    # Accumulators
    pre_recall_sum = 0.0
    pre_mrr_sum = 0.0
    post_recall_sum = 0.0
    post_mrr_sum = 0.0
    n = len(queries)

    for q in queries:
        # ---------- Pre‑rerank (raw hybrid) ----------
        raw_candidates = hybrid_search(q["query"], top_k=10)
        raw_ids = [c["id"] for c in raw_candidates]
        pre_recall_sum += recall_at_k(q["relevant"], raw_ids, k=5)
        pre_mrr_sum += reciprocal_rank(q["relevant"], raw_ids)

        # ---------- Post‑rerank ----------
        reranked = rerank(q["query"], raw_candidates, top_k=10)
        reranked_ids = [c["id"] for c in reranked]
        post_recall_sum += recall_at_k(q["relevant"], reranked_ids, k=5)
        post_mrr_sum += reciprocal_rank(q["relevant"], reranked_ids)

    print("=== Retrieval Evaluation (Recall@5 / MRR) ===")
    print(f"Queries evaluated: {n}")
    print("Before reranking:")
    print(f"  Recall@5 = {pre_recall_sum / n:.3f}")
    print(f"  MRR       = {pre_mrr_sum / n:.3f}")
    print("After reranking:")
    print(f"  Recall@5 = {post_recall_sum / n:.3f}")
    print(f"  MRR       = {post_mrr_sum / n:.3f}")

if __name__ == "__main__":
    evaluate()
