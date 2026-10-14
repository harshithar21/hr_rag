import json
import pathlib
import requests
import sqlite3
from typing import List, Dict, Tuple

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]  # rag_dw folder
DB_PATH = BASE_DIR / "data" / "search.db"
EMBED_MODEL = "nomic-embed-text"

def embed_text(text: str) -> List[float]:
    """Return a single embedding vector from Ollama for the given text."""
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"embedding": [...]} for a single vector
    return data["embedding"]

def _connect() -> sqlite3.Connection:
    # Ensure the data directory exists before opening the DB
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    # Enable sqlite‑vec extension (already loaded in ingestion script)
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    return conn

def _sanitize_fts5(q: str) -> str:
    """Escape special characters for FTS5 MATCH and wrap as a literal phrase.
    This prevents syntax errors for natural‑language queries containing punctuation
    or FTS5 operators ("*", "?", etc.)."""
    # Characters that have special meaning in FTS5 query syntax
    special_chars = "\"'`*?~^[](){}:"  # note the double‑quote is escaped
    for ch in special_chars:
        q = q.replace(ch, " ")
    # Collapse whitespace and wrap the whole query in double quotes for phrase search
    q = " ".join(q.split())
    return f'"{q}"'

def text_search(query: str, limit: int = 10) -> List[Tuple[int, float]]:
    """Return list of (rowid, txt_score) using BM25 on the FTS5 virtual table.
    The user query is sanitized to avoid FTS5 syntax errors.
    """
    conn = _connect()
    cur = conn.cursor()
    safe_query = _sanitize_fts5(query)
    cur.execute(
        "SELECT rowid, bm25(chunks_fts) AS txt_score FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
        (safe_query, limit),
    )
    results = cur.fetchall()
    conn.close()
    return results

def vector_search(query_vec: List[float], limit: int = 10) -> List[Tuple[int, float]]:
    """Return list of (rowid, distance) where distance is cosine distance.
    Smaller distance = more similar.
    """
    conn = _connect()
    cur = conn.cursor()
    # Compute cosine distance using sqlite-vec's vec_distance_cosine function.
    # The function returns a distance; we order ascending.
    cur.execute(
        "SELECT rowid, vec_distance_cosine(?, embedding) AS distance FROM chunks ORDER BY distance LIMIT ?",
        (json.dumps(query_vec), limit),
    )
    results = cur.fetchall()
    conn.close()
    return results

def _fetch_chunk_details(rowids: List[int]) -> Dict[int, Dict]:
    """Return a mapping rowid -> chunk dict (id, text, source, page, ...)."""
    if not rowids:
        return {}
    placeholders = ",".join(["?" for _ in rowids])
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        f"SELECT rowid, id, text, source, page, chunk_index, offset FROM chunks WHERE rowid IN ({placeholders})",
        tuple(rowids),
    )
    rows = cur.fetchall()
    conn.close()
    mapping = {}
    for row in rows:
        rowid, uid, txt, src, pg, idx, off = row
        mapping[rowid] = {
            "id": uid,
            "text": txt,
            "source": src,
            "page": pg,
            "chunk_index": idx,
            "offset": off,
        }
    return mapping

def hybrid_search(query: str, top_k: int = 10) -> List[Dict]:
    """Perform a 50/50 hybrid search and return a list of chunk dicts.

    The function:
    1️⃣ Embeds the query via Ollama.
    2️⃣ Retrieves text scores (BM25) and vector distances.
    3️⃣ Normalizes each score list to [0, 1].
    4️⃣ Combines them with equal weight.
    5️⃣ Returns the top_k chunks sorted by the combined score.
    """
    # 1️⃣ Embed query
    query_vec = embed_text(query)

    # 2️⃣ Retrieve raw scores
    txt_results = text_search(query, limit=top_k * 2)  # fetch extra for merging
    vec_results = vector_search(query_vec, limit=top_k * 2)

    # Convert to dicts for quick lookup
    txt_dict = {rowid: score for rowid, score in txt_results}
    vec_dict = {rowid: dist for rowid, dist in vec_results if dist is not None}

    # Build a unified set of candidate rowids
    candidate_ids = set(txt_dict) | set(vec_dict)

    # Normalization helpers
    def _norm(scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize a {rowid: score} dict to [0, 1]."""
        if not scores:
            return {}
        vals = scores.values()
        min_v = min(vals)
        max_v = max(vals)
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    # Normalize text scores (higher = better)
    txt_scores = {rid: txt_dict.get(rid, 0.0) for rid in candidate_ids}
    txt_norm = _norm(txt_scores)
    # Normalize vector similarity: we have distances, so convert to similarity first
    # similarity = 1 / (1 + distance)  (range (0,1])
    vec_sim = {rid: 1.0 / (1.0 + vec_dict.get(rid, 1e9)) for rid in candidate_ids}
    vec_norm = _norm(vec_sim)

    # Combine 0.5*txt + 0.5*vec
    combined_scores = {}
    for rid in candidate_ids:
        txt_score = txt_norm.get(rid, 0.0)
        vec_score = vec_norm.get(rid, 0.0)
        combined_scores[rid] = 0.5 * txt_score + 0.5 * vec_score

    # Sort candidates by combined score descending
    sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]

    # Fetch chunk details for the selected ids
    details = _fetch_chunk_details(sorted_ids)

    # Attach combined score
    results = []
    for rid in sorted_ids:
        chunk = details.get(rid, {})
        if chunk:
            chunk["combined_score"] = combined_scores[rid]
            results.append(chunk)
    return results
