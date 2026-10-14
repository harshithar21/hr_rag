import os
from sentence_transformers import CrossEncoder
from typing import List, Dict

# Load the cross‑encoder once at import time
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = CrossEncoder(MODEL_NAME)

def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank a list of candidate chunks using a cross‑encoder.

    Args:
        query: The user question.
        candidates: A list of dicts, each containing at least a ``"text"`` field
            (and optionally ``"id"``, ``"source"``, ``"page"``).
        top_k: Number of top results to return.

    Returns:
        Candidates sorted by descending relevance score, each augmented with a
        ``"score"`` key (higher is better). Only the top ``top_k`` items are kept.
    """
    if not candidates:
        return []

    # Build (query, text) pairs for the cross‑encoder
    pairs = [(query, cand["text"]) for cand in candidates]
    scores = _cross_encoder.predict(pairs)  # returns a list of floats

    # Attach scores to candidates
    for cand, sc in zip(candidates, scores):
        cand["score"] = float(sc)

    # Sort by score descending and limit
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
