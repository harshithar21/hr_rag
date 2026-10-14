import json
import pathlib
import sqlite3
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure the backend package is importable
from backend.search import _sanitize_fts5, text_search, hybrid_search, DB_PATH, _connect
from backend.app.main import app

client = TestClient(app)

def test_sanitize_fts5_removes_special_chars():
    raw = "What is the main topic? (see section 2) *special*"
    sanitized = _sanitize_fts5(raw)
    # Should be wrapped in double quotes and special chars removed/replaced with spaces
    assert sanitized.startswith('"') and sanitized.endswith('"')
    # No asterisk or parentheses inside the sanitized string
    inner = sanitized.strip('"')
    assert '*' not in inner and '(' not in inner and ')' not in inner
    # Multiple spaces collapsed
    assert '  ' not in inner

def test_text_search_no_error(tmp_path):
    # Ensure DB exists (the _connect will create the directory if missing)
    conn = _connect()
    cur = conn.cursor()
    # Insert a simple row for test if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    if not cur.fetchone():
        # If the FTS table doesn't exist, skip the test (nothing to search)
        pytest.skip("FTS5 virtual table not present in the test DB")
    # Perform a search with a natural language query
    results = text_search("sample query", limit=5)
    assert isinstance(results, list)
    # Each result should be a tuple of (rowid, score)
    if results:
        assert isinstance(results[0], tuple) and len(results[0]) == 2
    conn.close()

def test_hybrid_search_returns_chunks():
    # Use a simple query; hybrid_search will call embed_text which hits Ollama.
    # We'll mock embed_text to avoid external call.
    from backend import search as search_mod
    with patch.object(search_mod, "embed_text", return_value=[0.0] * 768):
        results = hybrid_search("test query", top_k=3)
        # Should return a list (may be empty if DB empty)
        assert isinstance(results, list)
        if results:
            # Each result should be a dict with required keys
            required = {"id", "text", "source", "page", "chunk_index", "offset", "combined_score"}
            assert required.issubset(results[0].keys())

def test_rag_endpoint_success(monkeypatch):
    # Mock embed_text, text_search, vector_search, and Ollama chat request
    from backend import search as search_mod
    # Mock embed_text to avoid calling Ollama embeddings
    monkeypatch.setattr(search_mod, "embed_text", lambda txt: [0.0] * 768)
    # Mock text_search and vector_search to return empty lists (no data)
    monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [])
    monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [])
    # Mock the rerank to simply return the candidates unchanged
    from backend import rerank as rerank_mod
    monkeypatch.setattr(rerank_mod, "rerank", lambda query, candidates, top_k=5: candidates[:top_k])
    # Mock requests.post for Ollama chat
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Mocked answer"}}
    mock_resp.raise_for_status = lambda: None
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_resp)
    # Send POST request
    response = client.post("/rag", json={"query": "What is the policy?", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data and data["answer"] == "Mocked answer"
    assert isinstance(data["sources"], list)
    assert isinstance(data["latency_ms"], int)
