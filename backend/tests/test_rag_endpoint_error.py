import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.app.main import app

client = TestClient(app)

def test_rag_endpoint_ollama_error(monkeypatch):
    # Mock requests.post to raise a connection error
    def mock_post(*args, **kwargs):
        raise Exception("Ollama not reachable")
    monkeypatch.setattr("requests.post", mock_post)
    response = client.post("/rag", json={"query": "test", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"].startswith("Error retrieving answer")
    # sources should be empty list
    assert data["sources"] == []
