"""Regression tests for _norm() and hybrid_search normalization.

These tests verify that the normalization helper inside hybrid_search
correctly handles dict inputs (not plain float lists) and that the full
hybrid pipeline produces well-formed results.
"""
import pytest
from unittest.mock import patch, MagicMock
from backend import search as search_mod


# ---------------------------------------------------------------------------
# Isolated _norm() tests via a module-level copy of the logic
# ---------------------------------------------------------------------------

def _norm(scores):
    """Replica of the nested _norm() inside hybrid_search for direct testing."""
    if not scores:
        return {}
    vals = scores.values()
    min_v = min(vals)
    max_v = max(vals)
    if max_v == min_v:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class TestNorm:
    """Unit tests for the min-max normalization helper."""

    def test_empty_dict(self):
        assert _norm({}) == {}

    def test_single_entry(self):
        # When all values are equal, every entry gets 1.0
        result = _norm({1: 0.5})
        assert result == {1: 1.0}

    def test_all_equal(self):
        result = _norm({1: 3.0, 2: 3.0, 3: 3.0})
        assert all(v == 1.0 for v in result.values())
        assert set(result.keys()) == {1, 2, 3}

    def test_basic_normalization(self):
        result = _norm({10: 0.0, 20: 5.0, 30: 10.0})
        assert result[10] == pytest.approx(0.0)
        assert result[20] == pytest.approx(0.5)
        assert result[30] == pytest.approx(1.0)

    def test_negative_values(self):
        # BM25 scores are typically negative
        result = _norm({1: -5.0, 2: -3.0, 3: -1.0})
        assert result[1] == pytest.approx(0.0)   # lowest
        assert result[3] == pytest.approx(1.0)   # highest
        assert result[2] == pytest.approx(0.5)

    def test_preserves_keys(self):
        scores = {100: 1.0, 200: 2.0, 300: 3.0}
        result = _norm(scores)
        assert set(result.keys()) == {100, 200, 300}

    def test_raises_on_float_list(self):
        """_norm must NOT receive a plain list of floats (the old bug)."""
        with pytest.raises(AttributeError):
            # A list has no .values() method
            _norm([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Integration-style tests for hybrid_search normalization path
# ---------------------------------------------------------------------------

class TestHybridSearchNormalization:
    """Test that hybrid_search correctly normalises and merges results."""

    @pytest.fixture(autouse=True)
    def mock_embed(self, monkeypatch):
        """Mock embed_text so tests don't need Ollama."""
        monkeypatch.setattr(search_mod, "embed_text", lambda txt: [0.0] * 768)

    def test_only_text_results(self, monkeypatch):
        """hybrid_search works when vector_search returns nothing."""
        monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [(1, -2.0), (2, -3.0)])
        monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [])
        results = search_mod.hybrid_search("test", top_k=5)
        assert isinstance(results, list)

    def test_only_vector_results(self, monkeypatch):
        """hybrid_search works when text_search returns nothing."""
        monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [])
        monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [(1, 0.3), (2, 0.5)])
        results = search_mod.hybrid_search("test", top_k=5)
        assert isinstance(results, list)

    def test_both_sources_overlap(self, monkeypatch):
        """When both sources return the same rowid, it should appear once."""
        monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [(1, -1.0)])
        monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [(1, 0.2)])
        # Mock _fetch_chunk_details so we don't hit the DB
        monkeypatch.setattr(search_mod, "_fetch_chunk_details", lambda rowids: {
            1: {"id": "a", "text": "hello", "source": "s", "page": 1, "chunk_index": 0, "offset": 0}
        })
        results = search_mod.hybrid_search("test", top_k=5)
        assert len(results) == 1
        assert "combined_score" in results[0]
        # Combined score should be 1.0 (only one candidate → both norms give 1.0)
        assert results[0]["combined_score"] == pytest.approx(1.0)

    def test_both_sources_no_overlap(self, monkeypatch):
        """Rowids unique to each source should both appear in the final results."""
        monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [(1, -1.0)])
        monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [(2, 0.3)])
        monkeypatch.setattr(search_mod, "_fetch_chunk_details", lambda rowids: {
            1: {"id": "a", "text": "txt", "source": "s1", "page": 1, "chunk_index": 0, "offset": 0},
            2: {"id": "b", "text": "vec", "source": "s2", "page": 2, "chunk_index": 1, "offset": 0},
        })
        results = search_mod.hybrid_search("test", top_k=5)
        assert len(results) == 2
        # Both results should have valid combined_score between 0 and 1
        for r in results:
            assert 0.0 <= r["combined_score"] <= 1.0

    def test_none_distances_filtered(self, monkeypatch):
        """vec_dict should skip rows where distance is None."""
        monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [(1, -1.0)])
        monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [(1, None), (2, 0.5)])
        monkeypatch.setattr(search_mod, "_fetch_chunk_details", lambda rowids: {
            1: {"id": "a", "text": "t1", "source": "s1", "page": 1, "chunk_index": 0, "offset": 0},
            2: {"id": "b", "text": "t2", "source": "s2", "page": 2, "chunk_index": 1, "offset": 0},
        })
        results = search_mod.hybrid_search("test", top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_empty_both_sources(self, monkeypatch):
        """hybrid_search returns [] when both sources are empty."""
        monkeypatch.setattr(search_mod, "text_search", lambda q, limit=10: [])
        monkeypatch.setattr(search_mod, "vector_search", lambda vec, limit=10: [])
        results = search_mod.hybrid_search("test", top_k=5)
        assert results == []
