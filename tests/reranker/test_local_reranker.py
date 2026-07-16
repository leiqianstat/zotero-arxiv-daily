"""Tests for LocalReranker — requires sentence-transformers, marked slow."""

import pytest
import sentence_transformers

from zotero_arxiv_daily.reranker.local import LocalReranker, _load_sentence_transformer
import zotero_arxiv_daily.reranker.local as local_reranker


def test_load_sentence_transformer_retries_download_errors(monkeypatch):
    sleeps: list[int] = []

    class FakeEncoder:
        attempts = 0

        def __init__(self, *args, **kwargs):
            type(self).attempts += 1
            if type(self).attempts < 3:
                raise OSError("model download unavailable")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEncoder)
    monkeypatch.setattr(local_reranker, "sleep", sleeps.append)

    encoder = _load_sentence_transformer("example/model")

    assert isinstance(encoder, FakeEncoder)
    assert FakeEncoder.attempts == 3
    assert sleeps == [30, 60]


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
