"""Regression tests for Kaggle Q2 citation HTTP 400 and checkpoint recovery."""
from __future__ import annotations
import json
from collections import Counter
import pytest
# The isolated CI container does not have diskcache; Kaggle installs it.
try:
    import diskcache
except ImportError:
    import sys
    import types
    stub = types.ModuleType("diskcache")
    class Cache:
        def __init__(self, *args, **kwargs): self._values = {}
        def get(self, key): return self._values.get(key)
        def set(self, key, value, expire=None): self._values[key] = value
        def expire(self): return 0
    stub.Cache = Cache
    sys.modules["diskcache"] = stub

from mrdre_baseline.evaluation import citations


class FakeJudge:
    model_name = "ollama:test"
    system_prompt = "Respond only with {result: supported|insufficient|contradictory}."

    def __call__(self, prompt):
        return '{"result": "insufficient"}'


def test_preserves_every_evidence_character_when_splitting():
    content = ('Paragraph ' + ('a ' * 1000) + '\n\n') * 10
    for chunk_size in (500, 3500, 16000):
        chunks = citations._partition_evidence(content, chunk_size)
        assert ''.join(chunks) == content
        assert all(chunks)
        assert len(chunks) > 1


def test_400_context_error_is_split_and_piece_results_checkpointed(tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv('MRDRE_LLM_BACKEND', 'ollama')
    monkeypatch.setenv('MRDRE_EVAL_WORKERS', '1')
    monkeypatch.setenv('MRDRE_CITATION_MAX_EVIDENCE_CHARS', '16000')
    monkeypatch.setenv('MRDRE_CITATION_CLAIM_RETRIES', '2')
    monkeypatch.setattr(citations.time, 'sleep', lambda _: None)
    calls = []

    class Response:
        def __init__(self, status, data):
            self.status_code = status
            self.data = data
            self.text = json.dumps(data)
        def raise_for_status(self):
            if self.status_code == 400:
                raise requests.HTTPError('400 Bad Request', response=self)
        def json(self):
            return self.data

    def post(url, json, timeout):
        prompt = json['messages'][-1]['content']
        calls.append(prompt)
        if len(prompt) > 3300:
            return Response(400, {'error': 'request exceeds the available context size'})
        return Response(200, {'message': {'content': '{"result":"supported"}'}})
    monkeypatch.setattr(requests, 'post', post)
    source = ('alpha beta gamma ' * 400)
    claims = [{'id': 23, 'claim': 'alpha', 'url': ['https://example.com'],
               'combined_content': source}]
    folder = tmp_path / 'citation_checkpoint'
    result = citations.judge_claims(claims, FakeJudge(), progress_dir=folder)
    assert result[0]['result'] == 'supported'
    assert any(len(p) > 3300 for p in calls)
    assert any(len(p) <= 3300 for p in calls)
    assert len(list(folder.rglob('*.json'))) > 3
    before = len(calls)
    # Per-claim checkpoint makes a repeated evaluation a no-op.
    result = citations.judge_claims(claims, FakeJudge(), progress_dir=folder)
    assert result[0]['result'] == 'supported'
    assert len(calls) == before


def test_http_400_short_prompt_includes_ollama_server_error(monkeypatch):
    import requests
    monkeypatch.setenv('MRDRE_LLM_BACKEND', 'ollama')
    monkeypatch.setenv('MRDRE_CITATION_CLAIM_RETRIES', '2')
    class Response:
        status_code = 400
        text = 'denied by server'
        def json(self):
            return {'error': 'format is unsupported on this runner'}
        def raise_for_status(self):
            raise requests.HTTPError('400', response=self)
    count = Counter()
    def post(url, json, timeout):
        count['requests'] += 1
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(RuntimeError, match='format is unsupported on this runner'):
        citations._judge_local_bounded(
            {'claim': 'short claim', 'combined_content': 'short evidence'}, FakeJudge())
    # Invalid payload is not retried unchanged four times.
    assert count['requests'] == 1


def test_mixed_supported_and_contradictory_is_not_falsely_supported():
    assert citations._merge_piece_judgments(['supported', 'contradictory']) == 'insufficient'
    assert citations._merge_piece_judgments(['insufficient', 'supported']) == 'supported'
    assert citations._merge_piece_judgments(['insufficient', 'contradictory']) == 'contradictory'
