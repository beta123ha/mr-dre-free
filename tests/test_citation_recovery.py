"""Regression tests for the last observed Kaggle failure (claim 85)."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pytest
# Only the isolated test container lacks this declared runtime dependency.
# Kaggle installs real diskcache through pyproject.toml.
try:
    import diskcache
except ImportError:
    import sys
    import types
    stub = types.ModuleType("diskcache")
    class Cache:
        def __init__(self, *args, **kwargs):
            self._values = {}
        def get(self, key):
            return self._values.get(key)
        def set(self, key, value, expire=None):
            self._values[key] = value
        def expire(self):
            return 0
    stub.Cache = Cache
    sys.modules["diskcache"] = stub
from mrdre_baseline.evaluation import citations
from mrdre_baseline.runtime.runner import order_questions, snapshot_zip


class FakeJudge:
    model_name = "ollama:fake"
    system_prompt = "Judge only according to the supplied evidence."

    def __call__(self, prompt):
        return '{"result": "supported"}'


def _claims(n=4):
    return [{"id": i, "claim": f"claim {i}",
             "url": [f"https://example.com/{i}"],
             "combined_content": f"source about claim {i}"} for i in range(n)]


def test_just_one_first_original_question_never_shuffles():
    items = [{"id": x} for x in (1, 5, 3, 7)]
    assert order_questions(items, 1, None, first_original=True) == [{"id": 1}]
    assert order_questions(items, 99, None, first_original=True) == [{"id": 1}]
    with pytest.raises(ValueError):
        order_questions(items, 1, "5", first_original=True)


def test_citation_one_bad_claim_keeps_completed_results(tmp_path, monkeypatch):
    monkeypatch.setenv("MRDRE_LLM_BACKEND", "ollama")
    monkeypatch.setenv("MRDRE_EVAL_WORKERS", "2")
    calls = Counter()
    fail_once = {85}

    def fake_local(prompt, judge):
        i = int(prompt.split("<claim>\nclaim ")[1].split("\n")[0])
        calls[i] += 1
        if i in fail_once:
            fail_once.remove(i)
            raise RuntimeError("simulated transient model output for claim 85")
        return {"result": "supported"}

    monkeypatch.setattr(citations, "_judge_local_ollama", fake_local)
    folder = tmp_path / "output" / ".citation_progress" / "test"
    with pytest.raises(RuntimeError, match="claim 85.*simulated transient"):
        citations.judge_claims(_claims(112), FakeJudge(), progress_dir=folder)
    assert len(list((folder / "judgments").glob("*.json"))) == 111
    before = calls.copy()
    completed = citations.judge_claims(_claims(112), FakeJudge(), progress_dir=folder)
    assert len(completed) == 112 and all(x["result"] == "supported" for x in completed)
    assert calls[85] == before[85] + 1
    assert all(calls[i] == before[i] for i in range(112) if i != 85)
    archive = snapshot_zip(tmp_path / "output")
    from zipfile import ZipFile
    with ZipFile(archive) as z:
        assert any(".citation_progress" in x for x in z.namelist())


def test_ollama_retries_invalid_json_without_fabricating_score(monkeypatch):
    monkeypatch.setenv("MRDRE_CITATION_CLAIM_RETRIES", "3")
    monkeypatch.setattr(citations.time, "sleep", lambda _: None)
    posted = []

    class Response:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            pass
        def json(self):
            return {"message": {"content": self.content}}

    def fake_post(url, json, timeout):
        posted.append((url, json, timeout))
        return Response("not json" if len(posted) == 1 else '{"result": "insufficient"}')

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    result = citations._judge_local_ollama("prompt", FakeJudge())
    assert result == {"result": "insufficient"}
    assert len(posted) == 2
    assert posted[0][1]["format"]["properties"]["result"]["enum"] == [
        "supported", "insufficient", "contradictory"]


def test_ollama_exhausted_error_has_actual_cause(monkeypatch):
    monkeypatch.setenv("MRDRE_CITATION_CLAIM_RETRIES", "2")
    monkeypatch.setattr(citations.time, "sleep", lambda _: None)
    import requests

    def broken(url, json, timeout):
        raise requests.exceptions.ConnectionError("mock connection refused")

    monkeypatch.setattr(requests, "post", broken)
    with pytest.raises(RuntimeError, match="ConnectionError.*connection refused"):
        citations._judge_local_ollama("prompt", FakeJudge())


def test_report_checkpoint_skips_extraction_crawl_and_summarization(tmp_path, monkeypatch):
    monkeypatch.setenv("MRDRE_LLM_BACKEND", "ollama")
    monkeypatch.setenv("MRDRE_EVAL_WORKERS", "1")
    calls = Counter()

    def extract(*args, **kwargs):
        calls["extract"] += 1
        return [{"claim": "fact", "url": ["https://example.com"]}]

    def crawl(*args):
        calls["crawl"] += 1
        return {"https://example.com": "fact"}

    def summarize(content, claims, model):
        calls["summarize"] += 1
        return content

    def local(prompt, model):
        calls["judge"] += 1
        return {"result": "supported"}

    monkeypatch.setattr(citations, "extract_claims", extract)
    monkeypatch.setattr(citations, "crawl_urls", crawl)
    monkeypatch.setattr(citations, "summarize_url_content", summarize)
    monkeypatch.setattr(citations, "_judge_local_ollama", local)
    report = {"id": 1, "full_response": {"report": "Fact [a](https://example.com)"}}
    args = dict(report_obj=report, claim_extraction_model=FakeJudge(),
                supported_judge_model=FakeJudge(), webpage_summarization_model=FakeJudge(),
                progress_dir=tmp_path / "output" / ".citation_progress" / "q1")
    first = citations.evaluate_citations(**args)
    second = citations.evaluate_citations(**args)
    assert first == second
    assert first["faithfulness_score"] == 1.0
    assert calls == Counter(extract=1, crawl=1, summarize=1, judge=1)


def test_checkpoint_key_changes_with_model_identity(monkeypatch):
    monkeypatch.setenv("MRDRE_MODEL_FINGERPRINT", "old")
    old = citations._progress_key("report", FakeJudge())
    monkeypatch.setenv("MRDRE_MODEL_FINGERPRINT", "new")
    new = citations._progress_key("report", FakeJudge())
    assert old != new
