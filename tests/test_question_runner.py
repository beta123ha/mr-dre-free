"""End-to-end DAG/checkpoint tests that require no GPU or external service."""
from __future__ import annotations

import json
from pathlib import Path
import time

import pytest
from mrdre_baseline.core.io import append_jsonl
from mrdre_baseline.runtime.runner import (QuestionJob, ResourcePolicy, task_plan,
    order_questions, snapshot_zip, ensure_manifest, completed_entry)
from mrdre_baseline.runtime.summary import summarize


def test_one_question_plan_exact_four_reports_and_all_six_axes():
    tasks = task_plan(("content", "format", "reflect"), "all")
    names = {x.name for x in tasks}
    assert len(names) == 19
    assert {x.name for x in tasks if x.name.startswith("gen")} == {
        "gen1", "gen2_content", "gen2_format", "gen2_reflect"}
    assert sum(x.mode == "checklist" for x in tasks) == 4
    assert sum(x.mode == "citation" for x in tasks) == 4
    assert sum(x.mode == "rubric" for x in tasks) == 4
    assert "formatinc" in names and "contentfb" in names and "formatfb" in names
    assert all(set(x.deps) <= names for x in tasks)


def test_stable_sampling_and_exact_id():
    data = [{"id": x} for x in range(65)]
    assert order_questions(data, 10, None) == order_questions(data, 10, None)
    assert len(order_questions(data, -1, "37")) == 1
    assert order_questions(data, -1, "37")[0]["id"] == 37
    with pytest.raises(ValueError):
        order_questions(data, -1, "missing")


def test_runner_completed_and_resumes_per_task(tmp_path, monkeypatch):
    out = tmp_path / "output"
    data = tmp_path / "ResearcherBench.jsonl"
    data.write_text('{"id": 7, "question":"q", "checklist":[]}', encoding="utf-8")
    policy = ResourcePolicy(2, 1, 1, 1)
    runtime = {"research_ollama": "http://localhost:11434", "judge_ollama": "http://localhost:11435"}
    job = QuestionJob("7", "ResearcherBench", "odr-local", out, data, {}, runtime,
                      policy, time.monotonic() + 60)
    executed = []

    def fake_call(task):
        assert all(job.exists(next(x for x in task_plan(("content", "format", "reflect"), "all") if x.name == dep))
                   for dep in task.deps)
        executed.append(task.name)
        path, field = job.path(task)
        if field == "report":
            row = {"id": 7, "full_response": {"report": "real mocked report"},
                   "signal": {"feedback_text": "change", "targeted_id": [1]}, "question": "q"}
        elif field == "feedback":
            row = {"id": 7, "feedback": "change"}
        else:
            row = {"id": 7, field: 0.8}
        append_jsonl(row, str(path))

    monkeypatch.setattr(job, "run_one", fake_call)
    state = job.execute(("content", "format", "reflect"), "all")
    assert state["status"] == "completed"
    assert len(executed) == 19
    assert json.loads((out / "run_state" / "7.json").read_text())["status"] == "completed"
    executed.clear()
    job.execute(("content", "format", "reflect"), "all")
    assert executed == []
    archive = snapshot_zip(out)
    assert archive.is_file()


def test_partial_run_does_not_appear_complete(tmp_path, monkeypatch):
    out = tmp_path / "output"
    data = tmp_path / "data.jsonl"
    data.write_text("{}", encoding="utf-8")
    job = QuestionJob("8", "ResearcherBench", "odr-local", out, data, {}, {},
                      ResourcePolicy(1, 1, 0, 1), time.monotonic() + 30)

    def only_first(task):
        if task.name == "gen1":
            path, _ = job.path(task)
            append_jsonl({"id": 8, "full_response": {"report": "R1"}}, str(path))
        else:
            raise RuntimeError("deliberate failure")

    monkeypatch.setattr(job, "run_one", only_first)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        job.execute(("content",), "all")
    assert json.loads((out / "run_state" / "8.json").read_text())["status"] == "interrupted"
    rows = summarize(out, "content")
    assert rows[0]["completed_questions"] == 0
    assert not rows[0]["complete"]


def test_manifest_rejects_mixed_protocols(tmp_path):
    output = tmp_path / "output"
    ensure_manifest(output, {"model": "A"})
    ensure_manifest(output, {"model": "A"})
    with pytest.raises(RuntimeError):
        ensure_manifest(output, {"model": "B"})
