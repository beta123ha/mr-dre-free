import hashlib
import json
from pathlib import Path

from mrdre_baseline.core.io import get_file_path, load_jsonl, save_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_datasets_unchanged():
    expectations = {
        "ResearchRubrics": (101, "5d9347c9c38666294a8ed49caa8320854e10c0a5c23aa35f184eca92a17dde0e"),
        "ResearcherBench": (65, "e345fe388af84ccb9cd1d92c52a7d409ab0ed1ca87f4279be8d9e8874e100391"),
        "RigorousBench": (214, "913074b6bd95eccfdde3423e49b03de50a9cebd44ffdc790ff63c1475595c19e"),
    }
    for name, (length, digest) in expectations.items():
        path = ROOT / "data" / f"{name}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        rows = load_jsonl(str(path))
        assert len(rows) == length
        assert len({str(x['id']) for x in rows}) == length
        assert all(x['question'] and isinstance(x['checklist'], list) for x in rows)


def test_atomic_jsonl_replaces_old_version(tmp_path):
    path = tmp_path / "records.jsonl"
    save_jsonl([{"id": 1}], str(path))
    save_jsonl([{"id": 1}, {"id": 2, "text": "Tiếng Việt"}], str(path))
    assert load_jsonl(str(path)) == [{"id": 1}, {"id": 2, "text": "Tiếng Việt"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_generated_paths_stable():
    assert get_file_path("output/ResearcherBench", "odr-local", 1, "init") == \
           "output/ResearcherBench/init/odr-local/gen_turn1.jsonl"
    assert get_file_path("output/ResearcherBench", "odr-local", 2, "content_feedback") == \
           "output/ResearcherBench/content_feedback/odr-local/gen_turn2.jsonl"


def test_kaggle_paired_group_targets():
    from kaggle.run import chunk_targets
    assert chunk_targets(65, 5) == list(range(5, 66, 5))
    assert chunk_targets(63, 5)[-2:] == [60, 63]
    assert chunk_targets(65, 0) == [65]
    assert chunk_targets(1, 5) == [1]


def test_paper_break_rate_includes_feedback_targets():
    from kaggle.metrics import break_rate
    before = {"id": 3, "checklist_details": [
        {"id": 1, "weight": 2, "covered": 1.0},
        {"id": 2, "weight": 2, "covered": 0.5},
        {"id": 3, "weight": -1, "covered": 0.0},
    ]}
    after = {"id": 3, "checklist_details": [
        {"id": 1, "weight": 2, "covered": 0.0},
        {"id": 2, "weight": 2, "covered": 1.0},
        {"id": 3, "weight": -1, "covered": 1.0},
    ]}
    assert break_rate(before, after) == 2 / 3


def test_fast_runner_has_no_reviser_flags():
    from kaggle.run import FEEDBACK
    assert FEEDBACK == {"content": "content_feedback", "format": "format_feedback", "reflect": "reflection"}


def test_atomic_jsonl_append(tmp_path):
    from mrdre_baseline.core.io import append_jsonl
    path = tmp_path / 'out.jsonl'
    append_jsonl({'id': 1}, str(path))
    append_jsonl({'id': 2}, str(path))
    assert load_jsonl(str(path)) == [{'id': 1}, {'id': 2}]
