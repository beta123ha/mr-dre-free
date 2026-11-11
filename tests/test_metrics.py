"""Protocol-level unit tests; do not call GPUs or remote APIs."""
import json
from pathlib import Path

from kaggle.metrics import break_rate, incorporation_content, collect


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')


def checklist(turn, qid=1):
    return {'id': qid, 'coverage_score': turn / 2, 'checklist_details': [
        {'id': 1, 'weight': 2, 'covered': float(turn == 1)},
        {'id': 2, 'weight': -1, 'covered': float(turn == 2)},
        {'id': 3, 'weight': 1, 'covered': float(turn == 2)},
    ]}


def test_paper_broken_include_negative_weights():
    assert break_rate(checklist(1), checklist(2)) == 1.0
    assert incorporation_content(checklist(2), {'signal': {'targeted_id': [3]}}) == 1
    assert incorporation_content(checklist(2), {'signal': {'targeted_id': [2]}}) == 0


def test_six_metrics_are_not_fake_zeros(tmp_path):
    root = tmp_path / 'output' / 'ResearcherBench'
    init = root / 'init' / 'odr-local'
    rev = root / 'content_feedback' / 'odr-local'
    write_rows(init / 'gen_turn1.jsonl', [{'id': 1, 'question': 'q', 'full_response': {'report': 'old'}}])
    write_rows(rev / 'gen_turn2.jsonl', [{'id': 1, 'question': 'q', 'full_response': {'report': 'new'},
                                          'signal': {'targeted_id': [3], 'feedback_text': 'fix'}}])
    write_rows(init / 'checklist_turn1.jsonl', [checklist(1)])
    write_rows(rev / 'checklist_turn2.jsonl', [checklist(2)])
    rows, summary = collect(tmp_path / 'output', 'content', 1)
    assert not summary['complete']
    assert summary['n_coverage_t1'] == 1
    assert summary['n_faithfulness_t1'] == 0
    assert rows[0]['faithfulness_t1'] is None
    for folder, turn, f, g, p in [(init, 1, .8, .4, .9), (rev, 2, .7, .3, .95)]:
        write_rows(folder / f'citation_turn{turn}.jsonl',
                   [{'id': 1, 'faithfulness_score': f, 'groundedness_score': g}])
        write_rows(folder / f'rubric_turn{turn}.jsonl', [{'id': 1, 'presentation': p}])
    rows, summary = collect(tmp_path / 'output', 'content', 1)
    assert summary['complete']
    assert abs(rows[0]['faithfulness_delta'] + .1) < 1e-8
    assert abs(rows[0]['groundedness_delta'] + .1) < 1e-8
    assert abs(rows[0]['presentation_delta'] - .05) < 1e-8
    assert rows[0]['incorporation'] == 1
    assert rows[0]['break_rate'] == 1


def test_format_incorporation_only_when_judged(tmp_path):
    root = tmp_path / 'output' / 'ResearcherBench'
    init = root / 'init' / 'odr-local'
    rev = root / 'format_feedback' / 'odr-local'
    write_rows(init / 'gen_turn1.jsonl', [{'id': 1}])
    write_rows(rev / 'gen_turn2.jsonl', [{'id': 1}])
    _, summary = collect(tmp_path / 'output', 'format', 1)
    assert summary['n_incorporation'] == 0
    write_rows(rev / 'format_incorporation_turn2.jsonl', [{'id': 1, 'score': 1}])
    rows, summary = collect(tmp_path / 'output', 'format', 1)
    assert summary['n_incorporation'] == 1
    assert rows[0]['incorporation'] == 1


def test_reflection_does_not_have_incorporation(tmp_path):
    root = tmp_path / 'output' / 'ResearcherBench'
    init = root / 'init' / 'odr-local'
    rev = root / 'reflection' / 'odr-local'
    write_rows(init / 'gen_turn1.jsonl', [{'id': 1}])
    write_rows(rev / 'gen_turn2.jsonl', [{'id': 1}])
    _, summary = collect(tmp_path / 'output', 'reflect', 1)
    assert summary['incorporation'] is None
    assert summary['n_incorporation'] == 0
