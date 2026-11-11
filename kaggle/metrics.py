"""MR DRE ACL 2026 metric aggregation over ID-aligned, *observed* evaluations.

Only the paper's six axes are aggregated. Missing runs never become zero scores.
Inc for reflection is undefined, exactly as in the paper's main table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import tempfile

DATASET = 'ResearcherBench'
AGENT = 'odr-local'
SETTING_NAMES = {'content': 'content_feedback', 'format': 'format_feedback', 'reflect': 'reflection'}


def records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                item = json.loads(line)
                if 'id' not in item:
                    raise ValueError(f'Missing id at {path}:{lineno}')
                out[str(item['id'])] = item
    return out


def break_rate(old: dict, new: dict) -> float | None:
    """Equation in Appendix B.4, all previously achieved criteria included."""
    previous = {str(x['id']): x for x in old['checklist_details']}
    latest = {str(x['id']): x for x in new['checklist_details']}
    if previous.keys() != latest.keys():
        raise ValueError('Checklist item ID mismatch across turns')
    achieved = broken = 0
    for key, prior in previous.items():
        now = latest[key]
        w = float(prior['weight'])
        if w == 0:
            continue
        s1, s2 = float(prior['covered']), float(now['covered'])
        if (w > 0 and s1 > 0) or (w < 0 and s1 < 1):
            achieved += 1
            broken += int(w * s2 < w * s1 - 1e-8)
    return broken / achieved if achieved else None


def incorporation_content(new: dict, report: dict) -> float | None:
    target_ids = (report.get('signal') or {}).get('targeted_id') or []
    if not target_ids:
        return None
    items = {str(x['id']): x for x in new['checklist_details']}
    scores = []
    for raw_id in target_ids:
        item = items.get(str(raw_id))
        if item is None:
            raise ValueError(f'Missing targeted checklist item {raw_id}')
        weight = float(item['weight'])
        score = float(item['covered'])
        scores.append(float(score == (1.0 if weight > 0 else 0.0)))
    return statistics.mean(scores)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def collect(output: Path, feedback: str, expected: int = 65) -> tuple[list[dict], dict]:
    setting = SETTING_NAMES[feedback]
    base = output / DATASET
    init = base / 'init' / AGENT
    rev = base / setting / AGENT
    sources = {
        'gen1': records(init / 'gen_turn1.jsonl'),
        'gen2': records(rev / 'gen_turn2.jsonl'),
        'cov1': records(init / 'checklist_turn1.jsonl'),
        'cov2': records(rev / 'checklist_turn2.jsonl'),
        'cite1': records(init / 'citation_turn1.jsonl'),
        'cite2': records(rev / 'citation_turn2.jsonl'),
        'pres1': records(init / 'rubric_turn1.jsonl'),
        'pres2': records(rev / 'rubric_turn2.jsonl'),
        'fmtinc': records(rev / 'format_incorporation_turn2.jsonl'),
    }
    ids = sorted(set(sources['gen1']) & set(sources['gen2']), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    fields = ['id', 'setting', 'coverage_t1', 'coverage_t2', 'coverage_delta',
              'faithfulness_t1', 'faithfulness_t2', 'faithfulness_delta',
              'groundedness_t1', 'groundedness_t2', 'groundedness_delta',
              'presentation_t1', 'presentation_t2', 'presentation_delta',
              'incorporation', 'break_rate']
    per_question = []
    for qid in ids:
        row = {field: None for field in fields}
        row.update({'id': qid, 'setting': feedback})
        if qid in sources['cov1'] and qid in sources['cov2']:
            a, b = sources['cov1'][qid], sources['cov2'][qid]
            row['coverage_t1'], row['coverage_t2'] = float(a['coverage_score']), float(b['coverage_score'])
            row['coverage_delta'] = row['coverage_t2'] - row['coverage_t1']
            row['break_rate'] = break_rate(a, b)
            if feedback == 'content':
                row['incorporation'] = incorporation_content(b, sources['gen2'][qid])
        if feedback == 'format' and qid in sources['fmtinc']:
            # The paper treats format incorporation as a binary decision.
            row['incorporation'] = float(sources['fmtinc'][qid]['score'] == 1.0)
        for basename, key in [('faithfulness', 'faithfulness_score'), ('groundedness', 'groundedness_score')]:
            if qid in sources['cite1'] and qid in sources['cite2']:
                row[f'{basename}_t1'] = float(sources['cite1'][qid][key])
                row[f'{basename}_t2'] = float(sources['cite2'][qid][key])
                row[f'{basename}_delta'] = row[f'{basename}_t2'] - row[f'{basename}_t1']
        if qid in sources['pres1'] and qid in sources['pres2']:
            row['presentation_t1'] = float(sources['pres1'][qid]['presentation'])
            row['presentation_t2'] = float(sources['pres2'][qid]['presentation'])
            row['presentation_delta'] = row['presentation_t2'] - row['presentation_t1']
        per_question.append(row)

    summary = {'dataset': DATASET, 'agent': AGENT, 'setting': feedback, 'expected': expected,
               'paired_reports': len(ids)}
    numeric_fields = fields[2:]
    for field in numeric_fields:
        vals = [r[field] for r in per_question if r[field] is not None]
        summary[field] = round(statistics.mean(vals), 6) if vals else None
        summary[f'n_{field}'] = len(vals)
    # INC in Reflect is intentionally N/A, rather than 0 or a fabricated number.
    mandatory = ['coverage_t1', 'coverage_t2', 'faithfulness_t1', 'faithfulness_t2',
                 'groundedness_t1', 'groundedness_t2', 'presentation_t1',
                 'presentation_t2', 'break_rate']
    if feedback != 'reflect':
        mandatory.append('incorporation')
    summary['complete'] = len(ids) == expected and all(summary[f'n_{field}'] == expected for field in mandatory)
    target_dir = output / 'metrics'
    atomic_csv(target_dir / f'{feedback}_per_question.csv', per_question, fields)
    atomic_csv(target_dir / f'{feedback}_summary.csv', [summary], list(summary))
    return per_question, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path('/kaggle/working/output'))
    parser.add_argument('--feedback', choices=list(SETTING_NAMES) + ['all'], default='all')
    parser.add_argument('--count', type=int, default=65)
    args = parser.parse_args()
    for mode in (list(SETTING_NAMES) if args.feedback == 'all' else [args.feedback]):
        _, row = collect(args.output_root, mode, expected=args.count)
        print(json.dumps(row, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
