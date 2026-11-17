"""Materialize original simulated feedback for exactly one question."""
from __future__ import annotations
import argparse
from pathlib import Path
from mrdre_baseline.core.io import load_jsonl
from mrdre_baseline.workflows.generate import get_feedbacks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ResearcherBench")
    ap.add_argument("--question-id", required=True)
    ap.add_argument("--kind", choices=["content_feedback", "format_feedback"], required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--model", default="odr-local")
    ap.add_argument("--dataset-file", required=True)
    args = ap.parse_args()
    records = load_jsonl(args.dataset_file)
    matched = [q for q in records if str(q["id"]) == args.question_id]
    if len(matched) != 1:
        raise ValueError(f"Dataset does not contain exactly one question ID {args.question_id}")
    feedback = get_feedbacks(
        base_output_dir=str(Path(args.output_root) / args.dataset),
        model_name=args.model,
        current_turn_type=args.kind,
        current_turn=2,
        needed_qids=[args.question_id],
        questions_by_id={args.question_id: matched[0]["question"]},
        num_items=1,
    ).get(args.question_id)
    if not feedback or not str(feedback.get("feedback_text", "")).strip():
        raise RuntimeError(f"Cannot create {args.kind} feedback for question {args.question_id}")
    print(f"[OK] {args.kind} feedback Q{args.question_id}", flush=True)


if __name__ == "__main__":
    main()
