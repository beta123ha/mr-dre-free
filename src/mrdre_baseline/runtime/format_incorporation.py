"""Paper's original pairwise incorporation judge, scoped to one question."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mrdre_baseline.core.io import append_jsonl, load_jsonl
from mrdre_baseline.agents.openai_client import GPT
from mrdre_baseline.evaluation.prompts import PAIRWISE_JUDGE_SYSTEM_PROMPT, PAIRWISE_JUDGE_USER_PROMPT


def judge_one(output_root: Path, dataset: str, model: str, question_id: str) -> float:
    base = output_root / dataset
    init = base / "init" / model / "gen_turn1.jsonl"
    rev = base / "format_feedback" / model / "gen_turn2.jsonl"
    dst = base / "format_feedback" / model / "format_incorporation_turn2.jsonl"
    existing = {str(item["id"]): item for item in load_jsonl(str(dst))} if dst.exists() else {}
    if question_id in existing:
        return float(existing[question_id]["score"])

    def get_row(file: Path) -> dict:
        hits = [r for r in load_jsonl(str(file)) if str(r["id"]) == question_id]
        if len(hits) != 1:
            raise ValueError(f"Question {question_id} missing/duplicated in {file}")
        return hits[0]

    before, after = get_row(init), get_row(rev)
    feedback = str((after.get("signal") or {}).get("feedback_text") or "")
    if not feedback.strip():
        raise ValueError("Format feedback absent")
    prompt = PAIRWISE_JUDGE_USER_PROMPT.format(
        question=before["question"], report=before["full_response"]["report"],
        revised_report=after["full_response"]["report"], feedback=feedback)
    score = float(json.loads(GPT("gpt-4.1-mini", system_prompt=PAIRWISE_JUDGE_SYSTEM_PROMPT)(prompt))["score"])
    if score not in (0.0, 0.5, 1.0):
        raise ValueError(f"Invalid format-incorporation score: {score}")
    append_jsonl({"id": before["id"], "score": score, "feedback": feedback}, str(dst))
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-id", required=True)
    ap.add_argument("--dataset", default="ResearcherBench")
    ap.add_argument("--model", default="odr-local")
    ap.add_argument("--output-root", type=Path, required=True)
    a = ap.parse_args()
    print("[OK] format incorporation", judge_one(a.output_root, a.dataset, a.model, a.question_id))


if __name__ == "__main__":
    main()
