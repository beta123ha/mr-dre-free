"""Aggregate ONLY completed full-question checkpoints, never partial tasks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mrdre_baseline.runtime.metrics import collect


def completed_ids(output: Path, feedback: str, extra_eval: str) -> set[str]:
    done = set()
    state_dir = output / "run_state"
    for path in state_dir.glob("*.json") if state_dir.exists() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        if (item.get("status") == "completed" and feedback in item.get("feedback", [])
                and item.get("extra_eval") == extra_eval):
            done.add(str(item["question_id"]))
    return done


def summarize(output: Path, feedback: str = "all", dataset: str = "ResearcherBench",
              extra_eval: str = "all") -> list[dict]:
    choices = ["content", "format", "reflect"] if feedback == "all" else [feedback]
    summaries = []
    for name in choices:
        ids = completed_ids(output, name, extra_eval)
        # Important: zero complete samples is not evidence of a zero metric.
        _, row = collect(output, name, expected=len(ids), eligible_ids=ids, dataset=dataset)
        row["completed_questions"] = len(ids)
        row["complete"] = bool(ids) and row["complete"]
        summaries.append(row)
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--feedback", choices=["all", "content", "format", "reflect"], default="all")
    ap.add_argument("--dataset", default="ResearcherBench")
    ap.add_argument("--extra-eval", choices=["all", "presentation", "none"], default="all")
    args = ap.parse_args()
    for row in summarize(args.output_root, args.feedback, args.dataset, args.extra_eval):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
