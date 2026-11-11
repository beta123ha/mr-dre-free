"""Resumable MR DRE baseline: original LC ODR, no edit planning and no reviser.

GPU0: research; GPU1 (if available): checklist judging. The optional slow
citation and presentation evaluations are run after paired reports are saved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution as `python kaggle/run.py` and module imports alike.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = "ResearcherBench"
AGENT = "odr-local"
FEEDBACK = {
    "content": "content_feedback",
    "format": "format_feedback",
    "reflect": "reflection",
}


def read_ids(path: Path, required_report: bool = False) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if required_report and not str((obj.get("full_response") or {}).get("report") or "").strip():
                continue
            ids.add(str(obj["id"]))
    return ids


def snapshot(output: Path) -> Path:
    """Atomic, persistent checkpoint for Kaggle Save & Run All output."""
    archive = output.parent / "baseline-results.zip"
    temp = output.parent / "baseline-results.zip.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for f in output.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(output.parent))
        config = output.parent / "baseline-runtime.json"
        if config.exists():
            z.write(config, config.name)
    os.replace(temp, archive)
    return archive


def command(args: list[str], env: dict) -> subprocess.Popen:
    print("$", " ".join(args), flush=True)
    return subprocess.Popen(args, cwd=ROOT, env=env)


def run_stage(turn: int, setting: str, target: int, output: Path,
              gen_env: dict, judge_env: dict, deadline: float) -> bool:
    label = "init" if turn == 1 else setting
    folder = output / DATASET / label / AGENT
    gen_file = folder / f"gen_turn{turn}.jsonl"
    check_file = folder / f"checklist_turn{turn}.jsonl"
    print(f"\n[TURN {turn}] {label} | target: {target}", flush=True)

    generate = command([
        sys.executable, "-m", "mrdre_baseline.workflows.generate",
        "--turn", str(turn), "--type", label, "--dataset", DATASET,
        "--model", AGENT, "--num_questions", str(target), "--batch_size", "1",
        "--output-root", str(output),
    ], gen_env)
    judge = None
    previous_count = (-1, -1)
    last_save = time.monotonic()
    generated_status = None

    while True:
        if generate.poll() is not None:
            generated_status = generate.returncode
        gen_count = len(read_ids(gen_file, required_report=True))
        judge_count = len(read_ids(check_file))
        if (gen_count, judge_count) != previous_count:
            print(f"[TURN {turn}] reports {gen_count}/{target} | evaluated {judge_count}/{target}", flush=True)
            previous_count = (gen_count, judge_count)
            if gen_count and time.monotonic() - last_save >= 30:
                snapshot(output)
                last_save = time.monotonic()

        if judge is not None and judge.poll() is not None:
            if judge.returncode:
                print(f"[ERROR] judge exit={judge.returncode}", flush=True)
                if generate.poll() is None:
                    generate.terminate()
                snapshot(output)
                return False
            judge = None

        # Judges use GPU1 and grade only the reports available when their
        # process starts. A fresh evaluator picks up later report checkpoints.
        if judge is None and gen_count > judge_count:
            judge = command([
                sys.executable, "-m", "mrdre_baseline.workflows.evaluate",
                "--report_file", str(gen_file),
                "--dataset_file", str(ROOT / "data" / f"{DATASET}.jsonl"),
                "--eval_mode", "checklist",
            ], judge_env)

        if generated_status is not None and judge is None and judge_count >= gen_count:
            snapshot(output)
            if generated_status == 0 and gen_count >= target and judge_count >= target:
                print(f"[OK] Turn {turn}: {target} paired reports/judgements", flush=True)
                return True
            print(f"[INCOMPLETE] turn={turn}, generated={gen_count}, judged={judge_count}, exit={generated_status}", flush=True)
            return False

        if time.monotonic() >= deadline:
            print("[BUDGET] Checkpointing before Kaggle session limit.", flush=True)
            for child in (generate, judge):
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        child.kill()
            snapshot(output)
            return False
        time.sleep(8)


def write_scores(output: Path, setting: str, target: int) -> dict:
    """Write per-question and aggregate six-metric scores; missing = blank."""
    from kaggle.metrics import collect
    kind = next(k for k, v in FEEDBACK.items() if v == setting)
    _, summary = collect(output, kind, target)
    print("[METRICS]", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def evaluate_format_incorporation(output: Path, target: int, judge_env: dict, deadline: float) -> bool:
    """Original pairwise incorporation judge; resume each independently scored pair."""
    from mrdre_baseline.core.io import append_jsonl, load_jsonl
    from mrdre_baseline.agents.openai_client import GPT
    from mrdre_baseline.evaluation.prompts import PAIRWISE_JUDGE_SYSTEM_PROMPT, PAIRWISE_JUDGE_USER_PROMPT
    folder = output / DATASET / "format_feedback" / AGENT
    original = output / DATASET / "init" / AGENT / "gen_turn1.jsonl"
    revision = folder / "gen_turn2.jsonl"
    destination = folder / "format_incorporation_turn2.jsonl"
    before = {str(row["id"]): row for row in load_jsonl(str(original))}
    after = {str(row["id"]): row for row in load_jsonl(str(revision))}
    done = read_ids(destination)
    os.environ.update({"MRDRE_LLM_BACKEND": "ollama",
                       "MRDRE_LOCAL_MODEL": judge_env["MRDRE_LOCAL_MODEL"],
                       "OLLAMA_BASE_URL": judge_env["OLLAMA_BASE_URL"]})
    judge = GPT("gpt-4.1-mini", system_prompt=PAIRWISE_JUDGE_SYSTEM_PROMPT)
    for qid in sorted(set(before) & set(after), key=str):
        if qid in done:
            continue
        if time.monotonic() >= deadline:
            return False
        old, new = before[qid], after[qid]
        feedback = (new.get("signal") or {}).get("feedback_text")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError(f"Q{qid} missing format feedback")
        question = old["question"]
        prompt = PAIRWISE_JUDGE_USER_PROMPT.format(
            question=question,
            report=old["full_response"]["report"],
            revised_report=new["full_response"]["report"],
            feedback=feedback,
        )
        score = float(json.loads(judge(prompt))["score"])
        if score not in (0.0, 0.5, 1.0):
            raise ValueError(f"Q{qid} invalid incorporation score: {score}")
        append_jsonl({"id": old["id"], "score": score, "feedback": feedback}, str(destination))
        done.add(qid)
    return len(done) >= target


def chunk_targets(n: int, batch: int) -> list[int]:
    if n < 1 or n > 65 or batch < 0:
        raise ValueError("n must be 1..65 and batch nonnegative")
    return [n] if batch == 0 else [min(i, n) for i in range(batch, n+batch, batch)]


def extra_scoring(output: Path, settings: list[str], generation_env: dict,
                  judge_env: dict, deadline: float, mode: str) -> bool:
    """Grade different report sets on both GPUs after report generation.

    Each (report set, metric) writes to separate JSONL files. No shared write
    targets, and interrupted jobs resume by ID. Citation is prioritized because
    it is much more expensive than presentation.
    """
    if mode == "none":
        return True
    from collections import deque
    labels = [("init", 1)] + [(setting, 2) for setting in settings]
    jobs = deque()
    if mode == "all":
        jobs.extend((label, turn, "citation") for label, turn in labels)
    jobs.extend((label, turn, "rubric") for label, turn in labels)
    envs = [generation_env]
    if generation_env["OLLAMA_BASE_URL"] != judge_env["OLLAMA_BASE_URL"]:
        envs.append(judge_env)
    active: dict[int, tuple[subprocess.Popen, str, int, str]] = {}
    while jobs or active:
        if time.monotonic() >= deadline:
            print("[BUDGET] Stop independent evaluations safely", flush=True)
            for child, *_ in active.values():
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
            snapshot(output)
            return False
        while jobs and len(active) < len(envs):
            slot = next(i for i in range(len(envs)) if i not in active)
            label, turn, eval_type = jobs.popleft()
            folder = output / DATASET / label / AGENT
            report = folder / f"gen_turn{turn}.jsonl"
            args = [sys.executable, "-m", "mrdre_baseline.workflows.evaluate",
                    "--report_file", str(report),
                    "--dataset_file", str(ROOT / "data" / f"{DATASET}.jsonl"),
                    "--eval_mode", eval_type]
            proc = command(args, envs[slot])
            print(f"[EVALUATE] GPU{slot} {label} turn{turn} {eval_type}", flush=True)
            active[slot] = (proc, label, turn, eval_type)
        time.sleep(5)
        still_running = {}
        for slot, (child, label, turn, eval_type) in active.items():
            if child.poll() is None:
                still_running[slot] = (child, label, turn, eval_type)
                continue
            if child.returncode != 0:
                print(f"[ERROR] {eval_type} {label} turn {turn}: {child.returncode}", flush=True)
                for other, *_ in active.values():
                    if other is not child and other.poll() is None:
                        other.terminate()
                snapshot(output)
                return False
            print(f"[OK] {eval_type} {label} turn{turn}", flush=True)
            snapshot(output)
        active = still_running
    return True


def main() -> int:
    cli = argparse.ArgumentParser(description="MR DRE baseline on ResearcherBench, Kaggle Free")
    cli.add_argument("--count", type=int, default=1)
    cli.add_argument("--feedback", choices=list(FEEDBACK) + ["all"], default="content")
    cli.add_argument("--chunk-size", type=int, default=5)
    cli.add_argument("--budget-minutes", type=int, default=630)
    cli.add_argument("--extra-eval", choices=["none", "presentation", "all"], default="all",
                     help="After paired checklist reports, evaluate presentation and optionally citations")
    cli.add_argument("--output-root", default="/kaggle/working/output" if Path("/kaggle/working").exists() else str(ROOT / "output"))
    args = cli.parse_args()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = output.parent / "baseline-runtime.json"
    if not metadata.exists():
        cli.error(f"Missing {metadata}; first run kaggle/bootstrap.py")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    settings = [FEEDBACK[key] for key in (list(FEEDBACK) if args.feedback == "all" else [args.feedback])]
    common = os.environ.copy()
    common.update({
        "MRDRE_LLM_BACKEND": "ollama", "MRDRE_LOCAL_MODEL": info["model"],
        "ODR_LOCAL_MODEL": info["model"], "ODR_URL": info["odr_url"],
        "ODR_MCP_URL": info["mcp_url"], "MRDRE_EVAL_WORKERS": "1",
        "MRDRE_SUMMARY_WORKERS": "1", "MRDRE_CRAWLER": "direct",
        "REQUEST_TIMEOUT_SECONDS": "2700", "POLL_INTERVAL_SECONDS": "2",
        "BATCH_TIMEOUT_MINUTES": "75", "PYTHONUNBUFFERED": "1",
        "ODR_RESEARCH_ITERATIONS": os.getenv("ODR_RESEARCH_ITERATIONS", "3"),
        "ODR_MAX_TOOL_CALLS": os.getenv("ODR_MAX_TOOL_CALLS", "6"),
        "ODR_RESEARCH_CONCURRENCY": "1",
        "ODR_FINAL_MAX_TOKENS": os.getenv("ODR_FINAL_MAX_TOKENS", "6144"),
    })
    generation_env = {**common, "OLLAMA_BASE_URL": info["research_ollama"]}
    evaluation_env = {**common, "OLLAMA_BASE_URL": info["judge_ollama"]}
    deadline = time.monotonic() + args.budget_minutes * 60
    for count in chunk_targets(args.count, args.chunk_size):
        print(f"\n[GROUP] {count}/{args.count} | {', '.join(settings)}", flush=True)
        if not run_stage(1, "init", count, output, generation_env, evaluation_env, deadline):
            snapshot(output)
            return 2
        for setting in settings:
            if not run_stage(2, setting, count, output, generation_env, evaluation_env, deadline):
                snapshot(output)
                return 2
            # Metrics are written even before the slow judges have finished.
            # Unmeasured axes remain NULL, never reported as zero.
            write_scores(output, setting, count)
        if args.extra_eval != "none":
            # During this (expensive) phase GPU0 is no longer generating,
            # so both GPUs can grade *different* reports simultaneously.
            if not extra_scoring(output, settings, generation_env, evaluation_env,
                                 deadline, args.extra_eval):
                snapshot(output)
                return 3
            for setting in settings:
                if setting == "format_feedback":
                    try:
                        if not evaluate_format_incorporation(output, count, evaluation_env, deadline):
                            snapshot(output)
                            return 3
                    except Exception as e:
                        print(f"[ERROR] Format incorporation judgement: {e}", flush=True)
                        snapshot(output)
                        return 3
                write_scores(output, setting, count)
        print("[CHECKPOINT]", snapshot(output), flush=True)
    if args.extra_eval == "all":
        from kaggle.metrics import collect
        for setting in settings:
            key = next(k for k, v in FEEDBACK.items() if v == setting)
            _, row = collect(output, key, args.count)
            if not row["complete"]:
                print("[INCOMPLETE] Missing some metric judgments:", row, flush=True)
                snapshot(output)
                return 4
    print("[COMPLETE]", snapshot(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
