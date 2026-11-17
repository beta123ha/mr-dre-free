"""One-question-at-a-time MR DRE runner, with parallel *stages* within a question.

Every question uses one initial report and three independent Turn-2 branches.
Each stage saves its original JSONL outputs immediately; the next question is
admitted only after all requested scores and reports for the current one exist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any, Callable

from .resources import ResourcePolicy, judge_env, snapshot as resource_snapshot

ROOT = Path(__file__).resolve().parents[3]
FEEDBACK = {"content": "content_feedback", "format": "format_feedback", "reflect": "reflection"}


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    deps: tuple[str, ...]
    priority: int
    label: str = ""
    turn: int = 0
    mode: str = ""


def task_plan(feedback: tuple[str, ...], extra_eval: str = "all") -> list[Task]:
    """Explicit dependency DAG. Never change the original MR DRE scoring prompts."""
    tasks = [Task("gen1", "agent", (), 0, "init", 1),
             Task("cov1", "judge", ("gen1",), 1, "init", 1, "checklist")]
    if "content" in feedback:
        tasks += [Task("contentfb", "judge", ("cov1",), 2, "content_feedback"),
                  Task("gen2_content", "agent", ("contentfb",), 2, "content_feedback", 2),
                  Task("cov2_content", "judge", ("gen2_content",), 5, "content_feedback", 2, "checklist")]
    if "format" in feedback:
        tasks += [Task("formatfb", "judge", ("gen1",), 3, "format_feedback"),
                  Task("gen2_format", "agent", ("formatfb",), 3, "format_feedback", 2),
                  Task("cov2_format", "judge", ("gen2_format",), 5, "format_feedback", 2, "checklist"),
                  Task("formatinc", "judge", ("gen2_format",), 6, "format_feedback", 2)]
    if "reflect" in feedback:
        tasks += [Task("gen2_reflect", "agent", ("gen1",), 1, "reflection", 2),
                  Task("cov2_reflect", "judge", ("gen2_reflect",), 5, "reflection", 2, "checklist")]
    if extra_eval != "none":
        report_tasks = [("gen1", "init", 1)] + [(f"gen2_{f}", FEEDBACK[f], 2) for f in feedback]
        for gen, label, turn in report_tasks:
            tag = "1" if turn == 1 else label
            tasks.append(Task(f"rubric_{tag}", "judge", (gen,), 7, label, turn, "rubric"))
            if extra_eval == "all":
                tasks.append(Task(f"cite_{tag}", "judge", (gen,), 9, label, turn, "citation"))
    return tasks


def atomic_json(path: Path, value: dict) -> None:
    from mrdre_baseline.core.io import save_json
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    save_json(value, str(temp))
    os.replace(temp, path)


def completed_entry(path: Path, qid: str, required: str) -> bool:
    if not path.exists():
        return False
    from mrdre_baseline.core.io import load_jsonl
    try:
        for row in load_jsonl(str(path)):
            if str(row.get("id")) == qid:
                if required == "report":
                    if str((row.get("full_response") or {}).get("report", "")).strip():
                        return True
                elif required == "feedback":
                    if str(row.get("feedback", "")).strip():
                        return True
                elif required in row and row[required] is not None:
                    return True
    except (ValueError, OSError, KeyError):
        return False
    return False


class QuestionJob:
    def __init__(self, question_id: str, dataset: str, model: str, output: Path,
                 dataset_file: Path, common: dict, runtime: dict, policy: ResourcePolicy,
                 deadline: float, retries: int = 0):
        self.qid = str(question_id)
        self.dataset = dataset
        self.model = model
        self.output = output
        self.dataset_file = dataset_file
        self.common = common
        self.runtime = runtime
        self.policy = policy
        self.deadline = deadline
        self.retries = retries
        self.timings: list[dict] = []
        self.cancel = threading.Event()

    def folder(self, label: str) -> Path:
        return self.output / self.dataset / label / self.model

    def path(self, task: Task) -> tuple[Path, str]:
        if task.name == "contentfb":
            return self.folder("init") / "content_feedback_k=1_turn1.jsonl", "feedback"
        if task.name == "formatfb":
            return self.folder("init") / "format_feedback_turn1.jsonl", "feedback"
        if task.name == "formatinc":
            return self.folder("format_feedback") / "format_incorporation_turn2.jsonl", "score"
        if task.name.startswith("gen"):
            return self.folder(task.label) / f"gen_turn{task.turn}.jsonl", "report"
        suffix = {"checklist": "checklist", "citation": "citation", "rubric": "rubric"}[task.mode]
        column = {"checklist": "coverage_score", "citation": "faithfulness_score", "rubric": "presentation"}[task.mode]
        return self.folder(task.label) / f"{suffix}_turn{task.turn}.jsonl", column

    def exists(self, task: Task) -> bool:
        path, field = self.path(task)
        return completed_entry(path, self.qid, field)

    def command(self, task: Task) -> list[str]:
        py = sys.executable
        common = ["--question-id", self.qid, "--dataset", self.dataset,
                  "--output-root", str(self.output), "--model", self.model]
        if task.name.startswith("gen"):
            return [py, "-m", "mrdre_baseline.workflows.generate", "--turn", str(task.turn),
                    "--type", task.label, "--batch_size", "1", *common]
        if task.name in ("contentfb", "formatfb"):
            return [py, "-m", "mrdre_baseline.runtime.feedback_cli", *common,
                    "--kind", task.label, "--dataset-file", str(self.dataset_file)]
        if task.name == "formatinc":
            return [py, "-m", "mrdre_baseline.runtime.format_incorporation", *common]
        return [py, "-m", "mrdre_baseline.workflows.evaluate", "--report_file",
                str(self.folder(task.label) / f"gen_turn{task.turn}.jsonl"),
                "--dataset_file", str(self.dataset_file), "--eval_mode", task.mode,
                "--question-id", self.qid]

    def run_one(self, task: Task) -> None:
        if self.exists(task):
            return
        env = dict(self.common) if task.kind == "agent" else judge_env(self.common, self.runtime, self.policy)
        if task.kind == "agent":
            env["MRDRE_LLM_BACKEND"] = "ollama"
            env["MRDRE_MODEL_FINGERPRINT"] = self.runtime.get("model_digest") or ""
            env["OLLAMA_BASE_URL"] = self.runtime.get("research_ollama", "http://127.0.0.1:11434")
        command = self.command(task)
        before = resource_snapshot()
        t0 = time.monotonic()
        for attempt in range(self.retries + 1):
            if self.cancel.is_set():
                raise RuntimeError(f"Task {task.name} cancelled")
            if time.monotonic() >= self.deadline:
                raise TimeoutError(f"Session budget exhausted before {task.name}")
            print(f"[Q{self.qid}] {task.name} ({task.kind}) attempt={attempt + 1}", flush=True)
            proc = subprocess.Popen(command, cwd=ROOT, env=env,
                                    start_new_session=(os.name == "posix"))
            while proc.poll() is None:
                if self.cancel.is_set() or time.monotonic() >= self.deadline:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                        proc.wait()
                    raise TimeoutError(f"Session budget/cancellation ended during {task.name}")
                time.sleep(1)
            if proc.returncode == 0 and self.exists(task):
                duration = round(time.monotonic() - t0, 2)
                self.timings.append({"task": task.name, "seconds": duration,
                                     "workers": int(env.get("MRDRE_EVAL_WORKERS", "1")),
                                     "before": before, "after": resource_snapshot()})
                print(f"[Q{self.qid}] {task.name} OK ({duration}s)", flush=True)
                return
            print(f"[Q{self.qid}] {task.name} failed (exit={proc.returncode}); checkpoint retained", flush=True)
        raise RuntimeError(f"{task.name} did not produce a valid Q{self.qid} record")

    def execute(self, feedback: tuple[str, ...], extra_eval: str) -> dict:
        tasks = task_plan(feedback, extra_eval)
        done: set[str] = {t.name for t in tasks if self.exists(t)}
        state_path = self.output / "run_state" / f"{self.qid}.json"
        state = {"question_id": self.qid, "dataset": self.dataset, "feedback": list(feedback),
                 "extra_eval": extra_eval, "status": "running", "completed_tasks": sorted(done),
                 "last_update": datetime.now(timezone.utc).isoformat()}
        atomic_json(state_path, state)
        # Two independent GPU lanes on T4x2; a single lane on one GPU.
        separate = (self.policy.gpu_count > 1 and
                    (self.runtime.get("judge_backend") == "vllm" or
                     self.runtime.get("research_ollama") != self.runtime.get("judge_ollama")))
        agent_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")
        judge_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge") if separate else agent_pool
        futures: dict[Future, Task] = {}
        busy: set[str] = set()
        try:
            while len(done) < len(tasks):
                available = sorted((t for t in tasks if t.name not in done
                                    and t.name not in {v.name for v in futures.values()}
                                    and all(d in done for d in t.deps)), key=lambda t: (t.priority, t.name))
                for task in available:
                    lane = task.kind if separate else "single"
                    if lane in busy:
                        continue
                    futures[(agent_pool if task.kind == "agent" else judge_pool).submit(self.run_one, task)] = task
                    busy.add(lane)
                if not futures:
                    raise RuntimeError(f"Deadlocked task plan for Q{self.qid}: {done}")
                finished, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
                for future in finished:
                    task = futures.pop(future)
                    busy.discard(task.kind if separate else "single")
                    future.result()  # Propagate real evaluation failures.
                    done.add(task.name)
                    state["completed_tasks"] = sorted(done)
                    state["last_update"] = datetime.now(timezone.utc).isoformat()
                    atomic_json(state_path, state)
            state["status"] = "completed"
            state["timings"] = self.timings
            state["last_update"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
            return state
        except (Exception, KeyboardInterrupt) as exc:
            self.cancel.set()
            # Running subprocesses are deadline-bounded and finish before pool shutdown;
            # each successful stage is individually checkpointed regardless.
            state["status"] = "interrupted"
            state["error"] = str(exc)
            atomic_json(state_path, state)
            raise
        finally:
            agent_pool.shutdown(wait=True)
            if judge_pool is not agent_pool:
                judge_pool.shutdown(wait=True)


def snapshot_zip(output: Path, destination: Path | None = None) -> Path:
    """Build an atomic archive; never include the archive inside itself."""
    destination = destination or output.parent / "baseline-results.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in output.rglob("*"):
            if path.is_file() and path.resolve() not in (destination.resolve(), temp.resolve()):
                archive.write(path, Path("output") / path.relative_to(output))
        for metadata in (output.parent / "baseline-runtime.json", output / "run_manifest.json"):
            if metadata.exists():
                archive.write(metadata, metadata.name)
    os.replace(temp, destination)
    return destination


def manifest_identity(dataset_file: Path, settings: dict, runtime: dict) -> dict:
    return {"dataset_sha256": hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
            "model": runtime.get("model") or settings["MRDRE_LOCAL_MODEL"],
            "model_digest": runtime.get("model_digest"), "odr_commit": runtime.get("odr_commit"),
            "judge_backend": runtime.get("judge_backend", "ollama"),
            "judge_model": runtime.get("judge_model", runtime.get("model")),
            "odr_iterations": settings.get("ODR_RESEARCH_ITERATIONS", "3"),
            "odr_tool_calls": settings.get("ODR_MAX_TOOL_CALLS", "6"),
            "odr_final_tokens": settings.get("ODR_FINAL_MAX_TOKENS", "6144"),
            "odr_concurrency": settings.get("ODR_RESEARCH_CONCURRENCY", "1"),
            "context_length": runtime.get("ollama_context_length"),
            "dataset_seed": 42, "content_targets": 1}


def ensure_manifest(output: Path, identity: dict) -> None:
    path = output / "run_manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != identity:
            raise RuntimeError("Existing output uses a different model/dataset/ODR configuration; choose a new --output-root")
    else:
        atomic_json(path, identity)


def order_questions(data: list[dict], count: int, only_id: str | None) -> list[dict]:
    values = data.copy()
    random.Random(42).shuffle(values)
    if only_id is not None:
        values = [item for item in values if str(item["id"]) == str(only_id)]
        if len(values) != 1:
            raise ValueError(f"Missing or duplicate question ID {only_id}")
        return values
    return values if count < 0 else values[:count]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Complete one MR DRE question before admitting the next")
    ap.add_argument("--dataset", default="ResearcherBench")
    ap.add_argument("--count", type=int, default=-1)
    ap.add_argument("--question-id")
    ap.add_argument("--feedback", choices=[*FEEDBACK, "all"], default="all")
    ap.add_argument("--extra-eval", choices=["all", "presentation", "none"], default="all")
    ap.add_argument("--budget-minutes", type=float, default=0, help="0=no time limit; leave a platform safety margin")
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--checkpoint-zip", type=Path, default=None)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Display the execution DAG without running models")
    args = ap.parse_args(argv)
    if args.count == 0 or args.count < -1 or args.retries < 0:
        ap.error("--count must be -1 or positive, and --retries nonnegative")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    dataset_file = ROOT / "data" / f"{args.dataset}.jsonl"
    if not dataset_file.is_file():
        ap.error(f"Dataset missing: {dataset_file}")
    from mrdre_baseline.core.io import load_jsonl
    questions = order_questions(load_jsonl(str(dataset_file)), args.count, args.question_id)
    modes = tuple(FEEDBACK) if args.feedback == "all" else (args.feedback,)
    if args.dry_run:
        print("\n".join(f"{t.kind:5} {t.name:18} after {','.join(t.deps) or '-'}" for t in task_plan(modes, args.extra_eval)))
        print(f"Questions: {len(questions)}")
        return 0
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta = output.parent / "baseline-runtime.json"
    if not meta.exists():
        ap.error(f"Missing {meta}; run python scripts/bootstrap.py --profile local first")
    runtime = json.loads(meta.read_text(encoding="utf-8"))
    settings = os.environ.copy()
    settings.update({"MRDRE_LLM_BACKEND": "ollama", "MRDRE_LOCAL_MODEL": runtime.get("model", os.getenv("MRDRE_LOCAL_MODEL", "qwen3:4b-instruct")),
                     "ODR_LOCAL_MODEL": runtime.get("model", os.getenv("ODR_LOCAL_MODEL", "qwen3:4b-instruct")),
                     "ODR_URL": runtime.get("odr_url", os.getenv("ODR_URL", "http://127.0.0.1:2024")),
                     "ODR_MCP_URL": runtime.get("mcp_url", os.getenv("ODR_MCP_URL", "http://127.0.0.1:8001")),
                     "MRDRE_CRAWLER": "direct", "PYTHONUNBUFFERED": "1"})
    for key, default in (("ODR_RESEARCH_ITERATIONS", "3"), ("ODR_MAX_TOOL_CALLS", "6"),
                         ("ODR_FINAL_MAX_TOKENS", "6144"), ("ODR_RESEARCH_CONCURRENCY", "1"),
                         ("REQUEST_TIMEOUT_SECONDS", "2700"), ("POLL_INTERVAL_SECONDS", "2"),
                         ("BATCH_TIMEOUT_MINUTES", "75")):
        settings.setdefault(key, default)
    ensure_manifest(output, manifest_identity(dataset_file, settings, runtime))
    policy = ResourcePolicy.detect(runtime)
    print(f"[RESOURCES] {policy.gpu_count} GPU(s), {policy.max_workers} max judge workers, one question at a time", flush=True)
    deadline = time.monotonic() + args.budget_minutes * 60 if args.budget_minutes else float("inf")
    complete = 0
    for question in questions:
        if time.monotonic() >= deadline:
            break
        qid = str(question["id"])
        job = QuestionJob(qid, args.dataset, "odr-local", output, dataset_file,
                          settings, runtime, policy, deadline, args.retries)
        try:
            result = job.execute(modes, args.extra_eval)
            complete += 1
            print(f"[COMPLETE] Q{qid} ({complete}/{len(questions)} in this invocation)", flush=True)
            snapshot_zip(output, args.checkpoint_zip)
        except (Exception, KeyboardInterrupt) as exc:
            print(f"[STOP] Q{qid}: {exc}; completed stages retained", file=sys.stderr, flush=True)
            snapshot_zip(output, args.checkpoint_zip)
            return 2
    print(f"[FINISHED] {complete}/{len(questions)} questions completed/verified during this invocation", flush=True)
    return 0 if complete == len(questions) else 3


if __name__ == "__main__":
    raise SystemExit(main())
