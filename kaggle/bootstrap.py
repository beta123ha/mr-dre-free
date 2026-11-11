"""Prepare a free Kaggle GPU VM for a reproducible MR DRE baseline run.

Run from the repository root: python kaggle/bootstrap.py
All services bind only to localhost. No paid API credentials are required.
"""
from __future__ import annotations

import json
import socket
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.getenv("MRDRE_BASELINE_WORKDIR", "/kaggle/working" if Path("/kaggle/working").exists() else str(ROOT)))
LOGS = WORK / "baseline-logs"
MODELS = WORK / "ollama-models"
ODR = WORK / "open_deep_research"
MODEL = os.getenv("MRDRE_BASELINE_MODEL", "qwen3:4b-instruct")


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def wait_http(url: str, label: str, seconds: int = 100) -> None:
    for _ in range(seconds):
        try:
            with urlopen(url, timeout=2) as r:
                if 200 <= r.status < 500:
                    print(f"[OK] {label} {url}")
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{label} did not start at {url}; see {LOGS}")


def service(command: list[str], name: str, env: dict, cwd: Path) -> None:
    log = open(LOGS / f"{name}.log", "ab", buffering=0)
    proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    (LOGS / f"{name}.pid").write_text(str(proc.pid))
    print(f"Started {name} (pid={proc.pid})")


def main() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Requires Python 3.11+, switch the Kaggle environment")
    LOGS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Verify GPU before spending quota on downloads.
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                           text=True, capture_output=True)
    if query.returncode:
        raise RuntimeError("GPU not visible. In Kaggle select Settings > Accelerator > GPU T4 x2")
    gpu_lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    gpu_count = min(2, len(gpu_lines))
    print("Detected GPUs:", *gpu_lines, sep="\n  ")

    run(sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT))
    prior_metadata = WORK / "baseline-runtime.json"
    previous = json.loads(prior_metadata.read_text(encoding="utf-8")) if prior_metadata.exists() else {}
    if previous and previous.get("model") != MODEL:
        raise RuntimeError("Cannot resume with a different model; keep the same model tag")
    if not ODR.exists():
        run("git", "clone", "--depth", "1", "https://github.com/langchain-ai/open_deep_research.git", str(ODR))
    old_sha = previous.get("odr_commit")
    if old_sha:
        current_sha = subprocess.check_output(["git", "-C", str(ODR), "rev-parse", "HEAD"], text=True).strip()
        if current_sha != old_sha:
            run("git", "-C", str(ODR), "fetch", "--depth", "1", "origin", old_sha)
            run("git", "-C", str(ODR), "checkout", "--detach", old_sha)
    run(sys.executable, "-m", "pip", "install", "-q", "-e", str(ODR), "langgraph-cli[inmem]", "langchain-ollama")
    # langgraph.json points to .env; make an empty one if the upstream clone
    # does not provide it, without depending on any external secrets.
    (ODR / ".env").touch(exist_ok=True)
    run(sys.executable, str(ROOT / "scripts/configure_odr_local.py"), str(ODR))
    # Kaggle's Python image may differ from the Python version in the ODR
    # deployment JSON. Match the local process interpreter, not the graph.
    graph_cfg = ODR / "langgraph.json"
    obj = json.loads(graph_cfg.read_text(encoding="utf-8"))
    expected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if obj.get("python_version") != expected_python:
        obj["python_version"] = expected_python
        graph_cfg.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        print("[CONFIG] LangGraph Python version:", expected_python)
    # Keep official LC ODR graph and prompts unmodified. The non-thinking
    # Qwen3:4b-instruct model is selected instead of patching upstream code.
    sha = subprocess.check_output(["git", "-C", str(ODR), "rev-parse", "HEAD"], text=True).strip()

    if not shutil.which("ollama"):
        run("bash", "-o", "pipefail", "-c", "curl -fsSL https://ollama.com/install.sh | sh")
    ollama = shutil.which("ollama")
    if not ollama:
        raise RuntimeError("Ollama executable not found after installing")

    # Ollama installer may auto-start a default server without GPU isolation.
    subprocess.run(["pkill", "-x", "ollama"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)
    time.sleep(2)
    base = os.environ.copy()
    base.update({"OLLAMA_MODELS": str(MODELS), "OLLAMA_FLASH_ATTENTION": "1",
                 "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_NUM_PARALLEL": "1",
                 "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_CONTEXT_LENGTH": "24576",
                 "OLLAMA_KEEP_ALIVE": "15m", "OLLAMA_MAX_QUEUE": "16"})
    for i in range(gpu_count):
        port = 11434 + i
        env = base | {"OLLAMA_HOST": f"127.0.0.1:{port}", "CUDA_VISIBLE_DEVICES": str(i)}
        service([ollama, "serve"], f"ollama_gpu{i}", env, WORK)
        wait_http(f"http://127.0.0.1:{port}/api/tags", f"Ollama GPU {i}")

    pull_env = base | {"OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": "0"}
    run(ollama, "pull", MODEL, env=pull_env)
    # Both local Ollama instances share their immutable model files.
    for i in range(gpu_count):
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{11434+i}/api/tags", timeout=10) as r:
            tags = json.load(r)
        if not any(x.get("name", "") == MODEL for x in tags.get("models", [])):
            raise RuntimeError(f"GPU {i} server cannot find model {MODEL}")

    mcp_env = base | {"PYTHONUNBUFFERED": "1"}
    service([sys.executable, "-m", "mrdre_baseline.services.search_mcp"], "mcp", mcp_env, ROOT)
    for attempt in range(90):
        try:
            with socket.create_connection(("127.0.0.1", 8001), timeout=2):
                print("[OK] MCP listening at 127.0.0.1:8001")
                break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError(f"MCP did not start; see {LOGS / 'mcp.log'}")

    odr_env = base | {"OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": "0",
                      "PYTHONUNBUFFERED": "1", "LANGSMITH_TRACING": "false"}
    langgraph = shutil.which("langgraph")
    if not langgraph:
        raise RuntimeError("langgraph CLI missing after pip install")
    service([langgraph, "dev", "--allow-blocking", "--host", "127.0.0.1"],
            "langgraph", odr_env, ODR)
    wait_http("http://127.0.0.1:2024/docs", "LangGraph", 120)

    # Write only non-secret runtime settings. These are safe to place in logs.
    from datetime import datetime, timezone
    model_digest = next((m.get("digest") for m in tags.get("models", []) if m.get("name") == MODEL), None)
    if previous.get("model_digest") and previous["model_digest"] != model_digest:
        raise RuntimeError("Ollama model digest changed between runs: stop rather than mix baseline outputs")
    info = {"model": MODEL, "model_digest": model_digest,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "gpu_names": gpu_lines, "gpu_count": gpu_count, "odr_commit": sha,
            "experiment": "MR DRE baseline / official LC ODR graph / ResearcherBench",
            "research_iterations": 3, "max_tool_calls": 6,
            "agent_model": "ollama instruct 4b", "judge_model": "ollama instruct 4b",
            "research_ollama": "http://127.0.0.1:11434",
            "judge_ollama": f"http://127.0.0.1:{11435 if gpu_count == 2 else 11434}",
            "odr_url": "http://127.0.0.1:2024", "mcp_url": "http://127.0.0.1:8001"}
    (WORK / "baseline-runtime.json").write_text(json.dumps(info, indent=2) + "\n")
    print("\n[READY] MR DRE baseline services running. Model:", MODEL)
    print("Judge GPU:", 1 if gpu_count > 1 else 0)
    print("Run: python kaggle/run.py --count 1 --feedback all --extra-eval all")


if __name__ == "__main__":
    main()
