"""Portable local/WSL/Kaggle/Colab bootstrap. Keeps services running across questions.

Uses the official LangChain ODR graph unchanged and two independent Ollama
servers only when two GPUs exist. Existing servers are NEVER killed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from .resources import gpus

ROOT = Path(__file__).resolve().parents[3]


def command(args: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def online(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as resp:
            return 200 <= resp.status < 500
    except (OSError, URLError, TimeoutError):
        return False


def wait_online(url: str, label: str, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if online(url):
            print(f"[READY] {label}: {url}", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"{label} did not start: {url}")


def service(name: str, args: list[str], env: dict, workdir: Path, logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{name}.log").open("ab", buffering=0) as log:
        proc = subprocess.Popen(args, env=env, cwd=workdir, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    (logs / f"{name}.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"[STARTED] {name}: PID {proc.pid}", flush=True)


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def choose_profile(requested: str) -> str:
    if requested != "auto":
        return requested
    if Path("/kaggle/working").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start MR DRE services; no GPU quota is used by the runner itself")
    ap.add_argument("--profile", choices=["auto", "local", "kaggle", "colab"], default="auto")
    ap.add_argument("--workdir", type=Path)
    ap.add_argument("--model", default=os.getenv("MRDRE_BASELINE_MODEL", "qwen3:4b-instruct"))
    ap.add_argument("--odr-ref", default=os.getenv("ODR_REF", ""), help="Pin upstream ODR commit/tag for reproducibility")
    ap.add_argument("--install", dest="install", action="store_true", default=True)
    ap.add_argument("--no-install", dest="install", action="store_false")
    ap.add_argument("--allow-cpu", action="store_true", help="Run serially without NVIDIA GPU (very slow)")
    ap.add_argument("--judge-backend", choices=["ollama", "vllm"], default="ollama")
    ap.add_argument("--vllm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--vllm-max-model-len", type=int, default=16384)
    args = ap.parse_args(argv)
    if sys.version_info < (3, 11):
        ap.error("Use Python 3.11+ in WSL, Kaggle or Colab")
    profile = choose_profile(args.profile)
    work = (args.workdir or (Path("/kaggle/working") if profile == "kaggle" else
                            Path("/content") if profile == "colab" else ROOT)).resolve()
    logs, odr, model_dir = work / "baseline-logs", work / "open_deep_research", work / "ollama-models"
    for path in (work, logs, model_dir):
        path.mkdir(parents=True, exist_ok=True)
    gpu_devices = gpus()
    if not gpu_devices and not args.allow_cpu:
        raise RuntimeError("No NVIDIA GPU found. Ensure WSL NVIDIA driver or enable Kaggle/Colab GPU. ")
    count = min(2, len(gpu_devices)) or 1
    if args.judge_backend == "vllm" and len(gpu_devices) < 2:
        raise RuntimeError("vLLM judge is experimental and requires a separate GPU; keep Ollama on a single GPU")
    mem = gpu_devices[0]["total_mb"] if gpu_devices else 0
    # Conservative default prevents multi-context VRAM exhaustion on 6 GB.
    context = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192" if mem and mem < 8500 else "16384"))
    parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "1" if mem < 12000 else "2"))
    print(f"[PROFILE] {profile}; GPUs={gpu_devices}; context={context}; parallel={parallel}", flush=True)

    if args.install:
        command([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])
    previous_file = work / "baseline-runtime.json"
    previous = json.loads(previous_file.read_text(encoding="utf-8")) if previous_file.exists() else {}
    if previous.get("ollama_context_length"):
        saved_context = int(previous["ollama_context_length"])
        if os.getenv("OLLAMA_CONTEXT_LENGTH") and context != saved_context:
            raise RuntimeError("Resuming with a changed context length would mix benchmark configurations")
        context = saved_context
        base["OLLAMA_CONTEXT_LENGTH"] = str(context)
    odr_ref = args.odr_ref or previous.get("odr_commit", "")
    if not odr.exists():
        command(["git", "clone", "--depth", "1", "https://github.com/langchain-ai/open_deep_research.git", str(odr)])
    if odr_ref:
        current = subprocess.check_output(["git", "-C", str(odr), "rev-parse", "HEAD"], text=True).strip()
        if current != odr_ref:
            command(["git", "-C", str(odr), "fetch", "--depth", "1", "origin", odr_ref])
            command(["git", "-C", str(odr), "checkout", "--detach", "FETCH_HEAD"])
    sha = subprocess.check_output(["git", "-C", str(odr), "rev-parse", "HEAD"], text=True).strip()
    if args.install:
        command([sys.executable, "-m", "pip", "install", "-q", "-e", str(odr), "langgraph-cli[inmem]", "langchain-ollama"])
    (odr / ".env").touch(exist_ok=True)
    command([sys.executable, str(ROOT / "scripts" / "configure_odr_local.py"), str(odr)])
    cfg = odr / "langgraph.json"
    graph_json = json.loads(cfg.read_text(encoding="utf-8"))
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if graph_json.get("python_version") != python_version:
        graph_json["python_version"] = python_version
        cfg.write_text(json.dumps(graph_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ollama = shutil.which("ollama")
    if not ollama and args.install and os.name == "posix":
        command(["bash", "-o", "pipefail", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        ollama = shutil.which("ollama")
    if not ollama:
        raise RuntimeError("Ollama not installed. Install it or run --install on Linux/WSL")
    base = os.environ.copy()
    base.update({"OLLAMA_MODELS": str(model_dir), "OLLAMA_FLASH_ATTENTION": "1",
                 "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_NUM_PARALLEL": str(parallel),
                 "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_CONTEXT_LENGTH": str(context),
                 "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_MAX_QUEUE": "32", "PYTHONUNBUFFERED": "1"})
    server_count = 1 if args.judge_backend == "vllm" else count
    for i in range(server_count):
        port = 11434 + i
        url = f"http://127.0.0.1:{port}/api/tags"
        if not online(url):
            env = {**base, "OLLAMA_HOST": f"127.0.0.1:{port}", "CUDA_VISIBLE_DEVICES": str(i)}
            service(f"ollama_gpu{i}", [ollama, "serve"], env, work, logs)
            wait_online(url, f"Ollama {i}", 90)
        else:
            print(f"[REUSE] Ollama {i} on {port}; existing server settings cannot be changed in this process")
        pull_env = {**base, "OLLAMA_HOST": f"127.0.0.1:{port}"}
        with urlopen(url, timeout=20) as response:
            tags = json.load(response)
        if not any(m.get("name") == args.model for m in tags.get("models", [])):
            command([ollama, "pull", args.model], env=pull_env)

    if args.judge_backend == "vllm":
        if not online("http://127.0.0.1:8002/v1/models"):
            vllm = shutil.which("vllm")
            if not vllm:
                raise RuntimeError("vLLM not installed. Install vllm separately on the 2-GPU runtime")
            vllm_env = {**base, "CUDA_VISIBLE_DEVICES": "1"}
            service("vllm_judge", [vllm, "serve", args.vllm_model,
                                   "--host", "127.0.0.1", "--port", "8002",
                                   "--max-model-len", str(args.vllm_max_model_len),
                                   "--enable-prefix-caching"], vllm_env, work, logs)
            wait_online("http://127.0.0.1:8002/v1/models", "vLLM judge", 240)

    if not port_open("127.0.0.1", 8001):
        service("mcp", [sys.executable, "-m", "mrdre_baseline.services.search_mcp"], base, ROOT, logs)
        for _ in range(90):
            if port_open("127.0.0.1", 8001):
                break
            time.sleep(1)
        else:
            raise RuntimeError("MCP did not start; see baseline-logs/mcp.log")
    if not online("http://127.0.0.1:2024/docs"):
        cli = shutil.which("langgraph")
        if not cli:
            raise RuntimeError("langgraph CLI absent; run with --install")
        graph_env = {**base, "OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": "0",
                     "LANGSMITH_TRACING": "false"}
        service("langgraph", [cli, "dev", "--allow-blocking", "--host", "127.0.0.1"], graph_env, odr, logs)
        wait_online("http://127.0.0.1:2024/docs", "LangGraph", 120)

    with urlopen("http://127.0.0.1:11434/api/tags", timeout=20) as response:
        tags = json.load(response)
    model_digest = next((m.get("digest") for m in tags.get("models", []) if m.get("name") == args.model), None)
    info = {"profile": profile, "model": args.model, "model_digest": model_digest,
            "odr_commit": sha, "gpu_count": count, "gpu_names": gpu_devices,
            "judge_backend": args.judge_backend,
            "judge_model": args.vllm_model if args.judge_backend == "vllm" else args.model,
            "judge_model_fingerprint": args.vllm_model if args.judge_backend == "vllm" else model_digest,
            "judge_base_url": "http://127.0.0.1:8002/v1" if args.judge_backend == "vllm" else f"http://127.0.0.1:{11435 if count > 1 else 11434}",
            "ollama_num_parallel": parallel, "ollama_context_length": context,
            "research_ollama": "http://127.0.0.1:11434",
            "judge_ollama": f"http://127.0.0.1:{11435 if count > 1 else 11434}",
            "odr_url": "http://127.0.0.1:2024", "mcp_url": "http://127.0.0.1:8001",
            "started_at": datetime.now(timezone.utc).isoformat()}
    path = work / "baseline-runtime.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        for key in ("model", "model_digest", "odr_commit"):
            if prior.get(key) and prior[key] != info.get(key):
                raise RuntimeError(f"Existing run model/revision differs at {key}; use a new workdir")
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[READY] Services available; runtime settings in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
