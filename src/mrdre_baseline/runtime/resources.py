"""Conservative adaptive resource admission for local inference servers.

This module observes physical free memory; it does not pretend to reallocate
VRAM or increase throughput merely by increasing the number of Python threads.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


def gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=8, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            devices.append({"index": int(fields[0]), "name": fields[1],
                            "total_mb": int(fields[2]), "free_mb": int(fields[3]),
                            "utilization": int(fields[4])})
        except ValueError:
            continue
    return devices


def cpu_memory() -> dict[str, Any]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"cpu_percent": psutil.cpu_percent(interval=None),
                "ram_total_mb": round(vm.total / 1048576),
                "ram_available_mb": round(vm.available / 1048576)}
    except ImportError:
        return {}


def snapshot() -> dict[str, Any]:
    return {"gpus": gpus(), **cpu_memory()}


@dataclass(frozen=True)
class ResourcePolicy:
    gpu_count: int
    parallel_capacity: int
    judge_gpu: int
    max_workers: int

    @classmethod
    def detect(cls, runtime: dict | None = None) -> "ResourcePolicy":
        devices = gpus()
        runtime = runtime or {}
        count = int(runtime.get("gpu_count", len(devices)))
        if count < 1:
            count = 1  # single CPU/Ollama endpoint, still serialize
        parallel = max(1, int(runtime.get("ollama_num_parallel", os.getenv("OLLAMA_NUM_PARALLEL", "1"))))
        judge_gpu = 1 if count > 1 else 0
        max_workers = max(1, min(4, parallel))
        return cls(count, parallel, judge_gpu, max_workers)

    def judge_workers(self) -> int:
        """Set once per task using *currently* free VRAM; stay within Ollama capacity."""
        if self.max_workers == 1:
            return 1
        devices = gpus()
        gpu = next((d for d in devices if d["index"] == self.judge_gpu), None)
        if gpu is None:
            return 1
        # A model normally remains resident in VRAM. Additional concurrent
        # 8k-24k contexts require substantial headroom beyond model weights.
        return min(self.max_workers, 2) if gpu["free_mb"] >= 4500 else 1


def judge_env(common: dict[str, str], runtime: dict, policy: ResourcePolicy) -> dict[str, str]:
    env = dict(common)
    env["OLLAMA_BASE_URL"] = runtime.get("judge_ollama", common.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    backend = runtime.get("judge_backend", "ollama")
    env["MRDRE_LLM_BACKEND"] = backend
    env["MRDRE_LOCAL_MODEL"] = runtime.get("judge_model", runtime.get("model", "qwen3:4b-instruct"))
    env["MRDRE_MODEL_FINGERPRINT"] = runtime.get("judge_model_fingerprint", runtime.get("model_digest") or "")
    if backend == "vllm":
        env["VLLM_BASE_URL"] = runtime.get("judge_base_url", "http://127.0.0.1:8002/v1")
    workers = (4 if backend == "vllm" else policy.judge_workers())
    env["MRDRE_EVAL_WORKERS"] = str(workers)
    env["MRDRE_SUMMARY_WORKERS"] = str(workers)
    return env
