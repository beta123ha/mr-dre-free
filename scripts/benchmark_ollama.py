"""Short, optional local inference speed check; does not run Mr Dre."""
import concurrent.futures
import json
import os
import time
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()
base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
model = os.getenv("MRDRE_LOCAL_MODEL", "qwen3:4b")

def infer(_=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain in three short sentences what a reliable research citation is."}],
        "stream": False,
        "options": {"num_predict": 120},
        "keep_alive": "30m",
    }
    request = Request(base + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result.get("eval_count", 0)

def measure(parallel):
    start = time.monotonic()
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            counts = list(pool.map(infer, range(2)))
    else:
        counts = [infer(), infer()]
    elapsed = time.monotonic() - start
    return elapsed, sum(counts) / max(elapsed, 0.001)

print("Model:", model, "Server:", base)
print("Warmup...")
infer()
a, at = measure(False)
b, bt = measure(True)
print(f"2 sequential: {a:.1f}s, {at:.1f} output tokens/s")
print(f"2 concurrent: {b:.1f}s, {bt:.1f} output tokens/s")
print("This short probe is not a full context benchmark. Check 'ollama ps' for GPU residency.")
print("Use 2 workers only if it is actually faster and memory fits.")
