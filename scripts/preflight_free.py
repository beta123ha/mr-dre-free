"""Quick connectivity and configuration check; never runs an expensive research task."""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


def check(label, url, verifier):
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        result = verifier(response)
        if not result:
            raise RuntimeError("Unexpected response")
        print("[OK]", label)
        return True
    except Exception as exc:
        print("[FAIL]", label, "-", exc)
        return False


def main():
    model = os.getenv("MRDRE_LOCAL_MODEL", "qwen3:4b")
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    passed = [check("Ollama + correct local model", base + "/api/tags",
                    lambda r: any(m.get("name", "") == model or m.get("model", "") == model
                                  for m in r.json().get("models", []))) ]
    odr_url = os.getenv("ODR_URL", "http://127.0.0.1:2024").rstrip("/")
    passed.append(check("LangGraph ODR server", odr_url + "/docs", lambda r: bool(r.content)))
    mcp = os.getenv("ODR_MCP_URL", "http://127.0.0.1:8001").rstrip("/") + "/mcp"
    # MCP GET returns 405 or 406 for clients lacking required Accept headers;
    # connectivity of service is verified without claiming protocol integration.
    try:
        r = requests.get(mcp, timeout=8, headers={"Accept": "text/event-stream"})
        if r.status_code in (200, 400, 405, 406):
            print("[OK] MCP server reachable (protocol handshake not tested)")
            passed.append(True)
        else:
            print("[FAIL] MCP server returned", r.status_code)
            passed.append(False)
    except Exception as exc:
        print("[FAIL] MCP server", exc)
        passed.append(False)
    if not all(passed):
        sys.exit(1)
    print("All endpoints reachable. NEXT: run 1 REAL ODR question and inspect its citations.")


if __name__ == "__main__":
    main()
