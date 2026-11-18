"""Mr Dre adapter for the *upstream* LangChain Open Deep Research LangGraph server.

The research graph is NOT reimplemented here. This is a thin transport adapter
for the officially evaluated DRA family (LC ODR) and the original Mr Dre DRA
interface. Results are never synthesized on failure.
"""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import os
import time
import uuid
from typing import Any, List, Optional, Tuple, Union

import requests

from . import DeepResearchAgentBase


class OpenDeepResearchAgent(DeepResearchAgentBase):
    def __init__(self, model: str = "odr-local", **kwargs):
        self.model = model
        self.url = os.getenv("ODR_URL", "http://127.0.0.1:2024").rstrip("/")
        # Name is EXACTLY what is in the upstream langgraph.json. Override only
        # when the user intentionally registered a differently named graph.
        self.graph = os.getenv("ODR_GRAPH_ID", "Deep Researcher")
        self.local_model = os.getenv("ODR_LOCAL_MODEL", "qwen3:4b-instruct")
        # Upstream ODR's load_mcp_tools() APPENDS '/mcp' to the supplied URL.
        # This must be the base address (without the suffix), or no tools load.
        self.mcp_url = os.getenv("ODR_MCP_URL", "http://127.0.0.1:8001").rstrip("/")
        if self.mcp_url.endswith("/mcp"):
            raise ValueError("ODR_MCP_URL must be the MCP server BASE URL without /mcp: " + self.mcp_url)
        self.timeout = float(os.getenv("ODR_HTTP_TIMEOUT", "60"))
        self.transport = os.getenv("ODR_TRANSPORT", "server").strip().lower()
        if self.transport not in ("server", "direct"):
            raise ValueError(f"Unsupported ODR_TRANSPORT: {self.transport}")
        self._runs: dict[str, str] = {}
        self._futures: dict[str, Future] = {}
        self._executor = None
        self._session = requests.Session()

    def _request(self, method: str, route: str, payload: dict = None) -> dict:
        try:
            response = self._session.request(
                method, self.url + route, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Local ODR server {method} {route} failed: {exc}") from exc

    def _config(self) -> dict:
        # Only the service/model settings differ from LangChain's graph defaults.
        configurable = {
            "allow_clarification": False,  # batch runs cannot answer questions
            "summarization_model": f"ollama:{self.local_model}",
            "research_model": f"ollama:{self.local_model}",
            "compression_model": f"ollama:{self.local_model}",
            "final_report_model": f"ollama:{self.local_model}",
            "search_api": "none",
            "mcp_config": {
                "url": self.mcp_url,
                "tools": ["web_search", "read_webpage"],
                "auth_required": False,
            },
            "mcp_prompt": (
                "Research the user's question using web_search and read_webpage. "
                "Use retrieved pages as evidence and provide inline citations "
                "with URLs. Do not invent sources."
            ),
            "max_concurrent_research_units": int(os.getenv("ODR_RESEARCH_CONCURRENCY", "1")),
            "max_researcher_iterations": int(os.getenv("ODR_RESEARCH_ITERATIONS", "3")),
            "max_react_tool_calls": int(os.getenv("ODR_MAX_TOOL_CALLS", "6")),
            "final_report_model_max_tokens": int(os.getenv("ODR_FINAL_MAX_TOKENS", "6144")),
        }
        return {"configurable": configurable}

    def __call__(self, input_data: Union[str, List[dict]], **kwargs) -> str:
        messages = ([{"role": "user", "content": input_data}]
                    if isinstance(input_data, str) else input_data)
        if not isinstance(messages, list) or not messages:
            raise ValueError("Expected a non-empty conversation")
        # Mr Dre reconstructs and supplies complete history each turn. We must
        # start a NEW LangGraph thread rather than append the whole history to
        # an existing thread (which would silently duplicate old messages).
        if self.transport == "direct":
            request_id = str(uuid.uuid4())
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="odr-direct")
            self._futures[request_id] = self._executor.submit(
                self._invoke_direct, messages, self._config()
            )
            return request_id
        thread = self._request("POST", "/threads", {})
        thread_id = thread["thread_id"]
        run = self._request("POST", f"/threads/{thread_id}/runs", {
            "assistant_id": self.graph,
            "input": {"messages": messages},
            "config": self._config(),
        })
        run_id = run["run_id"]
        self._runs[run_id] = thread_id
        return run_id

    @staticmethod
    def _invoke_direct(messages: list[dict], config: dict) -> str:
        # Run the UNMODIFIED official LangChain ODR graph in-process. This
        # avoids the separate LangGraph development server on cloud notebooks.
        # The exact same messages/configuration are passed to the same graph.
        from open_deep_research.deep_researcher import deep_researcher
        result = asyncio.run(deep_researcher.ainvoke({"messages": messages}, config=config))
        report = result.get("final_report") if isinstance(result, dict) else None
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError(
                "Official ODR graph produced no final_report; "
                f"result keys: {list(result) if isinstance(result, dict) else type(result).__name__}"
            )
        return report

    def _run(self, request_id: str) -> dict:
        if request_id not in self._runs:
            raise ValueError(f"Unknown ODR run id {request_id}")
        return self._request("GET", f"/threads/{self._runs[request_id]}/runs/{request_id}")

    def poll(self, request_id: str) -> Tuple[bool, Optional[dict]]:
        if self.transport == "direct":
            future = self._futures.get(request_id)
            if future is None:
                raise ValueError(f"Unknown ODR request id {request_id}")
            if not future.done():
                return False, None
            try:
                report = future.result()
            except Exception as exc:
                raise RuntimeError(f"Direct official ODR graph failed: {exc}") from exc
            return True, {
                "resp_id": request_id,
                "report": report,
                "citations": [],
                "usage": {"input_tokens": None, "output_tokens": None,
                          "total_tokens": None, "cost": {}},
                "metadata": {"agent": "langchain-open-deep-research",
                             "transport": "direct", "thread_id": request_id,
                             "local_model": self.local_model,
                             "search_provider": "local_mcp_ddg"},
            }
        run = self._run(request_id)
        status = run.get("status", "")
        if status in ("pending", "running"):
            return False, None
        if status not in ("success", "completed"):
            raise RuntimeError(f"ODR graph finished unsuccessfully: {status}: {run}")
        state = self._request("GET", f"/threads/{self._runs[request_id]}/state")
        values = state.get("values") or {}
        report = values.get("final_report")
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError(
                "ODR returned no final_report; inspect local LangGraph execution logs. "
                f"State keys: {list(values)}"
            )
        return True, {
            "resp_id": request_id,
            "report": report,
            "citations": [],   # citations remain inside the DRA's original report
            "usage": {"input_tokens": None, "output_tokens": None,
                      "total_tokens": None, "cost": {}},
            "metadata": {"agent": "langchain-open-deep-research", "thread_id": self._runs[request_id],
                         "local_model": self.local_model, "search_provider": "local_mcp_ddg"},
        }

    def wait_for_completion(self, request_id: str) -> dict:
        while True:
            finished, result = self.poll(request_id)
            if finished:
                return result
            time.sleep(float(os.getenv("ODR_POLL_INTERVAL", "5")))

    def get_response(self, request_id: str) -> Tuple[str, Any]:
        if self.transport == "direct":
            future = self._futures.get(request_id)
            if future is None:
                raise ValueError(f"Unknown ODR request id {request_id}")
            if not future.done():
                return "running", None
            if future.exception() is not None:
                return "error", {"error": str(future.exception())}
            return "success", {"report": future.result()}
        run = self._run(request_id)
        return run.get("status", "unknown"), run
