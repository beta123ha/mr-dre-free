import json
import uuid
from typing import Any, List, Optional, Tuple, Union

from ddgs import DDGS
from openai import OpenAI

from . import DeepResearchAgentBase


class LocalOllamaDeepResearchAgent(DeepResearchAgentBase):
    def __init__(
        self,
        model: str = "local-qwen3.5:4b",
        base_url: str = "http://127.0.0.1:11434/v1",
        **kwargs,
    ):
        self.model = model.removeprefix("local-")
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=3600,
        )
        self._responses = {}

    def _get_user_messages(self, messages: List[dict]) -> List[str]:
        return [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user" and message.get("content")
        ]

    def _make_search_queries(self, messages: List[dict]) -> List[str]:
        user_messages = self._get_user_messages(messages)

        if not user_messages:
            return []

        original_question = user_messages[0]
        latest_request = user_messages[-1]

        prompt = f"""
Generate exactly 3 concise web search queries for researching the task below.

Original question:
{original_question}

Latest user request:
{latest_request}

Return only a JSON array of strings.
Example:
["query one", "query two", "query three"]
"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0,
                extra_body={"think": False},
            )

            queries = json.loads(response.output_text)

            if isinstance(queries, list):
                queries = [
                    str(query).strip()
                    for query in queries
                    if str(query).strip()
                ]
                if queries:
                    return queries[:3]
        except Exception:
            pass

        return [f"{original_question} {latest_request}".strip()]

    def _search_web(self, messages: List[dict]) -> List[dict]:
        queries = self._make_search_queries(messages)

        results = []
        seen_urls = set()

        for query in queries:
            try:
                search_results = DDGS().text(
                    query,
                    max_results=3,
                )

                for result in search_results:
                    url = result.get("href")

                    if not url or url in seen_urls:
                        continue

                    seen_urls.add(url)

                    results.append(
                        {
                            "title": result.get("title", ""),
                            "url": url,
                            "snippet": result.get("body", ""),
                        }
                    )
            except Exception:
                continue

        return results[:8]

    def _build_source_context(self, search_results: List[dict]) -> str:
        if not search_results:
            return "No web search results were available."

        parts = []

        for index, result in enumerate(search_results, 1):
            parts.append(
                f"""[S{index}]
Title: {result["title"]}
URL: {result["url"]}
Snippet: {result["snippet"]}"""
            )

        return "\n\n".join(parts)

    def __call__(
        self,
        input_data: Union[str, List[dict]],
        **kwargs,
    ) -> str:
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = list(input_data)

        search_results = self._search_web(messages)
        source_context = self._build_source_context(search_results)

        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "You are a research report writing agent. "
                    "Write a clear and comprehensive research report that directly "
                    "answers the user's request. "
                    "When revising an existing report, follow the latest user feedback "
                    "while preserving useful existing content. "
                    "Use the provided web research sources when relevant. "
                    "Cite factual claims using Markdown links with the exact source URL, "
                    "for example [source](https://example.com). "
                    "Do not invent URLs or citations. "
                    "Return only the report."
                ),
            },
        )

        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Web research results:\n\n"
                    f"{source_context}"
                ),
            },
        )

        response = self.client.responses.create(
            model=self.model,
            input=messages,
            temperature=0,
            extra_body={"think": False},
        )

        request_id = str(uuid.uuid4())
        usage = getattr(response, "usage", None)

        result = {
            "resp_id": request_id,
            "report": response.output_text,
            "citations": [
                result["url"]
                for result in search_results
            ],
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "cached_input_tokens": 0,
                "output_tokens": getattr(usage, "output_tokens", None),
                "reasoning_tokens": 0,
                "total_tokens": getattr(usage, "total_tokens", None),
                "tool_call_count": len(search_results),
                "cost": {},
            },
            "metadata": {
                "provider": "ollama",
                "model": self.model,
                "search_results": search_results,
            },
        }

        self._responses[request_id] = result
        return request_id

    def poll(self, request_id: str) -> Tuple[bool, Optional[dict]]:
        result = self._responses.get(request_id)

        if result is None:
            raise RuntimeError(f"Unknown request id: {request_id}")

        return True, result

    def wait_for_completion(self, request_id: str) -> dict:
        result = self._responses.get(request_id)

        if result is None:
            raise RuntimeError(f"Unknown request id: {request_id}")

        return result

    def get_response(self, request_id: str) -> Tuple[str, Any]:
        result = self._responses.get(request_id)

        if result is None:
            raise RuntimeError(f"Unknown request id: {request_id}")

        return "completed", result