import uuid
from typing import Any, List, Optional, Tuple, Union

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

    def __call__(
        self,
        input_data: Union[str, List[dict]],
        **kwargs,
    ) -> str:
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = list(input_data)

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
                    "Return only the report."
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
            "citations": [],
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "cached_input_tokens": 0,
                "output_tokens": getattr(usage, "output_tokens", None),
                "reasoning_tokens": 0,
                "total_tokens": getattr(usage, "total_tokens", None),
                "tool_call_count": 0,
                "cost": {},
            },
            "metadata": {
                "provider": "ollama",
                "model": self.model,
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