"""Offline tests for the direct transport of the official ODR graph."""
from __future__ import annotations

import sys
import time
from types import ModuleType

import pytest

from mrdre_baseline.agents.open_deep_research import OpenDeepResearchAgent


def _fake_graph(monkeypatch, *, report="mocked research report", error=None):
    calls = []
    fake_package = ModuleType("open_deep_research")
    fake_package.__path__ = []
    fake_module = ModuleType("open_deep_research.deep_researcher")

    class FakeGraph:
        async def ainvoke(self, values, config):
            calls.append((values, config))
            if error:
                raise error
            return {"final_report": report}

    fake_module.deep_researcher = FakeGraph()
    monkeypatch.setitem(sys.modules, "open_deep_research", fake_package)
    monkeypatch.setitem(sys.modules, "open_deep_research.deep_researcher", fake_module)
    return calls


def _finish(agent, request_id):
    for _ in range(200):
        done, result = agent.poll(request_id)
        if done:
            return result
        time.sleep(0.005)
    pytest.fail("Direct ODR did not finish")


def test_direct_transport_invokes_original_graph_and_preserves_config(monkeypatch):
    monkeypatch.setenv("ODR_TRANSPORT", "direct")
    monkeypatch.setenv("ODR_LOCAL_MODEL", "qwen3:4b-instruct")
    calls = _fake_graph(monkeypatch)
    agent = OpenDeepResearchAgent()
    messages = [{"role": "user", "content": "test"}]
    request_id = agent(messages)
    result = _finish(agent, request_id)
    assert result["report"] == "mocked research report"
    assert result["metadata"]["transport"] == "direct"
    assert calls[0][0] == {"messages": messages}
    assert calls[0][1]["configurable"]["research_model"] == "ollama:qwen3:4b-instruct"
    assert calls[0][1]["configurable"]["mcp_config"]["tools"] == ["web_search", "read_webpage"]
    assert agent.get_response(request_id)[0] == "success"
    agent._executor.shutdown(wait=True)


def test_direct_transport_propagates_graph_error(monkeypatch):
    monkeypatch.setenv("ODR_TRANSPORT", "direct")
    _fake_graph(monkeypatch, error=ValueError("tool error"))
    agent = OpenDeepResearchAgent()
    rid = agent("test")
    with pytest.raises(RuntimeError, match="tool error"):
        _finish(agent, rid)
    assert agent.get_response(rid)[0] == "error"
    agent._executor.shutdown(wait=True)


def test_direct_transport_rejects_empty_report(monkeypatch):
    monkeypatch.setenv("ODR_TRANSPORT", "direct")
    _fake_graph(monkeypatch, report=" ")
    agent = OpenDeepResearchAgent()
    rid = agent("test")
    with pytest.raises(RuntimeError, match="no final_report"):
        _finish(agent, rid)
    agent._executor.shutdown(wait=True)
