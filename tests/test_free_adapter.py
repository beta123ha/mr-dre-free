"""Tests exercise Mr Dre's ORIGINAL report-generation orchestration without paid APIs.

Mocks replace the external HTTP server only: no fake DRA is shipped for real use.
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import bootstrap  # noqa: F401 (offline mocks for absent dependencies)
from engine.odr import OpenDeepResearchAgent
from utils import get_file_path, load_jsonl


class TestOriginalODRTransport(unittest.TestCase):
    def test_submits_real_graph_messages_and_reads_final_report(self):
        agent = OpenDeepResearchAgent()
        calls = []
        responses = [
            {"thread_id": "threadA"}, {"run_id": "runA"},
            {"status": "running"}, {"status": "success"},
            {"values": {"messages": [], "final_report": "# Evidence\nSupported [source](https://example.org)"}},
        ]

        def http(method, route, payload=None):
            calls.append((method, route, payload))
            return responses.pop(0)

        agent._request = http
        conversation = [
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": "Old cited report"},
            {"role": "user", "content": "Correct section 2"},
        ]
        rid = agent(conversation)
        self.assertEqual(rid, "runA")
        self.assertEqual(calls[1][2]["assistant_id"], "Deep Researcher")
        self.assertEqual(calls[1][2]["input"]["messages"], conversation)
        cfg = calls[1][2]["config"]["configurable"]
        self.assertEqual(cfg["search_api"], "none")
        self.assertIn("web_search", cfg["mcp_config"]["tools"])
        self.assertEqual(cfg["mcp_config"]["url"], "http://127.0.0.1:8001")
        self.assertTrue(cfg["research_model"].startswith("ollama:"))
        self.assertEqual(agent.poll(rid), (False, None))
        done, result = agent.poll(rid)
        self.assertTrue(done)
        self.assertIn("# Evidence", result["report"])

    def test_no_empty_or_failed_report_fabrication(self):
        agent = OpenDeepResearchAgent()
        agent._runs["bad"] = "t"
        agent._request = lambda method, route, payload=None: {"status": "error"}
        with self.assertRaises(RuntimeError):
            agent.poll("bad")
        agent._request = lambda method, route, payload=None: (
            {"status": "success"} if route.endswith("/runs/bad") else
            {"values": {"final_report": ""}}
        )
        with self.assertRaises(RuntimeError):
            agent.poll("bad")


class DummyExternalDRA:
    """Mock of external transport for orchestration tests, not part of benchmark."""
    calls = []
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.results = {}
    def __call__(self, data):
        self.calls.append(data)
        rid = str(len(self.calls))
        self.results[rid] = {"report": "# Revised report\n" + str(len(self.calls)), "citations": [], "usage": {}}
        return rid
    def poll(self, rid):
        return True, self.results[rid]


class TestBenchmarkOrchestration(unittest.TestCase):
    def test_original_one_to_two_turns(self):
        import call_dra
        questions = [{"id": "sample-1", "question": "Research topic", "metadata": {}}]
        DummyExternalDRA.calls = []
        with tempfile.TemporaryDirectory() as temp, patch.object(call_dra, "DRA", DummyExternalDRA):
            for turn, typ in ((1, "init"), (2, "reflection"), (3, "reflection")):
                results, timeout, failed = call_dra.call_dra_api_single_turn(
                    questions, model_name="odr-local", base_output_dir=temp,
                    current_turn=turn, current_turn_type=typ, batch_size=1,
                )
                self.assertFalse(timeout)
                self.assertFalse(failed)
                self.assertEqual(len(results), 1)
            self.assertEqual(len(DummyExternalDRA.calls[0]), 1)
            self.assertEqual(len(DummyExternalDRA.calls[1]), 3)
            self.assertEqual(len(DummyExternalDRA.calls[2]), 5)
            self.assertEqual(DummyExternalDRA.calls[2][1]["role"], "assistant")
            self.assertEqual(DummyExternalDRA.calls[2][3]["role"], "assistant")
            p3 = get_file_path(temp, "odr-local", 3, "reflection", 1)
            result = load_jsonl(p3)[0]
            self.assertIn("report", result["full_response"])
            self.assertIn("feedback_text", result["signal"])

    def test_content_feedback_reaches_original_dra_unchanged(self):
        import call_dra
        question = [{"id": "sample-2", "question": "What is X?", "metadata": {}}]
        feedback = {"sample-2": {"feedback_text": "Add the missing evidence for X.",
                                 "targeted_item": ["Evidence"], "targeted_id": [11]}}
        DummyExternalDRA.calls = []
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(call_dra, "DRA", DummyExternalDRA), \
             patch.object(call_dra, "get_feedbacks", return_value=feedback) as get_feedbacks:
            for turn, typ in ((1, "init"), (2, "content_feedback")):
                results, timed_out, failed = call_dra.call_dra_api_single_turn(
                    question, model_name="odr-local", base_output_dir=temp,
                    current_turn=turn, current_turn_type=typ, batch_size=1)
                self.assertEqual((len(results), timed_out, failed), (1, [], []))
            self.assertEqual(DummyExternalDRA.calls[1][-1]["content"], feedback["sample-2"]["feedback_text"])
            self.assertEqual(len(DummyExternalDRA.calls[1]), 3)
            self.assertTrue(get_feedbacks.called)
            output = get_file_path(temp, "odr-local", 2, "content_feedback", 1)
            self.assertEqual(load_jsonl(output)[0]["signal"]["targeted_id"], [11])

    def test_format_feedback_reaches_original_dra_unchanged(self):
        import call_dra
        question = [{"id": "sample-3", "question": "What is Y?", "metadata": {}}]
        feedback = {"sample-3": {"feedback_text": "Add meaningful headings to this report.",
                                 "targeted_item": [], "targeted_id": []}}
        DummyExternalDRA.calls = []
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(call_dra, "DRA", DummyExternalDRA), \
             patch.object(call_dra, "get_feedbacks", return_value=feedback):
            for turn, typ in ((1, "init"), (2, "format_feedback")):
                results, timed_out, failed = call_dra.call_dra_api_single_turn(
                    question, model_name="odr-local", base_output_dir=temp,
                    current_turn=turn, current_turn_type=typ, batch_size=1)
                self.assertEqual((len(results), timed_out, failed), (1, [], []))
            self.assertEqual(DummyExternalDRA.calls[1][-1]["content"], feedback["sample-3"]["feedback_text"])
            self.assertEqual(len(DummyExternalDRA.calls[1]), 3)

    def test_input_datasets_are_present_and_unmodified(self):
        base = Path(__file__).parents[1]
        for name in ("ResearchRubrics", "RigorousBench", "ResearcherBench"):
            entries = load_jsonl(str(base / "data" / (name + ".jsonl")))
            self.assertGreater(len(entries), 50)
            self.assertIn("checklist", entries[0])


class TestFreeServiceSubstitution(unittest.TestCase):
    def test_paid_model_name_redirected_but_prompt_unchanged(self):
        from engine.oai import GPT
        with patch.dict(os.environ, {"MRDRE_LLM_BACKEND": "ollama", "MRDRE_LOCAL_MODEL": "qwen3:4b"}):
            g = GPT("gpt-4.1-mini", system_prompt="judge this report")
            self.assertEqual(g.model_name, "ollama:qwen3:4b")
            self.assertEqual(g.system_prompt, "judge this report")

    def test_direct_crawler_never_requires_jina_key(self):
        from dr_eval.crawler.jina import fetch_webpage_content_jina
        with patch.dict(os.environ, {"MRDRE_CRAWLER": "direct"}), patch(
            "dr_eval.crawler.direct.fetch_webpage_content_direct",
            return_value={"url": "https://example.org", "content": "Evidence", "success": True},
        ):
            result = fetch_webpage_content_jina("https://example.org")
        self.assertEqual(result["content"], "Evidence")

    def test_prompt_arrays_are_not_forced_to_json_objects(self):
        from engine.oai import _cached_gpt_call
        class FakeMessage:
            content = '[{"claim": "Fact", "url": []}]'
        class FakeChoice:
            message = FakeMessage()
        class FakeResp:
            choices = [FakeChoice()]
        class FakeCompletion:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeResp()
        client = type('FakeClient', (), {})()
        client.chat = type('FakeChat', (), {})()
        client.chat.completions = FakeCompletion()
        with patch("engine.oai.openai.OpenAI", return_value=client):
            raw = _cached_gpt_call("ollama:mock-model@unit-test", "Return only the JSON list", "Extract claims")
        self.assertNotIn("response_format", client.chat.completions.kwargs)
        self.assertEqual(json.loads(raw)[0]["claim"], "Fact")

    def test_public_url_does_not_accept_localhost(self):
        from local.free_web import _validate_public_url
        with self.assertRaises(ValueError):
            _validate_public_url("http://localhost:11434/api/tags")
        with self.assertRaises(ValueError):
            _validate_public_url("file:///etc/passwd")


class TestCompleteScoringPipeline(unittest.TestCase):
    """Offline synthetic data. This never asserts model quality, only plumbing."""

    def test_real_mrdre_checklist_citation_presentation_file_formats(self):
        from evaluate_report import evaluate_reports
        from dr_eval import eval_prompts

        class DeterministicTestJudge:
            def __init__(self, model_name, system_prompt=""):
                self.model_name = model_name
                self.sp = system_prompt
            def __call__(self, prompt):
                if self.sp == eval_prompts.CLAIM_EXTRACTION_SYSTEM_PROMPT:
                    return json.dumps([{"claim": "The sky is blue", "url": ["https://example.org"]}])
                if self.sp == eval_prompts.SUPPORTED_JUDGE_SYSTEM_PROMPT:
                    return json.dumps({"result": "supported"})
                if self.sp == eval_prompts.CONTENT_SUMMARIZATION_SYSTEM_PROMPT:
                    return "The sky is blue."
                if self.sp == eval_prompts.CHECKLIST_EVAL_SYSTEM_PROMPT:
                    return json.dumps({"score": 1.0, "justification": "Mentioned"})
                if self.sp == eval_prompts.RUBRIC_EVAL_SYSTEM_PROMPT:
                    return json.dumps({"score": 1, "justification": "Well formatted"})
                raise ValueError("Unexpected judge prompt")

        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            generation = temp / 'gen_turn1.jsonl'
            dataset = temp / 'testdataset.jsonl'
            generation.write_text(json.dumps({"id": "q1", "question": "What is the sky color?",
                "full_response": {"report": "# Report\n\nThe sky is blue [1].\n\n## References\n[1] https://example.org"},
                "signal": {}}, ensure_ascii=False) + '\n')
            dataset.write_text(json.dumps({"id": "q1", "question": "What is the sky color?",
                "checklist": [{"id": 1, "item": "Color of sky", "weight": 1}], "metadata": {}}, ensure_ascii=False) + '\n')
            with patch('evaluate_report.GPT', DeterministicTestJudge), patch(
                'dr_eval.citation_eval.fetch_webpage_content_jina',
                return_value={"url": "https://example.org", "content": "The sky is blue.", "success": True}
            ):
                evaluate_reports(str(generation), str(dataset), str(temp), eval_mode='all',
                    evaluator_model_name='test', claim_extraction_model_name='test',
                    supported_judge_model_name='test', webpage_summarization_model_name='test',
                    rubric_judge_model_name='test')
            coverage = load_jsonl(str(temp / 'checklist_turn1.jsonl'))[0]
            citation = load_jsonl(str(temp / 'citation_turn1.jsonl'))[0]
            presentation = load_jsonl(str(temp / 'rubric_turn1.jsonl'))[0]
            self.assertEqual(coverage['coverage_score'], 1.0)
            self.assertEqual(citation['faithfulness_score'], 1.0)
            self.assertEqual(citation['groundedness_score'], 1.0)
            self.assertEqual(presentation['presentation'], 1.0)

    def test_paper_scoring_field_used_for_presentation_delta(self):
        from evaluate_change import rubric_stats
        actual = rubric_stats([{'id': 1, 'presentation': .8}], [{'id':1, 'presentation': .9}])
        self.assertAlmostEqual(actual, .1)


if __name__ == "__main__":
    unittest.main()
