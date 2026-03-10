import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER_ROOT) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER_ROOT))

from llm.api import TaskResult
from persona.prompt_template import gpt_structure


class GPTStructureTests(unittest.TestCase):
  def test_chatgpt_single_request_uses_system_and_user_messages(self):
    with patch.object(gpt_structure, "temp_sleep", return_value=None):
      with patch.object(gpt_structure, "generate_text", return_value=SimpleNamespace(value="hello")) as mock_generate_text:
        result = gpt_structure.ChatGPT_single_request("Prompt body")
    self.assertEqual(result, "hello")
    messages = mock_generate_text.call_args.args[1]
    self.assertEqual([message["role"] for message in messages], ["system", "user"])
    self.assertIn("Return plain text only.", messages[0]["content"])
    self.assertEqual(messages[1]["content"], "Prompt body")

  def test_gpt_request_preserves_stop_and_uses_system_and_user_messages(self):
    with patch.object(gpt_structure, "temp_sleep", return_value=None):
      with patch.object(
        gpt_structure,
        "generate_text",
        return_value=SimpleNamespace(value="done", fallback_used=False),
      ) as mock_generate_text:
        result = gpt_structure.GPT_request("Prompt body", {"temperature": 0.3, "max_tokens": 12, "stop": ["END"]})
    self.assertEqual(result, "done")
    messages = mock_generate_text.call_args.args[1]
    self.assertEqual([message["role"] for message in messages], ["system", "user"])
    self.assertEqual(mock_generate_text.call_args.kwargs["config"]["stop"], ["END"])

  def test_safe_generate_response_builds_generic_json_structured_task(self):
    def fake_run_task(spec):
      self.assertEqual(spec.lane, "structured")
      self.assertEqual(spec.metadata["output_family"], "generic_json")
      self.assertEqual([message["role"] for message in spec.messages_builder(None)], ["system", "user"])
      self.assertEqual(spec.schema["properties"]["output"]["properties"]["answer"]["type"], "string")
      parsed = spec.parser({"output": {"answer": "fixed"}})
      return TaskResult(
        value=parsed,
        raw_text='{"output": {"answer": "fixed"}}',
        model="mock-model",
        usage={},
        trace_id="trace",
        retries=0,
        fallback_used=False,
        parsed_ok=True,
        error_type=None,
        provider="mock",
        lane="structured",
      )

    with patch.object(gpt_structure, "run_task", side_effect=fake_run_task):
      result = gpt_structure.ChatGPT_safe_generate_response(
        "Prompt body",
        {"answer": "example"},
        "Return the answer field.",
        fail_safe_response={"answer": "fallback"},
      )
    self.assertEqual(result, {"answer": "fixed"})

  def test_safe_generate_response_returns_fail_safe_when_runner_falls_back(self):
    with patch.object(
      gpt_structure,
      "run_task",
      return_value=TaskResult(
        value={"answer": "fallback"},
        raw_text=None,
        model="mock-model",
        usage={},
        trace_id=None,
        retries=1,
        fallback_used=True,
        parsed_ok=False,
        error_type="StructuredOutputError",
        provider="mock",
        lane="structured",
      ),
    ):
      result = gpt_structure.ChatGPT_safe_generate_response(
        "Prompt body",
        {"answer": "example"},
        "Return the answer field.",
        fail_safe_response={"answer": "fallback"},
      )
    self.assertEqual(result, {"answer": "fallback"})

  def test_get_embedding_uses_facade_and_returns_first_vector(self):
    with patch.object(
      gpt_structure,
      "embed_texts",
      return_value=SimpleNamespace(value=[[1.0, 2.0], [3.0, 4.0]]),
    ) as mock_embed_texts:
      result = gpt_structure.get_embedding("line one\nline two")
    self.assertEqual(result, [1.0, 2.0])
    self.assertEqual(mock_embed_texts.call_args.args[0], ["line one line two"])


if __name__ == "__main__":
  unittest.main()
