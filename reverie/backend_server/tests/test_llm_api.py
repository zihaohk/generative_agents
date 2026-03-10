import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER_ROOT) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER_ROOT))

from llm.api import TaskSpec, generate_structured, generate_text, run_task
from llm.client import ProviderResponse
from llm.config import LLMConfig


class DummyProvider:
  def __init__(
    self,
    chat_responses=None,
    structured_responses=None,
    embeddings=None,
    provider_name="siliconflow",
    trace_id="trace-123",
  ):
    self.provider_name = provider_name
    self.chat_responses = list(chat_responses or [])
    self.structured_responses = list(structured_responses or [])
    self.embeddings = embeddings or [[0.1, 0.2, 0.3]]
    self.trace_id = trace_id
    self.chat_calls = 0
    self.chat_structured_calls = 0
    self.embed_calls = 0
    self.messages = []

  def _response(self, model, raw_text):
    return ProviderResponse(
      provider=self.provider_name,
      model=model,
      raw_text=raw_text,
      usage={"total_tokens": 1},
      trace_id=self.trace_id,
    )

  def chat(self, model, messages, temperature, max_tokens, stop):
    self.chat_calls += 1
    self.messages.append(list(messages))
    raw_text = self.chat_responses.pop(0) if self.chat_responses else "ok"
    return self._response(model, raw_text)

  def chat_structured(self, model, messages, temperature, max_tokens, stop):
    self.chat_structured_calls += 1
    self.messages.append(list(messages))
    raw_text = self.structured_responses.pop(0) if self.structured_responses else '{"output": "ok"}'
    return self._response(model, raw_text)

  def embed(self, model, texts):
    self.embed_calls += 1
    return ProviderResponse(
      provider=self.provider_name,
      model=model,
      embeddings=self.embeddings,
      usage={"total_tokens": len(texts)},
      trace_id="trace-embed",
    )


class LLMApiTests(unittest.TestCase):
  def _config(self, audit_dir, shadow_stack="none", legacy_api_key=""):
    return LLMConfig(
      active_stack="siliconflow",
      shadow_stack=shadow_stack,
      timeout_seconds=30,
      max_retries=2,
      audit_log_dir=audit_dir,
      run_live_siliconflow_tests=False,
      siliconflow_api_key="sf-key",
      siliconflow_base_url="https://api.siliconflow.cn/v1",
      siliconflow_primary_model="primary-model",
      siliconflow_structured_model="structured-model",
      siliconflow_embedding_model="embedding-model",
      legacy_api_key=legacy_api_key,
      legacy_base_url=None,
      legacy_primary_model="legacy-primary",
      legacy_structured_model="legacy-structured",
      legacy_embedding_model="legacy-embedding",
    )

  def _audit_payload(self, audit_dir):
    audit_files = list(Path(audit_dir).glob("audit-*.jsonl"))
    self.assertEqual(len(audit_files), 1)
    return json.loads(audit_files[0].read_text(encoding="utf-8").strip())

  def test_generate_text_writes_full_audit_record(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      provider = DummyProvider(chat_responses=["hello"])
      with patch("llm.api.get_llm_config", return_value=self._config(audit_dir)):
        with patch("llm.api._get_provider", return_value=provider):
          result = generate_text("text_task", [{"role": "user", "content": "Hi"}])
      self.assertEqual(result.value, "hello")
      self.assertEqual(provider.chat_calls, 1)
      self.assertEqual(provider.chat_structured_calls, 0)
      payload = self._audit_payload(audit_dir)
      expected_keys = {
        "task",
        "lane",
        "provider",
        "model",
        "messages_hash",
        "duration_ms",
        "retry_count",
        "usage",
        "x-siliconcloud-trace-id",
        "parsed_ok",
        "fallback_used",
        "error_type",
        "primary_value",
        "shadow_value",
        "shadow_provider",
        "shadow_model",
        "shadow_parsed_ok",
        "shadow_fallback_used",
        "shadow_diff",
        "shadow_error_type",
      }
      self.assertTrue(expected_keys.issubset(payload.keys()))
      self.assertEqual(payload["task"], "text_task")
      self.assertEqual(payload["provider"], "siliconflow")
      self.assertEqual(payload["x-siliconcloud-trace-id"], "trace-123")
      self.assertEqual(payload["primary_value"], "hello")
      self.assertIsNone(payload["shadow_value"])

  def test_structured_task_repairs_once_via_chat_structured(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      provider = DummyProvider(
        structured_responses=[
          "not-json",
          '{"output": {"answer": "fixed"}}',
        ]
      )
      with patch("llm.api.get_llm_config", return_value=self._config(audit_dir)):
        with patch("llm.api._get_provider", return_value=provider):
          result = generate_structured(
            "structured_task",
            [{"role": "user", "content": "Return JSON"}],
            {
              "type": "object",
              "properties": {
                "output": {
                  "type": "object",
                  "properties": {
                    "answer": {"type": "string"},
                  },
                  "required": ["answer"],
                },
              },
              "required": ["output"],
            },
          )
      self.assertEqual(result.value["output"]["answer"], "fixed")
      self.assertEqual(provider.chat_calls, 0)
      self.assertEqual(provider.chat_structured_calls, 2)

  def test_structured_task_uses_fallback_after_failed_repair(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      provider = DummyProvider(structured_responses=["not-json", "still-not-json"])
      spec = TaskSpec(
        name="broken_structured",
        lane="structured",
        messages_builder=lambda _context=None: [{"role": "user", "content": "Return JSON"}],
        schema={
          "type": "object",
          "properties": {
            "output": {
              "type": "object",
              "properties": {
                "answer": {"type": "string"},
              },
              "required": ["answer"],
            },
          },
          "required": ["output"],
        },
        parser=lambda payload: payload["output"]["answer"],
        fallback="fallback-answer",
      )
      with patch("llm.api.get_llm_config", return_value=self._config(audit_dir)):
        with patch("llm.api._get_provider", return_value=provider):
          result = run_task(spec)
      self.assertEqual(result.value, "fallback-answer")
      self.assertTrue(result.fallback_used)
      self.assertEqual(result.error_type, "StructuredOutputError")
      self.assertEqual(provider.chat_structured_calls, 2)

  def test_schema_normalizes_additional_properties_and_integer_bounds(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      provider = DummyProvider(
        structured_responses=['{"output": {"insight": ["1", "2"], "backup": [3]}}']
      )
      with patch("llm.api.get_llm_config", return_value=self._config(audit_dir)):
        with patch("llm.api._get_provider", return_value=provider):
          result = generate_structured(
            "dict_list_int_task",
            [{"role": "user", "content": "Return JSON"}],
            {
              "type": "object",
              "properties": {
                "output": {
                  "type": "object",
                  "additionalProperties": {
                    "type": "array",
                    "items": {"type": "integer"},
                  },
                },
              },
              "required": ["output"],
            },
          )
      self.assertEqual(result.value["output"]["insight"], [1, 2])
      self.assertEqual(result.value["output"]["backup"], [3])

      with tempfile.TemporaryDirectory() as bounded_audit_dir:
        bounded_provider = DummyProvider(structured_responses=['{"output": "7"}'])
        with patch("llm.api.get_llm_config", return_value=self._config(bounded_audit_dir)):
          with patch("llm.api._get_provider", return_value=bounded_provider):
            bounded_result = generate_structured(
              "bounded_integer",
              [{"role": "user", "content": "Return JSON"}],
              {
                "type": "object",
                "properties": {
                  "output": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                  },
                },
                "required": ["output"],
              },
            )
      self.assertEqual(bounded_result.value["output"], 7)

  def test_shadow_skip_without_legacy_key(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      provider = DummyProvider(chat_responses=["shadow test"])
      config = self._config(audit_dir, shadow_stack="legacy", legacy_api_key="")
      with patch("llm.api.get_llm_config", return_value=config):
        with patch("llm.api._get_provider", return_value=provider):
          result = generate_text("shadowed_text", [{"role": "user", "content": "Hi"}])
      self.assertEqual(result.value, "shadow test")
      payload = self._audit_payload(audit_dir)
      self.assertEqual(payload["shadow_error_type"], "skipped_missing_legacy_key")
      self.assertTrue(payload["shadow_fallback_used"])

  def test_shadow_audit_records_shadow_value_and_diff(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      primary = DummyProvider(chat_responses=["primary answer"], provider_name="siliconflow", trace_id="trace-main")
      shadow = DummyProvider(chat_responses=["shadow answer"], provider_name="legacy", trace_id="trace-shadow")

      def provider_for_stack(_settings, stack):
        return primary if stack == "siliconflow" else shadow

      config = self._config(audit_dir, shadow_stack="legacy", legacy_api_key="legacy-key")
      with patch("llm.api.get_llm_config", return_value=config):
        with patch("llm.api._get_provider", side_effect=provider_for_stack):
          result = generate_text("shadow_diff_task", [{"role": "user", "content": "Hi"}])
      self.assertEqual(result.value, "primary answer")
      payload = self._audit_payload(audit_dir)
      self.assertEqual(payload["shadow_provider"], "legacy")
      self.assertEqual(payload["shadow_value"], "shadow answer")
      self.assertIn("primary answer", payload["shadow_diff"])
      self.assertIn("shadow answer", payload["shadow_diff"])

  def test_embedding_lane_uses_embed(self):
    with tempfile.TemporaryDirectory() as audit_dir:
      provider = DummyProvider(embeddings=[[1.0, 2.0]])
      spec = TaskSpec(
        name="embed_task",
        lane="embedding",
        messages_builder=lambda _context=None: ["hello"],
      )
      with patch("llm.api.get_llm_config", return_value=self._config(audit_dir)):
        with patch("llm.api._get_provider", return_value=provider):
          result = run_task(spec)
      self.assertEqual(result.value, [[1.0, 2.0]])
      self.assertEqual(provider.embed_calls, 1)
      self.assertEqual(provider.chat_calls, 0)
      self.assertEqual(provider.chat_structured_calls, 0)


if __name__ == "__main__":
  unittest.main()
