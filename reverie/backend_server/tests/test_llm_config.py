import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER_ROOT) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER_ROOT))

from llm.config import get_llm_config, refresh_llm_config


class LLMConfigTests(unittest.TestCase):
  def tearDown(self):
    refresh_llm_config()

  def test_defaults(self):
    with patch.dict(os.environ, {}, clear=True):
      with tempfile.TemporaryDirectory() as temp_dir:
        with patch("llm.config._dotenv_path", return_value=Path(temp_dir) / ".env"):
          config = refresh_llm_config()
      self.assertEqual(config.active_stack, "siliconflow")
      self.assertEqual(config.shadow_stack, "none")
      self.assertEqual(config.siliconflow_primary_model, "Pro/deepseek-ai/DeepSeek-V3.2")
      self.assertEqual(config.siliconflow_structured_model, "Qwen/Qwen2.5-72B-Instruct")
      self.assertEqual(config.siliconflow_embedding_model, "BAAI/bge-m3")
      self.assertEqual(config.timeout_seconds, 60)
      self.assertEqual(config.max_retries, 3)

  def test_environment_overrides(self):
    with patch.dict(
      os.environ,
      {
        "LLM_ACTIVE_STACK": "legacy",
        "LLM_SHADOW_STACK": "siliconflow",
        "LLM_TIMEOUT_SECONDS": "9",
        "LLM_MAX_RETRIES": "5",
        "SILICONFLOW_PRIMARY_MODEL": "a",
        "SILICONFLOW_STRUCTURED_MODEL": "b",
        "SILICONFLOW_EMBEDDING_MODEL": "c",
        "OPENAI_API_KEY": "legacy-key",
      },
      clear=True,
    ):
      with tempfile.TemporaryDirectory() as temp_dir:
        with patch("llm.config._dotenv_path", return_value=Path(temp_dir) / ".env"):
          config = refresh_llm_config()
      self.assertEqual(config.active_stack, "legacy")
      self.assertEqual(config.shadow_stack, "siliconflow")
      self.assertEqual(config.timeout_seconds, 9)
      self.assertEqual(config.max_retries, 5)
      self.assertEqual(config.siliconflow_primary_model, "a")
      self.assertEqual(config.siliconflow_structured_model, "b")
      self.assertEqual(config.siliconflow_embedding_model, "c")
      self.assertEqual(config.legacy_api_key, "legacy-key")

  def test_root_dotenv_is_loaded(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      env_path = Path(temp_dir) / ".env"
      env_path.write_text(
        "\n".join(
          [
            "LLM_ACTIVE_STACK=legacy",
            "LLM_SHADOW_STACK=siliconflow",
            "LLM_TIMEOUT_SECONDS=12",
            "SILICONFLOW_API_KEY=sf-from-dotenv",
            "OPENAI_API_KEY=legacy-from-dotenv",
            "SILICONFLOW_PRIMARY_MODEL=dotenv-primary",
          ]
        ),
        encoding="utf-8",
      )
      with patch.dict(os.environ, {}, clear=True):
        with patch("llm.config._dotenv_path", return_value=env_path):
          config = refresh_llm_config()
      self.assertEqual(config.active_stack, "legacy")
      self.assertEqual(config.shadow_stack, "siliconflow")
      self.assertEqual(config.timeout_seconds, 12)
      self.assertEqual(config.siliconflow_api_key, "sf-from-dotenv")
      self.assertEqual(config.legacy_api_key, "legacy-from-dotenv")
      self.assertEqual(config.siliconflow_primary_model, "dotenv-primary")

  def test_environment_variables_override_root_dotenv(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      env_path = Path(temp_dir) / ".env"
      env_path.write_text(
        "\n".join(
          [
            "LLM_ACTIVE_STACK=siliconflow",
            "SILICONFLOW_API_KEY=sf-from-dotenv",
            "OPENAI_API_KEY=legacy-from-dotenv",
          ]
        ),
        encoding="utf-8",
      )
      with patch.dict(
        os.environ,
        {
          "LLM_ACTIVE_STACK": "legacy",
          "SILICONFLOW_API_KEY": "sf-from-env",
        },
        clear=True,
      ):
        with patch("llm.config._dotenv_path", return_value=env_path):
          config = refresh_llm_config()
      self.assertEqual(config.active_stack, "legacy")
      self.assertEqual(config.siliconflow_api_key, "sf-from-env")
      self.assertEqual(config.legacy_api_key, "legacy-from-dotenv")


if __name__ == "__main__":
  unittest.main()
