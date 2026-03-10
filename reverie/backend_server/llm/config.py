import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _dotenv_path() -> Path:
  return Path(__file__).resolve().parents[3] / ".env"


@lru_cache(maxsize=1)
def _load_repo_dotenv() -> dict[str, str]:
  env_path = _dotenv_path()
  if not env_path.exists():
    return {}

  values: dict[str, str] = {}
  for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    if line.startswith("export "):
      line = line[len("export "):].strip()
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
      continue
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
      value = value[1:-1]
    values[key] = value
  return values


def _get_env(name: str, default: str = "") -> str:
  if name in os.environ:
    return os.environ.get(name, "")
  return _load_repo_dotenv().get(name, default)


def _normalize_stack(value: str, allowed: set[str], default: str) -> str:
  value = (value or "").strip().lower()
  if value in allowed:
    return value
  return default


def _get_int(name: str, default: int) -> int:
  raw = _get_env(name, "").strip()
  if not raw:
    return default
  try:
    return int(raw)
  except ValueError:
    return default


@dataclass(frozen=True)
class LLMConfig:
  active_stack: str
  shadow_stack: str
  timeout_seconds: int
  max_retries: int
  audit_log_dir: str
  run_live_siliconflow_tests: bool
  siliconflow_api_key: str
  siliconflow_base_url: str
  siliconflow_primary_model: str
  siliconflow_structured_model: str
  siliconflow_embedding_model: str
  legacy_api_key: str
  legacy_base_url: Optional[str]
  legacy_primary_model: str
  legacy_structured_model: str
  legacy_embedding_model: str


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
  return LLMConfig(
    active_stack=_normalize_stack(
      _get_env("LLM_ACTIVE_STACK", "siliconflow"),
      {"siliconflow", "legacy"},
      "siliconflow",
    ),
    shadow_stack=_normalize_stack(
      _get_env("LLM_SHADOW_STACK", "none"),
      {"none", "legacy", "siliconflow"},
      "none",
    ),
    timeout_seconds=max(1, _get_int("LLM_TIMEOUT_SECONDS", 60)),
    max_retries=max(1, _get_int("LLM_MAX_RETRIES", 3)),
    audit_log_dir=_get_env("LLM_AUDIT_LOG_DIR", "logs/llm_audit"),
    run_live_siliconflow_tests=_get_env("RUN_LIVE_SILICONFLOW_TESTS", "").strip() == "1",
    siliconflow_api_key=_get_env("SILICONFLOW_API_KEY", "").strip(),
    siliconflow_base_url=_get_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").strip(),
    siliconflow_primary_model=_get_env(
      "SILICONFLOW_PRIMARY_MODEL",
      "Pro/deepseek-ai/DeepSeek-V3.2",
    ).strip(),
    siliconflow_structured_model=_get_env(
      "SILICONFLOW_STRUCTURED_MODEL",
      "Qwen/Qwen2.5-72B-Instruct",
    ).strip(),
    siliconflow_embedding_model=_get_env(
      "SILICONFLOW_EMBEDDING_MODEL",
      "BAAI/bge-m3",
    ).strip(),
    legacy_api_key=_get_env("OPENAI_API_KEY", "").strip(),
    legacy_base_url=_get_env("OPENAI_BASE_URL", "").strip() or None,
    legacy_primary_model=_get_env("LEGACY_PRIMARY_MODEL", "gpt-4.1-mini").strip(),
    legacy_structured_model=_get_env("LEGACY_STRUCTURED_MODEL", "gpt-4.1-mini").strip(),
    legacy_embedding_model=_get_env("LEGACY_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
  )


def refresh_llm_config() -> LLMConfig:
  _load_repo_dotenv.cache_clear()
  get_llm_config.cache_clear()
  return get_llm_config()


def is_siliconflow_configured() -> bool:
  return bool(get_llm_config().siliconflow_api_key)


def is_legacy_configured() -> bool:
  return bool(get_llm_config().legacy_api_key)
