import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from .client import LegacyOpenAIClient, ProviderClient, ProviderResponse, SiliconFlowClient
from .config import LLMConfig, get_llm_config, refresh_llm_config


ParserFunc = Callable[[Any], Any]
MessagesBuilder = Callable[[Any], Any]


class StructuredOutputError(ValueError):
  pass


@dataclass(frozen=True)
class TaskSpec:
  name: str
  lane: str
  messages_builder: MessagesBuilder
  parser: Optional[ParserFunc] = None
  schema: Optional[Dict[str, Any]] = None
  sampling: Dict[str, Any] = field(default_factory=dict)
  stop: Optional[Sequence[str]] = None
  fallback: Any = None
  max_retries: Optional[int] = None
  model_override: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
  value: Any
  raw_text: Optional[str]
  model: str
  usage: Dict[str, Any]
  trace_id: Optional[str]
  retries: int
  fallback_used: bool
  parsed_ok: bool
  error_type: Optional[str]
  provider: str
  lane: str


@dataclass
class AuditEvent:
  task: str
  lane: str
  provider: str
  model: str
  messages_hash: str
  duration_ms: int
  retry_count: int
  usage: Dict[str, Any]
  trace_id: Optional[str]
  parsed_ok: bool
  fallback_used: bool
  error_type: Optional[str]
  shadow_provider: Optional[str] = None
  shadow_model: Optional[str] = None
  shadow_parsed_ok: Optional[bool] = None
  shadow_fallback_used: Optional[bool] = None
  shadow_diff: Optional[str] = None
  shadow_error_type: Optional[str] = None
  primary_value: Any = None
  shadow_value: Any = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

  def to_record(self) -> Dict[str, Any]:
    return {
      "timestamp": self.timestamp,
      "task": self.task,
      "lane": self.lane,
      "provider": self.provider,
      "model": self.model,
      "messages_hash": self.messages_hash,
      "duration_ms": self.duration_ms,
      "retry_count": self.retry_count,
      "usage": self.usage,
      "x-siliconcloud-trace-id": self.trace_id,
      "parsed_ok": self.parsed_ok,
      "fallback_used": self.fallback_used,
      "error_type": self.error_type,
      "primary_value": _json_safe_value(self.primary_value),
      "shadow_value": _json_safe_value(self.shadow_value),
      "shadow_provider": self.shadow_provider,
      "shadow_model": self.shadow_model,
      "shadow_parsed_ok": self.shadow_parsed_ok,
      "shadow_fallback_used": self.shadow_fallback_used,
      "shadow_diff": self.shadow_diff,
      "shadow_error_type": self.shadow_error_type,
      "metadata": self.metadata,
    }


_PROVIDER_CACHE: Dict[Any, ProviderClient] = {}


def refresh_llm_state() -> LLMConfig:
  _PROVIDER_CACHE.clear()
  return refresh_llm_config()


def _json_safe_value(value: Any) -> Any:
  try:
    json.dumps(value, ensure_ascii=False)
    return value
  except TypeError:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _get_audit_path(settings: LLMConfig) -> str:
  os.makedirs(settings.audit_log_dir, exist_ok=True)
  return os.path.join(
    settings.audit_log_dir,
    "audit-%s.jsonl" % datetime.now(timezone.utc).strftime("%Y-%m-%d"),
  )


def _log_audit(settings: LLMConfig, event: AuditEvent) -> None:
  with open(_get_audit_path(settings), "a", encoding="utf-8") as audit_file:
    audit_file.write(json.dumps(event.to_record(), ensure_ascii=False, default=str) + "\n")


def _normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
  normalized = []
  for message in messages:
    normalized.append(
      {
        "role": str(message.get("role", "user")),
        "content": str(message.get("content", "")),
      }
    )
  return normalized


def _messages_hash(messages: Sequence[Dict[str, str]]) -> str:
  payload = json.dumps(_normalize_messages(messages), ensure_ascii=False, sort_keys=True)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_json_value(raw_text: str) -> Any:
  text = (raw_text or "").strip()
  if not text:
    raise StructuredOutputError("empty structured response")

  if text.startswith("```"):
    text = text.strip("`")
    if text.startswith("json"):
      text = text[4:].strip()

  try:
    return json.loads(text)
  except json.JSONDecodeError:
    pass

  start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
  if not start_candidates:
    raise StructuredOutputError("no json object found")
  start = min(start_candidates)

  for end in range(len(text), start, -1):
    candidate = text[start:end].strip()
    try:
      return json.loads(candidate)
    except json.JSONDecodeError:
      continue
  raise StructuredOutputError("unable to parse json payload")


def _normalize_schema_value(value: Any, schema: Optional[Dict[str, Any]]) -> Any:
  if not schema:
    return value

  expected_type = schema.get("type")
  enum_values = schema.get("enum")
  if expected_type == "object":
    if not isinstance(value, dict):
      raise StructuredOutputError("expected object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    additional_properties = schema.get("additionalProperties")
    normalized = dict(value)
    for required_key in required:
      if required_key not in value:
        raise StructuredOutputError("missing required field: %s" % required_key)
    for key, sub_schema in properties.items():
      if key in value:
        normalized[key] = _normalize_schema_value(value[key], sub_schema)
    if additional_properties:
      for key, item in value.items():
        if key not in properties:
          normalized[key] = _normalize_schema_value(item, additional_properties)
    value = normalized
  elif expected_type == "array":
    if not isinstance(value, list):
      raise StructuredOutputError("expected array")
    item_schema = schema.get("items")
    value = [_normalize_schema_value(item, item_schema) for item in value]
  elif expected_type == "string":
    if value is None:
      raise StructuredOutputError("expected string")
    value = str(value)
  elif expected_type == "integer":
    if isinstance(value, bool):
      raise StructuredOutputError("expected integer")
    if isinstance(value, str):
      value = value.strip()
      if not value:
        raise StructuredOutputError("expected integer")
      value = int(value)
    elif not isinstance(value, int):
      raise StructuredOutputError("expected integer")
  elif expected_type == "number":
    if isinstance(value, bool):
      raise StructuredOutputError("expected number")
    if isinstance(value, str):
      value = float(value.strip())
    elif not isinstance(value, (int, float)):
      raise StructuredOutputError("expected number")
  elif expected_type == "boolean":
    if isinstance(value, bool):
      pass
    elif isinstance(value, str):
      lowered = value.strip().lower()
      if lowered in ("true", "yes", "1"):
        value = True
      elif lowered in ("false", "no", "0"):
        value = False
      else:
        raise StructuredOutputError("expected boolean")
    else:
      raise StructuredOutputError("expected boolean")

  if enum_values is not None and value not in enum_values:
    raise StructuredOutputError("value not in enum")
  minimum = schema.get("minimum")
  maximum = schema.get("maximum")
  if minimum is not None and value < minimum:
    raise StructuredOutputError("value below minimum")
  if maximum is not None and value > maximum:
    raise StructuredOutputError("value above maximum")
  return value


def _build_repair_messages(
  messages: Sequence[Dict[str, str]],
  raw_text: str,
  schema: Optional[Dict[str, Any]],
  error: Exception,
) -> List[Dict[str, str]]:
  repair_instruction = (
    "Your previous response was invalid. Return only valid JSON that matches this schema: "
    + json.dumps(schema or {"type": "object"}, ensure_ascii=False)
    + ". Do not include markdown or extra commentary."
    + " Previous error: "
    + str(error)
  )
  repaired = list(_normalize_messages(messages))
  repaired.append({"role": "assistant", "content": raw_text or ""})
  repaired.append({"role": "user", "content": repair_instruction})
  return repaired


def _resolve_model(settings: LLMConfig, stack: str, lane: str, override: Optional[str]) -> str:
  if override:
    return override
  if stack == "siliconflow":
    if lane == "structured":
      return settings.siliconflow_structured_model
    if lane == "embedding":
      return settings.siliconflow_embedding_model
    return settings.siliconflow_primary_model
  if lane == "embedding":
    return settings.legacy_embedding_model
  if lane == "structured":
    return settings.legacy_structured_model
  return settings.legacy_primary_model


def _get_provider(settings: LLMConfig, stack: str) -> ProviderClient:
  key = (stack, settings)
  if key in _PROVIDER_CACHE:
    return _PROVIDER_CACHE[key]

  if stack == "siliconflow":
    provider = SiliconFlowClient(settings)
  elif stack == "legacy":
    provider = LegacyOpenAIClient(settings)
  else:
    raise ValueError("Unsupported LLM stack: %s" % stack)
  _PROVIDER_CACHE[key] = provider
  return provider


def _coerce_sampling(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  if not config:
    return {}
  return dict(config)


def _default_messages_builder(messages: Sequence[Dict[str, Any]]) -> MessagesBuilder:
  normalized = _normalize_messages(messages)
  return lambda _context=None: normalized


def _execute_primary(
  task_name: str,
  provider: ProviderClient,
  model: str,
  messages: Sequence[Dict[str, str]],
  parser: Optional[ParserFunc],
  sampling: Dict[str, Any],
  stop: Optional[Sequence[str]],
) -> ProviderResponse:
  return provider.chat(
    model=model,
    messages=messages,
    temperature=float(sampling.get("temperature", 0.0)),
    max_tokens=sampling.get("max_tokens"),
    stop=stop or sampling.get("stop"),
  )


def _execute_embedding(
  provider: ProviderClient,
  model: str,
  texts: Sequence[str],
) -> ProviderResponse:
  return provider.embed(model=model, texts=list(texts))


def _compare_results(primary: Optional[TaskResult], shadow: Optional[TaskResult]) -> Optional[str]:
  if not primary or not shadow:
    return None
  if primary.value == shadow.value:
    return "equal"
  primary_preview = json.dumps(primary.value, ensure_ascii=False, default=str)[:200]
  shadow_preview = json.dumps(shadow.value, ensure_ascii=False, default=str)[:200]
  return "primary=%s | shadow=%s" % (primary_preview, shadow_preview)


def _run_text_task(
  task_name: str,
  lane: str,
  messages: Sequence[Dict[str, str]],
  parser: Optional[ParserFunc],
  sampling: Dict[str, Any],
  stop: Optional[Sequence[str]],
  model: str,
  provider: ProviderClient,
) -> TaskResult:
  response = _execute_primary(task_name, provider, model, messages, parser, sampling, stop)
  value = response.raw_text
  if parser is not None:
    value = parser(response.raw_text)
  return TaskResult(
    value=value,
    raw_text=response.raw_text,
    model=model,
    usage=response.usage,
    trace_id=response.trace_id,
    retries=0,
    fallback_used=False,
    parsed_ok=True,
    error_type=None,
    provider=provider.provider_name,
    lane=lane,
  )


def _run_structured_task(
  task_name: str,
  messages: Sequence[Dict[str, str]],
  parser: Optional[ParserFunc],
  schema: Optional[Dict[str, Any]],
  sampling: Dict[str, Any],
  stop: Optional[Sequence[str]],
  max_retries: int,
  model: str,
  provider: ProviderClient,
) -> TaskResult:
  attempts = 0
  current_messages = list(messages)
  last_response: Optional[ProviderResponse] = None
  last_error: Optional[Exception] = None

  while attempts < max_retries:
    last_response = provider.chat_structured(
      model=model,
      messages=current_messages,
      temperature=float(sampling.get("temperature", 0.0)),
      max_tokens=sampling.get("max_tokens"),
      stop=stop or sampling.get("stop"),
    )
    try:
      structured_value = _extract_json_value(last_response.raw_text or "")
      normalized = _normalize_schema_value(structured_value, schema)
      final_value = normalized if parser is None else parser(normalized)
      return TaskResult(
        value=final_value,
        raw_text=last_response.raw_text,
        model=model,
        usage=last_response.usage,
        trace_id=last_response.trace_id,
        retries=attempts,
        fallback_used=False,
        parsed_ok=True,
        error_type=None,
        provider=provider.provider_name,
        lane="structured",
      )
    except Exception as exc:
      last_error = exc
      attempts += 1
      if attempts >= max_retries:
        break
      current_messages = _build_repair_messages(messages, last_response.raw_text or "", schema, exc)

  raise StructuredOutputError(str(last_error or "structured task failed"))


def _run_lane(
  settings: LLMConfig,
  stack: str,
  lane: str,
  task_name: str,
  messages_or_payload: Any,
  parser: Optional[ParserFunc],
  schema: Optional[Dict[str, Any]],
  sampling: Dict[str, Any],
  stop: Optional[Sequence[str]],
  max_retries: int,
  model_override: Optional[str],
) -> TaskResult:
  provider = _get_provider(settings, stack)
  model = _resolve_model(settings, stack, lane, model_override)
  if lane == "embedding":
    response = _execute_embedding(provider, model, messages_or_payload)
    value = response.embeddings
    if parser is not None:
      value = parser(value)
    return TaskResult(
      value=value,
      raw_text=None,
      model=model,
      usage=response.usage,
      trace_id=response.trace_id,
      retries=0,
      fallback_used=False,
      parsed_ok=True,
      error_type=None,
      provider=provider.provider_name,
      lane=lane,
    )
  if lane == "structured":
    return _run_structured_task(
      task_name=task_name,
      messages=messages_or_payload,
      parser=parser,
      schema=schema,
      sampling=sampling,
      stop=stop,
      max_retries=max_retries,
      model=model,
      provider=provider,
    )
  return _run_text_task(
    task_name=task_name,
    lane=lane,
    messages=messages_or_payload,
    parser=parser,
    sampling=sampling,
    stop=stop,
    model=model,
    provider=provider,
  )


def _shadow_result_for_spec(
  settings: LLMConfig,
  spec: TaskSpec,
  payload: Any,
  main_result: TaskResult,
) -> Optional[TaskResult]:
  if settings.shadow_stack == "none" or settings.shadow_stack == settings.active_stack:
    return None
  if settings.shadow_stack == "legacy" and not settings.legacy_api_key:
    return TaskResult(
      value=None,
      raw_text=None,
      model=_resolve_model(settings, "legacy", spec.lane, spec.model_override),
      usage={},
      trace_id=None,
      retries=0,
      fallback_used=True,
      parsed_ok=False,
      error_type="skipped_missing_legacy_key",
      provider="legacy",
      lane=spec.lane,
    )
  try:
    return _run_lane(
      settings=settings,
      stack=settings.shadow_stack,
      lane=spec.lane,
      task_name=spec.name,
      messages_or_payload=payload,
      parser=spec.parser,
      schema=spec.schema,
      sampling=spec.sampling,
      stop=spec.stop,
      max_retries=spec.max_retries or settings.max_retries,
      model_override=spec.model_override,
    )
  except Exception as exc:
    return TaskResult(
      value=None,
      raw_text=None,
      model=_resolve_model(settings, settings.shadow_stack, spec.lane, spec.model_override),
      usage={},
      trace_id=None,
      retries=0,
      fallback_used=True,
      parsed_ok=False,
      error_type=type(exc).__name__,
      provider=settings.shadow_stack,
      lane=spec.lane,
    )


def run_task(task_spec: TaskSpec, context: Any = None) -> TaskResult:
  settings = get_llm_config()
  payload = task_spec.messages_builder(context)
  if task_spec.lane != "embedding":
    payload = _normalize_messages(payload)

  started_at = time.time()
  main_result: Optional[TaskResult] = None
  shadow_result: Optional[TaskResult] = None
  fallback_used = False
  error_type = None
  try:
    main_result = _run_lane(
      settings=settings,
      stack=settings.active_stack,
      lane=task_spec.lane,
      task_name=task_spec.name,
      messages_or_payload=payload,
      parser=task_spec.parser,
      schema=task_spec.schema,
      sampling=task_spec.sampling,
      stop=task_spec.stop,
      max_retries=task_spec.max_retries or settings.max_retries,
      model_override=task_spec.model_override,
    )
  except Exception as exc:
    error_type = type(exc).__name__
    fallback_used = True
    main_result = TaskResult(
      value=task_spec.fallback,
      raw_text=None,
      model=_resolve_model(settings, settings.active_stack, task_spec.lane, task_spec.model_override),
      usage={},
      trace_id=None,
      retries=task_spec.max_retries or settings.max_retries,
      fallback_used=True,
      parsed_ok=False,
      error_type=error_type,
      provider=settings.active_stack,
      lane=task_spec.lane,
    )

  shadow_result = _shadow_result_for_spec(settings, task_spec, payload, main_result)
  duration_ms = int((time.time() - started_at) * 1000)
  _log_audit(
    settings,
    AuditEvent(
      task=task_spec.name,
      lane=task_spec.lane,
      provider=main_result.provider,
      model=main_result.model,
      messages_hash=_messages_hash(payload if task_spec.lane != "embedding" else [{"role": "embed", "content": json.dumps(payload)}]),
      duration_ms=duration_ms,
      retry_count=main_result.retries,
      usage=main_result.usage,
      trace_id=main_result.trace_id,
      parsed_ok=main_result.parsed_ok,
      fallback_used=main_result.fallback_used or fallback_used,
      error_type=main_result.error_type or error_type,
      shadow_provider=shadow_result.provider if shadow_result else None,
      shadow_model=shadow_result.model if shadow_result else None,
      shadow_parsed_ok=shadow_result.parsed_ok if shadow_result else None,
      shadow_fallback_used=shadow_result.fallback_used if shadow_result else None,
      shadow_diff=_compare_results(main_result, shadow_result),
      shadow_error_type=shadow_result.error_type if shadow_result else None,
      primary_value=main_result.value,
      shadow_value=shadow_result.value if shadow_result else None,
      metadata=task_spec.metadata,
    ),
  )
  return main_result


def generate_text(
  task: str,
  messages: Sequence[Dict[str, Any]],
  config: Optional[Dict[str, Any]] = None,
) -> TaskResult:
  config = _coerce_sampling(config)
  spec = TaskSpec(
    name=task,
    lane="primary",
    messages_builder=_default_messages_builder(messages),
    sampling=config,
    stop=config.get("stop"),
    model_override=config.get("model"),
  )
  return run_task(spec)


def generate_structured(
  task: str,
  messages: Sequence[Dict[str, Any]],
  schema: Optional[Dict[str, Any]],
  config: Optional[Dict[str, Any]] = None,
) -> TaskResult:
  config = _coerce_sampling(config)
  spec = TaskSpec(
    name=task,
    lane="structured",
    messages_builder=_default_messages_builder(messages),
    schema=schema,
    sampling=config,
    stop=config.get("stop"),
    model_override=config.get("model"),
  )
  return run_task(spec)


def embed_texts(
  texts: Sequence[str],
  config: Optional[Dict[str, Any]] = None,
) -> TaskResult:
  config = config or {}
  spec = TaskSpec(
    name="embedding",
    lane="embedding",
    messages_builder=lambda _context=None: list(texts),
    sampling={},
    model_override=config.get("model"),
  )
  return run_task(spec)
