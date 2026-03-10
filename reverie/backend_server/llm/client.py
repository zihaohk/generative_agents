from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import LLMConfig


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
  if usage is None:
    return {}
  if hasattr(usage, "model_dump"):
    return usage.model_dump()
  if isinstance(usage, dict):
    return usage
  return {
    key: value
    for key, value in getattr(usage, "__dict__", {}).items()
    if not key.startswith("_")
  }


def _message_content_to_text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, dict):
        if item.get("type") == "text":
          parts.append(item.get("text", ""))
      elif hasattr(item, "text"):
        parts.append(getattr(item, "text", ""))
      else:
        parts.append(str(item))
    return "".join(parts)
  if content is None:
    return ""
  return str(content)


@dataclass
class ProviderResponse:
  provider: str
  model: str
  raw_text: Optional[str] = None
  embeddings: Optional[List[List[float]]] = None
  usage: Dict[str, Any] = field(default_factory=dict)
  trace_id: Optional[str] = None
  response_id: Optional[str] = None
  headers: Dict[str, Any] = field(default_factory=dict)
  raw_payload: Any = None


class ProviderClient(ABC):
  provider_name: str = "unknown"

  @abstractmethod
  def chat(
    self,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    stop: Optional[Sequence[str]],
  ) -> ProviderResponse:
    raise NotImplementedError

  @abstractmethod
  def chat_structured(
    self,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    stop: Optional[Sequence[str]],
  ) -> ProviderResponse:
    raise NotImplementedError

  @abstractmethod
  def embed(self, model: str, texts: Sequence[str]) -> ProviderResponse:
    raise NotImplementedError


class _OpenAICompatibleClient(ProviderClient):
  trace_header_names: Sequence[str] = ()

  def __init__(
    self,
    provider_name: str,
    api_key: str,
    base_url: Optional[str],
    timeout_seconds: int,
    trace_header_names: Sequence[str],
  ):
    self.provider_name = provider_name
    self._api_key = api_key
    self._base_url = base_url
    self._timeout_seconds = timeout_seconds
    self.trace_header_names = trace_header_names
    self._client = None

  def _ensure_client(self):
    if self._client is not None:
      return self._client
    if not self._api_key:
      raise RuntimeError(f"{self.provider_name} API key is not configured")
    try:
      from openai import OpenAI
    except ImportError as exc:
      raise RuntimeError("openai>=1,<2 is required for the new LLM layer") from exc

    kwargs: Dict[str, Any] = {
      "api_key": self._api_key,
      "timeout": self._timeout_seconds,
    }
    if self._base_url:
      kwargs["base_url"] = self._base_url
    self._client = OpenAI(**kwargs)
    return self._client

  def _extract_trace_id(self, headers: Dict[str, Any]) -> Optional[str]:
    for name in self.trace_header_names:
      if name in headers and headers[name]:
        return headers[name]
    return None

  def _create_with_headers(self, resource: Any, **kwargs: Any):
    headers: Dict[str, Any] = {}
    if hasattr(resource, "with_raw_response"):
      raw_response = resource.with_raw_response.create(**kwargs)
      parsed = raw_response.parse()
      headers = dict(getattr(raw_response, "headers", {}) or {})
      return parsed, headers
    parsed = resource.create(**kwargs)
    return parsed, headers

  def _chat_with_options(
    self,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    stop: Optional[Sequence[str]],
    response_format: Optional[Dict[str, Any]] = None,
  ) -> ProviderResponse:
    client = self._ensure_client()
    kwargs: Dict[str, Any] = {
      "model": model,
      "messages": list(messages),
      "temperature": temperature,
    }
    if max_tokens is not None:
      kwargs["max_tokens"] = max_tokens
    if stop:
      kwargs["stop"] = list(stop)
    if response_format:
      kwargs["response_format"] = response_format

    parsed, headers = self._create_with_headers(client.chat.completions, **kwargs)
    message = parsed.choices[0].message
    content = _message_content_to_text(getattr(message, "content", ""))
    return ProviderResponse(
      provider=self.provider_name,
      model=model,
      raw_text=content,
      usage=_usage_to_dict(getattr(parsed, "usage", None)),
      trace_id=self._extract_trace_id(headers),
      response_id=getattr(parsed, "id", None),
      headers=headers,
      raw_payload=parsed,
    )

  def chat(
    self,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    stop: Optional[Sequence[str]],
  ) -> ProviderResponse:
    return self._chat_with_options(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop,
    )

  def chat_structured(
    self,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    stop: Optional[Sequence[str]],
  ) -> ProviderResponse:
    return self._chat_with_options(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop,
      response_format={"type": "json_object"},
    )

  def embed(self, model: str, texts: Sequence[str]) -> ProviderResponse:
    client = self._ensure_client()
    parsed, headers = self._create_with_headers(
      client.embeddings,
      model=model,
      input=list(texts),
    )
    embeddings = [item.embedding for item in parsed.data]
    return ProviderResponse(
      provider=self.provider_name,
      model=model,
      embeddings=embeddings,
      usage=_usage_to_dict(getattr(parsed, "usage", None)),
      trace_id=self._extract_trace_id(headers),
      response_id=getattr(parsed, "id", None),
      headers=headers,
      raw_payload=parsed,
    )


class SiliconFlowClient(_OpenAICompatibleClient):
  def __init__(self, settings: LLMConfig):
    super().__init__(
      provider_name="siliconflow",
      api_key=settings.siliconflow_api_key,
      base_url=settings.siliconflow_base_url,
      timeout_seconds=settings.timeout_seconds,
      trace_header_names=("x-siliconcloud-trace-id", "x-request-id"),
    )


class LegacyOpenAIClient(_OpenAICompatibleClient):
  def __init__(self, settings: LLMConfig):
    super().__init__(
      provider_name="legacy",
      api_key=settings.legacy_api_key,
      base_url=settings.legacy_base_url,
      timeout_seconds=settings.timeout_seconds,
      trace_header_names=("x-request-id",),
    )
