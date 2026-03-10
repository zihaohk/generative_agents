"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Compatibility shim over the new provider-agnostic LLM task layer.
"""
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm import TaskSpec, embed_texts, generate_text, run_task
from llm.families import (
  build_output_schema,
  build_primary_messages,
  build_structured_messages,
  parse_structured_output,
)
from utils import *  # noqa: F401,F403


ValidateFn = Optional[Callable[[Any], bool]]
CleanupFn = Optional[Callable[[Any], Any]]


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def _bool_or_default(value: Optional[bool], default: bool) -> bool:
  if value is None:
    return default
  return bool(value)


def _default_validate(_value: Any, prompt: str = "") -> bool:
  return True


def _default_cleanup(value: Any, prompt: str = "") -> Any:
  return value


def _legacy_sampling_from_gpt_params(gpt_parameter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  gpt_parameter = gpt_parameter or {}
  return {
    "temperature": gpt_parameter.get("temperature", 0),
    "max_tokens": gpt_parameter.get("max_tokens"),
  }


def _schema_from_example(example_output: Any) -> Dict[str, Any]:
  if isinstance(example_output, bool):
    return {"type": "boolean"}
  if isinstance(example_output, int):
    return {"type": "integer"}
  if isinstance(example_output, float):
    return {"type": "number"}
  if isinstance(example_output, str):
    return {"type": "string"}
  if isinstance(example_output, list):
    item_schema = _schema_from_example(example_output[0]) if example_output else {}
    return {
      "type": "array",
      "items": item_schema,
    }
  if isinstance(example_output, dict):
    return {
      "type": "object",
      "properties": {
        str(key): _schema_from_example(value)
        for key, value in example_output.items()
      },
      "required": [str(key) for key in example_output.keys()],
    }
  return {}


def _primary_task_spec(
  task_name: str,
  prompt: str,
  sampling: Optional[Dict[str, Any]] = None,
  stop: Optional[Any] = None,
  parser: Optional[Callable[[str], Any]] = None,
  fallback: Any = None,
  max_retries: int = 1,
) -> TaskSpec:
  sampling = dict(sampling or {})
  if stop is None:
    stop = sampling.get("stop")
  return TaskSpec(
    name=task_name,
    lane="primary",
    messages_builder=lambda _context=None: build_primary_messages(prompt),
    parser=parser,
    sampling=sampling,
    stop=stop,
    fallback=fallback,
    max_retries=max_retries,
    metadata={"compat_layer": "gpt_structure", "output_family": "plain_text"},
  )


def _structured_task_spec(
  task_name: str,
  prompt: str,
  parser: Callable[[Dict[str, Any]], Any],
  fallback: Any,
  example_output: Any,
  special_instruction: str,
  max_retries: int,
) -> TaskSpec:
  inner_schema = _schema_from_example(example_output)
  return TaskSpec(
    name=task_name,
    lane="structured",
    messages_builder=lambda _context=None: build_structured_messages(
      prompt,
      family="generic_json",
      options={
        "schema": inner_schema,
        "example_output": example_output,
      },
      extra_system=[
        special_instruction,
        "Example output JSON:\n" + json.dumps({"output": example_output}, ensure_ascii=False),
      ],
    ),
    parser=parser,
    schema=build_output_schema(inner_schema),
    fallback=fallback,
    max_retries=max_retries,
    metadata={"compat_layer": "gpt_structure", "output_family": "generic_json"},
  )


def _validate_and_clean(
  value: Any,
  prompt: str,
  func_validate: ValidateFn,
  func_clean_up: CleanupFn,
) -> Any:
  validate = func_validate or _default_validate
  clean_up = func_clean_up or _default_cleanup
  if not validate(value, prompt=prompt):
    raise ValueError("legacy validation failed")
  return clean_up(value, prompt=prompt)


def _run_primary_with_legacy_validation(
  task_name: str,
  prompt: str,
  sampling: Optional[Dict[str, Any]],
  repeat: int,
  fallback: Any,
  func_validate: ValidateFn,
  func_clean_up: CleanupFn,
) -> Any:
  last_value = fallback
  for attempt in range(max(1, repeat)):
    spec = _primary_task_spec(
      task_name=task_name,
      prompt=prompt,
      sampling=sampling,
      fallback=fallback,
      parser=None,
      max_retries=1,
    )
    result = run_task(spec)
    raw_value = result.value
    if result.fallback_used and result.error_type:
      last_value = fallback
      continue
    try:
      return _validate_and_clean(raw_value, prompt, func_validate, func_clean_up)
    except Exception:
      last_value = fallback
      if attempt + 1 >= max(1, repeat):
        break
  return last_value


def _run_structured_with_legacy_validation(
  task_name: str,
  prompt: str,
  example_output: Any,
  special_instruction: str,
  repeat: int,
  fallback: Any,
  func_validate: ValidateFn,
  func_clean_up: CleanupFn,
) -> Any:
  def parser(payload: Dict[str, Any]) -> Any:
    candidate = parse_structured_output(payload["output"], "generic_json")
    return _validate_and_clean(candidate, prompt, func_validate, func_clean_up)

  spec = _structured_task_spec(
    task_name=task_name,
    prompt=prompt,
    parser=parser,
    fallback=fallback,
    example_output=example_output,
    special_instruction=special_instruction,
    max_retries=max(1, repeat),
  )
  result = run_task(spec)
  if result.fallback_used:
    return fallback
  return result.value


def ChatGPT_single_request(prompt):
  temp_sleep()
  result = generate_text(
    "ChatGPT_single_request",
    build_primary_messages(prompt),
  )
  return result.value


def GPT4_request(prompt):
  temp_sleep()
  result = generate_text(
    "GPT4_request",
    build_primary_messages(prompt),
  )
  if result.fallback_used:
    print("ChatGPT ERROR")
    return "ChatGPT ERROR"
  return result.value


def ChatGPT_request(prompt):
  temp_sleep()
  result = generate_text(
    "ChatGPT_request",
    build_primary_messages(prompt),
  )
  if result.fallback_used:
    print("ChatGPT ERROR")
    return "ChatGPT ERROR"
  return result.value


def GPT4_safe_generate_response(
  prompt,
  example_output,
  special_instruction,
  repeat=3,
  fail_safe_response="error",
  func_validate=None,
  func_clean_up=None,
  verbose=False,
):
  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)
  return _run_structured_with_legacy_validation(
    "GPT4_safe_generate_response",
    prompt,
    example_output,
    special_instruction,
    repeat,
    fail_safe_response,
    func_validate,
    func_clean_up,
  )


def ChatGPT_safe_generate_response(
  prompt,
  example_output,
  special_instruction,
  repeat=3,
  fail_safe_response="error",
  func_validate=None,
  func_clean_up=None,
  verbose=False,
):
  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)
  return _run_structured_with_legacy_validation(
    "ChatGPT_safe_generate_response",
    prompt,
    example_output,
    special_instruction,
    repeat,
    fail_safe_response,
    func_validate,
    func_clean_up,
  )


def ChatGPT_safe_generate_response_OLD(
  prompt,
  repeat=3,
  fail_safe_response="error",
  func_validate=None,
  func_clean_up=None,
  verbose=False,
):
  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)
  return _run_primary_with_legacy_validation(
    "ChatGPT_safe_generate_response_OLD",
    prompt,
    sampling={"temperature": 0, "max_tokens": 400},
    repeat=repeat,
    fallback=fail_safe_response,
    func_validate=func_validate,
    func_clean_up=func_clean_up,
  )


def GPT_request(prompt, gpt_parameter):
  temp_sleep()
  sampling = _legacy_sampling_from_gpt_params(gpt_parameter)
  result = generate_text(
    "GPT_request",
    build_primary_messages(prompt),
    config={
      **sampling,
      "stop": (gpt_parameter or {}).get("stop"),
    },
  )
  if result.fallback_used:
    print("TOKEN LIMIT EXCEEDED or ERROR")
    return "TOKEN LIMIT EXCEEDED"
  return result.value


def generate_prompt(curr_input, prompt_lib_file):
  if isinstance(curr_input, str):
    curr_input = [curr_input]
  curr_input = [str(item) for item in curr_input]

  with open(prompt_lib_file, "r", encoding="utf-8") as prompt_file:
    prompt = prompt_file.read()
  for count, item in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", item)
  marker = "<commentblockmarker>###</commentblockmarker>"
  if marker in prompt:
    prompt = prompt.split(marker)[1]
  return prompt.strip()


def safe_generate_response(
  prompt,
  gpt_parameter,
  repeat=5,
  fail_safe_response="error",
  func_validate=None,
  func_clean_up=None,
  verbose=False,
):
  if verbose:
    print(prompt)
  return _run_primary_with_legacy_validation(
    "safe_generate_response",
    prompt,
    sampling={
      **_legacy_sampling_from_gpt_params(gpt_parameter),
      "stop": (gpt_parameter or {}).get("stop"),
    },
    repeat=repeat,
    fallback=fail_safe_response,
    func_validate=func_validate,
    func_clean_up=func_clean_up,
  )


def get_embedding(text, model=None):
  text = (text or "").replace("\n", " ").strip()
  if not text:
    text = "this is blank"
  result = embed_texts([text], config={"model": model} if model else None)
  embeddings = result.value or []
  if not embeddings:
    return []
  return embeddings[0]
