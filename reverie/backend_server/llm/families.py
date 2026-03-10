import re
from typing import Any, Dict, List, Optional, Sequence


class OutputFamilyError(ValueError):
  pass


def build_output_schema(inner_schema: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "type": "object",
    "properties": {
      "output": inner_schema,
    },
    "required": ["output"],
  }


def structured_family_schema(family: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  options = dict(options or {})
  if family == "integer":
    schema: Dict[str, Any] = {"type": "integer"}
    if "minimum" in options:
      schema["minimum"] = options["minimum"]
    if "maximum" in options:
      schema["maximum"] = options["maximum"]
    return schema
  if family == "enum_string":
    return {
      "type": "string",
      "enum": list(options.get("choices", [])),
    }
  if family == "boolean":
    return {"type": "boolean"}
  if family == "triple":
    return {
      "type": "object",
      "properties": {
        "predicate": {"type": "string"},
        "object": {"type": "string"},
      },
      "required": ["predicate", "object"],
    }
  if family == "list_str":
    return {
      "type": "array",
      "items": {"type": "string"},
    }
  if family == "plain_text_json":
    return {"type": "string"}
  if family == "generic_json":
    return dict(options.get("schema") or {})
  if family == "chat_transcript":
    return {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "speaker": {"type": "string"},
          "utterance": {"type": "string"},
        },
        "required": ["speaker", "utterance"],
      },
    }
  if family == "dict_list_int":
    return {
      "type": "object",
      "additionalProperties": {
        "type": "array",
        "items": {"type": "integer"},
      },
    }
  if family == "utterance_turn":
    return {
      "type": "object",
      "properties": {
        "utterance": {"type": "string"},
        "end": {"type": "boolean"},
      },
      "required": ["utterance", "end"],
    }
  if family == "schedule_decomp":
    return {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "task": {"type": "string"},
          "duration": {"type": "integer", "minimum": 0},
        },
        "required": ["task", "duration"],
      },
    }
  if family == "safety_score":
    return {
      "type": "integer",
      "minimum": 0,
      "maximum": 10,
    }
  raise OutputFamilyError("Unsupported structured family: %s" % family)


def primary_family_instruction(family: str, options: Optional[Dict[str, Any]] = None) -> str:
  options = dict(options or {})
  if family == "plain_text":
    return "Return plain text only. Do not use JSON, markdown fences, or explanatory prefixes."
  if family == "numbered_list":
    return "Return only the numbered items as plain text. Do not add commentary before or after the list."
  if family == "quoted_text":
    return "Return plain text only, without wrapping the answer in quotes."
  return "Return plain text only."


def structured_family_instruction(family: str, options: Optional[Dict[str, Any]] = None) -> str:
  options = dict(options or {})
  if family == "integer":
    return "Return a JSON object with an integer field named output."
  if family == "enum_string":
    return "Return a JSON object with string field output. Allowed values: %s." % ", ".join(
      list(options.get("choices", []))
    )
  if family == "boolean":
    return "Return a JSON object with boolean field output."
  if family == "triple":
    return 'Return a JSON object like {"output": {"predicate": "...", "object": "..."}}.'
  if family == "list_str":
    return 'Return a JSON object like {"output": ["item 1", "item 2"]}.'
  if family == "plain_text_json":
    return 'Return a JSON object like {"output": "text"} with a single string output.'
  if family == "generic_json":
    return (
      "Return a JSON object with a field named output that matches the requested structure. "
      "Keep the value types consistent with the prompt requirements."
    )
  if family == "chat_transcript":
    return (
      'Return a JSON object like {"output": [{"speaker": "Name", "utterance": "Text"}]}. '
      "Keep speaker names and utterances concise and complete."
    )
  if family == "dict_list_int":
    return (
      'Return a JSON object where output is a dictionary whose keys are thoughts '
      "and whose values are lists of integer evidence ids."
    )
  if family == "utterance_turn":
    return 'Return a JSON object like {"output": {"utterance": "Text", "end": false}}.'
  if family == "schedule_decomp":
    return (
      'Return a JSON object where output is an array of {"task": "...", "duration": <minutes>} objects. '
      "duration must be an integer number of minutes."
    )
  if family == "safety_score":
    return "Return a JSON object with integer field output in the range 0 to 10."
  raise OutputFamilyError("Unsupported structured family: %s" % family)


def build_primary_messages(
  prompt: str,
  family: str = "plain_text",
  options: Optional[Dict[str, Any]] = None,
  extra_system: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
  system_parts = [
    "You are helping simulate generative agents. Follow the user prompt exactly.",
    primary_family_instruction(family, options),
  ]
  if extra_system:
    system_parts.extend([part for part in extra_system if part])
  return [
    {"role": "system", "content": "\n".join(system_parts).strip()},
    {"role": "user", "content": prompt},
  ]


def build_structured_messages(
  prompt: str,
  family: str,
  options: Optional[Dict[str, Any]] = None,
  extra_system: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
  system_parts = [
    "You are helping simulate generative agents. Follow the user prompt exactly.",
    structured_family_instruction(family, options),
    "Return valid JSON only. Do not wrap the JSON in markdown fences.",
  ]
  if extra_system:
    system_parts.extend([part for part in extra_system if part])
  return [
    {"role": "system", "content": "\n".join(system_parts).strip()},
    {"role": "user", "content": prompt},
  ]


def parse_primary_output(raw_text: str, family: str = "plain_text", options: Optional[Dict[str, Any]] = None) -> Any:
  options = dict(options or {})
  text = (raw_text or "").strip()
  if family in ("plain_text", "quoted_text"):
    if options.get("split_quote"):
      text = text.split('"')[0].strip()
    if options.get("trim_trailing_period") and text.endswith("."):
      text = text[:-1].rstrip()
    if options.get("max_length") is not None:
      text = text[: int(options["max_length"])].strip()
    if options.get("prefix"):
      return str(options["prefix"]) + text
    return text
  if family == "numbered_list":
    return _parse_numbered_list(text)
  raise OutputFamilyError("Unsupported primary family: %s" % family)


def parse_structured_output(value: Any, family: str, options: Optional[Dict[str, Any]] = None) -> Any:
  options = dict(options or {})
  if family in ("integer", "safety_score"):
    return int(value)
  if family == "enum_string":
    normalized = str(value).strip()
    choices = list(options.get("choices", []))
    if choices and normalized not in choices:
      raise OutputFamilyError("Value %r is not one of %s" % (normalized, choices))
    return normalized
  if family == "boolean":
    return bool(value)
  if family == "triple":
    return [str(value["predicate"]).strip(), str(value["object"]).strip()]
  if family == "list_str":
    return [str(item).strip() for item in list(value)]
  if family == "plain_text_json":
    return str(value).strip()
  if family == "generic_json":
    return value
  if family == "chat_transcript":
    transcript = []
    for item in list(value):
      transcript.append([str(item["speaker"]).strip(), str(item["utterance"]).strip()])
    return transcript
  if family == "dict_list_int":
    normalized_dict: Dict[str, List[int]] = {}
    for key, entries in dict(value).items():
      normalized_dict[str(key).strip()] = [int(entry) for entry in list(entries)]
    return normalized_dict
  if family == "utterance_turn":
    return {
      "utterance": str(value["utterance"]).strip(),
      "end": bool(value["end"]),
    }
  if family == "schedule_decomp":
    normalized_schedule = []
    for item in list(value):
      normalized_schedule.append([str(item["task"]).strip(), int(item["duration"])])
    return normalized_schedule
  raise OutputFamilyError("Unsupported structured family: %s" % family)


def _parse_numbered_list(text: str) -> List[str]:
  normalized = " ".join((text or "").replace("\r", "\n").split())
  if not normalized:
    return []

  marker_regex = re.compile(r"(?<!\w)(\d+[\).])\s+")
  matches = list(marker_regex.finditer(normalized))
  items: List[str] = []
  if matches:
    for index, match in enumerate(matches):
      start = match.end()
      end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
      candidate = normalized[start:end].strip(" ,.;")
      if candidate:
        items.append(candidate)
  if items:
    return items

  for line in (text or "").splitlines():
    candidate = re.sub(r"^\s*(?:[-*]|\d+[\).])\s*", "", line).strip(" ,.;")
    if candidate:
      items.append(candidate)
  return items
