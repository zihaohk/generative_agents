"""
Simple smoke script for the new LLM layer.

Usage:
  python test.py text
  python test.py structured
  python test.py embed
"""
import json
import sys

from llm import embed_texts, generate_structured, generate_text


def run_text():
  result = generate_text(
    "smoke_text",
    [{"role": "user", "content": "Reply with exactly: smoke test ok"}],
  )
  print(result.value)


def run_structured():
  result = generate_structured(
    "smoke_structured",
    [{"role": "user", "content": "Return JSON with {'output': {'status': 'ok'}}"}],
    {
      "type": "object",
      "properties": {
        "output": {
          "type": "object",
          "properties": {
            "status": {"type": "string"},
          },
          "required": ["status"],
        },
      },
      "required": ["output"],
    },
  )
  print(json.dumps(result.value, ensure_ascii=False))


def run_embed():
  result = embed_texts(["smoke embedding"])
  vector = result.value[0]
  print(f"embedding_length={len(vector)}")


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "text"
  if mode == "text":
    run_text()
  elif mode == "structured":
    run_structured()
  elif mode == "embed":
    run_embed()
  else:
    raise SystemExit(f"Unknown mode: {mode}")
