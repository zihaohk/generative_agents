import datetime
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER_ROOT) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER_ROOT))

from llm.api import TaskResult
from persona import persona as persona_module
from persona.prompt_template import run_gpt_prompt as prompts


class ScratchStub:
  def __init__(self, name):
    self.name = name
    self.first_name = name.split()[0]
    self.last_name = name.split()[-1]
    self.curr_time = None
    self.curr_tile = (0, 0)
    self.living_area = "world:home"
    self.currently = f"{name} is planning the day."
    self.act_description = "reading a book"
    self.planned_path = []
    self.act_address = "world:library:reading room:desk"
    self.act_event = (name, "is", "reading")

  def get_str_iss(self):
    return f"{self.name} likes routines."

  def get_str_lifestyle(self):
    return "Sleeps early and works in the morning."

  def get_str_firstname(self):
    return self.first_name


class AssociativeMemoryStub:
  def __init__(self):
    self.seq_chat = []

  def get_last_chat(self, _name):
    return None


class PersonaMoveSmokeTests(unittest.TestCase):
  def _persona(self, name):
    persona = persona_module.Persona.__new__(persona_module.Persona)
    persona.name = name
    persona.scratch = ScratchStub(name)
    persona.a_mem = AssociativeMemoryStub()
    return persona

  def test_move_smoke_runs_prompt_tasks_and_returns_execution_triple(self):
    prompts.debug = False
    init_persona = self._persona("Alice Smith")
    target_persona = self._persona("Bob Jones")
    maze = SimpleNamespace(
      access_tile=lambda _tile: {
        "world": "world",
        "sector": "library",
        "arena": "reading room",
      }
    )
    personas = {
      "Alice Smith": init_persona,
      "Bob Jones": target_persona,
    }
    retrieved = {
      "events": [SimpleNamespace(description="Alice is reading quietly")],
      "thoughts": [SimpleNamespace(description="Alice enjoys focused mornings")],
    }

    def fake_run_task(spec):
      payloads = {
        "wake_up_hour": 7,
        "decide_to_react": "2",
      }
      raw_value = payloads[spec.name]
      if spec.lane == "structured":
        parsed = spec.parser({"output": raw_value})
        raw_text = json.dumps({"output": raw_value}, ensure_ascii=False)
      else:
        parsed = spec.parser(raw_value) if spec.parser else raw_value
        raw_text = str(raw_value)
      return TaskResult(
        value=parsed,
        raw_text=raw_text,
        model="mock-model",
        usage={},
        trace_id="trace",
        retries=0,
        fallback_used=False,
        parsed_ok=True,
        error_type=None,
        provider="mock",
        lane=spec.lane,
      )

    def fake_plan(persona, current_maze, current_personas, new_day, current_retrieved):
      wake_up_hour = prompts.run_gpt_prompt_wake_up_hour(persona)[0]
      reaction = prompts.run_gpt_prompt_decide_to_react(
        persona,
        current_personas["Bob Jones"],
        current_retrieved,
      )[0]
      return {
        "wake_up_hour": wake_up_hour,
        "reaction": reaction,
        "new_day": new_day,
      }

    def fake_execute(_persona, _maze, _personas, plan_payload):
      return ((1, 1), ":)", f"wake={plan_payload['wake_up_hour']} react={plan_payload['reaction']}")

    with ExitStack() as stack:
      stack.enter_context(
        patch.object(
          prompts,
          "generate_prompt",
          side_effect=lambda prompt_input, prompt_template: f"{prompt_template}::{len(prompt_input)}",
        )
      )
      stack.enter_context(patch.object(prompts, "run_task", side_effect=fake_run_task))
      stack.enter_context(patch.object(persona_module, "perceive", return_value=["perceived"]))
      stack.enter_context(patch.object(persona_module, "retrieve", return_value=retrieved))
      stack.enter_context(patch.object(persona_module, "plan", side_effect=fake_plan))
      stack.enter_context(patch.object(persona_module, "reflect", return_value=None))
      stack.enter_context(patch.object(persona_module, "execute", side_effect=fake_execute))
      execution = init_persona.move(
        maze,
        personas,
        (3, 4),
        datetime.datetime(2026, 3, 10, 9, 0, 0),
      )

    self.assertEqual(execution[0], (1, 1))
    self.assertEqual(execution[1], ":)")
    self.assertEqual(execution[2], "wake=7 react=2")


if __name__ == "__main__":
  unittest.main()
