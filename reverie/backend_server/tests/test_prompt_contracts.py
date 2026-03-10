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
from persona.prompt_template import run_gpt_prompt as prompts


class ScratchStub:
  def __init__(self, name):
    self.name = name
    self.first_name = name.split()[0]
    self.last_name = name.split()[-1]
    self.curr_time = datetime.datetime(2026, 3, 10, 9, 0, 0)
    self.curr_tile = (0, 0)
    self.living_area = "world:home"
    self.currently = f"{name} is planning the day."
    self.daily_req = ["eat breakfast", "work on art"]
    self.f_daily_schedule_hourly_org = [
      ["cook breakfast", 60],
      ["work on art", 120],
      ["relax", 60],
    ]
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

  def get_str_curr_date_str(self):
    return "Tuesday March 10"

  def get_str_name(self):
    return self.name

  def get_f_daily_schedule_hourly_org_index(self):
    return 0

  def get_str_daily_plan_req(self):
    return "1) breakfast, 2) work on art"


class SpatialMemoryStub:
  def get_str_accessible_sectors(self, world):
    return "home, library"

  def get_str_accessible_sector_arenas(self, key):
    if key.endswith(":home"):
      return "kitchen, bedroom"
    return "reading room, lobby"

  def get_str_accessible_arena_game_objects(self, address):
    if "reading room" in address:
      return "desk, chair"
    return "stove, table"


class AssociativeMemoryStub:
  def __init__(self):
    self.seq_chat = []

  def get_last_chat(self, _name):
    return None

  def retrieve_relevant_thoughts(self, *_args):
    return [SimpleNamespace(description="remembers a useful detail")]


class PersonaStub:
  def __init__(self, name):
    self.name = name
    self.scratch = ScratchStub(name)
    self.s_mem = SpatialMemoryStub()
    self.a_mem = AssociativeMemoryStub()


class MazeStub:
  def access_tile(self, _tile):
    return {
      "world": "world",
      "sector": "library",
      "arena": "reading room",
    }


class PromptContractTests(unittest.TestCase):
  def setUp(self):
    prompts.debug = False
    self.persona = PersonaStub("Alice Smith")
    self.target = PersonaStub("Bob Jones")
    self.maze = MazeStub()
    self.retrieved = {
      "events": [SimpleNamespace(description="Alice is reading quietly")],
      "thoughts": [SimpleNamespace(description="Alice enjoys focused mornings")],
    }

  def _run_prompt(self, func, args, raw_value, expected_lane, assert_spec=None):
    def fake_run_task(spec):
      self.assertEqual(spec.lane, expected_lane)
      messages = spec.messages_builder(None)
      self.assertEqual([message["role"] for message in messages], ["system", "user"])
      if assert_spec:
        assert_spec(spec, messages)
      if spec.lane == "structured":
        parsed = spec.parser({"output": raw_value}) if spec.parser else {"output": raw_value}
        raw_text = json.dumps({"output": raw_value}, ensure_ascii=False)
      else:
        parsed = spec.parser(raw_value) if spec.parser else raw_value
        raw_text = raw_value if isinstance(raw_value, str) else str(raw_value)
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

    with ExitStack() as stack:
      stack.enter_context(
        patch.object(
          prompts,
          "generate_prompt",
          side_effect=lambda prompt_input, prompt_template: f"{prompt_template}::{len(prompt_input)}",
        )
      )
      stack.enter_context(patch.object(prompts, "run_task", side_effect=fake_run_task))
      return func(*args)

  def test_wake_up_hour_uses_structured_integer_schema(self):
    def assert_spec(spec, messages):
      self.assertEqual(spec.metadata["output_family"], "integer")
      schema = spec.schema["properties"]["output"]
      self.assertEqual(schema["type"], "integer")
      self.assertEqual(schema["minimum"], 1)
      self.assertEqual(schema["maximum"], 12)
      self.assertIn("Return a JSON object with an integer field named output.", messages[0]["content"])

    output, _details = self._run_prompt(
      prompts.run_gpt_prompt_wake_up_hour,
      (self.persona,),
      7,
      "structured",
      assert_spec,
    )
    self.assertEqual(output, 7)

  def test_daily_plan_uses_primary_numbered_list_and_preserves_shape(self):
    def assert_spec(spec, messages):
      self.assertEqual(spec.metadata["output_family"], "numbered_list")
      self.assertEqual(spec.stop, None)
      self.assertIn("Return only the numbered items as plain text.", messages[0]["content"])

    output, _details = self._run_prompt(
      prompts.run_gpt_prompt_daily_plan,
      (self.persona, 6),
      "1. eat breakfast at 7:00 am\n2. work on art from 8:00 am to 10:00 am",
      "primary",
      assert_spec,
    )
    self.assertEqual(output[0], "wake up and complete the morning routine at 6:00 am")
    self.assertIn("eat breakfast at 7:00 am", output[1])

  def test_task_decomp_and_new_schedule_keep_list_of_lists_shape(self):
    task_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_task_decomp,
      (self.persona, "cook breakfast", 60),
      [
        {"task": "prep ingredients", "duration": 20},
        {"task": "cook food", "duration": 40},
      ],
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "schedule_decomp"),
    )
    self.assertEqual(
      task_output,
      [["cook breakfast (prep ingredients)", 20], ["cook breakfast (cook food)", 40]],
    )

    new_schedule, _details = self._run_prompt(
      prompts.run_gpt_prompt_new_decomp_schedule,
      (
        self.persona,
        [["work on art", 60], ["relax", 30]],
        [["work on art", 30]],
        datetime.datetime(2026, 3, 10, 9, 0, 0),
        datetime.datetime(2026, 3, 10, 10, 30, 0),
        "talk with Bob",
        15,
      ),
      [
        {"task": "work on art", "duration": 45},
        {"task": "talk with Bob", "duration": 15},
        {"task": "relax", "duration": 30},
      ],
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "schedule_decomp"),
    )
    self.assertEqual(new_schedule[0], ["work on art", 45])
    self.assertEqual(new_schedule[1], ["talk with Bob", 15])

  def test_dynamic_enum_tasks_use_current_choices(self):
    sector_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_action_sector,
      ("reading", self.persona, self.maze),
      "library",
      "structured",
      lambda spec, _messages: self.assertEqual(
        spec.schema["properties"]["output"]["enum"],
        ["home", "library"],
      ),
    )
    self.assertEqual(sector_output, "library")

    arena_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_action_arena,
      ("reading", self.persona, self.maze, "world", "library"),
      "reading room",
      "structured",
      lambda spec, _messages: self.assertEqual(
        spec.schema["properties"]["output"]["enum"],
        ["reading room", "lobby"],
      ),
    )
    self.assertEqual(arena_output, "reading room")

    object_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_action_game_object,
      ("reading", self.persona, self.maze, "world:library:reading room"),
      "desk",
      "structured",
      lambda spec, _messages: self.assertEqual(
        spec.schema["properties"]["output"]["enum"],
        ["desk", "chair"],
      ),
    )
    self.assertEqual(object_output, "desk")

    react_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_decide_to_react,
      (self.persona, self.target, self.retrieved),
      "2",
      "structured",
      lambda spec, _messages: self.assertEqual(
        spec.schema["properties"]["output"]["enum"],
        ["1", "2", "3"],
      ),
    )
    self.assertEqual(react_output, "2")

  def test_decide_to_talk_uses_yes_no_enum(self):
    output, _details = self._run_prompt(
      prompts.run_gpt_prompt_decide_to_talk,
      (self.persona, self.target, self.retrieved),
      "yes",
      "structured",
      lambda spec, _messages: self.assertEqual(
        spec.schema["properties"]["output"]["enum"],
        ["yes", "no"],
      ),
    )
    self.assertEqual(output, "yes")

  def test_triple_tasks_return_existing_tuple_shape(self):
    event_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_event_triple,
      ("reading a book", self.persona),
      {"predicate": "is", "object": "reading"},
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "triple"),
    )
    self.assertEqual(event_output, ("Alice Smith", "is", "reading"))

    object_output, _details = self._run_prompt(
      prompts.run_gpt_prompt_act_obj_event_triple,
      ("desk", "desk is being used", self.persona),
      {"predicate": "is", "object": "in use"},
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "triple"),
    )
    self.assertEqual(object_output, ("desk", "is", "in use"))

  def test_list_and_dict_families_keep_existing_shapes(self):
    keywords, _details = self._run_prompt(
      prompts.run_gpt_prompt_extract_keywords,
      (self.persona, "Morning Routine\nWork Session"),
      ["Routine.", "Work"],
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "list_str"),
    )
    self.assertEqual(keywords, {"routine", "work"})

    focal_points, _details = self._run_prompt(
      prompts.run_gpt_prompt_focal_pt,
      (self.persona, "Observation one. Observation two.", 2),
      ["Who am I", "What matters today"],
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "list_str"),
    )
    self.assertEqual(focal_points, ["Who am I", "What matters today"])

    insights, _details = self._run_prompt(
      prompts.run_gpt_prompt_insight_and_guidance,
      (self.persona, "Observation one. Observation two.", 2),
      {"Alice values routine": [1, 2]},
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "dict_list_int"),
    )
    self.assertEqual(insights, {"Alice values routine": [1, 2]})

    safety_score, _details = self._run_prompt(
      prompts.run_gpt_generate_safety_score,
      (self.persona, "This is safe."),
      4,
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "safety_score"),
    )
    self.assertEqual(safety_score, 4)

  def test_chat_transcript_and_iterative_turn_tasks_keep_existing_shapes(self):
    conversation, _details = self._run_prompt(
      prompts.run_gpt_prompt_create_conversation,
      (self.persona, self.target, {"arena": "reading room"}),
      [
        {"speaker": "Alice Smith", "utterance": "Morning."},
        {"speaker": "Bob Jones", "utterance": "Morning."},
      ],
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "chat_transcript"),
    )
    self.assertEqual(conversation[0], ["Alice Smith", "Morning."])

    agent_chat, _details = self._run_prompt(
      prompts.run_gpt_prompt_agent_chat,
      (self.maze, self.persona, self.target, "At the library", "Alice wants to focus", "Bob wants to say hello"),
      [
        {"speaker": "Alice Smith", "utterance": "Hi Bob."},
        {"speaker": "Bob Jones", "utterance": "Hi Alice."},
      ],
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "chat_transcript"),
    )
    self.assertEqual(agent_chat[1], ["Bob Jones", "Hi Alice."])

    iterative_turn, _details = self._run_prompt(
      prompts.run_gpt_generate_iterative_chat_utt,
      (self.maze, self.persona, self.target, self.retrieved, "At the library", [["Alice Smith", "Hi"]]),
      {"utterance": "How is your morning?", "end": False},
      "structured",
      lambda spec, _messages: self.assertEqual(spec.metadata["output_family"], "utterance_turn"),
    )
    self.assertEqual(iterative_turn, {"utterance": "How is your morning?", "end": False})

  def test_static_regression_removed_old_prompt_parsers_and_models(self):
    run_prompt_source = (BACKEND_SERVER_ROOT / "persona" / "prompt_template" / "run_gpt_prompt.py").read_text(encoding="utf-8")
    gpt_structure_source = (BACKEND_SERVER_ROOT / "persona" / "prompt_template" / "gpt_structure.py").read_text(encoding="utf-8")
    self.assertNotIn("text-davinci", run_prompt_source)
    self.assertNotIn("extract_first_json_dict", run_prompt_source)
    self.assertNotIn("json.loads(", run_prompt_source)
    self.assertNotIn("ast.literal_eval", run_prompt_source)
    self.assertNotIn('{"role": "user", "content": prompt}', gpt_structure_source)
    self.assertNotIn("openai.ChatCompletion.create", run_prompt_source + gpt_structure_source)
    self.assertNotIn("openai.Completion.create", run_prompt_source + gpt_structure_source)
    self.assertNotIn("openai.Embedding.create", run_prompt_source + gpt_structure_source)


if __name__ == "__main__":
  unittest.main()
