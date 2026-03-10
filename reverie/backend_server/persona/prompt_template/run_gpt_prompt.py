"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: run_gpt_prompt.py
Description: Defines all run gpt prompt functions. These functions directly
interface with the safe_generate_response function.
"""
import re
import datetime
import sys

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *
from llm import run_task, TaskSpec
from llm.families import (
  build_output_schema,
  build_primary_messages,
  build_structured_messages,
  parse_primary_output,
  parse_structured_output,
  structured_family_schema,
)

def get_random_alphanumeric(i=6, j=6): 
  """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
  k = random.randint(i, j)
  x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
  return x


def _task_details(output, prompt, sampling, prompt_input, fallback):
  return [output, prompt, sampling, prompt_input, fallback]


def _emit_prompt_debug(prompt_template, persona, sampling, prompt_input, prompt, output, verbose):
  if debug or verbose:
    print_run_prompts(prompt_template, persona, sampling, prompt_input, prompt, output)


def _sampling_config(sampling):
  sampling = dict(sampling or {})
  return {
    "temperature": sampling.get("temperature", 0),
    "max_tokens": sampling.get("max_tokens"),
    "stop": sampling.get("stop"),
  }


def _build_primary_parser(prompt, family="plain_text", family_options=None, func_validate=None, func_clean_up=None):
  family_options = dict(family_options or {})

  def parser(raw_text):
    value = parse_primary_output(raw_text, family, family_options)
    if func_validate and not func_validate(value, prompt=prompt):
      raise ValueError("legacy parser validation failed")
    if func_clean_up:
      return func_clean_up(value, prompt=prompt)
    return value

  return parser


def _build_structured_parser(prompt, family, family_options=None, func_validate=None, func_clean_up=None):
  family_options = dict(family_options or {})

  def parser(payload):
    candidate = parse_structured_output(payload["output"], family, family_options)
    if func_validate and not func_validate(candidate, prompt=prompt):
      raise ValueError("structured output validation failed")
    if func_clean_up:
      return func_clean_up(candidate, prompt=prompt)
    return candidate

  return parser


def _run_primary_prompt_task(
  task_name,
  prompt_template,
  prompt_input,
  persona,
  sampling,
  fallback,
  family="plain_text",
  family_options=None,
  func_validate=None,
  func_clean_up=None,
  verbose=False,
  metadata=None,
  extra_system=None,
):
  prompt = generate_prompt(prompt_input, prompt_template)
  sampling_config = _sampling_config(sampling)
  task_spec = TaskSpec(
    name=task_name,
    lane="primary",
    messages_builder=lambda _context=None: build_primary_messages(
      prompt,
      family=family,
      options=family_options,
      extra_system=extra_system,
    ),
    parser=_build_primary_parser(prompt, family, family_options, func_validate, func_clean_up),
    sampling=sampling_config,
    stop=sampling_config.get("stop"),
    fallback=fallback,
    metadata={
      "prompt_template": prompt_template,
      "output_family": family,
      **(metadata or {}),
    },
  )
  result = run_task(task_spec)
  output = result.value
  _emit_prompt_debug(prompt_template, persona, sampling, prompt_input, prompt, output, verbose)
  return output, _task_details(output, prompt, sampling, prompt_input, fallback)


def _run_raw_primary_task(
  task_name,
  prompt,
  persona,
  sampling,
  fallback,
  family="plain_text",
  family_options=None,
  func_validate=None,
  func_clean_up=None,
  verbose=False,
  metadata=None,
  extra_system=None,
):
  sampling_config = _sampling_config(sampling)
  task_spec = TaskSpec(
    name=task_name,
    lane="primary",
    messages_builder=lambda _context=None: build_primary_messages(
      prompt,
      family=family,
      options=family_options,
      extra_system=extra_system,
    ),
    parser=_build_primary_parser(prompt, family, family_options, func_validate, func_clean_up),
    sampling=sampling_config,
    stop=sampling_config.get("stop"),
    fallback=fallback,
    metadata={
      "output_family": family,
      **(metadata or {}),
    },
  )
  result = run_task(task_spec)
  output = result.value
  if debug or verbose:
    print_run_prompts(None, persona, sampling, None, prompt, output)
  return output, _task_details(output, prompt, sampling, None, fallback)


def _run_structured_prompt_task(
  task_name,
  prompt_template,
  prompt_input,
  persona,
  sampling,
  fallback,
  family,
  family_options=None,
  func_validate=None,
  func_clean_up=None,
  verbose=False,
  max_retries=3,
  metadata=None,
  extra_system=None,
):
  prompt = generate_prompt(prompt_input, prompt_template)
  sampling_config = _sampling_config(sampling)
  schema = build_output_schema(structured_family_schema(family, family_options))
  task_spec = TaskSpec(
    name=task_name,
    lane="structured",
    messages_builder=lambda _context=None: build_structured_messages(
      prompt,
      family=family,
      options=family_options,
      extra_system=extra_system,
    ),
    parser=_build_structured_parser(prompt, family, family_options, func_validate, func_clean_up),
    schema=schema,
    sampling=sampling_config,
    stop=sampling_config.get("stop"),
    fallback=fallback,
    max_retries=max_retries,
    metadata={
      "prompt_template": prompt_template,
      "output_family": family,
      **(metadata or {}),
    },
  )
  result = run_task(task_spec)
  output = result.value
  _emit_prompt_debug(prompt_template, persona, sampling, prompt_input, prompt, output, verbose)
  return output, _task_details(output, prompt, sampling, prompt_input, fallback)


def _run_json_output_task(
  task_name,
  prompt_template,
  prompt_input,
  persona,
  sampling,
  fallback,
  example_output,
  special_instruction,
  func_validate=None,
  func_clean_up=None,
  verbose=False,
  max_retries=3,
  metadata=None,
):
  return _run_structured_prompt_task(
    task_name=task_name,
    prompt_template=prompt_template,
    prompt_input=prompt_input,
    persona=persona,
    sampling=sampling,
    fallback=fallback,
    family="plain_text_json",
    family_options={
      "example_output": example_output,
      "special_instruction": special_instruction,
    },
    func_validate=func_validate,
    func_clean_up=func_clean_up,
    verbose=verbose,
    max_retries=max_retries,
    metadata=metadata,
    extra_system=[special_instruction],
  )


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False): 
  """
  Given the persona, returns an integer that indicates the hour when the 
  persona wakes up.  
  """
  prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
  prompt_input = [persona.scratch.get_str_iss(),
                  persona.scratch.get_str_lifestyle(),
                  persona.scratch.get_str_firstname()]
  sampling = {"temperature": 0.8, "max_tokens": 5, "stop": ["\n"]}
  return _run_structured_prompt_task(
    "wake_up_hour",
    prompt_template,
    prompt_input,
    persona,
    sampling,
    8,
    family="integer",
    family_options={"minimum": 1, "maximum": 12},
    verbose=verbose,
  )


def run_gpt_prompt_daily_plan(persona, 
                              wake_up_hour, 
                              test_input=None, 
                              verbose=False):
  """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. 
  """
  prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
  prompt_input = [persona.scratch.get_str_iss(),
                  persona.scratch.get_str_lifestyle(),
                  persona.scratch.get_str_curr_date_str(),
                  persona.scratch.get_str_firstname(),
                  f"{str(wake_up_hour)}:00 am"]
  sampling = {"temperature": 1.0, "max_tokens": 500}
  fallback = ['wake up and complete the morning routine at 6:00 am', 
              'eat breakfast at 7:00 am', 
              'read a book from 8:00 am to 12:00 pm', 
              'have lunch at 12:00 pm', 
              'take a nap from 1:00 pm to 4:00 pm', 
              'relax and watch TV from 7:00 pm to 8:00 pm', 
              'go to bed at 11:00 pm']
  output, debug_info = _run_primary_prompt_task(
    "daily_plan",
    prompt_template,
    prompt_input,
    persona,
    sampling,
    fallback,
    family="numbered_list",
    verbose=verbose,
  )
  output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)
  return output, _task_details(output, debug_info[1], sampling, prompt_input, fallback)


def run_gpt_prompt_generate_hourly_schedule(persona, 
                                            curr_hour_str,
                                            p_f_ds_hourly_org, 
                                            hour_str,
                                            intermission2=None,
                                            test_input=None, 
                                            verbose=False): 
  schedule_format = ""
  for i in hour_str: 
    schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
    schedule_format += f" Activity: [Fill in]\n"
  schedule_format = schedule_format[:-1]

  intermission_str = f"Here the originally intended hourly breakdown of"
  intermission_str += f" {persona.scratch.get_str_firstname()}'s schedule today: "
  for count, i in enumerate(persona.scratch.daily_req): 
    intermission_str += f"{str(count+1)}) {i}, "
  intermission_str = intermission_str[:-2]

  prior_schedule = ""
  if p_f_ds_hourly_org: 
    prior_schedule = "\n"
    for count, i in enumerate(p_f_ds_hourly_org): 
      prior_schedule += f"[(ID:{get_random_alphanumeric()})" 
      prior_schedule += f" {persona.scratch.get_str_curr_date_str()} --"
      prior_schedule += f" {hour_str[count]}] Activity:"
      prior_schedule += f" {persona.scratch.get_str_firstname()}"
      prior_schedule += f" is {i}\n"

  prompt_ending = f"[(ID:{get_random_alphanumeric()})"
  prompt_ending += f" {persona.scratch.get_str_curr_date_str()}"
  prompt_ending += f" -- {curr_hour_str}] Activity:"
  prompt_ending += f" {persona.scratch.get_str_firstname()} is"

  prompt_input = [schedule_format,
                  persona.scratch.get_str_iss(),
                  prior_schedule + "\n",
                  intermission_str,
                  f"\n{intermission2}" if intermission2 else "",
                  prompt_ending]
  return _run_primary_prompt_task(
    "generate_hourly_schedule",
    "persona/prompt_template/v2/generate_hourly_schedule_v2.txt",
    prompt_input,
    persona,
    {"temperature": 0.5, "max_tokens": 50, "stop": ["\n"]},
    "asleep",
    family="plain_text",
    family_options={"trim_trailing_period": True},
    verbose=verbose,
  )








def run_gpt_prompt_task_decomp(persona, 
                               task, 
                               duration, 
                               test_input=None, 
                               verbose=False): 
  curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
  all_indices = [curr_f_org_index]
  if curr_f_org_index+1 <= len(persona.scratch.f_daily_schedule_hourly_org): 
    all_indices += [curr_f_org_index+1]
  if curr_f_org_index+2 <= len(persona.scratch.f_daily_schedule_hourly_org): 
    all_indices += [curr_f_org_index+2]

  curr_time_range = ""
  summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
  summ_str += f'From '
  for index in all_indices: 
    if index < len(persona.scratch.f_daily_schedule_hourly_org): 
      start_min = 0
      for i in range(index): 
        start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
      end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
      start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                    + datetime.timedelta(minutes=start_min)) 
      end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
                  + datetime.timedelta(minutes=end_min)) 
      start_time_str = start_time.strftime("%H:%M%p")
      end_time_str = end_time.strftime("%H:%M%p")
      summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
      if curr_f_org_index+1 == index:
        curr_time_range = f'{start_time_str} ~ {end_time_str}'
  summ_str = summ_str[:-2] + "."

  prompt_input = [persona.scratch.get_str_iss(),
                  summ_str,
                  persona.scratch.get_str_firstname(),
                  persona.scratch.get_str_firstname(),
                  task,
                  curr_time_range,
                  duration,
                  persona.scratch.get_str_firstname()]
  sampling = {"temperature": 0, "max_tokens": 1000}
  fallback = [["asleep", duration]]
  output, debug_info = _run_structured_prompt_task(
    "task_decomp",
    "persona/prompt_template/v2/task_decomp_v3.txt",
    prompt_input,
    persona,
    sampling,
    fallback,
    family="schedule_decomp",
    extra_system=[f"The durations in output must sum to exactly {int(duration)} minutes."],
    verbose=verbose,
  )

  fin_output = []
  time_sum = 0
  for i_task, i_duration in output: 
    time_sum += i_duration
    if time_sum <= duration: 
      fin_output += [[i_task, i_duration]]
    else: 
      break
  ftime_sum = 0
  for fi_task, fi_duration in fin_output: 
    ftime_sum += fi_duration
  
  if fin_output:
    fin_output[-1][1] += (duration - ftime_sum)
  else:
    fin_output = [[task, duration]]
  output = fin_output 

  ret = []
  for decomp_task, dur in output: 
    ret += [[f"{task} ({decomp_task})", dur]]
  output = ret

  return output, _task_details(output, debug_info[1], sampling, prompt_input, fallback)



def run_gpt_prompt_action_sector(action_description, 
                                persona, 
                                maze, 
                                test_input=None, 
                                verbose=False):
  act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
  curr = accessible_sector_str.split(", ")
  fin_accessible_sectors = []
  for item in curr:
    if "'s house" in item:
      if persona.scratch.last_name in item:
        fin_accessible_sectors += [item]
    else:
      fin_accessible_sectors += [item]
  accessible_sector_str = ", ".join(fin_accessible_sectors)

  action_description_1 = action_description
  action_description_2 = action_description
  if "(" in action_description:
    action_description_1 = action_description.split("(")[0].strip()
    action_description_2 = action_description.split("(")[-1][:-1]

  prompt_input = [persona.scratch.get_str_name(),
                  persona.scratch.living_area.split(":")[1],
                  persona.s_mem.get_str_accessible_sector_arenas(f"{act_world}:{persona.scratch.living_area.split(':')[1]}"),
                  persona.scratch.get_str_name(),
                  f"{maze.access_tile(persona.scratch.curr_tile)['sector']}",
                  persona.s_mem.get_str_accessible_sector_arenas(f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"),
                  f"\n{persona.scratch.get_str_daily_plan_req()}" if persona.scratch.get_str_daily_plan_req() != "" else "",
                  accessible_sector_str,
                  persona.scratch.get_str_name(),
                  action_description_1,
                  action_description_2,
                  persona.scratch.get_str_name()]
  fallback = persona.scratch.living_area.split(":")[1]
  choices = fin_accessible_sectors or [fallback]
  output, debug_info = _run_structured_prompt_task(
    "action_sector",
    "persona/prompt_template/v1/action_location_sector_v1.txt",
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fallback,
    family="enum_string",
    family_options={"choices": choices},
    verbose=verbose,
  )

  y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
  if output not in x: 
    output = persona.scratch.living_area.split(":")[1]

  return output, _task_details(output, debug_info[1], {"temperature": 0, "max_tokens": 15}, prompt_input, fallback)



def run_gpt_prompt_action_arena(action_description, 
                                persona, 
                                maze, act_world, act_sector,
                                test_input=None, 
                                verbose=False):
  x = f"{act_world}:{act_sector}"
  accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
  curr = accessible_arena_str.split(", ")
  fin_accessible_arenas = []
  for item in curr: 
    if "'s room" in item: 
      if persona.scratch.last_name in item: 
        fin_accessible_arenas += [item]
    else: 
      fin_accessible_arenas += [item]
  accessible_arena_str = ", ".join(fin_accessible_arenas)

  action_description_1 = action_description
  action_description_2 = action_description
  if "(" in action_description: 
    action_description_1 = action_description.split("(")[0].strip()
    action_description_2 = action_description.split("(")[-1][:-1]

  prompt_input = [persona.scratch.get_str_name(),
                  act_sector,
                  accessible_arena_str,
                  persona.scratch.get_str_name(),
                  action_description_1,
                  action_description_2,
                  persona.scratch.get_str_name(),
                  act_sector,
                  accessible_arena_str]
  fallback = fin_accessible_arenas[0] if fin_accessible_arenas else "kitchen"
  return _run_structured_prompt_task(
    "action_arena",
    "persona/prompt_template/v1/action_location_object_vMar11.txt",
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fallback,
    family="enum_string",
    family_options={"choices": fin_accessible_arenas or [fallback]},
    verbose=verbose,
  )



def run_gpt_prompt_action_game_object(action_description, 
                                      persona, 
                                      maze,
                                      temp_address,
                                      test_input=None, 
                                      verbose=False): 
  desc = action_description.split("(")[-1][:-1] if "(" in action_description else action_description
  prompt_input = [desc, persona.s_mem.get_str_accessible_arena_game_objects(temp_address)]
  choices = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",") if i.strip()]
  fallback = choices[0] if choices else "bed"
  output, debug_info = _run_structured_prompt_task(
    "action_game_object",
    "persona/prompt_template/v1/action_object_v2.txt",
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fallback,
    family="enum_string",
    family_options={"choices": choices or [fallback]},
    verbose=verbose,
  )
  if output not in choices and choices:
    output = random.choice(choices)
  return output, _task_details(output, debug_info[1], {"temperature": 0, "max_tokens": 15}, prompt_input, fallback)




def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False): 
  prompt_input = [action_description.split("(")[-1].split(")")[0] if "(" in action_description else action_description]
  return _run_primary_prompt_task(
    "pronunciatio",
    "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt",
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    ":)",
    family="plain_text",
    family_options={"max_length": 3},
    verbose=verbose,
  )












def run_gpt_prompt_event_triple(action_description, persona, verbose=False): 
  desc = action_description.split("(")[-1].split(")")[0] if "(" in action_description else action_description
  prompt_input = [persona.name, desc, persona.name]
  sampling = {"temperature": 0, "max_tokens": 30, "stop": ["\n"]}
  fallback = (persona.name, "is", "idle")
  output, debug_info = _run_structured_prompt_task(
    "event_triple",
    "persona/prompt_template/v2/generate_event_triple_v1.txt",
    prompt_input,
    persona,
    sampling,
    fallback,
    family="triple",
    verbose=verbose,
  )
  try:
    if len(output) == 3:
      output = tuple(output)
    else:
      output = (persona.name, output[0], output[1])
  except Exception:
    output = fallback
  return output, _task_details(output, debug_info[1], sampling, prompt_input, fallback)













def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False): 
  prompt_input = [act_game_object, persona.name, act_desp, act_game_object, act_game_object]
  return _run_primary_prompt_task(
    "act_obj_desc",
    "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt",
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    f"{act_game_object} is idle",
    family="plain_text",
    family_options={"trim_trailing_period": True},
    verbose=verbose,
  )









def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False): 
  prompt_input = [act_game_object, act_obj_desc, act_game_object]
  sampling = {"temperature": 0, "max_tokens": 30, "stop": ["\n"]}
  fallback = (act_game_object, "is", "idle")
  output, debug_info = _run_structured_prompt_task(
    "act_obj_event_triple",
    "persona/prompt_template/v2/generate_event_triple_v1.txt",
    prompt_input,
    persona,
    sampling,
    fallback,
    family="triple",
    verbose=verbose,
  )
  try:
    if len(output) == 3:
      output = tuple(output)
    else:
      output = (act_game_object, output[0], output[1])
  except Exception:
    output = fallback
  return output, _task_details(output, debug_info[1], sampling, prompt_input, fallback)





def run_gpt_prompt_new_decomp_schedule(persona, 
                                       main_act_dur, 
                                       truncated_act_dur, 
                                       start_time_hour,
                                       end_time_hour, 
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None, 
                                       verbose=False): 
  def create_prompt_input(persona, 
                           main_act_dur, 
                           truncated_act_dur, 
                           start_time_hour,
                           end_time_hour, 
                           inserted_act,
                           inserted_act_dur,
                           test_input=None): 
    persona_name = persona.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    for_time = start_time_hour
    for i in main_act_dur: 
      original_plan += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      original_plan += "\n"
      for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    for_time = start_time_hour
    for count, i in enumerate(truncated_act_dur): 
      new_plan_init += f'{for_time.strftime("%H:%M")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0]
      new_plan_init += "\n"
      if count < len(truncated_act_dur) - 1: 
        for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

    prompt_input = [persona_name, 
                    start_hour_str,
                    end_hour_str,
                    original_plan,
                    persona_name,
                    inserted_act,
                    inserted_act_dur,
                    persona_name,
                    start_hour_str,
                    end_hour_str,
                    end_hour_str,
                    new_plan_init]
    return prompt_input
  
  def get_fail_safe(main_act_dur, truncated_act_dur): 
    dur_sum = 0
    for act, dur in main_act_dur: dur_sum += dur

    ret = truncated_act_dur[:]
    ret += main_act_dur[len(ret)-1:]

    # If there are access, we need to trim... 
    ret_dur_sum = 0
    count = 0
    over = None
    for act, dur in ret: 
      ret_dur_sum += dur
      if ret_dur_sum == dur_sum: 
        break
      if ret_dur_sum > dur_sum: 
        over = ret_dur_sum - dur_sum
        break
      count += 1 

    if over: 
      ret = ret[:count+1]
      ret[-1][1] -= over

    return ret

  prompt_template = "persona/prompt_template/v2/new_decomp_schedule_v1.txt"
  prompt_input = create_prompt_input(persona, 
                                     main_act_dur, 
                                     truncated_act_dur, 
                                     start_time_hour,
                                     end_time_hour, 
                                     inserted_act,
                                     inserted_act_dur,
                                     test_input)
  fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)
  target_duration = sum(int(duration_value) for _action, duration_value in main_act_dur)
  return _run_structured_prompt_task(
    "new_decomp_schedule",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 1000},
    fail_safe,
    family="schedule_decomp",
    extra_system=[f"The durations in output must sum to exactly {target_duration} minutes."],
    verbose=verbose,
  )






def run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved,test_input=None, 
                                       verbose=False): 
  def create_prompt_input(init_persona, target_persona, retrieved, 
                          test_input=None): 
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_chatted_time = ""
    last_chat_about = ""
    if last_chat: 
      last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
      last_chat_about = last_chat.description

    context = ""
    for c_node in retrieved["events"]: 
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context +=  f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]: 
      context +=  f"{c_node.description}. "

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc: 
      init_act_desc = init_act_desc.split("(")[-1][:-1]
    
    if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
      init_p_desc = f"{init_persona.name} is already {init_act_desc}"
    elif "waiting" in init_act_desc:
      init_p_desc = f"{init_persona.name} is {init_act_desc}"
    else: 
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc: 
      target_act_desc = target_act_desc.split("(")[-1][:-1]
    
    if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
      target_p_desc = f"{target_persona.name} is already {target_act_desc}"
    elif "waiting" in init_act_desc:
      target_p_desc = f"{init_persona.name} is {init_act_desc}"
    else: 
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"


    prompt_input = []
    prompt_input += [context]

    prompt_input += [curr_time]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [last_chatted_time]
    prompt_input += [last_chat_about]


    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    return prompt_input
  
  def get_fail_safe(): 
    fs = "yes"
    return fs

  prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  fail_safe = get_fail_safe()
  return _run_structured_prompt_task(
    "decide_to_talk",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 20},
    fail_safe,
    family="enum_string",
    family_options={"choices": ["yes", "no"]},
    verbose=verbose,
  )




def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved,test_input=None, 
                                       verbose=False): 
  def create_prompt_input(init_persona, target_persona, retrieved, 
                          test_input=None): 

    


    context = ""
    for c_node in retrieved["events"]: 
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context +=  f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]: 
      context +=  f"{c_node.description}. "

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc: 
      init_act_desc = init_act_desc.split("(")[-1][:-1]
    if len(init_persona.scratch.planned_path) == 0: 
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
      init_p_desc = f"{init_persona.name} is already {init_act_desc} at {loc}"
    else: 
      loc = ""
      if ":" in init_persona.scratch.act_address:
        loc = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2]
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc} at {loc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc: 
      target_act_desc = target_act_desc.split("(")[-1][:-1]
    if len(target_persona.scratch.planned_path) == 0: 
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
      target_p_desc = f"{target_persona.name} is already {target_act_desc} at {loc}"
    else: 
      loc = ""
      if ":" in target_persona.scratch.act_address:
        loc = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2]
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc} at {loc}"

    prompt_input = []
    prompt_input += [context]
    prompt_input += [curr_time]
    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]

    prompt_input += [init_persona.name]
    prompt_input += [init_act_desc]
    prompt_input += [target_persona.name]
    prompt_input += [target_act_desc]

    prompt_input += [init_act_desc]
    return prompt_input
  
  def get_fail_safe(): 
    fs = "3"
    return fs

  prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
  prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                     test_input)
  fail_safe = get_fail_safe()
  return _run_structured_prompt_task(
    "decide_to_react",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 20},
    fail_safe,
    family="enum_string",
    family_options={"choices": ["1", "2", "3"]},
    verbose=verbose,
  )

















def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False): 
  def create_prompt_input(init_persona, target_persona, curr_loc, 
                          test_input=None): 

    prev_convo_insert = "\n"
    if init_persona.a_mem.seq_chat: 
      for i in init_persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((init_persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
          for row in i.filling: 
            prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if init_persona.a_mem.seq_chat: 
      if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""


    init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                target_persona.scratch.act_event[1],
                                target_persona.scratch.act_event[2])
    init_persona_thought = ""
    for i in init_persona_thought_nodes: 
      init_persona_thought += f"-- {i.description}\n"

    target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(init_persona.scratch.act_event[0],
                                init_persona.scratch.act_event[1],
                                init_persona.scratch.act_event[2])
    target_persona_thought = ""
    for i in target_persona_thought_nodes: 
      target_persona_thought += f"-- {i.description}\n"

    init_persona_curr_desc = ""
    if init_persona.scratch.planned_path: 
      init_persona_curr_desc = f"{init_persona.name} is on the way to {init_persona.scratch.act_description}"
    else: 
      init_persona_curr_desc = f"{init_persona.name} is {init_persona.scratch.act_description}"

    target_persona_curr_desc = ""
    if target_persona.scratch.planned_path: 
      target_persona_curr_desc = f"{target_persona.name} is on the way to {target_persona.scratch.act_description}"
    else: 
      target_persona_curr_desc = f"{target_persona.name} is {target_persona.scratch.act_description}"
 

    curr_loc = curr_loc["arena"]

    prompt_input = []
    prompt_input += [init_persona.scratch.get_str_iss()]
    prompt_input += [target_persona.scratch.get_str_iss()]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [init_persona_thought]

    prompt_input += [target_persona.name]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona_thought]

    prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S")]

    prompt_input += [init_persona_curr_desc]
    prompt_input += [target_persona_curr_desc]

    prompt_input += [prev_convo_insert]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]

    prompt_input += [curr_loc]
    prompt_input += [init_persona.name]
    return prompt_input
  
  def get_fail_safe(init_persona, target_persona): 
    convo = [[init_persona.name, "Hi!"], 
             [target_persona.name, "Hi!"]]
    return convo

  prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, curr_loc, 
                                     test_input)
  fail_safe = get_fail_safe(persona, target_persona)
  return _run_structured_prompt_task(
    "create_conversation",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0.7, "max_tokens": 1000},
    fail_safe,
    family="chat_transcript",
    verbose=verbose,
  )










def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False): 
  def create_prompt_input(conversation, test_input=None): 
    convo_str = ""
    for row in conversation: 
      convo_str += f'{row[0]}: "{row[1]}"\n'

    prompt_input = [convo_str]
    return prompt_input
  
  def get_fail_safe(): 
    return "conversing with a housemate about morning greetings"
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt"
  prompt_input = create_prompt_input(conversation, test_input)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "summarize_conversation",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="plain_text",
    family_options={"prefix": "conversing about "},
    verbose=verbose,
  )




def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False): 
  def create_prompt_input(description, test_input=None): 
    if "\n" in description: 
      description = description.replace("\n", " <LINE_BREAK> ")
    prompt_input = [description]
    return prompt_input
  
  def get_fail_safe(): 
    return set()

  prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
  prompt_input = create_prompt_input(description, test_input)
  fail_safe = get_fail_safe()
  output, debug_info = _run_structured_prompt_task(
    "extract_keywords",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 50},
    fail_safe,
    family="list_str",
    verbose=verbose,
  )
  output = {keyword.lower().rstrip(".") for keyword in output if keyword}
  return output, _task_details(output, debug_info[1], {"temperature": 0, "max_tokens": 50}, prompt_input, fail_safe)









def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False): 
  def create_prompt_input(persona, keyword, concept_summary, test_input=None): 
    prompt_input = [keyword, concept_summary, persona.name]
    return prompt_input
  
  def get_fail_safe(): 
    return ""
  prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(persona, keyword, concept_summary)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "keyword_to_thoughts",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0.7, "max_tokens": 40},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )









def run_gpt_prompt_convo_to_thoughts(persona, 
                                    init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None, verbose=False): 
  def create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None): 
    prompt_input = [init_persona_name,
                    target_persona_name,
                    convo_str,
                    init_persona_name,
                    fin_target]
    return prompt_input
  
  def get_fail_safe(): 
    return ""
  prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "convo_to_thoughts",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0.7, "max_tokens": 40},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )



























def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def get_fail_safe(): 
    return 4
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt"
  prompt_input = create_prompt_input(persona, event_description)
  fail_safe = get_fail_safe()
  return _run_structured_prompt_task(
    "event_poignancy",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="integer",
    family_options={"minimum": 1, "maximum": 10},
    verbose=verbose,
  )


def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def get_fail_safe(): 
    return 4
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt"
  prompt_input = create_prompt_input(persona, event_description)
  fail_safe = get_fail_safe()
  return _run_structured_prompt_task(
    "thought_poignancy",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="integer",
    family_options={"minimum": 1, "maximum": 10},
    verbose=verbose,
  )



def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def get_fail_safe(): 
    return 4
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt"
  prompt_input = create_prompt_input(persona, event_description)
  fail_safe = get_fail_safe()
  return _run_structured_prompt_task(
    "chat_poignancy",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="integer",
    family_options={"minimum": 1, "maximum": 10},
    verbose=verbose,
  )





def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def get_fail_safe(n): 
    return ["Who am I"] * n
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  fail_safe = get_fail_safe(n)
  return _run_structured_prompt_task(
    "focal_pt",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="list_str",
    verbose=verbose,
  )




  
def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def get_fail_safe(n): 
    return {}

  prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  fail_safe = get_fail_safe(n)
  return _run_structured_prompt_task(
    "insight_and_guidance",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0.5, "max_tokens": 150},
    fail_safe,
    family="dict_list_int",
    verbose=verbose,
  )








def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None): 
    prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently, 
                    statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt"
  prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "agent_chat_summarize_ideas",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )




def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, statements, test_input=None): 
    prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt"
  prompt_input = create_prompt_input(persona, target_persona, statements)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "agent_chat_summarize_relationship",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )





def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                               curr_context, 
                               init_summ_idea, 
                               target_summ_idea, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None): 
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"
    

    prompt_input = [persona.scratch.currently, 
                    target_persona.scratch.currently, 
                    prev_convo_insert,
                    curr_context, 
                    curr_location,

                    persona.scratch.name,
                    init_summ_idea, 
                    persona.scratch.name,
                    target_persona.scratch.name,

                    target_persona.scratch.name,
                    target_summ_idea, 
                    target_persona.scratch.name,
                    persona.scratch.name,

                    persona.scratch.name]
    return prompt_input
  
  def get_fail_safe(): 
    return [[persona.scratch.name, "Hi!"], [target_persona.scratch.name, "Hi!"]]

  prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt"
  prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)
  fail_safe = get_fail_safe()
  return _run_structured_prompt_task(
    "agent_chat",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="chat_transcript",
    verbose=verbose,
  )


# =======================
# =======================
# =======================
# =======================







def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, question, test_input=None): 
    prompt_input = [statements, persona.scratch.name, question]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt"
  prompt_input = create_prompt_input(persona, statements, question)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "summarize_ideas",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )



def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None, verbose=False): 
  def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None): 
    prompt_input = [persona.scratch.name, 
                    persona.scratch.get_str_iss(),
                    persona.scratch.name, 
                    interlocutor_desc, 
                    prev_convo, 
                    persona.scratch.name,
                    retrieved_summary, 
                    persona.scratch.name,]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
  prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "generate_next_convo_line",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 1, "max_tokens": 250},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )






def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False): 
  def create_prompt_input(persona, whisper, test_input=None): 
    prompt_input = [persona.scratch.name, whisper]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
  prompt_input = create_prompt_input(persona, whisper)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "generate_whisper_inner_thought",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 50},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )



def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "planning_thought_on_convo",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 50},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )



def run_gpt_prompt_revise_identity_plan_note(persona, statements, verbose=False):
  prompt = statements + "\n"
  prompt += f"Given the statements above, is there anything that {persona.scratch.name} should remember as they plan for"
  prompt += f" *{persona.scratch.curr_time.strftime('%A %B %d')}*? "
  prompt += "If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)\n\n"
  prompt += f"Write the response from {persona.scratch.name}'s perspective."
  return _run_raw_primary_task(
    "revise_identity_plan_note",
    prompt,
    persona,
    {"temperature": 0.2, "max_tokens": 300},
    "",
    family="plain_text",
    verbose=verbose,
    metadata={"identity_revision": "plan_note"},
  )


def run_gpt_prompt_revise_identity_thought_note(persona, statements, verbose=False):
  prompt = statements + "\n"
  prompt += f"Given the statements above, how might we summarize {persona.scratch.name}'s feelings about their days up to now?\n\n"
  prompt += f"Write the response from {persona.scratch.name}'s perspective."
  return _run_raw_primary_task(
    "revise_identity_thought_note",
    prompt,
    persona,
    {"temperature": 0.2, "max_tokens": 250},
    "",
    family="plain_text",
    verbose=verbose,
    metadata={"identity_revision": "thought_note"},
  )


def run_gpt_prompt_revise_identity_currently(persona, plan_note, thought_note, verbose=False):
  p_name = persona.scratch.name
  prompt = f"{p_name}'s status from {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n"
  prompt += f"{persona.scratch.currently}\n\n"
  prompt += f"{p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n"
  prompt += (plan_note + thought_note).replace('\n', '') + "\n\n"
  prompt += f"It is now {persona.scratch.curr_time.strftime('%A %B %d')}. Given the above, write {p_name}'s status for {persona.scratch.curr_time.strftime('%A %B %d')} that reflects {p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}. Write this in third-person talking about {p_name}."
  prompt += "If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n"
  prompt += "Follow this format below:\nStatus: <new status>"
  return _run_raw_primary_task(
    "revise_identity_currently",
    prompt,
    persona,
    {"temperature": 0.2, "max_tokens": 350},
    persona.scratch.currently,
    family="plain_text",
    verbose=verbose,
    metadata={"identity_revision": "currently"},
  )


def run_gpt_prompt_revise_identity_daily_plan(persona, verbose=False):
  prompt = persona.scratch.get_str_iss() + "\n"
  prompt += f"Today is {persona.scratch.curr_time.strftime('%A %B %d')}. Here is {persona.scratch.name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
  prompt += "Follow this format (the list should have 4~6 items but no more):\n"
  prompt += "1. wake up and complete the morning routine at <time>, 2. ..."
  return _run_raw_primary_task(
    "revise_identity_daily_plan",
    prompt,
    persona,
    {"temperature": 0.4, "max_tokens": 250},
    persona.scratch.get_str_daily_plan_req() or "",
    family="plain_text",
    verbose=verbose,
    metadata={"identity_revision": "daily_plan_req"},
  )


def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def get_fail_safe(): 
    return "..."
  prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  fail_safe = get_fail_safe()
  return _run_primary_prompt_task(
    "memo_on_convo",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 15},
    fail_safe,
    family="plain_text",
    verbose=verbose,
  )




def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False): 
  def create_prompt_input(comment, test_input=None):
    prompt_input = [comment]
    return prompt_input

  def get_fail_safe():
    return 0

  prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt" 
  prompt_input = create_prompt_input(comment) 
  fail_safe = get_fail_safe() 
  return _run_structured_prompt_task(
    "safety_score",
    prompt_template,
    prompt_input,
    persona,
    {"temperature": 0, "max_tokens": 50},
    fail_safe,
    family="safety_score",
    verbose=verbose,
  )

def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None, verbose=False): 
  def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
    persona = init_persona
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items(): 
      for v in vals: 
        retrieved_str += f"- {v.description}\n"


    convo_str = ""
    for i in curr_chat:
      convo_str += ": ".join(i) + "\n"
    if convo_str == "": 
      convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
    prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
      curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
      convo_str, init_persona.scratch.name, target_persona.scratch.name,
      init_persona.scratch.name, init_persona.scratch.name,
      init_persona.scratch.name
      ]
    return prompt_input

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict

  prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt" 
  prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat) 
  fail_safe = get_fail_safe() 
  return _run_structured_prompt_task(
    "iterative_chat_utt",
    prompt_template,
    prompt_input,
    init_persona,
    {"temperature": 0, "max_tokens": 50},
    fail_safe,
    family="utterance_turn",
    verbose=verbose,
  )
