"""Shared samples for factory tests: a known-good declarative persona quartet,
plus the scripted fake-backend helpers reused across the factory test files."""

import json

from janus.core.backends.generic import GenericBackend
from janus.core.validation.judge import judge_registry

GOOD_NAME = "haiku_scout"

GOOD_MANIFEST = """\
[persona]
name = "haiku_scout"
description = "Writes a haiku about a subject."
domain = "poetry"

[prompt]
file = "prompt.md"

[task]
template = "Write a haiku about: {subject}"

[tools]
builtins = []

[output]
schema_file = "output_schema.json"

[validation]
rubric_file = "rubric.toml"
"""

GOOD_PROMPT = (
    "You are a poet. Write a single haiku about the subject, "
    "then call emit_output with it.\n"
)

GOOD_SCHEMA = """\
{
  "type": "object",
  "properties": {"haiku": {"type": "string"}},
  "required": ["haiku"],
  "additionalProperties": false
}
"""

GOOD_RUBRIC = """\
tasks = ["autumn rain"]
pass_threshold = 0.7
mode = "all"

[[criteria]]
name = "form"
description = "The output reads like a haiku: three short lines, evocative imagery."
"""

GOOD_DELIVERABLE = {"haiku": "cold rain on the roof\nthe kettle begins to sing\nsteam against the glass"}
PASSING_VERDICT = {"scores": {"form": 0.9}, "feedback": "reads like a haiku"}
FAILING_VERDICT = {"scores": {"form": 0.2}, "feedback": "this is prose, not a haiku"}


def get_tool(mod, name):
    """Fetch a ToolSpec from a loaded factory tools module by name."""
    return next(t for t in mod.TOOLS if t.name == name)


def scaffold_good(mod, ctx, **overrides):
    """Call scaffold_persona with the known-good quartet, allowing per-field overrides."""
    args = {
        "name": GOOD_NAME,
        "manifest_toml": GOOD_MANIFEST,
        "prompt_md": GOOD_PROMPT,
        "output_schema_json": GOOD_SCHEMA,
        "rubric_toml": GOOD_RUBRIC,
    }
    args.update(overrides)
    return get_tool(mod, "scaffold_persona").handler(ctx, **args)


def emit_backend_factory(payload):
    """GenericBackend subclass whose first completion calls emit_output with payload.
    Same pattern as tests/core/validation/test_harness.py."""

    class _B(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1 and payload is not None:
                return {"message": {"content": "x", "tool_calls": [
                    {"id": "c1",
                     "function": {"name": "emit_output", "arguments": json.dumps(payload)}}]}}
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tc, result):
            return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}

    return _B


def make_fake_factories(deliverable, verdict):
    """ctx.extra backend-factory overrides for validate_persona: the generated
    persona's agent emits `deliverable`; the judge emits `verdict`."""
    AgentB = emit_backend_factory(deliverable)
    JudgeB = emit_backend_factory(verdict)

    def make_agent(persona, wd):
        return AgentB(working_directory=wd, system_prompt=persona.system_prompt,
                      model="m", registry=persona.registry)

    def make_judge(rubric, wd):
        return JudgeB(working_directory=wd, system_prompt="judge",
                      model="m", registry=judge_registry(rubric))

    return {"make_agent_backend": make_agent, "make_judge_backend": make_judge}
