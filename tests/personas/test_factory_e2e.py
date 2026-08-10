"""Hermetic end-to-end: a scripted factory LLM drives the real controller,
registry, and factory tools through scaffold -> validate -> export -> report."""

import json
from pathlib import Path

import jsonschema
import pytest

from janus.core.backends.generic import GenericBackend
from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.events import EventBus
from janus.core.persona import Persona
from janus.core.session import SessionStore
from tests.personas.factory_samples import (
    GOOD_DELIVERABLE,
    GOOD_MANIFEST,
    GOOD_NAME,
    GOOD_PROMPT,
    GOOD_RUBRIC,
    GOOD_SCHEMA,
    PASSING_VERDICT,
    make_fake_factories,
)

PERSONA_DIR = Path(__file__).parent.parent.parent / "personas" / "factory"

_REPORT = {
    "status": "exported",
    "agent": {"name": GOOD_NAME, "domain": "poetry",
              "description": "Writes a haiku about a subject."},
    "attempts": [{"smoke_passed": True, "judge_passed": True,
                  "scores": {"form": 0.9}, "feedback_digest": "reads like a haiku",
                  "changes_made": "initial version"}],
    "export_path": f"fleet/{GOOD_NAME}",
    "how_to_run": 'python agent.py "<subject>"',
}

_SCRIPT = [
    ("scaffold_persona", {
        "name": GOOD_NAME,
        "manifest_toml": GOOD_MANIFEST,
        "prompt_md": GOOD_PROMPT,
        "output_schema_json": GOOD_SCHEMA,
        "rubric_toml": GOOD_RUBRIC,
    }),
    ("validate_persona", {"name": GOOD_NAME}),
    ("export_persona", {"name": GOOD_NAME}),
    ("emit_output", _REPORT),
]


class ScriptedFactoryBackend(GenericBackend):
    """Plays the factory LLM: one scripted tool call per completion, then done."""

    def __init__(self, *args, fake_factories=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._fake_factories = fake_factories or {}
        self.tool_results = []

    def _tool_context(self):
        ctx = super()._tool_context()
        ctx.extra.update(self._fake_factories)
        return ctx

    async def _chat_completion(self):
        if self._step < len(_SCRIPT):
            name, args = _SCRIPT[self._step]
            self._step += 1
            return {"message": {"content": "", "tool_calls": [
                {"id": f"c{self._step}",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}}
        return {"message": {"content": "Build complete."}}

    def _tool_result_message(self, tc, result):
        self.tool_results.append(result)
        return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}


@pytest.mark.asyncio
async def test_scripted_factory_builds_validates_and_exports(tmp_path):
    persona = Persona.load(PERSONA_DIR)
    ws = tmp_path / "ws"
    persona.prepare_workspace(ws)

    fake_factories = make_fake_factories(GOOD_DELIVERABLE, PASSING_VERDICT)
    fake_factories["fleet_dir"] = str(ws / "fleet")

    backend = ScriptedFactoryBackend(
        working_directory=ws,
        system_prompt=persona.system_prompt,
        model="m",
        registry=persona.registry,
        fake_factories=fake_factories,
    )
    controller = AgentController(
        load_config(persona=persona.name, working_directory=ws),
        backend=backend,
        session_store=SessionStore(sessions_dir=ws / ".sessions"),
        events=EventBus(),   # explicit private bus for this controller
    )
    result = await controller.run(persona.build_task("an agent that writes haiku"))

    assert result["status"] == "completed"
    # every scripted tool call succeeded
    for tool_result in backend.tool_results:
        assert not tool_result.startswith("Error"), tool_result
        assert "rejected" not in tool_result.lower(), tool_result

    # scaffolded persona is on disk and validation recorded one passing attempt
    build = ws / "build" / GOOD_NAME
    assert (build / "manifest.toml").exists()
    attempts = json.loads(
        (ws / "build" / ".state" / GOOD_NAME / "attempts.json").read_text()
    )["attempts"]
    assert len(attempts) == 1 and attempts[0]["passed"] is True

    # the exported repo exists and is clean (in the fleet)
    export = ws / "fleet" / GOOD_NAME
    assert (export / "agent.py").exists()
    assert (export / "janus" / "__init__.py").exists()
    assert not (export / "persona" / "validation").exists()

    # the factory's own build report was emitted and validates against its schema
    report = json.loads((ws / "output.json").read_text())
    jsonschema.validate(report, persona.output_schema)
    assert report["status"] == "exported"


_CONVO_SCRIPT = [
    ("ask_user", {"question": "What domain should the agent cover, and who reads its output?"}),
    ("ask_user", {"question": "SPEC: haiku_scout, poetry domain, emit haiku JSON. Approve?", "choices": ["Approve the spec", "Request changes"]}),
    ("scaffold_persona", {
        "name": GOOD_NAME,
        "manifest_toml": GOOD_MANIFEST,
        "prompt_md": GOOD_PROMPT,
        "output_schema_json": GOOD_SCHEMA,
        "rubric_toml": GOOD_RUBRIC,
    }),
    ("validate_persona", {"name": GOOD_NAME}),
    ("export_persona", {"name": GOOD_NAME}),
    ("emit_output", _REPORT),
]


class ConversationalFactoryBackend(ScriptedFactoryBackend):
    """Same scripted factory, but its first two turns are ask_user calls."""

    async def _chat_completion(self):
        if self._step < len(_CONVO_SCRIPT):
            name, args = _CONVO_SCRIPT[self._step]
            self._step += 1
            return {"message": {"content": "", "tool_calls": [
                {"id": f"c{self._step}",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}}
        return {"message": {"content": "Build complete."}}


@pytest.mark.asyncio
async def test_factory_holds_a_conversation_inside_one_run(tmp_path):
    """Cycle-3A demo, hermetic: two ask_user turns answered via
    controller.reply(), then scaffold -> validate -> export -> report —
    ALL inside a single controller.run()."""
    import asyncio

    from janus.core.controller import AgentState

    persona = Persona.load(PERSONA_DIR)
    ws = tmp_path / "ws"
    persona.prepare_workspace(ws)

    fake_factories = make_fake_factories(GOOD_DELIVERABLE, PASSING_VERDICT)
    fake_factories["fleet_dir"] = str(ws / "fleet")

    backend = ConversationalFactoryBackend(
        working_directory=ws,
        system_prompt=persona.system_prompt,
        model="m",
        registry=persona.registry,
        fake_factories=fake_factories,
    )
    controller = AgentController(
        load_config(persona=persona.name, working_directory=ws),
        backend=backend,
        session_store=SessionStore(sessions_dir=ws / ".sessions"),
    )
    controller.enable_user_replies()

    replies = ["Poetry; the reader is a poetry-club newsletter.", "Approve the spec"]

    async def answer_questions():
        for reply in replies:
            for _ in range(200):
                await asyncio.sleep(0.05)
                if controller.state is AgentState.AWAITING_INPUT:
                    controller.reply(reply)
                    break
            else:
                raise AssertionError("factory never asked")
            await asyncio.sleep(0.05)

    result, _ = await asyncio.wait_for(
        asyncio.gather(
            controller.run(persona.build_task("an agent that writes haiku")),
            answer_questions(),
        ),
        timeout=60,
    )

    assert result["status"] == "completed"
    # both replies became ask_user tool results
    assert "poetry-club" in backend.tool_results[0]
    assert backend.tool_results[1] == "Approve the spec"
    # and the build still happened end-to-end (exported to the fleet)
    report = json.loads((ws / "output.json").read_text())
    assert report["status"] == "exported"
    assert (ws / "fleet" / GOOD_NAME / "agent.py").exists()
