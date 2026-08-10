"""Hermetic e2e: a scripted factory loads a fleet agent, baseline-validates,
tightens, re-validates, and commits the improvement — inside one run."""

import json
import subprocess
from pathlib import Path

import pytest

from janus.core.backends.generic import GenericBackend
from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.events import EventBus
from janus.core.persona import Persona
from janus.core.session import SessionStore
from janus.fleet.registry import FleetRegistry
from tests.personas.factory_samples import (
    GOOD_MANIFEST, GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA, GOOD_DELIVERABLE,
    PASSING_VERDICT, make_fake_factories)

FACTORY_DIR = Path(__file__).parent.parent.parent / "personas" / "factory"

_TIGHTER_PROMPT = GOOD_PROMPT + "\nAlways cite a named source.\n"


def _seed_fleet_agent(fleet):
    agent = fleet / "haiku_scout"
    persona = agent / "persona"
    persona.mkdir(parents=True)
    (persona / "manifest.toml").write_text(GOOD_MANIFEST)
    (persona / "prompt.md").write_text(GOOD_PROMPT)
    (persona / "output_schema.json").write_text(GOOD_SCHEMA)
    (persona / "rubric.toml").write_text(GOOD_RUBRIC)
    subprocess.run(["git", "init", "-q"], cwd=agent, check=True)
    subprocess.run(["git", "add", "-A"], cwd=agent, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "init"], cwd=agent, check=True)
    FleetRegistry(fleet).register("haiku_scout", domain="poetry", description="d",
                                  source="factory", path=str(agent))
    return agent


_SCRIPT = [
    ("load_fleet_persona", {"name": "haiku_scout"}),
    ("validate_persona", {"name": "haiku_scout"}),          # baseline (attempt 1)
    ("scaffold_persona", {                                    # tighten the prompt
        "name": "haiku_scout", "manifest_toml": GOOD_MANIFEST,
        "prompt_md": _TIGHTER_PROMPT, "output_schema_json": GOOD_SCHEMA,
        "rubric_toml": GOOD_RUBRIC}),
    ("validate_persona", {"name": "haiku_scout"}),          # re-validate (attempt 2)
    ("export_improved_persona", {"name": "haiku_scout", "summary": "require a named source"}),
    ("emit_output", {
        "status": "exported",
        "agent": {"name": "haiku_scout", "domain": "poetry", "description": "d"},
        "attempts": [{"smoke_passed": True, "judge_passed": True,
                      "scores": {"form": 0.9}, "feedback_digest": "ok",
                      "changes_made": "cite a source"}],
        "export_path": "fleet/haiku_scout", "how_to_run": "python agent.py '...'"}),
]


class ScriptedImproveBackend(GenericBackend):
    def __init__(self, *a, fake_factories=None, fleet_dir=None, **k):
        super().__init__(*a, **k)
        self._step = 0
        self._fake = fake_factories or {}
        self._fleet_dir = fleet_dir
        self.tool_results = []

    def _tool_context(self):
        ctx = super()._tool_context()
        ctx.extra.update(self._fake)
        if self._fleet_dir:
            ctx.extra["fleet_dir"] = self._fleet_dir
        return ctx

    async def _chat_completion(self):
        if self._step < len(_SCRIPT):
            name, args = _SCRIPT[self._step]
            self._step += 1
            return {"message": {"content": "", "tool_calls": [
                {"id": f"c{self._step}",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}}
        return {"message": {"content": "done"}}

    def _tool_result_message(self, tool_call, result):
        self.tool_results.append(result)
        return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}


@pytest.mark.asyncio
async def test_scripted_factory_improves_a_fleet_agent(tmp_path):
    fleet = tmp_path / "fleet"
    agent = _seed_fleet_agent(fleet)

    factory = Persona.load(FACTORY_DIR)
    ws = tmp_path / "ws"
    factory.prepare_workspace(ws)
    backend = ScriptedImproveBackend(
        working_directory=ws, system_prompt=factory.system_prompt, model="m",
        registry=factory.registry,
        fake_factories=make_fake_factories(GOOD_DELIVERABLE, PASSING_VERDICT),
        fleet_dir=str(fleet))
    controller = AgentController(
        load_config(persona=factory.name, working_directory=ws),
        backend=backend,
        session_store=SessionStore(sessions_dir=ws / ".sessions"),
        events=EventBus())

    result = await controller.run("IMPROVEMENT REQUEST for 'haiku_scout' ...")
    assert result["status"] == "completed"
    for tr in backend.tool_results:
        assert not tr.startswith(("Error", "Scaffold rejected", "Infrastructure error")), tr

    # the improvement was committed in the agent's own repo, history preserved
    log = subprocess.run(["git", "log", "--oneline"], cwd=agent,
                         capture_output=True, text=True).stdout
    assert "improve: require a named source" in log
    assert "init" in log
    assert "Always cite a named source." in (agent / "persona" / "prompt.md").read_text()
    # registry recorded the improvement scores
    hist = FleetRegistry(fleet).get("haiku_scout")["validation_history"]
    assert hist and hist[-1]["note"].startswith("improve:")
