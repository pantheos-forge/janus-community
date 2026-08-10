import asyncio
import json
from pathlib import Path

import pytest
from janus.core.persona import Persona
from janus.core.backends.generic import GenericBackend
from janus.core.validation.rubric import Rubric, Criterion
from janus.core.validation.judge import judge_registry
from janus.core.validation.harness import ValidationReport, validate

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "personas" / "echo_brief"
RUBRIC_PATH = FIXTURE / "rubric.toml"

_CONTAINER_MANIFEST = '''
[persona]
name = "tinycontainer"
description = "tiny containerized persona for hermetic tests"
domain = "demo"
[prompt]
file = "prompt.md"
[tools]
builtins = ["bash"]
[task]
template = "Do: {subject}"
[output]
schema_file = "output_schema.json"
'''


def _containerized_persona(tmp_path):
    """A minimal containerized persona: manifest + prompt + container.toml + output_schema."""
    d = tmp_path / "tinycontainer"
    d.mkdir()
    (d / "manifest.toml").write_text(_CONTAINER_MANIFEST)
    (d / "prompt.md").write_text("You are a tiny tool agent.")
    (d / "container.toml").write_text(
        '[install]\napt = ["ripgrep"]\n[[tool]]\nname = "rg"\ndescription = "search"\n'
    )
    (d / "output_schema.json").write_text(
        '{"type":"object","properties":{"summary":{"type":"string"}},'
        '"required":["summary"],"additionalProperties":false}'
    )
    return Persona.load(d)


def _emit_backend_factory(payload):
    class _B(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0
        async def _chat_completion(self):
            self._n += 1
            if self._n == 1 and payload is not None:
                return {"message": {"content": "x", "tool_calls": [
                    {"id": "c1", "function": {"name": "emit_output", "arguments": json.dumps(payload)}}]}}
            return {"message": {"content": "done"}}
        def _tool_result_message(self, tc, result):
            return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}
    return _B


@pytest.mark.asyncio
async def test_validate_passes_end_to_end(tmp_path):
    persona = Persona.load(FIXTURE)
    rubric = Rubric.load(RUBRIC_PATH)  # criteria: coverage, clarity; threshold 0.7; mode all

    AgentB = _emit_backend_factory({"summary": "EVs are growing"})
    JudgeB = _emit_backend_factory({"scores": {"coverage": 0.9, "clarity": 0.8}, "feedback": "great"})

    def make_agent(persona, wd):
        return AgentB(working_directory=wd, system_prompt=persona.system_prompt, model="m",
                      registry=persona.registry)

    def make_judge(rubric, wd):
        return JudgeB(working_directory=wd, system_prompt="judge", model="m",
                      registry=judge_registry(rubric))

    report = await validate(persona, rubric, make_agent, make_judge, tmp_path / "run")
    assert isinstance(report, ValidationReport)
    assert report.smoke.passed
    assert report.judge is not None and report.judge.passed
    assert report.passed


@pytest.mark.asyncio
async def test_validate_short_circuits_when_smoke_fails(tmp_path):
    persona = Persona.load(FIXTURE)
    rubric = Rubric.load(RUBRIC_PATH)

    AgentB = _emit_backend_factory(None)  # emits nothing -> no deliverable -> smoke fails
    JudgeB = _emit_backend_factory({"scores": {"coverage": 0.9, "clarity": 0.9}, "feedback": "x"})

    def make_agent(persona, wd):
        return AgentB(working_directory=wd, system_prompt=persona.system_prompt, model="m",
                      registry=persona.registry)

    judge_called = {"n": 0}
    def make_judge(rubric, wd):
        judge_called["n"] += 1
        return JudgeB(working_directory=wd, system_prompt="judge", model="m",
                      registry=judge_registry(rubric))

    report = await validate(persona, rubric, make_agent, make_judge, tmp_path / "run")
    assert not report.smoke.passed
    assert report.judge is None           # judge skipped
    assert judge_called["n"] == 0         # judge backend never built
    assert not report.passed


@pytest.mark.asyncio
async def test_validate_runs_every_rubric_task(tmp_path):
    persona = Persona.load(FIXTURE)
    rubric = Rubric(
        tasks=["EV charging in Norway", "heat pumps in Germany"],
        criteria=[Criterion("coverage", "covers the topic"), Criterion("clarity", "clearly written")],
    )
    AgentB = _emit_backend_factory({"summary": "growing fast"})
    JudgeB = _emit_backend_factory({"scores": {"coverage": 0.9, "clarity": 0.8}, "feedback": "ok"})

    agent_ws, judge_ws = [], []

    def make_agent(p, wd):
        agent_ws.append(wd)
        return AgentB(working_directory=wd, system_prompt=p.system_prompt, model="m",
                      registry=p.registry)

    def make_judge(r, wd):
        judge_ws.append(wd)
        return JudgeB(working_directory=wd, system_prompt="judge", model="m",
                      registry=judge_registry(r))

    report = await validate(persona, rubric, make_agent, make_judge, tmp_path / "run")
    assert report.passed
    assert len(agent_ws) == 2 and len(judge_ws) == 2      # one run per rubric task
    assert agent_ws[0] != agent_ws[1]                     # separate workspaces
    check_names = [c.name for c in report.smoke.checks]
    assert any(n.startswith("task0:") for n in check_names)
    assert any(n.startswith("task1:") for n in check_names)
    assert report.judge is not None and len(report.judge.per_task) == 2


@pytest.mark.asyncio
async def test_validate_uses_container_smoke_for_containerized_persona(tmp_path, monkeypatch):
    """A containerized persona routes the smoke phase through container_smoke_run,
    not the in-process make_agent_backend."""
    import janus.core.validation.container_smoke as cs
    from janus.core.validation.smoke import SmokeCheck, SmokeResult

    calls = {"container": 0, "agent_backend": 0}

    async def fake_container_smoke(persona, subject, ws, **kw):
        calls["container"] += 1
        return SmokeResult(True, [SmokeCheck("run_completed", True, "ok")],
                           deliverable={"summary": "ok"})
    monkeypatch.setattr(cs, "container_smoke_run", fake_container_smoke)

    def make_agent(persona, ws):
        calls["agent_backend"] += 1
        raise AssertionError("in-process agent backend must NOT be built for a container persona")

    persona = _containerized_persona(tmp_path)  # local helper: persona with .container set
    rubric = Rubric(
        tasks=["find something"],
        criteria=[Criterion("coverage", "covers the topic")],
    )  # reuse the module's Rubric-construction pattern (see test_validate_runs_every_rubric_task)

    JudgeB = _emit_backend_factory({"scores": {"coverage": 0.9}, "feedback": "fine"})

    def make_judge(r, wd):
        return JudgeB(working_directory=wd, system_prompt="judge", model="m",
                      registry=judge_registry(r))

    report = await validate(persona, rubric, make_agent, make_judge, tmp_path / "val")
    assert calls["container"] == len(rubric.tasks)
    assert calls["agent_backend"] == 0
    assert report.smoke.passed


@pytest.mark.asyncio
async def test_validate_completes_when_persona_uses_ask_user(tmp_path):
    """Validation never enables user replies, so ask_user fails open and the
    run still completes and passes."""
    import json as _json

    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.builtins import builtin_registry

    persona = Persona.load(FIXTURE)
    persona.registry.register(builtin_registry(["ask_user"]).get("ask_user"))
    rubric = Rubric.load(RUBRIC_PATH)

    class AskThenEmit(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1:
                return {"message": {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "ask_user",
                     "arguments": _json.dumps({"question": "scope?"})}}]}}
            if self._n == 2:
                return {"message": {"content": "", "tool_calls": [
                    {"id": "c2", "function": {"name": "emit_output",
                     "arguments": _json.dumps({"summary": "assumed broad scope"})}}]}}
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tc, result):
            return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}

    JudgeB = _emit_backend_factory(
        {"scores": {"coverage": 0.9, "clarity": 0.9}, "feedback": "fine"})

    def make_agent(p, wd):
        return AskThenEmit(working_directory=wd, system_prompt=p.system_prompt,
                           model="m", registry=p.registry)

    def make_judge(r, wd):
        return JudgeB(working_directory=wd, system_prompt="judge", model="m",
                      registry=judge_registry(r))

    report = await asyncio.wait_for(
        validate(persona, rubric, make_agent, make_judge, tmp_path / "run"), timeout=30
    )
    assert report.passed
