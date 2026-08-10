from pathlib import Path

from janus.core.persona import Persona
from janus.core.validation import production
from janus.core.validation.judge import JUDGE_SYSTEM_PROMPT
from janus.core.validation.rubric import Criterion, Rubric

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "personas" / "echo_brief"


def test_agent_factory_wires_the_persona(monkeypatch, tmp_path):
    captured = {}

    def fake_build_backend_for_persona(config, persona):
        captured["persona"] = persona
        captured["config"] = config
        return object()

    monkeypatch.setattr(production, "build_backend_for_persona", fake_build_backend_for_persona)
    persona = Persona.load(FIXTURE)
    production.make_production_agent_backend(persona, tmp_path / "ws")
    assert captured["persona"] is persona
    assert captured["config"].working_directory == tmp_path / "ws"


def test_judge_factory_upholds_the_judge_contract(monkeypatch, tmp_path):
    captured = {}

    def fake_build_backend(config, system_prompt, registry):
        captured["system_prompt"] = system_prompt
        captured["registry"] = registry
        return object()

    monkeypatch.setattr(production, "build_backend", fake_build_backend)
    rubric = Rubric(tasks=["t"], criteria=[Criterion("a", "desc")])
    production.make_production_judge_backend(rubric, tmp_path / "jws")
    assert captured["system_prompt"] is JUDGE_SYSTEM_PROMPT
    assert captured["registry"].names() == ["emit_output"]
