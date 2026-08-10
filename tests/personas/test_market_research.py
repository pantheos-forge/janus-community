import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from janus.core.backends.generic import GenericBackend
from janus.core.persona import Persona
from janus.core.validation.judge import judge_registry
from janus.core.validation.rubric import Rubric
from janus.factory import export_agent

PERSONA_DIR = Path(__file__).parent.parent.parent / "personas" / "market_research"


@pytest.fixture
def persona():
    return Persona.load(PERSONA_DIR)


def test_persona_loads_with_expected_identity(persona):
    assert persona.name == "market_research"
    assert "research" in persona.description.lower()
    assert persona.provider_model == "claude-sonnet-5"


def test_registry_is_the_bash_free_research_toolset(persona):
    names = set(persona.registry.names())
    assert {"web_fetch", "write_file", "read_file", "update_plan", "emit_output"} <= names
    assert "bash" not in names          # deliberate: no shell
    assert "web_search" not in names     # deliberate: web_fetch only


def test_task_template_frames_a_research_brief(persona):
    task = persona.build_task("the market for at-home coffee equipment in the United States")
    assert "at-home coffee equipment" in task
    assert "brief" in task.lower()


def test_output_schema_accepts_a_valid_brief(persona):
    schema = persona.output_schema
    assert schema is not None
    valid = {
        "summary": "The US at-home coffee equipment market is large and growing.",
        "key_findings": [
            {"finding": "Premium espresso machines are the fastest-growing segment.",
             "sources": ["https://example.com/report"]},
        ],
        "competitive_landscape": [
            {"name": "Breville", "note": "Premium all-in-one machines.",
             "sources": ["https://example.com/breville"]},
        ],
        "risks": ["Discretionary spending is sensitive to recessions."],
        "opportunities": ["Subscription bean models bundled with hardware."],
    }
    jsonschema.validate(valid, schema)  # must not raise


def test_output_schema_rejects_a_finding_without_sources(persona):
    schema = persona.output_schema
    bad = {
        "summary": "x",
        "key_findings": [{"finding": "unsourced claim", "sources": []}],  # minItems 1 violated
        "competitive_landscape": [],
        "risks": [],
        "opportunities": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_output_schema_rejects_a_competitor_without_sources(persona):
    schema = persona.output_schema
    bad = {
        "summary": "x",
        "key_findings": [{"finding": "sourced", "sources": ["https://example.com"]}],
        # competitor missing the now-required "sources" -> rejected
        "competitive_landscape": [{"name": "Breville", "note": "premium"}],
        "risks": [],
        "opportunities": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_rubric_has_the_three_criteria(persona):
    assert persona.rubric_path is not None
    rubric = Rubric.load(persona.rubric_path)
    names = {c.name for c in rubric.criteria}
    assert names == {"coverage", "sourcing", "structure"}
    assert rubric.tasks  # at least one smoke/judge task
    assert rubric.pass_threshold == 0.7


def _emit_backend_factory(payload):
    """Build a GenericBackend subclass whose first _chat_completion emits a tool_call
    to emit_output with the given payload. Copied from
    tests/core/validation/test_harness.py's reusable fake-backend pattern."""

    class _B(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1 and payload is not None:
                return {
                    "message": {
                        "content": "x",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "emit_output",
                                    "arguments": json.dumps(payload),
                                },
                            }
                        ],
                    }
                }
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tc, result):
            return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}

    return _B


@pytest.mark.asyncio
async def test_validation_harness_passes_with_a_schema_valid_brief(tmp_path):
    """The persona validates end-to-end through the real harness with fake backends:
    a schema-valid brief + judge scores above threshold -> ValidationReport.passed."""
    persona = Persona.load(PERSONA_DIR)
    rubric = Rubric.load(persona.rubric_path)

    valid_brief = {
        "summary": "The US at-home coffee equipment market is large and growing.",
        "key_findings": [
            {
                "finding": "Premium espresso is the fastest-growing segment.",
                "sources": ["https://example.com/report"],
            }
        ],
        "competitive_landscape": [
            {"name": "Breville", "note": "Premium all-in-one machines.",
             "sources": ["https://example.com/breville"]}
        ],
        "risks": ["Discretionary spending is recession-sensitive."],
        "opportunities": ["Subscription bean+hardware bundles."],
    }

    AgentB = _emit_backend_factory(valid_brief)
    JudgeB = _emit_backend_factory(
        {
            "scores": {"coverage": 0.9, "sourcing": 0.9, "structure": 0.9},
            "feedback": "well sourced and complete",
        }
    )

    def make_agent(persona, wd):
        return AgentB(
            working_directory=wd,
            system_prompt=persona.system_prompt,
            model="m",
            registry=persona.registry,
        )

    def make_judge(rubric, wd):
        return JudgeB(
            working_directory=wd,
            system_prompt="judge",
            model="m",
            registry=judge_registry(rubric),
        )

    from janus.core.validation.harness import validate

    report = await validate(
        persona, rubric, make_agent, make_judge, tmp_path / "val"
    )
    assert report.smoke.passed
    assert report.judge is not None and report.judge.passed
    assert report.passed


def test_exported_market_research_repo_runs_headless(tmp_path):
    """export_agent produces a runnable vendored repo; with no provider the
    generated agent.py exits 1 with the friendly message (not a traceback)."""
    dest = export_agent(Persona.load(PERSONA_DIR), tmp_path / "agent", git_init=False)
    proc = subprocess.run(
        [sys.executable, "agent.py", "at-home coffee equipment in the US"],
        cwd=dest,
        env={"PYTHONPATH": str(dest), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "Configure a provider" in proc.stderr
    assert "Traceback" not in proc.stderr
