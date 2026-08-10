import json
import os
from pathlib import Path

import pytest

from janus.core.backends.select import build_backend, build_backend_for_persona
from janus.core.config import load_config
from janus.core.persona import Persona
from janus.core.validation.harness import validate
from janus.core.validation.judge import JUDGE_SYSTEM_PROMPT, judge_registry
from janus.core.validation.rubric import Rubric

PERSONA_DIR = Path(__file__).parent.parent.parent / "personas" / "market_research"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live capstone: set ANTHROPIC_API_KEY to run the real market-research proof",
)


@pytest.mark.asyncio
async def test_market_research_capstone_live(tmp_path):
    """CAPSTONE: run the market-research persona on live Claude and require a
    passing ValidationReport. Skipped unless ANTHROPIC_API_KEY is set."""
    persona = Persona.load(PERSONA_DIR)
    rubric = Rubric.load(persona.rubric_path)

    def make_agent_backend(p, ws):
        return build_backend_for_persona(
            load_config(persona=p.name, working_directory=ws), p
        )

    def make_judge_backend(r, jws):
        return build_backend(
            load_config(working_directory=jws), JUDGE_SYSTEM_PROMPT, judge_registry(r)
        )

    report = await validate(persona, rubric, make_agent_backend, make_judge_backend, tmp_path)

    # Persist the captured report as the proof artifact.
    artifact = tmp_path / "validation-report.json"
    artifact.write_text(json.dumps({
        "smoke_passed": report.smoke.passed,
        "judge_passed": report.judge.passed if report.judge else None,
        "scores": report.judge.scores if report.judge else None,
        "passed": report.passed,
    }, indent=2))
    print(f"\nCAPSTONE ValidationReport -> {artifact}\n{artifact.read_text()}")

    assert report.smoke.passed, "live smoke run did not complete with a schema-valid brief"
    scores = report.judge.scores if report.judge else None
    assert report.passed, f"live validation did not pass; scores={scores}"
