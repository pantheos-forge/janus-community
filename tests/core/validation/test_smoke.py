# tests/core/validation/test_smoke.py
import json
from pathlib import Path

import pytest
from janus.core.persona import Persona
from janus.core.backends.generic import GenericBackend
from janus.core.validation.smoke import smoke_run

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "personas" / "echo_brief"


class _EmitBackend(GenericBackend):
    """Scripts a call to emit_output with a given payload (or None to emit nothing)."""
    def __init__(self, *a, payload=None, **k):
        super().__init__(*a, **k)
        self._payload = payload
        self._n = 0

    async def _chat_completion(self):
        self._n += 1
        if self._n == 1 and self._payload is not None:
            return {"message": {"content": "here", "tool_calls": [
                {"id": "c1", "function": {"name": "emit_output",
                                          "arguments": json.dumps(self._payload)}}]}}
        return {"message": {"content": "done"}}

    def _tool_result_message(self, tc, result):
        return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}


@pytest.mark.asyncio
async def test_smoke_passes_on_valid_deliverable(tmp_path):
    persona = Persona.load(FIXTURE)  # output_schema requires {"summary": str}
    wd = tmp_path / "wd"
    backend = _EmitBackend(working_directory=wd, system_prompt=persona.system_prompt,
                           model="m", registry=persona.registry, payload={"summary": "ok"})
    res = await smoke_run(persona, backend, "solar", wd)
    assert res.passed
    assert res.deliverable == {"summary": "ok"}
    assert {c.name for c in res.checks} >= {"run_completed", "deliverable_valid"}


@pytest.mark.asyncio
async def test_smoke_fails_when_no_deliverable(tmp_path):
    persona = Persona.load(FIXTURE)
    wd = tmp_path / "wd"
    backend = _EmitBackend(working_directory=wd, system_prompt=persona.system_prompt,
                           model="m", registry=persona.registry, payload=None)  # never emits
    res = await smoke_run(persona, backend, "solar", wd)
    assert not res.passed
    assert any(c.name == "deliverable_valid" and not c.ok for c in res.checks)


@pytest.mark.asyncio
async def test_smoke_fails_on_invalid_deliverable(tmp_path):
    persona = Persona.load(FIXTURE)
    wd = tmp_path / "wd"
    # emit a payload that violates the schema (summary must be a string; additionalProperties false)
    backend = _EmitBackend(working_directory=wd, system_prompt=persona.system_prompt,
                           model="m", registry=persona.registry, payload={"summary": 123, "extra": 1})
    res = await smoke_run(persona, backend, "solar", wd)
    # emit_output itself rejects invalid payloads, so output.json is never written -> deliverable_valid fails
    assert not res.passed


@pytest.mark.asyncio
async def test_smoke_reads_custom_output_filename(tmp_path):
    """Verify smoke reads from persona.output_filename, not hardcoded output.json."""
    # Create a temporary persona with custom output.filename
    persona_dir = tmp_path / "custom_persona"
    persona_dir.mkdir()
    (persona_dir / "prompt.md").write_text("sys")
    (persona_dir / "output_schema.json").write_text('{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false}')
    manifest = '''
[persona]
name = "custom"
[prompt]
file = "prompt.md"
[task]
template = "Do: {subject}"
[output]
schema_file = "output_schema.json"
filename = "brief.json"
'''
    (persona_dir / "manifest.toml").write_text(manifest)

    persona = Persona.load(persona_dir)
    assert persona.output_filename == "brief.json"

    wd = tmp_path / "wd"
    backend = _EmitBackend(working_directory=wd, system_prompt=persona.system_prompt,
                           model="m", registry=persona.registry, payload={"summary": "ok"})
    res = await smoke_run(persona, backend, "test", wd)

    # Smoke should find the deliverable at brief.json (not output.json)
    assert res.passed
    assert res.deliverable == {"summary": "ok"}
    assert {c.name for c in res.checks} >= {"run_completed", "deliverable_valid"}
