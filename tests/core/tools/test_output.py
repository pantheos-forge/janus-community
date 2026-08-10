import json

import pytest
from janus.core.tools.output import make_emit_output_tool
from janus.core.tools.registry import ToolContext

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "score": {"type": "number"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def test_spec_uses_schema_as_parameters():
    spec = make_emit_output_tool(SCHEMA)
    assert spec.name == "emit_output"
    assert spec.parameters == SCHEMA


@pytest.mark.asyncio
async def test_valid_payload_writes_deliverable_and_notifies(tmp_path):
    seen = []
    ctx = ToolContext(cwd=tmp_path, emit_output=seen.append)
    spec = make_emit_output_tool(SCHEMA, output_path="brief.json")
    out = await spec.handler(ctx, summary="hello", score=0.9)
    assert "brief.json" in out
    written = json.loads((tmp_path / "brief.json").read_text())
    assert written == {"summary": "hello", "score": 0.9}
    assert seen == [{"summary": "hello", "score": 0.9}]


@pytest.mark.asyncio
async def test_invalid_payload_returns_error_and_does_not_write(tmp_path):
    ctx = ToolContext(cwd=tmp_path)  # emit_output None
    spec = make_emit_output_tool(SCHEMA, output_path="brief.json")
    out = await spec.handler(ctx, score="not-a-number")  # missing required 'summary', wrong type
    assert "rejected" in out.lower()
    assert not (tmp_path / "brief.json").exists()


@pytest.mark.asyncio
async def test_emit_output_none_is_tolerated(tmp_path):
    ctx = ToolContext(cwd=tmp_path)  # no emit_output callback
    spec = make_emit_output_tool(SCHEMA)
    out = await spec.handler(ctx, summary="ok")
    assert "output.json" in out
    assert (tmp_path / "output.json").exists()


CONDITIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["exported", "failed"]},
        "diagnosis": {"type": "string"},
    },
    "required": ["status"],
    "additionalProperties": False,
    "allOf": [
        {"if": {"properties": {"status": {"const": "failed"}}},
         "then": {"required": ["diagnosis"]}},
    ],
}


def test_parameters_are_provider_safe_for_conditional_schemas():
    """Anthropic-family APIs reject tool input_schemas with top-level
    oneOf/allOf/anyOf (live capstone 400, req_011CdMEikzaMWGdHXybm45Bb). The
    ToolSpec parameters must be a stripped projection; validation keeps the
    full schema."""
    spec = make_emit_output_tool(CONDITIONAL_SCHEMA)
    for kw in ("allOf", "anyOf", "oneOf", "if", "then", "else"):
        assert kw not in spec.parameters
    assert spec.parameters["properties"] == CONDITIONAL_SCHEMA["properties"]
    assert spec.parameters["required"] == CONDITIONAL_SCHEMA["required"]


@pytest.mark.asyncio
async def test_conditional_requirements_still_enforced_on_payloads(tmp_path):
    """Stripping combinators from the wire schema must NOT weaken validation:
    a payload violating the if/then conditional is still rejected."""
    ctx = ToolContext(cwd=tmp_path)
    spec = make_emit_output_tool(CONDITIONAL_SCHEMA, output_path="out.json")
    out = await spec.handler(ctx, status="failed")  # missing conditional 'diagnosis'
    assert "rejected" in out.lower()
    assert not (tmp_path / "out.json").exists()
    ok = await spec.handler(ctx, status="failed", diagnosis="the rubric is unmeetable")
    assert "rejected" not in ok.lower()
    assert (tmp_path / "out.json").exists()
