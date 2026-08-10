# Janus — an engine for building specialized AI agents.
# Copyright (C) 2026 Pantheos Forge
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. This program is distributed WITHOUT ANY WARRANTY;
# see the GNU AGPL <https://www.gnu.org/licenses/> for details.
#
# A persona exception applies — see LICENSE-EXCEPTION.

"""Schema-aware ``emit_output`` deliverable tool.

This is the generalized structured-deliverable tool (a domain-neutral successor to a
domain-specific reporting tool): ``make_emit_output_tool`` builds a ``ToolSpec`` bound
to a persona's JSON Schema, so the model can emit its final structured deliverable once
the task is complete. The payload is validated against the schema before anything is
written — an invalid payload returns an error string (not an exception) so the model can
see what's wrong and retry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from janus.core.tools.registry import ToolContext, ToolSpec, tool

# JSON Schema combinator keywords that Anthropic-family APIs reject at the TOP
# LEVEL of a tool's input_schema (the API 400s the whole request). Discovered
# live: the factory persona's build-report schema uses a top-level allOf for
# if/then conditional requirements. The wire schema sent as tool parameters is
# a stripped projection; payload VALIDATION always uses the full schema, so
# nothing is lost except the hint to the model — and an invalid payload comes
# back as a retryable "Output rejected" error naming what to fix.
_TOP_LEVEL_COMBINATORS = ("allOf", "anyOf", "oneOf", "if", "then", "else")


def _provider_safe_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``schema`` without top-level combinator keywords."""
    return {k: v for k, v in schema.items() if k not in _TOP_LEVEL_COMBINATORS}


def make_emit_output_tool(
    output_schema: dict[str, Any], output_path: str = "output.json"
) -> ToolSpec:
    """Build an ``emit_output`` ToolSpec validated against ``output_schema``.

    On a valid call, the payload is written as JSON to ``ctx.cwd / output_path``
    (creating parent directories as needed) and, if ``ctx.emit_output`` is set,
    passed to it as well. On an invalid call, nothing is written and the handler
    returns an error string describing what to fix.

    The ToolSpec's ``parameters`` (what providers see as the tool's
    input_schema) is a provider-safe projection of ``output_schema`` — top-level
    combinators are stripped because Anthropic-family APIs reject them — while
    the handler validates payloads against the full schema.
    """

    @tool(
        "emit_output",
        "Emit the final structured deliverable once the task is complete. Call this "
        "exactly once, with a payload matching the required output schema.",
        _provider_safe_parameters(output_schema),
    )
    async def _emit_output(ctx: ToolContext, **payload: Any) -> str:
        try:
            jsonschema.validate(instance=payload, schema=output_schema)
        except jsonschema.ValidationError as e:
            return f"Output rejected: {e.message}. Fix the fields and call emit_output again."

        filepath = Path(ctx.cwd) / output_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(payload, indent=2))

        if ctx.emit_output is not None:
            ctx.emit_output(payload)

        return f"Deliverable written to {output_path} ({len(payload)} fields)."

    return _emit_output
