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

"""Tool registry — the single source of truth for agent tools.

A `ToolSpec` couples a name, a JSON-schema `parameters` object, and a handler.
Handlers receive a `ToolContext` (cwd + optional output emitter) plus the
tool-call arguments as keyword args, and return a string result. The generic
loop (Plan 2) and the Claude-SDK backend (Plan 2) both consume a `ToolRegistry`.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ToolHandler = Callable[..., "Awaitable[str] | str"]

# ANSI/terminal control sequences leak from shelled-out tools into tool-result strings.
# Some model backends template message history through a parser that rejects raw ESC
# (U+001B) bytes. Strip CSI/OSC/other ESC sequences and stray C0 control bytes (keeping
# tab/newline/CR) before tool output enters message history.
_ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI: ESC [ ... letter
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC: ESC ] ... BEL or ST
    r"|\x1b[@-Z\\-_]"  # other 2-byte ESC sequences
)
_C0_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_terminal_control(text: str) -> str:
    """Remove ANSI escape sequences and stray C0 control bytes from tool output."""
    return _C0_CONTROL_RE.sub("", _ANSI_RE.sub("", text))


@dataclass
class ToolContext:
    cwd: Path
    emit_output: Callable[[dict[str, Any]], None] | None = None
    await_user_reply: Callable[[str, list[str] | None], Awaitable[str]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    name: str, description: str, parameters: dict[str, Any]
) -> Callable[[ToolHandler], ToolSpec]:
    def decorator(handler: ToolHandler) -> ToolSpec:
        return ToolSpec(name=name, description=description, parameters=parameters, handler=handler)

    return decorator


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def openai_payload(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._specs.values()]

    async def dispatch(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        spec = self._specs.get(name)
        if spec is None:
            return _strip_terminal_control(f"Unknown tool: {name}")
        params = list(inspect.signature(spec.handler).parameters.values())
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            call_args = args
        else:
            accepted_names = {p.name for p in params[1:]}
            call_args = {k: v for k, v in args.items() if k in accepted_names}
        try:
            result = spec.handler(ctx, **call_args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:  # noqa: BLE001 — the convention boundary:
            # handlers return error strings, never raise. Anything that
            # escapes one (e.g. an unanticipated exception class from a
            # malformed model-emitted argument) must become a retryable
            # tool error, not kill the agent loop. CancelledError is a
            # BaseException and still propagates (a parked ask_user dies
            # cleanly on stop).
            return _strip_terminal_control(f"Error: tool {name!r} failed: {e}")
        return _strip_terminal_control(result)
