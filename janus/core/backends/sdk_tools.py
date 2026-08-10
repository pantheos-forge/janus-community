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

"""Registry -> Claude Agent SDK custom-tools adapter.

Wraps every `ToolSpec` in a `ToolRegistry` as a `claude_agent_sdk.tool`, bundles
them into a single in-process MCP server, and computes the matching
`allowed_tools` names so native Claude (via the Claude Agent SDK) can use the
exact same persona tool set as every other backend.

The `claude_agent_sdk` import is lazy: importing this module must never
require the SDK to be installed, only calling `registry_to_sdk_server` does.
"""

from __future__ import annotations

from typing import Any

from janus.core.tools.registry import ToolContext, ToolRegistry

MCP_SERVER_NAME = "janus"


def _make_handler(registry: ToolRegistry, name: str, ctx: ToolContext):
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        result = await registry.dispatch(name, args, ctx)
        return {"content": [{"type": "text", "text": result}]}

    return handler


def registry_to_sdk_server(
    registry: ToolRegistry, ctx: ToolContext
) -> tuple[object, list[str], list]:
    """Wrap `registry`'s tools as an SDK MCP server plus allowed-tool names.

    Returns `(sdk_mcp_server, allowed_tool_names, wrapped_tools)`.
    `sdk_mcp_server` is the dict-shaped `McpSdkServerConfig` returned by
    `claude_agent_sdk.create_sdk_mcp_server` — kept SDK-clean (only
    `type`/`name`/`instance`) because the SDK's CLI transport JSON-serializes
    this dict (stripping only `instance`), so any extra key holding
    non-serializable objects breaks `connect()`. The raw list of wrapped
    `SdkMcpTool` objects (whose handlers are otherwise only closed over inside
    the MCP `Server` instance) is returned as the third element for callers
    and tests that need to reach them directly.
    """
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise RuntimeError(
            "The Claude Agent SDK backend requires 'claude-agent-sdk'. "
            "Install it: pip install claude-agent-sdk"
        ) from exc

    names = registry.names()
    sdk_tools = []
    for name in names:
        spec = registry.get(name)
        assert spec is not None
        handler = _make_handler(registry, name, ctx)
        sdk_tools.append(
            claude_agent_sdk.tool(spec.name, spec.description, spec.parameters)(handler)
        )

    server = claude_agent_sdk.create_sdk_mcp_server(
        name=MCP_SERVER_NAME, version="1.0.0", tools=sdk_tools
    )

    allowed = [f"mcp__{MCP_SERVER_NAME}__{n}" for n in names]
    return server, allowed, sdk_tools
