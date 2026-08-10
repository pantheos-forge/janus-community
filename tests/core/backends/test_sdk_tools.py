from pathlib import Path

import pytest

from janus.core.backends.sdk_tools import MCP_SERVER_NAME, registry_to_sdk_server
from janus.core.tools.registry import ToolContext, ToolRegistry, tool


def _registry():
    reg = ToolRegistry()

    @tool("echo", "echo tool", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)
    return reg


def test_allowed_tool_names(tmp_path: Path):
    reg = _registry()
    _server, allowed, _tools = registry_to_sdk_server(reg, ToolContext(cwd=tmp_path))
    assert allowed == [f"mcp__{MCP_SERVER_NAME}__echo"]


def test_server_has_expected_shape(tmp_path: Path):
    # The real claude_agent_sdk.create_sdk_mcp_server(...) returns a plain dict
    # (a McpSdkServerConfig TypedDict) with keys "type", "name", "instance".
    reg = _registry()
    server, _allowed, _tools = registry_to_sdk_server(reg, ToolContext(cwd=tmp_path))
    assert isinstance(server, dict)
    assert server["type"] == "sdk"
    assert server["name"] == MCP_SERVER_NAME
    assert "instance" in server
    # The dict must stay SDK-clean: no extra keys leaking non-serializable objects.
    assert set(server) == {"type", "name", "instance"}


def test_server_dict_is_json_serializable_for_the_cli(tmp_path: Path):
    # The SDK CLI transport json.dumps the mcp-server config, stripping only the
    # "instance" field (subprocess_cli._build_command). The rest must survive that
    # -- a regression guard for the SdkMcpTool-leak that broke every real connect().
    import json

    reg = _registry()
    server, allowed, tools = registry_to_sdk_server(reg, ToolContext(cwd=tmp_path))
    for_cli = {k: v for k, v in server.items() if k != "instance"}
    json.dumps({"mcpServers": {"janus": for_cli}})  # must NOT raise
    assert tools and allowed  # wrapped tools reachable via the 3rd return


@pytest.mark.asyncio
async def test_sdk_tool_handler_bridges_to_registry_dispatch(tmp_path: Path):
    reg = _registry()
    # The wrapped SdkMcpTool objects (whose handlers bridge to the registry) are
    # returned as the third element -- NOT stuffed into the server dict, which
    # must stay JSON-serializable for the SDK CLI transport.
    _server, _allowed, sdk_tools = registry_to_sdk_server(reg, ToolContext(cwd=tmp_path))
    echo_tool = next(t for t in sdk_tools if t.name == "echo")
    out = await echo_tool.handler({"text": "hi"})
    assert out == {"content": [{"type": "text", "text": "echoed:hi"}]}


def test_missing_sdk_raises_actionable_error(tmp_path: Path, monkeypatch):
    # Simulate the SDK being absent: make `import claude_agent_sdk` fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
            raise ImportError("no sdk")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="claude-agent-sdk"):
        registry_to_sdk_server(_registry(), ToolContext(cwd=tmp_path))
