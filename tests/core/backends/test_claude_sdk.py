import sys
from pathlib import Path

import claude_agent_sdk  # real SDK (installed in venv); we mock only ClaudeSDKClient
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from janus.core.backend import MessageType
from janus.core.backends.claude_sdk import ClaudeSDKBackend
from janus.core.tools.registry import ToolContext, ToolRegistry, tool


def _registry():
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)
    return reg


class _FakeClient:
    """Records the options + prompt; yields a scripted SDK message stream."""

    last_options = None

    def __init__(self, options=None):
        _FakeClient.last_options = options
        self.queried = None
        self._script = [
            AssistantMessage(
                content=[
                    TextBlock(text="hi"),
                    ToolUseBlock(id="tu1", name="echo", input={"text": "hi"}),
                ],
                model="m",
            ),
            UserMessage(
                content=[
                    ToolResultBlock(tool_use_id="tu1", content="echoed:hi", is_error=False),
                ]
            ),
            ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s1",
                total_cost_usd=0.01,
            ),
        ]

    async def connect(self): ...

    async def disconnect(self): ...

    async def query(self, prompt):
        self.queried = prompt

    async def receive_response(self):
        for m in self._script:
            yield m


@pytest.fixture
def _patch_client(monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)


@pytest.mark.asyncio
async def test_connect_wires_custom_tools_into_options(tmp_path, _patch_client):
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path),
        system_prompt="SYS",
        model="claude-sonnet-5",
        registry=_registry(),
    )
    await b.connect()
    opts = _FakeClient.last_options
    assert opts.model == "claude-sonnet-5"
    assert opts.system_prompt == "SYS"
    assert "janus" in opts.mcp_servers
    assert "mcp__janus__echo" in opts.allowed_tools


@pytest.mark.asyncio
async def test_receive_translates_text_and_result(tmp_path, _patch_client):
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path),
        system_prompt="s",
        model="claude-sonnet-5",
        registry=_registry(),
    )
    await b.connect()
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert any(m.type is MessageType.TEXT and m.content == "hi" for m in msgs)
    assert msgs[-1].type is MessageType.RESULT
    assert msgs[-1].metadata["cost_usd"] == 0.01


@pytest.mark.asyncio
async def test_receive_translates_tool_start_and_result(tmp_path, _patch_client):
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path),
        system_prompt="s",
        model="claude-sonnet-5",
        registry=_registry(),
    )
    await b.connect()
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    tool_start = next(m for m in msgs if m.type is MessageType.TOOL_START)
    assert tool_start.tool_name == "echo"
    assert tool_start.tool_args == {"text": "hi"}

    tool_result = next(m for m in msgs if m.type is MessageType.TOOL_RESULT)
    assert tool_result.tool_name == "echo"
    assert tool_result.content == "echoed:hi"
    assert tool_result.metadata["is_error"] is False


def test_supports_resume(tmp_path):
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path), system_prompt="s", model="m", registry=_registry()
    )
    assert b.supports_resume is True


def test_auth_manual_blanks_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("JANUS_AUTH_MODE", "manual")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
    assert ClaudeSDKBackend._build_auth_env_overrides()["ANTHROPIC_API_KEY"] == ""


@pytest.mark.asyncio
async def test_query_without_connect_raises(tmp_path):
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path), system_prompt="s", model="m", registry=_registry()
    )
    with pytest.raises(RuntimeError, match="not connected"):
        await b.query("go")


@pytest.mark.asyncio
async def test_tool_context_cwd_is_path(tmp_path, monkeypatch, _patch_client):
    """`ToolContext.cwd` must be a real `Path`, matching every other backend.

    A persona tool that does path arithmetic on `ctx.cwd` (e.g. `ctx.cwd /
    "sub"`) works under GenericBackend but would raise `TypeError` if
    `ClaudeSDKBackend` passed a bare `str` through instead.

    The wrapped `SdkMcpTool`s are NOT stored on the (JSON-serialized) server
    dict, so we spy on the backend's call to `registry_to_sdk_server` to
    capture BOTH the `ctx` the backend constructed and the wrapped tools it
    built with it -- proving the backend passes a real `Path` cwd.
    """
    import janus.core.backends.claude_sdk as claude_sdk_mod
    from janus.core.backends.sdk_tools import registry_to_sdk_server as _real

    reg = ToolRegistry()

    @tool("pathtool", "path arithmetic", {"type": "object", "properties": {}})
    def pathtool(ctx: ToolContext, **kw):
        return str(ctx.cwd / "sub")

    reg.register(pathtool)

    captured: dict = {}

    def _spy(registry, ctx):
        captured["ctx"] = ctx
        result = _real(registry, ctx)
        captured["tools"] = result[2]  # the 3rd return: wrapped SdkMcpTools
        return result

    monkeypatch.setattr(claude_sdk_mod, "registry_to_sdk_server", _spy)

    b = ClaudeSDKBackend(
        working_directory=str(tmp_path),
        system_prompt="s",
        model="m",
        registry=reg,
    )
    await b.connect()

    # The backend must construct ToolContext with a real Path cwd (not a str).
    assert isinstance(captured["ctx"].cwd, Path)
    wrapped = next(t for t in captured["tools"] if t.name == "pathtool")
    result = await wrapped.handler({})

    assert result == {
        "content": [{"type": "text", "text": str(Path(str(tmp_path)) / "sub")}]
    }


@pytest.mark.asyncio
async def test_connect_raises_friendly_error_when_sdk_missing(tmp_path, monkeypatch):
    """`connect()` should surface the friendly SDK-missing `RuntimeError`.

    `_build_options` must call `registry_to_sdk_server` (which raises a
    friendly `RuntimeError` when `claude_agent_sdk` can't be imported)
    before its own bare `import claude_agent_sdk` -- otherwise a plain
    `ModuleNotFoundError` would fire first instead.
    """
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

    b = ClaudeSDKBackend(
        working_directory=str(tmp_path), system_prompt="s", model="m", registry=_registry()
    )
    with pytest.raises(RuntimeError, match="claude-agent-sdk"):
        await b.connect()


@pytest.mark.asyncio
async def test_resume_rebuilds_options_and_sets_session_id(tmp_path, _patch_client):
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path), system_prompt="s", model="m", registry=_registry()
    )
    await b.connect()
    ok = await b.resume("sess-123")
    assert ok is True
    assert b.session_id == "sess-123"
    assert _FakeClient.last_options.resume == "sess-123"


class _FakeClientResultError:
    """Yields only a ResultMessage with is_error=True."""

    def __init__(self, options=None):
        pass

    async def connect(self): ...

    async def disconnect(self): ...

    async def query(self, prompt): ...

    async def receive_response(self):
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="s1",
            total_cost_usd=0.01,
            result="blocked by real-time safety classifier",
        )


class _FakeClientResultOk:
    """Yields only a ResultMessage with is_error=False."""

    def __init__(self, options=None):
        pass

    async def connect(self): ...

    async def disconnect(self): ...

    async def query(self, prompt): ...

    async def receive_response(self):
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s1",
            total_cost_usd=0.01,
        )


@pytest.mark.asyncio
async def test_sdk_result_error_maps_to_outcome_error(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClientResultError)
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path),
        system_prompt="s",
        model="m",
        registry=_registry(),
    )
    await b.connect()
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    # An error result now also surfaces the error DETAIL as an ERROR message so
    # the controller can persist it (last_error) — not just the outcome tag.
    assert [m.type for m in msgs] == [MessageType.ERROR, MessageType.RESULT]
    assert "blocked by real-time safety classifier" in str(msgs[0].content)
    assert msgs[1].metadata["outcome"] == "error"


@pytest.mark.asyncio
async def test_sdk_result_ok_maps_to_outcome_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClientResultOk)
    b = ClaudeSDKBackend(
        working_directory=str(tmp_path),
        system_prompt="s",
        model="m",
        registry=_registry(),
    )
    await b.connect()
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert len(msgs) == 1
    assert msgs[0].type is MessageType.RESULT
    assert msgs[0].metadata["outcome"] == "ok"


import asyncio


class _StallingClient:
    """A fake SDK client whose stream blocks — models the SDK stalled on a tool result."""
    last_options = None
    def __init__(self, options=None):
        _StallingClient.last_options = options
        self._release = asyncio.Event()
    async def connect(self): ...
    async def disconnect(self): ...
    async def query(self, prompt): self.queried = prompt
    async def receive_response(self):
        await self._release.wait()   # never fires in these tests → stream stays open
        return
        yield  # make it an async generator


@pytest.fixture
def _patch_stalling(monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _StallingClient)


@pytest.mark.asyncio
async def test_ask_user_surfaces_awaiting_input_and_parks_for_reply(_patch_stalling):
    b = ClaudeSDKBackend(working_directory="/tmp", system_prompt="s", model="m",
                         registry=_registry())
    b.user_reply_enabled = True
    await b.connect()
    seen = []
    async def consume():
        async for msg in b.receive_messages():
            seen.append(msg)
    consumer = asyncio.ensure_future(consume())
    reply_task = asyncio.ensure_future(b._await_user_reply("What is the secret?", ["a", "b"]))
    await asyncio.sleep(0.05)
    # AWAITING_INPUT surfaced through receive_messages while the SDK stream is stalled
    awaiting = [m for m in seen if m.type == MessageType.AWAITING_INPUT]
    assert awaiting and awaiting[0].content == "What is the secret?"
    assert awaiting[0].metadata["choices"] == ["a", "b"]
    # delivering the reply resumes the parked handler with that exact text
    b.deliver_reply("banana")
    assert await asyncio.wait_for(reply_task, timeout=1.0) == "banana"
    consumer.cancel()


@pytest.mark.asyncio
async def test_ask_user_disabled_wires_no_reply_callback(_patch_client, monkeypatch, tmp_path):
    # user_reply_enabled False → the bridged ToolContext has await_user_reply=None → fail-open.
    captured = {}
    import janus.core.backends.claude_sdk as m
    real = m.registry_to_sdk_server
    def spy(reg, ctx):
        captured["ctx"] = ctx
        return real(reg, ctx)
    monkeypatch.setattr(m, "registry_to_sdk_server", spy)
    b = ClaudeSDKBackend(working_directory=str(tmp_path), system_prompt="s", model="m",
                         registry=_registry())
    b.user_reply_enabled = False
    await b.connect()
    assert captured["ctx"].await_user_reply is None
    # and enabled → a real callback
    captured.clear()
    b2 = ClaudeSDKBackend(working_directory=str(tmp_path), system_prompt="s", model="m",
                          registry=_registry())
    b2.user_reply_enabled = True
    await b2.connect()
    assert captured["ctx"].await_user_reply is not None


@pytest.mark.asyncio
async def test_query_drains_stale_reply(_patch_client):
    b = ClaudeSDKBackend(working_directory="/tmp", system_prompt="s", model="m",
                         registry=_registry())
    await b.connect()
    b.deliver_reply("stale")            # left over from a cancelled park
    await b.query("go")
    assert b._reply_queue.empty()       # a fresh query must not inherit it


@pytest.mark.asyncio
async def test_disconnect_unblocks_a_parked_ask_user(_patch_stalling):
    b = ClaudeSDKBackend(working_directory="/tmp", system_prompt="s", model="m",
                         registry=_registry())
    b.user_reply_enabled = True
    await b.connect()
    # start the receive pump so _out_q has a live consumer, then park a reply
    consumer = asyncio.ensure_future(_drain(b))
    reply_task = asyncio.ensure_future(b._await_user_reply("q?"))
    await asyncio.sleep(0.05)
    await b.disconnect()                # must unblock the parked handler
    result = await asyncio.wait_for(reply_task, timeout=1.0)
    assert "stop" in result.lower()     # returns a stopped-string, does not hang
    consumer.cancel()


async def _drain(b):
    async for _ in b.receive_messages():
        pass
