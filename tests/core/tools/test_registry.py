from pathlib import Path

import pytest
from janus.core.tools.registry import ToolContext, ToolRegistry, ToolSpec, tool


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


def test_tool_decorator_builds_spec():
    @tool("echo", "Echo text", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    def echo(ctx, text=""):
        return text

    assert isinstance(echo, ToolSpec)
    assert echo.name == "echo"
    assert echo.to_openai_schema() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    }


@pytest.mark.asyncio
async def test_dispatch_sync_handler(tmp_path):
    @tool("echo", "Echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx, text=""):
        return f"got:{text}"

    reg = ToolRegistry()
    reg.register(echo)
    assert reg.names() == ["echo"]
    assert await reg.dispatch("echo", {"text": "hi"}, _ctx(tmp_path)) == "got:hi"


@pytest.mark.asyncio
async def test_dispatch_async_handler(tmp_path):
    @tool("aecho", "Async echo", {"type": "object", "properties": {}})
    async def aecho(ctx, **kw):
        return "async-ok"

    reg = ToolRegistry()
    reg.register(aecho)
    assert await reg.dispatch("aecho", {}, _ctx(tmp_path)) == "async-ok"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool(tmp_path):
    reg = ToolRegistry()
    assert await reg.dispatch("nope", {}, _ctx(tmp_path)) == "Unknown tool: nope"


@pytest.mark.asyncio
async def test_dispatch_strips_ansi_escape_sequences(tmp_path):
    @tool("colored", "Returns ANSI-colored text", {"type": "object", "properties": {}})
    def colored(ctx, **kw):
        return "before\x1b[31mred\x1b[0mafter"

    reg = ToolRegistry()
    reg.register(colored)
    out = await reg.dispatch("colored", {}, _ctx(tmp_path))
    assert out == "beforeredafter"
    assert "\x1b" not in out


@pytest.mark.asyncio
async def test_dispatch_strips_stray_control_bytes_but_preserves_whitespace(tmp_path):
    @tool("bel", "Returns a bare control byte", {"type": "object", "properties": {}})
    def bel(ctx, **kw):
        return "line1\x07line2\nkept\ttabs\rand-cr"

    reg = ToolRegistry()
    reg.register(bel)
    out = await reg.dispatch("bel", {}, _ctx(tmp_path))
    assert out == "line1line2\nkept\ttabs\rand-cr"
    assert "\x07" not in out


@pytest.mark.asyncio
async def test_dispatch_filters_unexpected_kwargs_for_explicit_handler(tmp_path):
    @tool("known", "Known handler", {"type": "object", "properties": {"path": {"type": "string"}}})
    def known(ctx, path=""):
        return f"path={path}"

    reg = ToolRegistry()
    reg.register(known)
    out = await reg.dispatch("known", {"path": "x", "bogus": 1, "limit": 5}, _ctx(tmp_path))
    assert out == "path=x"


@pytest.mark.asyncio
async def test_dispatch_passes_all_kwargs_to_var_keyword_handler(tmp_path):
    @tool("greedy", "Greedy handler", {"type": "object", "properties": {}})
    def greedy(ctx, **kw):
        return ",".join(sorted(kw))

    reg = ToolRegistry()
    reg.register(greedy)
    out = await reg.dispatch("greedy", {"a": 1, "b": 2}, _ctx(tmp_path))
    assert out == "a,b"


def test_openai_payload_lists_all_specs():
    @tool("a", "A", {"type": "object", "properties": {}})
    def a(ctx, **kw): return ""

    @tool("b", "B", {"type": "object", "properties": {}})
    def b(ctx, **kw): return ""

    reg = ToolRegistry()
    reg.register(a)
    reg.register(b)
    payload = reg.openai_payload()
    assert [p["function"]["name"] for p in payload] == ["a", "b"]


@pytest.mark.asyncio
async def test_dispatch_converts_handler_exceptions_to_error_strings(tmp_path):
    """Live-capstone crash: an unanticipated exception class escaping a tool
    handler killed the whole agent loop (run mislabeled completed). Dispatch
    is the convention boundary: handlers that raise become retryable error
    strings the model can act on."""
    from janus.core.tools.registry import ToolContext, ToolRegistry, tool

    @tool("boom", "always raises", {"type": "object", "properties": {}})
    def _boom(ctx):
        raise RuntimeError("URL can't contain control characters")

    reg = ToolRegistry()
    reg.register(_boom)
    result = await reg.dispatch("boom", {}, ToolContext(cwd=tmp_path))
    assert result.startswith("Error: tool 'boom' failed:")
    assert "control characters" in result


@pytest.mark.asyncio
async def test_dispatch_never_swallows_cancellation(tmp_path):
    """A parked ask_user dies via CancelledError propagating through its
    handler — the dispatch boundary must not absorb cancellation."""
    import asyncio

    from janus.core.tools.registry import ToolContext, ToolRegistry, tool

    @tool("parks", "parks forever", {"type": "object", "properties": {}})
    async def _parks(ctx):
        await asyncio.sleep(3600)
        return "never"

    reg = ToolRegistry()
    reg.register(_parks)
    task = asyncio.ensure_future(reg.dispatch("parks", {}, ToolContext(cwd=tmp_path)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
