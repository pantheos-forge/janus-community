import json

import httpx
import pytest
from janus.core.backend import MessageType
from janus.core.tools.registry import ToolContext, ToolRegistry, tool
from janus.core.backends.anthropic_api import AnthropicAPIBackend


def _registry():
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)
    return reg


def _make(tmp_path, responses, captured=None):
    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="SYS", model="claude-sonnet-5",
                            registry=_registry(), api_key="sk-ant-123")
    seq = iter(responses)

    def handler(req: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["headers"] = dict(req.headers)
            captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=next(seq))

    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return b


@pytest.mark.asyncio
async def test_request_shape_separates_system_and_omits_temperature(tmp_path):
    captured = {}
    body = {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3}}
    b = _make(tmp_path, [body], captured)
    b._messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    out = await b._chat_completion()
    assert out["message"]["content"] == "done"
    assert captured["body"]["system"] == "SYS"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["tools"][0]["input_schema"]["type"] == "object"
    assert "temperature" not in captured["body"]
    assert "thinking" not in captured["body"]
    assert captured["headers"]["x-api-key"] == "sk-ant-123"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    await b._client.aclose()


@pytest.mark.asyncio
async def test_tool_result_message_openai_shape(tmp_path):
    b = _make(tmp_path, [])
    assert b._tool_result_message({"id": "toolu_1"}, "res") == {
        "role": "tool", "tool_call_id": "toolu_1", "content": "res"}
    await b._client.aclose()


@pytest.mark.asyncio
async def test_full_run_through_anthropic_backend(tmp_path):
    responses = [
        {"content": [{"type": "text", "text": "calling"},
                     {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "yo"}}],
         "stop_reason": "tool_use", "usage": {"input_tokens": 5}},
        {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn", "usage": {"input_tokens": 6}},
    ]
    b = _make(tmp_path, responses)
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert any(m.type is MessageType.TOOL_RESULT and m.content == "echoed:yo" for m in msgs)
    assert msgs[-1].type is MessageType.RESULT
    await b.disconnect()
    # disconnect() now closes and nulls the client itself.
    assert b._client is None


@pytest.mark.asyncio
async def test_none_on_http_error(tmp_path, monkeypatch):
    # 500 is a transient status, so it will be retried until exhausted before
    # ultimately returning None; stub out the sleep so the test runs instantly.
    import janus.core.backends.anthropic_api as anthropic_api_module

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(anthropic_api_module.asyncio, "sleep", _no_sleep)

    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="s", model="claude-sonnet-5",
                            registry=_registry(), api_key="k")
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, json={"e": 1})))
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert await b._chat_completion() is None
    await b._client.aclose()


def test_default_context_window_is_200k(tmp_path):
    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="s", model="claude-sonnet-5",
                            registry=_registry(), api_key="k")
    assert b._context_window == 200_000


def test_context_window_override(tmp_path):
    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="s", model="claude-sonnet-5",
                            registry=_registry(), api_key="k", context_window=500_000)
    assert b._context_window == 500_000


@pytest.mark.asyncio
async def test_retries_transient_429_then_succeeds(tmp_path, monkeypatch):
    import janus.core.backends.anthropic_api as anthropic_api_module

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(anthropic_api_module.asyncio, "sleep", _no_sleep)

    calls = {"count": 0}
    ok_body = {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
               "usage": {"input_tokens": 10, "output_tokens": 3}}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=ok_body)

    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="s", model="claude-sonnet-5",
                            registry=_registry(), api_key="k")
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    out = await b._chat_completion()

    assert out is not None
    assert out["message"]["content"] == "done"
    assert calls["count"] >= 2
    await b._client.aclose()


@pytest.mark.asyncio
async def test_non_transient_400_returns_none_without_retry(tmp_path, monkeypatch):
    import janus.core.backends.anthropic_api as anthropic_api_module

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(anthropic_api_module.asyncio, "sleep", _no_sleep)

    calls = {"count": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="s", model="claude-sonnet-5",
                            registry=_registry(), api_key="k")
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    assert await b._chat_completion() is None
    assert calls["count"] == 1
    await b._client.aclose()


@pytest.mark.asyncio
async def test_disconnect_closes_httpx_client(tmp_path):
    """disconnect() must close the httpx client and null the reference.

    An unclosed AsyncClient keeps its connection pool alive and lingers after
    asyncio.run returns (the real cause of the headless process not exiting).
    """
    b = AnthropicAPIBackend(working_directory=tmp_path, system_prompt="s", model="claude-sonnet-5",
                            registry=_registry(), api_key="k")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    b._client = client

    await b.disconnect()

    assert client.is_closed
    assert b._client is None
    # Idempotent: a second disconnect on the already-closed backend is a no-op.
    await b.disconnect()
