import httpx
import pytest
from janus.core.backend import MessageType
from janus.core.tools.registry import ToolContext, ToolRegistry, tool
from janus.core.backends.openai_compat import MAX_TOKENS_CEILING, OpenAICompatBackend


def _registry():
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)
    return reg


def _make(tmp_path, monkeypatch, responses):
    """Patch the backend's httpx client with a transport returning scripted JSON bodies."""
    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k")
    seq = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(seq))

    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return b


@pytest.mark.asyncio
async def test_chat_completion_parses_openai_shape(tmp_path, monkeypatch):
    body = {
        "choices": [{"message": {"content": "hi", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "echo", "arguments": '{"text":"x"}'}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    b = _make(tmp_path, monkeypatch, [body])
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = await b._chat_completion()
    assert out["message"]["content"] == "hi"
    assert out["message"]["tool_calls"][0]["function"]["name"] == "echo"
    await b._client.aclose()


@pytest.mark.asyncio
async def test_tool_result_message_has_tool_call_id(tmp_path, monkeypatch):
    b = _make(tmp_path, monkeypatch, [])
    msg = b._tool_result_message({"id": "abc"}, "the result")
    assert msg == {"role": "tool", "tool_call_id": "abc", "content": "the result"}
    await b._client.aclose()


@pytest.mark.asyncio
async def test_full_run_through_openai_backend(tmp_path, monkeypatch):
    responses = [
        {"choices": [{"message": {"content": "calling", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": '{"text":"yo"}'}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5}},
        {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 6}},
    ]
    b = _make(tmp_path, monkeypatch, responses)
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert any(m.type is MessageType.TOOL_RESULT and m.content == "echoed:yo" for m in msgs)
    assert msgs[-1].type is MessageType.RESULT
    await b.disconnect()
    # disconnect() now closes and nulls the client itself.
    assert b._client is None


def _no_sleep(monkeypatch):
    """Patch away real backoff sleeps so retry tests run instantly."""

    async def _fake_sleep(*_a, **_k):
        return None

    monkeypatch.setattr("janus.core.backends.openai_compat.asyncio.sleep", _fake_sleep)


@pytest.mark.asyncio
async def test_chat_completion_retries_transient_429_then_succeeds(tmp_path, monkeypatch):
    """A single 429 must be retried (with backoff) rather than read as a clean
    end-of-session -- without the retry path, _post_completion would return None
    and the loop would treat the turn as finished."""
    _no_sleep(monkeypatch)
    calls = {"n": 0}
    ok_body = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=ok_body)

    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k")
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    out = await b._chat_completion()

    assert out["message"]["content"] == "hi"
    assert calls["n"] >= 2
    await b._client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_retries_transient_5xx_then_succeeds(tmp_path, monkeypatch):
    """Same retry path, but for a gateway 503 -- covers the rest of the
    _RETRYABLE_STATUS set beyond 429."""
    _no_sleep(monkeypatch)
    calls = {"n": 0}
    ok_body = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="bad gateway")
        return httpx.Response(200, json=ok_body)

    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k")
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    out = await b._chat_completion()

    assert out["message"]["content"] == "hi"
    assert calls["n"] >= 2
    await b._client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_recovers_from_context_overflow(tmp_path, monkeypatch):
    """A 400 context-overflow response must trigger history compaction/eviction
    and a retry of the same turn -- not a dead end. Bounded by
    MAX_CONTEXT_EVICTIONS so a persistent overflow can't loop forever."""
    _no_sleep(monkeypatch)
    calls = {"n": 0}
    ok_body = {
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400, text='{"error": {"message": "context_length_exceeded"}}'
            )
        return httpx.Response(200, json=ok_body)

    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k")
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # Seed several removable assistant/tool exchanges so eviction has room to work.
    b._messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
    ]
    for i in range(4):
        b._messages.append({
            "role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "echo", "arguments": "{}"}}],
        })
        b._messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i}"})
    before_len = len(b._messages)

    out = await b._chat_completion()

    assert out["message"]["content"] == "done"
    assert len(b._messages) < before_len
    assert calls["n"] >= 2
    await b._client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_bumps_max_tokens_on_length_truncation(tmp_path, monkeypatch):
    """A finish_reason=length response with no tool call must trigger exactly one
    retry at a higher max_tokens (capped at MAX_TOKENS_CEILING), not be treated as
    a normal final answer."""
    import json

    captured_max_tokens = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_max_tokens.append(body["max_tokens"])
        if len(captured_max_tokens) == 1:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5},
            })
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "final"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6},
        })

    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k",
                            max_tokens=100)
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    out = await b._chat_completion()

    assert out["message"]["content"] == "final"
    assert len(captured_max_tokens) == 2
    assert captured_max_tokens[1] > captured_max_tokens[0]
    assert captured_max_tokens[1] == min(100 * 2, MAX_TOKENS_CEILING)
    await b._client.aclose()


@pytest.mark.asyncio
async def test_connect_refines_context_window_without_overwriting_client(tmp_path):
    """connect() must not clobber an already-injected client (the guard tests
    and self-managed callers rely on), and should still refine the context
    window from the /models body when one is advertised."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "m", "context_length": 4096}]})

    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k")
    injected_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    b._client = injected_client

    await b.connect()

    # The guard: connect() must leave an already-set self._client untouched.
    assert b._client is injected_client
    assert b._context_window == 4096
    await b._client.aclose()


@pytest.mark.asyncio
async def test_disconnect_closes_httpx_client(tmp_path):
    """disconnect() must close the httpx client and null the reference.

    An unclosed AsyncClient keeps its connection pool alive and lingers after
    asyncio.run returns (the real cause of the headless process not exiting).
    """
    b = OpenAICompatBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                            registry=_registry(), base_url="http://fake/v1", api_key="k")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    b._client = client

    await b.disconnect()

    assert client.is_closed
    assert b._client is None
    # Idempotent: a second disconnect on the already-closed backend is a no-op.
    await b.disconnect()
