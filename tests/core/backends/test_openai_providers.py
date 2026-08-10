import httpx
import pytest
from janus.core.tools.registry import ToolContext, ToolRegistry, tool
from janus.core.backends.ds4 import Ds4Backend
from janus.core.backends.openrouter import OpenRouterBackend


def _registry():
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {}})
    def echo(ctx: ToolContext, **kw):
        return "ok"

    reg.register(echo)
    return reg


def test_ds4_defaults_and_extra_payload(tmp_path):
    b = Ds4Backend(working_directory=tmp_path, system_prompt="s", model="deepseek-v4-flash", registry=_registry())
    assert b._base_url == "http://127.0.0.1:8000"
    assert b._extra_payload() == {"reasoning_effort": "low"}
    assert b._request_headers() == {}  # local, no auth


def test_ds4_empty_reasoning_effort(tmp_path):
    b = Ds4Backend(working_directory=tmp_path, system_prompt="s", model="deepseek-v4-flash", registry=_registry(), reasoning_effort="")
    assert b._extra_payload() == {}


def test_openrouter_requires_api_key(tmp_path):
    with pytest.raises(ValueError):
        OpenRouterBackend(working_directory=tmp_path, system_prompt="s", model="anthropic/claude-sonnet-5",
                          registry=_registry(), api_key="")


def test_openrouter_headers_and_extra_payload(tmp_path):
    b = OpenRouterBackend(working_directory=tmp_path, system_prompt="s", model="anthropic/claude-sonnet-5",
                          registry=_registry(), api_key="sk-or-123", referer="https://x", title="Janus")
    assert b._base_url == "https://openrouter.ai/api/v1"
    h = b._request_headers()
    assert h["Authorization"] == "Bearer sk-or-123"
    assert h["HTTP-Referer"] == "https://x"
    assert h["X-Title"] == "Janus"
    assert b._extra_payload() == {"usage": {"include": True}}


@pytest.mark.asyncio
async def test_ds4_full_run_uses_reasoning_effort(tmp_path):
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 5}})

    b = Ds4Backend(working_directory=tmp_path, system_prompt="s", model="deepseek-v4-flash", registry=_registry())
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert captured["reasoning_effort"] == "low"
    await b.disconnect()
    # disconnect() now closes and nulls the client itself.
    assert b._client is None
