import httpx
import pytest
from janus.core.backend import MessageType
from janus.core.tools.registry import ToolContext, ToolRegistry, tool
from janus.core.backends.ollama import OllamaBackend


def _registry():
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)
    return reg


def _make(tmp_path, responses):
    b = OllamaBackend(working_directory=tmp_path, system_prompt="sys", model="qwen",
                      registry=_registry(), base_url="http://fake:11434")
    seq = iter(responses)
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=next(seq))))
    return b


@pytest.mark.asyncio
async def test_chat_completion_returns_ollama_shape(tmp_path):
    body = {"message": {"content": "hi", "tool_calls": [
        {"function": {"name": "echo", "arguments": {"text": "x"}}}]}, "prompt_eval_count": 12}
    b = _make(tmp_path, [body])
    b._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = await b._chat_completion()
    assert out["message"]["content"] == "hi"
    assert out["message"]["tool_calls"][0]["function"]["name"] == "echo"
    await b._client.aclose()


@pytest.mark.asyncio
async def test_tool_result_message_has_no_id(tmp_path):
    b = _make(tmp_path, [])
    assert b._tool_result_message({"id": "ignored"}, "the result") == {"role": "tool", "content": "the result"}
    await b._client.aclose()


@pytest.mark.asyncio
async def test_full_run_through_ollama(tmp_path):
    responses = [
        {"message": {"content": "calling", "tool_calls": [
            {"function": {"name": "echo", "arguments": {"text": "yo"}}}]}},
        {"message": {"content": "done"}},  # no tool_calls -> loop ends
    ]
    b = _make(tmp_path, responses)
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert any(m.type is MessageType.TOOL_RESULT and m.content == "echoed:yo" for m in msgs)
    assert msgs[-1].type is MessageType.RESULT
    await b.disconnect()
    # disconnect() now closes and nulls the client itself.
    assert b._client is None
