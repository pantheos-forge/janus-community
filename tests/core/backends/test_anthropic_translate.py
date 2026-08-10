import json

from janus.core.backends.anthropic_translate import (
    anthropic_response_to_openai_message,
    openai_messages_to_anthropic,
    openai_tools_to_anthropic,
)


def test_tools_mapping():
    payload = [{"type": "function", "function": {
        "name": "echo", "description": "e", "parameters": {"type": "object", "properties": {}}}}]
    assert openai_tools_to_anthropic(payload) == [
        {"name": "echo", "description": "e", "input_schema": {"type": "object", "properties": {}}}]


def test_system_extracted_and_user_preserved():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    system, out = openai_messages_to_anthropic(msgs)
    assert system == "SYS"
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_tool_call_becomes_tool_use():
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "let me call", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "echo", "arguments": '{"text": "x"}'}}]},
    ]
    _system, out = openai_messages_to_anthropic(msgs)
    assert out[1]["role"] == "assistant"
    blocks = out[1]["content"]
    assert blocks[0] == {"type": "text", "text": "let me call"}
    assert blocks[1] == {"type": "tool_use", "id": "t1", "name": "echo", "input": {"text": "x"}}


def test_consecutive_tool_results_coalesce_into_one_user_message():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "a", "arguments": "{}"}},
            {"id": "t2", "function": {"name": "b", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r1"},
        {"role": "tool", "tool_call_id": "t2", "content": "r2"},
    ]
    _system, out = openai_messages_to_anthropic(msgs)
    # one assistant message, then ONE user message holding both tool_result blocks
    assert out[-1]["role"] == "user"
    results = out[-1]["content"]
    assert results == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
    ]


def test_malformed_tool_arguments_yield_empty_input():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "function": {"name": "a", "arguments": "{not json"}}]}]
    _system, out = openai_messages_to_anthropic(msgs)
    assert out[0]["content"][0] == {"type": "tool_use", "id": "t1", "name": "a", "input": {}}


def test_response_with_text_only():
    data = {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}
    msg = anthropic_response_to_openai_message(data)
    assert msg["role"] == "assistant"
    assert msg["content"] == "done"
    assert not msg.get("tool_calls")


def test_response_with_tool_use_roundtrips_id_and_args():
    data = {"content": [
        {"type": "text", "text": "calling"},
        {"type": "tool_use", "id": "toolu_9", "name": "echo", "input": {"text": "hi"}}],
        "stop_reason": "tool_use"}
    msg = anthropic_response_to_openai_message(data)
    assert msg["content"] == "calling"
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_9"
    assert tc["function"]["name"] == "echo"
    assert json.loads(tc["function"]["arguments"]) == {"text": "hi"}
