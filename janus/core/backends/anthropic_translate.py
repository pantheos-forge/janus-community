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

"""Pure translation helpers between the OpenAI message/tool shape and the
Anthropic Messages API shape.

Janus Core's generic agent loop keeps its ``_messages`` in the OpenAI shape
(role system/user/assistant/tool; assistant may carry ``tool_calls``; tool
messages carry ``tool_call_id``/``content``) -- this is the shape every Janus
backend uses internally. The native Anthropic backend converts to/from the
Anthropic ``/v1/messages`` shape on the wire using the functions in this
module. All functions here are pure: no I/O, no network, stdlib ``json``
only.
"""

import json


def openai_tools_to_anthropic(payload: list[dict]) -> list[dict]:
    """Map OpenAI tool-spec dicts to Anthropic tool-spec dicts.

    ``{"type": "function", "function": {"name", "description", "parameters"}}``
    becomes ``{"name", "description", "input_schema": <parameters>}``.
    """
    tools = []
    for entry in payload:
        function = entry["function"]
        tools.append({
            "name": function["name"],
            "description": function.get("description"),
            "input_schema": function.get("parameters"),
        })
    return tools


def _parse_arguments(arguments: str) -> dict:
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}


def openai_messages_to_anthropic(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-shaped messages to (system, anthropic_messages).

    The leading system message (if present at index 0) is extracted into the
    returned ``system`` string and removed from the message list. Consecutive
    ``role: "tool"`` messages coalesce into a single Anthropic user message
    holding all of their ``tool_result`` blocks -- this is required so that
    all tool results for a turn arrive in one user message.
    """
    system: str | None = None
    remaining = messages
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content")
        remaining = messages[1:]

    anthropic_messages: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            anthropic_messages.append({
                "role": "user",
                "content": list(pending_tool_results),
            })
            pending_tool_results.clear()

    for message in remaining:
        role = message.get("role")

        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": message.get("content"),
            })
            continue

        # Any non-tool message ends a run of consecutive tool messages.
        flush_tool_results()

        if role == "user":
            anthropic_messages.append({
                "role": "user",
                "content": message.get("content"),
            })
        elif role == "assistant":
            content: list[dict] = []
            text = message.get("content")
            if text:
                content.append({"type": "text", "text": text})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call["function"]
                content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": function["name"],
                    "input": _parse_arguments(function.get("arguments")),
                })
            anthropic_messages.append({"role": "assistant", "content": content})
        else:
            anthropic_messages.append(message)

    flush_tool_results()

    return system, anthropic_messages


def anthropic_response_to_openai_message(data: dict) -> dict:
    """Convert an Anthropic response body to an OpenAI-shaped assistant message."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in data.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    message: dict = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message
