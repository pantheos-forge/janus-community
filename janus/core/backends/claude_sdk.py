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

"""`AgentBackend` implementation backed by the Claude Agent SDK.

`ClaudeSDKBackend` is a pull-model backend wrapping `claude_agent_sdk`'s
`ClaudeSDKClient` (native Claude via OAuth/Claude-Code auth). It ports the
read-only upstream `ClaudeCodeBackend`, adding one thing: it wires the
registry's custom tools into `ClaudeAgentOptions` via
`janus.core.backends.sdk_tools.registry_to_sdk_server`, so native Claude gets
the exact same persona tool set as every other backend.

`claude_agent_sdk` is imported lazily inside method bodies (rather than at
module import time) so tests can patch `claude_agent_sdk.ClaudeSDKClient`
directly on the module without requiring the SDK to be installed to even
import this module.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from janus.core.backend import AgentBackend, AgentMessage, MessageType
from janus.core.backends.sdk_tools import registry_to_sdk_server
from janus.core.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_STOP_REPLY = object()  # pushed on disconnect to unblock a parked ask_user


def _sdk_error_detail(msg: Any) -> str:
    """Best-effort human-readable error text from an errored ResultMessage.

    The SDK spreads the reason across several optional fields; prefer the most
    descriptive available so the controller can persist a real ``last_error``
    instead of a bare "run ended with an error".
    """
    result = getattr(msg, "result", None)
    if result and str(result).strip():
        return str(result).strip()
    errors = getattr(msg, "errors", None)
    if errors:
        return "; ".join(str(e) for e in errors)
    reason = getattr(msg, "terminal_reason", None) or getattr(msg, "subtype", None)
    api = getattr(msg, "api_error_status", None)
    if reason:
        return f"Claude Code error: {reason}" + (f" (HTTP {api})" if api else "")
    if api:
        return f"Claude Code API error (HTTP {api})"
    return "Claude Code run ended with an error"


class ClaudeSDKBackend(AgentBackend):
    """`AgentBackend` implementation backed by the Claude Agent SDK.

    **Capabilities (spec §5.4/§6):**
    - `ask_user()` parks for a real reply when the run is interactive
      (`user_reply_enabled`, set by `AgentController.enable_user_replies()`):
      the bridged handler surfaces `AWAITING_INPUT` and blocks until
      `deliver_reply()` (via `AgentController.reply()`) supplies the answer,
      which becomes the tool result. When NOT interactive (piped/keyless) it
      fails open, returning "No user is available in this run; proceed with
      reasonable assumptions and complete the task."
    - `pause()` is display-only (the ABC's no-op); there is no loop gate.
      Backends do not halt model calls; paused runs continue processing in the
      background.
    """

    #: Backoff between retry attempts on a transient SDK stream error.
    _RETRY_BACKOFF_SECONDS: float = 5.0
    #: Consecutive (not cumulative) transient errors before giving up.
    _MAX_CONSECUTIVE_ERRORS: int = 10

    def __init__(
        self,
        working_directory: str,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        permission_mode: str = "bypassPermissions",
    ) -> None:
        self._working_directory = working_directory
        self._system_prompt = system_prompt
        self._model = model
        self._registry = registry
        self._permission_mode = permission_mode
        self._client: Any = None
        self._session_id: str | None = None
        self.user_reply_enabled: bool = False
        self._reply_queue: asyncio.Queue = asyncio.Queue()
        self._out_q: asyncio.Queue = asyncio.Queue()

    @staticmethod
    def _build_auth_env_overrides() -> dict[str, str]:
        """Build the SDK `env` override dict for the current auth mode.

        Reads `JANUS_AUTH_MODE` (default `"manual"`) fresh from the
        environment each call, so tests using `monkeypatch.setenv` see the
        effect without needing to reconstruct the backend.
        """
        auth_mode = os.environ.get("JANUS_AUTH_MODE", "manual")
        env_overrides: dict[str, str] = {}

        if auth_mode == "manual":
            if "ANTHROPIC_API_KEY" in os.environ:
                # Blank it so the SDK falls back to its own OAuth flow
                # instead of picking up a stray env-level API key.
                env_overrides["ANTHROPIC_API_KEY"] = ""
        elif auth_mode == "openrouter":
            for key in (
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_AUTH_TOKEN",
                "NO_PROXY",
                "DISABLE_TELEMETRY",
                "DISABLE_COST_WARNINGS",
                "API_TIMEOUT_MS",
            ):
                if key in os.environ:
                    env_overrides[key] = os.environ[key]
        # "anthropic" mode (and anything else): no overrides -- the API key
        # in the environment is left as-is for the SDK to use directly.

        return env_overrides

    async def _await_user_reply(self, question: str,
                                choices: list[str] | None = None) -> str:
        """ask_user bridge: surface AWAITING_INPUT out-of-band, then park for the reply.

        The bridged MCP handler runs in-process, so it can await here while the
        SDK stream stalls on the tool result. ``receive_messages`` merges the
        ``AWAITING_INPUT`` we push onto ``_out_q`` so the question reaches the
        controller/TUI during the pause.
        """
        await self._out_q.put((
            "ctl",
            AgentMessage(type=MessageType.AWAITING_INPUT, content=question,
                         metadata={"choices": list(choices or [])}),
        ))
        reply = await self._reply_queue.get()
        if reply is _STOP_REPLY:
            return "The run was stopped before a reply was provided."
        return reply

    def deliver_reply(self, text: str) -> None:
        """Deliver the user's reply to a parked ask_user call."""
        self._reply_queue.put_nowait(text)

    def _build_options(self, *, resume: str | None = None) -> Any:
        """Build `ClaudeAgentOptions`, wiring the registry's custom tools in.

        Calls `registry_to_sdk_server` before importing `claude_agent_sdk`
        directly, so that when the SDK is absent, `registry_to_sdk_server`'s
        friendly `RuntimeError` ("... requires 'claude-agent-sdk' ...") is
        what surfaces -- not a bare `ModuleNotFoundError` from this method's
        own import.
        """
        server, allowed, _tools = registry_to_sdk_server(
            self._registry,
            ToolContext(
                cwd=Path(self._working_directory),
                await_user_reply=(self._await_user_reply
                                  if self.user_reply_enabled else None),
            ),
        )

        import claude_agent_sdk

        return claude_agent_sdk.ClaudeAgentOptions(
            cwd=self._working_directory,
            permission_mode=self._permission_mode,  # type: ignore[arg-type]
            system_prompt=self._system_prompt,
            model=self._model,
            env=self._build_auth_env_overrides(),
            mcp_servers={"janus": server},
            allowed_tools=allowed,
            **({"resume": resume} if resume else {}),
        )

    async def connect(self) -> None:
        """Create and connect a fresh `ClaudeSDKClient` for this session."""
        options = self._build_options()

        import claude_agent_sdk

        self._client = claude_agent_sdk.ClaudeSDKClient(options=options)

        result = self._client.connect()
        if result is not None:
            await result

    async def disconnect(self) -> None:
        """Disconnect the underlying SDK client, if one is connected.

        Idempotent under concurrent double-call: the client reference is
        captured and nulled BEFORE awaiting teardown, so a second concurrent
        caller sees ``None`` and no-ops while the first tears down a local.
        """
        # Unblock any ask_user handler parked on a reply so its coroutine
        # doesn't leak when the run is torn down / stopped.
        self._reply_queue.put_nowait(_STOP_REPLY)

        client, self._client = self._client, None
        if client is not None:
            result = client.disconnect()
            if result is not None:
                await result

    async def query(self, prompt: str) -> None:
        """Forward `prompt` to the connected SDK client."""
        if self._client is None:
            raise RuntimeError("Backend not connected")

        # A new query is a fresh turn: drop any reply left over from a
        # cancelled ask_user park so it can't satisfy a future question.
        while not self._reply_queue.empty():
            self._reply_queue.get_nowait()

        result = self._client.query(prompt)
        if result is not None:
            await result

    def _translate_sdk(
        self, msg: Any, pending_tool_names: dict[str, str]
    ) -> list[AgentMessage]:
        """Translate a single SDK message into zero or more `AgentMessage`s.

        `pending_tool_names` is mutated in place: SDK tool-result blocks carry
        only the `tool_use_id`, not the tool name -- it's remembered here from
        the matching `ToolUseBlock` so tool results can still be reported with
        a `tool_name`.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        out: list[AgentMessage] = []

        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    out.append(AgentMessage(type=MessageType.TEXT, content=block.text))
                elif isinstance(block, ToolUseBlock):
                    pending_tool_names[block.id] = block.name
                    out.append(
                        AgentMessage(
                            type=MessageType.TOOL_START,
                            content=None,
                            tool_name=block.name,
                            tool_args=block.input,
                        )
                    )
                # Other block kinds (e.g. ThinkingBlock) are silently
                # skipped -- no wire-format equivalent.
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        content = block.content
                        if isinstance(content, list):
                            content = "\n".join(
                                part["text"]
                                for part in content
                                if isinstance(part, dict) and "text" in part
                            )
                        tool_name = pending_tool_names.pop(block.tool_use_id, None)
                        out.append(
                            AgentMessage(
                                type=MessageType.TOOL_RESULT,
                                content=content,
                                tool_name=tool_name,
                                metadata={"is_error": block.is_error or False},
                            )
                        )
        elif isinstance(msg, ResultMessage):
            is_error = bool(getattr(msg, "is_error", False))
            if is_error:
                out.append(
                    AgentMessage(type=MessageType.ERROR, content=_sdk_error_detail(msg))
                )
            out.append(
                AgentMessage(
                    type=MessageType.RESULT,
                    content=None,
                    metadata={
                        "cost_usd": getattr(msg, "total_cost_usd", 0),
                        "outcome": "error" if is_error else "ok",
                    },
                )
            )
        # Any other SDK message type (e.g. SystemMessage) is skipped
        # silently -- no wire-format equivalent.
        return out

    async def receive_messages(self) -> AsyncIterator[AgentMessage]:
        """Merge the SDK stream with the ask_user AWAITING_INPUT control channel."""
        if self._client is None:
            raise RuntimeError("Backend not connected")
        # Fresh per run. `_await_user_reply` (running in the in-process MCP
        # handler) pushes onto this same `self._out_q`; that is safe because the
        # controller calls receive_messages() exactly once per run, before any
        # tool can fire, and never re-enters it concurrently.
        self._out_q = asyncio.Queue()
        pending_tool_names: dict[str, str] = {}

        async def _pump() -> None:
            try:
                async for m in self._receive_with_retry():
                    await self._out_q.put(("sdk", m))
            except Exception as e:  # let CancelledError propagate (real teardown)
                await self._out_q.put(("err", e))
            finally:
                await self._out_q.put(("end", None))

        pump_task = asyncio.ensure_future(_pump())
        try:
            while True:
                kind, payload = await self._out_q.get()
                if kind == "end":
                    break
                if kind == "err":
                    raise payload
                if kind == "ctl":
                    yield payload
                    continue
                for am in self._translate_sdk(payload, pending_tool_names):
                    yield am
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except BaseException:
                pass

    async def _receive_with_retry(self) -> AsyncIterator[Any]:
        """Wrap `client.receive_response()`, tolerating transient errors.

        Re-enters the SDK stream (a fresh `receive_response()` call, not a
        session resume) on recognized transient errors, backing off on rate
        limits and skipping silently on stray "unknown message type" errors.
        Anything else, or `_MAX_CONSECUTIVE_ERRORS` transient errors in a
        row with no successful message in between, is re-raised.
        """
        consecutive_errors = 0

        while True:
            try:
                async for msg in self._client.receive_response():
                    consecutive_errors = 0
                    yield msg
                return
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "Giving up after %d consecutive SDK stream errors: %s",
                        consecutive_errors,
                        exc,
                    )
                    raise

                error_text = str(exc).lower()
                if "rate_limit" in error_text or "rate limit" in error_text:
                    logger.warning(
                        "Transient rate-limit error from SDK stream, "
                        "backing off %.1fs: %s",
                        self._RETRY_BACKOFF_SECONDS,
                        exc,
                    )
                    await asyncio.sleep(self._RETRY_BACKOFF_SECONDS)
                    continue
                elif "unknown message type" in error_text:
                    logger.debug("Skipping unknown SDK message type: %s", exc)
                    continue
                else:
                    raise

    @property
    def session_id(self) -> str | None:
        """The session id set by a successful `resume()`, else None."""
        return self._session_id

    @property
    def supports_resume(self) -> bool:
        return True

    async def resume(self, session_id: str) -> bool:
        """Disconnect any existing client and reconnect resuming `session_id`."""
        if self._client is not None:
            result = self._client.disconnect()
            if result is not None:
                await result

        options = self._build_options(resume=session_id)

        import claude_agent_sdk

        self._client = claude_agent_sdk.ClaudeSDKClient(options=options)

        result = self._client.connect()
        if result is not None:
            await result

        self._session_id = session_id
        return True
