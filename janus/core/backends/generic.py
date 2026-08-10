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

"""GenericBackend — queue-driven agentic loop, registry-aware.

Ported from the upstream proprietary agent's ``OllamaBackend``, stripped of the
Ollama-specific
transport (client/base-url/health-check) and domain-specific bookkeeping
unrelated to the generic loop. The wire seams (``_chat_completion``,
``_tool_result_message``) are left abstract here; concrete subclasses (e.g. an
Ollama or OpenAI-compatible backend) implement them. ``_agent_loop`` calls the
model, emits any text, and dispatches tool calls through the registry-driven
tool context; it repeats until the model stops calling tools or the
``MAX_ITERATIONS`` ceiling is hit.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from janus.core.backend import AgentBackend, AgentMessage, MessageType
from janus.core.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

# Hard ceiling on agentic-loop iterations, guarding against a model that never
# stops calling tools.
MAX_ITERATIONS = 100

# Bounded number of pre-action nudges before conceding that the model won't act.
MAX_NUDGES = 2

# Compaction protects the most recent N messages from stubbing/eviction.
_COMPACT_KEEP_RECENT = 4
# Tool outputs below this many chars are left alone (not worth stubbing).
_COMPACT_STUB_THRESHOLD = 800
# Proactively compact once the estimated request crosses this fraction of the
# context window.
_COMPACT_HIGH_WATER = 0.80
# Compact down to this fraction of the context window (the low-water target).
_COMPACT_TARGET = 0.60
# Conservative default chars-per-token ratio before calibration from real usage.
_TOKEN_CHARS = 3.0


class GenericBackend(AgentBackend):
    """Backend scaffolding shared by queue-driven, registry-based agent loops."""

    def __init__(
        self,
        working_directory: Path,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        temperature: float = 0.3,
        num_ctx: int = 32768,
    ):
        self._cwd = working_directory
        self._system_prompt = system_prompt
        self._model = model
        self._registry = registry
        self._temperature = temperature
        self._num_ctx = num_ctx
        # Best-known context window for compaction decisions. Defaults to num_ctx.
        self._context_window = num_ctx
        self._session_id = str(uuid.uuid4())[:8]
        self._messages: list[dict[str, Any]] = []
        # Pinned "CURRENT PLAN" message (protected from compaction); Task 2+.
        self._plan_msg: dict[str, Any] | None = None
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        # Accumulated USD cost across every chat-completion request this session.
        self._total_cost_usd = 0.0
        # Chars-per-token ratio for the compaction size estimate; calibrated at
        # runtime from the server's real prompt-token counts (see Task 2+).
        self._chars_per_token = _TOKEN_CHARS
        # Cached char-size of the (constant) tool schemas re-sent on every request.
        self._tools_chars: int | None = None
        # ask_user bridge: interfaces enable replies; validation/piped runs
        # leave this False so ask_user fails open instead of parking forever.
        self.user_reply_enabled = False
        self._reply_queue: asyncio.Queue[str] = asyncio.Queue()
        # Turn generation: bumped at each query(); emitted messages are
        # tagged and receive_messages() filters to the current generation so
        # a stopped run's stranded messages never pollute a later run.
        self._generation = 0
        # Pause gate: cleared by hold() so the loop stops calling the model
        # (a real pause, not just display), set again by release().
        self._run_gate = asyncio.Event()
        self._run_gate.set()

    async def connect(self) -> None:
        """Establish the underlying agent session. No-op; subclasses override."""

    async def disconnect(self) -> None:
        """Clean up: stop the loop and await its task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def query(self, prompt: str) -> None:
        """Start the agentic loop, or deliver a mid-run instruction in-context.

        Called again mid-session on inject/resume-with-instruction, while a
        prior ``_agent_loop()`` task may still be running -- cancel and await it
        first so two loops never concurrently mutate ``self._messages`` or push
        to the same message queue.
        """
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        self._close_dangling_tool_calls()

        # A new query begins a fresh conversational turn: any reply left over
        # from a cancelled ask_user park is stale and must not satisfy a
        # future question.
        while not self._reply_queue.empty():
            self._reply_queue.get_nowait()

        self._generation += 1

        self._prepare_query_messages(prompt)
        self._running = True
        self._task = asyncio.create_task(self._agent_loop())

    def _prepare_query_messages(self, prompt: str) -> None:
        """Build the message list for a :meth:`query` call.

        Empty history (first call) seeds ``[system, task]``; later calls append
        the prompt as a new ``user`` turn, preserving the existing conversation.
        """
        if not self._messages:
            self._messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ]
            return
        self._messages.append({"role": "user", "content": prompt})

    def _close_dangling_tool_calls(self) -> None:
        """Append synthetic tool results for any trailing assistant tool_calls
        that never got one (the loop was cancelled mid-tool). Keeps message
        history valid for strict OpenAI/Anthropic-style servers.
        """
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") == "tool":
                continue
            calls = msg.get("tool_calls") or []
            if not calls:
                return
            answered = {
                m.get("tool_call_id")
                for m in self._messages[i + 1:]
                if m.get("role") == "tool"
            }
            for tc in calls:
                if tc.get("id") not in answered:
                    self._messages.append(
                        self._tool_result_message(tc, "[interrupted by user instruction]")
                    )
            return

    async def receive_messages(self) -> AsyncIterator[AgentMessage]:
        """Yield current-generation messages as the agent loop produces them.

        Messages tagged with an older generation were stranded by a previous
        (stopped/cancelled) turn and are silently discarded.
        """
        while True:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                if msg.metadata.get("generation") != self._generation:
                    continue
                yield msg
                if msg.type == MessageType.RESULT:
                    return
            except TimeoutError:
                # Check if the loop is still running.
                if self._task and self._task.done():
                    # Drain remaining messages.
                    while not self._message_queue.empty():
                        msg = self._message_queue.get_nowait()
                        if msg.metadata.get("generation") != self._generation:
                            continue
                        yield msg
                        if msg.type == MessageType.RESULT:
                            return
                    # If no RESULT was sent, send one now.
                    yield AgentMessage(
                        type=MessageType.RESULT,
                        content=None,
                        metadata={"generation": self._generation, **self._result_metadata()},
                    )
                    return

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def resume(self, session_id: str) -> bool:
        """Resume not supported for the generic loop."""
        return False

    async def _emit(self, msg: AgentMessage) -> None:
        """Push a message to the output queue, tagged with the current turn."""
        msg.metadata.setdefault("generation", self._generation)
        await self._message_queue.put(msg)

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            cwd=self._cwd,
            emit_output=self._emit_output_payload,
            await_user_reply=self._await_user_reply if self.user_reply_enabled else None,
        )

    def _emit_output_payload(self, payload: dict[str, Any]) -> None:
        """Sync, non-blocking emitter passed into `ToolContext.emit_output`.

        Called from within a tool handler (itself invoked from the async
        `_agent_loop`), so it can't `await`; `put_nowait` is safe here because
        `_message_queue` is an unbounded `asyncio.Queue` (see `__init__`).
        """
        msg = AgentMessage(MessageType.OUTPUT, payload)
        msg.metadata.setdefault("generation", self._generation)
        self._message_queue.put_nowait(msg)

    def deliver_reply(self, text: str) -> None:
        """Deliver the user's reply to a parked ask_user call (loop-thread only;
        cross-thread callers go through AgentController.reply's marshaling)."""
        self._reply_queue.put_nowait(text)

    def hold(self) -> None:
        self._run_gate.clear()

    def release(self) -> None:
        self._run_gate.set()

    async def _await_user_reply(self, question: str,
                                choices: list[str] | None = None) -> str:
        """ask_user bridge: surface the question (+ choices), park for the reply."""
        await self._emit(AgentMessage(
            MessageType.AWAITING_INPUT, question,
            metadata={"choices": list(choices or [])},
        ))
        return await self._reply_queue.get()

    def _result_metadata(self) -> dict[str, Any]:
        """Metadata for the terminal RESULT message."""
        return {"cost_usd": self._total_cost_usd}

    async def _chat_completion(self) -> dict[str, Any] | None:
        """Issue a chat-completion request to the underlying LLM. Wire seam."""
        raise NotImplementedError

    def _tool_result_message(self, tool_call: dict[str, Any], result: str) -> dict[str, Any]:
        """Build the ``role: tool`` history message for a completed tool call. Wire seam."""
        raise NotImplementedError

    def _tools_payload(self) -> list[dict[str, Any]]:
        """OpenAI-style tool schemas for the registered tools, sent with every request."""
        return self._registry.openai_payload()

    def _compact_history(self) -> bool:
        """Free context least-destructively; return True if history changed.

        Pass A stubs the oldest bulky ``role: tool`` output in place (preserving the
        conversational thread — the assistant's own reasoning stays). Only when nothing
        remains to stub does it fall back to dropping the oldest exchange
        (``_evict_oldest_exchange``). Protects: system prompt (0), task (1), the pinned
        plan (by identity), and the most recent ``_COMPACT_KEEP_RECENT`` messages.
        """
        n = len(self._messages)
        if n <= 3:
            return False
        recent_start = max(3, n - _COMPACT_KEEP_RECENT)

        # Pass A: stub the oldest bulky, not-yet-stubbed tool output in the middle band.
        for i in range(2, recent_start):
            msg = self._messages[i]
            if msg is self._plan_msg:
                continue
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if content.startswith("[evicted tool output"):
                continue
            if len(content) > _COMPACT_STUB_THRESHOLD:
                logger.debug(
                    "Compaction (stub): stubbing tool output at msg[%d] (%d chars) in place",
                    i, len(content),
                )
                msg["content"] = f"[evicted tool output: {len(content)} chars]"
                return True

        # Pass B: nothing bulky left to stub -> drop the oldest evictable exchange.
        return self._evict_oldest_exchange()

    def _evict_oldest_exchange(self) -> bool:
        """Drop the oldest removable turn to free context, keeping history valid.

        Preserves the system prompt (index 0) and the initial user task (index 1),
        then removes the oldest assistant turn together with its trailing ``tool``
        results — removing an assistant-with-tool_calls without its results (or vice
        versa) would produce a sequence OpenAI-style servers reject. Skips the pinned
        plan message if it sits at index 2 (protected from eviction like compaction
        protects it by identity). Returns False when nothing beyond the preserved head
        remains to evict.
        """
        if len(self._messages) <= 3:
            return False
        i = 2
        if self._messages[i] is self._plan_msg:
            i = 3
            if i >= len(self._messages):
                return False
        removed = 1
        del self._messages[i]
        while i < len(self._messages) and self._messages[i].get("role") == "tool":
            del self._messages[i]
            removed += 1
        logger.debug(
            "Compaction (evict): evicted oldest exchange at msg[%d] (%d messages, "
            "history now %d)", i, removed, len(self._messages),
        )
        return True

    def _tools_schema_chars(self) -> int:
        """Character size of the tool schemas sent on every request (cached).

        The tool list is constant for the lifetime of the backend, so this is
        computed once and reused by :meth:`_estimate_request_tokens`.
        """
        if self._tools_chars is None:
            try:
                self._tools_chars = len(json.dumps(self._tools_payload()))
            except Exception:
                self._tools_chars = 0
        return self._tools_chars

    def _raw_request_chars(self) -> int:
        """Total characters in the next request: message content, tool-call argument
        JSON, and the tool schemas re-sent every request."""
        chars = 0
        for msg in self._messages:
            content = msg.get("content")
            if isinstance(content, str):
                chars += len(content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                chars += len(str(fn.get("arguments", ""))) + len(str(fn.get("name", "")))
        chars += self._tools_schema_chars()
        return chars

    def _estimate_request_tokens(self) -> int:
        """Estimated tokens for the next request (chars / chars-per-token).

        Cheap and tokenizer-free — it only gates *when* to compact, never billing.
        The chars-per-token ratio starts at a conservative default and is calibrated
        from the server's real reported prompt-token counts (see
        :meth:`_calibrate_token_ratio`): tool output (JSON, hex, base64, etc.)
        tokenises far denser than the ~4-chars/token prose default, and an
        uncalibrated estimate makes proactive compaction fire too late.
        """
        return int(self._raw_request_chars() / self._chars_per_token)

    def _calibrate_token_ratio(self, prompt_tokens: int | None) -> None:
        """Adjust the chars-per-token ratio from a real prompt-token count.

        Called right after a completion (success ``usage.prompt_tokens`` or an
        overflow error's reported count) while ``self._messages`` still equals the
        request that produced it. EMA-smoothed so a single outlier can't swing it,
        and clamped to a sane band so a bogus count can't disable compaction.
        """
        if not prompt_tokens or prompt_tokens <= 0:
            return
        chars = self._raw_request_chars()
        if chars <= 0:
            return
        observed = chars / prompt_tokens
        blended = 0.5 * self._chars_per_token + 0.5 * observed
        old = self._chars_per_token
        self._chars_per_token = max(1.5, min(6.0, blended))
        logger.debug(
            "Token-ratio calibrated: %.3f -> %.3f chars/tok (observed %.3f from "
            "%d chars / %d prompt_tokens)",
            old, self._chars_per_token, observed, chars, prompt_tokens,
        )

    def _accumulate_cost(self, usage: dict[str, Any] | None) -> None:
        """Add an OpenAI-style ``usage.cost`` (USD) to the running session total.

        OpenRouter-style servers return a real ``cost`` when cost accounting is
        requested; local and token-only servers omit it, so this contributes
        nothing. Best-effort — a missing, ``None``, non-numeric, or ``bool`` value
        is ignored.
        """
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self._total_cost_usd += float(cost)

    def _compact_to_budget(self, target_tokens: int) -> bool:
        """Compact history until the estimate is at/below ``target_tokens``.

        Frees down to the target in ONE pass — so the caller re-prefills the model
        once, instead of once per freed message (avoiding compaction thrash where
        each single-message eviction triggers a full re-prefill). Returns True if
        the history changed. Bounded by the message count so a mis-estimate can
        never spin.
        """
        changed = False
        for _ in range(len(self._messages) + 1):
            if self._estimate_request_tokens() <= target_tokens:
                break
            if not self._compact_history():
                break
            changed = True
        return changed

    def _maybe_precompact(self) -> None:
        """Proactively compact before sending if the request would ride the ceiling.

        When the estimated request size crosses the high-water mark, trim to the
        low-water target so the request never reaches the server's context limit in
        the first place (avoiding the send -> 400 -> compact -> re-send round trip on
        strict servers, and silent context truncation on servers that drop tokens
        instead of erroring). A no-op while there is headroom.
        """
        window = self._context_window
        if window <= 0:
            return
        est = self._estimate_request_tokens()
        high_water = int(_COMPACT_HIGH_WATER * window)
        logger.debug(
            "Precompact check: est=%d tokens vs high-water=%d (window=%d, %.0f%% full, "
            "%.2f chars/tok)",
            est, high_water, window, 100.0 * est / window, self._chars_per_token,
        )
        if est > high_water:
            target = int(_COMPACT_TARGET * window)
            if self._compact_to_budget(target):
                logger.debug(
                    "Proactively compacted history to ~%d-token target (window=%d, "
                    "now %d messages)", target, window, len(self._messages),
                )

    async def _agent_loop(self) -> None:
        """Core agentic loop: query LLM -> execute tools -> feed back -> repeat."""
        iteration = 0
        nudges = 0
        tool_executions = 0
        emitted = False
        deliverable_nudged = False
        wants_deliverable = self._registry.get("emit_output") is not None
        try:
            while self._running and iteration < MAX_ITERATIONS:
                await self._run_gate.wait()
                iteration += 1

                # Proactively trim history before it reaches the context ceiling, so a
                # long session doesn't thrash on per-round re-prefills near the limit.
                self._maybe_precompact()

                response = await self._chat_completion()
                if response is None:
                    await self._emit(
                        AgentMessage(
                            type=MessageType.ERROR,
                            content="Failed to get response from model",
                        )
                    )
                    self._running = False
                    await self._emit(
                        AgentMessage(
                            type=MessageType.RESULT,
                            content=None,
                            metadata={"outcome": "error", **self._result_metadata()},
                        )
                    )
                    return

                message = response["message"]

                text_content = message.get("content", "")
                if text_content and text_content.strip():
                    await self._emit(
                        AgentMessage(type=MessageType.TEXT, content=text_content)
                    )

                tool_calls = message.get("tool_calls", []) or []
                if not tool_calls:
                    # No tool calls normally signals completion. But a model may emit a
                    # prose plan without acting; treating that as "done" would exit the
                    # session having accomplished nothing. If no tool has run yet,
                    # nudge the model to act and retry a bounded number of times before
                    # conceding.
                    if tool_executions == 0 and nudges < MAX_NUDGES:
                        nudges += 1
                        self._messages.append(message)
                        self._messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You have not called any tools yet. Call one of the "
                                    "available tools now to make progress, or provide "
                                    "your final answer if the task is complete."
                                ),
                            }
                        )
                        continue
                    if wants_deliverable and not emitted and not deliverable_nudged:
                        deliverable_nudged = True
                        self._messages.append(message)
                        self._messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You are ending without producing the deliverable. "
                                    "Call emit_output now with your best structured "
                                    "deliverable, using reasonable assumptions for any "
                                    "fields you could not fully verify."
                                ),
                            }
                        )
                        continue
                    break
                self._messages.append(message)
                for tool_call in tool_calls:
                    tool_executions += 1
                    func = tool_call.get("function", {})
                    name = func.get("name", "")
                    if name == "emit_output":
                        emitted = True
                    raw_args = func.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    else:
                        args = raw_args or {}
                    await self._emit(
                        AgentMessage(MessageType.TOOL_START, "", tool_name=name, tool_args=args)
                    )
                    result = await self._registry.dispatch(name, args, self._tool_context())
                    await self._emit(
                        AgentMessage(MessageType.TOOL_RESULT, result, tool_name=name)
                    )
                    self._messages.append(self._tool_result_message(tool_call, result))

            # Normal exit (tool-less turn, or the MAX_ITERATIONS ceiling): the loop
            # is genuinely done, so the flag shouldn't keep claiming otherwise.
            self._running = False
            await self._emit(
                AgentMessage(
                    type=MessageType.RESULT,
                    content=None,
                    metadata={"outcome": "ok", **self._result_metadata()},
                )
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self._emit(
                AgentMessage(type=MessageType.ERROR, content=f"Agent error: {e}")
            )
            await self._emit(
                AgentMessage(
                    type=MessageType.RESULT,
                    content=None,
                    metadata={"outcome": "error", **self._result_metadata()},
                )
            )
