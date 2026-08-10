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

"""OpenAI-compatible chat-completions wire backend.

Ported from the upstream proprietary agent's ``OpenAICompatBackend``, rebased
onto Janus's
``GenericBackend`` (queue-driven loop, registry-driven tools) instead of the
Ollama-specific base it originally extended. Talks to any server exposing an
OpenAI-compatible ``/v1/chat/completions`` endpoint with native tool-calling
(local Ollama's ``/v1``, OpenRouter, vLLM, etc.) -- the first real, runnable
provider for the generic loop.

Overrides only the points where the wire protocol differs from the abstract
seams in :class:`GenericBackend`:

* :meth:`connect`              -- health check via ``GET {base_url}/models``
* :meth:`_chat_completion`     -- OpenAI request/response shape (+ context eviction)
* :meth:`_tool_result_message` -- attach ``tool_call_id`` (OpenAI requires it)

The small hooks that differ between concrete OpenAI-compatible servers --
``_request_headers`` (auth), ``_extra_payload`` (per-server params) and
``_is_context_overflow`` (how the server signals a context-window overflow) --
are kept as overridable subclass hooks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import re
from pathlib import Path
from typing import Any

import httpx

from janus.core.backends.generic import _COMPACT_TARGET, GenericBackend
from janus.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Max oldest-exchange evictions to attempt when a single request overflows the
# server's context window before giving up on that turn.
MAX_CONTEXT_EVICTIONS = 20

# Cap for the one-shot bump when a reasoning model truncates (finish_reason=length).
MAX_TOKENS_CEILING = 16384

# Transient upstream failures worth retrying rather than ending the session: a
# provider rate-limit (429) and gateway hiccups (transient 5xx). Without this, a 429
# returned None from _post_completion, which the agentic loop reads as a clean
# end-of-session -- a false success mid-run. Retry with backoff instead. A 400
# context-overflow is NOT here: it is handled separately via _is_context_overflow.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 4
_MAX_RETRY_DELAY_S = 60.0

# Generous timeout headroom for slow local inference servers.
_REQUEST_TIMEOUT_S = 180.0


class OpenAICompatBackend(GenericBackend):
    """Backend for any OpenAI-compatible ``/v1/chat/completions`` server."""

    def __init__(
        self,
        working_directory: Path,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        super().__init__(
            working_directory=working_directory,
            system_prompt=system_prompt,
            model=model,
            registry=registry,
            temperature=temperature,
        )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    # --- Subclass hooks -----------------------------------------------------

    def _request_headers(self) -> dict[str, str]:
        """Extra HTTP headers (e.g. auth). Default: bearer token if configured."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _extra_payload(self) -> dict[str, Any]:
        """Extra chat-completion payload fields (e.g. ``reasoning_effort``)."""
        return {}

    def _connect_error_hint(self) -> str:
        """Human-readable hint appended to a connection-failure error."""
        return "Is the server running and reachable?"

    def _is_context_overflow(self, status_code: int, body: str) -> bool:
        """Whether an HTTP error indicates a context-window overflow."""
        low = body.lower()
        return status_code == 400 and any(
            s in low for s in ("context_length_exceeded", "context length", "maximum context")
        )

    def _extract_prompt_tokens_from_error(self, body: str) -> int | None:
        """Real prompt-token count from an overflow error body, if the server reports
        one. Best-effort; None when absent."""
        match = re.search(r'"?n_prompt_tokens"?\s*:\s*(\d+)', body)
        return int(match.group(1)) if match else None

    # --- Connection -----------------------------------------------------------

    def _parse_context_window(self, models: Any) -> int | None:
        """Best-effort context length for the served model from a ``/models`` body.

        Prefers the entry whose ``id`` matches the configured model (each entry may
        advertise ``context_length`` directly or under ``top_provider``); falls back
        to the first entry that advertises one. Returns None when nothing usable is
        present, leaving the conservative default in place.
        """

        def ctx_of(entry: dict[str, Any]) -> int | None:
            for key in ("context_length", "max_context_length"):
                val = entry.get(key)
                if isinstance(val, int) and val > 0:
                    return val
            provider = entry.get("top_provider") or {}
            val = provider.get("context_length") if isinstance(provider, dict) else None
            return val if isinstance(val, int) and val > 0 else None

        try:
            entries = [e for e in (models.get("data") or []) if isinstance(e, dict)]
        except (AttributeError, TypeError):
            return None
        for entry in entries:
            if entry.get("id") == self._model:
                window = ctx_of(entry)
                if window:
                    return window
        for entry in entries:
            window = ctx_of(entry)
            if window:
                return window
        return None

    async def connect(self) -> None:
        """Verify the server is reachable via its model listing endpoint.

        Does not overwrite ``self._client`` if one is already set -- tests (and
        callers that manage their own transport) may inject a client before
        driving the loop.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_S))
        try:
            resp = await self._client.get(
                f"{self._base_url}/models", headers=self._request_headers()
            )
            resp.raise_for_status()
            window = self._parse_context_window(resp.json())
            if window:
                self._context_window = window
            logger.debug(
                "Connected to OpenAI-compatible server at %s (context window=%d)",
                self._base_url, self._context_window,
            )
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to {self._base_url}. {self._connect_error_hint()}"
            ) from e

    async def disconnect(self) -> None:
        """Stop the loop (base class) then close the httpx client.

        ``GenericBackend.disconnect`` only cancels/awaits the loop task; it
        never closes ``self._client``. An unclosed ``httpx.AsyncClient`` keeps
        its connection pool (and the sockets/anyio resources behind it) alive,
        which lingers after ``asyncio.run`` returns and holds the process open
        on a headless run. Capture-and-null before awaiting ``aclose`` so a
        concurrent double-disconnect is a safe no-op, mirroring
        ``ClaudeSDKBackend.disconnect``.
        """
        await super().disconnect()
        if self._client is not None:
            client, self._client = self._client, None
            with contextlib.suppress(Exception):
                await client.aclose()

    # --- Shared OpenAI wire protocol --------------------------------------------

    async def _chat_completion(self) -> dict[str, Any] | None:
        """Send an OpenAI chat-completion request.

        Normalises to the ``{"message": {...}}`` shape the shared loop expects.
        Recovers from context overflow by compacting history and retrying; and,
        when a model truncates its output (``finish_reason == "length"``) without
        emitting a tool call, retries the turn once at a higher token cap.
        """
        if not self._client:
            return None

        headers = self._request_headers()
        overflow_retries = 0
        for _ in range(MAX_CONTEXT_EVICTIONS + 1):
            outcome = await self._post_completion(headers, self._max_tokens)
            if outcome is None:
                return None
            if isinstance(outcome, str):
                # outcome == "overflow": the server rejected the request as too large.
                # Compact down to the low-water target in ONE pass and re-prefill once,
                # rather than freeing a single message per full re-prefill. Each repeat
                # overflow means the local estimate was optimistic, so tighten the target
                # by 10% each round until the request fits.
                overflow_retries += 1
                target = int(
                    self._context_window * _COMPACT_TARGET * (0.9 ** (overflow_retries - 1))
                )
                # The server rejected the request, so we ARE over its real limit even
                # when the local estimate disagrees. Compact to target, but guarantee
                # forward progress with at least one eviction so a stale estimate can't
                # wedge us.
                compacted = self._compact_to_budget(max(target, 1))
                if not compacted:
                    compacted = self._compact_history()
                if compacted:
                    logger.warning(
                        "Context window exceeded; compacted to ~%d-token target and "
                        "retrying (history now %d messages)", target, len(self._messages),
                    )
                    continue
                logger.error("Still over context after compaction")
                return None

            message, finish_reason = outcome
            if finish_reason == "length" and not message.get("tool_calls"):
                bumped = min(self._max_tokens * 2, MAX_TOKENS_CEILING)
                if bumped > self._max_tokens:
                    logger.info(
                        "Output truncated (finish_reason=length, no tool call); "
                        "retrying once at max_tokens=%d", bumped,
                    )
                    retried = await self._post_completion(headers, bumped)
                    if isinstance(retried, tuple):
                        return {"message": retried[0]}
            return {"message": message}
        logger.error("Still over context after %d compactions", MAX_CONTEXT_EVICTIONS)
        return None

    async def _post_completion(
        self, headers: dict[str, str], max_tokens: int
    ) -> tuple[dict[str, Any], str | None] | str | None:
        """One chat-completion POST.

        Returns ``(message, finish_reason)`` on success, the string ``"overflow"``
        when the server signalled a context-window overflow, or ``None`` on any
        other failure (timeout / HTTP error / no choices).
        """
        if not self._client:
            return None
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "tools": self._tools_payload(),
            "tool_choice": "auto",
            "stream": False,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            **self._extra_payload(),
        }
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(_REQUEST_TIMEOUT_S),
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    logger.error("Server returned no choices: %s", str(data)[:500])
                    return None
                # Calibrate the compaction estimator from the server's real prompt-token
                # count while self._messages still equals the request that produced it.
                usage = data.get("usage")
                if isinstance(usage, dict):
                    self._calibrate_token_ratio(usage.get("prompt_tokens"))
                    # Accumulate real USD cost (OpenRouter reports usage.cost; local/
                    # token-only servers omit it -> contributes 0). Surfaced on RESULT.
                    self._accumulate_cost(usage)
                choice = choices[0]
                return choice.get("message", {}), choice.get("finish_reason")
            except httpx.HTTPStatusError as e:
                body = e.response.text
                if self._is_context_overflow(e.response.status_code, body):
                    # The rejected request's real token count is the most accurate
                    # calibration point -- self._messages is still the oversized
                    # request, not yet compacted.
                    self._calibrate_token_ratio(self._extract_prompt_tokens_from_error(body))
                    return "overflow"
                if (
                    e.response.status_code in _RETRYABLE_STATUS
                    and attempt < _MAX_TRANSIENT_RETRIES
                ):
                    delay = self._retry_after(e.response) or self._backoff_delay(attempt)
                    logger.warning(
                        "Transient upstream %s; backing off %.1fs then retrying "
                        "(attempt %d/%d)",
                        e.response.status_code, delay, attempt + 1, _MAX_TRANSIENT_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Chat-completion HTTP error: %s — %s", e.response.status_code, body[:500]
                )
                return None
            except httpx.TimeoutException:
                if attempt < _MAX_TRANSIENT_RETRIES:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Chat-completion request timed out; retrying in %.1fs "
                        "(attempt %d/%d)",
                        delay, attempt + 1, _MAX_TRANSIENT_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Chat-completion request timed out")
                return None
            except Exception as e:
                logger.error("Chat-completion request failed: %s", e)
                return None
        return None

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        """Honor an upstream ``Retry-After`` header (delta-seconds), capped."""
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return min(max(float(raw), 0.0), _MAX_RETRY_DELAY_S)
        except ValueError:
            return None

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff (~1, 2, 4, 8s, capped) with +/-20% jitter."""
        base = min(2.0**attempt, 8.0)
        return round(base * (0.8 + 0.4 * random.random()), 2)

    def _tool_result_message(self, tool_call: dict[str, Any], result: str) -> dict[str, Any]:
        """OpenAI requires each tool result to reference its ``tool_call_id``."""
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "content": result,
        }
