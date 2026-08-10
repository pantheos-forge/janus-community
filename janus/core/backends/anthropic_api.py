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

"""Native Anthropic ``/v1/messages`` wire backend.

Talks to Anthropic's Messages API directly over raw httpx (no ``anthropic``
SDK). Uses the pure translation helpers in
:mod:`janus.core.backends.anthropic_translate` to convert the shared loop's
OpenAI-shaped ``_messages``/tool schemas to and from the Anthropic wire
shape.

Overrides only the points where the wire protocol differs from the abstract
seams in :class:`GenericBackend`:

* :meth:`connect`              -- create the httpx client; no health check
* :meth:`_chat_completion`     -- Anthropic ``/v1/messages`` request/response shape
* :meth:`_tool_result_message` -- OpenAI ``role: tool`` shape (translated on the way out)

Critical wire constraint: current Claude models (Opus 4.8 / Sonnet 5) return
a 400 if the request body includes ``temperature``, ``top_p``, ``top_k``, or
``thinking`` alongside tool use in this configuration, so none of those
fields are ever sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import httpx

from janus.core.backends.anthropic_translate import (
    anthropic_response_to_openai_message,
    openai_messages_to_anthropic,
    openai_tools_to_anthropic,
)
from janus.core.backends.generic import GenericBackend
from janus.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Generous timeout headroom for slow inference.
_REQUEST_TIMEOUT_S = 180.0

# Transient upstream failures worth retrying rather than ending the session: a
# rate-limit (429) and gateway hiccups (transient 5xx). Without this, a 429
# returned None from _chat_completion, which the agentic loop reads as a clean
# end-of-session -- a false success mid-run.
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 3


class AnthropicAPIBackend(GenericBackend):
    """Backend for Anthropic's native ``/v1/messages`` API."""

    DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        working_directory: Path,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int = 8192,
        context_window: int = 200_000,
    ):
        super().__init__(
            working_directory=working_directory,
            system_prompt=system_prompt,
            model=model,
            registry=registry,
        )
        self._base_url = (base_url or self.DEFAULT_ANTHROPIC_URL).rstrip("/")
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Create the underlying httpx client.

        Does not overwrite ``self._client`` if one is already set -- tests
        (and callers that manage their own transport) may inject a client
        before driving the loop. No health check is performed.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_S))

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

    async def _chat_completion(self) -> dict[str, Any] | None:
        """Send a request to Anthropic's ``/v1/messages``.

        Translates the loop's OpenAI-shaped ``_messages``/tools into the
        Anthropic wire shape, POSTs, and translates the response back into
        the ``{"message": <openai assistant message>}`` shape the shared
        loop expects. Returns ``None`` on a non-2xx response (after
        exhausting transient retries), a timeout, or any other failure --
        the loop treats that as end-of-turn.

        Note: ``cost_usd`` stays 0 for this backend -- Anthropic's ``usage``
        reports ``input_tokens``/``output_tokens``, not a dollar ``cost``
        field, so token-based pricing is out of scope here.
        """
        if not self._client:
            return None

        system, messages = openai_messages_to_anthropic(self._messages)
        tools = openai_tools_to_anthropic(self._tools_payload())

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "tools": tools,
        }
        if system is not None:
            body["system"] = system

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                resp = await self._client.post(
                    f"{self._base_url}/v1/messages",
                    json=body,
                    headers=headers,
                    timeout=httpx.Timeout(_REQUEST_TIMEOUT_S),
                )
                if (
                    resp.status_code in _TRANSIENT_STATUSES
                    and attempt < _MAX_TRANSIENT_RETRIES
                ):
                    delay = self._retry_after(resp) or min(2**attempt, 8)
                    logger.warning(
                        "Anthropic transient error %s; backing off %.1fs then retrying "
                        "(attempt %d/%d)",
                        resp.status_code, delay, attempt + 1, _MAX_TRANSIENT_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                self._calibrate_token_ratio(data.get("usage", {}).get("input_tokens"))
                self._accumulate_cost(data.get("usage"))
                return {"message": anthropic_response_to_openai_message(data)}
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Anthropic HTTP error: %s — %s", e.response.status_code, e.response.text[:500]
                )
                return None
            except httpx.TimeoutException:
                logger.error("Anthropic request timed out")
                return None
            except Exception as e:
                logger.error("Anthropic request failed: %s", e)
                return None
        return None

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        """Honor an upstream ``Retry-After`` header (delta-seconds), if parseable."""
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _tool_result_message(self, tool_call: dict[str, Any], result: str) -> dict[str, Any]:
        """OpenAI ``role: tool`` shape; the translator converts it to an
        Anthropic ``tool_result`` content block."""
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "content": result,
        }
