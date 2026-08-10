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

"""Native Ollama ``/api/chat`` wire backend.

Ported from the upstream proprietary agent's ``OllamaBackend``, rebased onto
Janus's
``GenericBackend`` (queue-driven loop, registry-driven tools). Ollama's
native tool-calling response is already ``{"message": {"content",
"tool_calls": [...]}}`` -- exactly the shape the shared loop expects -- so
``_chat_completion`` returns it directly after a light calibration pass.

Overrides only the points where the wire protocol differs from the abstract
seams in :class:`GenericBackend`:

* :meth:`connect`              -- best-effort health check via ``GET {base_url}/api/tags``
* :meth:`_chat_completion`     -- Ollama ``/api/chat`` request/response shape
* :meth:`_tool_result_message` -- no ``tool_call_id`` (Ollama matches positionally)
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import httpx

from janus.core.backends.generic import GenericBackend
from janus.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Generous timeout headroom for slow local inference servers.
_REQUEST_TIMEOUT_S = 180.0


class OllamaBackend(GenericBackend):
    """Backend for a native Ollama ``/api/chat`` server."""

    def __init__(
        self,
        working_directory: Path,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        num_ctx: int = 32768,
    ):
        super().__init__(
            working_directory=working_directory,
            system_prompt=system_prompt,
            model=model,
            registry=registry,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Best-effort reachability check against Ollama's tag listing.

        Does not overwrite ``self._client`` if one is already set -- tests
        (and callers that manage their own transport) may inject a client
        before driving the loop. The health check itself is best-effort:
        Ollama being briefly unreachable at connect time shouldn't crash the
        backend, since the actual chat request will surface any real failure.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_S))
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            logger.debug("Connected to Ollama at %s", self._base_url)
        except Exception as e:
            logger.warning("Ollama health check failed (continuing anyway): %s", e)

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
        """Send a chat-completion request to Ollama's native ``/api/chat``.

        Ollama already returns the ``{"message": {...}}`` shape the shared
        loop expects, so the response is returned as-is after calibrating the
        compaction estimator from the real ``prompt_eval_count``. Returns
        ``None`` on timeout, HTTP error, or any other failure -- the loop
        treats that as end-of-turn.
        """
        if not self._client:
            return None

        payload = {
            "model": self._model,
            "messages": self._messages,
            "tools": self._tools_payload(),
            "stream": False,
            # Suppress extended thinking for reasoning models -- thinking tokens
            # delay tool calls and can produce advisory prose instead of action.
            "think": False,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
            },
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_S),
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            self._calibrate_token_ratio(data.get("prompt_eval_count"))
            self._accumulate_cost(data.get("usage"))
            return data
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                "Ollama HTTP error: %s — %s", e.response.status_code, e.response.text[:500]
            )
            return None
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
            return None

    def _tool_result_message(self, tool_call: dict[str, Any], result: str) -> dict[str, Any]:
        """Ollama matches tool results to the preceding assistant turn
        positionally and does not require a ``tool_call_id``."""
        return {"role": "tool", "content": result}
