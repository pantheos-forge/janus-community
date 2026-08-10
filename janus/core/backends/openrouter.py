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

"""OpenRouter backend -- routes to any OpenRouter-hosted model via one API key.

OpenRouter (https://openrouter.ai/api/v1) speaks the same OpenAI-compatible
``/v1/chat/completions`` protocol as :class:`OpenAICompatBackend`, so this
backend only adds bearer-token auth (and optional attribution headers) and
pins the OpenRouter defaults. The ``model`` is a routed slug such as
``anthropic/claude-3.5-sonnet``, ``openai/gpt-4o``, or ``deepseek/deepseek-chat``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from janus.core.backends.openai_compat import OpenAICompatBackend
from janus.core.tools.registry import ToolRegistry

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(OpenAICompatBackend):
    """Backend that routes through OpenRouter's OpenAI-compatible API."""

    DEFAULT_OPENROUTER_URL = DEFAULT_OPENROUTER_URL

    def __init__(
        self,
        working_directory: Path,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        api_key: str,
        base_url: str | None = None,
        referer: str | None = None,
        title: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY "
                "(or pass api_key). Get one at https://openrouter.ai/keys."
            )
        super().__init__(
            working_directory=working_directory,
            system_prompt=system_prompt,
            model=model,
            registry=registry,
            base_url=base_url or self.DEFAULT_OPENROUTER_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._referer = referer
        self._title = title

    def _request_headers(self) -> dict[str, str]:
        """Bearer auth + optional OpenRouter attribution headers."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        return headers

    def _connect_error_hint(self) -> str:
        return "Check OPENROUTER_API_KEY and network connectivity to openrouter.ai."

    def _extra_payload(self) -> dict[str, Any]:
        # No reasoning_effort: not universally supported across routed models.
        # Request usage accounting so the response's `usage` object carries `cost`
        # (USD) -- otherwise OpenRouter omits it and session cost reads $0.00.
        return {"usage": {"include": True}}
