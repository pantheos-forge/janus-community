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

"""ds4-server (DwarfStar) backend for local DeepSeek-V4 inference.

DwarfStar's ``ds4-server`` exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint with native tool-calling. The whole OpenAI wire protocol is shared
with :class:`OpenAICompatBackend`; this backend only pins the ds4 defaults
(local, no auth) and adds ``reasoning_effort`` to each request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from janus.core.backends.openai_compat import OpenAICompatBackend
from janus.core.tools.registry import ToolRegistry

DEFAULT_DS4_URL = "http://127.0.0.1:8000"
# Both aliases serve whatever GGUF ds4-server loaded with -m; the name is cosmetic.
DEFAULT_DS4_MODEL = "deepseek-v4-flash"


class Ds4Backend(OpenAICompatBackend):
    """Backend that talks to ds4-server's OpenAI-compatible API."""

    DEFAULT_DS4_URL = DEFAULT_DS4_URL
    DEFAULT_DS4_MODEL = DEFAULT_DS4_MODEL

    def __init__(
        self,
        working_directory: Path,
        system_prompt: str,
        model: str,
        registry: ToolRegistry,
        *,
        base_url: str | None = None,
        reasoning_effort: str = "low",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        super().__init__(
            working_directory=working_directory,
            system_prompt=system_prompt,
            model=model,
            registry=registry,
            base_url=base_url or self.DEFAULT_DS4_URL,
            api_key=None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # DeepSeek-V4 is a reasoning model; left unconstrained it spends turns emitting
        # <think>/DSML prose instead of acting. ds4-server advertises reasoning_effort
        # on /v1/models; "low" keeps native tool-calling prompt and cuts the
        # reasoning-token leakage/looping.
        self._reasoning_effort = reasoning_effort

    def _extra_payload(self) -> dict[str, Any]:
        """ds4-server accepts ``reasoning_effort`` to quiet DeepSeek's <think>/DSML."""
        return {"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}

    def _connect_error_hint(self) -> str:
        return (
            "Is it running? Start with: ds4-server -m <model.gguf> --cuda --ctx 100000"
        )
