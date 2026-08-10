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

"""Provider auto-selection: build a backend from ``JanusConfig`` by precedence.

Janus Core's ``AgentController`` is injection-only -- it never picks a
provider itself. ``build_backend()`` is the standalone helper callers use to
construct the concrete backend they inject, mirroring the upstream
proprietary agent's precedence: ds4-server first, then OpenRouter,
then local Ollama, then the Claude-default tier -- the Claude Agent SDK
backend (opt-in via ``use_claude_agent_sdk``) or else the native
``AnthropicAPIBackend`` built from ``anthropic_api_key`` / ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from janus.core.backend import AgentBackend
from janus.core.backends.anthropic_api import AnthropicAPIBackend
from janus.core.backends.claude_sdk import ClaudeSDKBackend
from janus.core.backends.ds4 import Ds4Backend
from janus.core.backends.ollama import OllamaBackend
from janus.core.backends.openrouter import OpenRouterBackend
from janus.core.config import JanusConfig
from janus.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from janus.core.persona import Persona


def build_backend(
    config: JanusConfig, system_prompt: str, registry: ToolRegistry
) -> AgentBackend:
    """Construct the backend indicated by ``config``, by precedence.

    Precedence (first match wins): ``ds4_url`` > ``openrouter_model`` >
    ``local_model`` > ``use_claude_agent_sdk`` > native Anthropic (from
    ``anthropic_api_key`` or the ``ANTHROPIC_API_KEY`` env var). Raises
    ``NotImplementedError`` if none are configured and no Anthropic API key
    is available.
    """
    if config.ds4_url:
        return Ds4Backend(
            working_directory=config.working_directory,
            system_prompt=system_prompt,
            model=config.ds4_model,
            registry=registry,
            base_url=config.ds4_url,
            max_tokens=config.ds4_max_tokens,
        )

    if config.openrouter_model:
        return OpenRouterBackend(
            working_directory=config.working_directory,
            system_prompt=system_prompt,
            model=config.openrouter_model,
            registry=registry,
            api_key=config.openrouter_api_key or "",
            base_url=config.openrouter_url,
            max_tokens=config.openrouter_max_tokens,
        )

    if config.local_model:
        return OllamaBackend(
            working_directory=config.working_directory,
            system_prompt=system_prompt,
            model=config.local_model,
            registry=registry,
            base_url=config.ollama_base_url,
            temperature=config.ollama_temperature,
            num_ctx=config.ollama_num_ctx,
        )

    if config.use_claude_agent_sdk:
        return ClaudeSDKBackend(
            working_directory=config.working_directory,
            system_prompt=system_prompt,
            model=config.llm_model,
            registry=registry,
        )

    if key := (config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")):
        return AnthropicAPIBackend(
            working_directory=config.working_directory,
            system_prompt=system_prompt,
            model=config.llm_model,
            registry=registry,
            api_key=key,
        )

    raise NotImplementedError(
        "No provider configured. Set use_claude_agent_sdk=True, or "
        "ANTHROPIC_API_KEY (or anthropic_api_key), or ds4_url / openrouter_model / "
        "local_model."
    )


def build_backend_for_persona(
    config: JanusConfig, persona: Persona
) -> AgentBackend:
    """Build a backend for a persona by wiring its system prompt and registry.

    Convenience wrapper around ``build_backend()`` that extracts the system
    prompt and tool registry from a ``Persona`` and passes them through.
    """
    return build_backend(config, persona.system_prompt, persona.registry)
