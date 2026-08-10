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

"""Configuration management for Janus using Pydantic."""

import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JanusConfig(BaseSettings):
    """Main configuration for Janus."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # LLM Configuration
    llm_model: str = Field(
        default="claude-sonnet-5", description="Claude model to use for the agent"
    )

    working_directory: Path = Field(
        default_factory=lambda: Path.cwd() / "workspace",
        description="Working directory for agent operations",
    )

    fleet_dir: Path = Field(
        default_factory=lambda: Path.home() / "janus-agents",
        validation_alias="JANUS_FLEET_DIR",
        description="Home directory for the managed fleet of generated agents.",
    )

    fleet_max_concurrent: int = Field(
        default=3,
        validation_alias="JANUS_FLEET_MAX_CONCURRENT",
        description="Max number of concurrently RUNNING dashboard sessions; excess queue.",
    )

    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key",
    )

    # Subject / persona configuration
    subject: str | None = Field(
        default=None,
        description="Subject for the agent to work on",
    )

    custom_instruction: str | None = Field(
        default=None, description="Optional custom instructions for the agent"
    )

    persona: str | None = Field(
        default=None,
        description="Persona/system prompt type to use",
    )

    # Permission Mode
    permission_mode: Literal["ask", "bypassPermissions"] = Field(
        default="bypassPermissions", description="Permission mode for Claude Code SDK"
    )

    # Local LLM Configuration
    local_model: str | None = Field(
        default=None,
        description=(
            "Local LLM model name via Ollama (e.g., "
            "'huihui_ai/qwen3-coder-next-abliterated:latest')"
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_temperature: float = Field(
        default=0.3,
        description="Sampling temperature for local Ollama models. Low values give "
        "deterministic, reliable native tool-calling; the raw model default (e.g. qwen3's "
        "0.6) can cause reasoning models to return a prose-only first turn and exit early.",
    )
    ollama_num_ctx: int = Field(
        default=32768,
        description="Context window (num_ctx) for local Ollama models. 32K fits the ~16K "
        "system prompt plus session history; 16K is the practical minimum.",
    )

    # ds4-server (DwarfStar) Configuration -- OpenAI-compatible local DeepSeek-V4 backend.
    # Setting ds4_url opts into the Ds4Backend (takes precedence over local_model).
    ds4_url: str | None = Field(
        default=None,
        description="ds4-server base URL (e.g. 'http://127.0.0.1:8000'). When set, "
        "Janus routes the task through the ds4-server OpenAI-compatible API.",
    )
    ds4_model: str = Field(
        default="deepseek-v4-flash",
        description="Model alias sent to ds4-server. Both deepseek-v4-flash and "
        "deepseek-v4-pro serve whatever GGUF the server loaded; the name is cosmetic.",
    )
    ds4_max_tokens: int = Field(
        default=8192,
        description="Max output tokens per ds4-server chat completion. The context "
        "window itself is fixed server-side via ds4-server's -c/--ctx flag.",
    )

    # OpenRouter Configuration -- OpenAI-compatible cloud aggregator.
    # Setting openrouter_model opts into the OpenRouterBackend (precedence:
    # ds4_url > openrouter_model > local_model > ClaudeCode default).
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias="OPENROUTER_API_KEY",
        description="OpenRouter API key (sk-or-v1-...). Required when openrouter_model is set.",
    )
    openrouter_model: str | None = Field(
        default=None,
        description="OpenRouter routed model slug (e.g. 'anthropic/claude-3.5-sonnet', "
        "'openai/gpt-4o', 'deepseek/deepseek-chat'). When set, routes the task "
        "through OpenRouter's OpenAI-compatible API.",
    )
    openrouter_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    openrouter_max_tokens: int = Field(
        default=8192,
        description="Max output tokens per OpenRouter chat completion.",
    )

    use_claude_agent_sdk: bool = Field(
        default=False,
        description="Use the Claude Agent SDK backend (OAuth/Claude-Code) instead of "
        "the direct Anthropic API for the default Claude path.",
    )

    @field_validator("use_claude_agent_sdk", mode="before")
    @classmethod
    def _empty_bool_is_default(cls, v: object) -> object:
        """Coerce an empty-string boolean to the field default.

        A containerized agent can receive ``USE_CLAUDE_AGENT_SDK=""`` when an
        unset compose var is expanded to an empty string; pydantic's bool parser
        rejects ``""`` and crashes load_config. Treat empty/whitespace as "unset".
        """
        if isinstance(v, str) and not v.strip():
            return False
        return v

    # Session ID (pre-generated so per-session directories are unique from the start)
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique session identifier",
    )


def load_config(**overrides: object) -> JanusConfig:
    """
    Load configuration from environment with optional overrides.

    Args:
        **overrides: Keyword arguments to override config values

    Returns:
        JanusConfig instance

    Example:
        >>> config = load_config(persona="market_research")
    """
    return JanusConfig(**overrides)  # type: ignore[arg-type]
