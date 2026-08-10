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

"""Production backend factories for the validation harness.

The one place that builds `validate()`'s two backends for real providers.
`make_production_judge_backend` upholds the judge contract documented in
`janus.core.validation.harness`: `JUDGE_SYSTEM_PROMPT` as the system prompt and
`judge_registry(rubric)` as the tool registry. Used by the `janus validate` CLI
and by the factory persona's `validate_persona` tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from janus.core.backends.select import build_backend, build_backend_for_persona
from janus.core.config import load_config
from janus.core.validation.judge import JUDGE_SYSTEM_PROMPT, judge_registry

if TYPE_CHECKING:
    from janus.core.backend import AgentBackend
    from janus.core.persona import Persona
    from janus.core.validation.rubric import Rubric


def make_production_agent_backend(persona: Persona, ws: Path) -> AgentBackend:
    """Build the persona-under-test's backend for one validation workspace."""
    return build_backend_for_persona(
        load_config(persona=persona.name, working_directory=ws), persona
    )


def make_production_judge_backend(rubric: Rubric, jws: Path) -> AgentBackend:
    """Build a correctly-framed judge backend (JUDGE_SYSTEM_PROMPT + judge_registry)."""
    return build_backend(
        load_config(working_directory=jws), JUDGE_SYSTEM_PROMPT, judge_registry(rubric)
    )
