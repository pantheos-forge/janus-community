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

"""Smoke-run validation check.

Runs a persona's agent on one task through the real ``AgentController`` and
checks it is mechanically sound: the run boots, executes, and terminates
``completed``; and, if the persona declares an ``output_schema``, that it
emitted a deliverable (the persona's ``output_filename`` in the workspace —
``output.json`` by default — written by ``emit_output``) that validates
against that schema. Both checks fail
gracefully (no exception) so a broken agent produces a readable
:class:`SmokeResult` instead of a crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema

from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.events import EventBus
from janus.core.session import SessionStore

if TYPE_CHECKING:
    from janus.core.backend import AgentBackend
    from janus.core.persona import Persona


@dataclass
class SmokeCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class SmokeResult:
    passed: bool
    checks: list[SmokeCheck] = field(default_factory=list)
    deliverable: dict[str, Any] | None = None


async def smoke_run(
    persona: Persona,
    backend: AgentBackend,
    subject: str,
    working_directory: Path,
    *,
    session_store: SessionStore | None = None,
    events: EventBus | None = None,
) -> SmokeResult:
    """Run ``persona``'s agent on ``subject`` once and check it's mechanically sound."""
    persona.prepare_workspace(working_directory)

    controller = AgentController(
        load_config(persona=persona.name),
        backend=backend,
        session_store=session_store or SessionStore(sessions_dir=working_directory / ".sessions"),
        events=events,
    )
    result = await controller.run(persona.build_task(subject))

    checks: list[SmokeCheck] = []

    completed = result["status"] == "completed"
    checks.append(
        SmokeCheck(
            "run_completed",
            completed,
            f"status={result['status']!r}" if not completed else "Run completed.",
        )
    )

    deliverable: dict[str, Any] | None = None
    if persona.output_schema is not None:
        output_path = working_directory / persona.output_filename
        try:
            if not output_path.exists():
                raise FileNotFoundError(f"{output_path} does not exist")
            deliverable = json.loads(output_path.read_text())
            jsonschema.validate(instance=deliverable, schema=persona.output_schema)
        except (FileNotFoundError, json.JSONDecodeError, jsonschema.ValidationError) as e:
            checks.append(SmokeCheck("deliverable_valid", False, str(e)))
            deliverable = None
        else:
            checks.append(SmokeCheck("deliverable_valid", True, "Deliverable is schema-valid."))

    return SmokeResult(
        passed=all(c.ok for c in checks),
        checks=checks,
        deliverable=deliverable,
    )
