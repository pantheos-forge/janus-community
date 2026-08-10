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

"""Build panel — event-derived factory pipeline view.

Watches TOOL events by name (no factory→TUI protocol): scaffold_persona /
validate_persona / export_persona. Hidden (`active=False`) until the first
factory-shaped event, so ordinary personas never see it. Unparseable results
degrade to phase glyphs (never a crash).
"""

from __future__ import annotations

import json
import re
from typing import Any

from textual.widgets import Static

FACTORY_TOOLS = ("scaffold_persona", "validate_persona", "export_persona")

_GLYPHS = {"pending": "○", "active": "▶", "done": "✓", "failed": "✗"}
_EXPORT_PATH_RE = re.compile(r"Exported .+? to (\S+?)\.?(?:\s|$)")


class BuildPanel(Static):
    """Right-side factory pipeline: phases, attempts, live judge scores."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.active = False
        self.phases: dict[str, str] = {
            "scaffold": "pending", "validate": "pending", "export": "pending"}
        self.attempt: int | None = None
        self.attempts_remaining: int | None = None
        self.scores: dict[str, float] = {}
        self.export_path: str | None = None

    def on_mount(self) -> None:
        self.display = False

    # -- state machine (pure; unit-testable without a mounted app) ---------

    def observe_tool(self, name: str, status: str, result: Any) -> None:
        if name not in FACTORY_TOOLS:
            return
        self.active = True
        phase = {"scaffold_persona": "scaffold", "validate_persona": "validate",
                 "export_persona": "export"}[name]
        if status == "running":
            self.phases[phase] = "active"
        elif status == "completed":
            self._complete(phase, str(result) if result is not None else "")
        self._refresh_view()

    def _complete(self, phase: str, result: str) -> None:
        if phase == "validate":
            self._absorb_validate_result(result)
            return
        if result.startswith("Error"):
            self.phases[phase] = "failed"
            return
        self.phases[phase] = "done"
        if phase == "export":
            m = _EXPORT_PATH_RE.search(result)
            if m:
                self.export_path = m.group(1)

    def _absorb_validate_result(self, result: str) -> None:
        if result.startswith("Error") or result.startswith("Infrastructure"):
            # "Error...": budget exhausted / refused.
            # "Infrastructure error during validation...": tool/runtime
            # failure with no attempt consumed — must not render as a pass.
            self.phases["validate"] = "failed"
            return
        try:
            data = json.loads(result)
            self.attempt = data.get("attempt")
            self.attempts_remaining = data.get("attempts_remaining")
            judge = data.get("judge") or {}
            self.scores = dict(judge.get("scores") or {})
            self.phases["validate"] = "done" if data.get("passed") else "failed"
        except (json.JSONDecodeError, TypeError, AttributeError):
            self.phases["validate"] = "done"       # degrade: glyph only
            self.scores = {}

    # -- rendering ---------------------------------------------------------

    def _refresh_view(self) -> None:
        if not self.is_mounted:
            return
        self.display = self.active
        lines = ["[b]Build[/b]", ""]
        for phase, label in (("scaffold", "Scaffold"),
                             ("validate", "Validate"),
                             ("export", "Export")):
            line = f"{_GLYPHS[self.phases[phase]]} {label}"
            if phase == "validate" and self.attempt is not None:
                total = self.attempt + (self.attempts_remaining or 0)
                line += f"  {self.attempt}/{total}"
            lines.append(line)
            if phase == "validate":
                for crit, score in self.scores.items():
                    lines.append(f"    {crit:<12} {score:.2f}")
        if self.export_path:
            lines.append("")
            lines.append(f"→ {self.export_path}")
        self.update("\n".join(lines))
