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

"""JSON-backed registry of managed fleet agents.

One ``registry.json`` per fleet home records each agent's identity and its
validation history. Every mutator persists immediately; a corrupt file raises
``FleetRegistryError`` with recovery guidance rather than crashing a caller.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


class FleetRegistryError(Exception):
    """The fleet registry file is missing required structure or is corrupt."""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class FleetRegistry:
    """Read/update the ``registry.json`` under a fleet home directory."""

    def __init__(self, fleet_dir: str | Path) -> None:
        self.fleet_dir = Path(fleet_dir)
        self.path = self.fleet_dir / "registry.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"agents": {}}
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise FleetRegistryError(
                f"cannot read {self.path}: {e}. Fix or delete it and re-adopt "
                "agents with `janus fleet adopt <path>`."
            ) from e
        if not isinstance(data, dict) or "agents" not in data:
            raise FleetRegistryError(
                f"{self.path} is not a valid fleet registry (missing 'agents')."
            )
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.fleet_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    def agents(self) -> dict[str, dict[str, Any]]:
        return self.load()["agents"]

    def get(self, name: str) -> dict[str, Any] | None:
        return self.load()["agents"].get(name)

    def register(
        self,
        name: str,
        *,
        domain: str,
        description: str,
        source: str,
        path: str,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        data = self.load()
        now = clock()
        existing = data["agents"].get(name)
        if existing is None:
            data["agents"][name] = {
                "name": name,
                "domain": domain,
                "description": description,
                "created": now,
                "updated": now,
                "source": source,
                "path": path,
                "validation_history": [],
            }
        else:
            existing.update(
                domain=domain, description=description, source=source,
                path=path, updated=now,
            )
        self._save(data)

    def append_validation(
        self,
        name: str,
        *,
        scores: dict[str, float],
        passed: bool,
        note: str,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        data = self.load()
        agent = data["agents"].get(name)
        if agent is None:
            raise FleetRegistryError(f"no fleet agent named {name!r} to record validation for")
        now = clock()
        agent.setdefault("validation_history", []).append(
            {"date": now, "scores": scores, "passed": passed, "note": note}
        )
        agent["updated"] = now
        self._save(data)

    def set_synced_to(
        self,
        name: str,
        sha: str,
        *,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        """Record the main-repo SHA an agent's runtime was last synced to."""
        data = self.load()
        agent = data["agents"].get(name)
        if agent is None:
            raise FleetRegistryError(f"no fleet agent named {name!r} to record sync for")
        agent["synced_to"] = sha
        agent["updated"] = clock()
        self._save(data)

    def rename(self, old: str, new: str, *, clock: Callable[[], str] = _now_iso) -> None:
        """Move an agent's record from key ``old`` to ``new`` (identity swap).

        Updates the record's ``name`` and ``path`` (to ``fleet_dir/new``) and
        ``updated``; preserves validation history, ``synced_to``, ``created``,
        and ``source``. Raises if ``old`` is unknown or ``new`` already exists.
        """
        data = self.load()
        agents = data["agents"]
        if old not in agents:
            raise FleetRegistryError(f"no fleet agent named {old!r} to rename")
        if new in agents:
            raise FleetRegistryError(f"a fleet agent named {new!r} already exists")
        record = agents.pop(old)
        record["name"] = new
        record["path"] = str(self.fleet_dir / new)
        record["updated"] = clock()
        agents[new] = record
        self._save(data)

    def remove(self, name: str) -> None:
        """Drop an agent from the registry. Raises if it isn't registered."""
        data = self.load()
        if name not in data["agents"]:
            raise FleetRegistryError(f"no fleet agent named {name!r} to remove")
        del data["agents"][name]
        self._save(data)
