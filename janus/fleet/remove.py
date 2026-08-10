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

"""Remove a fleet agent: deregister it, and optionally purge its directory.

Deregister-first: the registry key is dropped before any filesystem delete, so a
delete failure leaves a deregistered agent with a lingering dir (safe,
re-adoptable) rather than a registered agent whose dir is half-gone.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from janus.fleet.registry import FleetRegistry


class RemoveError(Exception):
    """A fleet agent could not be removed."""


@dataclass
class RemoveResult:
    name: str
    path: Path
    purged: bool
    dir_deleted: bool


def remove_agent(fleet_dir: str | Path, name: str, *, purge: bool = False) -> RemoveResult:
    fleet_dir = Path(fleet_dir)
    reg = FleetRegistry(fleet_dir)
    record = reg.get(name)
    if record is None:
        raise RemoveError(f"no fleet agent named {name!r}")
    path = Path(record.get("path") or (fleet_dir / name))

    reg.remove(name)                       # deregister FIRST

    dir_deleted = False
    if purge and path.exists():
        try:
            shutil.rmtree(path)
            dir_deleted = True
        except OSError:
            dir_deleted = False
    return RemoveResult(name=name, path=path, purged=purge, dir_deleted=dir_deleted)
