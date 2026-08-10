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
import subprocess
from dataclasses import dataclass
from pathlib import Path

from janus.fleet.registry import FleetRegistry

# Image used only to get a root shell over the bind mount. Any tiny image would
# do; this one is near-universally cached.
_HELPER_IMAGE = "alpine"


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
        except PermissionError:
            # A containerized agent's runs/ holds root-owned files: the container
            # runs as root and writes into the ./runs bind mount, so the host user
            # cannot delete them and --purge silently fails to purge. Borrow root
            # from a container to clear the contents, then remove the (host-owned)
            # directory itself.
            dir_deleted = _docker_assisted_delete(path)
        except OSError:
            dir_deleted = False
    return RemoveResult(name=name, path=path, purged=purge, dir_deleted=dir_deleted)


def _docker_assisted_delete(path: Path) -> bool:
    """Delete ``path`` by clearing its contents from inside a container.

    Returns True only if the directory is actually gone. Never raises: no Docker,
    a failed container, or a still-populated directory all degrade to False,
    which is the pre-existing "deregistered, dir left on disk" outcome.
    """
    resolved = path.resolve()
    # This runs `rm -rf` as root over a bind mount, so refuse obviously-wrong
    # targets instead of trusting the registry record.
    if not resolved.is_dir() or resolved == Path.home() or len(resolved.parts) < 3:
        return False
    try:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{resolved}:/target", _HELPER_IMAGE,
             "find", "/target", "-mindepth", "1", "-maxdepth", "1",
             "-exec", "rm", "-rf", "{}", "+"],
            capture_output=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    try:
        resolved.rmdir()          # the mount point itself is host-owned
    except OSError:
        return False
    return True
