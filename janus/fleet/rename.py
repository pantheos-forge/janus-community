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

"""Rename a fleet agent: move its directory, rewrite its manifest name, swap its
registry key, re-render name-derived wrappers, and commit — atomically enough to
keep the registry key and the directory name in agreement at all times.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from janus.core.persona import Persona
from janus.factory.render import render_compose, render_pyproject, render_readme
from janus.fleet.registry import FleetRegistry

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COMMITTER = ["-c", "user.email=agent@janus.local", "-c", "user.name=Janus"]


class RenameError(Exception):
    """A fleet agent could not be renamed (bad input or a mid-rename failure)."""


@dataclass
class RenameResult:
    old: str
    new: str
    new_path: Path
    sha: str
    committed: bool


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def rename_agent(fleet_dir: str | Path, old: str, new: str) -> RenameResult:
    fleet_dir = Path(fleet_dir)
    reg = FleetRegistry(fleet_dir)

    # 1. Validate (no mutation)
    if not _NAME_RE.fullmatch(new):
        raise RenameError(f"invalid name {new!r} — must be lowercase_snake_case")
    if new == old:
        raise RenameError("new name is the same as the old name")
    if reg.get(old) is None:
        raise RenameError(f"no fleet agent named {old!r}")
    if reg.get(new) is not None:
        raise RenameError(f"a fleet agent named {new!r} already exists")
    old_dir, new_dir = fleet_dir / old, fleet_dir / new
    if new_dir.exists():
        raise RenameError(f"{new_dir} already exists on disk — refusing to clobber")
    if not (old_dir / "persona" / "manifest.toml").exists():
        raise RenameError(f"{old_dir} is not a valid agent (no persona/manifest.toml)")
    if (old_dir / ".git").exists():
        st = _git(["status", "--porcelain"], old_dir)
        if st.stdout.strip():
            raise RenameError(
                f"{old!r} has uncommitted changes — commit or stash them first")

    # 2. Move, then rewrite+verify with rollback. `_write` snapshots each file's
    # original bytes before overwriting (None = file did not exist), and
    # `_rollback` restores every snapshotted file AND moves the dir back — so a
    # failure never leaves the persona's declared identity corrupted.
    try:
        old_dir.rename(new_dir)
    except OSError as e:
        raise RenameError(f"could not move {old_dir} to {new_dir}: {e}") from e

    snap: dict[Path, bytes | None] = {}

    def _write(path: Path, content: str) -> None:
        if path not in snap:
            snap[path] = path.read_bytes() if path.exists() else None
        path.write_text(content)

    def _rollback() -> None:
        for path, original in snap.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)
        new_dir.rename(old_dir)

    try:
        manifest = new_dir / "persona" / "manifest.toml"
        text = manifest.read_text()
        text = re.sub(r'(\[persona\][^\[]*?\bname\s*=\s*)"[^"]*"',
                      lambda mo: mo.group(1) + f'"{new}"', text, count=1, flags=re.DOTALL)
        _write(manifest, text)
        persona = Persona.load(new_dir / "persona")
        if persona.name != new:
            raise RenameError(f"manifest name did not update to {new!r}")
        _write(new_dir / "pyproject.toml", render_pyproject(new))
        _write(new_dir / "README.md", render_readme(persona, new))
        if persona.container is not None:
            _write(new_dir / "docker-compose.yml", render_compose(new))
    except Exception as e:
        _rollback()
        raise RenameError(f"rename failed, rolled back: {e}") from e

    # 3. Registry swap (registry + dir now both `new`). Any failure here —
    # FleetRegistryError or an OSError from persisting the JSON — must roll the
    # filesystem back so registry and directory never disagree.
    try:
        reg.rename(old, new)
    except Exception as e:
        _rollback()
        raise RenameError(f"registry rename failed, rolled back: {e}") from e

    # 4. Commit (agent stays renamed even if the commit fails)
    if not (new_dir / ".git").exists():
        _git(["init", "-q"], new_dir)
    _git(["add", "-A"], new_dir)
    commit = _git([*_COMMITTER, "commit", "-q", "-m", f"rename: {old} -> {new}"], new_dir)
    if commit.returncode != 0:
        return RenameResult(old, new, new_dir, "", committed=False)
    sha = _git(["rev-parse", "--short", "HEAD"], new_dir).stdout.strip()
    return RenameResult(old, new, new_dir, sha, committed=True)
