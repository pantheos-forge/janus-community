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

"""Git-preserving re-export of an improved persona into its fleet agent repo.

Improvement updates the agent's four declarative files IN PLACE and commits
them in the agent's own git repository, so its version history is the audit
trail and rollback is ``git revert`` — the repo is never wiped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_GENERATED_FILES = ("manifest.toml", "prompt.md", "output_schema.json", "rubric.toml")
# Optional art files: copied only when the improve run produced them, so an
# improvement that regenerates a banner carries it into the fleet repo instead
# of silently dropping it. A run that doesn't touch art leaves any existing
# fleet banner in place (these are copy-if-present, never deleted).
_OPTIONAL_FILES = ("banner.txt", "banner_source.txt", "container.toml")
_COMMITTER = ["-c", "user.email=agent@janus.local", "-c", "user.name=Janus"]


class ImproveError(Exception):
    """A fleet improvement re-export could not be committed."""


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def sync_and_commit(build_persona_dir: Path, fleet_agent_dir: Path, *, summary: str) -> str:
    """Copy the persona files into ``fleet_agent_dir/persona`` and commit them.

    The four declarative files always; ``banner.txt``/``banner_source.txt``
    only when the build dir has them (a run that didn't regenerate art leaves
    any existing fleet banner untouched).

    Ensures the agent dir is a git repo (initializes one if missing), stages
    ``persona/``, commits ``improve: <summary>``, and returns the new commit's
    short SHA. Raises :class:`ImproveError` on any git failure.
    """
    build_persona_dir = Path(build_persona_dir)
    fleet_agent_dir = Path(fleet_agent_dir)
    dest = fleet_agent_dir / "persona"
    dest.mkdir(parents=True, exist_ok=True)
    for f in _GENERATED_FILES:
        shutil.copy2(build_persona_dir / f, dest / f)
    for f in _OPTIONAL_FILES:
        src = build_persona_dir / f
        if src.exists():
            shutil.copy2(src, dest / f)

    if not (fleet_agent_dir / ".git").exists():
        r = _run_git(["init", "-q"], fleet_agent_dir)
        if r.returncode != 0:
            raise ImproveError(f"git init failed: {r.stderr.strip()}")

    r = _run_git(["add", "persona"], fleet_agent_dir)
    if r.returncode != 0:
        raise ImproveError(f"git add failed: {r.stderr.strip()}")

    r = _run_git([*_COMMITTER, "commit", "-q", "-m", f"improve: {summary}"], fleet_agent_dir)
    if r.returncode != 0:
        raise ImproveError(f"git commit failed: {r.stderr.strip() or r.stdout.strip()}")

    sha = _run_git(["rev-parse", "--short", "HEAD"], fleet_agent_dir).stdout.strip()
    return sha
