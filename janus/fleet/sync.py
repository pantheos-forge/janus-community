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

"""Re-vendor the current Janus runtime into exported fleet agents.

``sync_agent`` refreshes one agent's ``janus/`` package and wrapper files to
match the running source and commits the change in the agent's own git repo.
The persona directory and ``.env`` are never touched. Git content is the truth
for "stale": a sync commits only when something actually changed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import janus
from janus.core.persona import Persona
from janus.factory.export import write_runtime_and_wrappers
from janus.factory.render import (
    render_agent_runner,
    render_compose,
    render_dockerfile,
    render_env_example,
    render_gitignore,
    render_pyproject,
    render_readme,
    render_smoke_test,
)
from janus.fleet.registry import FleetRegistry

_COMMITTER = ["-c", "user.email=agent@janus.local", "-c", "user.name=Janus"]


@dataclass
class SyncResult:
    name: str
    status: str                                  # updated | current | skipped | error
    detail: str = ""
    sha: str | None = None
    changed_files: list[str] = field(default_factory=list)


@dataclass
class RuntimeStatus:
    label: str                 # "current" | "stale" | "unsynced" | "error"
    n_changed: int = 0

    @property
    def text(self) -> str:
        if self.label == "stale":
            return f"stale({self.n_changed})"
        return {"current": "current", "unsynced": "unsynced", "error": "?"}[self.label]


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def sync_agent(agent_dir: str | Path, *, source_sha: str, dry_run: bool = False,
               force: bool = False) -> SyncResult:
    """Sync one exported agent. Returns a SyncResult; never raises for expected
    failures (missing path, unloadable persona, git error) — those come back as
    ``status="error"``.

    A dirty repo is skipped unless ``force`` — then its current state is
    committed (``chore: checkpoint before sync``) before the sync writes, so no
    uncommitted change is ever lost. ``dry_run`` never writes, so a dirty repo is
    skipped in dry-run regardless of ``force``."""
    agent_dir = Path(agent_dir)
    name = agent_dir.name
    persona_dir = agent_dir / "persona"
    if not (persona_dir / "manifest.toml").exists():
        return SyncResult(name, "error", f"{agent_dir} is not an exported agent")

    was_git = (agent_dir / ".git").exists()
    dirty = False
    if was_git:
        st = _git(["status", "--porcelain"], agent_dir)
        dirty = st.returncode == 0 and bool(st.stdout.strip())
        if dirty and (dry_run or not force):
            return SyncResult(
                name, "skipped", "uncommitted changes — commit or stash, then re-sync")

    try:
        persona = Persona.load(persona_dir)
    except Exception as e:
        return SyncResult(name, "error", f"persona did not load: {e}")

    if dry_run:
        changed = _diff_against_source(persona, agent_dir, name)
        status = "updated" if changed else "current"
        detail = f"{len(changed)} files" if changed else "up to date"
        return SyncResult(name, status, detail, changed_files=changed)

    if dirty and force:
        checkpoint_add = _git(["add", "-A"], agent_dir)
        if checkpoint_add.returncode != 0:
            return SyncResult(
                name, "error",
                f"checkpoint git add failed: {checkpoint_add.stderr.strip()}")
        cp = _git([*_COMMITTER, "commit", "-q", "-m", "chore: checkpoint before sync"],
                  agent_dir)
        if cp.returncode != 0:
            return SyncResult(
                name, "error",
                f"checkpoint commit failed: {cp.stderr.strip() or cp.stdout.strip()}")

    write_runtime_and_wrappers(persona, agent_dir, name)

    if not was_git:
        init = _git(["init", "-q"], agent_dir)
        if init.returncode != 0:
            return SyncResult(name, "error", f"git init failed: {init.stderr.strip()}")
    add = _git(["add", "-A"], agent_dir)
    if add.returncode != 0:
        return SyncResult(name, "error", f"git add failed: {add.stderr.strip()}")

    st = _git(["status", "--porcelain"], agent_dir)
    changed = [line[3:] for line in st.stdout.splitlines() if line.strip()]
    if not changed:
        return SyncResult(name, "current", f"already at {source_sha}")

    commit = _git(
        [*_COMMITTER, "commit", "-q", "-m", f"sync: vendored runtime → {source_sha}"],
        agent_dir)
    if commit.returncode != 0:
        return SyncResult(
            name, "error",
            f"git commit failed (runtime left staged, uncommitted): "
            f"{commit.stderr.strip() or commit.stdout.strip()}")
    sha = _git(["rev-parse", "--short", "HEAD"], agent_dir).stdout.strip()
    detail = "initial commit (git-init'd)" if not was_git else f"{len(changed)} files"
    if dirty and force:
        detail = f"checkpointed dirty state, then {detail}"
    return SyncResult(name, "updated", detail, sha=sha, changed_files=changed)


def _diff_against_source(persona: Persona, agent_dir: Path, name: str) -> list[str]:
    """Files (janus/ + wrappers) that a real sync would change, without writing."""
    changed: list[str] = []
    janus_src = Path(janus.__file__).parent
    agent_janus = agent_dir / "janus"

    def _pkg_files(root: Path) -> set[Path]:
        if not root.exists():
            return set()
        return {
            p.relative_to(root)
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }

    for rel in sorted(_pkg_files(janus_src) | _pkg_files(agent_janus)):
        s, d = janus_src / rel, agent_janus / rel
        if not s.exists() or not d.exists() or s.read_bytes() != d.read_bytes():
            changed.append(f"janus/{rel}")

    # Mirror write_runtime_and_wrappers exactly: a containerized persona renders
    # its Ubuntu tool Dockerfile (not the slim default) and a docker-compose.yml.
    # Diverging here made a containerized agent read as perpetually 'stale' and
    # left compose drift invisible to staleness detection.
    wrappers = {
        "pyproject.toml": render_pyproject(name),
        "agent.py": render_agent_runner(),
        "README.md": render_readme(persona, name),
        "Dockerfile": render_dockerfile(persona.container),
        ".env.example": render_env_example(),
        ".gitignore": render_gitignore(),
        "tests/test_smoke.py": render_smoke_test(),
    }
    if persona.container is not None:
        wrappers["docker-compose.yml"] = render_compose(name)
    for wrapper_rel, content in wrappers.items():
        f = agent_dir / wrapper_rel
        if not f.exists() or f.read_text() != content:
            changed.append(wrapper_rel)
    return changed


def runtime_status(agent_dir: str | Path) -> RuntimeStatus:
    """Classify an agent's vendored runtime against the current source.

    error → no loadable persona; unsynced → persona but no vendored janus/;
    current → janus/ + wrappers match source; stale → they differ. Never raises.
    """
    agent_dir = Path(agent_dir)
    persona_dir = agent_dir / "persona"
    if not (persona_dir / "manifest.toml").exists():
        return RuntimeStatus("error")
    try:
        persona = Persona.load(persona_dir)
    except Exception:
        return RuntimeStatus("error")
    if not (agent_dir / "janus").exists():
        return RuntimeStatus("unsynced")
    changed = _diff_against_source(persona, agent_dir, agent_dir.name)
    return RuntimeStatus("stale", len(changed)) if changed else RuntimeStatus("current")


def sync_fleet(
    registry: FleetRegistry,
    *,
    only: str | None = None,
    source_sha: str,
    dry_run: bool = False,
    force: bool = False,
) -> list[SyncResult]:
    """Sync all registered agents (or just ``only``), isolating per-agent failures.

    Records ``synced_to`` for agents actually updated on a real run. Returns the
    per-agent results in registry (sorted) order.
    """
    agents = registry.agents()
    if only is not None:
        if only not in agents:
            return [SyncResult(only, "error", "no such fleet agent")]
        items = [(only, agents[only])]
    else:
        items = sorted(agents.items())

    results: list[SyncResult] = []
    for name, meta in items:
        try:
            res = sync_agent(Path(meta.get("path", "")), source_sha=source_sha,
                             dry_run=dry_run, force=force)
        except Exception as e:                      # defensive: never abort the sweep
            res = SyncResult(name, "error", f"unexpected: {e}")
        res.name = name                             # registry name, not dir basename
        if res.status == "updated" and not dry_run:
            registry.set_synced_to(name, source_sha)
        results.append(res)
    return results


def _source_sha() -> str:
    """Short SHA of the running janus source, or a version fallback when Janus
    is installed rather than run from a git checkout."""
    janus_src = Path(janus.__file__).parent
    r = subprocess.run(["git", "-C", str(janus_src), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return f"v{janus.__version__} (no git)"
