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

"""`janus fleet` CLI — list / status / run / validate / adopt / sync.

Thin argparse handlers over FleetRegistry and the existing runtime/validation
libraries. Every handler accepts ``args.fleet_dir`` so callers (and tests)
can point at a specific fleet home.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from janus.core.backends.select import build_backend_for_persona
from janus.core.persona import Persona
from janus.core.validation.container_smoke import container_run, docker_available
from janus.fleet.registry import FleetRegistry, FleetRegistryError
from janus.interface import launch
from janus.interface.fleet_app import run_fleet_dashboard

_SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def _registry(args: argparse.Namespace) -> FleetRegistry:
    return FleetRegistry(args.fleet_dir)


def cmd_list(args: argparse.Namespace) -> int:
    try:
        agents = _registry(args).agents()
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not agents:
        print("No agents in the fleet. Build one with the factory, or "
              "`janus fleet adopt <path>`.")
        return 0
    from janus.fleet.sync import runtime_status

    print(f"{'NAME':<28} {'DOMAIN':<24} {'RUNTIME':<11} LAST VALIDATION")
    for name, a in sorted(agents.items()):
        hist = a.get("validation_history") or []
        if hist:
            last = hist[-1]
            scores = " ".join(f"{k}={v:.2f}" for k, v in (last.get("scores") or {}).items())
            when = last.get("date", "")[:10]
            mark = "PASS" if last.get("passed") else "FAIL"
            tail = f"{mark} {when} {scores}"
        else:
            tail = "(never validated)"
        runtime = runtime_status(a.get("path", "")).text
        print(f"{name:<28} {a.get('domain', ''):<24} {runtime:<11} {tail}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        agent = _registry(args).get(args.agent)
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if agent is None:
        print(f"error: no fleet agent named {args.agent!r}", file=sys.stderr)
        return 1
    print(f"{agent['name']}  ({agent.get('domain', '')})")
    print(f"  description: {agent.get('description', '')}")
    print(f"  source:  {agent.get('source', '')}")
    print(f"  path:    {agent.get('path', '')}")
    print(f"  created: {agent.get('created', '')}   updated: {agent.get('updated', '')}")
    print("  validation history:")
    for h in agent.get("validation_history") or []:
        scores = " ".join(f"{k}={v:.2f}" for k, v in (h.get("scores") or {}).items())
        mark = "PASS" if h.get("passed") else "FAIL"
        print(f"    {h.get('date', '')}  {mark}  {scores}  {h.get('note', '')}")
    if not (agent.get("validation_history")):
        print("    (never validated)")
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    src = Path(args.path)
    persona_dir = src / "persona"
    if not (persona_dir / "manifest.toml").exists():
        print(f"error: {src} is not an exported agent (no persona/manifest.toml)",
              file=sys.stderr)
        return 1
    try:
        persona = Persona.load(persona_dir)
    except Exception as e:
        print(f"error: {src}/persona does not load: {e}", file=sys.stderr)
        return 1
    if not _SAFE_NAME.fullmatch(persona.name):
        print(f"error: refusing to adopt — unsafe persona name {persona.name!r} "
              "(expected lowercase_snake_case)", file=sys.stderr)
        return 1
    reg = FleetRegistry(args.fleet_dir)
    try:
        reg.load()
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    dest = Path(args.fleet_dir) / persona.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src, dest,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".venv", ".env", "runs", ".janus"),
    )
    try:
        reg.register(
            persona.name, domain=persona.domain, description=persona.description,
            source="adopted", path=str(dest))
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"adopted {persona.name!r} into {dest}")
    print(f"  for standalone runs: cd {dest} && pip install -e .")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStore

    try:
        agent = _registry(args).get(args.agent)
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if agent is None:
        print(f"error: no fleet agent named {args.agent!r}", file=sys.stderr)
        return 1
    persona_dir = Path(agent["path"]) / "persona"
    persona = Persona.load(persona_dir)
    from datetime import datetime
    workdir = Path(agent["path"]) / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")

    if persona.container is not None:
        import asyncio
        if not docker_available():
            print("error: Docker is required to run a containerized agent.", file=sys.stderr)
            return 1
        res = asyncio.run(container_run(persona, args.subject, workdir,
                                        on_line=lambda line: print(line)))
        if res.success:
            print(f"ran {persona.name} in-container -> deliverable: {res.output_path}")
            return 0
        print(f"error: {res.error}", file=sys.stderr)
        return 1

    persona.prepare_workspace(workdir)
    config = load_config(persona=persona.name, working_directory=workdir)
    try:
        backend = build_backend_for_persona(config, persona)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    controller = AgentController(
        config, backend=backend,
        session_store=SessionStore(sessions_dir=Path(agent["path"]) / ".janus" / "sessions"))
    launch(controller, persona.build_task(args.subject), title=persona.name, banner=persona.banner)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    import asyncio

    from janus.core.validation.harness import validate
    from janus.core.validation.production import (
        make_production_agent_backend,
        make_production_judge_backend,
    )
    from janus.core.validation.rubric import Rubric

    reg = _registry(args)
    try:
        agent = reg.get(args.agent)
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if agent is None:
        print(f"error: no fleet agent named {args.agent!r}", file=sys.stderr)
        return 1
    persona_dir = Path(agent["path"]) / "persona"
    persona = Persona.load(persona_dir)
    if persona.rubric_path is None:
        print(f"error: {args.agent!r} has no rubric", file=sys.stderr)
        return 1
    rubric = Rubric.load(persona.rubric_path)
    from datetime import datetime
    root = Path(agent["path"]) / "runs" / ("validate-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        report = asyncio.run(validate(
            persona, rubric,
            make_production_agent_backend, make_production_judge_backend, root))
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    scores = report.judge.scores if report.judge else {}
    reg.append_validation(args.agent, scores=scores, passed=report.passed,
                          note="fleet validate")
    print(f"smoke: {'PASS' if report.smoke.passed else 'FAIL'}")
    print(f"judge: {'PASS' if (report.judge and report.judge.passed) else 'FAIL'}")
    for k, v in scores.items():
        print(f"  {k}: {v:.2f}")
    return 0 if report.passed else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        print("The fleet dashboard needs an interactive terminal. "
              "Use `janus fleet list` for a non-interactive view.")
        return 0
    run_fleet_dashboard(args.fleet_dir)
    return 0


def _resolve_factory_persona_dir() -> Path:
    """Locate the factory persona directory (dev/live: repo personas/factory)."""
    candidate = Path("personas") / "factory"
    if (candidate / "manifest.toml").exists():
        return candidate
    # Fall back to the package-relative location if running from elsewhere.
    here = Path(__file__).resolve().parent.parent.parent / "personas" / "factory"
    return here


def cmd_improve(args: argparse.Namespace) -> int:
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStore

    try:
        agent = _registry(args).get(args.agent)
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if agent is None:
        print(f"error: no fleet agent named {args.agent!r}", file=sys.stderr)
        return 1

    # The factory tools resolve the fleet dir by calling load_config() fresh
    # (via JANUS_FLEET_DIR / the default), not from the config object built
    # here — so an explicit --fleet-dir must be propagated through the env
    # for load_fleet_persona / export_improved_persona to target it.
    import os
    os.environ["JANUS_FLEET_DIR"] = str(args.fleet_dir)

    factory = Persona.load(_resolve_factory_persona_dir())
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = Path(args.fleet_dir) / ".improve" / f"{args.agent}-{stamp}"
    factory.prepare_workspace(workdir)
    config = load_config(persona=factory.name, working_directory=workdir,
                         fleet_dir=args.fleet_dir)
    try:
        backend = build_backend_for_persona(config, factory)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    task = (
        f"IMPROVEMENT REQUEST for the existing fleet agent {args.agent!r}. "
        f"First call load_fleet_persona('{args.agent}'), then baseline-validate it, "
        f"diagnose, tighten, re-validate, and export_improved_persona when it passes. "
        f"The user's complaint: {args.complaint}"
    )
    controller = AgentController(
        config, backend=backend,
        session_store=SessionStore(
            sessions_dir=Path(args.fleet_dir) / ".janus" / "sessions"))
    launch(controller, task, title=f"improve:{args.agent}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    from janus.fleet.sync import _source_sha, sync_fleet

    reg = _registry(args)
    try:
        reg.load()
    except FleetRegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False))
    sha = _source_sha()
    results = sync_fleet(reg, only=getattr(args, "agent", None), source_sha=sha,
                         dry_run=dry_run, force=force)

    labels = {
        "updated": "would update" if dry_run else "updated",
        "current": "current",
        "skipped": "skipped",
        "error": "error",
    }
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        print(f"{r.name:<28} {labels[r.status]:<13} {r.detail}")

    summary = ", ".join(f"{counts[k]} {labels[k]}" for k in
                        ("updated", "current", "skipped", "error") if counts.get(k))
    tail = "  — dry run, nothing written" if dry_run else ""
    print(f"—\n{summary or 'no agents'}  (of {len(results)}){tail}")

    if getattr(args, "agent", None) is not None and len(results) == 1 \
            and results[0].status == "error":
        return 1
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    from janus.fleet.rename import RenameError, rename_agent
    reg = _registry(args)
    try:
        res = rename_agent(reg.fleet_dir, args.old, args.new)
    except RenameError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    tail = (f" (commit {res.sha})" if res.committed
            else " (renamed; commit failed — commit manually)")
    print(f"renamed {res.old} -> {res.new}{tail} at {res.new_path}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    from janus.fleet.remove import RemoveError, remove_agent
    reg = _registry(args)
    if args.purge and not args.yes:
        print(f"error: refusing to purge {args.name!r} without --yes — this "
              f"permanently deletes its directory and git history", file=sys.stderr)
        return 1
    try:
        res = remove_agent(reg.fleet_dir, args.name, purge=args.purge)
    except RemoveError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if res.purged:
        note = f"deleted {res.path}" if res.dir_deleted else \
               f"deregistered, but {res.path} could not be deleted"
        print(f"purged {res.name} — {note}")
    else:
        print(f"removed {res.name} from the fleet (files kept at {res.path}; "
              f"re-adopt with 'janus fleet adopt {res.path}')")
    return 0


def _fleet_default(args: argparse.Namespace) -> int:
    """`janus fleet` with no subcommand: dashboard on a TTY, else list."""
    if sys.stdout.isatty() and sys.stdin.isatty():
        return cmd_dashboard(args)
    return cmd_list(args)


def add_fleet_subparser(subparsers: argparse._SubParsersAction) -> None:
    fleet = subparsers.add_parser("fleet", help="Manage the fleet of generated agents")
    fleet.add_argument("--fleet-dir", dest="fleet_dir", default=None,
                       help="Fleet home (default from config: ~/janus-agents)")
    fsub = fleet.add_subparsers(dest="fleet_command")

    p = fsub.add_parser("list", help="List fleet agents")
    p.set_defaults(func=cmd_list)
    p = fsub.add_parser("status", help="Show one agent's details + history")
    p.add_argument("agent")
    p.set_defaults(func=cmd_status)
    p = fsub.add_parser("run", help="Run a fleet agent on a subject")
    p.add_argument("agent")
    p.add_argument("subject")
    p.set_defaults(func=cmd_run)
    p = fsub.add_parser("validate", help="Re-validate a fleet agent (records drift)")
    p.add_argument("agent")
    p.set_defaults(func=cmd_validate)
    p = fsub.add_parser("adopt", help="Import an exported agent repo into the fleet")
    p.add_argument("path")
    p.set_defaults(func=cmd_adopt)
    p = fsub.add_parser("improve", help="Improve a fleet agent via the factory")
    p.add_argument("agent")
    p.add_argument("complaint")
    p.set_defaults(func=cmd_improve)
    p = fsub.add_parser("sync", help="Re-vendor the current runtime into fleet agents")
    p.add_argument("agent", nargs="?", default=None,
                   help="Sync just this agent (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change; write nothing")
    p.add_argument("--force", action="store_true",
                   help="Sync even if an agent repo has uncommitted changes "
                        "(a checkpoint commit is made first, so nothing is lost)")
    p.set_defaults(func=cmd_sync)
    p = fsub.add_parser("rename", help="Rename a fleet agent")
    p.add_argument("old")
    p.add_argument("new")
    p.set_defaults(func=cmd_rename)
    p = fsub.add_parser("remove", help="Remove an agent from the fleet "
                        "(deregister; --purge to delete its files)")
    p.add_argument("name")
    p.add_argument("--purge", action="store_true",
                   help="also delete the agent's directory + git history")
    p.add_argument("--yes", action="store_true", help="confirm a --purge (required)")
    p.set_defaults(func=cmd_remove)
    p = fsub.add_parser("dashboard", help="Open the interactive fleet dashboard")
    p.set_defaults(func=cmd_dashboard)

    fleet.set_defaults(func=_fleet_default)
