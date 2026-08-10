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

"""The ``janus`` command-line interface — ``run`` / ``validate`` / ``export``.

Wires together everything built so far: persona loading, provider
auto-selection, the controller, the TUI/headless launcher, the validation
harness, and the export factory. This module is imported by the ``janus``
console-script entry point and by ``python -m janus``; it must stay
importable without the optional ``textual`` dependency (it never imports
``textual`` at module level — ``launch()`` lazily imports the TUI only when
one is actually needed).

``main()`` is synchronous by design: ``launch()`` and
``janus.core.validation.harness.validate`` each own their own event loop
(the former via an internal ``asyncio.run`` when the TUI path is taken, the
latter driven here via ``asyncio.run``), so nothing here may run under an
already-running loop.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from janus.core.backends.select import build_backend_for_persona
from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.persona import Persona
from janus.core.session import SessionStore
from janus.core.validation.harness import validate
from janus.core.validation.production import (
    make_production_agent_backend,
    make_production_judge_backend,
)
from janus.core.validation.rubric import Rubric
from janus.factory import export_agent
from janus.interface import launch


def resolve_persona_dir(spec: str) -> Path:
    """Resolve a persona spec (a directory path, or a name under ``personas/``).

    Accepts a raw path to a persona directory, or a bare name looked up under
    Janus's own ``personas/`` directory. Raises ``SystemExit`` with a clear
    message if neither resolves to an existing directory.
    """
    direct = Path(spec)
    if direct.is_dir():
        return direct

    under_personas = Path("personas") / spec
    if under_personas.is_dir():
        return under_personas

    raise SystemExit(
        f"error: no persona directory found for {spec!r} "
        f"(tried {direct} and {under_personas})"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    persona_dir = resolve_persona_dir(args.persona)
    persona = Persona.load(persona_dir)

    workdir = Path("runs") / persona.name
    persona.prepare_workspace(workdir)

    config = load_config(persona=persona.name, working_directory=workdir)

    try:
        backend = build_backend_for_persona(config, persona)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Configure a provider (see .env.example): set ANTHROPIC_API_KEY, or "
            "openrouter_model / local_model / ds4_url.",
            file=sys.stderr,
        )
        return 1

    sessions = SessionStore(sessions_dir=Path(".janus") / "sessions")
    controller = AgentController(config, backend=backend, session_store=sessions)

    launch(
        controller,
        persona.build_task(args.task),
        title=persona.name,
        banner=persona.banner,
        resume_session_id=args.resume,
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    persona_dir = resolve_persona_dir(args.persona)
    persona = Persona.load(persona_dir)

    if persona.rubric_path is None:
        print(
            f"error: persona {persona.name!r} has no validation rubric configured",
            file=sys.stderr,
        )
        return 1

    rubric = Rubric.load(persona.rubric_path)

    try:
        report = asyncio.run(
            validate(
                persona,
                rubric,
                make_production_agent_backend,
                make_production_judge_backend,
                Path("runs") / f"{persona.name}-validation",
            )
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Configure a provider (see .env.example): set ANTHROPIC_API_KEY, or "
            "openrouter_model / local_model / ds4_url.",
            file=sys.stderr,
        )
        return 1

    print(f"smoke: {'PASS' if report.smoke.passed else 'FAIL'}")
    if not report.smoke.passed:
        # Without this the CLI reports a bare "smoke: FAIL" and discards the
        # detail it already captured (container stderr, schema errors), leaving
        # a user nothing actionable to act on.
        for check in report.smoke.checks:
            if not check.ok:
                print(f"  {check.name}: {check.detail or 'failed'}", file=sys.stderr)
    if report.judge is None:
        print("judge: skipped (smoke failed)")
    else:
        print(f"judge: {'PASS' if report.judge.passed else 'FAIL'}")
        for name, score in report.judge.scores.items():
            print(f"  {name}: {score:.2f}")

    return 0 if report.passed else 1


def _cmd_export(args: argparse.Namespace) -> int:
    persona_dir = resolve_persona_dir(args.persona)
    persona = Persona.load(persona_dir)

    dest = export_agent(
        persona,
        args.dest,
        agent_name=args.name,
        git_init=not args.no_git,
        force=args.force,
    )
    print(f"exported {persona.name!r} to {dest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="janus", description="Janus agent runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a persona on a task")
    run_parser.add_argument("--persona", default="factory",
                            help="Persona name or directory (default: factory)")
    run_parser.add_argument("--task", required=True, help="Task subject")
    run_parser.add_argument("--resume", default=None, help="Session id to resume")
    run_parser.set_defaults(func=_cmd_run)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a persona against its rubric"
    )
    validate_parser.add_argument("--persona", required=True, help="Persona name or directory")
    validate_parser.set_defaults(func=_cmd_validate)

    export_parser = subparsers.add_parser(
        "export", help="Export a persona as a self-contained agent"
    )
    export_parser.add_argument("--persona", required=True, help="Persona name or directory")
    export_parser.add_argument("--dest", required=True, help="Destination directory")
    export_parser.add_argument("--name", default=None, help="Override the exported agent's name")
    export_parser.add_argument("--no-git", action="store_true", help="Skip git init in the export")
    export_parser.add_argument(
        "--force", action="store_true", help="Replace the destination if it already exists"
    )
    export_parser.set_defaults(func=_cmd_export)

    from janus.fleet.cli import add_fleet_subparser
    add_fleet_subparser(subparsers)

    dash = subparsers.add_parser("dashboard", help="Open the fleet dashboard")
    dash.add_argument("--fleet-dir", dest="fleet_dir", default=None)
    from janus.fleet.cli import cmd_dashboard
    dash.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, dispatch to the matching subcommand, and return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if (getattr(args, "fleet_dir", None) is None
            and getattr(args, "command", "") in ("fleet", "dashboard")):
        from janus.core.config import load_config
        args.fleet_dir = str(load_config().fleet_dir)
    return args.func(args)
