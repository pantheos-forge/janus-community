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

"""In-process supervisor for concurrent dashboard sessions.

Each session is an ``(EventBus, controller, asyncio.Task)`` triple running on
the caller's event loop (the Textual loop, in the dashboard). Sessions are
isolated by their own bus (a crash lands ERROR on that session only), capped
at ``max_concurrent`` running at once with the rest queued.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from janus.core.events import Event, EventBus, EventType
from janus.core.validation.container_smoke import container_run
from janus.fleet.registry import FleetRegistry

MakeController = Callable[[dict, str, EventBus], "tuple[Any, Any]"]

_TERMINAL_STATES = frozenset({"completed", "error", "stopped", "validated", "failed"})


@dataclass
class SessionInfo:
    id: str
    agent: str
    state: str = "queued"
    cost_usd: float = 0.0
    pending_question: tuple[str, list[str]] | None = None
    kind: str = "run"


@dataclass
class ValidationOutcome:
    smoke_passed: bool
    judge_passed: bool
    scores: dict[str, float]
    error: str | None = None


@dataclass
class ContainerRunOutcome:
    success: bool
    output_path: Path | None
    error: str | None


@dataclass
class _Session:
    info: SessionInfo
    bus: EventBus | None
    subject: str
    controller: Any = None
    task: asyncio.Task | None = field(default=None)
    done: asyncio.Event | None = None
    outcome: Any = None
    term_rank: int | None = None
    log_lines: list[str] = field(default_factory=list)


class FleetSupervisor:
    """Runs and tracks concurrent agent sessions on the current event loop."""

    def __init__(
        self,
        fleet_dir: str | Path,
        *,
        make_controller: MakeController | None = None,
        make_improve_controller: MakeController | None = None,
        make_containerize_controller: MakeController | None = None,
        validate_fn: Callable[[Any, Any, Path], Awaitable[Any]] | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self.fleet_dir = Path(fleet_dir)
        self._make_controller = make_controller or self._default_make_controller
        self._make_improve_controller = (
            make_improve_controller or self._default_make_improve_controller)
        self._make_containerize_controller = (
            make_containerize_controller or self._default_make_containerize_controller)
        self._validate_fn = validate_fn or self._default_validate
        self._max_concurrent = max_concurrent if max_concurrent is not None else 3
        self._sessions: dict[str, _Session] = {}
        self._order: list[str] = []
        self._shutting_down = False
        self._term_counter = 0

    # -- public API --------------------------------------------------------

    async def spawn(self, agent: str, subject: str) -> str:
        self._prune_sessions()
        session_id = uuid.uuid4().hex[:8]
        bus = EventBus()
        info = SessionInfo(id=session_id, agent=agent, state="queued")
        sess = _Session(info=info, bus=bus, subject=subject)
        self._sessions[session_id] = sess
        self._order.append(session_id)
        bus.subscribe(EventType.STATE_CHANGED, self._make_state_listener(session_id))
        bus.subscribe(EventType.MESSAGE, self._make_message_listener(session_id))
        self._maybe_start_queued()
        return session_id

    async def spawn_improve(self, agent: str, complaint: str) -> str:
        self._prune_sessions()
        existing = self._inflight(agent, "improve")
        if existing is not None:
            return existing
        session_id = uuid.uuid4().hex[:8]
        bus = EventBus()
        info = SessionInfo(id=session_id, agent=agent, state="queued", kind="improve")
        sess = _Session(info=info, bus=bus, subject=complaint)
        self._sessions[session_id] = sess
        self._order.append(session_id)
        bus.subscribe(EventType.STATE_CHANGED, self._make_state_listener(session_id))
        bus.subscribe(EventType.MESSAGE, self._make_message_listener(session_id))
        self._maybe_start_queued()
        return session_id

    async def spawn_containerize(self, agent: str, intent: str) -> str:
        self._prune_sessions()
        existing = self._inflight(agent, "containerize")
        if existing is not None:
            return existing
        session_id = uuid.uuid4().hex[:8]
        bus = EventBus()
        info = SessionInfo(id=session_id, agent=agent, state="queued", kind="containerize")
        sess = _Session(info=info, bus=bus, subject=intent, done=asyncio.Event())
        self._sessions[session_id] = sess
        self._order.append(session_id)
        bus.subscribe(EventType.STATE_CHANGED, self._make_state_listener(session_id))
        bus.subscribe(EventType.MESSAGE, self._make_message_listener(session_id))
        self._maybe_start_queued()
        return session_id

    async def spawn_validate(self, agent: str) -> str:
        self._prune_sessions()
        existing = self._inflight(agent, "validate")
        if existing is not None:
            return existing
        session_id = uuid.uuid4().hex[:8]
        info = SessionInfo(id=session_id, agent=agent, state="queued", kind="validate")
        sess = _Session(info=info, bus=None, subject="", done=asyncio.Event())
        self._sessions[session_id] = sess
        self._order.append(session_id)
        self._maybe_start_queued()
        return session_id

    async def spawn_container_run(self, agent: str, subject: str) -> str:
        self._prune_sessions()
        existing = self._inflight(agent, "container_run")
        if existing is not None:
            return existing
        session_id = uuid.uuid4().hex[:8]
        bus = EventBus()
        info = SessionInfo(id=session_id, agent=agent, state="queued", kind="container_run")
        sess = _Session(info=info, bus=bus, subject=subject, done=asyncio.Event())
        self._sessions[session_id] = sess
        self._order.append(session_id)
        self._maybe_start_queued()
        return session_id

    def sessions(self) -> list[SessionInfo]:
        return [self._sessions[sid].info for sid in self._order]

    def controller_for(self, session_id: str) -> Any:
        s = self._sessions.get(session_id)
        return s.controller if s else None

    def bus_for(self, session_id: str) -> EventBus | None:
        s = self._sessions.get(session_id)
        return s.bus if s else None

    def container_run_log(self, session_id: str) -> list[str]:
        sess = self._sessions.get(session_id)
        return list(sess.log_lines) if sess else []

    def validation_ready(self, session_id: str) -> bool:
        sess = self._sessions.get(session_id)
        return bool(sess and sess.done is not None and sess.done.is_set())

    async def await_session(self, session_id: str) -> None:
        sess = self._sessions.get(session_id)
        if sess is None or sess.done is None:
            return
        await sess.done.wait()

    async def shutdown(self) -> None:
        self._shutting_down = True
        for sess in self._sessions.values():
            if sess.controller is not None and hasattr(sess.controller, "stop"):
                sess.controller.stop()
        tasks = [s.task for s in self._sessions.values() if s.task and not s.task.done()]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # Anything still queued was never started (and never will be, now
        # that shutdown has run) — reflect that in state.
        for sess in self._sessions.values():
            if sess.info.state == "queued":
                sess.info.state = "stopped"
        # Unblock any validation_result awaiters: a validate session that
        # never made it into _run_validation (queued and cancelled/never
        # started) leaves `done` unset forever otherwise.
        for sess in self._sessions.values():
            if sess.done is not None and not sess.done.is_set():
                if sess.outcome is None:
                    sess.outcome = ValidationOutcome(False, False, {}, error="stopped")
                sess.done.set()

    # -- internals ---------------------------------------------------------

    def _running_count(self) -> int:
        return sum(1 for s in self._sessions.values()
                   if s.info.state in ("running", "awaiting_input", "paused", "validating"))

    def _maybe_start_queued(self) -> None:
        """Start queued sessions up to the concurrency cap (FIFO)."""
        if self._shutting_down:
            return
        for sid in self._order:
            if self._running_count() >= self._max_concurrent:
                return
            sess = self._sessions[sid]
            if sess.info.state != "queued":
                continue
            self._start(sess)

    def _inflight(self, agent: str, kind: str) -> str | None:
        for sid in self._order:
            s = self._sessions[sid]
            if (s.info.agent == agent and s.info.kind == kind
                    and s.info.state not in _TERMINAL_STATES):
                return sid
        return None

    def _prune_sessions(self) -> None:
        """Keep every non-terminal session + the terminal one that completed
        most recently per agent.

        "Most recent" tracks completion order, not spawn order (``self._order``
        is insertion/spawn order — two concurrent same-agent sessions can
        finish out of that order). Pruning runs in each session's terminal
        ``finally`` on the single-threaded loop, so stamping ``term_rank``
        the first time a session is seen terminal here follows completion
        order for the common case. (A session that turns terminal WITHOUT an
        immediate prune — e.g. a builder-error in ``_start`` or a terminal
        state emitted mid-coroutine — is only ranked at the next prune; if two
        such sessions are both unstamped then, they tie-break by spawn order.
        This only affects WHICH terminal row the table retains — display-only,
        and no worse than spawn-order — never reachability of a live session.)
        """
        for sid in self._order:
            s = self._sessions[sid]
            if s.info.state in _TERMINAL_STATES and s.term_rank is None:
                s.term_rank = self._term_counter
                self._term_counter += 1
        keep: set[str] = set()
        latest_terminal: dict[str, str] = {}
        latest_rank: dict[str, int] = {}
        for sid in self._order:
            s = self._sessions[sid]
            if s.info.state not in _TERMINAL_STATES:
                keep.add(sid)
            else:
                agent = s.info.agent
                if agent not in latest_rank or s.term_rank > latest_rank[agent]:
                    latest_rank[agent] = s.term_rank
                    latest_terminal[agent] = sid
        keep.update(latest_terminal.values())
        if len(keep) == len(self._order):
            return
        self._order = [sid for sid in self._order if sid in keep]
        self._sessions = {sid: self._sessions[sid] for sid in self._order}

    def _start(self, sess: _Session) -> None:
        if sess.info.kind == "validate":
            sess.info.state = "validating"
            sess.task = asyncio.ensure_future(self._run_validation(sess))
            return
        if sess.info.kind == "container_run":
            sess.info.state = "running"
            sess.task = asyncio.ensure_future(self._run_container(sess))
            return
        agent_record = {"name": sess.info.agent, "path": str(self.fleet_dir / sess.info.agent)}
        if sess.info.kind == "containerize":
            builder = self._make_containerize_controller
        elif sess.info.kind == "improve":
            builder = self._make_improve_controller
        else:
            builder = self._make_controller
        assert sess.bus is not None
        try:
            controller, run_coro = builder(agent_record, sess.subject, sess.bus)
        except Exception:
            sess.info.state = "error"
            if sess.done is not None and not sess.done.is_set():
                sess.done.set()
            return
        sess.controller = controller
        sess.info.state = "running"
        sess.task = asyncio.ensure_future(self._run_wrapped(sess, run_coro))

    async def _run_wrapped(self, sess: _Session, run_coro: Any) -> None:
        try:
            await run_coro
        except asyncio.CancelledError:
            sess.info.state = "stopped"
            raise
        except Exception:
            sess.info.state = "error"
        finally:
            if sess.done is not None and not sess.done.is_set():
                sess.done.set()
            # A finished session frees a slot; pull the next queued one.
            if sess.info.state not in ("running", "awaiting_input", "paused"):
                self._prune_sessions()
                self._maybe_start_queued()

    async def _run_validation(self, sess: _Session) -> None:
        from datetime import datetime

        from janus.core.persona import Persona
        from janus.core.validation.rubric import Rubric

        try:
            agent_path = self.fleet_dir / sess.info.agent
            persona = Persona.load(agent_path / "persona")
            if persona.rubric_path is None:
                sess.outcome = ValidationOutcome(False, False, {},
                                                 error="no rubric — cannot validate")
                sess.info.state = "error"
                return
            rubric = Rubric.load(persona.rubric_path)
            root = agent_path / "runs" / ("validate-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
            report = await self._validate_fn(persona, rubric, root)
            scores = report.judge.scores if report.judge else {}
            FleetRegistry(self.fleet_dir).append_validation(
                sess.info.agent, scores=scores, passed=report.passed,
                note="dashboard validate")
            sess.outcome = ValidationOutcome(
                report.smoke.passed, bool(report.judge and report.judge.passed), scores)
            sess.info.state = "validated" if report.passed else "failed"
        except asyncio.CancelledError:
            sess.info.state = "stopped"
            raise
        except Exception as e:
            sess.outcome = ValidationOutcome(False, False, {}, error=str(e))
            sess.info.state = "error"
        finally:
            if sess.done is not None:
                sess.done.set()
            if not self._shutting_down:
                self._prune_sessions()
                self._maybe_start_queued()

    async def _run_container(self, sess: _Session) -> None:
        from datetime import datetime

        from janus.core.persona import Persona

        def _sink(line: str) -> None:
            sess.log_lines.append(line)
            if sess.bus is not None:
                sess.bus.emit_message(line)

        try:
            agent_path = self.fleet_dir / sess.info.agent
            persona = Persona.load(agent_path / "persona")
            workdir = agent_path / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
            res = await container_run(persona, sess.subject, workdir, on_line=_sink)
            _sink(f"— completed — deliverable: {res.output_path}" if res.success
                  else f"— failed: {res.error}")
            sess.outcome = ContainerRunOutcome(res.success, res.output_path, res.error)
            sess.info.state = "completed" if res.success else "error"
        except Exception as e:
            _sink(f"— error: {e}")
            sess.outcome = ContainerRunOutcome(False, None, str(e))
            sess.info.state = "error"
        finally:
            if sess.done is not None and not sess.done.is_set():
                sess.done.set()
            if not self._shutting_down:
                self._prune_sessions()
                self._maybe_start_queued()

    async def _default_validate(self, persona: Any, rubric: Any, working_root: Path):
        from janus.core.validation.harness import validate
        from janus.core.validation.production import (
            make_production_agent_backend,
            make_production_judge_backend,
        )
        return await validate(persona, rubric, make_production_agent_backend,
                              make_production_judge_backend, working_root)

    async def validation_result(self, session_id: str) -> ValidationOutcome:
        sess = self._sessions.get(session_id)
        if sess is None:
            return ValidationOutcome(False, False, {}, error="session no longer available")
        if sess.done is not None:
            await sess.done.wait()
        return sess.outcome

    async def run_result(self, session_id: str) -> ContainerRunOutcome:
        sess = self._sessions.get(session_id)
        if sess is None or sess.done is None:
            return ContainerRunOutcome(False, None, "no such session")
        await sess.done.wait()
        return sess.outcome

    def _make_state_listener(self, session_id: str) -> Callable[[Event], None]:
        def listener(event: Event) -> None:
            sess = self._sessions.get(session_id)
            if sess is None:
                return
            state = event.data.get("state")
            if state:
                sess.info.state = state
                # The cached question is only meaningful while the agent is
                # actually awaiting input; any other state means it has been
                # answered or superseded.
                if state != "awaiting_input":
                    sess.info.pending_question = None
            cost = event.data.get("cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                sess.info.cost_usd = float(cost)
        return listener

    def _make_message_listener(self, session_id: str) -> Callable[[Event], None]:
        def listener(event: Event) -> None:
            sess = self._sessions.get(session_id)
            if sess is None:
                return
            if event.data.get("type") != "question":
                return
            text = event.data.get("text", "")
            choices = list(event.data.get("choices") or [])
            sess.info.pending_question = (text, choices)
        return listener

    def _improve_sessions_dir(self) -> Path:
        return self.fleet_dir / ".janus" / "sessions"

    def _default_make_controller(self, agent_record: dict, subject: str, bus: EventBus):
        from datetime import datetime

        from janus.core.backends.select import build_backend_for_persona
        from janus.core.config import load_config
        from janus.core.controller import AgentController
        from janus.core.persona import Persona
        from janus.core.session import SessionStore

        agent_path = Path(agent_record["path"])
        persona = Persona.load(agent_path / "persona")
        workdir = agent_path / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        persona.prepare_workspace(workdir)
        config = load_config(persona=persona.name, working_directory=workdir)
        backend = build_backend_for_persona(config, persona)
        controller = AgentController(
            config, backend=backend,
            session_store=SessionStore(sessions_dir=agent_path / ".janus" / "sessions"),
            events=bus,
        )
        controller.enable_user_replies()
        return controller, controller.run(persona.build_task(subject))

    def _default_make_improve_controller(self, agent_record: dict, complaint: str,
                                         bus: EventBus):
        agent_name = agent_record["name"]
        task = (
            f"IMPROVEMENT REQUEST for the existing fleet agent {agent_name!r}. "
            f"First call load_fleet_persona('{agent_name}'), then baseline-validate it, "
            f"diagnose, tighten, re-validate, and export_improved_persona when it passes. "
            f"The user's complaint: {complaint}"
        )
        return self._factory_controller(agent_name, task, bus)

    def _default_make_containerize_controller(self, agent_record: dict, intent: str,
                                              bus: EventBus):
        agent_name = agent_record["name"]
        task = (
            f"CONTAINERIZE REQUEST for the existing fleet agent {agent_name!r}. Give it "
            f"the command-line tools it needs to satisfy: {intent}. First call "
            f"load_fleet_persona('{agent_name}'), then check_docker. Research the EXACT "
            f"apt/pip/go package names (do not guess) and present the exact install list "
            f"at the spec gate (ask_user) for approval. Then call scaffold_persona "
            f"re-passing this agent's EXISTING prompt.md, output_schema.json and "
            f"rubric.toml VERBATIM, a manifest.toml that adds \"bash\" to "
            f"[tools].builtins, and the authored container_toml. Then validate_persona "
            f"(it runs in-container), and export_improved_persona when it passes. Do NOT "
            f"change the prompt, schema, or rubric beyond adding the container and bash."
        )
        return self._factory_controller(agent_name, task, bus)

    def _factory_controller(self, agent_name: str, task: str, bus: EventBus):
        import os
        from datetime import datetime

        from janus.core.backends.select import build_backend_for_persona
        from janus.core.config import load_config
        from janus.core.controller import AgentController
        from janus.core.persona import Persona
        from janus.core.session import SessionStore
        from janus.fleet.cli import _resolve_factory_persona_dir

        # Factory tools resolve the fleet dir from the env, not the config
        # object — set it to this dashboard's fleet (idempotent).
        os.environ["JANUS_FLEET_DIR"] = str(self.fleet_dir)
        factory = Persona.load(_resolve_factory_persona_dir())
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        workdir = self.fleet_dir / ".improve" / f"{agent_name}-{stamp}"
        factory.prepare_workspace(workdir)
        config = load_config(persona=factory.name, working_directory=workdir,
                             fleet_dir=self.fleet_dir)
        backend = build_backend_for_persona(config, factory)
        controller = AgentController(
            config, backend=backend,
            session_store=SessionStore(sessions_dir=self._improve_sessions_dir()),
            events=bus)
        controller.enable_user_replies()
        return controller, controller.run(task)
