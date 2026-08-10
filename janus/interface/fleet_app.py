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

"""FleetDashboardApp — the multi-session fleet TUI.

A FleetScreen (agents table) is the home; activating an agent pushes a
SessionScreen wrapping a SessionView driven by that agent's live supervisor
session. Sessions live in the FleetSupervisor (not the screens), so switching
screens never disturbs a running session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Label, RichLog, Static

from janus.core.events import EventType
from janus.core.validation.container_smoke import docker_available
from janus.fleet.registry import FleetRegistry
from janus.fleet.remove import RemoveError, remove_agent
from janus.fleet.rename import RenameError, rename_agent
from janus.fleet.supervisor import FleetSupervisor
from janus.interface.components.session_view import SessionView
from janus.interface.fleet_screen import FleetScreen


class PromptModal(ModalScreen[str]):
    """A one-line text prompt; dismisses with the entered value (or '')."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield Vertical(Label(self._prompt), Input(id="prompt_input"), id="prompt_box")

    def on_mount(self) -> None:
        self.query_one("#prompt_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss("")


class ValidationResultModal(ModalScreen[None]):
    """Shows a completed validation's smoke/judge verdict + scores."""

    BINDINGS = [Binding("escape", "dismiss_none", "Close"),
                Binding("enter", "dismiss_none", "Close")]

    def __init__(self, agent: str, outcome: Any) -> None:
        super().__init__()
        self._agent = agent
        self._outcome = outcome

    def compose(self) -> ComposeResult:
        o = self._outcome
        lines = [Label(f"Validation — {self._agent}")]
        if o.error:
            lines.append(Static(f"error: {o.error}"))
        else:
            lines.append(Static(f"smoke: {'PASS' if o.smoke_passed else 'FAIL'}"))
            lines.append(Static(f"judge: {'PASS' if o.judge_passed else 'FAIL'}"))
            for k, v in o.scores.items():
                lines.append(Static(f"  {k}: {v:.2f}"))
        yield Vertical(*lines, id="validation_box")

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class ValidationDetailModal(ModalScreen[None]):
    """Full per-criterion breakdown of an agent's latest validation."""

    BINDINGS = [Binding("escape", "dismiss_none", "Close"),
                Binding("enter", "dismiss_none", "Close")]

    def __init__(self, agent: str, entry: dict | None) -> None:
        super().__init__()
        self._agent = agent
        self._entry = entry

    def compose(self) -> ComposeResult:
        e = self._entry
        lines = [Label(f"Validation — {self._agent}")]
        if not e:
            lines.append(Static("(never validated)"))
        else:
            mark = "PASS" if e.get("passed") else "FAIL"
            lines.append(Static(f"{mark} · {(e.get('date') or '')[:10]}"))
            for k, v in (e.get("scores") or {}).items():
                lines.append(Static(f"  {k}: {v:.2f}"))
        yield Vertical(*lines, id="validation_box")

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class SyncResultModal(ModalScreen[None]):
    """Shows the outcome of a dashboard-triggered fleet sync."""

    BINDINGS = [Binding("escape", "dismiss_none", "Close"),
                Binding("enter", "dismiss_none", "Close")]

    def __init__(self, agent: str, result: Any) -> None:
        super().__init__()
        self._agent = agent
        self._result = result

    def compose(self) -> ComposeResult:
        r = self._result
        lines = [Label(f"Sync — {self._agent}")]
        if r.status == "updated":
            lines.append(Static(f"updated — {r.detail}"))
            if r.sha:
                lines.append(Static(f"commit {r.sha}"))
        elif r.status == "current":
            lines.append(Static("already current"))
        elif r.status == "skipped":
            lines.append(Static(f"skipped: {r.detail}"))
        else:
            lines.append(Static(f"error: {r.detail}"))
        yield Vertical(*lines, id="sync_box")

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class ContainerRunScreen(Screen[None]):
    """Live scrolling log of a batch container run (read-only)."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back to fleet")]

    def __init__(self, supervisor: Any, session_id: str, agent: str) -> None:
        super().__init__()
        self._sup = supervisor
        self._sid = session_id
        self._agent = agent
        self._bus: Any = None

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(f"Container run — {self._agent}"),
            RichLog(id="crun_log", wrap=True, markup=False, highlight=False),
            id="crun_screen",
        )

    def on_mount(self) -> None:
        log = self.query_one("#crun_log", RichLog)
        for line in self._sup.container_run_log(self._sid):
            log.write(line)
        self._bus = self._sup.bus_for(self._sid)
        if self._bus is not None:
            self._bus.subscribe(EventType.MESSAGE, self._on_bus_message)

    def on_unmount(self) -> None:
        # Drop the bus subscription so re-opening (Enter after Escape) doesn't
        # accumulate dead handlers for the life of the session bus.
        if self._bus is not None:
            self._bus.unsubscribe(EventType.MESSAGE, self._on_bus_message)

    def _on_bus_message(self, event: Any) -> None:
        # NOTE: this is deliberately NOT named `_on_message` — that name is
        # Textual MessagePump's own private message-dispatch method, and a
        # bus-event handler here would silently shadow it (breaking
        # on_event's internal `await self._on_message(event)` dispatch).
        try:
            self.query_one("#crun_log", RichLog).write(event.data.get("text", ""))
        except Exception:
            pass  # screen popped / not mounted — ignore late events


class MessageModal(ModalScreen[None]):
    """A one-line informational modal (close with esc/enter)."""

    BINDINGS = [Binding("escape", "dismiss_none", "Close"),
                Binding("enter", "dismiss_none", "Close")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Vertical(Static(self._message), id="message_box")

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """A yes/no confirmation; dismisses True on Yes, False on No.

    Both are clickable ``Button``s and keyboard shortcuts (``y``/``n``, ``esc``).
    """

    BINDINGS = [Binding("y", "yes", "Yes"), Binding("n", "no", "No"),
                Binding("escape", "no", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._message),
            Horizontal(
                Button("Yes", id="confirm_yes", variant="success"),
                Button("No", id="confirm_no", variant="error"),
                id="confirm_buttons",
            ),
            id="confirm_box",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm_yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class SessionScreen(Screen[None]):
    """Wraps a SessionView for one supervisor session."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back to fleet")]

    def __init__(
        self,
        controller: Any,
        bus: Any,
        *,
        initial_state: str | None = None,
        pending_question: tuple[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._bus = bus
        self._initial_state = initial_state
        self._pending_question = pending_question

    def compose(self) -> ComposeResult:
        yield SessionView(
            self._controller,
            self._bus,
            id="session_view",
            initial_state=self._initial_state,
            pending_question=self._pending_question,
        )


class FleetDashboardApp(App[None]):
    """The fleet dashboard."""

    CSS_PATH = "styles.tcss"

    def __init__(self, fleet_dir: str | Path, *, make_controller: Any = None,
                 max_concurrent: int | None = None) -> None:
        super().__init__()
        self._fleet_dir = Path(fleet_dir)
        # NOTE: Textual's own `App` already has an internal `self._registry`
        # (its DOM node registry, used by mounting/query_one/teardown) —
        # naming this attribute `_registry` silently shadows it and breaks
        # widget lookup and app shutdown. Use a distinct name.
        self._fleet_registry = FleetRegistry(self._fleet_dir)
        self.supervisor = FleetSupervisor(
            self._fleet_dir, make_controller=make_controller,
            max_concurrent=max_concurrent)

    def on_mount(self) -> None:
        self.push_screen(FleetScreen(self._fleet_registry, self.supervisor))

    def compose(self) -> ComposeResult:
        return iter(())

    def _latest_validation(self, agent: str) -> dict | None:
        try:
            a = self._fleet_registry.get(agent)
        except Exception:
            return None
        hist = (a or {}).get("validation_history") or []
        return hist[-1] if hist else None

    def _mark_synced_on_screen(self, agent: str) -> None:
        for screen in self.screen_stack:
            if isinstance(screen, FleetScreen):
                screen.mark_synced(agent)
                return

    async def _run_sync(self, agent: str) -> Any:
        import asyncio

        from janus.fleet.sync import SyncResult, _source_sha, sync_agent
        try:
            meta = self._fleet_registry.get(agent)
        except Exception:
            return SyncResult(agent, "error", "registry unreadable")
        path = (meta or {}).get("path")
        if not path:
            return SyncResult(agent, "error", "no such agent")

        def _do_sync(path: str):
            sha = _source_sha()
            return sha, sync_agent(path, source_sha=sha)

        sha, result = await asyncio.to_thread(_do_sync, path)
        result.name = agent
        if result.status == "updated":
            try:
                self._fleet_registry.set_synced_to(agent, sha)
            except Exception:
                pass
        if result.status in ("updated", "current"):
            self._mark_synced_on_screen(agent)
        return result

    # `push_screen_wait` may only be called from a Textual worker (it does
    # `get_current_worker()` internally and raises `NoActiveWorker`
    # otherwise) — a plain async message handler is *not* run as a worker,
    # so this handler must be `@work`-decorated. Verified against the
    # installed Textual (8.2.8) source at app.py's `push_screen`.
    @work
    async def on_fleet_screen_action_requested(self, m: FleetScreen.ActionRequested) -> None:
        if m.agent is None and m.kind != "adopt":
            return
        if m.kind == "run":
            from pathlib import Path
            assert m.agent is not None  # guarded by the m.agent-is-None check above
            subject = await self.push_screen_wait(PromptModal(f"Run {m.agent} on subject:"))
            if not subject:
                return
            containerized = (Path(self._fleet_dir) / str(m.agent)
                             / "persona" / "container.toml").exists()
            if containerized:
                if not docker_available():
                    await self.push_screen_wait(MessageModal(
                        "Docker is required to run a containerized agent."))
                    return
                sid = await self.supervisor.spawn_container_run(m.agent, subject)
                self.push_screen(ContainerRunScreen(self.supervisor, sid, m.agent))
            else:
                await self.supervisor.spawn(m.agent, subject)
        elif m.kind == "improve":
            complaint = await self.push_screen_wait(PromptModal(f"Improve {m.agent} — complaint:"))
            if complaint:
                await self.supervisor.spawn_improve(m.agent, complaint)
        elif m.kind == "validate":
            sid = await self.supervisor.spawn_validate(m.agent)
            outcome = await self.supervisor.validation_result(sid)
            # A double `v` press spawns two concurrent workers (Task's
            # `@work` isn't exclusive); Task 1's idempotent spawn_validate
            # makes both await the same sid/outcome, but each would still
            # independently push_screen_wait a *new* ValidationResultModal
            # instance, stacking two. Guard: if a validation modal is
            # already the top screen, the first worker already showed this
            # result — do not push a second one.
            if (isinstance(self.screen, ValidationResultModal)
                    and getattr(self.screen, "_agent", None) == m.agent):
                return
            await self.push_screen_wait(ValidationResultModal(m.agent, outcome))
        elif m.kind == "sync":
            result = await self._run_sync(m.agent)
            # A double `s` press spawns two concurrent workers, same as the
            # validate branch above — guard against stacking two modals for
            # the same agent's result.
            if (isinstance(self.screen, SyncResultModal)
                    and getattr(self.screen, "_agent", None) == m.agent):
                return
            self.push_screen(SyncResultModal(m.agent, result))
        elif m.kind == "containerize":
            from pathlib import Path

            from janus.fleet.supervisor import _TERMINAL_STATES

            try:
                meta = self._fleet_registry.get(m.agent)
            except Exception:
                meta = None
            path = (meta or {}).get("path")
            if not path:
                await self.push_screen_wait(MessageModal(f"No such agent: {m.agent}"))
                return
            if (Path(path) / "persona" / "container.toml").exists():
                await self.push_screen_wait(MessageModal(
                    f"{m.agent} is already containerized — use Improve to change its tools."))
                return
            if not docker_available():
                await self.push_screen_wait(MessageModal(
                    "Docker is required to containerize an agent (in-container "
                    "validation). Start Docker and try again."))
                return
            # In-flight guard: a second `c` press must not spawn a second
            # awaiter+sync for the same agent (two concurrent git syncs race
            # on the agent repo's index.lock). If a non-terminal containerize
            # session already exists for this agent, refuse the second press.
            if any(s.agent == m.agent and s.kind == "containerize"
                   and s.state not in _TERMINAL_STATES
                   for s in self.supervisor.sessions()):
                await self.push_screen_wait(MessageModal(
                    f"Containerization already in progress for {m.agent}."))
                return
            intent = await self.push_screen_wait(
                PromptModal(f"Containerize {m.agent} — what tools/capabilities?"))
            if not intent:
                return
            sid = await self.supervisor.spawn_containerize(m.agent, intent)
            await self.supervisor.await_session(sid)     # blocks this worker until done
            # The factory's export commits persona/container.toml only on a
            # SUCCESSFUL containerization; a failed run leaves the plain agent
            # untouched. Only render Docker wrappers (auto-sync) + show a
            # success-toned modal when it actually landed.
            if (Path(path) / "persona" / "container.toml").exists():
                result = await self._run_sync(m.agent)   # render Ubuntu Dockerfile + compose
                if (isinstance(self.screen, SyncResultModal)
                        and getattr(self.screen, "_agent", None) == m.agent):
                    return
                self.push_screen(SyncResultModal(m.agent, result))
            else:
                self.push_screen(MessageModal(
                    f"Containerization did not complete for {m.agent} — open its session "
                    "view for details."))
        elif m.kind == "rename":
            import asyncio

            from janus.fleet.supervisor import _TERMINAL_STATES
            assert m.agent is not None  # guarded by the m.agent-is-None check above
            if any(s.agent == m.agent and s.state not in _TERMINAL_STATES
                   for s in self.supervisor.sessions()):
                await self.push_screen_wait(MessageModal(
                    f"Can't rename {m.agent} while it has a live session — "
                    "wait for it to finish."))
                return
            new = await self.push_screen_wait(PromptModal(f"Rename {m.agent} to:"))
            if not new or new == m.agent:
                return
            try:
                res = await asyncio.to_thread(rename_agent, self._fleet_dir, m.agent, new)
            except RenameError as e:
                await self.push_screen_wait(MessageModal(str(e)))
                return
            # refresh the fleet table so the row shows the new name
            for screen in self.screen_stack:
                if isinstance(screen, FleetScreen):
                    screen.refresh_table()
                    break
            self.push_screen(MessageModal(f"renamed {res.old} -> {res.new}"))
        elif m.kind == "remove":
            import asyncio

            from janus.fleet.supervisor import _TERMINAL_STATES
            assert m.agent is not None  # guarded by the m.agent-is-None check above
            if any(s.agent == m.agent and s.state not in _TERMINAL_STATES
                   for s in self.supervisor.sessions()):
                await self.push_screen_wait(MessageModal(
                    f"Can't remove {m.agent} while it has a live session — "
                    "wait for it to finish."))
                return
            ok = await self.push_screen_wait(ConfirmModal(
                f"Remove {m.agent} from the fleet? Its files stay on disk "
                "(re-adopt to restore)."))
            if not ok:
                return
            try:
                rm_result = await asyncio.to_thread(
                    remove_agent, self._fleet_dir, m.agent, purge=False)
            except RemoveError as e:
                await self.push_screen_wait(MessageModal(str(e)))
                return
            for screen in self.screen_stack:
                if isinstance(screen, FleetScreen):
                    screen.refresh_table()
                    break
            self.push_screen(
                MessageModal(f"removed {rm_result.name} from the fleet (files kept)"))
        elif m.kind == "details":
            self.push_screen(ValidationDetailModal(m.agent, self._latest_validation(m.agent)))
        elif m.kind == "adopt":
            path = await self.push_screen_wait(PromptModal("Path to an exported agent to adopt:"))
            if path:
                from types import SimpleNamespace

                from janus.fleet.cli import cmd_adopt

                cmd_adopt(SimpleNamespace(fleet_dir=str(self._fleet_dir), path=path))

    async def on_fleet_screen_agent_activated(self, m: FleetScreen.AgentActivated) -> None:
        sessions = [s for s in self.supervisor.sessions() if s.agent == m.agent]
        if not sessions:
            entry = self._latest_validation(m.agent)
            if entry is not None:
                self.push_screen(ValidationDetailModal(m.agent, entry))
            return
        # Prefer the latest run/improve session — it has a live feed to
        # open, and must not be shadowed by a later-completed validate.
        attachable = [s for s in sessions if s.kind != "validate"]
        if attachable:
            info = attachable[-1]
            if info.kind == "container_run":
                self.push_screen(ContainerRunScreen(self.supervisor, info.id, m.agent))
                return
            controller = self.supervisor.controller_for(info.id)
            bus = self.supervisor.bus_for(info.id)
            if controller is not None and bus is not None:
                self.push_screen(
                    SessionScreen(
                        controller,
                        bus,
                        initial_state=info.state,
                        pending_question=info.pending_question,
                    )
                )
            # An attachable (run/improve) session exists but isn't openable
            # (builder-raised `error` session → no controller/bus): do
            # nothing rather than fall through to the validate branch below,
            # which could pop a STALE, unrelated validate's scores.
            return
        # No attachable session: if the latest validation is READY, show its
        # scores (never block the pump waiting on an in-flight one — the
        # design says "else ignore").
        validates = [s for s in sessions if s.kind == "validate"]
        if validates and self.supervisor.validation_ready(validates[-1].id):
            outcome = await self.supervisor.validation_result(validates[-1].id)
            self.push_screen(ValidationResultModal(m.agent, outcome))

    async def on_unmount(self) -> None:
        await self.supervisor.shutdown()


def _build_dashboard_app(fleet_dir: str | Path) -> FleetDashboardApp:
    """Construct the dashboard app, wiring the configured concurrency cap.

    Split out from :func:`run_fleet_dashboard` so it's testable without
    calling `App.run()`.
    """
    from janus.core.config import load_config

    config = load_config()
    return FleetDashboardApp(fleet_dir, max_concurrent=config.fleet_max_concurrent)


def run_fleet_dashboard(fleet_dir: str | Path) -> None:
    """Launch the fleet dashboard (blocks until the user quits)."""
    _build_dashboard_app(fleet_dir).run()
