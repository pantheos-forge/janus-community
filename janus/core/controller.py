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

"""Async lifecycle orchestrator for a single agent run.

`AgentController` drives one `AgentBackend` through a run: connects it,
feeds it the initial task, streams its messages back out onto the
`EventBus`, and persists session state (status, cost, errors) via
`SessionStore`.

In this build the controller REQUIRES an injected backend -- automatic
backend selection from config is deferred to a later plan.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
from typing import TYPE_CHECKING, Any

from janus.core.backend import MessageType
from janus.core.events import Event, EventBus, EventType
from janus.core.session import SessionStatus, SessionStore

if TYPE_CHECKING:
    from janus.core.backend import AgentBackend, AgentMessage
    from janus.core.config import JanusConfig
    from janus.core.session import SessionInfo


class AgentState(enum.Enum):
    """Lifecycle states an :class:`AgentController` can be in."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class AgentController:
    """Owns one run's backend lifecycle and control surface."""

    def __init__(
        self,
        config: JanusConfig,
        backend: AgentBackend | None = None,
        session_store: SessionStore | None = None,
        events: EventBus | None = None,
    ) -> None:
        """Create the controller for one run.

        Args:
            config: Fully-resolved run configuration.
            backend: Pre-built backend to drive. This build requires a
                backend to be injected; :meth:`run` raises if it is ``None``.
            session_store: Session persistence; defaults to a fresh
                :class:`SessionStore`.
            events: Event bus to publish/subscribe on; defaults to a fresh
                per-controller bus. Inject a shared instance to co-observe
                multiple controllers.
        """
        self.config = config
        self.backend = backend
        self.session_store = session_store or SessionStore()
        self.events = events or EventBus()

        self._state = AgentState.IDLE
        self._pause_requested = False
        self._inject_requested = False
        self._stop_requested = False
        self._resume_event = asyncio.Event()
        self._pending_instruction: str | None = None
        self._current_tool_name: str | None = None
        self._last_outcome = "ok"
        self._last_error: str | None = None

        # Captured by run() so control methods called from foreign threads
        # (e.g. the Textual UI thread) can marshal onto the run's loop.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._disconnect_task: asyncio.Task | None = None

        self.events.subscribe(EventType.USER_COMMAND, self._on_user_command)
        self.events.subscribe(EventType.USER_INPUT, self._on_user_input)

    # -- State -----------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """The controller's current lifecycle state."""
        return self._state

    def _set_state(self, state: AgentState, details: str = "") -> None:
        """Transition to ``state`` and emit a ``STATE_CHANGED`` event."""
        self._state = state
        self.events.emit_state(state.value, details)

    def _unsubscribe_user_events(self) -> None:
        """Unsubscribe from this controller's bus (best-effort, idempotent)."""
        with contextlib.suppress(Exception):
            self.events.unsubscribe(EventType.USER_COMMAND, self._on_user_command)
        with contextlib.suppress(Exception):
            self.events.unsubscribe(EventType.USER_INPUT, self._on_user_input)

    def _marshal(self, fn) -> None:
        """Run ``fn`` on the run's event loop if one is live, else inline.

        asyncio primitives are not thread-safe; the TUI calls control methods
        from its own thread. Marshaling through ``call_soon_threadsafe``
        guarantees wakeups are seen by the loop.
        """
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(fn)
        else:
            fn()

    # -- Control surface -------------------------------------------------

    def pause(self) -> bool:
        """Request a pause at the next message boundary.

        Only honored while :attr:`state` is ``RUNNING``.

        Pause halts display at the next message boundary AND gates the
        backend's agent loop (``backend.hold()``): gated backends stop
        issuing model calls until :meth:`resume` releases them. Backends
        without a gate (the ABC no-op) still only halt display.
        """
        if self._state is not AgentState.RUNNING:
            return False
        self._pause_requested = True
        if self.backend is not None:
            self._marshal(self.backend.hold)
        return True

    def resume(self, instruction: str | None = None) -> bool:
        """Resume a paused run, optionally delivering an instruction first.

        Only honored while :attr:`state` is ``PAUSED``.
        """
        if self._state is not AgentState.PAUSED:
            return False

        def _do() -> None:
            if instruction is not None:
                self._pending_instruction = instruction
            self._resume_event.set()
            if self.backend is not None:
                self.backend.release()

        self._marshal(_do)
        return True

    def stop(self) -> bool:
        """Request a stop at the next message boundary.

        Also unblocks a pending pause so a stopped-while-paused run
        doesn't hang forever waiting on a resume.
        """
        def _do() -> None:
            self._stop_requested = True
            self._resume_event.set()
            if self.backend is not None:
                self.backend.release()
            if (
                self.backend is not None
                and self._loop is not None
                and self._loop.is_running()
            ):
                # A run parked inside a blocking tool produces no messages, so
                # the message-boundary stop check never fires; cancelling the
                # backend unwinds the parked handler and ends the stream. Only
                # meaningful while a loop is live (marshaled path). Retain the
                # future and consume its exception so a failing disconnect
                # never surfaces as an unretrieved-exception warning.
                task = asyncio.ensure_future(self.backend.disconnect())
                self._disconnect_task = task
                task.add_done_callback(lambda t: t.cancelled() or t.exception())

        self._marshal(_do)
        return True

    def inject(self, instruction: str) -> bool:
        """Queue ``instruction`` to be delivered to the running backend.

        Only honored while :attr:`state` is ``RUNNING`` or ``PAUSED``. While
        ``RUNNING`` this delivers inline at the next message boundary
        without pausing; while ``PAUSED`` it is delivered once a
        subsequent :meth:`resume` unblocks the run.
        """
        if self._state not in (AgentState.RUNNING, AgentState.PAUSED):
            return False

        def _do() -> None:
            self._pending_instruction = instruction
            if self._state is AgentState.RUNNING:
                self._inject_requested = True

        self._marshal(_do)
        return True

    def reply(self, text: str) -> bool:
        """Deliver the user's answer to a run parked in ask_user.

        Thread-safe; only honored while :attr:`state` is ``AWAITING_INPUT``.
        """
        if self._state is not AgentState.AWAITING_INPUT:
            return False

        def _do() -> None:
            if self._state is not AgentState.AWAITING_INPUT:
                return  # re-check on the loop; state may have moved
            deliver = getattr(self.backend, "deliver_reply", None)
            if deliver is None:
                return
            deliver(text)
            self._set_state(AgentState.RUNNING, "Reply delivered")

        self._marshal(_do)
        return True

    def enable_user_replies(self) -> None:
        """Mark this run interactive: ask_user parks for a real reply."""
        if self.backend is not None and hasattr(self.backend, "user_reply_enabled"):
            self.backend.user_reply_enabled = True

    async def _deliver_pending_instruction(self) -> None:
        """Send the queued instruction (if any) to the backend and record it."""
        if not self._pending_instruction:
            return
        instruction = self._pending_instruction
        self._pending_instruction = None
        self.session_store.add_instruction(instruction)
        self.events.emit_message(f"Injecting: {instruction[:50]}...", "info")
        assert self.backend is not None
        await self.backend.query(instruction)

    # -- User-originated events -------------------------------------------

    def _on_user_command(self, event: Event) -> None:
        """Dispatch a ``USER_COMMAND`` event to the matching control method."""
        command = event.data.get("command")
        if command == "pause":
            self.pause()
        elif command == "resume":
            self.resume()
        elif command == "stop":
            self.stop()

    def _on_user_input(self, event: Event) -> None:
        """Forward ``USER_INPUT`` text: a reply while awaiting, else inject."""
        text = event.data.get("text", "")
        if not text:
            return
        if self._state is AgentState.AWAITING_INPUT:
            self.reply(text)
        else:
            self.inject(text)

    # -- Run -----------------------------------------------------------------

    async def run(self, task: str, resume_session_id: str | None = None) -> dict[str, Any]:
        """Run (or resume) one task to completion and return its outcome.

        Returns ``{"status": ..., "session_id": ..., "cost_usd": ...}``.
        """
        self._loop = asyncio.get_running_loop()

        self._pause_requested = False
        self._stop_requested = False
        self._last_outcome = "ok"
        self._last_error = None
        self._resume_event.clear()

        session: SessionInfo
        if resume_session_id:
            loaded = self.session_store.load(resume_session_id)
            session = loaded if loaded is not None else self.session_store.create(
                subject=self.config.subject or "",
                task=task,
                model=self.config.llm_model,
                persona=self.config.persona or "",
            )
        else:
            session = self.session_store.create(
                subject=self.config.subject or "",
                task=task,
                model=self.config.llm_model,
                persona=self.config.persona or "",
            )

        if self.backend is None:
            self._loop = None
            self._unsubscribe_user_events()
            raise ValueError("AgentController requires an injected backend in this build")

        try:
            self._set_state(AgentState.RUNNING, "Connecting...")

            if resume_session_id and self.backend.supports_resume:
                await self.backend.resume(session.backend_session_id or resume_session_id)
            else:
                await self.backend.connect()

            await self.backend.query(task)
            self.session_store.update_status(SessionStatus.RUNNING)

            async for msg in self.backend.receive_messages():
                if self._stop_requested:
                    self._set_state(AgentState.STOPPED, "Stopped")
                    self.session_store.update_status(SessionStatus.STOPPED)
                    break

                if self._pause_requested:
                    self._set_state(AgentState.PAUSED)
                    await self._resume_event.wait()
                    self._resume_event.clear()
                    self._pause_requested = False
                    self._set_state(AgentState.RUNNING)
                    await self._deliver_pending_instruction()

                if self._inject_requested:
                    self._inject_requested = False
                    await self._deliver_pending_instruction()

                self._process_message(msg, session)

            else:
                if self._last_outcome == "error":
                    self._set_state(AgentState.ERROR, "Run ended with an error")
                    self.session_store.set_error(
                        self._last_error or "Run ended with an error")
                    self.session_store.update_status(SessionStatus.ERROR)
                else:
                    self._set_state(AgentState.COMPLETED)
                    self.session_store.update_status(SessionStatus.COMPLETED)

            return {
                "status": self._state.name.lower(),
                "session_id": session.session_id,
                "cost_usd": session.total_cost_usd,
            }

        except Exception as e:
            self._set_state(AgentState.ERROR, str(e))
            self.session_store.set_error(str(e))
            self.session_store.update_status(SessionStatus.ERROR)
            return {
                "status": self._state.name.lower(),
                "session_id": session.session_id,
                "cost_usd": session.total_cost_usd,
            }
        finally:
            self._loop = None
            self._unsubscribe_user_events()
            with contextlib.suppress(Exception):
                await self.backend.disconnect()

    # -- Message processing ---------------------------------------------------

    def _process_message(self, msg: AgentMessage, session: SessionInfo) -> None:
        """Translate one backend message into EventBus events + session updates."""
        if msg.type is MessageType.TEXT:
            self.events.emit_message(msg.content)

        elif msg.type is MessageType.TOOL_START:
            self._current_tool_name = msg.tool_name
            self.events.emit_tool("start", msg.tool_name, msg.tool_args)

        elif msg.type is MessageType.TOOL_RESULT:
            tool_name = msg.tool_name or self._current_tool_name
            self.events.emit_tool("result", tool_name, result=msg.content)

        elif msg.type is MessageType.OUTPUT:
            self.events.emit_output(msg.content)

        elif msg.type is MessageType.AWAITING_INPUT:
            self._set_state(AgentState.AWAITING_INPUT, str(msg.content)[:200])
            data: dict[str, Any] = {"text": str(msg.content), "type": "question"}
            choices = list((msg.metadata or {}).get("choices") or [])
            if choices:
                data["choices"] = choices
            self.events.emit(Event(EventType.MESSAGE, data))

        elif msg.type is MessageType.RESULT:
            self._last_outcome = (msg.metadata or {}).get("outcome", "ok")
            cost = (msg.metadata or {}).get("cost_usd", 0)
            if cost:
                self.session_store.add_cost_to(session, cost)

        elif msg.type is MessageType.ERROR:
            # Retain the text so an outcome=error termination can persist it as
            # last_error; without this the failure only ever shows in the feed.
            self._last_error = str(msg.content)
            self.events.emit_message(str(msg.content), "error")
