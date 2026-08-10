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

"""SessionView — a self-contained agent-session widget.

Reuses the 3B single-agent layout (activity feed + question panel + build
panel + input) and the bus->post_message->handler chain from JanusApp
(``janus/interface/tui.py``), scoped to one widget so the fleet dashboard
can mount one per session. The 3B pattern (bus handlers only ever call
``post_message``, which is safe from both a background thread and an
on-loop task; the actual widget mutation happens in the matching Textual
``on_*`` handler on the event loop) is reused verbatim. ``JanusApp`` itself
is untouched — this is a parallel, reusable extraction of the same chain.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input

from janus.core.events import Event, EventBus, EventType
from janus.interface.components.activity_feed import ActivityFeed
from janus.interface.components.build_panel import BuildPanel
from janus.interface.components.command_input import CommandInput
from janus.interface.components.question_panel import QuestionPanel

# ---------------------------------------------------------------------------
# Cross-thread/task Message classes
# ---------------------------------------------------------------------------
#
# Top-level (not nested) classes, matching tui.py's proven naming pattern:
# Textual derives on_<snake>_<snake>... from the class name, and nested or
# leading-underscore class names (e.g. `_StateChanged`) produce awkward
# double-underscored handler names (`on__state_changed`). Verified against
# the installed Textual (8.2.8) via Message.handler_name.


class SessionStateChanged(Message):
    """The session controller's lifecycle state changed."""

    def __init__(self, state: str, details: str) -> None:
        super().__init__()
        self.state = state
        self.details = details


class SessionMessageReceived(Message):
    """A prose message was emitted onto the session's EventBus."""

    def __init__(self, text: str, msg_type: str = "info") -> None:
        super().__init__()
        self.text = text
        self.msg_type = msg_type


class SessionQuestionAsked(Message):
    """The agent asked a question (possibly with quick-reply choices)."""

    def __init__(self, question: str, choices: list[str]) -> None:
        super().__init__()
        self.question = question
        self.choices = choices


class SessionToolEvent(Message):
    """A tool started or finished executing (feed-normalized status)."""

    def __init__(self, status: str, name: str, args: dict[str, Any], result: Any) -> None:
        super().__init__()
        self.status = status
        self.name = name
        self.args = args
        self.result = result


class SessionView(Vertical):
    """One agent session's feed + question panel + build panel + input.

    Encapsulates the 3B single-agent layout and bus wiring behind an
    injected ``(controller, bus)`` pair so a fleet dashboard can mount many
    of these — one per session — without JanusApp knowing anything about
    fleets, and without JanusApp itself being touched.
    """

    DEFAULT_CLASSES = "session-view"
    agent_state: reactive[str] = reactive("running")

    def __init__(
        self,
        controller: Any,
        bus: EventBus,
        *,
        id: str | None = None,
        initial_state: str | None = None,
        pending_question: tuple[str, list[str]] | None = None,
    ) -> None:
        super().__init__(id=id)
        self._controller = controller
        self._events = bus
        self._initial_state = initial_state
        self._initial_pending_question = pending_question

    def compose(self) -> ComposeResult:
        yield Horizontal(
            VerticalScroll(
                ActivityFeed(id="activity_feed", can_focus=False),
                id="content_area",
                can_focus=False,
            ),
            BuildPanel(id="build_panel"),
            id="main_row",
        )
        yield QuestionPanel(id="question_panel")
        yield CommandInput(commands=[], placeholder="Type your reply...", id="user_input")

    def on_mount(self) -> None:
        self._events.subscribe(EventType.STATE_CHANGED, self._on_agent_state_change)
        self._events.subscribe(EventType.MESSAGE, self._on_agent_message)
        self._events.subscribe(EventType.TOOL, self._on_agent_tool)
        if hasattr(self._controller, "enable_user_replies"):
            self._controller.enable_user_replies()

        # Late attach: the EventBus has no replay, so a session that was
        # already awaiting input (and asked its question) before this view
        # mounted would otherwise show a dead, unanswerable view. Seed both
        # the reactive state and the question panel from what the supervisor
        # cached, mirroring on_session_question_asked below.
        if self._initial_state is not None:
            self.agent_state = self._initial_state
        if self._initial_pending_question is not None:
            text, choices = self._initial_pending_question
            self.query_one("#activity_feed", ActivityFeed).add_message(
                f"Question: {text}", message_type="info"
            )
            self.query_one("#question_panel", QuestionPanel).show_question(text, choices)

    def on_unmount(self) -> None:
        self._events.unsubscribe(EventType.STATE_CHANGED, self._on_agent_state_change)
        self._events.unsubscribe(EventType.MESSAGE, self._on_agent_message)
        self._events.unsubscribe(EventType.TOOL, self._on_agent_tool)

    # -- bus handlers (may run off-loop: background thread or on-loop task,
    # depending on the controller) -> post_message only. NOTE: these must
    # NOT be named `_on_message`/`_on_state`/`_on_tool` etc. — those (and
    # similar dunder-ish names) collide with Textual's own MessagePump
    # internals (e.g. `_on_message`), which would silently shadow Textual's
    # dispatch and break the widget's own Compose/Mount handling.
    # ------------------------------------------------------------------------

    def _on_agent_state_change(self, event: Event) -> None:
        self.post_message(
            SessionStateChanged(
                event.data.get("state", "unknown").lower(), event.data.get("details", "")
            )
        )

    def _on_agent_message(self, event: Event) -> None:
        if event.data.get("type") == "question":
            self.post_message(
                SessionQuestionAsked(
                    event.data.get("text", ""), list(event.data.get("choices") or [])
                )
            )
            return
        text = event.data.get("text", "")
        if text:
            self.post_message(SessionMessageReceived(text, event.data.get("type", "info")))

    def _on_agent_tool(self, event: Event) -> None:
        status = event.data.get("status", "")
        if status == "start":
            feed_status = "running"
        elif status == "result":
            feed_status = "completed"
        else:
            return
        self.post_message(
            SessionToolEvent(
                status=feed_status,
                name=event.data.get("name", ""),
                args=event.data.get("args", {}),
                result=event.data.get("result"),
            )
        )

    # -- Textual handlers (on the loop) --------------------------------------

    def on_session_state_changed(self, event: SessionStateChanged) -> None:
        self.agent_state = event.state
        if event.details:
            self.query_one("#activity_feed", ActivityFeed).add_message(
                f"Agent: {event.details}", message_type="info"
            )

    def on_session_message_received(self, event: SessionMessageReceived) -> None:
        self.query_one("#activity_feed", ActivityFeed).add_message(
            event.text, message_type=event.msg_type
        )

    def on_session_question_asked(self, event: SessionQuestionAsked) -> None:
        self.query_one("#activity_feed", ActivityFeed).add_message(
            f"Question: {event.question}", message_type="info"
        )
        self.query_one("#question_panel", QuestionPanel).show_question(
            event.question, event.choices
        )

    def on_session_tool_event(self, event: SessionToolEvent) -> None:
        self.query_one("#activity_feed", ActivityFeed).add_tool_execution(
            event.name, event.args, event.status, event.result
        )
        try:
            self.query_one("#build_panel", BuildPanel).observe_tool(
                event.name, event.status, event.result
            )
        except Exception:
            pass

    # -- Agent-state reactive -------------------------------------------------

    def watch_agent_state(self, state: str) -> None:
        try:
            inp = self.query_one("#user_input", Input)
        except Exception:
            pass
        else:
            inp.disabled = state in ("completed", "error", "stopped", "idle")
            if state != "idle":
                inp.focus()

        if state != "awaiting_input":
            try:
                self.query_one("#question_panel", QuestionPanel).dismiss()
            except Exception:
                pass

    # -- Input routing --------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "user_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if self.agent_state in ("awaiting_input", "paused"):
            self._events.emit_input(text)
            return

        if self.agent_state == "running":
            self.query_one("#activity_feed", ActivityFeed).add_message(
                "Agent is running. Pause it first, or inject an instruction.",
                message_type="warning",
            )

    # -- Question-panel selection ---------------------------------------------

    def _select_choice(self, index: int) -> bool:
        """Answer the pending question with the choice at ``index`` (0-based).

        The single sink for BOTH answer paths — chip clicks
        (:meth:`on_button_pressed`) and digit keys
        (:meth:`select_question_choice`) — so both deliver the choice text
        through the same USER_INPUT emit_input path, exactly as a chip click
        does. Fails closed: returns False (no reply sent) when there is no
        pending question or ``index`` is out of range.
        """
        try:
            panel = self.query_one("#question_panel", QuestionPanel)
        except Exception:
            return False
        if not panel.display:
            return False
        choices = panel.current_choices
        if not (0 <= index < len(choices)):
            return False
        self._events.emit_input(choices[index])
        panel.dismiss()
        return True

    def select_question_choice(self, index: int) -> bool:
        """Digit-key entry point, resolved by ``CommandInput._handle_digit``.

        ``CommandInput`` intercepts a bare digit typed into an empty reply
        box and delegates to the nearest ancestor exposing this method (see
        ``CommandInput._resolve_choice_selector``). This makes digit
        selection self-contained to a mounted ``SessionView`` — it works in
        the fleet dashboard and in a bare pilot host with no app-level
        ``select_question_choice`` — mirroring what ``JanusApp`` provides at
        the app level for its single session.
        """
        return self._select_choice(index)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        panel = None
        try:
            panel = self.query_one("#question_panel", QuestionPanel)
        except Exception:
            return
        if not panel.display:
            return
        buttons = list(panel.query(Button).nodes)
        if event.button not in buttons:
            return
        index = buttons.index(event.button)
        self._select_choice(index)
