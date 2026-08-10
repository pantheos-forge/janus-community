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

"""Textual TUI application for Janus.

``JanusApp`` is the terminal front-end: it owns the splash -> main-interface
lifecycle, bridges the controller's :class:`~janus.core.events.EventBus`
(which the controller emits on a background thread) onto Textual's
single-threaded message pump via a handful of :class:`~textual.message.Message`
subclasses, and exposes an in-app slash-command surface (``/pause``,
``/resume``, ``/status`` and friends) via :class:`SlashCommandProcessor`.

The app is generic: it drives an injected :class:`~janus.core.controller.AgentController`
and speaks only the domain-neutral event vocabulary (STATE_CHANGED, MESSAGE,
TOOL, OUTPUT, USER_INPUT).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from janus.core.controller import AgentController, AgentState
from janus.core.events import Event, EventBus, EventType
from janus.interface.components.activity_feed import ActivityFeed, escape_markup
from janus.interface.components.animations import STATUS_ANIMATION, get_frame
from janus.interface.components.build_panel import BuildPanel
from janus.interface.components.command_input import CommandInput
from janus.interface.components.crush_spinner import CrushSpinner
from janus.interface.components.info_overlay import InfoOverlay
from janus.interface.components.question_panel import QuestionPanel
from janus.interface.components.splash import SplashScreen
from janus.interface.theme import ERROR, PRIMARY, SECONDARY, SUCCESS, WARNING

if TYPE_CHECKING:
    from textual.timer import Timer as TextualTimer

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

_TIMER_INTERVAL = 1.0  # run-timer tick, seconds
_BUILD_UI_DELAY = 0.05  # deferral after splash removal before building the main UI
_ANIMATION_INTERVAL = 0.1  # shared spinner/status animation tick
_SPLASH_HIDE_DELAY = 4.0  # how long the splash stays up

# Placeholder/disabled contract for #user_input, keyed by agent_state.
_STATE_INPUT_TABLE: dict[str, tuple[str, bool]] = {
    "idle": ("Waiting for agent to start...", True),
    "running": ("Type / for commands...", False),
    "paused": ("Type instruction or /resume...", False),
    "awaiting_input": ("Type your answer and press Enter...", False),
    "completed": ("Run complete. Type / for commands...", False),
    "error": ("Run ended with error. Type / for commands...", False),
    "stopped": ("Run stopped. Type / for commands...", False),
}

# Status-bar label/style contract, keyed by agent_state.
_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "idle": ("Idle", "dim"),
    "running": ("Running", f"bold {PRIMARY}"),
    "paused": ("PAUSED", f"bold {WARNING}"),
    "awaiting_input": ("AWAITING INPUT", f"bold {WARNING}"),
    "completed": ("Completed", f"bold {SUCCESS}"),
    "error": ("Error", f"bold {ERROR}"),
    "stopped": ("STOPPED", f"bold {WARNING}"),
}


# ---------------------------------------------------------------------------
# Cross-thread Message classes
# ---------------------------------------------------------------------------
#
# EventBus handlers run on the agent thread; they must never touch Textual's
# blocking call_from_thread(). Instead they build one of these Message
# subclasses and post it via App.post_message() (thread-safe, non-blocking).


class AgentStateChanged(Message):
    """The controller's lifecycle state changed."""

    def __init__(self, state: str, details: str) -> None:
        super().__init__()
        self.state = state
        self.details = details


class AgentMessageReceived(Message):
    """A prose message was emitted onto the EventBus."""

    def __init__(self, text: str, msg_type: str = "info") -> None:
        super().__init__()
        self.text = text
        self.msg_type = msg_type


class AgentToolEvent(Message):
    """A tool started or finished executing (feed-normalized status)."""

    def __init__(self, status: str, name: str, args: dict[str, Any], result: Any) -> None:
        super().__init__()
        self.status = status
        self.name = name
        self.args = args
        self.result = result


class AgentOutput(Message):
    """A structured deliverable was emitted onto the EventBus."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class AgentQuestionAsked(Message):
    """The agent asked a question (possibly with quick-reply choices)."""

    def __init__(self, question: str, choices: list[str]) -> None:
        super().__init__()
        self.question = question
        self.choices = choices


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class HelpScreen(ModalScreen[None]):
    """Keyboard/command reference. Closes on any keypress (no BINDINGS table)."""

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Janus — Help", id="help_title"),
            Label(self._body(), id="help_content"),
            id="help_dialog",
        )

    def on_key(self, _event: events.Key) -> None:
        """Any keypress dismisses the overlay."""
        self.app.pop_screen()

    @staticmethod
    def _body() -> str:
        lines = [
            "Keyboard Shortcuts:",
            "  F1              Help",
            "  Ctrl+P          Pause/Resume",
            "  Ctrl+Q / Ctrl+C Quit",
            "  Arrow keys / PgUp / PgDn   Scroll the activity feed",
            "",
            "Commands:",
        ]
        for name in sorted(SlashCommandProcessor.COMMANDS):
            _handler, desc, usage = SlashCommandProcessor.COMMANDS[name]
            lines.append(f"  {usage:<25} {desc}")
        lines.append("")
        lines.append("Typing while paused injects an instruction into the running agent.")
        return "\n".join(lines)


class QuitScreen(ModalScreen[None]):
    """Confirmation modal for quitting the app."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left,right", "toggle_focus", "Toggle focus", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Quit Janus?", id="quit_title"),
            Label(
                "Are you sure you want to quit? Any running agent will be stopped.",
                id="quit_message",
            ),
            Grid(
                Button("Yes", variant="error", id="btn_quit_confirm"),
                Button("No", variant="default", id="btn_quit_cancel"),
                id="quit_buttons",
            ),
            id="quit_dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#btn_quit_cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Handled here; must never bubble to the App (whose Button.Pressed
        # handler would treat it as a question-panel chip press).
        event.stop()
        if event.button.id == "btn_quit_confirm":
            self.app.exit()
        else:
            self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def action_toggle_focus(self) -> None:
        confirm = self.query_one("#btn_quit_confirm", Button)
        cancel = self.query_one("#btn_quit_cancel", Button)
        if self.focused is confirm:
            cancel.focus()
        else:
            confirm.focus()


# ---------------------------------------------------------------------------
# Slash-command processor
# ---------------------------------------------------------------------------


class SlashCommandProcessor:
    """Parses and dispatches ``/command`` input lines against an app-like object.

    Deliberately only touches the documented ``app`` interface (``_controller``,
    ``agent_state``, ``_show_feed_message``, ``_format_timer``, ``push_screen``)
    so it stays testable without a mounted Textual App.
    """

    COMMANDS: ClassVar[dict[str, tuple[str, str, str]]] = {
        "pause": ("_cmd_pause", "Pause the run", "/pause"),
        "resume": ("_cmd_resume", "Resume the run", "/resume"),
        "stop": ("_cmd_stop", "Stop the run", "/stop"),
        "instruction": ("_cmd_instruction", "Inject a new instruction", "/instruction <text>"),
        "status": ("_cmd_status", "Show current status", "/status"),
        "help": ("_cmd_help", "Show available commands", "/help"),
    }

    def __init__(self, app: Any) -> None:
        self._app = app

    def process(self, text: str) -> None:
        """Parse and dispatch a raw ``/command arg...`` line."""
        body = text.lstrip("/")
        parts = body.split(None, 1)
        cmd_name = parts[0].lower() if parts else ""
        cmd_arg = parts[1] if len(parts) > 1 else ""

        entry = self.COMMANDS.get(cmd_name)
        if entry is None:
            self._app._show_feed_message(
                f"Unknown command: /{cmd_name}. Type /help for available commands.", "warning"
            )
            return
        handler_name, _desc, _usage = entry
        getattr(self, handler_name)(cmd_arg)

    # -- run lifecycle -------------------------------------------------------

    def _cmd_pause(self, _arg: str) -> None:
        self._app._controller.pause()

    def _cmd_resume(self, _arg: str) -> None:
        self._app._controller.resume()

    def _cmd_stop(self, _arg: str) -> None:
        self._app._controller.stop()
        self._app._show_feed_message("Stopping run...", "info")

    def _cmd_instruction(self, arg: str) -> None:
        app = self._app
        stripped = arg.strip()
        if not stripped:
            app._show_feed_message("Usage: /instruction <text>", "warning")
            return
        app._controller.inject(stripped)
        app._show_feed_message(f"Instruction queued: {stripped[:80]}", "info")

    def _cmd_status(self, _arg: str) -> None:
        app = self._app
        text = (
            f"State:    {app.agent_state}\n"
            f"Elapsed:  {app._format_timer()}"
        )
        app.push_screen(InfoOverlay("Status", text))

    def _cmd_help(self, _arg: str) -> None:
        app = self._app
        lines = [
            "Keyboard Shortcuts:",
            "  F1              Help",
            "  Ctrl+P          Pause/Resume",
            "  Ctrl+Q / Ctrl+C Quit",
            "",
            "Commands:",
        ]
        for name in sorted(self.COMMANDS):
            _handler, desc, usage = self.COMMANDS[name]
            lines.append(f"  {usage:<25} {desc}")
        lines.append("")
        lines.append("Typing while paused injects an instruction into the running agent.")
        app.push_screen(InfoOverlay("Help — Commands", "\n".join(lines)))


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class JanusApp(App[None]):
    """The Janus Textual TUI application."""

    CSS_PATH = "styles.tcss"
    TITLE = "Janus"

    show_splash: reactive[bool] = reactive(True)
    agent_state: reactive[str] = reactive("idle")

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("f1", "toggle_help", "Help", priority=True),
        Binding("ctrl+p", "toggle_pause", "Pause/Resume", priority=True),
        Binding("ctrl+q", "request_quit", "Quit", priority=True),
        Binding("ctrl+c", "request_quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        controller: AgentController,
        task: str,
        *,
        title: str | None = None,
        banner: str | None = None,
        resume_session_id: str | None = None,
    ) -> None:
        super().__init__()

        # Injected run driver + task. The controller is never constructed here.
        self._controller = controller
        self._task_text = task
        self._app_title = title or "Janus"
        self.title = self._app_title
        self._banner = banner
        self._resume_session_id = resume_session_id

        # Agent-thread state.
        self._agent_thread: threading.Thread | None = None
        self._activity_feed: ActivityFeed | None = None
        self._events: EventBus | None = None

        # Run timer state.
        self._timer_seconds: int = 0
        self._timer_running: bool = False
        self._timer_handle: TextualTimer | None = None

        # Slash commands.
        self._command_processor = SlashCommandProcessor(self)

        # Shared spinner/status animation.
        self._anim_step: int = 0
        self._header_spinner_obj = CrushSpinner()
        self._anim_timer: TextualTimer | None = None

    # -- small helpers ----------------------------------------------------

    def _show_feed_message(self, text: str, message_type: str = "info") -> None:
        """Append a message to the activity feed, if it exists yet."""
        if self._activity_feed is not None:
            self._activity_feed.add_message(text, message_type=message_type)

    # -- EventBus wiring ----------------------------------------------------

    def _setup_event_handlers(self) -> None:
        if self._events is None:
            return
        self._events.subscribe(EventType.STATE_CHANGED, self._on_state_change)
        self._events.subscribe(EventType.MESSAGE, self._on_agent_message)
        self._events.subscribe(EventType.TOOL, self._on_tool)
        self._events.subscribe(EventType.OUTPUT, self._on_output)

    def _teardown_event_handlers(self) -> None:
        if self._events is None:
            return
        self._events.unsubscribe(EventType.STATE_CHANGED, self._on_state_change)
        self._events.unsubscribe(EventType.MESSAGE, self._on_agent_message)
        self._events.unsubscribe(EventType.TOOL, self._on_tool)
        self._events.unsubscribe(EventType.OUTPUT, self._on_output)

    # -- _on_* bridges: AGENT THREAD ONLY, must only call post_message() ---

    def _on_state_change(self, event: Event) -> None:
        self.post_message(
            AgentStateChanged(
                event.data.get("state", "unknown").lower(), event.data.get("details", "")
            )
        )

    def _on_agent_message(self, event: Event) -> None:
        if event.data.get("type") == "question":
            self.post_message(
                AgentQuestionAsked(
                    event.data.get("text", ""),
                    list(event.data.get("choices") or []),
                )
            )
            return
        text = event.data.get("text", "")
        if text:
            self.post_message(AgentMessageReceived(text, event.data.get("type", "info")))

    def _on_tool(self, event: Event) -> None:
        # Translate the controller's tool vocabulary ("start"/"result") into
        # the activity feed's vocabulary ("running"/"completed") here, so no
        # raw controller status ever reaches ActivityFeed.add_tool_execution
        # (which would render as an "Unknown" row and drop the result).
        status = event.data.get("status", "")
        if status == "start":
            feed_status = "running"
        elif status == "result":
            feed_status = "completed"
        else:
            return
        self.post_message(
            AgentToolEvent(
                status=feed_status,
                name=event.data.get("name", ""),
                args=event.data.get("args", {}),
                result=event.data.get("result"),
            )
        )

    def _on_output(self, event: Event) -> None:
        self.post_message(AgentOutput(dict(event.data)))

    # -- Timer helpers ------------------------------------------------------

    def _start_timer(self) -> None:
        if self._timer_handle is None:
            self._timer_handle = self.set_interval(_TIMER_INTERVAL, self._tick_timer)
        self._timer_running = True

    def _pause_timer(self) -> None:
        self._timer_running = False

    def _stop_timer(self) -> None:
        self._timer_running = False
        if self._timer_handle is not None:
            self._timer_handle.stop()

    def _tick_timer(self) -> None:
        if not self._timer_running:
            return
        self._timer_seconds += 1
        self._update_header_timer()

    def _format_timer(self) -> str:
        hours, rem = divmod(self._timer_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _update_header_timer(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#header_timer", Static).update(self._format_timer())

    # -- Textual message handlers: TEXTUAL EVENT LOOP ONLY ------------------

    def on_agent_state_changed(self, event: AgentStateChanged) -> None:
        self.agent_state = event.state
        if event.state == "running":
            self._start_timer()
        elif event.state == "paused":
            self._pause_timer()
        elif event.state in ("completed", "error", "stopped"):
            self._stop_timer()

        if self._activity_feed is not None and event.details:
            self._activity_feed.add_message(f"Agent: {event.details}", message_type="info")

    def on_agent_message_received(self, event: AgentMessageReceived) -> None:
        if self._activity_feed is not None:
            self._activity_feed.add_message(event.text, message_type=event.msg_type)

    def on_agent_tool_event(self, event: AgentToolEvent) -> None:
        if self._activity_feed is None:
            return
        # event.status is already feed-normalized ("running"/"completed").
        self._activity_feed.add_tool_execution(
            event.name, event.args, event.status, event.result
        )
        try:
            self.query_one("#build_panel", BuildPanel).observe_tool(
                event.name, event.status, event.result)
        except Exception:
            pass

    def on_agent_output(self, event: AgentOutput) -> None:
        if self._activity_feed is None:
            return
        keys = ", ".join(str(k) for k in event.data) if event.data else ""
        text = "Deliverable emitted"
        if keys:
            text = f"Deliverable emitted ({keys})"
        self._activity_feed.add_message(text, message_type="success")

    def on_agent_question_asked(self, event: AgentQuestionAsked) -> None:
        if self._activity_feed is not None:
            self._activity_feed.add_message(f"Question: {event.question}", message_type="info")
        with contextlib.suppress(Exception):
            self.query_one("#question_panel", QuestionPanel).show_question(
                event.question, event.choices
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        panel = None
        with contextlib.suppress(Exception):
            panel = self.query_one("#question_panel", QuestionPanel)
        if panel is None or not panel.display:
            return
        # Only the panel's OWN chips may produce a reply. Button presses from
        # elsewhere (e.g. a modal's buttons bubbling after being handled)
        # must never answer a pending question.
        buttons = list(panel.query(Button).nodes)
        if event.button not in buttons:
            return
        self.select_question_choice(buttons.index(event.button))

    # -- Agent-state reactive -------------------------------------------------

    def watch_agent_state(self, state: str) -> None:
        with contextlib.suppress(Exception):
            user_input = self.query_one("#user_input", CommandInput)
            placeholder, disabled = _STATE_INPUT_TABLE.get(state, ("", False))
            user_input.placeholder = placeholder
            user_input.disabled = disabled
            if state != "idle":
                user_input.focus()

        self._update_status_bar()

        if state != "running":
            with contextlib.suppress(Exception):
                self.query_one("#header_spinner", Static).update("")

        if state != "awaiting_input":
            with contextlib.suppress(Exception):
                self.query_one("#question_panel", QuestionPanel).dismiss()

    # -- Lifecycle: splash -> main interface ---------------------------------

    def compose(self) -> ComposeResult:
        if self.show_splash:
            yield SplashScreen(
                id="splash_screen", app_name=self._app_title, banner=self._banner
            )

    def watch_show_splash(self, show_splash: bool) -> None:
        if show_splash or not self.is_running:
            return
        with contextlib.suppress(Exception):
            self.query_one("#splash_screen").remove()
        self.set_timer(_BUILD_UI_DELAY, self._build_main_interface)

    async def _build_main_interface(self) -> None:
        feed = ActivityFeed(id="activity_feed", can_focus=False)
        self._activity_feed = feed

        container = Vertical(
            self._create_header(),
            Horizontal(
                VerticalScroll(feed, id="content_area", can_focus=False),
                BuildPanel(id="build_panel"),
                id="main_row",
            ),
            QuestionPanel(id="question_panel"),
            CommandInput(
                commands=list(SlashCommandProcessor.COMMANDS),
                placeholder="Waiting for agent to start...",
                id="user_input",
                disabled=True,
            ),
            self._create_status_bar(),
            id="main_container",
        )
        await self.mount(container)

        self._anim_timer = self.set_interval(_ANIMATION_INTERVAL, self._tick_animation)
        self.call_later(self._start_agent)

    def _create_header(self) -> Horizontal:
        from janus import __version__

        info = Static(
            f"[bold {PRIMARY}]{escape_markup(self._app_title)}[/] [dim]v{__version__}[/]",
            id="header_info",
        )
        spinner = Static("", id="header_spinner")
        timer = Static("00:00:00", id="header_timer")
        return Horizontal(info, spinner, timer, id="header")

    def _create_status_bar(self) -> Static:
        return Static(self._build_status_text(0), id="status_bar")

    # -- Status bar -----------------------------------------------------------

    def _build_status_text(self, anim_step: int = 0) -> Text:
        text = Text()
        if self.agent_state == "running" and anim_step > 0:
            text.append(get_frame(STATUS_ANIMATION, anim_step), style=PRIMARY)
            text.append(" ")

        text.append("Agent: ", style="dim")
        label, style = _STATUS_LABELS.get(self.agent_state, ("Unknown", "dim"))
        text.append(label, style=style)

        text.append("  |  ", style="dim")
        for key, hint in (("Ctrl+P", "Pause"), ("F1", "Help"), ("Ctrl+Q", "Quit")):
            text.append(key, style=f"bold {SECONDARY}")
            text.append(f" {hint}  ", style="dim")

        return text

    def _update_status_bar(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#status_bar", Static).update(self._build_status_text(0))

    def _update_status_animation(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#status_bar", Static).update(self._build_status_text(self._anim_step))

    # -- Shared animation ticking ---------------------------------------------

    def _tick_animation(self) -> None:
        self._anim_step += 1
        if self.agent_state == "running":
            self._update_header_spinner()
            self._update_status_animation()

    def _update_header_spinner(self) -> None:
        with contextlib.suppress(Exception):
            self._header_spinner_obj.advance()
            self.query_one("#header_spinner", Static).update(self._header_spinner_obj.render())

    # -- Mount / splash ---------------------------------------------------------

    def on_mount(self) -> None:
        self._events = self._controller.events
        self._setup_event_handlers()
        self.set_timer(_SPLASH_HIDE_DELAY, self._hide_splash)

    def _hide_splash(self) -> None:
        self.show_splash = False

    # -- Agent thread -----------------------------------------------------------

    def _start_agent(self) -> None:
        if self._activity_feed is not None:
            self._activity_feed.add_message("Initializing...", message_type="info")

        if hasattr(self._controller, "enable_user_replies"):
            self._controller.enable_user_replies()

        # `group` positional is None; the run worker is passed positionally.
        self._agent_thread = threading.Thread(None, self._run_agent, daemon=True)
        self._agent_thread.start()

    def _run_agent(self) -> None:
        """Run worker body. Runs entirely on a background daemon thread."""
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._controller.run(self._task_text, resume_session_id=self._resume_session_id)
            )

            status = result.get("status", "unknown")
            cost = result.get("cost_usd", 0.0)
            session_id = result.get("session_id", "")
            if status == "error":
                msg_type = "error"
            elif status == "completed":
                msg_type = "success"
            else:
                msg_type = "warning"
            self.post_message(
                AgentMessageReceived(
                    f"Run {status} | Cost: ${cost:.4f} | Session: {session_id}",
                    msg_type,
                )
            )
        except Exception as e:  # final backstop, surfaced to the UI
            self.post_message(AgentMessageReceived(f"Agent error: {e}", "error"))
        except BaseException:
            # SystemExit/KeyboardInterrupt etc surfacing on a daemon thread:
            # nothing useful to do but let the thread die quietly.
            pass
        finally:
            if loop is not None:
                with contextlib.suppress(Exception):
                    loop.close()
            with contextlib.suppress(Exception):
                if self._controller is not None and self._controller.state in (
                    AgentState.RUNNING,
                    AgentState.PAUSED,
                ):
                    self.post_message(AgentStateChanged("error", "Agent thread terminated"))

    # -- Input routing ------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "user_input":
            return

        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if isinstance(event.input, CommandInput):
            event.input.add_to_history(text)

        if text.startswith("/"):
            self._command_processor.process(text)
            return

        if self.agent_state == "awaiting_input":
            if self._events is not None:
                self._events.emit_input(text)
            self._show_feed_message(f"Reply sent: {text[:80]}", "info")
            return

        if self.agent_state == "paused":
            if self._events is not None:
                self._events.emit_input(text)
            self._show_feed_message(f"Instruction queued: {text[:80]}", "info")
            return

        if self.agent_state == "running":
            self._show_feed_message(
                "Agent is running. Use /pause first, or /instruction <text> to inject directly.",
                "warning",
            )

    # -- Question-panel selection -------------------------------------------

    def select_question_choice(self, index: int) -> bool:
        """Answer the pending question with the choice at ``index`` (0-based).

        The single implementation behind every path that can answer a
        pending question: the panel's own chip buttons
        (:meth:`on_button_pressed`) and the reply box's digit-key shortcut
        (``CommandInput.on_key``, which calls this via ``app``). Delivers
        the choice text through the same USER_INPUT path as a typed reply.

        Returns True if a reply was sent, False if there is no pending
        question or ``index`` is out of range (fails closed).
        """
        panel = None
        with contextlib.suppress(Exception):
            panel = self.query_one("#question_panel", QuestionPanel)
        if panel is None or not panel.display:
            return False

        choices = panel.current_choices
        if not (0 <= index < len(choices)):
            return False

        choice = choices[index]
        if self._events is not None:
            self._events.emit_input(choice)
        self._show_feed_message(f"Reply sent: {choice}", "info")
        panel.dismiss()
        return True

    # -- Key actions --------------------------------------------------------

    def action_toggle_pause(self) -> None:
        if self.show_splash:
            return
        if self.agent_state == "paused":
            self._controller.resume()
        elif self.agent_state == "running":
            self._controller.pause()

    def action_toggle_help(self) -> None:
        if self.show_splash:
            return
        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
        else:
            self.push_screen(HelpScreen())

    def action_request_quit(self) -> None:
        if self.show_splash:
            self.exit()
            return
        self.push_screen(QuitScreen())

    # -- Shutdown -------------------------------------------------------------

    def on_unmount(self) -> None:
        """Best-effort teardown. Each step is independent -- one exploding
        step must never prevent the rest from running."""
        with contextlib.suppress(Exception):
            if self._anim_timer is not None:
                self._anim_timer.stop()

        with contextlib.suppress(Exception):
            if self._controller is not None:
                self._controller.stop()

        with contextlib.suppress(Exception):
            if self._agent_thread is not None and self._agent_thread.is_alive():
                self._agent_thread.join(timeout=2.0)

        with contextlib.suppress(Exception):
            self._teardown_event_handlers()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_tui(
    controller: AgentController,
    task: str,
    *,
    title: str | None = None,
    banner: str | None = None,
    resume_session_id: str | None = None,
) -> None:
    """Construct and run the Janus TUI."""
    app = JanusApp(
        controller,
        task,
        title=title,
        banner=banner,
        resume_session_id=resume_session_id,
    )
    await app.run_async()
