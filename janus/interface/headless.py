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

"""Dependency-free plaintext renderer for headless (non-TTY) runs.

Mirrors the domain-neutral event vocabulary consumed by the Textual TUI
(``janus.interface.tui``) but prints one concise line per event to stdout
instead of driving a terminal UI. This module must never import
``textual`` — it is the fallback path used whenever a terminal isn't
available or the ``[tui]`` extra isn't installed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Any

from janus.core.events import Event, EventBus, EventType


def _snippet(value: Any, limit: int = 80) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _translate_reply(line: str, choices: list[str]) -> str:
    """Map a bare menu number to its choice text; pass everything else through."""
    stripped = line.strip()
    if stripped.isdigit() and choices:
        n = int(stripped)
        if 1 <= n <= len(choices):
            return choices[n - 1]
    return line


def _handle_stdin_line(line: str, renderer: HeadlessRenderer, controller: Any) -> None:
    """Translate one stdin line against the last-seen question menu and
    emit it as USER_INPUT.

    Once a translation actually replaces a bare menu number with its choice
    text, the menu it was resolved against is stale — clear
    ``renderer.last_choices`` so a later bare-number line (e.g. during a
    RUNNING state with no pending question) is never silently rewritten by
    that same menu again.
    """
    if not line.strip():
        return
    translated = _translate_reply(line, renderer.last_choices)
    if translated != line:
        renderer.last_choices = []
    controller.events.emit_input(translated.strip())


class HeadlessRenderer:
    """Subscribes to the event bus and prints one line per event."""

    def __init__(self, events: EventBus | None = None) -> None:
        self.events = events or EventBus()
        self.last_choices: list[str] = []

    def attach(self) -> None:
        self.events.subscribe(EventType.STATE_CHANGED, self._on_state)
        self.events.subscribe(EventType.MESSAGE, self._on_message)
        self.events.subscribe(EventType.TOOL, self._on_tool)
        self.events.subscribe(EventType.OUTPUT, self._on_output)

    def detach(self) -> None:
        self.events.unsubscribe(EventType.STATE_CHANGED, self._on_state)
        self.events.unsubscribe(EventType.MESSAGE, self._on_message)
        self.events.unsubscribe(EventType.TOOL, self._on_tool)
        self.events.unsubscribe(EventType.OUTPUT, self._on_output)

    def _on_state(self, event: Event) -> None:
        try:
            state = event.data.get("state", "")
            details = event.data.get("details", "")
            line = f"[state] {state}"
            if details:
                line += f" {details}"
            print(line)
        except Exception:
            pass

    def _on_message(self, event: Event) -> None:
        try:
            msg_type = event.data.get("type", "info")
            text = event.data.get("text", "")
            if msg_type == "question":
                self.last_choices = list(event.data.get("choices") or [])
                print(f"[question] {text}")
                for i, choice in enumerate(self.last_choices, start=1):
                    print(f"  {i}) {choice}")
                return
            print(f"[{msg_type}] {text}")
        except Exception:
            pass

    def _on_tool(self, event: Event) -> None:
        try:
            status = event.data.get("status", "")
            name = event.data.get("name", "")
            if status == "result":
                snippet = _snippet(event.data.get("result"))
            else:
                snippet = _snippet(event.data.get("args"))
            print(f"[tool:{status}] {name} {snippet}")
        except Exception:
            pass

    def _on_output(self, event: Event) -> None:
        try:
            keys = ", ".join(sorted(event.data.keys()))
            print(f"[output] {keys}")
        except Exception:
            pass


def run_headless(controller: Any, task: str, *, resume_session_id: str | None = None) -> dict:
    """Run ``controller.run(task, ...)`` to completion, printing plaintext status.

    When stdin is a TTY, the run is interactive: replies are enabled and a
    daemon thread forwards stdin lines onto the bus as USER_INPUT (the
    controller routes them to reply() while awaiting, inject() otherwise).
    Piped runs stay non-interactive so ask_user fails open.
    """
    renderer = HeadlessRenderer(controller.events)
    renderer.attach()

    interactive = sys.stdin.isatty()
    stop = threading.Event()
    if interactive and hasattr(controller, "enable_user_replies"):
        controller.enable_user_replies()

        def _stdin_pump() -> None:
            # EOFError and OSError (stdin closed, redirected to an invalid
            # handle, or unavailable in a daemonized/captured context) are
            # both normal "no more input" conditions for a best-effort pump.
            while not stop.is_set():
                try:
                    line = input()
                except (EOFError, OSError):
                    return
                if stop.is_set():
                    return
                _handle_stdin_line(line, renderer, controller)

        threading.Thread(target=_stdin_pump, daemon=True).start()

    try:
        result = asyncio.run(controller.run(task, resume_session_id=resume_session_id))
        print(f"status: {result['status']}")
        if result.get("session_id"):
            print(f"session: {result['session_id']}")
        return result
    finally:
        renderer.detach()
        # The pump thread is a daemon and, once run() has returned, nobody
        # joins it — a daemon thread parked in input() does not by itself
        # block interpreter shutdown. Setting the flag here is belt-and-
        # braces: it lets a pump that wakes up between reads notice the run
        # is over and stop forwarding stale input instead of racing the
        # detached renderer, without ever blocking this return on the
        # still-blocked-on-input() thread.
        stop.set()
