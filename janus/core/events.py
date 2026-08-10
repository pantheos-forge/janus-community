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

"""In-process publish/subscribe event bus (domain-neutral, per-consumer instances).

Decouples the agent's core loop from its front-ends (TUI, CLI, future web).
Ported from the upstream proprietary agent's EventBus; the mechanism is
unchanged, the event
vocabulary is generic.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    STATE_CHANGED = auto()
    MESSAGE = auto()
    TOOL = auto()
    OUTPUT = auto()
    USER_COMMAND = auto()
    USER_INPUT = auto()


Handler = Callable[["Event"], None]


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.setdefault(event_type, [])
            if handler not in handlers:
                handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers is not None and handler in handlers:
                handlers.remove(handler)

    def emit(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event.type, ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                continue

    # -- Convenience emitters ------------------------------------------------
    def emit_state(
        self,
        state: str,
        details: str = "",
        subject: str | None = None,
        task: str | None = None,
        persona: str | None = None,
    ) -> None:
        data: dict[str, Any] = {"state": state, "details": details}
        if subject is not None:
            data["subject"] = subject
        if task is not None:
            data["task"] = task
        if persona is not None:
            data["persona"] = persona
        self.emit(Event(EventType.STATE_CHANGED, data))

    def emit_message(self, text: str, msg_type: str = "info") -> None:
        self.emit(Event(EventType.MESSAGE, {"text": text, "type": msg_type}))

    def emit_tool(
        self,
        status: str,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any | None = None,
    ) -> None:
        self.emit(
            Event(
                EventType.TOOL,
                {
                    "status": status,
                    "name": name,
                    "args": args or {},
                    "result": result,
                },
            )
        )

    def emit_output(self, payload: dict[str, Any]) -> None:
        self.emit(Event(EventType.OUTPUT, payload))

    def emit_command(self, command: str) -> None:
        self.emit(Event(EventType.USER_COMMAND, {"command": command}))

    def emit_input(self, text: str) -> None:
        self.emit(Event(EventType.USER_INPUT, {"text": text}))
