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

from __future__ import annotations

import abc
import dataclasses
import enum
from collections.abc import AsyncIterator
from typing import Any


class MessageType(enum.Enum):
    """Kinds of message an `AgentBackend` can emit from `receive_messages()`."""

    TEXT = "text"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    OUTPUT = "output"
    RESULT = "result"
    ERROR = "error"
    AWAITING_INPUT = "awaiting_input"


@dataclasses.dataclass
class AgentMessage:
    """A single backend-agnostic message translated from the underlying SDK."""

    type: MessageType
    content: Any
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class AgentBackend(abc.ABC):
    """Abstract interface the rest of the app drives an LLM agent through."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the underlying agent session."""
        raise NotImplementedError

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Tear down the underlying agent session, if any."""
        raise NotImplementedError

    @abc.abstractmethod
    async def query(self, prompt: str) -> None:
        """Send a prompt to the connected agent session."""
        raise NotImplementedError

    @abc.abstractmethod
    async def receive_messages(self) -> AsyncIterator[AgentMessage]:
        """Stream translated `AgentMessage`s from the agent session."""
        # The `if False: yield` makes this an async-generator function (as
        # opposed to a plain coroutine) even though the body never actually
        # runs -- concrete subclasses override it entirely.
        if False:  # pragma: no cover - unreachable, just fixes the func kind
            yield
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def session_id(self) -> str | None:
        """The backend's current session id, or None if not established."""
        raise NotImplementedError

    @abc.abstractmethod
    async def resume(self, session_id: str) -> bool:
        """Resume a previously established session by id."""
        raise NotImplementedError

    @property
    def supports_resume(self) -> bool:
        """Whether this backend implementation can resume a prior session."""
        return False

    def hold(self) -> None:  # noqa: B027
        """Request the agent loop pause before its next model call.

        Base implementation is a no-op; backends with a gated loop override.
        """
        pass

    def release(self) -> None:  # noqa: B027
        """Release a :meth:`hold` so the agent loop proceeds. No-op by default."""
        pass
