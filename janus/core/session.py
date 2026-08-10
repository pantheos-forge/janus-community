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

"""File-based persistence for sessions.

A "session" tracks the state of a single unit of work (subject, task,
model, status, running cost, etc). Each session is serialized as one
pretty-printed JSON file on disk, keyed by its session id, under
``~/.janus/sessions`` by default.

``SessionStore`` provides CRUD-style operations (create/load/list/
delete) plus a small set of convenience mutators that operate on
whichever session is "current" and immediately persist the change.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class SessionStatus(Enum):
    """Lifecycle status of a session."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class SessionInfo:
    """Serializable snapshot of a single session.

    Field order and defaults are part of the public contract: this
    dataclass is round-tripped to/from JSON via :meth:`to_dict` /
    :meth:`from_dict`, and both consumers and the test oracle depend
    on the exact field names below.
    """

    session_id: str
    subject: str
    created_at: datetime
    status: SessionStatus = SessionStatus.RUNNING
    backend_session_id: str | None = None
    updated_at: datetime | None = None
    task: str = ""
    user_instructions: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    model: str = ""
    persona: str = ""
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        ``created_at``/``updated_at`` are rendered as ISO-8601 strings
        (``updated_at`` stays ``None`` if unset) and ``status`` as its
        string value; every other field is copied verbatim.
        """
        return {
            "session_id": self.session_id,
            "subject": self.subject,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "backend_session_id": self.backend_session_id,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "task": self.task,
            "user_instructions": self.user_instructions,
            "total_cost_usd": self.total_cost_usd,
            "model": self.model,
            "persona": self.persona,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionInfo:
        """Build a :class:`SessionInfo` from a dict (inverse of :meth:`to_dict`).

        Raises:
            ValueError: if any of ``session_id``, ``subject``, ``created_at``,
                or ``status`` is missing from ``data``.
        """
        required = ("session_id", "subject", "created_at", "status")
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required field: {key}")

        updated_at_raw = data.get("updated_at")
        updated_at = datetime.fromisoformat(updated_at_raw) if updated_at_raw else None

        kwargs: dict[str, Any] = {
            "session_id": data["session_id"],
            "subject": data["subject"],
            "created_at": datetime.fromisoformat(data["created_at"]),
            "status": SessionStatus(data["status"]),
            "backend_session_id": data.get("backend_session_id"),
            "updated_at": updated_at,
        }

        optional_defaults = (
            "task",
            "user_instructions",
            "total_cost_usd",
            "model",
            "persona",
            "last_error",
        )
        for key in optional_defaults:
            if key in data:
                kwargs[key] = data[key]

        return cls(**kwargs)


class SessionStore:
    """File-based store for :class:`SessionInfo` records.

    Each session lives in its own ``<session_id>.json`` file inside
    the store's directory. The store also tracks a single "current"
    session, which the convenience mutators (``update_status``,
    ``add_cost``, etc.) operate on.
    """

    SESSIONS_DIR: Path = Path.home() / ".janus" / "sessions"

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self._dir: Path = sessions_dir if sessions_dir is not None else self.SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._current: SessionInfo | None = None

    @property
    def current(self) -> SessionInfo | None:
        """The currently active session, if any."""
        return self._current

    def _path_for(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def create(
        self,
        subject: str,
        task: str = "",
        model: str = "",
        persona: str = "",
    ) -> SessionInfo:
        """Create, persist, and set as current a new session."""
        resolved_id = str(uuid.uuid4())[:8]
        session = SessionInfo(
            session_id=resolved_id,
            subject=subject,
            created_at=datetime.now(),
            task=task,
            model=model,
            persona=persona,
        )
        self._current = session
        self.save()
        return session

    def save(self) -> None:
        """Persist the current session to disk, refreshing ``updated_at``.

        No-op if there is no current session.
        """
        if self._current is None:
            return
        self._current.updated_at = datetime.now()
        path = self._path_for(self._current.session_id)
        path.write_text(json.dumps(self._current.to_dict(), indent=2))

    def load(self, session_id: str) -> SessionInfo | None:
        """Load a session by id from disk and set it as current.

        Returns ``None`` (without raising) if the file is missing,
        unreadable, or fails to parse into a valid ``SessionInfo``.
        """
        path = self._path_for(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            session = SessionInfo.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
            return None
        self._current = session
        return session

    def list_sessions(self, subject: str | None = None) -> list[SessionInfo]:
        """List all sessions, newest first, optionally filtered by subject.

        Session files that fail to parse are silently skipped.
        """
        sessions: list[SessionInfo] = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sessions.append(SessionInfo.from_dict(data))
            except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
                continue

        if subject is not None:
            sessions = [s for s in sessions if s.subject == subject]

        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def get_latest(self, subject: str | None = None) -> SessionInfo | None:
        """Return the newest session (optionally subject-filtered), or ``None``."""
        sessions = self.list_sessions(subject)
        return sessions[0] if sessions else None

    def delete(self, session_id: str) -> bool:
        """Delete a session's file; clear it as current if it was.

        Returns ``True`` if a file was deleted, ``False`` if none existed.
        """
        path = self._path_for(session_id)
        if not path.exists():
            return False
        path.unlink()
        if self._current is not None and self._current.session_id == session_id:
            self._current = None
        return True

    def update_status(self, status: SessionStatus) -> None:
        """Set the current session's status and persist."""
        if self._current is None:
            return
        self._current.status = status
        self.save()

    def add_instruction(self, instruction: str) -> None:
        """Append a user instruction to the current session and persist."""
        if self._current is None:
            return
        self._current.user_instructions.append(instruction)
        self.save()

    def set_backend_session_id(self, backend_id: str) -> None:
        """Set the current session's backend session id and persist."""
        if self._current is None:
            return
        self._current.backend_session_id = backend_id
        self.save()

    def add_cost(self, cost: float) -> None:
        """Add to the current session's running cost and persist."""
        if self._current is None:
            return
        self._current.total_cost_usd += cost
        self.save()

    def add_cost_to(self, session: SessionInfo, cost: float) -> None:
        """Add to a specific session's running cost and persist it explicitly.

        Unlike :meth:`add_cost`, this does not rely on the store's notion
        of the "current" session: callers pass the exact ``SessionInfo``
        whose cost should be updated, and it is written to disk regardless
        of whether it happens to be ``self.current``.
        """
        session.total_cost_usd += cost
        session.updated_at = datetime.now()
        path = self._path_for(session.session_id)
        path.write_text(json.dumps(session.to_dict(), indent=2))

    def set_error(self, error: str) -> None:
        """Record the current session's last error and persist."""
        if self._current is None:
            return
        self._current.last_error = error
        self.save()
