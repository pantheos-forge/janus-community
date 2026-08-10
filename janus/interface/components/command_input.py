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

"""Command input widget with tab-completion and history for slash commands."""

from __future__ import annotations

from collections.abc import Callable

from textual import events
from textual.suggester import Suggester
from textual.widgets import Input


class SlashCommandSuggester(Suggester):
    """Inline suggestion engine for slash commands.

    Shows a grayed-out suffix as the user types (e.g. `/he` shows `lp`).
    """

    def __init__(self, commands: list[str]) -> None:
        super().__init__(use_cache=True, case_sensitive=False)
        self._commands = sorted(commands)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        partial = value[1:].lower()
        if not partial:
            return None
        for cmd in self._commands:
            if cmd.lower().startswith(partial):
                return f"/{cmd}"
        return None


def complete_command(value: str, commands: list[str]) -> str | None:
    """Tab-complete a slash command value against sorted command list.

    Returns the completed string, or None if no completion applies.
    Single match: ``/cmd`` (no trailing space).
    Multiple matches: longest common prefix.
    """
    if not value.startswith("/"):
        return None

    partial = value[1:].lower()
    matches = [c for c in commands if c.lower().startswith(partial)]
    if not matches:
        return None

    if len(matches) == 1:
        return f"/{matches[0]}"

    # Longest common prefix
    prefix = matches[0]
    for m in matches[1:]:
        while not m.lower().startswith(prefix.lower()):
            prefix = prefix[:-1]
    completed = f"/{prefix}"
    return completed if completed != value else None


class CommandHistory:
    """In-memory command history with up/down navigation."""

    MAX_ENTRIES = 50

    def __init__(self) -> None:
        self._entries: list[str] = []
        self._index: int = -1
        self._draft: str = ""

    @property
    def entries(self) -> list[str]:
        return self._entries

    def add(self, text: str) -> None:
        """Record a submitted input (deduped against last entry, capped)."""
        if not text:
            return
        if self._entries and self._entries[-1] == text:
            return
        self._entries.append(text)
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES :]
        self._index = -1

    def navigate_up(self, current_value: str) -> str | None:
        """Move to an older history entry. Returns new value or None."""
        if not self._entries:
            return None
        if self._index == -1:
            self._draft = current_value
            self._index = len(self._entries) - 1
        elif self._index > 0:
            self._index -= 1
        else:
            return None  # already at oldest
        return self._entries[self._index]

    def navigate_down(self) -> str | None:
        """Move to a newer history entry or restore draft. Returns new value or None."""
        if self._index == -1:
            return None  # not in history mode
        if self._index < len(self._entries) - 1:
            self._index += 1
            return self._entries[self._index]
        # Past newest — restore draft
        self._index = -1
        return self._draft

    def reset(self) -> None:
        """Exit history navigation (call on any non-history key)."""
        self._index = -1

    # -- Thin aliases: pure up/down navigation without a "current draft"
    # value, for callers (and the generic-components test contract) that
    # don't track an in-progress input line. --

    def previous(self) -> str | None:
        """Alias for `navigate_up` using the empty string as the draft."""
        return self.navigate_up(self._draft)

    def next(self) -> str | None:
        """Alias for `navigate_down`."""
        return self.navigate_down()


class CommandInput(Input):
    """Input widget with tab-completion and up/down history for slash commands."""

    def __init__(
        self,
        commands: list[str],
        *,
        placeholder: str = "",
        id: str | None = None,
        disabled: bool = False,
    ) -> None:
        self._commands = sorted(commands)
        self._history = CommandHistory()
        suggester = SlashCommandSuggester(commands)
        super().__init__(
            placeholder=placeholder,
            id=id,
            disabled=disabled,
            suggester=suggester,
        )

    def add_to_history(self, text: str) -> None:
        """Record a submitted input line."""
        self._history.add(text)

    def on_key(self, event: events.Key) -> None:
        if event.key == "tab":
            self._handle_tab(event)
        elif event.key == "up":
            self._handle_up(event)
        elif event.key == "down":
            self._handle_down(event)
        elif event.character and event.character.isdigit() and not self.value:
            self._handle_digit(event)
        else:
            self._history.reset()

    def _handle_digit(self, event: events.Key) -> None:
        """Delegate a bare digit typed into an empty box to the nearest
        ancestor's pending-question selection, iff one is pending.

        Textual dispatches handlers from most- to least-derived class for
        the same widget, and ``Message.prevent_default()`` short-circuits
        that walk (see ``MessagePump._get_dispatch_methods``): calling it
        here — before Textual's base ``Input._on_key`` runs — is what stops
        the digit from also being inserted as text. (``Input._on_key`` also
        ``event.stop()``s printable keys, so a bare digit never bubbles to a
        parent widget — hence the selection target must be resolved here, at
        the input, not in an ancestor ``on_key`` handler.)

        The target is the closest ancestor exposing ``select_question_choice``:
        a mounted :class:`~janus.interface.components.session_view.SessionView`
        (fleet dashboard) or the :class:`~janus.interface.tui.JanusApp` itself
        (single session) — the App is included in ``ancestors``. Only consumed
        when that method reports a real selection; otherwise the key is left
        alone and falls through to normal text insertion, so digits typed with
        no pending question behave exactly as before.
        """
        assert event.character is not None
        select = self._resolve_choice_selector()
        if select is None or not select(int(event.character) - 1):
            return
        event.stop()
        event.prevent_default()

    def _resolve_choice_selector(self) -> Callable[[int], bool] | None:
        """Nearest ancestor's ``select_question_choice`` (``ancestors`` runs
        child->App, so the closest enclosing session widget wins)."""
        for node in self.ancestors:
            select = getattr(node, "select_question_choice", None)
            if callable(select):
                return select
        return None

    def _handle_tab(self, event: events.Key) -> None:
        event.prevent_default()
        event.stop()
        result = complete_command(self.value, self._commands)
        if result is not None:
            self.value = result
            self.cursor_position = len(self.value)

    def _handle_up(self, event: events.Key) -> None:
        event.prevent_default()
        event.stop()
        result = self._history.navigate_up(self.value)
        if result is not None:
            self.value = result
            self.cursor_position = len(self.value)

    def _handle_down(self, event: events.Key) -> None:
        event.prevent_default()
        event.stop()
        result = self._history.navigate_down()
        if result is not None:
            self.value = result
            self.cursor_position = len(self.value)
