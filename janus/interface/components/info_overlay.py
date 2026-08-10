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

"""Reusable floating info overlay dismissed with Escape or q."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class InfoOverlay(ModalScreen[None]):
    """Reusable floating info panel dismissed with Escape or q."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "dismiss_overlay", "Close", priority=True),
        Binding("q", "dismiss_overlay", "Close", priority=True),
    ]

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self._title = title
        self._content = content

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(self._title, id="info_title"),
            VerticalScroll(
                Static(self._content, id="info_body"),
                id="info_scroll",
            ),
            id="info_dialog",
        )

    def action_dismiss_overlay(self) -> None:
        """Close the overlay."""
        self.app.pop_screen()
