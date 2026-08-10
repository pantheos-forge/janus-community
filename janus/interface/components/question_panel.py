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

"""Question panel — renders an ask_user question and optional choice chips.

Hidden by default; JanusApp shows it on a question event and dismisses it
when the agent state leaves ``awaiting_input``. Chips deliver the choice
TEXT through the same USER_INPUT path as typing it.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static


class QuestionPanel(Vertical):
    """A dismissible panel: question text + optional quick-reply chips.

    The question text lives inside a bounded ``VerticalScroll`` so an
    arbitrarily long question (e.g. the factory presenting a full spec at
    the gate) scrolls within the panel instead of growing it past the
    viewport and clipping the chips and input box below the fold.
    """

    DEFAULT_CLASSES = "question-panel"

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static("", id="question_text"), id="question_scroll")
        yield Horizontal(id="question_chips")

    def on_mount(self) -> None:
        self.display = False

    def show_question(self, question: str, choices: list[str]) -> None:
        self.query_one("#question_text", Static).update(question)
        self.query_one("#question_scroll", VerticalScroll).scroll_home(animate=False)
        chips = self.query_one("#question_chips", Horizontal)
        chips.remove_children()
        for i, choice in enumerate(choices, start=1):
            chips.mount(Button(choice, id=f"chip_{i}"))
        self.display = True

    def dismiss(self) -> None:
        self.display = False

    @property
    def current_choices(self) -> list[str]:
        return [str(b.label) for b in self.query("Button")]
