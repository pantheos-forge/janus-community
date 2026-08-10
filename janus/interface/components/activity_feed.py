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

"""Live activity feed widget for the Janus TUI.

Renders agent messages and tool executions as a scrolling Rich-markup feed,
including an animated spinner for in-progress tool calls.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.content import Content
from textual.markup import escape as _textual_escape
from textual.widgets import Static

from janus.interface.components.animations import FEED_SPINNERS, AnimationType, get_frame
from janus.interface.theme import ERROR, PRIMARY, SUCCESS, WARNING

if TYPE_CHECKING:
    from textual.timer import Timer

_MAX_ACTIVITIES = 200


def escape_markup(text: str) -> str:
    """Escape Textual console-markup special characters in ``text``.

    The feed's widgets render through Textual's ``Content.from_markup``, so
    content must be escaped in Textual's dialect (not Rich's). Returns a plain
    ``str``.
    """
    return str(_textual_escape(text))


class SpinnerWidget(Static):
    """A ``Static`` that animates a spinner glyph inside a fixed markup template.

    The template contains a literal ``{spinner}`` placeholder that gets
    replaced with the current animation frame on every tick.
    """

    def __init__(self, markup_template: str, anim_type: AnimationType, **kwargs: Any) -> None:
        """Store the template/animation and validate the template renders as markup."""
        probe = markup_template.replace("{spinner}", "⠀")
        try:
            Content.from_markup(probe)
            markup_ok = True
        except Exception:
            markup_ok = False

        super().__init__(markup=markup_ok, **kwargs)
        self._template = markup_template
        self._anim_type = anim_type
        self._markup_ok = markup_ok
        self._step = 0
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        """Start the animation timer."""
        self._timer = self.set_interval(0.1, self._tick)

    def on_unmount(self) -> None:
        """Stop the animation timer if it is running."""
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        """Advance the animation by one frame and update the rendered content."""
        self._step += 1
        frame = get_frame(self._anim_type, self._step)
        replacement = f"[{WARNING}]{frame}[/]" if self._markup_ok else frame
        content = self._template.replace("{spinner}", replacement)
        try:
            self.update(content)
        except Exception:
            if self._timer is not None:
                self._timer.stop()
            with contextlib.suppress(Exception):
                self.update(self._template.replace("{spinner}", frame))


class ActivityFeed(VerticalScroll):
    """Scrolling feed of agent messages and tool-execution activity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the feed with an empty activity count and no placeholder."""
        super().__init__(*args, **kwargs)
        self._count = 0
        self._placeholder: Static | None = None

    def compose(self) -> ComposeResult:
        """Yield the initial placeholder shown before any activity arrives."""
        placeholder = Static(
            "\n\n[dim italic]Waiting for agent to start...[/]\n\n"
            "[dim]The activity feed will show real-time updates here.[/]",
            id="activity_content",
            classes="placeholder",
        )
        self._placeholder = placeholder
        yield placeholder

    @staticmethod
    def _render_message(message: str, message_type: str, timestamp: datetime) -> str:
        """Render a plain feed message as a Rich-markup string."""
        ts = timestamp.strftime("%H:%M:%S")
        safe = escape_markup(message)

        if message_type == "success":
            icon = f"[{SUCCESS}]✓[/]"
            style = SUCCESS
        elif message_type == "error":
            icon = f"[{ERROR}]✗[/]"
            style = ERROR
        elif message_type == "warning":
            icon = f"[{WARNING}]⚠[/]"
            style = ""
        else:
            icon = f"[{PRIMARY}]●[/]"
            style = PRIMARY

        if style:
            return f"[dim]{ts}[/] {icon} [{style}]{safe}[/]"
        return f"{safe}"

    @staticmethod
    def _render_tool(
        tool_name: str,
        args: dict[str, Any],
        status: str,
        result: Any,
        timestamp: datetime,
    ) -> str:
        """Render a completed/failed/running (non-animated) tool activity."""
        ts = timestamp.strftime("%H:%M:%S")

        if status == "running":
            status_icon = f"[{WARNING}]●[/] In progress..."
        elif status == "completed":
            status_icon = f"[{SUCCESS}]✓[/] Done"
        elif status == "failed":
            status_icon = f"[{ERROR}]✗[/] Failed"
        else:
            status_icon = "[dim]○[/] Unknown"

        lines = [f"[dim]{ts}[/]", f"[bold]▍ {escape_markup(tool_name)}[/] {status_icon}"]

        if args:
            for key, value in list(args.items())[:3]:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                lines.append(f"  [dim]{key}:[/] {escape_markup(value_str)}")

        if status in ("completed", "failed") and result:
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:197] + "..."
            lines.append(f"  [dim]→[/] {escape_markup(result_str)}")

        return "\n".join(lines)

    @staticmethod
    def _render_tool_animated(tool_name: str, args: dict[str, Any], timestamp: datetime) -> str:
        """Render the markup template for a running tool with a spinner placeholder."""
        ts = timestamp.strftime("%H:%M:%S")

        header = f"[bold]▍ {escape_markup(tool_name)}[/] {{spinner}} In progress..."
        lines = [f"[dim]{ts}[/]", header]

        if args:
            for key, value in list(args.items())[:3]:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                lines.append(f"  [dim]{key}:[/] {escape_markup(value_str)}")

        return "\n".join(lines)

    def add_message(
        self,
        message: str,
        message_type: str = "info",
        timestamp: datetime | None = None,
    ) -> None:
        """Append a plain feed message."""
        if timestamp is None:
            timestamp = datetime.now()
        content = self._render_message(message, message_type, timestamp)
        self._mount_activity(content)

    def add_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        status: str = "running",
        result: Any = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a tool-execution activity, animated while running."""
        if timestamp is None:
            timestamp = datetime.now()

        if status == "running":
            template = self._render_tool_animated(tool_name, args, timestamp)
            anim_type = FEED_SPINNERS[self._count % len(FEED_SPINNERS)]
            widget = SpinnerWidget(template, anim_type)
            self._mount_activity(widget)
        else:
            content = self._render_tool(tool_name, args, status, result, timestamp)
            self._mount_activity(content)

    def clear(self) -> None:
        """Remove all mounted activity widgets and reset the count."""
        for child in list(self.children):
            child.remove()
        self._count = 0

    def _mount_activity(self, content: str | Static) -> None:
        """Mount a new activity widget, evicting the placeholder/oldest entries as needed."""
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None

        if isinstance(content, str):
            try:
                Content.from_markup(content)
            except Exception:
                widget = Static(content, markup=False, classes="activity-item")
            else:
                widget = Static(content, classes="activity-item")
        else:
            widget = content
            widget.add_class("activity-item")

        self.mount(widget)
        self._count += 1
        if self._count > _MAX_ACTIVITIES:
            self._evict_oldest()

        self.call_later(self.scroll_end, animate=False)

    def _evict_oldest(self) -> None:
        """Remove the oldest mounted activity widget, if any."""
        if self.children:
            self.children[0].remove()
            self._count -= 1
