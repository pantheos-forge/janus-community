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

"""Animated Textual splash screen shown at Janus startup.

Renders a braille-art banner of a Janus coin medallion — the two-faced
janiform head in negative space inside a lit roundel — alongside the app
name, title, tagline, and version, plus a small looping spinner/progress-bar
animation driven by a repeating Textual timer.

The banner is derived from "Janus, from Illustrium imagines" (1517 woodcut),
Wikimedia Commons, public domain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.align import Align
from rich.console import Group
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from janus.interface.components.animations import (
    SPLASH_BAR,
    SPLASH_SPINNER,
    get_frame,
)
from janus.interface.theme import PRIMARY, SECONDARY, TEXT_DIM, TEXT_SECONDARY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from textual.timer import Timer

_VERSION = "0.1.0"

_BANNER: str = """⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⠶⣖⣛⣿⣿⣿⣟⣛⣲⡶⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⣠⢖⣫⣵⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣝⡶⣄⡀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⡴⢻⣷⣿⣿⣿⠿⢛⣋⣭⣥⣤⢤⣭⣭⣝⡛⠿⣿⣿⣿⣮⡻⣦⡀⠀⠀⠀
⠀⠀⢠⢞⣴⣿⣿⡿⢋⣥⣚⣍⡝⢏⠋⢛⢸⢫⣻⢛⠩⣓⢤⡙⢿⣿⣿⣮⣳⣄⠀⠀
⠀⢠⢏⣾⣿⣿⢏⣴⣿⣿⡥⣉⡴⠗⢈⡬⣄⡌⢒⢓⡥⠦⣴⣿⣦⡙⣿⣿⣷⣹⣆⠀
⢀⡏⣿⣿⣿⠃⣾⣿⣿⣿⠀⠀⠉⢲⡉⡁⡇⢱⡅⡁⠀⡀⢹⣿⣿⣷⡘⣿⣿⣷⢻⡄
⣸⢹⣿⣿⡏⣸⣿⣿⣿⣿⠘⡼⠓⠀⠹⣟⠳⡉⠀⠐⢸⠇⠙⣿⣿⣿⣇⢹⣿⣿⡏⣷
⣿⣼⣿⣿⠁⣿⣿⣿⣿⣃⣠⡀⠀⠀⠀⡋⣰⣄⡠⠀⠀⢠⣀⣹⣿⣿⣿⠸⣿⣿⣧⣿
⣿⢻⣿⣿⠀⣿⣿⣿⣿⣿⡴⢤⢀⣀⣗⠀⢣⡏⢄⠷⢄⣡⢤⣿⣿⣿⣿⢠⣿⣿⣿⣿
⢻⢸⣿⣿⡇⢹⣿⣿⣿⡿⠙⠀⢄⠃⣋⡼⠁⠙⢶⡀⠺⠉⡹⣿⣿⣿⡟⣸⣿⣿⣧⡿
⠘⣇⢿⣿⣿⡈⢻⣿⣿⣧⣯⣴⣼⠃⠁⠀⠀⠀⠈⠙⣶⣤⣽⣿⣿⡿⢠⣿⣿⡿⣸⠃
⠀⠘⣎⢿⣿⣷⣄⠻⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⠟⣰⣿⣿⡿⣱⠇⠀
⠀⠀⠘⢮⠻⣿⣿⣦⡈⠛⢿⣿⣤⣬⣭⣥⣬⣭⣭⣶⣿⡿⠛⣡⣾⣿⣿⢟⡵⠃⠀⠀
⠀⠀⠀⠈⠳⣌⠻⣿⣿⣷⣦⣌⣉⡛⠛⠛⠛⠛⣛⣉⣥⣶⣿⣿⣿⠟⣡⠞⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⠑⠮⣝⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢟⣫⠵⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠶⠭⢭⣍⣛⣛⣫⡭⠭⠖⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀"""

_TAGLINE = "AI agent runtime"


class SplashScreen(Static):
    """Animated startup banner widget.

    Displays a braille-art banner of a janiform Janus coin medallion, the app
    name, title, version, and tagline, along with a looping spinner +
    progress-bar animation that advances on a 0.1s Textual timer.
    """

    BANNER: str = _BANNER

    PRIMARY_COLOR = PRIMARY
    SECONDARY_COLOR = SECONDARY

    def __init__(
        self,
        *args: Any,
        app_name: str = "Janus",
        banner: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize animation/render state.

        Args:
            *args: Forwarded to ``Static.__init__``.
            app_name: Name of the application shown on the banner. Defaults
                to ``"Janus"``.
            banner: Custom braille banner art. ``None`` uses the default
                Janus medallion.
            **kwargs: Forwarded to ``Static.__init__``.
        """
        super().__init__(*args, **kwargs)
        self.app_name = app_name
        self._banner = banner or self.BANNER
        self._animation_step: int = 0
        self._animation_timer: Timer | None = None
        self._panel_static: Static | None = None
        self._version: str = _VERSION

    def compose(self) -> Iterator[Static]:
        """Build and yield the single content widget at step 0."""
        self._animation_step = 0
        spinner_line, bar_line = self._build_loading_text(self._animation_step)
        content = self._build_content(spinner_line, bar_line)
        self._panel_static = Static(content, id="splash_content")
        yield self._panel_static

    def on_mount(self) -> None:
        """Start the repeating animation timer."""
        self._animation_timer = self.set_interval(0.1, self._animate_loading)

    def on_unmount(self) -> None:
        """Stop and clear the animation timer, if one is running."""
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

    def _animate_loading(self) -> None:
        """Advance the animation by one step and refresh the panel."""
        if self._panel_static is None:
            return
        self._animation_step += 1
        spinner_line, bar_line = self._build_loading_text(self._animation_step)
        content = self._build_content(spinner_line, bar_line)
        self._panel_static.update(content)

    def _build_loading_text(self, step: int) -> tuple[Text, Text]:
        """Build the spinner and progress-bar lines for a given step.

        Args:
            step: The current animation step.

        Returns:
            A ``(spinner_line, bar_line)`` tuple of ``rich.text.Text`` objects.
        """
        spinner_frame = get_frame(SPLASH_SPINNER, step)
        bar_frame = get_frame(SPLASH_BAR, step)

        spinner_line = Text()
        spinner_line.append(spinner_frame, style=Style(color=self.SECONDARY_COLOR))
        spinner_line.append(
            " Initializing ", style=Style(color=self.PRIMARY_COLOR, bold=True)
        )
        spinner_line.append(spinner_frame, style=Style(color=self.SECONDARY_COLOR))

        bar_line = Text()
        bar_line.append(bar_frame, style=Style(color=self.SECONDARY_COLOR))

        return spinner_line, bar_line

    def _build_content(self, spinner_line: Text, bar_line: Text) -> Group:
        """Assemble the full centered splash layout.

        Args:
            spinner_line: The spinner line built by ``_build_loading_text``.
            bar_line: The progress-bar line built by ``_build_loading_text``.

        Returns:
            A ``rich.console.Group`` of 12 center-aligned lines.
        """
        banner_text = Text(justify="center")
        banner_text.append(self._banner, style=self.PRIMARY_COLOR)

        name_text = Text(
            " ".join(self.app_name.upper()), style=Style(color=self.PRIMARY_COLOR, bold=True)
        )

        title_text = Text()
        title_text.append(self.app_name, style=Style(color=self.PRIMARY_COLOR, bold=True))
        title_text.append(" — Agent Runtime", style=Style(color=TEXT_SECONDARY))

        version_text = Text(f"v{self._version}", style=Style(color=TEXT_DIM, dim=True))

        tagline_text = Text(_TAGLINE, style=Style(color=self.SECONDARY_COLOR, italic=True))

        return Group(
            Align.center(banner_text),
            Align.center(Text(" ")),
            Align.center(name_text),
            Align.center(Text(" ")),
            Align.center(title_text),
            Align.center(version_text),
            Align.center(Text(" ")),
            Align.center(tagline_text),
            Align.center(Text(" ")),
            Align.center(spinner_line),
            Align.center(Text(" ")),
            Align.center(bar_line),
        )

    def render_banner_text(self) -> str:
        """Return the plain-text banner, app name, and tagline.

        Used by tests (and safe to call outside a running Textual app) to
        verify the rendered splash content without needing a live console.
        """
        lines = [
            self.app_name.upper(),
            self.app_name,
            _TAGLINE,
        ]
        return "\n".join(lines)
