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

"""Janus runtime interface — Textual TUI, headless renderer, and CLI.

This package is importable without the optional ``textual`` dependency:
only submodules that actually need a terminal UI (added in later tasks)
import ``textual``. Keep this module, ``theme.py``, and ``headless.py``
free of that import so headless/CLI code paths never require the
``[tui]`` extra.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import Any

from janus.interface.headless import run_headless


def _stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def _textual_available() -> bool:
    return importlib.util.find_spec("textual") is not None


def launch(
    controller: Any,
    task: str,
    *,
    title: str | None = None,
    banner: str | None = None,
    resume_session_id: str | None = None,
) -> dict | None:
    """Run ``controller`` on ``task``, choosing the Textual TUI or headless renderer.

    Uses the Textual TUI when stdout is a TTY and ``textual`` is installed;
    otherwise falls back to the dependency-free headless renderer. Returns
    the controller's result dict for the headless path, or ``None`` when
    the TUI ran (the TUI owns its own lifecycle/output). ``banner``, when
    given, is an optional custom braille splash banner; it is ignored on
    the headless path, which has no splash screen.
    """
    if title is None:
        title = "Janus"

    use_tui = _stdout_is_tty() and _textual_available()
    if use_tui:
        from janus.interface.tui import run_tui

        asyncio.run(
            run_tui(
                controller,
                task,
                title=title,
                banner=banner,
                resume_session_id=resume_session_id,
            )
        )
        return None

    return run_headless(controller, task, resume_session_id=resume_session_id)
