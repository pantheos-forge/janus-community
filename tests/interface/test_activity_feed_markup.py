"""Regression tests: the activity feed must never crash the TUI on agent output
containing Textual-markup-like bracket sequences (a real crash seen in the field).

Follows the pilot pattern from tests/interface/test_session_view.py: a tiny host
App mounts one ActivityFeed and drives its public API, asserting against mounted
widgets (never synthetic events).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Static

from janus.interface.components.activity_feed import (
    ActivityFeed,
    SpinnerWidget,
    escape_markup,
)
from janus.interface.components.animations import FEED_SPINNERS

# JSON-in-JSON with nested brackets — the shape that caused the crash.
CRASH = '[{"cve":"CVE-2026-63030"},{"a":"-2026-63030"}]'
MINIMAL = "x [a=-2026-63030]"
# Textual-unparseable in BOTH Rich and Textual escape dialects (the [[ escape gap).
BAD_MARKUP = "[[a=-2026]]"


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield ActivityFeed(id="feed")


@pytest.mark.asyncio
async def test_add_message_with_markup_like_content_does_not_crash():
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        feed = app.query_one("#feed", ActivityFeed)
        feed.add_message(CRASH)
        feed.add_message(MINIMAL, message_type="error")
        await pilot.pause(0.1)
        assert len(feed.query(".activity-item")) == 2


@pytest.mark.asyncio
async def test_add_completed_tool_with_markup_like_result_does_not_crash():
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        feed = app.query_one("#feed", ActivityFeed)
        feed.add_tool_execution(
            "web_fetch", {"url": CRASH}, status="completed", result=CRASH
        )
        await pilot.pause(0.1)
        assert len(feed.query(".activity-item")) == 1


@pytest.mark.asyncio
async def test_add_running_tool_with_markup_like_arg_does_not_crash():
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        feed = app.query_one("#feed", ActivityFeed)
        feed.add_tool_execution("web_fetch", {"url": CRASH}, status="running")
        await pilot.pause(0.25)  # let the spinner tick at least once (renders)
        assert len(feed.query(".activity-item")) == 1


@pytest.mark.asyncio
async def test_mount_activity_falls_back_to_non_markup_on_bad_markup():
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        feed = app.query_one("#feed", ActivityFeed)
        feed._mount_activity(BAD_MARKUP)      # Textual-unparseable -> literal fallback
        feed._mount_activity("[dim]clean[/]")  # valid Textual markup -> keeps markup
        await pilot.pause(0.1)
        items = list(feed.query(".activity-item").results(Static))
        assert len(items) == 2
        assert items[0]._render_markup is False  # fallback fired
        assert items[1]._render_markup is True   # normal styled path


def test_spinner_widget_probes_with_textual_dialect():
    # Rich's Text.from_markup accepts this template; Textual's Content cannot,
    # so the widget must fall back to markup off (renders the frame literally).
    bad = SpinnerWidget(f"{BAD_MARKUP} {{spinner}}", FEED_SPINNERS[0])
    assert bad._markup_ok is False
    good = SpinnerWidget("[bold]{spinner}[/]", FEED_SPINNERS[0])
    assert good._markup_ok is True


def test_escape_markup_output_is_textual_parseable_for_common_case():
    # Common single-bracket tag must escape into something Textual can render.
    Content.from_markup(escape_markup("[ref=x]"))  # must not raise
