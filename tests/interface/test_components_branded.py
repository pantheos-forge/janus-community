import pytest
from textual.app import App, ComposeResult

from janus.interface.components.activity_feed import ActivityFeed, escape_markup
from janus.interface.components.splash import SplashScreen


def test_escape_markup_neutralizes_brackets():
    assert "[" not in escape_markup("[bold]x[/bold]") or "\\[" in escape_markup("[bold]x[/bold]")


class _FeedHarness(App):
    def compose(self) -> ComposeResult:
        yield ActivityFeed(id="activity_feed")


def _feed_text(feed) -> str:
    """Concatenate the rendered content of every mounted feed child."""
    parts = []
    for child in feed.children:
        try:
            parts.append(str(child.render()))
        except Exception:
            parts.append(str(getattr(child, "renderable", "")))
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_activity_feed_adds_message_and_tool():
    app = _FeedHarness()
    async with app.run_test() as pilot:
        feed = app.query_one(ActivityFeed)
        feed.add_message("hello world", "info")
        feed.add_tool_execution("web_fetch", {"url": "x"}, "running", None)
        feed.add_tool_execution("web_fetch", {"url": "x"}, "completed", "ok")
        await pilot.pause()
        # The feed mounted child rows for the message and the tool execution.
        assert len(feed.children) >= 2
        # Use the feed's real vocabulary ("running"/"completed") — anything
        # else falls through to the "Unknown" branch and drops the result.
        assert "Unknown" not in _feed_text(feed)


def test_splash_is_rebranded():
    banner = SplashScreen(app_name="Janus").render_banner_text()  # see Step 3 for this helper
    lowered = banner.lower()
    assert "janus" in lowered
    for pentest in ("penetration", "security agent", "pentest", "target"):
        assert pentest not in lowered


def test_banner_is_braille_and_fits_layout():
    """Spec: banner is braille-only, <=70 cols, ~15 rows (13-17)."""
    from janus.interface.components import splash

    lines = splash._BANNER.split("\n")
    assert 13 <= len(lines) <= 17
    assert max(len(line) for line in lines) <= 70
    for line in lines:
        for ch in line:
            assert 0x2800 <= ord(ch) <= 0x28FF


def test_splash_docstrings_describe_janus_head():
    """Spec: docstrings no longer call the banner 'generic'."""
    from janus.interface.components import splash

    assert "generic" not in (splash.__doc__ or "").lower()
    assert "generic" not in (splash.SplashScreen.__doc__ or "").lower()
