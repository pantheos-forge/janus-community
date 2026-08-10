"""Custom banner threading: SplashScreen, JanusApp, launch, rendered agent.py."""

import asyncio

import pytest
from rich.console import Console

from janus.core.events import EventBus
from janus.interface.components.splash import SplashScreen
from janus.interface.tui import JanusApp

ART = "\n".join(["⣶" * 12] * 8)  # distinct from the default medallion


class _FakeController:
    def __init__(self) -> None:
        self.events = EventBus()

    def pause(self) -> bool:
        return True

    def resume(self, instruction=None) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def inject(self, instruction: str) -> bool:
        return True

    async def run(self, task: str, resume_session_id=None) -> dict:
        self.events.emit_state("completed")
        return {"status": "completed", "session_id": "s1", "cost_usd": 0.0}


def _render(splash: SplashScreen) -> str:
    spinner, bar = splash._build_loading_text(0)
    console = Console(record=True, width=100)
    console.print(splash._build_content(spinner, bar))
    return console.export_text()


def test_splash_renders_custom_banner():
    out = _render(SplashScreen(app_name="demo_agent", banner=ART))
    assert "⣶⣶⣶" in out


def test_splash_default_banner_unchanged():
    out = _render(SplashScreen(app_name="Janus"))
    assert "⣶⣶⣶" not in out  # the custom art did not leak into the default
    # the default content contains non-blank braille from the class banner
    assert any(ch in out for ch in SplashScreen.BANNER if 0x2801 <= ord(ch) <= 0x28FF)


@pytest.mark.asyncio
async def test_app_threads_banner_to_splash():
    app = JanusApp(_FakeController(), "t", title="demo_agent", banner=ART)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)  # well inside the 4s splash window
        splash = app.query_one(SplashScreen)
        assert splash._banner == ART


def test_launch_passes_banner_to_run_tui(monkeypatch):
    import janus.interface as iface
    import janus.interface.tui as tui_mod

    captured = {}

    async def fake_run_tui(controller, task, *, title=None, banner=None,
                           resume_session_id=None):
        captured["banner"] = banner

    monkeypatch.setattr(iface, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr(iface, "_textual_available", lambda: True)
    monkeypatch.setattr(tui_mod, "run_tui", fake_run_tui)
    iface.launch(object(), "t", title="x", banner=ART)
    assert captured["banner"] == ART


def test_rendered_agent_template_passes_banner():
    import janus.factory.render as render_mod

    source = open(render_mod.__file__).read()
    assert "banner=persona.banner" in source
