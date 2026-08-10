import sys

import janus.interface as interface
from janus.core.events import EventBus


class FakeController:
    def __init__(self, events): self.events = events
    async def run(self, task, resume_session_id=None):
        self.events.emit_state("completed")
        return {"status": "completed", "session_id": "x", "cost_usd": 0.0}


def test_launch_uses_headless_when_not_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(interface, "_stdout_is_tty", lambda: False)
    # Keep the headless path non-interactive regardless of the actual
    # terminal this test happens to run in (real-terminal pytest runs
    # would otherwise spawn a genuine stdin-reading daemon thread).
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    result = interface.launch(FakeController(EventBus()), "t")
    assert result["status"] == "completed"
    assert "status: completed" in capsys.readouterr().out


def test_launch_uses_tui_when_tty_and_textual(monkeypatch):
    called = {}
    monkeypatch.setattr(interface, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr(interface, "_textual_available", lambda: True)

    def fake_run_tui(controller, task, *, title=None, banner=None, resume_session_id=None):
        called["tui"] = (task, title)
        async def _noop():
            return None
        return _noop()

    import janus.interface.tui as tui
    monkeypatch.setattr(tui, "run_tui", fake_run_tui)
    result = interface.launch(FakeController(EventBus()), "t", title="Janus")
    assert result is None
    assert called["tui"] == ("t", "Janus")
