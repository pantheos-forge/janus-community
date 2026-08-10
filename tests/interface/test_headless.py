from janus.core.events import EventBus
from janus.interface.headless import HeadlessRenderer, run_headless


def _repo_root() -> str:
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[2])


def test_renderer_prints_lines_per_event(capsys):
    events = EventBus()
    r = HeadlessRenderer(events)
    r.attach()
    events.emit_state("running")
    events.emit_message("hello", "info")
    events.emit_tool("start", "web_fetch", {"url": "u"})
    events.emit_output({"summary": "s", "findings": []})
    r.detach()
    out = capsys.readouterr().out
    assert "[state] running" in out
    assert "hello" in out
    assert "web_fetch" in out
    assert "[output]" in out and "summary" in out


class FakeController:
    def __init__(self, events): self.events = events
    async def run(self, task, resume_session_id=None):
        self.events.emit_state("running")
        self.events.emit_message("working on " + task)
        self.events.emit_state("completed")
        return {"status": "completed", "session_id": "sid", "cost_usd": 0.0}


def test_run_headless_returns_result_and_prints_status(capsys):
    events = EventBus()
    result = run_headless(FakeController(events), "topic")
    assert result["status"] == "completed"
    out = capsys.readouterr().out
    assert "status: completed" in out
    assert "working on topic" in out


def test_run_headless_piped_stays_noninteractive(tmp_path, monkeypatch):
    """No TTY: replies are not enabled and no stdin thread starts."""
    import sys as _sys

    from janus.core.events import EventBus
    from janus.interface.headless import run_headless

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False, raising=False)

    class FakeController:
        def __init__(self):
            self.events = EventBus()
            self.enabled = False

        def enable_user_replies(self):
            self.enabled = True

        async def run(self, task, resume_session_id=None):
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    c = FakeController()
    result = run_headless(c, "task")
    assert result["status"] == "completed"
    assert c.enabled is False


def test_run_headless_tty_enables_replies(tmp_path, monkeypatch):
    import sys as _sys

    from janus.core.events import EventBus
    from janus.interface.headless import run_headless

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: True, raising=False)

    class FakeController:
        def __init__(self):
            self.events = EventBus()
            self.enabled = False

        def enable_user_replies(self):
            self.enabled = True

        async def run(self, task, resume_session_id=None):
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    c = FakeController()
    run_headless(c, "task")
    assert c.enabled is True


def test_translate_reply_maps_bare_numbers_to_choices():
    from janus.interface.headless import _translate_reply

    choices = ["Approve the spec", "Request changes"]
    assert _translate_reply("1", choices) == "Approve the spec"
    assert _translate_reply("2", choices) == "Request changes"
    assert _translate_reply("3", choices) == "3"          # out of range: literal
    assert _translate_reply("1", []) == "1"               # no choices: literal
    assert _translate_reply("free text", choices) == "free text"


def test_stdin_pump_clears_stale_menu_after_translating_once():
    """Once a bare-number line has been translated against a menu, that
    menu is stale. A later bare-number inject during RUNNING (no pending
    question) must pass through literally instead of being silently
    rewritten by the old menu."""
    from janus.core.events import EventType
    from janus.interface.headless import HeadlessRenderer, _handle_stdin_line

    events = EventBus()
    seen = []
    events.subscribe(EventType.USER_INPUT, seen.append)

    class FakeController:
        def __init__(self, events):
            self.events = events

    renderer = HeadlessRenderer(events)
    renderer.last_choices = ["Approve the spec", "Request changes"]
    ctrl = FakeController(events)

    _handle_stdin_line("1", renderer, ctrl)
    assert seen[-1].data.get("text") == "Approve the spec"
    assert renderer.last_choices == []

    _handle_stdin_line("1", renderer, ctrl)
    assert seen[-1].data.get("text") == "1"


def test_renderer_prints_question_menu_and_tracks_choices(capsys):
    from janus.core.events import Event, EventBus, EventType
    from janus.interface.headless import HeadlessRenderer

    bus = EventBus()
    renderer = HeadlessRenderer(bus)
    renderer.attach()
    try:
        bus.emit(Event(EventType.MESSAGE, {
            "text": "Approve the spec?", "type": "question",
            "choices": ["Approve the spec", "Request changes"]}))
    finally:
        renderer.detach()
    out = capsys.readouterr().out
    assert "Approve the spec?" in out
    assert "1) Approve the spec" in out
    assert "2) Request changes" in out
    assert renderer.last_choices == ["Approve the spec", "Request changes"]


_HEADLESS_RUN_SCRIPT = '''
    import asyncio
    from janus.core.events import EventBus
    from janus.interface.headless import run_headless

    class C:
        def __init__(self):
            self.events = EventBus()
        def enable_user_replies(self):  # interactive path is enabled...
            pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    run_headless(C(), "task")
    print("RETURNED")
'''


def test_headless_run_process_exits_promptly(tmp_path):
    """A generated agent run in headless mode must return the shell prompt on
    its own — the stdin pump must not hold the process open (live ^C bug)."""
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "run_once.py"
    script.write_text(textwrap.dedent(_HEADLESS_RUN_SCRIPT))
    # stdin is a pipe (not a TTY) -> the pump does not start; but if the impl
    # ever starts a pump, this still must exit. Give it real stdin via a pipe
    # that stays open, to prove the process still exits.
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path, env={"PYTHONPATH": _repo_root(), "PATH": "/usr/bin:/bin"},
        stdin=subprocess.PIPE, capture_output=True, text=True, timeout=15,
    )
    assert "RETURNED" in proc.stdout
    assert proc.returncode == 0


def test_headless_run_process_exits_promptly_interactive_tty(tmp_path):
    """Same as above, but stdin is a real pty (isatty() == True), which is
    the path that actually starts the stdin pump thread and blocks it in
    input(). This is the scenario the live ^C bug was reported against —
    the pipe-based variant above never starts the pump at all, so it can't
    exercise the bug even if the bug is real."""
    import os
    import pty
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "run_once_tty.py"
    script.write_text(textwrap.dedent(_HEADLESS_RUN_SCRIPT))

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env={"PYTHONPATH": _repo_root(), "PATH": "/usr/bin:/bin"},
            stdin=slave_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            out, _ = proc.communicate()
            raise AssertionError(
                f"headless run did not exit within the timeout with a TTY "
                f"stdin (the live ^C bug); captured output so far:\n{out}"
            ) from exc
    finally:
        os.close(master_fd)
        if slave_fd != -1:
            os.close(slave_fd)

    assert "RETURNED" in out
    assert proc.returncode == 0
