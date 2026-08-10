"""Tests for the JanusApp Textual TUI on the generic event vocabulary.

Uses Textual's headless ``run_test()`` pilot with a ``FakeController`` that
carries its own fresh ``EventBus`` per test. No network, no real backend.
"""

import asyncio

import pytest

from janus.core.events import EventBus
from janus.interface.tui import JanusApp


class FakeController:
    """Records control calls; emits generic events when run."""

    def __init__(self, events: EventBus) -> None:
        self.events = events
        self.calls: list[str] = []
        self._resume = asyncio.Event()

    def pause(self) -> bool:
        self.calls.append("pause")
        return True

    def resume(self, instruction=None) -> bool:
        self.calls.append("resume")
        self._resume.set()
        return True

    def stop(self) -> bool:
        self.calls.append("stop")
        self._resume.set()
        return True

    def inject(self, instruction: str) -> bool:
        self.calls.append(f"inject:{instruction}")
        return True

    async def run(self, task: str, resume_session_id=None) -> dict:
        self.events.emit_state("running")
        self.events.emit_message("thinking about " + task)
        self.events.emit_tool("start", "web_fetch", {"url": "u"})
        self.events.emit_tool("result", "web_fetch", result="fetched")
        self.events.emit_output({"summary": "done"})
        self.events.emit_state("completed")
        return {"status": "completed", "session_id": "s1", "cost_usd": 0.0}


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
async def test_tui_renders_generic_events_and_completes():
    events = EventBus()
    ctrl = FakeController(events)
    app = JanusApp(ctrl, "market X", title="Janus")
    async with app.run_test() as pilot:
        # Skip the splash delay: force the main interface up.
        app.show_splash = False
        await pilot.pause(0.3)
        # Let the worker-thread run() emit its events.
        await pilot.pause(0.3)
        feed = app.query_one("#activity_feed")
        # The feed received the message + tool rows (no crash on OUTPUT).
        assert len(feed.children) >= 2

        # Strengthened: the controller emits tool status "start"/"result";
        # the TUI must TRANSLATE those to the feed's running/completed vocab.
        # A mistranslation would render the tool row as "Unknown" and drop
        # the result. Assert a real status was rendered instead.
        text = _feed_text(feed)
        assert "web_fetch" in text
        assert "Unknown" not in text
        # completed tool result flowed through
        assert "fetched" in text


@pytest.mark.asyncio
async def test_status_bar_tolerates_uppercase_state_emission():
    """Guards the defensive `.lower()` in `_on_state_change`: a producer that
    emits an uppercase state string (not matching our lowercase convention)
    must not leave the status bar showing "Unknown"."""
    events = EventBus()
    ctrl = FakeController(events)
    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.3)

        # Simulate a non-normalizing producer emitting an uppercase state.
        events.emit_state("RUNNING")
        await pilot.pause(0.2)

        status_bar = app.query_one("#status_bar")
        text = str(getattr(status_bar, "renderable", status_bar.render()))
        assert "Unknown" not in text


@pytest.mark.asyncio
async def test_slash_pause_calls_controller():
    events = EventBus()
    ctrl = FakeController(events)
    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.2)
        inp = app.query_one("#user_input")
        inp.value = "/pause"
        from textual.widgets import Input

        app.post_message(Input.Submitted(inp, "/pause"))
        await pilot.pause(0.2)
        assert "pause" in ctrl.calls


@pytest.mark.asyncio
async def test_typing_while_awaiting_input_emits_user_input():
    """awaiting_input state routes REAL keypresses onto the bus as USER_INPUT
    (the controller's _on_user_input turns it into reply()).

    This drives actual key events through the mounted CommandInput (not a
    posted Input.Submitted or a direct .value assignment) so it also proves
    the reply box is enabled and focused while awaiting_input -- a disabled
    Input silently drops keypresses, which a synthetic-event test can't
    catch."""
    from janus.core.events import EventType

    events = EventBus()
    ctrl = FakeController(events)
    seen = []
    events.subscribe(EventType.USER_INPUT, seen.append)

    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.2)
        app.agent_state = "awaiting_input"
        await pilot.pause(0.1)

        inp = app.query_one("#user_input")
        assert inp.disabled is False
        assert app.focused is inp

        await pilot.press("y", "e", "s", "enter")
        await pilot.pause(0.2)

        assert seen and seen[0].data.get("text") == "yes"
        assert inp.value == ""


@pytest.mark.asyncio
async def test_digit_keypress_selects_question_choice():
    """Pressing a digit while the reply box is empty and a question panel
    with that many choices is visible selects the corresponding choice --
    via real keypresses, exercising CommandInput's on_key digit path."""
    from janus.core.events import Event, EventType

    events = EventBus()
    ctrl = FakeController(events)
    seen = []
    events.subscribe(EventType.USER_INPUT, seen.append)

    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.2)
        app.agent_state = "awaiting_input"
        events.emit(
            Event(
                EventType.MESSAGE,
                {
                    "text": "Approve the spec?",
                    "type": "question",
                    "choices": ["Approve the spec", "Request changes"],
                },
            )
        )
        await pilot.pause(0.2)

        inp = app.query_one("#user_input")
        assert inp.value == ""

        await pilot.press("1")
        await pilot.pause(0.2)

        assert len(seen) == 1
        assert seen[0].data.get("text") == "Approve the spec"
        assert inp.value == ""
        assert not app.query_one("#question_panel").display


@pytest.mark.asyncio
async def test_digit_keypress_passes_through_when_input_nonempty():
    """A digit typed as part of a longer answer must never be hijacked as a
    choice selection -- only the FIRST digit into an empty box can select;
    once the box holds text, digits type normally."""
    from janus.core.events import Event, EventType

    events = EventBus()
    ctrl = FakeController(events)
    seen = []
    events.subscribe(EventType.USER_INPUT, seen.append)

    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.2)
        app.agent_state = "awaiting_input"
        events.emit(
            Event(
                EventType.MESSAGE,
                {
                    "text": "Approve the spec?",
                    "type": "question",
                    "choices": ["Approve the spec", "Request changes"],
                },
            )
        )
        await pilot.pause(0.2)

        await pilot.press("4")
        await pilot.pause(0.1)
        # "4" is out of range (only 2 choices) so it fell through to normal
        # text insertion -- the box is no longer empty.
        inp = app.query_one("#user_input")
        assert inp.value == "4"

        await pilot.press("2")
        await pilot.pause(0.2)

        assert seen == []
        assert inp.value == "42"
        assert app.query_one("#question_panel").display


def test_awaiting_input_state_in_lookup_tables():
    """Ensure awaiting_input state is present in both TUI lookup tables
    so status bar never shows 'Unknown' when agent is waiting for input."""
    from janus.interface.tui import _STATE_INPUT_TABLE, _STATUS_LABELS

    assert "awaiting_input" in _STATE_INPUT_TABLE
    assert "awaiting_input" in _STATUS_LABELS
    assert "stopped" in _STATE_INPUT_TABLE
    assert "stopped" in _STATUS_LABELS


@pytest.mark.asyncio
async def test_question_event_shows_panel_with_chips():
    from janus.core.events import Event, EventType

    events = EventBus()
    ctrl = FakeController(events)
    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.3)
        events.emit(
            Event(
                EventType.MESSAGE,
                {
                    "text": "Approve the spec?",
                    "type": "question",
                    "choices": ["Approve the spec", "Request changes"],
                },
            )
        )
        await pilot.pause(0.2)
        panel = app.query_one("#question_panel")
        assert panel.display  # visible
        buttons = app.query("QuestionPanel Button")
        assert [str(b.label) for b in buttons] == [
            "Approve the spec",
            "Request changes",
        ]
        assert app.query_one("#main_row").size.height > 1   # feed/panel not collapsed by the question panel


@pytest.mark.asyncio
async def test_chip_press_sends_choice_text_as_user_input():
    from janus.core.events import Event, EventType

    events = EventBus()
    ctrl = FakeController(events)
    seen = []
    events.subscribe(EventType.USER_INPUT, seen.append)

    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.3)
        events.emit(
            Event(
                EventType.MESSAGE,
                {
                    "text": "Approve the spec?",
                    "type": "question",
                    "choices": ["Approve the spec", "Request changes"],
                },
            )
        )
        await pilot.pause(0.2)
        await pilot.click("QuestionPanel Button")  # first chip
        await pilot.pause(0.1)
        assert seen and seen[0].data.get("text") == "Approve the spec"


@pytest.mark.asyncio
async def test_panel_dismisses_when_state_leaves_awaiting():
    from janus.core.events import Event, EventType

    events = EventBus()
    ctrl = FakeController(events)
    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.3)
        events.emit(
            Event(EventType.MESSAGE, {"text": "Q?", "type": "question", "choices": []})
        )
        await pilot.pause(0.2)
        assert app.query_one("#question_panel").display
        events.emit_state("running", "Reply delivered")
        await pilot.pause(0.2)
        assert not app.query_one("#question_panel").display


@pytest.mark.asyncio
async def test_modal_buttons_never_answer_a_pending_question():
    """A modal's buttons (e.g. QuitScreen's "No") must never bubble to the
    App-level Button.Pressed handler and silently answer a pending question."""
    from janus.core.events import Event, EventType
    from janus.interface.tui import QuitScreen

    events = EventBus()
    ctrl = FakeController(events)
    seen = []
    events.subscribe(EventType.USER_INPUT, seen.append)

    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test() as pilot:
        app.show_splash = False
        await pilot.pause(0.3)
        events.emit(
            Event(
                EventType.MESSAGE,
                {
                    "text": "Approve the spec?",
                    "type": "question",
                    "choices": ["Approve the spec", "Request changes"],
                },
            )
        )
        await pilot.pause(0.2)
        assert app.query_one("#question_panel").display

        # Simulate the collision: the quit modal opens over the pending
        # question, and the user clicks "No" to cancel quitting.
        app.push_screen(QuitScreen())
        await pilot.pause(0.2)
        await pilot.click("#btn_quit_cancel")
        await pilot.pause(0.2)

        # The modal is dismissed, no reply was emitted, and the question
        # is still pending.
        assert not isinstance(app.screen, QuitScreen)
        assert seen == []
        assert app.query_one("#question_panel").display


@pytest.mark.asyncio
async def test_long_question_never_pushes_input_off_screen():
    """3B live-capstone bug: the factory's spec-gate question (a full spec,
    ~100+ lines) grew the question panel past the viewport, clipping the
    chips and input with no scroll path. The question text must scroll
    inside a bounded area; chips and the input stay reachable."""
    from janus.core.events import Event, EventType

    events = EventBus()
    ctrl = FakeController(events)
    app = JanusApp(ctrl, "task", title="Janus")
    async with app.run_test(size=(100, 30)) as pilot:
        app.show_splash = False
        await pilot.pause(0.3)
        long_question = "## PROPOSED SPECIFICATION\n" + "\n".join(
            f"- requirement line {i}" for i in range(120)
        )
        events.emit(Event(EventType.MESSAGE, {
            "text": long_question, "type": "question",
            "choices": ["Approve the spec", "Request changes"]}))
        await pilot.pause(0.3)

        panel = app.query_one("#question_panel")
        assert panel.display
        # The panel is bounded well inside the 30-row viewport...
        assert panel.size.height <= 24, f"panel swallowed the screen: {panel.size.height}"
        # ...the input box is on-screen and usable...
        inp = app.query_one("#user_input")
        assert inp.region.height >= 1
        assert inp.region.y < app.size.height
        # ...and the chips are visible too.
        chips = app.query("QuestionPanel Button")
        assert len(chips) == 2
        assert all(b.region.height >= 1 and b.region.y < app.size.height for b in chips)
