"""Pilot tests for SessionView — the reusable 3B single-agent widget.

Follows the tests/interface/test_tui.py pilot pattern: a tiny host App
mounts one SessionView and presses real keys / clicks real buttons,
asserting against mounted widget content (never synthetic events or
display-only asserts — see the Cycle-3B house rule).
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from janus.core.events import Event, EventBus, EventType
from janus.interface.components.session_view import SessionView


class _FakeController:
    """Mirrors the invariant of the real AgentController that matters here:
    it always subscribes to USER_INPUT (janus/core/controller.py:83, in
    __init__, unconditional on enable_user_replies) and routes to reply()
    while awaiting input. SessionView must only ever call bus.emit_input —
    never controller.reply() directly — so this fake proves that path.
    """

    def __init__(self, bus: EventBus) -> None:
        self.events = bus
        self.replies: list[str] = []
        self.state = "running"
        bus.subscribe(EventType.USER_INPUT, self._on_user_input)

    def _on_user_input(self, event: Event) -> None:
        text = event.data.get("text", "")
        if text:
            self.reply(text)

    def enable_user_replies(self) -> None:
        pass

    def reply(self, text: str) -> bool:
        self.replies.append(text)
        return True

    def inject(self, text: str) -> bool:
        return True

    def pause(self) -> bool:
        return True

    def resume(self, instruction: str | None = None) -> bool:
        return True

    def stop(self) -> bool:
        return True


class _Host(App[None]):
    def __init__(self, controller: _FakeController, bus: EventBus) -> None:
        super().__init__()
        self._c = controller
        self._b = bus

    def compose(self) -> ComposeResult:
        yield SessionView(self._c, self._b, id="sv")


@pytest.mark.asyncio
async def test_session_view_renders_agent_messages():
    bus = EventBus()
    app = _Host(_FakeController(bus), bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        bus.emit_message("hello from the agent")
        await pilot.pause(0.2)
        feed = app.query_one("#sv ActivityFeed")
        rendered = "\n".join(str(c.render()) for c in feed.children)
        assert "hello from the agent" in rendered


@pytest.mark.asyncio
async def test_session_view_question_and_chip_reply():
    bus = EventBus()
    controller = _FakeController(bus)
    app = _Host(controller, bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        # simulate the controller entering awaiting_input with choices
        app.query_one("#sv", SessionView).agent_state = "awaiting_input"
        bus.emit(
            Event(
                EventType.MESSAGE,
                {
                    "text": "Approve?",
                    "type": "question",
                    "choices": ["Approve the spec", "Request changes"],
                },
            )
        )
        await pilot.pause(0.2)
        buttons = app.query("#sv QuestionPanel Button")
        assert [str(b.label) for b in buttons] == ["Approve the spec", "Request changes"]
        await pilot.click("#sv QuestionPanel Button")
        await pilot.pause(0.1)
        # the reply reached the controller via the bus USER_INPUT -> reply path
        assert controller.replies == ["Approve the spec"]


@pytest.mark.asyncio
async def test_session_view_input_submit_routes_to_reply_when_awaiting_input():
    bus = EventBus()
    controller = _FakeController(bus)
    app = _Host(controller, bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        app.query_one("#sv", SessionView).agent_state = "awaiting_input"
        await pilot.pause(0.1)
        await pilot.click("#sv #user_input")
        await pilot.press(*"helloreply")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert controller.replies == ["helloreply"]


@pytest.mark.asyncio
async def test_session_view_input_submit_warns_when_running():
    bus = EventBus()
    controller = _FakeController(bus)
    app = _Host(controller, bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        # default agent_state is "running"
        await pilot.click("#sv #user_input")
        await pilot.press(*"typedwhilerunning")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert controller.replies == []
        feed = app.query_one("#sv ActivityFeed")
        rendered = "\n".join(str(c.render()) for c in feed.children)
        assert "running" in rendered.lower()


@pytest.mark.asyncio
async def test_session_view_build_panel_activates_on_factory_tool_events():
    bus = EventBus()
    app = _Host(_FakeController(bus), bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        bus.emit_tool("start", "scaffold_persona", {})
        await pilot.pause(0.2)
        panel = app.query_one("#sv BuildPanel")
        assert panel.display is True


async def _show_two_choice_question(app, bus, pilot):
    """Put the view in awaiting_input with a 2-choice question showing, and
    focus the (empty) reply box — the precondition for digit selection."""
    app.query_one("#sv", SessionView).agent_state = "awaiting_input"
    bus.emit(
        Event(
            EventType.MESSAGE,
            {
                "text": "Approve?",
                "type": "question",
                "choices": ["Approve the spec", "Request changes"],
            },
        )
    )
    await pilot.pause(0.2)
    await pilot.click("#sv #user_input")
    await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_session_view_digit_key_selects_choice():
    bus = EventBus()
    controller = _FakeController(bus)
    app = _Host(controller, bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await _show_two_choice_question(app, bus, pilot)
        # empty reply box + question showing: "1" selects the first choice
        await pilot.press("1")
        await pilot.pause(0.1)
        assert controller.replies == ["Approve the spec"]
        assert app.query_one("#sv QuestionPanel").display is False
        # nothing leaked into the reply box
        assert app.query_one("#sv #user_input").value == ""


@pytest.mark.asyncio
async def test_session_view_digit_into_nonempty_box_is_literal_text():
    bus = EventBus()
    controller = _FakeController(bus)
    app = _Host(controller, bus)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await _show_two_choice_question(app, bus, pilot)
        # "9" is out of range (2 choices) -> typed literally, box now non-empty
        await pilot.press("9")
        # "2" with a non-empty box -> literal text, NOT a selection
        await pilot.press("2")
        await pilot.pause(0.1)
        assert controller.replies == []
        assert app.query_one("#sv #user_input").value == "92"
        assert app.query_one("#sv QuestionPanel").display is True
