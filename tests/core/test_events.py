import pytest
from janus.core.events import Event, EventBus, EventType


def test_subscribe_emit_delivers_event():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(EventType.MESSAGE, seen.append)
    bus.emit_message("hello")
    assert len(seen) == 1
    assert seen[0].type is EventType.MESSAGE
    assert seen[0].data == {"text": "hello", "type": "info"}


def test_duplicate_subscribe_is_noop():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(EventType.TOOL, seen.append)
    bus.subscribe(EventType.TOOL, seen.append)
    bus.emit_tool("start", "bash", {"command": "ls"})
    assert len(seen) == 1


def test_handler_exception_is_swallowed():
    bus = EventBus()
    calls: list[str] = []

    def boom(_e): raise RuntimeError("nope")
    def ok(_e): calls.append("ok")

    bus.subscribe(EventType.MESSAGE, boom)
    bus.subscribe(EventType.MESSAGE, ok)
    bus.emit_message("x")  # must not raise
    assert calls == ["ok"]


def test_emit_output_payload():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(EventType.OUTPUT, seen.append)
    bus.emit_output({"kind": "brief", "value": 1})
    assert seen[0].type is EventType.OUTPUT
    assert seen[0].data == {"kind": "brief", "value": 1}


def test_eventbus_has_no_singleton_api():
    assert not hasattr(EventBus, "get")
    assert not hasattr(EventBus, "reset")


def test_two_buses_are_isolated():
    bus_a, bus_b = EventBus(), EventBus()
    seen_a, seen_b = [], []
    bus_a.subscribe(EventType.MESSAGE, seen_a.append)
    bus_b.subscribe(EventType.MESSAGE, seen_b.append)
    bus_a.emit_message("for A only")
    assert len(seen_a) == 1 and seen_b == []


def test_no_production_code_references_a_global_bus():
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "EventBus.get()", "janus/"], capture_output=True, text=True
    )
    assert out.stdout == "", f"global-bus references remain:\n{out.stdout}"


def test_no_pentest_vocabulary():
    names = {e.name for e in EventType}
    assert names == {
        "STATE_CHANGED", "MESSAGE", "TOOL", "OUTPUT",
        "USER_COMMAND", "USER_INPUT",
    }
