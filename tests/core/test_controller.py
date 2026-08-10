import asyncio

import httpx
import pytest
from janus.core.backend import AgentBackend, AgentMessage, MessageType
from janus.core.backends.openai_compat import OpenAICompatBackend
from janus.core.config import load_config
from janus.core.controller import AgentController, AgentState
from janus.core.events import EventBus, EventType
from janus.core.session import SessionStore
from janus.core.tools.registry import ToolContext, ToolRegistry, tool


class ParkedBackend(AgentBackend):
    """Yields one message, waits on an event, then yields RESULT. Records injected queries."""
    def __init__(self):
        self._connected = False
        self._task = None

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        if self._task and not self._task.done():
            self._task.cancel()

    async def query(self, prompt):
        self._task = asyncio.create_task(asyncio.sleep(3600))  # parked forever

    async def receive_messages(self):
        while True:
            if self._task and self._task.done():
                yield AgentMessage(type=MessageType.RESULT, content=None)
                return
            await asyncio.sleep(0.05)

    @property
    def session_id(self):
        return "parked"

    async def resume(self, session_id):
        return False


class FakeBackend(AgentBackend):
    def __init__(self, messages):
        self._messages = messages
        self.connected = False
        self.queried = None

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def query(self, prompt):
        self.queried = prompt

    async def receive_messages(self):
        for m in self._messages:
            yield m

    @property
    def session_id(self):
        return "fake1234"

    async def resume(self, session_id):
        return False


def _controller(tmp_path, messages):
    cfg = load_config(subject="test subject", persona="tester")
    return AgentController(cfg, backend=FakeBackend(messages),
                           session_store=SessionStore(sessions_dir=tmp_path))


@pytest.mark.asyncio
async def test_run_drives_backend_to_completion(tmp_path):
    msgs = [
        AgentMessage(MessageType.TEXT, "thinking"),
        AgentMessage(MessageType.TOOL_START, "", tool_name="echo", tool_args={"text": "x"}),
        AgentMessage(MessageType.TOOL_RESULT, "echoed:x", tool_name="echo"),
        AgentMessage(MessageType.RESULT, "", metadata={"cost_usd": 0.02}),
    ]
    ctrl = _controller(tmp_path, msgs)
    seen = []
    ctrl.events.subscribe(EventType.STATE_CHANGED, lambda e: seen.append(e.data["state"]))
    result = await ctrl.run("do it")
    assert ctrl.state is AgentState.COMPLETED
    assert result["status"] == "completed"
    assert result["cost_usd"] == 0.02
    assert "running" in seen and "completed" in seen
    assert ctrl.backend.queried == "do it"


@pytest.mark.asyncio
async def test_run_without_backend_raises(tmp_path):
    bus = EventBus()
    before_cmd = len(bus._handlers.get(EventType.USER_COMMAND, []))
    before_input = len(bus._handlers.get(EventType.USER_INPUT, []))

    cfg = load_config()
    ctrl = AgentController(
        cfg, backend=None, session_store=SessionStore(sessions_dir=tmp_path), events=bus
    )
    with pytest.raises(ValueError, match="injected backend"):
        await ctrl.run("x")

    assert len(bus._handlers.get(EventType.USER_COMMAND, [])) == before_cmd
    assert len(bus._handlers.get(EventType.USER_INPUT, [])) == before_input


@pytest.mark.asyncio
async def test_run_emits_tool_and_message_events(tmp_path):
    msgs = [
        AgentMessage(MessageType.TEXT, "hello"),
        AgentMessage(MessageType.TOOL_START, "", tool_name="echo", tool_args={}),
        AgentMessage(MessageType.RESULT, "", metadata={"cost_usd": 0.0}),
    ]
    ctrl = _controller(tmp_path, msgs)
    tools, messages = [], []
    ctrl.events.subscribe(EventType.TOOL, lambda e: tools.append(e.data))
    ctrl.events.subscribe(EventType.MESSAGE, lambda e: messages.append(e.data))
    await ctrl.run("go")
    assert any(t["name"] == "echo" for t in tools)
    assert any(m["text"] == "hello" for m in messages)


class PausableBackend(AgentBackend):
    """Yields one message, waits on an event, then yields RESULT. Records injected queries."""
    def __init__(self):
        self.gate = asyncio.Event()
        self.injected = []
        self.connected = False
    async def connect(self): self.connected = True
    async def disconnect(self): self.connected = False
    async def query(self, prompt): self.injected.append(prompt)
    async def receive_messages(self):
        yield AgentMessage(MessageType.TEXT, "first")
        await self.gate.wait()
        yield AgentMessage(MessageType.RESULT, "", metadata={"cost_usd": 0.0})
    @property
    def session_id(self): return "p1"
    async def resume(self, session_id): return False


@pytest.mark.asyncio
async def test_pause_then_resume_with_instruction(tmp_path):
    backend = PausableBackend()
    cfg = load_config()
    ctrl = AgentController(cfg, backend=backend, session_store=SessionStore(sessions_dir=tmp_path))
    run_task = asyncio.create_task(ctrl.run("go"))
    await asyncio.sleep(0.05)          # let it emit "first" and start awaiting the gate
    ctrl.pause()
    backend.gate.set()                 # allow the loop to proceed to the pause check
    await asyncio.sleep(0.05)
    assert ctrl.state is AgentState.PAUSED
    ctrl.resume("focus on Europe")     # unblock + queue an instruction
    result = await run_task
    assert "focus on Europe" in backend.injected
    assert result["status"] in {"completed", "idle"}


@pytest.mark.asyncio
async def test_bare_resume_preserves_injected_instruction(tmp_path):
    backend = PausableBackend()
    cfg = load_config()
    ctrl = AgentController(cfg, backend=backend, session_store=SessionStore(sessions_dir=tmp_path))
    run_task = asyncio.create_task(ctrl.run("go"))
    await asyncio.sleep(0.05)          # let it emit "first" and start awaiting the gate
    ctrl.pause()
    backend.gate.set()                 # allow the loop to proceed to the pause check
    await asyncio.sleep(0.05)
    assert ctrl.state is AgentState.PAUSED
    ctrl.inject("remember this")       # queue an instruction while PAUSED (not RUNNING)
    ctrl.resume()                      # bare resume, no instruction argument
    result = await run_task
    assert "remember this" in backend.injected
    assert result["status"] in {"completed", "idle"}


@pytest.mark.asyncio
async def test_stop_requests_halt(tmp_path):
    backend = PausableBackend()
    cfg = load_config()
    ctrl = AgentController(cfg, backend=backend, session_store=SessionStore(sessions_dir=tmp_path))
    run_task = asyncio.create_task(ctrl.run("go"))
    await asyncio.sleep(0.05)
    ctrl.stop()
    backend.gate.set()
    result = await run_task
    assert result["status"] == "stopped"


@pytest.mark.asyncio
async def test_controller_drives_real_openai_compat_backend(tmp_path):
    """Full milestone path: controller -> real OpenAICompatBackend -> generic
    agent loop -> registry dispatch, with the wire faked at the httpx transport
    (a tool_call turn, then a final stop turn)."""
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)

    responses = iter([
        {"choices": [{"message": {"content": "calling", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "echo", "arguments": '{"text":"yo"}'}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "cost": 0.001}},
        {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6, "cost": 0.002}},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m", "context_length": 8192}]})
        return httpx.Response(200, json=next(responses))

    backend = OpenAICompatBackend(
        working_directory=tmp_path, system_prompt="sys", model="m", registry=reg,
        base_url="http://fake/v1", api_key="k",
    )
    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    cfg = load_config(subject="test subject", persona="tester")
    ctrl = AgentController(cfg, backend=backend, session_store=SessionStore(sessions_dir=tmp_path))
    tools = []
    ctrl.events.subscribe(EventType.TOOL, lambda e: tools.append(e.data))

    result = await ctrl.run("go")

    assert ctrl.state is AgentState.COMPLETED
    assert any(t["name"] == "echo" for t in tools)
    assert result["status"] == "completed"
    assert result["cost_usd"] == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_run_emits_output_event(tmp_path):
    msgs = [
        AgentMessage(MessageType.OUTPUT, {"kind": "brief", "value": 1}),
        AgentMessage(MessageType.RESULT, "", metadata={"cost_usd": 0.0}),
    ]
    ctrl = _controller(tmp_path, msgs)
    outputs = []
    ctrl.events.subscribe(EventType.OUTPUT, lambda e: outputs.append(e.data))
    await ctrl.run("go")
    assert outputs == [{"kind": "brief", "value": 1}]


def test_controllers_get_isolated_buses_by_default(tmp_path):
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.events import EventType
    from janus.core.session import SessionStore

    c1 = AgentController(load_config(), session_store=SessionStore(sessions_dir=tmp_path / "s1"))
    c2 = AgentController(load_config(), session_store=SessionStore(sessions_dir=tmp_path / "s2"))
    assert c1.events is not c2.events

    seen1, seen2 = [], []
    c1.events.subscribe(EventType.MESSAGE, seen1.append)
    c2.events.subscribe(EventType.MESSAGE, seen2.append)
    c1.events.emit_message("hello c1")
    assert len(seen1) == 1 and seen2 == []


@pytest.mark.asyncio
async def test_stop_from_foreign_thread_terminates_a_parked_run(tmp_path):
    """A backend that never emits messages simulates a run parked inside a
    blocking tool. stop() called from another thread (the TUI shape) must
    still terminate the run promptly."""
    import threading

    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStore

    controller = AgentController(
        load_config(), backend=ParkedBackend(),
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )

    async def stop_later():
        await asyncio.sleep(0.2)
        t = threading.Thread(target=controller.stop)
        t.start()
        t.join()

    result, _ = await asyncio.wait_for(
        asyncio.gather(controller.run("park"), stop_later()), timeout=10
    )
    assert result["status"] == "stopped"


@pytest.mark.asyncio
async def test_stop_marks_the_session_terminally_stopped(tmp_path):
    import threading

    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStatus, SessionStore

    store = SessionStore(sessions_dir=tmp_path / "s")
    controller = AgentController(load_config(), backend=ParkedBackend(),
                                 session_store=store)

    async def stop_later():
        await asyncio.sleep(0.2)
        t = threading.Thread(target=controller.stop)
        t.start()
        t.join()

    result, _ = await asyncio.wait_for(
        asyncio.gather(controller.run("park"), stop_later()), timeout=10
    )
    assert result["status"] == "stopped"
    session = store.load(result["session_id"])
    assert session is not None and session.status is SessionStatus.STOPPED


@pytest.mark.asyncio
async def test_loop_is_captured_before_session_io(tmp_path, monkeypatch):
    """The loop must be captured as run()'s first act, so foreign-thread
    controls marshal correctly even during session-store setup."""
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStore

    store = SessionStore(sessions_dir=tmp_path / "s")
    controller = AgentController(load_config(), backend=None, session_store=store)

    seen = {}
    orig_create = store.create

    def spy_create(*a, **k):
        seen["loop_at_session_create"] = controller._loop
        return orig_create(*a, **k)

    monkeypatch.setattr(store, "create", spy_create)
    with pytest.raises(ValueError):
        await controller.run("task")
    assert seen["loop_at_session_create"] is not None   # captured before I/O
    assert controller._loop is None                     # cleared on the raise path


@pytest.mark.asyncio
async def test_ask_user_round_trip_through_controller_reply(tmp_path):
    import json
    import threading

    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController, AgentState
    from janus.core.session import SessionStore
    from janus.core.tools.builtins import builtin_registry

    class AsksBackend(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1:
                return {"message": {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "ask_user",
                     "arguments": json.dumps({"question": "Which domain?"})}}]}}
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""),
                    "content": result}

    backend = AsksBackend(
        working_directory=tmp_path, system_prompt="s", model="m",
        registry=builtin_registry(["ask_user"]),
    )
    controller = AgentController(
        load_config(), backend=backend,
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )
    controller.enable_user_replies()

    states = []
    from janus.core.events import EventType
    controller.events.subscribe(
        EventType.STATE_CHANGED, lambda e: states.append(e.data.get("state"))
    )

    tool_results = []
    orig = backend._tool_result_message
    backend._tool_result_message = lambda tc, r: tool_results.append(r) or orig(tc, r)

    async def answer_when_asked():
        for _ in range(100):
            await asyncio.sleep(0.05)
            if controller.state is AgentState.AWAITING_INPUT:
                t = threading.Thread(target=controller.reply, args=("healthcare",))
                t.start()
                t.join()
                return
        raise AssertionError("never reached AWAITING_INPUT")

    result, _ = await asyncio.wait_for(
        asyncio.gather(controller.run("go"), answer_when_asked()), timeout=10
    )
    assert result["status"] == "completed"
    assert "awaiting_input" in states
    assert "healthcare" in tool_results          # the reply became the tool result
    assert controller.reply("late") is False     # only honored while awaiting


@pytest.mark.asyncio
async def test_stale_reply_never_leaks_into_a_new_run(tmp_path):
    """A reply that races with stop() may land in the backend's reply queue
    after the parked ask_user task is cancelled. A later run on the SAME
    backend must not have its ask_user satisfied by that stale text."""
    import json

    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController, AgentState
    from janus.core.session import SessionStore
    from janus.core.tools.builtins import builtin_registry

    class AsksBackend(GenericBackend):
        """Parks in ask_user as the first completion of every query turn.

        In the first run the post-ask completion parks forever (the stop's
        cancel unwinds it) so a stopped run leaves no RESULT behind; in the
        second run it finishes normally.
        """

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._asked = False
            self._runs = 0

        async def query(self, prompt):
            self._asked = False
            self._runs += 1
            await super().query(prompt)

        async def _chat_completion(self):
            if not self._asked:
                self._asked = True
                return {"message": {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "ask_user",
                     "arguments": json.dumps({"question": "Which domain?"})}}]}}
            if self._runs == 1:
                await asyncio.sleep(3600)  # parked until stop() cancels
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""),
                    "content": result}

    backend = AsksBackend(
        working_directory=tmp_path, system_prompt="s", model="m",
        registry=builtin_registry(["ask_user"]),
    )
    controller = AgentController(
        load_config(), backend=backend,
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )
    controller.enable_user_replies()

    async def await_state(state):
        for _ in range(100):
            await asyncio.sleep(0.05)
            if controller.state is state:
                return
        raise AssertionError(f"never reached {state}")

    # First run: park in ask_user, then reply("stale") and stop() back to
    # back — the reply may or may not be consumed before the cancel lands.
    async def stale_reply_then_stop():
        await await_state(AgentState.AWAITING_INPUT)
        controller.reply("stale")
        controller.stop()

    await asyncio.wait_for(
        asyncio.gather(controller.run("go"), stale_reply_then_stop()), timeout=10
    )

    # If the race resolved to "consumed", deterministically recreate the
    # losing branch too: text delivered but never consumed by a parked task.
    if backend._reply_queue.empty():
        backend.deliver_reply("stale")

    # Second run on the SAME backend + controller: only "fresh" may win.
    tool_results = []
    orig = backend._tool_result_message
    backend._tool_result_message = lambda tc, r: tool_results.append(r) or orig(tc, r)

    async def fresh_reply():
        await await_state(AgentState.AWAITING_INPUT)
        assert controller.reply("fresh") is True

    result, _ = await asyncio.wait_for(
        asyncio.gather(controller.run("again"), fresh_reply()), timeout=10
    )
    assert result["status"] == "completed"
    assert "fresh" in tool_results
    assert "stale" not in tool_results


@pytest.mark.asyncio
async def test_ask_user_fails_open_when_replies_not_enabled(tmp_path):
    """Same scripted backend, but enable_user_replies() is never called —
    the validation/piped shape. The run must complete without a reply."""
    import json

    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStore
    from janus.core.tools.builtins import builtin_registry

    class AsksBackend(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1:
                return {"message": {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "ask_user",
                     "arguments": json.dumps({"question": "Anyone there?"})}}]}}
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""),
                    "content": result}

    backend = AsksBackend(
        working_directory=tmp_path, system_prompt="s", model="m",
        registry=builtin_registry(["ask_user"]),
    )
    controller = AgentController(
        load_config(), backend=backend,
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )
    result = await asyncio.wait_for(controller.run("go"), timeout=10)
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_pause_resume_stop_drive_the_backend_gate(tmp_path):
    from janus.core.config import load_config
    from janus.core.controller import AgentController, AgentState
    from janus.core.session import SessionStore

    calls = []

    class GateSpy:
        def hold(self):
            calls.append("hold")

        def release(self):
            calls.append("release")

    controller = AgentController(
        load_config(), backend=GateSpy(),
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )
    controller._state = AgentState.RUNNING
    controller.pause()
    assert calls == ["hold"]
    controller._state = AgentState.PAUSED
    controller.resume()
    assert calls == ["hold", "release"]
    controller.stop()
    assert calls == ["hold", "release", "release"]


@pytest.mark.asyncio
async def test_resume_with_instruction_while_parked_keeps_history_wire_valid(tmp_path):
    """3A final-review repro: pause lands while the model turn is in flight,
    the turn parks in ask_user, the user resumes WITH an instruction. The
    cancelled turn's dangling tool_calls must get synthetic tool results so a
    strict server accepts the next request."""
    import json

    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController, AgentState
    from janus.core.session import SessionStore
    from janus.core.tools.builtins import builtin_registry

    class StrictAsksBackend(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        def _assert_history_wire_valid(self):
            answered = {m.get("tool_call_id") for m in self._messages
                        if m.get("role") == "tool"}
            for m in self._messages:
                for tc in (m.get("tool_calls") or []):
                    assert tc.get("id") in answered, \
                        f"dangling tool_call {tc.get('id')!r} — strict server would 400"

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1:
                await asyncio.sleep(0.2)   # window for pause() to land first
                return {"message": {"content": "", "tool_calls": [
                    {"id": "ask1", "function": {"name": "ask_user",
                     "arguments": json.dumps({"question": "Which flavor?"})}}]}}
            self._assert_history_wire_valid()   # the strict-server simulation
            return {"message": {"content": "done after interruption"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""),
                    "content": result}

    backend = StrictAsksBackend(
        working_directory=tmp_path, system_prompt="s", model="m",
        registry=builtin_registry(["ask_user"]),
    )
    controller = AgentController(
        load_config(), backend=backend,
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )
    controller.enable_user_replies()

    async def drive():
        await asyncio.sleep(0.05)
        assert controller.pause() is True          # lands while turn in flight
        for _ in range(100):
            await asyncio.sleep(0.05)
            if controller.state is AgentState.PAUSED:
                break
        else:
            raise AssertionError("never reached PAUSED")
        assert controller.resume("actually, make it about tea") is True

    result, _ = await asyncio.wait_for(
        asyncio.gather(controller.run("go"), drive()), timeout=15
    )
    assert result["status"] == "completed"
    assert any(
        m.get("role") == "tool"
        and m.get("content") == "[interrupted by user instruction]"
        for m in backend._messages
    )


@pytest.mark.asyncio
async def test_question_event_carries_choices(tmp_path):
    import json

    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController, AgentState
    from janus.core.events import EventType
    from janus.core.session import SessionStore
    from janus.core.tools.builtins import builtin_registry

    class AsksWithChoices(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0

        async def _chat_completion(self):
            self._n += 1
            if self._n == 1:
                return {"message": {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "ask_user",
                     "arguments": json.dumps({
                         "question": "Approve the spec?",
                         "choices": ["Approve the spec", "Request changes"]})}}]}}
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""),
                    "content": result}

    backend = AsksWithChoices(
        working_directory=tmp_path, system_prompt="s", model="m",
        registry=builtin_registry(["ask_user"]),
    )
    controller = AgentController(
        load_config(), backend=backend,
        session_store=SessionStore(sessions_dir=tmp_path / "s"),
    )
    controller.enable_user_replies()

    questions = []
    controller.events.subscribe(
        EventType.MESSAGE,
        lambda e: questions.append(e.data) if e.data.get("type") == "question" else None,
    )

    async def answer():
        for _ in range(100):
            await asyncio.sleep(0.05)
            if controller.state is AgentState.AWAITING_INPUT:
                controller.reply("Approve the spec")
                return
        raise AssertionError("never asked")

    result, _ = await asyncio.wait_for(
        asyncio.gather(controller.run("go"), answer()), timeout=10
    )
    assert result["status"] == "completed"
    assert questions and questions[0]["text"] == "Approve the spec?"
    assert questions[0]["choices"] == ["Approve the spec", "Request changes"]


@pytest.mark.asyncio
async def test_crashed_run_reports_error_not_completed(tmp_path):
    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStatus, SessionStore
    from janus.core.tools.registry import ToolRegistry

    class Boom(GenericBackend):
        async def _chat_completion(self):
            raise RuntimeError("provider exploded")

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    store = SessionStore(sessions_dir=tmp_path / "s")
    controller = AgentController(
        load_config(),
        backend=Boom(working_directory=tmp_path, system_prompt="s", model="m",
                     registry=ToolRegistry()),
        session_store=store,
    )
    result = await controller.run("go")
    assert result["status"] == "error"
    session = store.load(result["session_id"])
    assert session is not None and session.status is SessionStatus.ERROR


@pytest.mark.asyncio
async def test_error_outcome_persists_the_error_message(tmp_path):
    """A run that ends via outcome=error (backend-caught, not a controller-level
    exception) must persist last_error too, not just status=ERROR — otherwise the
    failure is invisible to `fleet status`/dashboard/reload, only in the feed."""
    from janus.core.backends.generic import GenericBackend
    from janus.core.config import load_config
    from janus.core.controller import AgentController
    from janus.core.session import SessionStatus, SessionStore
    from janus.core.tools.registry import ToolRegistry

    class Boom(GenericBackend):
        async def _chat_completion(self):
            raise RuntimeError("provider exploded")

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    store = SessionStore(sessions_dir=tmp_path / "s")
    controller = AgentController(
        load_config(),
        backend=Boom(working_directory=tmp_path, system_prompt="s", model="m",
                     registry=ToolRegistry()),
        session_store=store,
    )
    result = await controller.run("go")
    session = store.load(result["session_id"])
    assert session is not None and session.status is SessionStatus.ERROR
    assert session.last_error is not None, "error status persisted but last_error is null"
    assert "provider exploded" in session.last_error
