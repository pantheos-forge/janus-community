import asyncio
from pathlib import Path

import pytest
from janus.core.backend import AgentMessage, MessageType
from janus.core.tools.registry import ToolContext, ToolRegistry, tool
from janus.core.backends.generic import _COMPACT_TARGET, GenericBackend


def _backend(tmp_path: Path) -> GenericBackend:
    return GenericBackend(
        working_directory=tmp_path,
        system_prompt="You are a test agent.",
        model="test-model",
        registry=ToolRegistry(),
    )


def test_init_seeds_state(tmp_path):
    b = _backend(tmp_path)
    assert b._messages == []
    assert b._running is False
    assert b.session_id and len(b.session_id) == 8
    assert b.supports_resume is False


def test_prepare_query_messages_first_and_subsequent(tmp_path):
    b = _backend(tmp_path)
    b._prepare_query_messages("do the thing")
    assert b._messages == [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "do the thing"},
    ]
    b._prepare_query_messages("also this")
    assert b._messages[-1] == {"role": "user", "content": "also this"}
    assert b._messages[0]["role"] == "system"  # not reset


@pytest.mark.asyncio
async def test_query_then_receive_yields_result(tmp_path):
    # With the placeholder _agent_loop, a query produces a terminal RESULT.
    b = _backend(tmp_path)
    await b.query("hello")
    msgs = [m async for m in b.receive_messages()]
    assert msgs[-1].type is MessageType.RESULT
    await b.disconnect()


@pytest.mark.asyncio
async def test_resume_returns_false(tmp_path):
    b = _backend(tmp_path)
    assert await b.resume("abc") is False


def test_seams_are_abstract(tmp_path):
    b = _backend(tmp_path)
    with pytest.raises(NotImplementedError):
        b._tool_result_message({"id": "1"}, "result")


class _ScriptedBackend(GenericBackend):
    """GenericBackend whose _chat_completion replays a scripted list of responses."""

    def __init__(self, *a, script, **k):
        super().__init__(*a, **k)
        self._script = list(script)

    async def _chat_completion(self):
        return self._script.pop(0) if self._script else {"message": {"content": "done"}}

    def _tool_result_message(self, tool_call, result):
        return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}


def _echo_registry():
    reg = ToolRegistry()

    @tool("echo", "echo", {"type": "object", "properties": {"text": {"type": "string"}}})
    def echo(ctx: ToolContext, text=""):
        return f"echoed:{text}"

    reg.register(echo)
    return reg


@pytest.mark.asyncio
async def test_loop_dispatches_tool_then_finishes(tmp_path):
    script = [
        {"message": {"content": "I will call echo", "tool_calls": [
            {"id": "c1", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}]}},
        {"message": {"content": "all done"}},  # no tool_calls -> loop ends
    ]
    b = _ScriptedBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                         registry=_echo_registry(), script=script)
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    types = [m.type for m in msgs]
    assert MessageType.TOOL_START in types
    assert MessageType.TOOL_RESULT in types
    assert msgs[-1].type is MessageType.RESULT
    tr = next(m for m in msgs if m.type is MessageType.TOOL_RESULT)
    assert tr.content == "echoed:hi"
    # the tool result was appended to history in wire shape
    assert any(m.get("role") == "tool" and m.get("content") == "echoed:hi" for m in b._messages)
    await b.disconnect()


@pytest.mark.asyncio
async def test_loop_emits_error_on_none_completion(tmp_path):
    class _NoneBackend(_ScriptedBackend):
        async def _chat_completion(self):
            return None

    b = _NoneBackend(working_directory=tmp_path, system_prompt="s", model="m",
                     registry=_echo_registry(), script=[])
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert any(m.type is MessageType.ERROR for m in msgs)
    assert msgs[-1].type is MessageType.RESULT
    await b.disconnect()


@pytest.mark.asyncio
async def test_query_reinvoked_midrun_cancels_prior_loop_and_appends_turn(tmp_path):
    """A second query() call mid-run must cancel+await the prior loop, then append."""

    class _SlowThenDoneBackend(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._calls = 0

        async def _chat_completion(self):
            self._calls += 1
            if self._calls == 1:
                # Block "forever" so the first query()'s loop is still running
                # when the second query() call comes in.
                await asyncio.sleep(3600)
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = _SlowThenDoneBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                             registry=_echo_registry())
    await b.query("first")
    await asyncio.sleep(0.05)  # let the first loop start its (sleeping) _chat_completion
    first_task = b._task
    assert first_task is not None and not first_task.done()

    await b.query("more")

    assert first_task.done()  # the prior loop was cancelled and awaited
    assert b._messages[-1] == {"role": "user", "content": "more"}

    msgs = [m async for m in b.receive_messages()]
    assert msgs[-1].type is MessageType.RESULT
    await b.disconnect()


@pytest.mark.asyncio
async def test_prose_only_turn_triggers_nudge_then_completes(tmp_path):
    # Model never calls a tool, no matter how many times it's nudged -> the loop
    # nudges MAX_NUDGES times (1 initial call + MAX_NUDGES retries, mirroring the
    # Apophis source's own "model never acts" test) then gives up.
    from janus.core.backends.generic import MAX_NUDGES

    calls = {"n": 0}

    class _NudgeBackend(_ScriptedBackend):
        async def _chat_completion(self):
            calls["n"] += 1
            return {"message": {"content": "let me think..."}}  # no tools, ever

    b = _NudgeBackend(working_directory=tmp_path, system_prompt="s", model="m",
                      registry=_echo_registry(), script=[])
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert msgs[-1].type is MessageType.RESULT
    assert calls["n"] == MAX_NUDGES + 1  # 1 initial call + MAX_NUDGES retries
    nudge_msgs = [
        m for m in b._messages
        if m.get("role") == "user" and "have not called any tools" in m.get("content", "")
    ]
    assert len(nudge_msgs) == MAX_NUDGES
    await b.disconnect()


@pytest.mark.asyncio
async def test_malformed_tool_arguments_json_does_not_crash_loop(tmp_path):
    script = [
        {"message": {"content": "calling echo", "tool_calls": [
            {"id": "c1", "function": {"name": "echo", "arguments": "{not json"}}]}},
        {"message": {"content": "all done"}},  # no tool_calls -> loop ends
    ]
    b = _ScriptedBackend(working_directory=tmp_path, system_prompt="sys", model="m",
                         registry=_echo_registry(), script=script)
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    assert not any(m.type is MessageType.ERROR for m in msgs)
    assert msgs[-1].type is MessageType.RESULT
    ts = next(m for m in msgs if m.type is MessageType.TOOL_START)
    assert ts.tool_args == {}
    await b.disconnect()


def test_compaction_evicts_old_bulky_turns_protecting_anchors(tmp_path):
    b = _backend(tmp_path)
    b._context_window = 1000       # small window to force compaction
    b._chars_per_token = 3.0
    # system, task, then many bulky assistant/tool exchanges
    b._messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "the task"},
    ]
    for i in range(20):
        b._messages.append({"role": "assistant", "content": "x" * 900,
                            "tool_calls": [{"id": f"c{i}", "function": {"name": "echo", "arguments": "{}"}}]})
        b._messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "y" * 900})
    before = len(b._messages)
    b._maybe_precompact()
    # anchors preserved
    assert b._messages[0]["content"] == "S"
    assert b._messages[1]["content"] == "the task"
    # estimate is now under target, and history shrank (stubbed and/or evicted)
    assert b._estimate_request_tokens() <= int(_COMPACT_TARGET * b._context_window) or len(b._messages) < before


def test_evict_returns_false_when_nothing_removable(tmp_path):
    b = _backend(tmp_path)
    b._messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "t"}]
    assert b._evict_oldest_exchange() is False


@pytest.mark.asyncio
async def test_tool_emit_output_surfaces_as_output_message(tmp_path):
    from janus.core.tools.registry import ToolContext, ToolRegistry, tool

    reg = ToolRegistry()

    @tool("finish", "emit output", {"type": "object", "properties": {}})
    async def finish(ctx: ToolContext, **kw):
        if ctx.emit_output:
            ctx.emit_output({"kind": "brief", "value": 42})
        return "emitted"

    reg.register(finish)

    script = [
        {"message": {"content": "done", "tool_calls": [
            {"id": "c1", "function": {"name": "finish", "arguments": "{}"}}]}},
        {"message": {"content": "bye"}},
    ]
    b = _ScriptedBackend(working_directory=tmp_path, system_prompt="s", model="m",
                         registry=reg, script=script)
    await b.query("go")
    msgs = [m async for m in b.receive_messages()]
    outputs = [m for m in msgs if m.type is MessageType.OUTPUT]
    assert outputs and outputs[0].content == {"kind": "brief", "value": 42}
    await b.disconnect()


@pytest.mark.asyncio
async def test_hold_gates_the_agent_loop(tmp_path):
    """While held, no further _chat_completion calls happen; release resumes."""
    import asyncio

    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.registry import ToolRegistry

    calls = []

    class Chatty(GenericBackend):
        async def _chat_completion(self):
            calls.append(1)
            await asyncio.sleep(0.01)
            if len(calls) >= 8:
                return {"message": {"content": "done"}}
            return {"message": {"content": "", "tool_calls": [
                {"id": f"c{len(calls)}",
                 "function": {"name": "nope", "arguments": "{}"}}]}}

        def _tool_result_message(self, tc, result):
            return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}

    b = Chatty(working_directory=tmp_path, system_prompt="s", model="m",
               registry=ToolRegistry())
    await b.query("go")
    await asyncio.sleep(0.05)
    b.hold()
    count_at_hold = len(calls)
    await asyncio.sleep(0.3)
    assert len(calls) <= count_at_hold + 1   # at most the in-flight iteration
    b.release()
    async for msg in b.receive_messages():
        if msg.type.value == "result":
            break
    assert len(calls) == 8                   # ran to completion after release


@pytest.mark.asyncio
async def test_messages_carry_the_current_generation(tmp_path):
    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.registry import ToolRegistry

    class OneShot(GenericBackend):
        async def _chat_completion(self):
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = OneShot(working_directory=tmp_path, system_prompt="s", model="m",
                registry=ToolRegistry())
    await b.query("go")
    seen = []
    async for msg in b.receive_messages():
        seen.append(msg)
        if msg.type.value == "result":
            break
    assert seen, "expected at least the RESULT message"
    assert all(m.metadata.get("generation") == 1 for m in seen)


@pytest.mark.asyncio
async def test_stale_generation_messages_are_discarded(tmp_path):
    """A message stranded by a previous run (older generation) must never
    reach — or terminate — a later run's stream."""
    from janus.core.backend import AgentMessage, MessageType
    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.registry import ToolRegistry

    class TwoTurn(GenericBackend):
        async def _chat_completion(self):
            return {"message": {"content": "fresh turn done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = TwoTurn(working_directory=tmp_path, system_prompt="s", model="m",
                registry=ToolRegistry())
    await b.query("first")           # generation 1
    async for msg in b.receive_messages():
        if msg.type.value == "result":
            break

    # Seed the hazard deterministically: a stale RESULT stranded in the queue
    # (what a stopped run leaves behind), tagged with the OLD generation.
    b._message_queue.put_nowait(
        AgentMessage(type=MessageType.RESULT, content=None,
                     metadata={"generation": 1, "cost_usd": 0})
    )

    await b.query("second")          # generation 2
    seen = []
    async for msg in b.receive_messages():
        seen.append(msg)
        if msg.type.value == "result":
            break
    # The stale RESULT was discarded: the stream did NOT end instantly; the
    # fresh turn's TEXT arrived before its own RESULT.
    assert any(m.type.value == "text" and "fresh turn done" in str(m.content)
               for m in seen)
    assert all(m.metadata.get("generation") == 2 for m in seen)


@pytest.mark.asyncio
async def test_result_outcome_is_error_when_the_loop_raises(tmp_path):
    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.registry import ToolRegistry

    class Boom(GenericBackend):
        async def _chat_completion(self):
            raise RuntimeError("kaboom")

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = Boom(working_directory=tmp_path, system_prompt="s", model="m", registry=ToolRegistry())
    await b.query("go")
    results = [m async for m in b.receive_messages()]
    result = next(m for m in results if m.type.value == "result")
    assert result.metadata.get("outcome") == "error"


@pytest.mark.asyncio
async def test_result_outcome_is_ok_on_normal_completion(tmp_path):
    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.registry import ToolRegistry

    class Fine(GenericBackend):
        async def _chat_completion(self):
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = Fine(working_directory=tmp_path, system_prompt="s", model="m", registry=ToolRegistry())
    await b.query("go")
    result = None
    async for m in b.receive_messages():
        if m.type.value == "result":
            result = m
            break
    assert result is not None
    assert result.metadata.get("outcome") == "ok"


@pytest.mark.asyncio
async def test_deliverable_nudge_when_emit_output_never_called(tmp_path):
    """A persona with emit_output that stops without emitting gets exactly one
    nudge to emit before the run concludes."""
    import json

    from janus.core.backends.generic import GenericBackend
    from janus.core.tools.output import make_emit_output_tool
    from janus.core.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(make_emit_output_tool({"type": "object",
                                        "properties": {"summary": {"type": "string"}},
                                        "required": ["summary"]}))

    class WindsDown(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n = 0
            self.saw_nudge = False

        async def _chat_completion(self):
            self._n += 1
            # look for the nudge in the last user message
            if self._messages and self._messages[-1].get("role") == "user" \
               and "emit_output" in str(self._messages[-1].get("content", "")):
                self.saw_nudge = True
                return {"message": {"content": "", "tool_calls": [
                    {"id": "e1", "function": {"name": "emit_output",
                     "arguments": json.dumps({"summary": "here it is"})}}]}}
            if self._n == 1:
                # did some work (a non-emit tool) then a tool-less turn
                return {"message": {"content": "I think I'm done."}}
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = WindsDown(working_directory=tmp_path, system_prompt="s", model="m", registry=reg)
    await b.query("go")
    async for m in b.receive_messages():
        if m.type.value == "result":
            break
    assert b.saw_nudge, "expected a deliverable nudge"
    assert (tmp_path / "output.json").exists(), "the nudge produced the deliverable"


@pytest.mark.asyncio
async def test_no_deliverable_nudge_without_emit_output(tmp_path):
    """A persona with NO emit_output (no output schema) is never deliverable-nudged."""
    from janus.core.backends.generic import GenericBackend, MAX_NUDGES
    from janus.core.tools.registry import ToolRegistry

    class Fine(GenericBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.turns = 0

        async def _chat_completion(self):
            self.turns += 1
            return {"message": {"content": "done"}}

        def _tool_result_message(self, tool_call, result):
            return {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result}

    b = Fine(working_directory=tmp_path, system_prompt="s", model="m", registry=ToolRegistry())
    await b.query("go")
    async for m in b.receive_messages():
        if m.type.value == "result":
            break
    # Without emit_output, no deliverable nudge is applied (only zero-tools nudge).
    # Zero-tools nudge results in 1 initial call + MAX_NUDGES retries.
    assert b.turns == MAX_NUDGES + 1
