import asyncio
from types import SimpleNamespace

import pytest

from janus.core.events import EventBus
from janus.fleet.registry import FleetRegistry
from janus.fleet.supervisor import FleetSupervisor, SessionInfo, _Session
from tests.personas.factory_samples import (
    GOOD_MANIFEST, GOOD_PROMPT, GOOD_SCHEMA, GOOD_RUBRIC)


def _scripted_controller_factory(events_to_emit):
    """Build a make_controller that returns a fake controller whose run()
    emits a scripted sequence of (state, cost) onto the session bus."""

    class FakeController:
        def __init__(self, bus):
            self.events = bus
            self.stopped = False

        def stop(self):
            self.stopped = True

        async def run(self, task, resume_session_id=None):
            for state, cost in events_to_emit:
                self.events.emit_state(state)
                if cost:
                    from janus.core.events import Event, EventType
                    self.events.emit(Event(EventType.STATE_CHANGED,
                                           {"state": state, "cost_usd": cost}))
                await asyncio.sleep(0)
            return {"status": events_to_emit[-1][0], "session_id": "x",
                    "cost_usd": events_to_emit[-1][1]}

    def make_controller(agent_record, subject, bus):
        c = FakeController(bus)
        return c, c.run(subject)

    return make_controller


class _NoopCtl:
    def stop(self): pass

async def _forever():
    await asyncio.Event().wait()


def _make_fleet_agent(fleet, name="alpha", with_rubric=True):
    """A registered fleet agent with a persona/ dir (optionally a rubric)."""
    persona = fleet / name / "persona"
    persona.mkdir(parents=True)
    manifest = GOOD_MANIFEST if with_rubric else GOOD_MANIFEST.replace(
        '[validation]\nrubric_file = "rubric.toml"\n', "")
    (persona / "manifest.toml").write_text(manifest)
    (persona / "prompt.md").write_text(GOOD_PROMPT)
    (persona / "output_schema.json").write_text(GOOD_SCHEMA)
    if with_rubric:
        (persona / "rubric.toml").write_text(GOOD_RUBRIC)
    reg = FleetRegistry(fleet)
    reg.register(name, domain="testing", description="d", source="factory",
                 path=str(fleet / name), clock=lambda: "2026-07-25T00:00:00")
    return reg


def _fake_report(passed):
    return SimpleNamespace(
        smoke=SimpleNamespace(passed=True),
        judge=SimpleNamespace(passed=passed, scores={"form": 0.9 if passed else 0.4}),
        passed=passed)


@pytest.mark.asyncio
async def test_spawn_runs_a_session_and_reports_state(tmp_path):
    sup = FleetSupervisor(
        tmp_path,
        make_controller=_scripted_controller_factory([("running", 0), ("completed", 0.5)]),
    )
    sid = await sup.spawn("alpha", "do the thing")
    # let the scripted run complete
    await asyncio.sleep(0.05)
    sessions = sup.sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert isinstance(s, SessionInfo)
    assert s.agent == "alpha" and s.id == sid
    assert s.state == "completed"
    assert s.cost_usd == 0.5
    assert sup.controller_for(sid) is not None
    assert sup.bus_for(sid) is not None
    await sup.shutdown()


@pytest.mark.asyncio
async def test_each_session_gets_its_own_bus(tmp_path):
    sup = FleetSupervisor(
        tmp_path,
        make_controller=_scripted_controller_factory([("running", 0), ("completed", 0)]),
    )
    a = await sup.spawn("alpha", "x")
    b = await sup.spawn("beta", "y")
    await asyncio.sleep(0.05)
    assert sup.bus_for(a) is not sup.bus_for(b)
    assert {s.agent for s in sup.sessions()} == {"alpha", "beta"}
    await sup.shutdown()


@pytest.mark.asyncio
async def test_controller_for_unknown_session_is_none(tmp_path):
    sup = FleetSupervisor(tmp_path, make_controller=_scripted_controller_factory([("completed", 0)]))
    assert sup.controller_for("nope") is None
    assert sup.bus_for("nope") is None


@pytest.mark.asyncio
async def test_concurrency_cap_queues_excess(tmp_path):
    started = []
    release = asyncio.Event()

    class Fake:
        def __init__(self, bus, name):
            self.events = bus
            self.name = name
        def stop(self):
            release.set()
        async def run(self, task, resume_session_id=None):
            started.append(self.name)
            self.events.emit_state("running")
            await release.wait()          # block so the slot stays occupied
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Fake(bus, agent_record["name"])
        return c, c.run(subject)

    sup = FleetSupervisor(tmp_path, make_controller=make_controller, max_concurrent=2)
    for name in ("a", "b", "c"):
        await sup.spawn(name, "x")
    await asyncio.sleep(0.05)
    # only 2 started; the third is queued
    assert len(started) == 2
    states = {s.agent: s.state for s in sup.sessions()}
    assert list(states.values()).count("running") == 2
    assert states["c"] == "queued"

    release.set()                          # let the two running finish
    await asyncio.sleep(0.05)
    # the queued one now started
    assert "c" in started
    await sup.shutdown()


@pytest.mark.asyncio
async def test_pending_question_is_cached_and_cleared_on_non_awaiting_state(tmp_path):
    """Critical #1 (supervisor half): a question MESSAGE arriving on a
    session's bus must be cached on SessionInfo (there is no bus replay for
    a view that mounts later), and the cache must clear once the session
    moves past awaiting_input."""
    from janus.core.events import Event, EventType

    class Fake:
        def __init__(self, bus):
            self.events = bus
        def stop(self):
            pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            await asyncio.sleep(0)
            self.events.emit(
                Event(EventType.MESSAGE,
                      {"type": "question", "text": "Approve?", "choices": ["Yes", "No"]})
            )
            self.events.emit_state("awaiting_input")
            # block here until the test resumes it via the bus
            await asyncio.sleep(0.05)
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Fake(bus)
        return c, c.run(subject)

    sup = FleetSupervisor(tmp_path, make_controller=make_controller)
    sid = await sup.spawn("alpha", "x")
    await asyncio.sleep(0.02)
    info = sup.sessions()[0]
    assert info.state == "awaiting_input"
    assert info.pending_question == ("Approve?", ["Yes", "No"])

    await asyncio.sleep(0.06)
    info = sup.sessions()[0]
    assert info.state == "completed"
    assert info.pending_question is None
    await sup.shutdown()


@pytest.mark.asyncio
async def test_shutdown_does_not_start_queued_sessions(tmp_path):
    """Critical #2: shutdown() cancels+awaits running tasks; awaiting a
    cancelled task runs the finally-clause -> _maybe_start_queued(), which
    must NOT start a still-queued session mid-shutdown."""
    release = asyncio.Event()  # never set -> A blocks forever until cancelled

    class Blocker:
        def __init__(self, bus):
            self.events = bus
        def stop(self):
            pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            await release.wait()
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Blocker(bus)
        return c, c.run(subject)

    sup = FleetSupervisor(tmp_path, make_controller=make_controller, max_concurrent=1)
    a = await sup.spawn("a", "x")
    b = await sup.spawn("b", "y")
    await asyncio.sleep(0.02)
    states = {s.id: s.state for s in sup.sessions()}
    assert states[a] == "running"
    assert states[b] == "queued"

    await sup.shutdown()

    assert sup.controller_for(b) is None
    b_info = next(s for s in sup.sessions() if s.id == b)
    assert b_info.state != "running"


@pytest.mark.asyncio
async def test_a_session_crash_does_not_disturb_neighbors(tmp_path):
    class Crasher:
        def __init__(self, bus): self.events = bus
        def stop(self): pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            raise RuntimeError("boom")

    class Finisher:
        def __init__(self, bus): self.events = bus
        def stop(self): pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            await asyncio.sleep(0)
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Crasher(bus) if agent_record["name"] == "bad" else Finisher(bus)
        return c, c.run(subject)

    sup = FleetSupervisor(tmp_path, make_controller=make_controller, max_concurrent=3)
    await sup.spawn("bad", "x")
    await sup.spawn("good", "y")
    await asyncio.sleep(0.05)
    states = {s.agent: s.state for s in sup.sessions()}
    assert states["bad"] == "error"
    assert states["good"] == "completed"   # neighbor unaffected
    await sup.shutdown()


@pytest.mark.asyncio
async def test_spawn_time_factory_exception_is_isolated_not_raised(tmp_path):
    """Important #2: a factory exception at spawn time (e.g. a corrupt
    persona dir raising inside _make_controller) must not propagate out of
    spawn() and must not starve a subsequently queued good session."""

    _good_factory = _scripted_controller_factory([("running", 0), ("completed", 0.1)])

    def make_controller(agent_record, subject, bus):
        if agent_record["name"] == "bad":
            raise RuntimeError("corrupt persona dir")
        return _good_factory(agent_record, subject, bus)

    sup = FleetSupervisor(tmp_path, make_controller=make_controller, max_concurrent=1)
    sid = await sup.spawn("bad", "x")   # must not raise
    assert sid
    bad_info = sup.sessions()[0]
    assert bad_info.state == "error"

    good_id = await sup.spawn("good", "y")
    await asyncio.sleep(0.05)
    good_info = next(s for s in sup.sessions() if s.id == good_id)
    assert good_info.state == "completed"
    await sup.shutdown()


@pytest.mark.asyncio
async def test_spawn_improve_registers_improve_kind_and_runs(tmp_path):
    seen = {}

    def fake_improve(agent_record, complaint, bus):
        seen["agent"] = agent_record["name"]
        seen["complaint"] = complaint

        class _Ctl:
            def stop(self): pass
        async def _run():
            bus.emit_state("running")
            bus.emit_state("completed")
        return _Ctl(), _run()

    sup = FleetSupervisor(tmp_path, make_improve_controller=fake_improve)
    sid = await sup.spawn_improve("alpha", "sources are weak")
    await asyncio.sleep(0.05)
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.kind == "improve"
    assert seen == {"agent": "alpha", "complaint": "sources are weak"}
    assert info.state == "completed"


@pytest.mark.asyncio
async def test_improve_builder_crash_is_isolated(tmp_path):
    def boom(agent_record, complaint, bus):
        raise RuntimeError("no factory persona")

    sup = FleetSupervisor(tmp_path, make_improve_controller=boom)
    sid = await sup.spawn_improve("alpha", "x")
    await asyncio.sleep(0.05)
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.state == "error"  # crash contained, app not affected


@pytest.mark.asyncio
async def test_spawn_validate_records_pass_and_returns_outcome(tmp_path):
    reg = _make_fleet_agent(tmp_path)
    async def fake_validate(persona, rubric, root):
        return _fake_report(True)
    sup = FleetSupervisor(tmp_path, validate_fn=fake_validate)
    sid = await sup.spawn_validate("alpha")
    outcome = await sup.validation_result(sid)
    assert outcome.smoke_passed and outcome.judge_passed
    assert outcome.scores == {"form": 0.9}
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.state == "validated"
    # scores were appended to the registry (drives LAST VALIDATION)
    hist = reg.get("alpha")["validation_history"]
    assert hist[-1]["passed"] is True and hist[-1]["scores"] == {"form": 0.9}


@pytest.mark.asyncio
async def test_spawn_validate_fail_sets_failed_state(tmp_path):
    _make_fleet_agent(tmp_path)
    async def fake_validate(persona, rubric, root):
        return _fake_report(False)
    sup = FleetSupervisor(tmp_path, validate_fn=fake_validate)
    sid = await sup.spawn_validate("alpha")
    outcome = await sup.validation_result(sid)
    assert not outcome.judge_passed
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.state == "failed"


@pytest.mark.asyncio
async def test_validate_no_rubric_is_error_not_crash(tmp_path):
    reg = _make_fleet_agent(tmp_path, with_rubric=False)
    called = {"n": 0}
    async def fake_validate(persona, rubric, root):
        called["n"] += 1
        return _fake_report(True)
    sup = FleetSupervisor(tmp_path, validate_fn=fake_validate)
    sid = await sup.spawn_validate("alpha")
    outcome = await sup.validation_result(sid)
    assert outcome.error and "rubric" in outcome.error.lower()
    assert called["n"] == 0  # harness never ran
    assert len(reg.get("alpha")["validation_history"]) == 0  # registry untouched
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.state == "error"


@pytest.mark.asyncio
async def test_validate_counts_toward_the_cap(tmp_path):
    _make_fleet_agent(tmp_path)
    gate = asyncio.Event()
    async def slow_validate(persona, rubric, root):
        await gate.wait()
        return _fake_report(True)
    # cap 1: a validate in flight must keep a run queued
    sup = FleetSupervisor(tmp_path, validate_fn=slow_validate,
                          make_controller=lambda *a: (_NoopCtl(), _forever()), max_concurrent=1)
    vsid = await sup.spawn_validate("alpha")
    rsid = await sup.spawn("alpha", "subject")
    await asyncio.sleep(0.02)
    assert next(s for s in sup.sessions() if s.id == vsid).state == "validating"
    assert next(s for s in sup.sessions() if s.id == rsid).state == "queued"
    gate.set()
    await sup.validation_result(vsid)
    await sup.shutdown()


@pytest.mark.asyncio
async def test_shutdown_settles_done_for_never_run_validate_sessions(tmp_path):
    """Important #1: a validate session that never got past 'queued' (its
    task was cancelled, or it was never started) must still have `done` set
    by shutdown() — otherwise any awaiter of validation_result() (e.g. the
    dashboard's own validate action mid-spawn) hangs forever."""
    _make_fleet_agent(tmp_path)

    sup = FleetSupervisor(tmp_path, make_controller=lambda *a: (_NoopCtl(), _forever()),
                          max_concurrent=1)
    await sup.spawn("alpha", "x")           # occupies the single slot
    vsid = await sup.spawn_validate("alpha")  # queues; done unset, never runs
    await asyncio.sleep(0.02)
    info = next(s for s in sup.sessions() if s.id == vsid)
    assert info.state == "queued"

    await sup.shutdown()

    outcome = await asyncio.wait_for(sup.validation_result(vsid), timeout=1.0)
    assert outcome.error


@pytest.mark.asyncio
async def test_double_spawn_validate_is_idempotent_while_inflight(tmp_path):
    reg = _make_fleet_agent(tmp_path)
    gate = asyncio.Event()
    calls = {"n": 0}
    async def slow_validate(persona, rubric, root):
        calls["n"] += 1
        await gate.wait()
        return _fake_report(True)
    sup = FleetSupervisor(tmp_path, validate_fn=slow_validate)
    a = await sup.spawn_validate("alpha")
    b = await sup.spawn_validate("alpha")   # in flight → same session
    assert a == b
    assert len([s for s in sup.sessions() if s.agent == "alpha"]) == 1
    gate.set()
    await sup.validation_result(a)
    assert calls["n"] == 1                   # harness ran once, one registry append
    assert len(reg.get("alpha")["validation_history"]) == 1
    c = await sup.spawn_validate("alpha")    # prior is terminal → new session
    assert c != a


@pytest.mark.asyncio
async def test_double_spawn_improve_is_idempotent_while_inflight(tmp_path):
    started = {"n": 0}
    gate = asyncio.Event()
    def fake_improve(agent_record, complaint, bus):
        started["n"] += 1
        class _Ctl:
            def stop(self): pass
        async def _run():
            bus.emit_state("running")
            await gate.wait()
            bus.emit_state("completed")
        return _Ctl(), _run()
    sup = FleetSupervisor(tmp_path, make_improve_controller=fake_improve)
    a = await sup.spawn_improve("alpha", "x")
    await asyncio.sleep(0.02)
    b = await sup.spawn_improve("alpha", "y")   # in flight → same session
    assert a == b and started["n"] == 1
    gate.set()
    await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_sessions_pruned_to_live_plus_one_terminal(tmp_path):
    # three completed runs for one agent collapse to the most-recent terminal
    async def _done():
        pass
    def quick(agent_record, subject, bus):
        class _Ctl:
            def stop(self): pass
        async def _run():
            bus.emit_state("running"); bus.emit_state("completed")
        return _Ctl(), _run()
    sup = FleetSupervisor(tmp_path, make_controller=quick)
    ids = []
    for _ in range(3):
        ids.append(await sup.spawn("alpha", "s"))
        await asyncio.sleep(0.02)
    alpha = [s for s in sup.sessions() if s.agent == "alpha"]
    assert len(alpha) == 1 and alpha[0].id == ids[-1]


@pytest.mark.asyncio
async def test_prune_keeps_last_completed_not_last_spawned(tmp_path):
    """Important bug: pruning kept the terminal session that was spawned
    last (later in self._order), not the one that actually completed last.
    Two concurrent same-agent run sessions that finish out of spawn order
    must retain the one that completed last."""
    events = {}

    def make_controller(agent_record, subject, bus):
        ev = asyncio.Event()
        events[subject] = ev

        class _Ctl:
            def stop(self): pass

        async def _run():
            bus.emit_state("running")
            await ev.wait()
            bus.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

        return _Ctl(), _run()

    sup = FleetSupervisor(tmp_path, make_controller=make_controller, max_concurrent=2)
    a = await sup.spawn("alpha", "A")   # spawned first
    b = await sup.spawn("alpha", "B")   # spawned second
    await asyncio.sleep(0.02)

    events["B"].set()                   # B completes first (spawned later)
    await asyncio.sleep(0.02)
    events["A"].set()                   # A completes last (spawned first)
    await asyncio.sleep(0.02)

    alpha = [s for s in sup.sessions() if s.agent == "alpha"]
    assert len(alpha) == 1 and alpha[0].id == a, (
        "pruning must keep the last-COMPLETED session (A), not the "
        "last-SPAWNED session (B)"
    )
    await sup.shutdown()


@pytest.mark.asyncio
async def test_validate_session_has_no_live_bus(tmp_path):
    _make_fleet_agent(tmp_path)
    async def ok(persona, rubric, root):
        return _fake_report(True)
    sup = FleetSupervisor(tmp_path, validate_fn=ok)
    sid = await sup.spawn_validate("alpha")
    assert sup.bus_for(sid) is None
    await sup.validation_result(sid)


@pytest.mark.asyncio
async def test_bool_cost_is_ignored(tmp_path):
    sup = FleetSupervisor(tmp_path)
    sid = await sup.spawn("alpha", "s") if False else None  # no run needed
    # drive the state listener directly with a boolean cost
    listener = sup._make_state_listener("x")
    sup._sessions["x"] = _Session(
        info=SessionInfo(id="x", agent="alpha"), bus=None, subject="")
    from janus.core.events import Event, EventType
    listener(Event(EventType.STATE_CHANGED, {"state": "running", "cost_usd": True}))
    assert sup._sessions["x"].info.cost_usd == 0.0


def test_improve_sessions_dir_is_under_fleet_dir(tmp_path):
    sup = FleetSupervisor(tmp_path)
    assert sup._improve_sessions_dir() == tmp_path / ".janus" / "sessions"


@pytest.mark.asyncio
async def test_spawn_containerize_runs_factory_with_intent_in_task():
    captured = {}

    def make_containerize(agent_record, subject, bus):
        captured["agent"] = agent_record["name"]
        captured["subject"] = subject

        class C:
            def __init__(self, bus): self.events = bus
            def enable_user_replies(self): pass
            def stop(self): pass
            async def run(self, task, resume_session_id=None):
                self.events.emit_state("running")
                self.events.emit_state("completed")
                return {"status": "completed"}
        c = C(bus)
        return c, c.run(subject)

    sup = FleetSupervisor("/tmp/fleet-x", make_containerize_controller=make_containerize)
    sid = await sup.spawn_containerize("demo_agent", "give it curl and nmap")
    assert isinstance(sid, str)
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.kind == "containerize"
    await sup.await_session(sid)          # resolves once the session completes
    assert captured["agent"] == "demo_agent"
    assert "curl and nmap" in captured["subject"]


@pytest.mark.asyncio
async def test_await_session_returns_for_unknown_or_pending_done():
    sup = FleetSupervisor("/tmp/fleet-x")
    await sup.await_session("nope")       # unknown id → returns immediately


@pytest.mark.asyncio
async def test_await_session_resolves_when_containerize_builder_crashes():
    """A synchronous builder crash in _start sets state=error and returns
    BEFORE _run_wrapped runs — so a containerize session's `done` event must
    be settled in that branch too, or await_session() hangs forever."""
    def boom(agent_record, intent, bus):
        raise RuntimeError("no provider configured")

    sup = FleetSupervisor("/tmp/fleet-x", make_containerize_controller=boom)
    sid = await sup.spawn_containerize("demo_agent", "give it curl")
    await asyncio.wait_for(sup.await_session(sid), timeout=1.0)  # must NOT hang
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.state == "error"


async def test_spawn_container_run_batch_session_and_result(tmp_path, monkeypatch):
    import janus.fleet.supervisor as sup_mod
    from janus.core.validation.container_smoke import ContainerRunResult

    async def fake_container_run(persona, subject, workdir, *, timeout=1800, on_line=None):
        return ContainerRunResult(True, {"ok": 1}, workdir / "out" / "output.json", 0, None)
    monkeypatch.setattr(sup_mod, "container_run", fake_container_run, raising=False)

    # a containerized agent on disk
    from tests.fleet.test_sync import _make_containerized_agent
    _make_containerized_agent(tmp_path, "toolbox_stub")
    sup = FleetSupervisor(tmp_path)
    sid = await sup.spawn_container_run("toolbox_stub", "do it")
    info = next(s for s in sup.sessions() if s.id == sid)
    assert info.kind == "container_run"
    outcome = await sup.run_result(sid)
    assert outcome.success and outcome.output_path is not None


async def test_container_run_completion_frees_the_concurrency_slot(tmp_path, monkeypatch):
    """A finished container-run must prune + pull the next queued session, or a
    session queued behind it at the concurrency cap starves forever."""
    import janus.fleet.supervisor as sup_mod
    from janus.core.validation.container_smoke import ContainerRunResult

    async def fake_container_run(persona, subject, workdir, *, timeout=1800, on_line=None):
        return ContainerRunResult(True, {"ok": 1}, workdir / "out" / "output.json", 0, None)
    monkeypatch.setattr(sup_mod, "container_run", fake_container_run, raising=False)

    from tests.fleet.test_sync import _make_containerized_agent
    _make_containerized_agent(tmp_path, "first_agent")
    _make_containerized_agent(tmp_path, "second_agent")
    sup = FleetSupervisor(tmp_path, max_concurrent=1)

    first = await sup.spawn_container_run("first_agent", "do it")
    second = await sup.spawn_container_run("second_agent", "do it too")
    # At the cap of 1, the second session must be queued behind the first.
    assert next(s for s in sup.sessions() if s.id == second).state == "queued"

    await sup.run_result(first)          # first completes → frees the slot
    # Only resolves if the queue advanced; without the finally-fix `second`
    # never starts, its `done` never sets, and this times out.
    await asyncio.wait_for(sup.run_result(second), timeout=1.0)
    assert next(s for s in sup.sessions() if s.id == second).state != "queued"


async def test_container_run_streams_to_bus_and_log(tmp_path, monkeypatch):
    import janus.fleet.supervisor as sup_mod
    from janus.core.validation.container_smoke import ContainerRunResult

    async def fake_container_run(persona, subject, workdir, *, timeout=1800, on_line=None):
        if on_line:
            on_line("line-1"); on_line("line-2")
        return ContainerRunResult(True, {"ok": 1}, workdir / "out" / "output.json", 0, None)
    monkeypatch.setattr(sup_mod, "container_run", fake_container_run, raising=False)

    from tests.fleet.test_sync import _make_containerized_agent
    _make_containerized_agent(tmp_path, "toolbox_stub")
    sup = FleetSupervisor(tmp_path)
    sid = await sup.spawn_container_run("toolbox_stub", "go")
    await sup.run_result(sid)
    log = sup.container_run_log(sid)
    assert "line-1" in log and "line-2" in log
    assert any("completed" in line for line in log)          # final summary line
    assert sup.bus_for(sid) is not None                 # streamable
