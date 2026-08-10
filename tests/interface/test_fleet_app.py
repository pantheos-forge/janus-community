import asyncio

import pytest

from janus.core.events import Event, EventBus, EventType
from janus.fleet.registry import FleetRegistry
from janus.interface.components.question_panel import QuestionPanel
from janus.interface.components.session_view import SessionView
from janus.interface.fleet_app import FleetDashboardApp, PromptModal, SessionScreen, _build_dashboard_app
from janus.interface.fleet_screen import FleetScreen


def _seed(fleet):
    FleetRegistry(fleet).register("alpha", domain="poetry", description="d",
                                  source="factory", path=str(fleet / "alpha"),
                                  clock=lambda: "2026-07-24T00:00:00")


def _fake_factory():
    import asyncio

    class Fake:
        def __init__(self, bus): self.events = bus
        def enable_user_replies(self): pass
        def stop(self): pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            self.events.emit_message("working: " + task)
            await asyncio.sleep(0)
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Fake(bus)
        return c, c.run(subject)
    return make_controller


@pytest.mark.asyncio
async def test_dashboard_shows_the_fleet_table(tmp_path):
    _seed(tmp_path)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        # NOTE: Textual 8.2.8's App.query_one only searches the app's
        # default_screen, not the currently pushed/active screen. Query
        # through app.screen (the true top-of-stack) instead.
        table = app.screen.query_one("DataTable")
        # DataTable has no `column_count` in this Textual version (8.2.8);
        # use `len(table.columns)`, matching tests/interface/test_fleet_screen.py.
        cells = " ".join(str(table.get_cell_at((0, c))) for c in range(len(table.columns)))
        assert "alpha" in cells


@pytest.mark.asyncio
async def test_run_action_spawns_a_session_via_modal(tmp_path):
    _seed(tmp_path)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("r")                       # run action -> modal
        await pilot.pause(0.2)
        # type a subject and submit
        for ch in "haiku":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
        # a session now exists in the supervisor for alpha
        assert any(s.agent == "alpha" for s in app.supervisor.sessions())


@pytest.mark.asyncio
async def test_run_containerized_agent_routes_to_container_and_shows_screen(tmp_path, monkeypatch):
    _seed(tmp_path)
    (tmp_path / "alpha" / "persona").mkdir(parents=True, exist_ok=True)
    (tmp_path / "alpha" / "persona" / "container.toml").write_text("[install]\napt=[]\n")
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: True, raising=False)
    spawned = {}
    async def fake_spawn(agent, subject):
        spawned["agent"], spawned["subject"] = agent, subject
        return "crun1"
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    app.supervisor.spawn_container_run = fake_spawn
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("r")
        await pilot.pause(0.2)
        for ch in "email":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.ContainerRunScreen)
    assert spawned["agent"] == "alpha"


@pytest.mark.asyncio
async def test_run_containerized_opens_streaming_log_screen(tmp_path, monkeypatch):
    _seed(tmp_path)
    (tmp_path / "alpha" / "persona").mkdir(parents=True, exist_ok=True)
    (tmp_path / "alpha" / "persona" / "container.toml").write_text("[install]\napt=[]\n")
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: True, raising=False)
    from janus.core.events import EventBus
    bus = EventBus()
    async def fake_spawn(agent, subject):
        return "crun1"
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    app.supervisor.spawn_container_run = fake_spawn
    app.supervisor.bus_for = lambda sid: bus
    app.supervisor.container_run_log = lambda sid: ["seeded line"]
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("r")
        await pilot.pause(0.2)
        for ch in "email":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.ContainerRunScreen)   # a live screen, not a modal
        from textual.widgets import RichLog
        log = app.screen.query_one(RichLog)
        seeded = "\n".join(s.text for s in log.lines)
        assert "seeded line" in seeded                          # buffer seeded on mount
        bus.emit_message("streamed live")                       # a new line arrives over the bus
        await pilot.pause(0.1)
        streamed = "\n".join(s.text for s in log.lines)
        assert "streamed live" in streamed                      # ...and lands in the RichLog


@pytest.mark.asyncio
async def test_run_containerized_agent_blocked_without_docker(tmp_path, monkeypatch):
    _seed(tmp_path)
    (tmp_path / "alpha" / "persona").mkdir(parents=True, exist_ok=True)
    (tmp_path / "alpha" / "persona" / "container.toml").write_text("[install]\napt=[]\n")
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: False, raising=False)
    called = {"spawned": False}
    async def fake_spawn(agent, subject):
        called["spawned"] = True
        return "x"
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    app.supervisor.spawn_container_run = fake_spawn
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("r")
        await pilot.pause(0.2)
        for ch in "email":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)      # "Docker required"
    assert called["spawned"] is False                        # never ran the container


def _long_lived_factory(release: asyncio.Event):
    """A make_controller whose run() emits 'running' then blocks on `release`
    (never completing), so a spawned session is genuinely still running while
    the test switches screens — the real subject of spec §6's HARD invariant.
    """

    class Fake:
        def __init__(self, bus): self.events = bus
        def enable_user_replies(self): pass
        def stop(self): pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            self.events.emit_message("working: " + task)
            await release.wait()          # block until the test releases us
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Fake(bus)
        return c, c.run(subject)
    return make_controller


@pytest.mark.asyncio
async def test_activating_an_agent_opens_its_session_and_switching_back_preserves_it(tmp_path):
    """Spec §6 HARD invariant: switching screens never disturbs a running
    session (sessions live in the FleetSupervisor, not the screen)."""
    _seed(tmp_path)
    release = asyncio.Event()
    app = FleetDashboardApp(tmp_path, make_controller=_long_lived_factory(release))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        # Spawn a genuinely-running session for alpha.
        await app.supervisor.spawn("alpha", "haiku")
        await pilot.pause(0.2)
        running = [s for s in app.supervisor.sessions() if s.agent == "alpha"]
        assert len(running) == 1 and running[0].state == "running"
        sid = running[0].id
        controller = app.supervisor.controller_for(sid)

        # Activate the agent the way the app really receives it: select the
        # table row and press Enter, so FleetScreen emits AgentActivated via
        # its real on_data_table_row_selected path. Query through app.screen
        # (Textual 8.2.8's App.query_one skips pushed screens).
        table = app.screen.query_one("DataTable")
        table.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")            # RowSelected -> AgentActivated
        await pilot.pause(0.2)
        assert isinstance(app.screen, SessionScreen)

        # Switch back to the fleet via SessionScreen's escape binding.
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, FleetScreen)

        # The HARD invariant: same session, same id, still running, same
        # controller object — the screen switch disturbed nothing.
        after = [s for s in app.supervisor.sessions() if s.agent == "alpha"]
        assert len(after) == 1
        assert after[0].id == sid
        assert after[0].state == "running"
        assert app.supervisor.controller_for(sid) is controller

        # Release the block so the session can finish; shutdown in teardown.
        release.set()
        await pilot.pause(0.1)


def _awaiting_input_factory(release: asyncio.Event):
    """A make_controller whose run() asks a question and goes awaiting_input
    BEFORE any SessionView exists to see it, then blocks on `release` — the
    exact late-attach scenario of Critical #1."""

    class Fake:
        def __init__(self, bus): self.events = bus
        def enable_user_replies(self): pass
        def stop(self): pass
        async def run(self, task, resume_session_id=None):
            self.events.emit_state("running")
            await asyncio.sleep(0)
            self.events.emit(
                Event(EventType.MESSAGE,
                      {"type": "question", "text": "Approve the plan?",
                       "choices": ["Approve", "Reject"]})
            )
            self.events.emit_state("awaiting_input")
            await release.wait()
            self.events.emit_state("completed")
            return {"status": "completed", "session_id": "x", "cost_usd": 0}

    def make_controller(agent_record, subject, bus):
        c = Fake(bus)
        return c, c.run(subject)
    return make_controller


@pytest.mark.asyncio
async def test_late_attach_seeds_state_and_pending_question(tmp_path):
    """Critical #1: a session that asked its question and went
    awaiting_input while the user was still on FleetScreen must NOT open to
    a dead view — SessionScreen must seed the SessionView's agent_state and
    question panel from the supervisor's cached SessionInfo."""
    _seed(tmp_path)
    release = asyncio.Event()
    app = FleetDashboardApp(tmp_path, make_controller=_awaiting_input_factory(release))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await app.supervisor.spawn("alpha", "haiku")
        # let the question + awaiting_input land on the supervisor's cache
        # while no SessionView is mounted to observe it (no bus replay).
        await pilot.pause(0.2)
        info = next(s for s in app.supervisor.sessions() if s.agent == "alpha")
        assert info.state == "awaiting_input"
        assert info.pending_question == ("Approve the plan?", ["Approve", "Reject"])

        # Now activate the agent, the real way (table + Enter).
        table = app.screen.query_one("DataTable")
        table.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, SessionScreen)

        view = app.screen.query_one(SessionView)
        assert view.agent_state == "awaiting_input"
        panel = app.screen.query_one("#question_panel", QuestionPanel)
        assert panel.display is True
        assert panel.current_choices == ["Approve", "Reject"]

        release.set()
        await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_build_dashboard_app_wires_fleet_max_concurrent(tmp_path, monkeypatch):
    """Important #1: JANUS_FLEET_MAX_CONCURRENT must actually reach the
    supervisor at the production entrypoint. Testable without App.run()."""
    monkeypatch.setenv("JANUS_FLEET_MAX_CONCURRENT", "7")
    app = _build_dashboard_app(tmp_path)
    assert app.supervisor._max_concurrent == 7


@pytest.mark.asyncio
async def test_prompt_modal_escape_cancels():
    """Minor: escape on PromptModal dismisses with '' (the same value every
    caller already treats as cancel)."""
    from textual.app import App, ComposeResult

    class _Host(App[str]):
        def compose(self) -> ComposeResult:
            return iter(())

    app = _Host()
    async with app.run_test() as pilot:
        result_holder = {}

        async def _run_modal():
            result_holder["value"] = await app.push_screen_wait(PromptModal("test:"))

        app.run_worker(_run_modal())
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert result_holder.get("value") == ""


from janus.fleet.supervisor import ValidationOutcome


@pytest.mark.asyncio
async def test_pressing_v_shows_validation_result_modal(tmp_path):
    from janus.interface import fleet_app as fa
    # one registered agent
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="testing", description="d",
                                     source="factory", path=str(tmp_path / "alpha"),
                                     clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)

    async def fake_spawn_validate(agent):
        return "vsid"
    async def fake_result(sid):
        return ValidationOutcome(True, True, {"form": 0.83})
    app.supervisor.spawn_validate = fake_spawn_validate
    app.supervisor.validation_result = fake_result

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("v")
        await pilot.pause(0.2)
        modal = app.screen  # top of stack is the modal
        assert isinstance(modal, fa.ValidationResultModal)
        # Textual 8.2.8's Static/Label expose their text via `.content`, not
        # `.renderable` (which doesn't exist on this version) — read that
        # instead; the assertions on the actual rendered text are unchanged.
        text = " ".join(str(w.content) for w in modal.query("Static, Label"))
        assert "alpha" in text and "0.83" in text and "PASS" in text


@pytest.mark.asyncio
async def test_pressing_i_prompts_then_spawns_improve(tmp_path):
    from janus.interface import fleet_app as fa
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="testing", description="d",
                                     source="factory", path=str(tmp_path / "alpha"),
                                     clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    seen = {}
    async def fake_spawn_improve(agent, complaint):
        seen["agent"] = agent
        seen["complaint"] = complaint
        return "isid"
    app.supervisor.spawn_improve = fake_spawn_improve

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("i")
        await pilot.pause(0.2)
        # PromptModal is up; type a complaint and submit
        app.screen.query_one("Input").focus()
        await pilot.press("s", "l", "o", "w", "enter")
        await pilot.pause(0.2)
        assert seen == {"agent": "alpha", "complaint": "slow"}


@pytest.mark.asyncio
async def test_activating_a_validate_session_does_not_open_a_sessionview(tmp_path):
    from janus.interface import fleet_app as fa
    from janus.fleet.supervisor import SessionInfo
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="testing", description="d",
                                     source="factory", path=str(tmp_path / "alpha"),
                                     clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    # supervisor reports a single validate-kind session for alpha
    app.supervisor.sessions = lambda: [
        SessionInfo(id="v1", agent="alpha", state="validated", kind="validate")]
    async def fake_result(sid):
        return ValidationOutcome(True, True, {"form": 0.9})
    app.supervisor.validation_result = fake_result
    # The validate-modal fallback only fires once a result is ready.
    app.supervisor.validation_ready = lambda sid: True
    # Stub controller_for and bus_for so that removing the validate guard
    # would actually try to build a SessionScreen (making the guard testable).
    app.supervisor.controller_for = lambda sid: object()
    app.supervisor.bus_for = lambda sid: object()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        # The validate branch must have re-showed the result modal.
        assert isinstance(app.screen, fa.ValidationResultModal)
        # No SessionScreen was pushed; the validate modal (or the table) is on top
        from janus.interface.components.session_view import SessionView
        assert not app.screen.query("SessionView")


@pytest.mark.asyncio
async def test_activating_a_not_ready_validate_does_not_block_or_show_modal(tmp_path):
    """Critical #1: activating a still-validating (or queued-validate) row
    must NOT await validation_result — that would block the whole app pump
    (and can permanently deadlock once the cap is saturated). The handler
    must ignore the row until a result is ready."""
    from janus.interface import fleet_app as fa
    from janus.fleet.supervisor import SessionInfo
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="testing", description="d",
                                     source="factory", path=str(tmp_path / "alpha"),
                                     clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    app.supervisor.sessions = lambda: [
        SessionInfo(id="v1", agent="alpha", state="validating", kind="validate")]
    app.supervisor.validation_ready = lambda sid: False

    async def _must_not_be_called(sid):
        raise AssertionError("validation_result awaited while not ready — would block the pump")
    app.supervisor.validation_result = _must_not_be_called

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        # No modal was pushed — the row was ignored, exactly as the design
        # says ("show its ValidationResultModal if a result is available,
        # else ignore").
        assert isinstance(app.screen, fa.FleetScreen)
        assert not app.screen.query("SessionView")
        # The app pump is still alive and responsive (not frozen behind a
        # blocked await) — a subsequent key still gets processed normally.
        await pilot.press("down")
        await pilot.pause(0.1)
        assert isinstance(app.screen, fa.FleetScreen)


@pytest.mark.asyncio
async def test_activating_agent_prefers_attachable_session_over_stale_validate(tmp_path):
    """Important #2: a terminal validate session must not permanently shadow
    a live run/improve session of the same agent — the latest attachable
    (run/improve) session wins, even if a later validate has completed."""
    from janus.interface import fleet_app as fa
    from janus.fleet.supervisor import SessionInfo
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="testing", description="d",
                                     source="factory", path=str(tmp_path / "alpha"),
                                     clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    improve_info = SessionInfo(id="i1", agent="alpha", state="awaiting_input", kind="improve")
    validate_info = SessionInfo(id="v1", agent="alpha", state="validated", kind="validate")
    # improve spawned first, validate completed later -- old sessions[-1]
    # logic would pick the validate and show the (wrong) modal.
    app.supervisor.sessions = lambda: [improve_info, validate_info]
    app.supervisor.validation_ready = lambda sid: True

    async def fake_result(sid):
        raise AssertionError("validation_result should not be reached: an attachable "
                              "session exists")
    app.supervisor.validation_result = fake_result

    stub_controller = object()
    stub_bus = EventBus()
    app.supervisor.controller_for = lambda sid: stub_controller if sid == "i1" else None
    app.supervisor.bus_for = lambda sid: stub_bus if sid == "i1" else None

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.SessionScreen)
        assert not isinstance(app.screen, fa.ValidationResultModal)


@pytest.mark.asyncio
async def test_pressing_d_shows_validation_detail_modal(tmp_path):
    from janus.interface import fleet_app as fa
    from janus.fleet.registry import FleetRegistry
    reg = FleetRegistry(tmp_path)
    reg.register("alpha", domain="poetry", description="d", source="factory",
                 path=str(tmp_path / "alpha"), clock=lambda: "2026-07-25T00:00:00")
    reg.append_validation("alpha", scores={"form": 0.9, "depth": 0.8}, passed=True,
                          note="x", clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("d")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.ValidationDetailModal)
        text = " ".join(str(w.content) for w in app.screen.query("Static, Label"))
        assert "alpha" in text and "form" in text and "0.90" in text


@pytest.mark.asyncio
async def test_enter_on_idle_validated_agent_shows_detail(tmp_path):
    from janus.interface import fleet_app as fa
    from janus.fleet.registry import FleetRegistry
    reg = FleetRegistry(tmp_path)
    reg.register("alpha", domain="poetry", description="d", source="factory",
                 path=str(tmp_path / "alpha"), clock=lambda: "2026-07-25T00:00:00")
    reg.append_validation("alpha", scores={"form": 0.9}, passed=True, note="x",
                          clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)  # no live sessions for alpha
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.ValidationDetailModal)


@pytest.mark.asyncio
async def test_activating_errored_attachable_does_not_show_stale_validate_modal(tmp_path):
    """An attachable (run/improve) session that errored out (builder raised,
    so there's no controller/bus to attach to) must not fall through to the
    validate branch and pop a stale, unrelated validate's scores."""
    from janus.interface import fleet_app as fa
    from janus.fleet.supervisor import SessionInfo
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="d", description="d", source="factory",
                                     path=str(tmp_path / "alpha"), clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    # alpha has an errored run session (attachable, but no controller/bus) AND a ready validate
    app.supervisor.sessions = lambda: [
        SessionInfo(id="r1", agent="alpha", state="error", kind="run"),
        SessionInfo(id="v1", agent="alpha", state="validated", kind="validate")]
    app.supervisor.controller_for = lambda sid: None
    app.supervisor.bus_for = lambda sid: None
    app.supervisor.validation_ready = lambda sid: True

    async def _must_not_be_called(sid):
        raise AssertionError("validation_result must not be called: an errored attachable "
                              "session exists and must fully own the activation")
    app.supervisor.validation_result = _must_not_be_called

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert not isinstance(app.screen, fa.ValidationResultModal)
        assert isinstance(app.screen, fa.FleetScreen)


@pytest.mark.asyncio
async def test_double_v_shows_single_modal(tmp_path):
    """Pressing `v` twice on one agent must not stack two ValidationResultModals.

    TEST-MECHANICS NOTE (deviates from the brief's literal `await
    pilot.press("v"); await pilot.press("v")`): `pilot.press` awaits
    `_wait_for_screen`, which fully drains the app's message queue —
    including running the `@work` validate handler to completion (it pushes
    the modal via `push_screen_wait` before `press` returns). By the time
    the second `press("v")` is dispatched, the ValidationResultModal is
    already the top screen, so the FleetScreen `v` binding (which lives on
    FleetScreen, not on the modal) never fires again. Verified empirically:
    this two-`press()` version passes even with the guard reverted, i.e. it
    is vacuous — it cannot observe stacking either way.

    A real double keypress can race two `on_fleet_screen_action_requested`
    workers *before* either has pushed a screen (Textual's `@work` isn't
    exclusive by default, and posting the message is fire-and-forget). This
    test reproduces that directly: post two `ActionRequested` messages
    back-to-back (no await between them) so both workers are scheduled
    before either resolves `spawn_validate`, which mirrors two real
    rapid-fire `v` presses landing before the first modal renders. Without
    the guard in `on_fleet_screen_action_requested`'s "validate" branch,
    this reproducibly stacks 2 modals; with the guard, exactly 1.
    """
    from janus.interface import fleet_app as fa
    from janus.interface.fleet_screen import FleetScreen
    from janus.fleet.registry import FleetRegistry
    FleetRegistry(tmp_path).register("alpha", domain="d", description="d", source="factory",
                                     path=str(tmp_path / "alpha"), clock=lambda: "2026-07-25T00:00:00")
    app = fa.FleetDashboardApp(tmp_path)
    spawns = {"n": 0}

    async def one_shot(agent):
        spawns["n"] += 1
        await asyncio.sleep(0.05)  # widen the race window between the two workers
        return "vsid"

    async def result(sid):
        return ValidationOutcome(True, True, {"form": 0.9})
    app.supervisor.spawn_validate = one_shot
    app.supervisor.validation_result = result

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()
        # Fire both "v" activations before either worker has resolved
        # spawn_validate — a genuine race, unlike two serialized presses.
        app.post_message(FleetScreen.ActionRequested("validate", "alpha"))
        app.post_message(FleetScreen.ActionRequested("validate", "alpha"))
        await pilot.pause(0.5)
        # exactly one ValidationResultModal on the stack
        assert isinstance(app.screen, fa.ValidationResultModal)
        modals = [s for s in app.screen_stack if isinstance(s, fa.ValidationResultModal)]
        assert len(modals) == 1


@pytest.mark.asyncio
async def test_cross_agent_validation_not_dropped(tmp_path):
    """Validating agent B while agent A's ValidationResultModal is the top
    screen must show B's result (not silently drop it). The validate guard
    should only suppress duplicates for the SAME agent."""
    from janus.interface import fleet_app as fa
    from janus.interface.fleet_screen import FleetScreen
    from janus.fleet.registry import FleetRegistry

    reg = FleetRegistry(tmp_path)
    reg.register("alpha", domain="d", description="d", source="factory",
                 path=str(tmp_path / "alpha"), clock=lambda: "2026-07-25T00:00:00")
    reg.register("bravo", domain="d", description="d", source="factory",
                 path=str(tmp_path / "bravo"), clock=lambda: "2026-07-25T00:00:00")

    app = fa.FleetDashboardApp(tmp_path)

    async def fake_spawn_validate(agent):
        return f"vsid_{agent}"

    async def fake_result(sid):
        # Different outcome for each agent to verify we're actually showing the right one
        if "alpha" in sid:
            return ValidationOutcome(True, False, {"form": 0.5})
        else:
            return ValidationOutcome(False, True, {"form": 0.9})

    app.supervisor.spawn_validate = fake_spawn_validate
    app.supervisor.validation_result = fake_result

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.screen.query_one("DataTable").focus()

        # Press 'v' to validate alpha, showing alpha's ValidationResultModal
        await pilot.press("v")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.ValidationResultModal)
        assert app.screen._agent == "alpha"

        # Post a validate message for bravo (agent B).
        # Without the agent-aware guard, bravo's result would be silently dropped
        # (because a ValidationResultModal is already top screen).
        # With the fix, bravo's modal should appear.
        app.post_message(FleetScreen.ActionRequested("validate", "bravo"))
        await pilot.pause(0.2)

        # The top screen must be a ValidationResultModal for bravo, not alpha
        assert isinstance(app.screen, fa.ValidationResultModal)
        assert app.screen._agent == "bravo"


@pytest.mark.asyncio
async def test_sync_action_syncs_selected_agent_and_shows_modal(tmp_path):
    from janus.fleet.registry import FleetRegistry
    from janus.interface.fleet_app import FleetDashboardApp, SyncResultModal
    from tests.fleet.test_sync import _make_exported_agent

    fleet = tmp_path / "fleet"
    agent = _make_exported_agent(fleet, name="haiku_scout")     # stale vendored copy
    FleetRegistry(fleet).register("haiku_scout", domain="d", description="x",
                                  source="adopted", path=str(agent),
                                  clock=lambda: "2026-07-27T00:00:00")

    app = FleetDashboardApp(fleet, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("s")                                  # sync the selected agent
        await pilot.pause(0.3)
        assert isinstance(app.screen, SyncResultModal)
        # real runtime landed in the agent's vendored copy
        assert (agent / "janus" / "core" / "controller.py").exists()
        await pilot.press("escape")                             # close modal
        await pilot.pause()
        # the fleet table's runtime cell now reads current
        table = app.screen.query_one("DataTable")
        row = table.get_row("haiku_scout")
        assert any(str(cell) == "current" for cell in row)


@pytest.mark.asyncio
async def test_containerize_action_spawns_when_docker_present(tmp_path, monkeypatch):
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: True, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    spawned = {}
    async def fake_spawn_containerize(agent, intent):
        spawned["agent"], spawned["intent"] = agent, intent
        return "sid1"
    async def fake_await(sid):
        # The factory's export commits persona/container.toml only on success;
        # simulate a successful containerization landing that file.
        (tmp_path / "alpha" / "persona").mkdir(parents=True, exist_ok=True)
        (tmp_path / "alpha" / "persona" / "container.toml").write_text("[image]\n")
        return None
    async def fake_run_sync(agent):
        from janus.fleet.sync import SyncResult
        return SyncResult(agent, "current", "")
    app.supervisor.spawn_containerize = fake_spawn_containerize
    app.supervisor.await_session = fake_await
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app._run_sync = fake_run_sync            # avoid real git/sync in the test
        await pilot.press("c")
        await pilot.pause(0.2)
        for ch in "curl":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
        # success path: container.toml landed → success-toned SyncResultModal
        assert isinstance(app.screen, fa.SyncResultModal)
    assert spawned["agent"] == "alpha"
    assert "curl" in spawned["intent"]


@pytest.mark.asyncio
async def test_containerize_failure_shows_message_not_sync_modal(tmp_path, monkeypatch):
    """Fix 1: a session that completes WITHOUT landing persona/container.toml
    (containerization failed) must NOT pop a success-toned SyncResultModal —
    it shows a MessageModal and does not auto-sync."""
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: True, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async def fake_spawn_containerize(agent, intent): return "sid1"
    async def fake_await(sid): return None            # completes, but no container.toml
    called = {"sync": False}
    async def fake_run_sync(agent):
        called["sync"] = True
        from janus.fleet.sync import SyncResult
        return SyncResult(agent, "current", "")
    app.supervisor.spawn_containerize = fake_spawn_containerize
    app.supervisor.await_session = fake_await
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app._run_sync = fake_run_sync
        await pilot.press("c")
        await pilot.pause(0.2)
        for ch in "curl":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)
        assert not isinstance(app.screen, fa.SyncResultModal)
    assert called["sync"] is False                    # no auto-sync on failure


@pytest.mark.asyncio
async def test_containerize_blocked_when_already_containerized(tmp_path, monkeypatch):
    """Fix 3: an agent that already has persona/container.toml is refused
    (use Improve) — MessageModal shown, spawn_containerize NOT called."""
    _seed(tmp_path)
    (tmp_path / "alpha" / "persona").mkdir(parents=True, exist_ok=True)
    (tmp_path / "alpha" / "persona" / "container.toml").write_text("[image]\n")
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: True, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    called = {"spawned": False}
    async def fake_spawn_containerize(agent, intent):
        called["spawned"] = True
        return "sid"
    app.supervisor.spawn_containerize = fake_spawn_containerize
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("c")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)   # "already containerized"
    assert called["spawned"] is False


@pytest.mark.asyncio
async def test_containerize_in_flight_guard_blocks_second_press(tmp_path, monkeypatch):
    """Fix 2: a second `c` while a non-terminal containerize session already
    exists for the agent must not spawn a second awaiter+sync (concurrent git
    syncs race on index.lock) — it shows a MessageModal and returns."""
    from janus.fleet.supervisor import SessionInfo
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: True, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    # A containerize session for alpha is already running.
    app.supervisor.sessions = lambda: [
        SessionInfo(id="c1", agent="alpha", state="running", kind="containerize")]
    called = {"spawned": False}
    async def fake_spawn_containerize(agent, intent):
        called["spawned"] = True
        return "c2"
    app.supervisor.spawn_containerize = fake_spawn_containerize
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("c")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)   # "already in progress"
    assert called["spawned"] is False


@pytest.mark.asyncio
async def test_containerize_action_blocked_without_docker(tmp_path, monkeypatch):
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    monkeypatch.setattr(fa, "docker_available", lambda: False, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    called = {"spawned": False}
    async def fake_spawn_containerize(agent, intent):
        called["spawned"] = True
        return "sid"
    app.supervisor.spawn_containerize = fake_spawn_containerize
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("c")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)   # "Docker required"
    assert called["spawned"] is False


@pytest.mark.asyncio
async def test_rename_action_renames_and_refreshes(tmp_path, monkeypatch):
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    called = {}
    def fake_rename(fleet_dir, old, new):
        called["old"], called["new"] = old, new
        from janus.fleet.rename import RenameResult
        from pathlib import Path
        return RenameResult(old, new, Path(fleet_dir) / new, "abc1234", True)
    monkeypatch.setattr(fa, "rename_agent", fake_rename, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("n")
        await pilot.pause(0.2)
        for ch in "beta":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.2)
    assert called == {"old": "alpha", "new": "beta"}


@pytest.mark.asyncio
async def test_rename_action_blocked_during_live_session(tmp_path, monkeypatch):
    # NOTE: a real `supervisor.spawn` + the fast-completing `_fake_factory`
    # (a single `asyncio.sleep(0)`) reaches the terminal "completed" state
    # well within a 0.2s pilot pause, so the guard never sees it as live —
    # this raced deterministically, not flakily. Mirror the same
    # non-racy idiom already used by test_containerize_in_flight_guard_
    # blocks_second_press: stub `sessions()` to return a fixed non-terminal
    # session directly.
    from janus.fleet.supervisor import SessionInfo
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    called = {"renamed": False}
    def fake_rename(*a, **k):
        called["renamed"] = True
    monkeypatch.setattr(fa, "rename_agent", fake_rename, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    app.supervisor.sessions = lambda: [
        SessionInfo(id="r1", agent="alpha", state="running")]
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("n")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)         # refused
    assert called["renamed"] is False


@pytest.mark.asyncio
async def test_remove_action_deregisters_on_confirm(tmp_path, monkeypatch):
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    called = {}
    def fake_remove(fleet_dir, name, *, purge=False):
        called["name"], called["purge"] = name, purge
        from janus.fleet.remove import RemoveResult
        from pathlib import Path
        return RemoveResult(name, Path(fleet_dir) / name, False, False)
    monkeypatch.setattr(fa, "remove_agent", fake_remove, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("x")
        await pilot.pause(0.2)
        await pilot.press("y")               # confirm
        await pilot.pause(0.2)
    assert called == {"name": "alpha", "purge": False}


@pytest.mark.asyncio
async def test_confirm_modal_yes_is_a_clickable_button(tmp_path, monkeypatch):
    """The confirm dialog's Yes/No must be real clickable Buttons, not just keys."""
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    called = {}
    def fake_remove(fleet_dir, name, *, purge=False):
        called["name"] = name
        from pathlib import Path

        from janus.fleet.remove import RemoveResult
        return RemoveResult(name, Path(fleet_dir) / name, False, False)
    monkeypatch.setattr(fa, "remove_agent", fake_remove, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("x")
        await pilot.pause(0.2)
        await pilot.click("#confirm_yes")        # CLICK the button, not press a key
        await pilot.pause(0.2)
    assert called.get("name") == "alpha"


@pytest.mark.asyncio
async def test_remove_action_cancel_does_nothing(tmp_path, monkeypatch):
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    called = {"removed": False}
    def fake_remove(*a, **k):
        called["removed"] = True
    monkeypatch.setattr(fa, "remove_agent", fake_remove, raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("x")
        await pilot.pause(0.2)
        await pilot.press("n")               # decline
        await pilot.pause(0.2)
    assert called["removed"] is False


@pytest.mark.asyncio
async def test_remove_action_blocked_during_live_session(tmp_path, monkeypatch):
    _seed(tmp_path)
    import janus.interface.fleet_app as fa
    from janus.fleet.supervisor import SessionInfo
    called = {"removed": False}
    monkeypatch.setattr(fa, "remove_agent",
                        lambda *a, **k: called.__setitem__("removed", True), raising=False)
    app = FleetDashboardApp(tmp_path, make_controller=_fake_factory())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        app.supervisor.sessions = lambda: [SessionInfo(id="r1", agent="alpha", state="running")]
        await pilot.press("x")
        await pilot.pause(0.2)
        assert isinstance(app.screen, fa.MessageModal)
    assert called["removed"] is False
