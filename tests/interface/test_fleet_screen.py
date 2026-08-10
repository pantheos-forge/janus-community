import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from janus.fleet.registry import FleetRegistry
from janus.fleet.supervisor import FleetSupervisor
from janus.interface.fleet_screen import FleetScreen


def _seed(fleet):
    reg = FleetRegistry(fleet)
    reg.register("alpha", domain="poetry", description="d", source="factory",
                 path=str(fleet / "alpha"), clock=lambda: "2026-07-24T00:00:00")
    reg.append_validation("alpha", scores={"form": 0.9}, passed=True, note="x",
                          clock=lambda: "2026-07-24T00:00:00")
    return reg


def _seed_two(fleet):
    reg = FleetRegistry(fleet)
    for n in ("alpha", "beta"):
        reg.register(n, domain="poetry", description="d", source="factory",
                     path=str(fleet / n), clock=lambda: "2026-07-24T00:00:00")
    reg.append_validation("alpha", scores={"form": 0.9}, passed=True, note="x",
                          clock=lambda: "2026-07-24T00:00:00")
    return reg


class _Host(App[None]):
    def __init__(self, screen):
        super().__init__()
        self._screen = screen
    def on_mount(self):
        self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_fleet_table_shows_registered_agents(tmp_path):
    reg = _seed(tmp_path)
    sup = FleetSupervisor(tmp_path, make_controller=lambda *a: (None, None))
    app = _Host(FleetScreen(reg, sup))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        # NOTE: Textual 8.2.8's App.query_one only searches the App's
        # default_screen (the one auto-created at startup), not the
        # currently active pushed screen — App._get_dom_base() returns
        # self.default_screen, and default_screen tracks _compose_screen,
        # not the screen stack top. Query through the active screen instead
        # (app.screen is the true top-of-stack, verified against source).
        table = app.screen.query_one("DataTable")
        # gather all rendered cell text
        cells = [str(table.get_cell_at((0, c))) for c in range(len(table.columns))]
        joined = " ".join(cells)
        assert "alpha" in joined and "poetry" in joined and "0.9" in joined


@pytest.mark.asyncio
async def test_fleet_table_never_collapses(tmp_path):
    reg = _seed(tmp_path)
    sup = FleetSupervisor(tmp_path, make_controller=lambda *a: (None, None))
    app = _Host(FleetScreen(reg, sup))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.2)
        table = app.screen.query_one("DataTable")
        assert table.size.height > 3  # the table fills the screen, not a sliver


from janus.fleet.supervisor import SessionInfo


class _FakeSup:
    def __init__(self, infos):
        self._infos = infos
    def sessions(self):
        return self._infos


def test_session_cell_is_kind_aware():
    infos = [
        SessionInfo(id="1", agent="a", state="validating", kind="validate"),
        SessionInfo(id="2", agent="b", state="running", kind="improve"),
        SessionInfo(id="3", agent="c", state="awaiting_input", kind="improve"),
        SessionInfo(id="4", agent="d", state="running", kind="run"),
    ]
    screen = FleetScreen(registry=None, supervisor=_FakeSup(infos))
    assert screen._session_cell("a") == "validating"
    assert screen._session_cell("b") == "improving"
    cell_c = screen._session_cell("c")
    assert "improving" in cell_c and "[awaiting]" in cell_c
    assert screen._session_cell("d") == "running"
    assert screen._session_cell("z") == "—"


@pytest.mark.asyncio
async def test_cursor_survives_periodic_refresh(tmp_path):
    reg = _seed_two(tmp_path)
    sup = FleetSupervisor(tmp_path, make_controller=lambda *a: (None, None))
    screen = FleetScreen(reg, sup)
    app = _Host(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.2)
        table = app.screen.query_one(DataTable)
        table.focus()
        await pilot.press("down")            # move cursor to row 1
        assert table.cursor_coordinate.row == 1
        screen.refresh_table()               # the 1s timer's action
        screen.refresh_table()
        await pilot.pause(0.05)
        assert table.cursor_coordinate.row == 1   # NOT snapped back to 0


@pytest.mark.asyncio
async def test_validation_cell_is_compact(tmp_path):
    reg = _seed_two(tmp_path)  # alpha has one PASS validation with scores={"form":0.9}
    sup = FleetSupervisor(tmp_path, make_controller=lambda *a: (None, None))
    app = _Host(FleetScreen(reg, sup))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.2)
        table = app.screen.query_one(DataTable)
        # alpha is row 0 (sorted); LAST VALIDATION is column index 3
        # (NAME, DOMAIN, RUNTIME, LAST VALIDATION, SESSION)
        cell = str(table.get_cell_at((0, 3)))
        assert "PASS" in cell and "μ0.90" in cell
        assert "form=" not in cell  # full per-criterion scores are NOT in the cell


@pytest.mark.asyncio
async def test_table_has_runtime_column_and_shows_stale(tmp_path):
    from janus.fleet.registry import FleetRegistry
    from tests.fleet.test_sync import _make_exported_agent

    fleet = tmp_path / "fleet"
    agent = _make_exported_agent(fleet, name="haiku_scout")     # stale vendored copy
    FleetRegistry(fleet).register("haiku_scout", domain="d", description="x",
                                  source="adopted", path=str(agent),
                                  clock=lambda: "2026-07-27T00:00:00")

    from janus.interface.fleet_app import FleetDashboardApp
    app = FleetDashboardApp(fleet)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        table = app.screen.query_one("DataTable")
        headers = [str(c.label) for c in table.columns.values()]
        assert "RUNTIME" in headers
        row = table.get_row("haiku_scout")
        assert any("stale(" in str(cell) for cell in row)


@pytest.mark.asyncio
async def test_runtime_not_recomputed_on_plain_refresh(tmp_path, monkeypatch):
    import janus.interface.fleet_screen as fs
    from janus.fleet.registry import FleetRegistry
    from tests.fleet.test_sync import _make_exported_agent

    fleet = tmp_path / "fleet"
    agent = _make_exported_agent(fleet, name="haiku_scout")
    FleetRegistry(fleet).register("haiku_scout", domain="d", description="x",
                                  source="adopted", path=str(agent),
                                  clock=lambda: "2026-07-27T00:00:00")

    calls = {"n": 0}
    real = fs.runtime_status
    def counting(path):
        calls["n"] += 1
        return real(path)
    monkeypatch.setattr(fs, "runtime_status", counting)

    from janus.interface.fleet_app import FleetDashboardApp
    app = FleetDashboardApp(fleet)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        first = calls["n"]
        assert first >= 1
        # force several plain refreshes (same rows) — cache must serve them
        screen = app.screen
        screen.refresh_table()
        screen.refresh_table()
        screen.refresh_table()
        assert calls["n"] == first
