# Janus — an engine for building specialized AI agents.
# Copyright (C) 2026 Pantheos Forge
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. This program is distributed WITHOUT ANY WARRANTY;
# see the GNU AGPL <https://www.gnu.org/licenses/> for details.
#
# A persona exception applies — see LICENSE-EXCEPTION.

"""FleetScreen — the dashboard's agents table.

Renders every registered agent with its last validation and any live session
state, refreshing on a timer. Row activation and the r/v/i/a keys post
messages the dashboard app acts on.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header

from janus.fleet.sync import runtime_status


class FleetScreen(Screen[None]):
    """A table of fleet agents + live session state."""

    DEFAULT_CSS = """
    FleetScreen #fleet_table {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("r", "action('run')", "Run"),
        Binding("v", "action('validate')", "Validate"),
        Binding("i", "action('improve')", "Improve"),
        Binding("s", "action('sync')", "Sync"),
        Binding("c", "action('containerize')", "Containerize"),
        Binding("n", "action('rename')", "Rename"),
        Binding("x", "action('remove')", "Remove"),
        Binding("d", "action('details')", "Details"),
        Binding("a", "action('adopt')", "Adopt"),
    ]

    class AgentActivated(Message):
        def __init__(self, agent: str) -> None:
            super().__init__()
            self.agent = agent

    class ActionRequested(Message):
        def __init__(self, kind: str, agent: str | None) -> None:
            super().__init__()
            self.kind = kind
            self.agent = agent

    def __init__(self, registry: Any, supervisor: Any) -> None:
        super().__init__()
        self._registry = registry
        self._supervisor = supervisor
        self._runtime: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="fleet_table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#fleet_table", DataTable)
        table.add_columns(("NAME", "name"), ("DOMAIN", "domain"),
                          ("RUNTIME", "runtime"),
                          ("LAST VALIDATION", "validation"), ("SESSION", "session"))
        self.refresh_table()
        self.set_interval(1.0, self.refresh_table)

    @staticmethod
    def _label(kind: str, state: str) -> str:
        if kind == "validate":
            return state  # queued/validating/validated/failed/error render literally
        if kind == "improve":
            if state in ("running", "paused", "awaiting_input"):
                return "improving"
            if state == "completed":
                return "improved"
            return state
        return state  # run

    def _session_cell(self, agent_name: str) -> str:
        sessions = [s for s in self._supervisor.sessions() if s.agent == agent_name]
        if not sessions:
            return "—"
        cell = ", ".join(self._label(s.kind, s.state) for s in sessions)
        if any(s.state == "awaiting_input" for s in sessions):
            cell += " [awaiting]"
        return cell

    def _runtime_cell(self, name: str, a: dict) -> str:
        if name not in self._runtime:
            self._runtime[name] = runtime_status(a.get("path", "")).text
        return self._runtime[name]

    def _row_values(self, name: str, a: dict) -> tuple[str, str, str, str]:
        hist = a.get("validation_history") or []
        if hist:
            last = hist[-1]
            scores = last.get("scores") or {}
            mark = "PASS" if last.get("passed") else "FAIL"
            date = (last.get("date") or "")[:10]
            if scores:
                mean = sum(scores.values()) / len(scores)
                validation = f"{mark} {date} μ{mean:.2f}"
            else:
                validation = f"{mark} {date}".strip()
        else:
            validation = "(never validated)"
        return (a.get("domain", ""), self._runtime_cell(name, a),
                validation, self._session_cell(name))

    def refresh_table(self) -> None:
        table = self.query_one("#fleet_table", DataTable)
        try:
            agents = self._registry.agents()
        except Exception as e:
            table.clear()
            table.add_row("(registry error)", str(e)[:40], "", "", "")
            return
        desired = sorted(agents.items())
        desired_keys = [name for name, _ in desired]
        current_keys = [row.key.value for row in table.ordered_rows]
        if current_keys == desired_keys:
            # Same rows in the same order: update cells in place so the
            # row cursor and scroll position are left untouched.
            for name, a in desired:
                domain, runtime, validation, session = self._row_values(name, a)
                table.update_cell(name, "domain", domain, update_width=True)
                table.update_cell(name, "runtime", runtime, update_width=True)
                table.update_cell(name, "validation", validation, update_width=True)
                table.update_cell(name, "session", session, update_width=True)
            return
        # Row set changed (adopt/remove): rebuild, then restore the cursor by key.
        try:
            cursor_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            cursor_key = None
        table.clear()
        for name, a in desired:
            domain, runtime, validation, session = self._row_values(name, a)
            table.add_row(name, domain, runtime, validation, session, key=name)
        if cursor_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(cursor_key))
            except Exception:
                pass

    def _selected_agent(self) -> str | None:
        table = self.query_one("#fleet_table", DataTable)
        if table.row_count == 0:
            return None
        row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        return row_key.value

    def mark_synced(self, name: str) -> None:
        """After a sync, an agent's runtime is current — update cache + cell."""
        self._runtime[name] = "current"
        try:
            self.query_one("#fleet_table", DataTable).update_cell(
                name, "runtime", "current", update_width=True)
        except Exception:
            pass

    def action_action(self, kind: str) -> None:
        self.post_message(self.ActionRequested(kind, self._selected_agent()))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self.post_message(self.AgentActivated(event.row_key.value))
