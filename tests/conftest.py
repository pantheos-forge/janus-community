"""Shared pytest fixtures for the Janus test suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_fleet_home(tmp_path_factory, monkeypatch):
    """No test may touch the real ~/janus-agents/. Point the fleet home at a
    throwaway dir for every test; tests that need a specific fleet still pass
    their own fleet_dir explicitly (this only catches missing overrides)."""
    monkeypatch.setenv("JANUS_FLEET_DIR", str(tmp_path_factory.mktemp("fleet_home")))
