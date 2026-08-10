import json

import pytest

from janus.fleet.registry import FleetRegistry, FleetRegistryError


def test_empty_registry_reads_as_no_agents(tmp_path):
    reg = FleetRegistry(tmp_path)
    assert reg.agents() == {}
    assert reg.get("nope") is None


def test_register_creates_and_persists(tmp_path):
    reg = FleetRegistry(tmp_path)
    reg.register("alpha", domain="d", description="desc", source="factory",
                 path=str(tmp_path / "alpha"), clock=lambda: "2026-07-24T00:00:00")
    on_disk = json.loads((tmp_path / "registry.json").read_text())
    a = on_disk["agents"]["alpha"]
    assert a["domain"] == "d" and a["source"] == "factory"
    assert a["created"] == "2026-07-24T00:00:00" == a["updated"]
    assert a["validation_history"] == []


def test_register_twice_preserves_created_updates_updated(tmp_path):
    reg = FleetRegistry(tmp_path)
    reg.register("a", domain="d", description="x", source="factory", path="/p",
                 clock=lambda: "2026-01-01T00:00:00")
    reg.register("a", domain="d2", description="y", source="adopted", path="/p2",
                 clock=lambda: "2026-02-02T00:00:00")
    a = reg.get("a")
    assert a["created"] == "2026-01-01T00:00:00"
    assert a["updated"] == "2026-02-02T00:00:00"
    assert a["domain"] == "d2" and a["source"] == "adopted"


def test_append_validation_history(tmp_path):
    reg = FleetRegistry(tmp_path)
    reg.register("a", domain="d", description="x", source="factory", path="/p")
    reg.append_validation("a", scores={"coverage": 0.9}, passed=True, note="baseline",
                          clock=lambda: "2026-03-03T00:00:00")
    hist = reg.get("a")["validation_history"]
    assert hist == [{"date": "2026-03-03T00:00:00", "scores": {"coverage": 0.9},
                     "passed": True, "note": "baseline"}]


def test_corrupt_registry_raises_readable_error(tmp_path):
    (tmp_path / "registry.json").write_text("{ not json")
    reg = FleetRegistry(tmp_path)
    with pytest.raises(FleetRegistryError) as ei:
        reg.load()
    assert "registry.json" in str(ei.value)


def test_append_validation_to_unknown_agent_errors(tmp_path):
    reg = FleetRegistry(tmp_path)
    with pytest.raises(FleetRegistryError):
        reg.append_validation("ghost", scores={}, passed=False, note="x")


def test_rename_swaps_key_and_updates_name_and_path(tmp_path):
    reg = FleetRegistry(tmp_path)
    reg.register("old_name", domain="d", description="x", source="factory",
                 path=str(tmp_path / "old_name"), clock=lambda: "2026-07-28T00:00:00")
    reg.append_validation("old_name", scores={"a": 0.9}, passed=True, note="v",
                          clock=lambda: "2026-07-28T00:00:01")
    reg.set_synced_to("old_name", "abc1234", clock=lambda: "2026-07-28T00:00:02")

    reg.rename("old_name", "new_name")

    assert reg.get("old_name") is None
    a = reg.get("new_name")
    assert a is not None
    assert a["name"] == "new_name"
    assert a["path"] == str(tmp_path / "new_name")
    assert a["synced_to"] == "abc1234"                 # preserved
    assert a["validation_history"] and a["validation_history"][0]["scores"] == {"a": 0.9}


def test_rename_missing_old_raises(tmp_path):
    reg = FleetRegistry(tmp_path)
    with pytest.raises(FleetRegistryError):
        reg.rename("nope", "whatever")


def test_rename_existing_new_raises(tmp_path):
    reg = FleetRegistry(tmp_path)
    for n in ("a_one", "b_two"):
        reg.register(n, domain="d", description="x", source="factory",
                     path=str(tmp_path / n), clock=lambda: "2026-07-28T00:00:00")
    with pytest.raises(FleetRegistryError):
        reg.rename("a_one", "b_two")


def test_remove_drops_the_agent(tmp_path):
    from janus.fleet.registry import FleetRegistry
    reg = FleetRegistry(tmp_path)
    for n in ("keep_me", "drop_me"):
        reg.register(n, domain="d", description="x", source="factory",
                     path=str(tmp_path / n), clock=lambda: "2026-07-28T00:00:00")
    reg.remove("drop_me")
    assert reg.get("drop_me") is None
    assert reg.get("keep_me") is not None          # others untouched


def test_remove_unknown_raises(tmp_path):
    import pytest
    from janus.fleet.registry import FleetRegistry, FleetRegistryError
    with pytest.raises(FleetRegistryError):
        FleetRegistry(tmp_path).remove("nope")
