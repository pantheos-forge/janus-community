import subprocess
from pathlib import Path

import pytest

from janus.fleet.registry import FleetRegistryError
from janus.fleet.sync import SyncResult, sync_agent
from tests.personas.factory_samples import GOOD_MANIFEST, GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _make_exported_agent(tmp_path, name="haiku_scout", *, git=True, stale=True):
    """A minimal exported-agent repo: persona/ + a vendored janus/ + a wrapper.

    With stale=True the vendored janus/ deliberately differs from the real source
    (a lone marker file), so a sync has something to change.
    """
    agent = tmp_path / name
    (agent / "persona").mkdir(parents=True)
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (agent / "persona" / fn).write_text(content)
    (agent / "janus").mkdir()
    if stale:
        (agent / "janus" / "__init__.py").write_text("# stale placeholder\n")
    (agent / "agent.py").write_text("# stale entrypoint\n")
    if git:
        _git(["init", "-q"], agent)
        _git(["add", "-A"], agent)
        _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], agent)
    return agent


def test_sync_agent_updates_and_commits(tmp_path):
    agent = _make_exported_agent(tmp_path)
    before = (agent / "persona" / "prompt.md").read_text()

    res = sync_agent(agent, source_sha="abc1234")

    assert isinstance(res, SyncResult)
    assert res.status == "updated"
    assert res.sha
    # real vendored package landed (source janus/ has many modules)
    assert (agent / "janus" / "core" / "controller.py").exists()
    # persona preserved byte-for-byte
    assert (agent / "persona" / "prompt.md").read_text() == before
    # commit exists with the provenance SHA in its message
    log = _git(["log", "--oneline", "-1"], agent).stdout
    assert "sync: vendored runtime" in log and "abc1234" in log


def test_sync_agent_current_makes_no_commit(tmp_path):
    agent = _make_exported_agent(tmp_path)
    sync_agent(agent, source_sha="abc1234")                 # first sync updates
    count1 = _git(["rev-list", "--count", "HEAD"], agent).stdout.strip()

    res = sync_agent(agent, source_sha="abc1234")           # nothing changed now
    count2 = _git(["rev-list", "--count", "HEAD"], agent).stdout.strip()

    assert res.status == "current"
    assert count2 == count1                                 # no empty commit


def test_sync_agent_skips_dirty_repo(tmp_path):
    agent = _make_exported_agent(tmp_path)
    (agent / "persona" / "prompt.md").write_text("locally edited, uncommitted\n")

    res = sync_agent(agent, source_sha="abc1234")

    assert res.status == "skipped"
    assert "uncommitted" in res.detail
    assert (agent / "persona" / "prompt.md").read_text() == "locally edited, uncommitted\n"


def test_sync_agent_git_inits_a_non_repo(tmp_path):
    agent = _make_exported_agent(tmp_path, git=False)
    res = sync_agent(agent, source_sha="abc1234")
    assert res.status == "updated"
    assert (agent / ".git").exists()


def test_sync_agent_errors_on_non_agent_path(tmp_path):
    res = sync_agent(tmp_path / "nope", source_sha="abc1234")
    assert res.status == "error"


def test_sync_agent_propagates_deletions(tmp_path):
    agent = _make_exported_agent(tmp_path)
    sync_agent(agent, source_sha="abc1234")                 # bring to canonical
    stale = agent / "janus" / "_removed_in_main.py"
    stale.write_text("# gone upstream\n")
    _git(["add", "-A"], agent)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "add stale"], agent)

    sync_agent(agent, source_sha="def5678")
    assert not stale.exists()


def test_sync_agent_dry_run_reports_without_writing(tmp_path):
    agent = _make_exported_agent(tmp_path)
    before_files = {p.relative_to(agent) for p in agent.rglob("*") if p.is_file()}
    rev_before = _git(["rev-list", "--count", "HEAD"], agent).stdout.strip()

    res = sync_agent(agent, source_sha="abc1234", dry_run=True)

    assert res.status == "updated"                    # would update
    assert res.changed_files                          # lists what would change
    assert res.sha is None                            # nothing committed
    # nothing written: same files, same commit count
    after_files = {p.relative_to(agent) for p in agent.rglob("*") if p.is_file()}
    assert after_files == before_files
    assert _git(["rev-list", "--count", "HEAD"], agent).stdout.strip() == rev_before


def test_sync_agent_dry_run_current_when_already_synced(tmp_path):
    agent = _make_exported_agent(tmp_path)
    sync_agent(agent, source_sha="abc1234")           # real sync -> canonical
    res = sync_agent(agent, source_sha="abc1234", dry_run=True)
    assert res.status == "current"
    assert res.changed_files == []


from janus.fleet.registry import FleetRegistry
from janus.fleet.sync import sync_fleet


def _register(fleet_dir, name, path):
    FleetRegistry(fleet_dir).register(
        name, domain="d", description="x", source="adopted", path=str(path),
        clock=lambda: "2026-07-27T00:00:00")


def test_set_synced_to_records_sha(tmp_path):
    _register(tmp_path, "alpha", tmp_path / "alpha")
    FleetRegistry(tmp_path).set_synced_to("alpha", "abc1234",
                                          clock=lambda: "2026-07-27T01:00:00")
    agent = FleetRegistry(tmp_path).get("alpha")
    assert agent["synced_to"] == "abc1234"
    assert agent["updated"] == "2026-07-27T01:00:00"


def test_sync_fleet_syncs_all_and_records_synced_to(tmp_path):
    fleet = tmp_path / "fleet"
    a1 = _make_exported_agent(fleet, name="haiku_scout")
    a2 = _make_exported_agent(fleet, name="haiku_two")
    _register(fleet, "haiku_scout", a1)
    _register(fleet, "haiku_two", a2)

    results = sync_fleet(FleetRegistry(fleet), source_sha="abc1234")

    assert {r.name for r in results} == {"haiku_scout", "haiku_two"}
    assert all(r.status == "updated" for r in results)
    assert FleetRegistry(fleet).get("haiku_scout")["synced_to"] == "abc1234"


def test_sync_fleet_only_one_agent(tmp_path):
    fleet = tmp_path / "fleet"
    a1 = _make_exported_agent(fleet, name="haiku_scout")
    _make_exported_agent(fleet, name="haiku_two")
    _register(fleet, "haiku_scout", a1)
    _register(fleet, "haiku_two", fleet / "haiku_two")

    results = sync_fleet(FleetRegistry(fleet), only="haiku_scout", source_sha="abc1234")
    assert [r.name for r in results] == ["haiku_scout"]


def test_sync_fleet_isolates_a_failing_agent(tmp_path):
    fleet = tmp_path / "fleet"
    good = _make_exported_agent(fleet, name="haiku_scout")
    _register(fleet, "haiku_scout", good)
    _register(fleet, "ghost", fleet / "does_not_exist")   # path missing

    results = sync_fleet(FleetRegistry(fleet), source_sha="abc1234")
    by = {r.name: r for r in results}
    assert by["haiku_scout"].status == "updated"
    assert by["ghost"].status == "error"                  # isolated, not fatal


def test_sync_fleet_dry_run_does_not_record_synced_to(tmp_path):
    fleet = tmp_path / "fleet"
    a1 = _make_exported_agent(fleet, name="haiku_scout")
    _register(fleet, "haiku_scout", a1)
    sync_fleet(FleetRegistry(fleet), source_sha="abc1234", dry_run=True)
    assert "synced_to" not in (FleetRegistry(fleet).get("haiku_scout") or {})


def test_sync_fleet_isolates_a_raising_agent(tmp_path, monkeypatch):
    """Prove the sync_fleet try/except wrapper itself, not just sync_agent's
    own error-return guard: monkeypatch sync_agent to raise for one agent."""
    import janus.fleet.sync as sync_mod

    fleet = tmp_path / "fleet"
    good_dir = fleet / "haiku_scout"
    bad_dir = fleet / "haiku_two"
    _register(fleet, "haiku_scout", good_dir)
    _register(fleet, "haiku_two", bad_dir)

    real_sync_agent = sync_mod.sync_agent

    def flaky(agent_dir, *, source_sha, dry_run=False, force=False):
        if Path(agent_dir) == bad_dir:
            raise RuntimeError("boom: simulated crash mid-sync")
        return real_sync_agent(agent_dir, source_sha=source_sha, dry_run=dry_run, force=force)

    _make_exported_agent(fleet, name="haiku_scout")
    monkeypatch.setattr(sync_mod, "sync_agent", flaky)

    results = sync_mod.sync_fleet(FleetRegistry(fleet), source_sha="abc1234")

    by = {r.name: r for r in results}
    assert len(results) == 2
    assert by["haiku_two"].status == "error"
    assert "boom" in by["haiku_two"].detail
    assert by["haiku_scout"].status == "updated"


def test_set_synced_to_unknown_agent_raises(tmp_path):
    _register(tmp_path, "alpha", tmp_path / "alpha")
    with pytest.raises(FleetRegistryError):
        FleetRegistry(tmp_path).set_synced_to("ghost", "abc1234")


def test_sync_fleet_reports_registry_name_not_dir_basename(tmp_path):
    fleet = tmp_path / "fleet"
    agent_dir = _make_exported_agent(fleet, name="haiku_scout")
    _register(fleet, "renamed_agent", agent_dir)

    results = sync_fleet(FleetRegistry(fleet), source_sha="abc1234")

    assert len(results) == 1
    assert results[0].name == "renamed_agent"


from janus.fleet.sync import RuntimeStatus, runtime_status, _source_sha


def test_runtime_status_current_after_sync(tmp_path):
    agent = _make_exported_agent(tmp_path)
    sync_agent(agent, source_sha="abc1234")          # bring vendored runtime to canonical
    rs = runtime_status(agent)
    assert rs.label == "current"
    assert rs.text == "current"


def test_runtime_status_stale_on_a_stale_copy(tmp_path):
    agent = _make_exported_agent(tmp_path)            # stub janus/__init__.py only -> differs
    rs = runtime_status(agent)
    assert rs.label == "stale"
    assert rs.n_changed > 0
    assert rs.text == f"stale({rs.n_changed})"


def test_runtime_status_unsynced_when_no_vendored_janus(tmp_path):
    agent = tmp_path / "stub"
    (agent / "persona").mkdir(parents=True)
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA)
    for fn, c in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT),
                  ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (agent / "persona" / fn).write_text(c)
    rs = runtime_status(agent)                        # no janus/ dir at all
    assert rs.label == "unsynced"
    assert rs.text == "unsynced"


def test_runtime_status_error_on_non_agent(tmp_path):
    rs = runtime_status(tmp_path / "nope")
    assert rs.label == "error"
    assert rs.text == "?"


_CONTAINER_MANIFEST = """\
[persona]
name = "toolbox_stub"
description = "Scans a directory."
domain = "application-security"

[prompt]
file = "prompt.md"

[task]
template = "Scan: {subject}"

[tools]
builtins = ["bash"]

[output]
schema_file = "output_schema.json"

[validation]
rubric_file = "rubric.toml"
"""

_CONTAINER_TOML = """\
[install]
apt = ["ripgrep"]

[[tool]]
name = "rg"
description = "ripgrep search"
usage = "rg -n pat ."
"""


def _make_containerized_agent(tmp_path, name="toolbox_stub"):
    from tests.personas.factory_samples import GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA
    agent = tmp_path / name
    (agent / "persona").mkdir(parents=True)
    for fn, content in (("manifest.toml", _CONTAINER_MANIFEST), ("prompt.md", GOOD_PROMPT),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC),
                        ("container.toml", _CONTAINER_TOML)):
        (agent / "persona" / fn).write_text(content)
    (agent / "janus").mkdir()
    _git(["init", "-q"], agent)
    _git(["add", "-A"], agent)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], agent)
    return agent


def test_sync_containerized_agent_writes_ubuntu_dockerfile_and_compose(tmp_path):
    from janus.core.persona import Persona
    from janus.factory.render import render_compose, render_dockerfile
    agent = _make_containerized_agent(tmp_path)

    sync_agent(agent, source_sha="abc1234")

    persona = Persona.load(agent / "persona")
    # Dockerfile is the Ubuntu tool image (NOT the slim default that clobbers it).
    dockerfile = (agent / "Dockerfile").read_text()
    assert dockerfile == render_dockerfile(persona.container)
    assert "FROM ubuntu:24.04" in dockerfile
    # docker-compose.yml is rendered with the fixed (bare-name) env passthrough.
    compose = (agent / "docker-compose.yml").read_text()
    assert compose == render_compose(agent.name)
    assert ":-}" not in compose


def test_runtime_status_current_after_sync_containerized(tmp_path):
    # Staleness detection must be container-aware too: a freshly synced
    # containerized agent is 'current', not falsely 'stale' from a slim-vs-Ubuntu
    # Dockerfile mismatch or an unchecked docker-compose.yml.
    agent = _make_containerized_agent(tmp_path)
    sync_agent(agent, source_sha="abc1234")
    rs = runtime_status(agent)
    assert rs.label == "current", rs.text


def test_source_sha_importable_from_sync():
    sha = _source_sha()
    assert isinstance(sha, str) and sha


def test_sync_agent_force_checkpoints_then_syncs(tmp_path):
    agent = _make_exported_agent(tmp_path)
    (agent / "persona" / "prompt.md").write_text("locally edited, uncommitted\n")
    rev_before = int(_git(["rev-list", "--count", "HEAD"], agent).stdout.strip())

    res = sync_agent(agent, source_sha="abc1234", force=True)

    assert res.status == "updated"
    rev_after = int(_git(["rev-list", "--count", "HEAD"], agent).stdout.strip())
    assert rev_after == rev_before + 2                # checkpoint commit + sync commit
    # the local edit is preserved in history (not destroyed by the sync)
    log = _git(["log", "--oneline"], agent).stdout
    assert "checkpoint before sync" in log
    assert "locally edited, uncommitted" in (agent / "persona" / "prompt.md").read_text()


def test_sync_agent_force_on_clean_repo_makes_no_checkpoint(tmp_path):
    agent = _make_exported_agent(tmp_path)            # clean, but stale runtime
    res = sync_agent(agent, source_sha="abc1234", force=True)
    assert res.status == "updated"
    log = _git(["log", "--oneline"], agent).stdout
    assert "checkpoint before sync" not in log        # nothing to checkpoint


def test_sync_agent_dirty_without_force_still_skips(tmp_path):
    agent = _make_exported_agent(tmp_path)
    (agent / "persona" / "prompt.md").write_text("dirty\n")
    res = sync_agent(agent, source_sha="abc1234")     # force defaults False
    assert res.status == "skipped"


def test_sync_fleet_threads_force(tmp_path):
    fleet = tmp_path / "fleet"
    a1 = _make_exported_agent(fleet, name="haiku_scout")
    (a1 / "persona" / "prompt.md").write_text("dirty\n")
    _register(fleet, "haiku_scout", a1)
    results = sync_fleet(FleetRegistry(fleet), source_sha="abc1234", force=True)
    assert results[0].status == "updated"             # force reached the agent
