import subprocess

import pytest

from janus.fleet.registry import FleetRegistry
from janus.fleet.rename import RenameError, rename_agent
from tests.personas.factory_samples import GOOD_MANIFEST, GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _make_agent(fleet, name="haiku_scout"):
    agent = fleet / name
    (agent / "persona").mkdir(parents=True)
    manifest = GOOD_MANIFEST.replace('name = "haiku_scout"', f'name = "{name}"')
    for fn, c in (("manifest.toml", manifest), ("prompt.md", GOOD_PROMPT),
                  ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (agent / "persona" / fn).write_text(c)
    (agent / "pyproject.toml").write_text('[project]\nname = "old"\n')
    (agent / "README.md").write_text("# old\n")
    _git(["init", "-q"], agent)
    _git(["add", "-A"], agent)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], agent)
    reg = FleetRegistry(fleet)
    reg.register(name, domain="poetry", description="Writes a haiku.",
                 source="factory", path=str(agent), clock=lambda: "2026-07-28T00:00:00")
    return agent, reg


def test_rename_agent_moves_dir_manifest_registry_and_commits(tmp_path):
    _make_agent(tmp_path, "haiku_scout")
    res = rename_agent(tmp_path, "haiku_scout", "sonnet_scout")

    assert res.new == "sonnet_scout" and res.committed
    assert not (tmp_path / "haiku_scout").exists()
    new = tmp_path / "sonnet_scout"
    assert new.exists()
    assert 'name = "sonnet_scout"' in (new / "persona" / "manifest.toml").read_text()
    assert 'name = "sonnet-scout"' in (new / "pyproject.toml").read_text()   # re-rendered
    reg = FleetRegistry(tmp_path)
    assert reg.get("haiku_scout") is None
    assert reg.get("sonnet_scout")["path"] == str(new)
    # git history preserved (the init commit is still reachable) + a rename commit exists
    log = _git(["log", "--oneline"], new).stdout
    assert "rename: haiku_scout" in log and log.count("\n") >= 2


def test_rename_agent_refuses_existing_target(tmp_path):
    _make_agent(tmp_path, "haiku_scout")
    _make_agent(tmp_path, "other_one")
    with pytest.raises(RenameError):
        rename_agent(tmp_path, "haiku_scout", "other_one")


def test_rename_agent_refuses_dirty_repo(tmp_path):
    agent, _ = _make_agent(tmp_path, "haiku_scout")
    (agent / "persona" / "prompt.md").write_text("dirty change\n")   # uncommitted
    with pytest.raises(RenameError):
        rename_agent(tmp_path, "haiku_scout", "sonnet_scout")
    assert (tmp_path / "haiku_scout").exists()          # untouched


def test_rename_agent_refuses_invalid_name(tmp_path):
    _make_agent(tmp_path, "haiku_scout")
    with pytest.raises(RenameError):
        rename_agent(tmp_path, "haiku_scout", "Bad-Name")


def test_rename_agent_rolls_back_when_manifest_verify_fails(tmp_path, monkeypatch):
    _make_agent(tmp_path, "haiku_scout")
    import janus.fleet.rename as m

    def _boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(m.Persona, "load", staticmethod(_boom))
    with pytest.raises(RenameError):
        rename_agent(tmp_path, "haiku_scout", "sonnet_scout")
    assert (tmp_path / "haiku_scout").exists()           # rolled back
    assert not (tmp_path / "sonnet_scout").exists()
    # content restored, not just the directory location
    manifest = (tmp_path / "haiku_scout" / "persona" / "manifest.toml").read_text()
    assert 'name = "haiku_scout"' in manifest
    assert 'name = "sonnet_scout"' not in manifest


def test_rename_agent_rolls_back_when_registry_rename_fails(tmp_path, monkeypatch):
    _make_agent(tmp_path, "haiku_scout")
    import janus.fleet.rename as m

    # Inject a plain OSError (NOT a FleetRegistryError) so this test proves the
    # `except Exception` broadening is load-bearing: a narrow `except
    # FleetRegistryError` would let this escape and skip the rollback.
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m.FleetRegistry, "rename", _boom)
    with pytest.raises(RenameError):
        rename_agent(tmp_path, "haiku_scout", "sonnet_scout")
    assert (tmp_path / "haiku_scout").exists()           # rolled back
    assert not (tmp_path / "sonnet_scout").exists()
    manifest = (tmp_path / "haiku_scout" / "persona" / "manifest.toml").read_text()
    assert 'name = "haiku_scout"' in manifest
