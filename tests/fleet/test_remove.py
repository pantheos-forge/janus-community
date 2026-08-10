import pytest

from janus.fleet.registry import FleetRegistry
from janus.fleet.remove import RemoveError, remove_agent
from tests.fleet.test_rename import _make_agent


def test_remove_deregisters_keeps_dir(tmp_path):
    _make_agent(tmp_path, "haiku_scout")
    res = remove_agent(tmp_path, "haiku_scout")            # purge=False
    assert res.purged is False and res.dir_deleted is False
    assert FleetRegistry(tmp_path).get("haiku_scout") is None
    assert (tmp_path / "haiku_scout").exists()             # dir kept


def test_remove_purge_deletes_dir(tmp_path):
    _make_agent(tmp_path, "haiku_scout")
    res = remove_agent(tmp_path, "haiku_scout", purge=True)
    assert res.purged and res.dir_deleted
    assert FleetRegistry(tmp_path).get("haiku_scout") is None
    assert not (tmp_path / "haiku_scout").exists()


def test_remove_unknown_raises(tmp_path):
    with pytest.raises(RemoveError):
        remove_agent(tmp_path, "nope")


def test_remove_purge_rmtree_failure_still_deregisters(tmp_path, monkeypatch):
    _make_agent(tmp_path, "haiku_scout")
    import janus.fleet.remove as m
    def boom(*a, **k):
        raise OSError("busy")
    monkeypatch.setattr(m.shutil, "rmtree", boom)
    res = remove_agent(tmp_path, "haiku_scout", purge=True)
    assert res.purged and res.dir_deleted is False         # deregistered anyway
    assert FleetRegistry(tmp_path).get("haiku_scout") is None
    assert (tmp_path / "haiku_scout").exists()              # dir survived the failed delete


def test_purge_falls_back_to_docker_when_rmtree_is_permission_denied(tmp_path, monkeypatch):
    """A containerized agent's runs/ holds root-owned files (the container runs
    as root and writes into the bind mount), so `shutil.rmtree` raises
    PermissionError and `--purge` silently fails to purge. Fall back to a
    root-capable delete inside a container."""
    import janus.fleet.remove as m

    _make_agent(tmp_path, "haiku_scout")
    monkeypatch.setattr(
        m.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "denied"))
    )

    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        # emulate the container clearing the directory's contents
        for child in sorted((tmp_path / "haiku_scout").rglob("*"), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink()
        return m.subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    res = remove_agent(tmp_path, "haiku_scout", purge=True)
    assert res.dir_deleted is True
    assert not (tmp_path / "haiku_scout").exists()
    assert calls and calls[0][0] == "docker", "fallback must go through docker"
    assert any(str(tmp_path / "haiku_scout") in part for part in calls[0]), "must mount the agent dir"


def test_purge_reports_not_deleted_when_the_docker_fallback_also_fails(tmp_path, monkeypatch):
    """No Docker (or a failing container) must degrade to today's behaviour:
    deregistered, dir left on disk, dir_deleted False — never a traceback."""
    import janus.fleet.remove as m

    _make_agent(tmp_path, "haiku_scout")
    monkeypatch.setattr(
        m.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "denied"))
    )
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no docker"))
    )

    res = remove_agent(tmp_path, "haiku_scout", purge=True)
    assert res.purged and res.dir_deleted is False
    assert FleetRegistry(tmp_path).get("haiku_scout") is None
    assert (tmp_path / "haiku_scout").exists()


def test_docker_fallback_refuses_an_unsafe_path(tmp_path, monkeypatch):
    """The fallback runs `rm -rf` as root inside a container, so it must refuse
    obviously-wrong targets rather than trust the registry record."""
    import janus.fleet.remove as m

    calls: list[list[str]] = []
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd))
    from pathlib import Path as _P

    assert m._docker_assisted_delete(_P("/")) is False
    assert m._docker_assisted_delete(_P.home()) is False
    assert calls == [], "must not invoke docker for an unsafe path"
