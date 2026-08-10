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
