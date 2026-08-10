from datetime import datetime

import pytest
from janus.core.session import SessionInfo, SessionStatus, SessionStore


def test_sessioninfo_roundtrip():
    info = SessionInfo(
        session_id="abc123",
        subject="EV charging market",
        created_at=datetime(2026, 7, 23, 12, 0, 0),
        task="research it",
        persona="market_research",
    )
    restored = SessionInfo.from_dict(info.to_dict())
    assert restored == info


def test_from_dict_requires_core_fields():
    with pytest.raises(ValueError, match="Missing required field"):
        SessionInfo.from_dict({"session_id": "x"})


def test_no_flags_field():
    info = SessionInfo(session_id="a", subject="s", created_at=datetime.now())
    assert not hasattr(info, "flags_found")


def test_store_create_load_delete(tmp_path):
    store = SessionStore(sessions_dir=tmp_path)
    info = store.create(subject="topic", task="do", persona="market_research")
    assert (tmp_path / f"{info.session_id}.json").exists()
    assert store.load(info.session_id).subject == "topic"
    store.delete(info.session_id)
    assert not (tmp_path / f"{info.session_id}.json").exists()


def test_store_mutators_persist(tmp_path):
    store = SessionStore(sessions_dir=tmp_path)
    info = store.create(subject="topic")
    store.add_cost(1.5)
    store.add_instruction("focus on EU")
    store.update_status(SessionStatus.COMPLETED)
    reloaded = store.load(info.session_id)
    assert reloaded.total_cost_usd == 1.5
    assert reloaded.user_instructions == ["focus on EU"]
    assert reloaded.status is SessionStatus.COMPLETED


def test_list_sessions_filter(tmp_path):
    store = SessionStore(sessions_dir=tmp_path)
    store.create(subject="a")
    store.create(subject="b")
    assert len(store.list_sessions()) == 2
    assert len(store.list_sessions(subject="a")) == 1
