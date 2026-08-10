# tests/interface/test_cli.py
import shutil
from pathlib import Path

import pytest

from janus.interface import cli


def test_run_persona_defaults_to_factory(monkeypatch):
    # `janus run --task ...` with no --persona should default the persona to
    # "factory" (Janus's own builder), not error on a missing required arg.
    captured = {}

    def fake_cmd_run(args):
        captured["persona"] = args.persona
        return 0

    monkeypatch.setattr(cli, "_cmd_run", fake_cmd_run)
    rc = cli.main(["run", "--task", "build an agent that reviews contracts"])
    assert rc == 0
    assert captured["persona"] == "factory"


def test_resolve_persona_dir_accepts_path(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    assert cli.resolve_persona_dir(str(d)) == d


def test_resolve_persona_dir_missing_raises():
    with pytest.raises(SystemExit):
        cli.resolve_persona_dir("definitely-not-a-persona-xyz")


def test_run_wires_controller_and_launches(monkeypatch, tmp_path):
    # Copy the echo_brief fixture persona and give this test's copy a real
    # banner, so the banner assertion below actually bites: asserting
    # `banner is None` would also pass if _cmd_run stopped forwarding
    # persona.banner to launch() entirely.
    src = Path("tests/fixtures/personas/echo_brief").resolve()
    persona_dir = tmp_path / "echo_brief"
    shutil.copytree(src, persona_dir, ignore=shutil.ignore_patterns("__pycache__"))
    art = "\n".join(["⣿" * 10] * 8)
    (persona_dir / "banner.txt").write_text(art + "\n")
    captured = {}

    fake_backend = object()
    monkeypatch.setattr(cli, "build_backend_for_persona", lambda cfg, p: fake_backend)

    class FakeController:
        def __init__(self, config, backend=None, session_store=None):
            captured["backend"] = backend

    monkeypatch.setattr(cli, "AgentController", FakeController)

    def fake_launch(controller, task, *, title=None, banner=None, resume_session_id=None):
        captured["task"] = task
        captured["title"] = title
        captured["banner"] = banner
        return None

    monkeypatch.setattr(cli, "launch", fake_launch)
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["run", "--persona", str(persona_dir), "--task", "widgets"])
    assert rc == 0
    assert captured["backend"] is fake_backend
    assert "widgets" in captured["task"]
    assert captured["banner"] == art  # non-None: proves banner= is really passed


def test_export_invokes_export_agent(monkeypatch, tmp_path):
    persona_dir = Path("tests/fixtures/personas/echo_brief").resolve()
    seen = {}

    def fake_export_agent(persona, dest, agent_name=None, git_init=True, force=False):
        seen["dest"] = Path(dest)
        return Path(dest)

    monkeypatch.setattr(cli, "export_agent", fake_export_agent)
    rc = cli.main(
        ["export", "--persona", str(persona_dir), "--dest", str(tmp_path / "out"), "--no-git"]
    )
    assert rc == 0
    assert seen["dest"] == tmp_path / "out"


def test_run_missing_provider_returns_1(monkeypatch, tmp_path):
    persona_dir = Path("tests/fixtures/personas/echo_brief").resolve()

    def boom(cfg, p):
        raise NotImplementedError("no provider configured")

    monkeypatch.setattr(cli, "build_backend_for_persona", boom)
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["run", "--persona", str(persona_dir), "--task", "x"])
    assert rc == 1


def test_validate_missing_provider_returns_1(monkeypatch, tmp_path):
    # echo_brief has a rubric configured, so validate() proceeds past the
    # "no rubric" early-return and reaches build_backend_for_persona (via the
    # make_agent_backend factory), which is where a missing-provider
    # NotImplementedError actually surfaces.
    persona_dir = Path("tests/fixtures/personas/echo_brief").resolve()

    def boom(cfg, p):
        raise NotImplementedError("no provider")

    monkeypatch.setattr(cli, "build_backend_for_persona", boom)
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["validate", "--persona", str(persona_dir)])
    assert rc == 1
