import json
import os

import pytest

from janus.fleet import cli as fleetcli
from janus.fleet.registry import FleetRegistry


def _seed(fleet, name="alpha", domain="d"):
    reg = FleetRegistry(fleet)
    reg.register(name, domain=domain, description="an agent", source="adopted",
                 path=str(fleet / name), clock=lambda: "2026-07-24T00:00:00")
    reg.append_validation(name, scores={"coverage": 0.9}, passed=True, note="x",
                          clock=lambda: "2026-07-24T00:00:00")


def test_list_shows_registered_agents(tmp_path, capsys):
    _seed(tmp_path)
    rc = fleetcli.cmd_list(_ns(fleet_dir=str(tmp_path)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "alpha" in out and "0.9" in out


def test_list_shows_runtime_staleness_column(tmp_path, capsys):
    # a real stale exported agent (stub janus/ differs from source)
    fleet, agent = _export_and_register(tmp_path, name="haiku_scout")
    rc = fleetcli.cmd_list(_ns(fleet_dir=str(fleet)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "RUNTIME" in out                       # header
    assert "stale(" in out                        # the stale agent's status


def test_status_shows_history(tmp_path, capsys):
    _seed(tmp_path)
    rc = fleetcli.cmd_status(_ns(fleet_dir=str(tmp_path), agent="alpha"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "coverage" in out and "2026-07-24" in out


def test_status_unknown_agent_errors(tmp_path, capsys):
    rc = fleetcli.cmd_status(_ns(fleet_dir=str(tmp_path), agent="ghost"))
    assert rc == 1
    assert "ghost" in capsys.readouterr().err


def test_adopt_imports_an_exported_repo(tmp_path, capsys):
    # build a minimal "exported repo" with a loadable persona/
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_SCHEMA, GOOD_RUBRIC, GOOD_NAME)
    src = tmp_path / "some_export"
    (src / "persona").mkdir(parents=True)
    (src / "persona" / "manifest.toml").write_text(GOOD_MANIFEST)
    (src / "persona" / "prompt.md").write_text(GOOD_PROMPT)
    (src / "persona" / "output_schema.json").write_text(GOOD_SCHEMA)
    (src / "persona" / "rubric.toml").write_text(GOOD_RUBRIC)
    (src / "agent.py").write_text("# entrypoint\n")

    fleet = tmp_path / "fleet"
    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(fleet), path=str(src)))
    assert rc == 0
    reg = FleetRegistry(fleet)
    a = reg.get(GOOD_NAME)
    assert a is not None and a["source"] == "adopted"
    assert (fleet / GOOD_NAME / "agent.py").exists()


def _build_export(tmp_path):
    """A fake exported agent repo that also carries env/scratch/secret artifacts."""
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_SCHEMA, GOOD_RUBRIC)
    src = tmp_path / "some_export"
    (src / "persona").mkdir(parents=True)
    (src / "persona" / "manifest.toml").write_text(GOOD_MANIFEST)
    (src / "persona" / "prompt.md").write_text(GOOD_PROMPT)
    (src / "persona" / "output_schema.json").write_text(GOOD_SCHEMA)
    (src / "persona" / "rubric.toml").write_text(GOOD_RUBRIC)
    (src / "agent.py").write_text("# entrypoint\n")
    # kept: vendored source + git
    (src / "janus").mkdir()
    (src / "janus" / "__init__.py").write_text("# vendored\n")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/master\n")
    # excluded: non-relocatable venv, secrets, scratch
    (src / ".venv" / "bin").mkdir(parents=True)
    (src / ".venv" / "pyvenv.cfg").write_text("home = /somewhere/else\n")
    (src / ".env").write_text("OPENROUTER_API_KEY=sk-secret\n")
    (src / "runs" / "old").mkdir(parents=True)
    (src / "runs" / "old" / "output.json").write_text("{}\n")
    (src / ".janus" / "sessions").mkdir(parents=True)
    (src / ".janus" / "sessions" / "s.json").write_text("{}\n")
    return src


def test_adopt_excludes_venv_env_and_scratch(tmp_path, capsys):
    from tests.personas.factory_samples import GOOD_NAME
    src = _build_export(tmp_path)
    fleet = tmp_path / "fleet"
    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(fleet), path=str(src)))
    assert rc == 0
    dest = fleet / GOOD_NAME
    # excluded — the mis-wired venv, secrets, and scratch must NOT be copied
    assert not (dest / ".venv").exists()
    assert not (dest / ".env").exists()
    assert not (dest / "runs").exists()
    assert not (dest / ".janus").exists()


def test_adopt_keeps_source_and_git(tmp_path, capsys):
    from tests.personas.factory_samples import GOOD_NAME
    src = _build_export(tmp_path)
    fleet = tmp_path / "fleet"
    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(fleet), path=str(src)))
    assert rc == 0
    dest = fleet / GOOD_NAME
    assert (dest / "persona" / "manifest.toml").exists()
    assert (dest / "agent.py").exists()
    assert (dest / "janus" / "__init__.py").exists()
    assert (dest / ".git" / "HEAD").exists()  # improve flow commits in-repo


def test_adopt_prints_standalone_venv_hint(tmp_path, capsys):
    from tests.personas.factory_samples import GOOD_NAME
    src = _build_export(tmp_path)
    fleet = tmp_path / "fleet"
    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(fleet), path=str(src)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "pip install -e" in out
    assert GOOD_NAME in out


def test_adopt_rejects_a_non_export_dir(tmp_path, capsys):
    bad = tmp_path / "not_an_export"
    bad.mkdir()
    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(tmp_path / "fleet"), path=str(bad)))
    assert rc == 1
    assert "persona" in capsys.readouterr().err.lower()


def test_adopt_refuses_unsafe_name(tmp_path, capsys):
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_SCHEMA, GOOD_RUBRIC)
    evil_manifest = GOOD_MANIFEST.replace(
        'name = "haiku_scout"', 'name = "../evil"')
    src = tmp_path / "some_export"
    (src / "persona").mkdir(parents=True)
    (src / "persona" / "manifest.toml").write_text(evil_manifest)
    (src / "persona" / "prompt.md").write_text(GOOD_PROMPT)
    (src / "persona" / "output_schema.json").write_text(GOOD_SCHEMA)
    (src / "persona" / "rubric.toml").write_text(GOOD_RUBRIC)
    (src / "agent.py").write_text("# entrypoint\n")

    fleet = tmp_path / "fleet"
    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(fleet), path=str(src)))
    assert rc == 1
    assert "unsafe" in capsys.readouterr().err.lower()
    # nothing escaped the fleet dir (no sibling "evil" written next to it)
    assert not (tmp_path / "evil").exists()


def test_adopt_on_corrupt_registry_errors_cleanly(tmp_path, capsys):
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_SCHEMA, GOOD_RUBRIC)
    src = tmp_path / "some_export"
    (src / "persona").mkdir(parents=True)
    (src / "persona" / "manifest.toml").write_text(GOOD_MANIFEST)
    (src / "persona" / "prompt.md").write_text(GOOD_PROMPT)
    (src / "persona" / "output_schema.json").write_text(GOOD_SCHEMA)
    (src / "persona" / "rubric.toml").write_text(GOOD_RUBRIC)
    (src / "agent.py").write_text("# entrypoint\n")

    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "registry.json").write_text("{ this is not valid json")

    rc = fleetcli.cmd_adopt(_ns(fleet_dir=str(fleet), path=str(src)))
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_run_on_corrupt_registry_errors_cleanly(tmp_path, capsys):
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "registry.json").write_text("{ this is not valid json")

    rc = fleetcli.cmd_run(_ns(fleet_dir=str(fleet), agent="whatever", subject="x"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err
    assert "Traceback" not in err


def test_run_launches_with_the_personas_banner(tmp_path, monkeypatch):
    """cmd_run forwards the loaded persona's banner art into launch().

    The banner is deliberately non-None: asserting ``banner is None`` would
    also pass if cmd_run silently stopped passing ``banner=`` at all.
    """
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_SCHEMA, GOOD_RUBRIC, GOOD_NAME)

    monkeypatch.chdir(tmp_path)

    fleet = tmp_path / "fleet"
    agent_dir = fleet / GOOD_NAME
    (agent_dir / "persona").mkdir(parents=True)
    (agent_dir / "persona" / "manifest.toml").write_text(GOOD_MANIFEST)
    (agent_dir / "persona" / "prompt.md").write_text(GOOD_PROMPT)
    (agent_dir / "persona" / "output_schema.json").write_text(GOOD_SCHEMA)
    (agent_dir / "persona" / "rubric.toml").write_text(GOOD_RUBRIC)
    art = "\n".join(["⣿" * 10] * 8)
    (agent_dir / "persona" / "banner.txt").write_text(art + "\n")

    FleetRegistry(fleet).register(GOOD_NAME, domain="poetry", description="x",
                                  source="factory", path=str(agent_dir))

    captured = {}

    def fake_launch(controller, task, *, title=None, banner=None,
                    resume_session_id=None):
        captured["title"] = title
        captured["banner"] = banner

    monkeypatch.setattr(fleetcli, "launch", fake_launch, raising=False)
    monkeypatch.setattr(fleetcli, "build_backend_for_persona",
                        lambda config, persona: object(), raising=False)

    rc = fleetcli.cmd_run(_ns(fleet_dir=str(fleet), agent=GOOD_NAME, subject="dawn"))
    assert rc == 0
    assert captured["title"] == GOOD_NAME
    assert captured["banner"] == art


def test_validate_on_corrupt_registry_errors_cleanly(tmp_path, capsys):
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "registry.json").write_text("{ this is not valid json")

    rc = fleetcli.cmd_validate(_ns(fleet_dir=str(fleet), agent="whatever"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err
    assert "Traceback" not in err


def test_improve_builds_the_factory_task_and_launches(tmp_path, monkeypatch):
    """cmd_improve resolves the agent, loads the factory persona, and launches
    it on an improvement task naming the agent + complaint. We stub launch to
    capture the task rather than run a real agent."""
    from janus.fleet import cli as fleetcli
    from janus.fleet.registry import FleetRegistry

    monkeypatch.chdir(tmp_path)  # belt-and-braces: nothing should write to cwd

    fleet = tmp_path / "fleet"
    FleetRegistry(fleet).register("alpha", domain="d", description="x",
                                  source="factory", path=str(fleet / "alpha"))

    captured = {}

    def fake_launch(controller, task, *, title=None, resume_session_id=None):
        captured["task"] = task
        captured["title"] = title

    def fake_backend(config, persona):
        return object()

    monkeypatch.setattr(fleetcli, "launch", fake_launch, raising=False)
    monkeypatch.setattr(fleetcli, "build_backend_for_persona", fake_backend, raising=False)

    rc = fleetcli.cmd_improve(_ns(fleet_dir=str(fleet), agent="alpha",
                                  complaint="it drifts geographically"))
    assert rc == 0
    assert "alpha" in captured["task"]
    assert "drifts geographically" in captured["task"]
    assert "load_fleet_persona" in captured["task"]  # steers the factory to improve mode


def test_improve_wires_fleet_dir_into_the_environment(tmp_path, monkeypatch):
    """The factory tools resolve their fleet dir via a fresh load_config() call
    (JANUS_FLEET_DIR / default), not from the config object cmd_improve builds —
    so an explicit --fleet-dir only reaches load_fleet_persona/export_improved_persona
    if cmd_improve also sets the env var."""
    from janus.fleet import cli as fleetcli
    from janus.fleet.registry import FleetRegistry

    monkeypatch.chdir(tmp_path)

    fleet = tmp_path / "fleet"
    FleetRegistry(fleet).register("alpha", domain="d", description="x",
                                  source="factory", path=str(fleet / "alpha"))

    def fake_launch(controller, task, *, title=None, resume_session_id=None):
        pass

    def fake_backend(config, persona):
        return object()

    monkeypatch.setattr(fleetcli, "launch", fake_launch, raising=False)
    monkeypatch.setattr(fleetcli, "build_backend_for_persona", fake_backend, raising=False)

    rc = fleetcli.cmd_improve(_ns(fleet_dir=str(fleet), agent="alpha", complaint="x"))
    assert rc == 0
    assert os.environ["JANUS_FLEET_DIR"] == str(fleet)


def test_improve_unknown_agent_errors(tmp_path, capsys):
    from janus.fleet import cli as fleetcli
    rc = fleetcli.cmd_improve(_ns(fleet_dir=str(tmp_path / "fleet"),
                                  agent="ghost", complaint="x"))
    assert rc == 1
    assert "ghost" in capsys.readouterr().err


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _ns(**kw):
    return _NS(**kw)


def test_dashboard_without_a_tty_prints_guidance(tmp_path, monkeypatch, capsys):
    import sys as _sys

    from janus.fleet import cli as fleetcli

    monkeypatch.setattr(_sys.stdout, "isatty", lambda: False, raising=False)
    rc = fleetcli.cmd_dashboard(_ns(fleet_dir=str(tmp_path)))
    assert rc == 0
    assert "terminal" in capsys.readouterr().out.lower()


def test_dashboard_on_a_tty_launches_the_app(tmp_path, monkeypatch):
    import sys as _sys

    from janus.fleet import cli as fleetcli

    called = {}
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(_sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(fleetcli, "run_fleet_dashboard",
                        lambda fd: called.setdefault("fleet_dir", fd), raising=False)
    rc = fleetcli.cmd_dashboard(_ns(fleet_dir=str(tmp_path)))
    assert rc == 0
    assert called["fleet_dir"] == str(tmp_path)


def test_dashboard_needs_interactive_stdin_too(tmp_path, monkeypatch, capsys):
    import janus.fleet.cli as fleetcli
    from types import SimpleNamespace
    launched = {"n": 0}
    monkeypatch.setattr(fleetcli, "run_fleet_dashboard", lambda d: launched.__setitem__("n", 1))
    monkeypatch.setattr(fleetcli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(fleetcli.sys.stdin, "isatty", lambda: False)   # piped stdin
    rc = fleetcli.cmd_dashboard(SimpleNamespace(fleet_dir=str(tmp_path)))
    assert rc == 0 and launched["n"] == 0
    assert "interactive terminal" in capsys.readouterr().out


def _export_and_register(tmp_path, name="haiku_scout"):
    """Register an exported-agent repo under a fleet dir so cmd_sync can find it."""
    from tests.personas.factory_samples import (
        GOOD_MANIFEST, GOOD_PROMPT, GOOD_RUBRIC, GOOD_SCHEMA)
    fleet = tmp_path / "fleet"
    agent = fleet / name
    (agent / "persona").mkdir(parents=True)
    for fn, content in (("manifest.toml", GOOD_MANIFEST), ("prompt.md", GOOD_PROMPT),
                        ("output_schema.json", GOOD_SCHEMA), ("rubric.toml", GOOD_RUBRIC)):
        (agent / "persona" / fn).write_text(content)
    (agent / "janus").mkdir()
    (agent / "janus" / "__init__.py").write_text("# stale\n")
    import subprocess
    for c in (["init", "-q"], ["add", "-A"],
              ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"]):
        subprocess.run(["git", *c], cwd=agent, capture_output=True)
    FleetRegistry(fleet).register(name, domain="d", description="x", source="adopted",
                                  path=str(agent), clock=lambda: "2026-07-27T00:00:00")
    return fleet, agent


def test_sync_updates_a_registered_agent(tmp_path, capsys):
    fleet, agent = _export_and_register(tmp_path)
    rc = fleetcli.cmd_sync(_ns(fleet_dir=str(fleet), agent=None, dry_run=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "haiku_scout" in out and "updated" in out
    assert (agent / "janus" / "core" / "controller.py").exists()   # real runtime landed


def test_sync_dry_run_writes_nothing(tmp_path, capsys):
    fleet, agent = _export_and_register(tmp_path)
    rc = fleetcli.cmd_sync(_ns(fleet_dir=str(fleet), agent=None, dry_run=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "would update" in out
    assert not (agent / "janus" / "core").exists()                 # nothing written


def test_sync_one_named_agent(tmp_path, capsys):
    fleet, _ = _export_and_register(tmp_path, name="haiku_scout")
    rc = fleetcli.cmd_sync(_ns(fleet_dir=str(fleet), agent="haiku_scout", dry_run=False))
    assert rc == 0
    assert "haiku_scout" in capsys.readouterr().out


def test_sync_on_corrupt_registry_errors_cleanly(tmp_path, capsys):
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "registry.json").write_text("{ not json")
    rc = fleetcli.cmd_sync(_ns(fleet_dir=str(fleet), agent=None, dry_run=False))
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_sync_unknown_agent_exits_nonzero(tmp_path, capsys):
    fleet, _ = _export_and_register(tmp_path, name="haiku_scout")   # different agent registered
    rc = fleetcli.cmd_sync(_ns(fleet_dir=str(fleet), agent="ghost", dry_run=False))
    out = capsys.readouterr().out
    assert rc == 1
    assert "ghost" in out and "error" in out


def test_sync_force_flag_reaches_the_agent(tmp_path, capsys):
    fleet, agent = _export_and_register(tmp_path)
    (agent / "persona" / "prompt.md").write_text("dirty\n")   # would skip without --force
    rc = fleetcli.cmd_sync(_ns(fleet_dir=str(fleet), agent=None, dry_run=False, force=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "updated" in out                                   # forced through the dirty repo


def test_cmd_rename_happy_path(tmp_path, capsys):
    from types import SimpleNamespace
    from janus.fleet.cli import cmd_rename
    # seed an agent via the shared helper used by the rename tests
    from tests.fleet.test_rename import _make_agent
    _make_agent(tmp_path, "haiku_scout")
    rc = cmd_rename(SimpleNamespace(fleet_dir=str(tmp_path), old="haiku_scout", new="sonnet_scout"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "sonnet_scout" in out
    from janus.fleet.registry import FleetRegistry
    assert FleetRegistry(tmp_path).get("sonnet_scout") is not None


def test_cmd_rename_error_returns_1(tmp_path, capsys):
    from types import SimpleNamespace
    from janus.fleet.cli import cmd_rename
    rc = cmd_rename(SimpleNamespace(fleet_dir=str(tmp_path), old="nope", new="whatever"))
    assert rc == 1
    assert "nope" in capsys.readouterr().err


def test_cmd_remove_deregisters(tmp_path, capsys):
    from types import SimpleNamespace
    from janus.fleet.cli import cmd_remove
    from tests.fleet.test_rename import _make_agent
    _make_agent(tmp_path, "haiku_scout")
    rc = cmd_remove(SimpleNamespace(fleet_dir=str(tmp_path), name="haiku_scout",
                                    purge=False, yes=False))
    assert rc == 0
    assert FleetRegistry(tmp_path).get("haiku_scout") is None
    assert (tmp_path / "haiku_scout").exists()
    assert "re-adopt" in capsys.readouterr().out


def test_cmd_remove_purge_requires_yes(tmp_path, capsys):
    from types import SimpleNamespace
    from janus.fleet.cli import cmd_remove
    from tests.fleet.test_rename import _make_agent
    _make_agent(tmp_path, "haiku_scout")
    rc = cmd_remove(SimpleNamespace(fleet_dir=str(tmp_path), name="haiku_scout",
                                    purge=True, yes=False))
    assert rc == 1
    assert (tmp_path / "haiku_scout").exists()                 # NOT deleted
    assert "--yes" in capsys.readouterr().err


def test_cmd_remove_purge_with_yes_deletes(tmp_path):
    from types import SimpleNamespace
    from janus.fleet.cli import cmd_remove
    from tests.fleet.test_rename import _make_agent
    _make_agent(tmp_path, "haiku_scout")
    rc = cmd_remove(SimpleNamespace(fleet_dir=str(tmp_path), name="haiku_scout",
                                    purge=True, yes=True))
    assert rc == 0
    assert not (tmp_path / "haiku_scout").exists()


def test_cmd_run_containerized_routes_to_container(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import janus.fleet.cli as cli
    from tests.fleet.test_sync import _make_containerized_agent
    from janus.core.validation.container_smoke import ContainerRunResult
    agent_dir = _make_containerized_agent(tmp_path, "toolbox_stub")
    FleetRegistry(tmp_path).register("toolbox_stub", domain="d", description="x",
                                     source="factory", path=str(agent_dir))
    called = {}
    async def fake_container_run(persona, subject, workdir, *, timeout=1800, on_line=None):
        called["subject"] = subject
        return ContainerRunResult(True, {"ok": 1}, workdir / "out" / "output.json", 0, None)
    monkeypatch.setattr(cli, "container_run", fake_container_run, raising=False)
    monkeypatch.setattr(cli, "docker_available", lambda: True, raising=False)
    rc = cli.cmd_run(SimpleNamespace(fleet_dir=str(tmp_path), agent="toolbox_stub", subject="go"))
    assert rc == 0
    assert called["subject"] == "go"
    assert "in-container" in capsys.readouterr().out


def test_cmd_run_containerized_streams_stdout(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import janus.fleet.cli as cli
    from tests.fleet.test_sync import _make_containerized_agent
    from janus.core.validation.container_smoke import ContainerRunResult
    _make_containerized_agent(tmp_path, "toolbox_stub")
    FleetRegistry(tmp_path).register("toolbox_stub", domain="d", description="x",
                                     source="factory", path=str(tmp_path / "toolbox_stub"))
    async def fake_container_run(persona, subject, workdir, *, timeout=1800, on_line=None):
        if on_line:
            on_line("[tool:start] bash")
        return ContainerRunResult(True, {"ok": 1}, workdir / "out" / "output.json", 0, None)
    monkeypatch.setattr(cli, "container_run", fake_container_run, raising=False)
    monkeypatch.setattr(cli, "docker_available", lambda: True, raising=False)
    rc = cli.cmd_run(SimpleNamespace(fleet_dir=str(tmp_path), agent="toolbox_stub", subject="go"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "[tool:start] bash" in out          # streamed live
    assert "in-container" in out               # final deliverable line
