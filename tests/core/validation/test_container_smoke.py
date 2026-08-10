import subprocess
from pathlib import Path

import pytest

from janus.core.persona import Persona
from janus.core.validation.container_smoke import container_smoke_run, docker_available

_MANIFEST = '''
[persona]
name = "tinycontainer"
description = "tiny containerized persona for hermetic tests"
domain = "demo"
[prompt]
file = "prompt.md"
[tools]
builtins = ["bash"]
[task]
template = "Do: {subject}"
[output]
schema_file = "output_schema.json"
'''


def _tiny_container_persona(tmp_path):
    """A minimal containerized persona: manifest + prompt + container.toml + output_schema."""
    d = tmp_path / "tinycontainer"
    d.mkdir()
    (d / "manifest.toml").write_text(_MANIFEST)
    (d / "prompt.md").write_text("You are a tiny tool agent.")
    (d / "container.toml").write_text(
        '[install]\napt = ["ripgrep"]\n[[tool]]\nname = "rg"\ndescription = "search"\n'
    )
    (d / "output_schema.json").write_text(
        '{"type":"object","properties":{"summary":{"type":"string"}},'
        '"required":["summary"],"additionalProperties":false}'
    )
    return Persona.load(d)


@pytest.mark.asyncio
async def test_container_smoke_run_mount_source_is_absolute(tmp_path, monkeypatch):
    """A RELATIVE working_directory must still yield an ABSOLUTE `docker run -v`
    source. Docker treats a relative -v source as a named volume, and a name with
    slashes is rejected ("invalid characters for a local volume name") — which is
    exactly what broke the factory's in-container validation (its workspace is a
    relative path). Regression guard."""
    import janus.core.validation.container_smoke as m

    monkeypatch.setattr(m, "docker_available", lambda: True)
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    persona = _tiny_container_persona(tmp_path)
    monkeypatch.chdir(tmp_path)                    # so a relative path resolves here
    await container_smoke_run(persona, "subj", Path("relwork"))   # RELATIVE workdir

    run_cmd = next(c for c in calls if len(c) > 1 and c[1] == "run")
    mount = run_cmd[run_cmd.index("-v") + 1]
    source = mount.split(":")[0]
    assert source.startswith("/"), f"mount source must be absolute, got {source!r}"


@pytest.mark.asyncio
async def test_container_run_sets_pythonunbuffered(tmp_path, monkeypatch):
    """`docker run` has no TTY, so Python block-buffers stdout — the streaming
    reader (and the dashboard live log) would get nothing until exit. The run
    command must pass PYTHONUNBUFFERED=1 so lines flush live."""
    import janus.core.validation.container_smoke as m
    monkeypatch.setattr(m, "docker_available", lambda: True)
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    await container_smoke_run(_tiny_container_persona(tmp_path), "x", tmp_path / "ws")

    run_cmd = next(c for c in calls if len(c) > 1 and c[1] == "run")
    assert "PYTHONUNBUFFERED=1" in run_cmd


def test_docker_available_false_when_cli_missing(monkeypatch):
    import janus.core.validation.container_smoke as m

    def boom(*a, **k):
        raise FileNotFoundError("no docker")
    monkeypatch.setattr(m.subprocess, "run", boom)
    assert docker_available() is False


@pytest.mark.asyncio
async def test_container_smoke_run_without_docker_fails_cleanly(tmp_path, monkeypatch):
    import janus.core.validation.container_smoke as m
    monkeypatch.setattr(m, "docker_available", lambda: False)

    # a minimal containerized persona (bash + container.toml); reuse a helper
    persona = _tiny_container_persona(tmp_path)
    result = await container_smoke_run(persona, "x", tmp_path / "ws")
    assert result.passed is False
    assert any(c.name == "docker_available" and not c.ok for c in result.checks)


@pytest.mark.asyncio
async def test_container_smoke_run_build_timeout_degrades_cleanly(tmp_path, monkeypatch):
    """A stalled `docker build`/`docker run` must degrade to a failed SmokeResult,
    never raise TimeoutExpired out of container_smoke_run (violates its
    'never raises' contract)."""
    import janus.core.validation.container_smoke as m
    monkeypatch.setattr(m, "docker_available", lambda: True)

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)
    monkeypatch.setattr(m.subprocess, "run", boom)

    persona = _tiny_container_persona(tmp_path)
    result = await container_smoke_run(persona, "x", tmp_path / "ws")

    assert result.passed is False
    assert any(not c.ok for c in result.checks)


@pytest.mark.asyncio
async def test_container_smoke_run_offloads_docker_to_a_thread(tmp_path, monkeypatch):
    """The blocking docker build/run must execute OFF the event-loop thread (via
    asyncio.to_thread) so the dashboard TUI stays responsive during the
    multi-minute in-container validation. Regression guard for the capstone
    UI-freeze during a containerize run."""
    import threading

    import janus.core.validation.container_smoke as m
    monkeypatch.setattr(m, "docker_available", lambda: True)
    main_ident = threading.get_ident()
    seen_threads: list[int] = []

    def spy_run(cmd, *a, **k):
        seen_threads.append(threading.get_ident())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", spy_run)
    persona = _tiny_container_persona(tmp_path)
    await container_smoke_run(persona, "x", tmp_path / "ws")

    assert seen_threads, "docker subprocess.run was never called"
    assert all(t != main_ident for t in seen_threads), (
        "docker build/run must run off the event-loop thread (asyncio.to_thread)")


@pytest.mark.asyncio
async def test_container_smoke_default_timeout_is_generous(tmp_path, monkeypatch):
    """The default smoke timeout must be generous enough for slow-tool agents
    (wide CLI sweeps). The 900s default failed a real capstone run even
    though the container + deliverable were healthy."""
    import janus.core.validation.container_smoke as m
    monkeypatch.setattr(m, "docker_available", lambda: True)
    monkeypatch.delenv("JANUS_CONTAINER_SMOKE_TIMEOUT", raising=False)
    seen: list[int] = []

    def spy_run(cmd, *a, **k):
        seen.append(k.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", spy_run)
    await container_smoke_run(_tiny_container_persona(tmp_path), "x", tmp_path / "ws")
    assert seen and all(t >= 1800 for t in seen), f"default smoke timeout too tight: {seen}"


@pytest.mark.asyncio
async def test_container_smoke_timeout_env_override(tmp_path, monkeypatch):
    """JANUS_CONTAINER_SMOKE_TIMEOUT overrides the smoke timeout without plumbing
    through validate() and every caller."""
    import janus.core.validation.container_smoke as m
    monkeypatch.setattr(m, "docker_available", lambda: True)
    monkeypatch.setenv("JANUS_CONTAINER_SMOKE_TIMEOUT", "1234")
    seen: list[int] = []

    def spy_run(cmd, *a, **k):
        seen.append(k.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", spy_run)
    await container_smoke_run(_tiny_container_persona(tmp_path), "x", tmp_path / "ws")
    assert seen and all(t == 1234 for t in seen), f"env override not applied: {seen}"


@pytest.mark.asyncio
async def test_container_run_returns_deliverable_without_schema_gate(tmp_path, monkeypatch):
    """A run surfaces whatever output.json was produced — it does NOT schema-gate
    it the way smoke does."""
    import janus.core.validation.container_smoke as m
    from janus.core.validation.container_smoke import container_run
    monkeypatch.setattr(m, "docker_available", lambda: True)

    def fake_run(cmd, *a, **k):
        # emulate: build ok; run ok and writes a NON-schema output.json
        if cmd[:2] == ["docker", "run"]:
            out = [p for p in cmd if ":/agent/runs/" in p][0].split(":")[0]
            (Path(out)).mkdir(parents=True, exist_ok=True)
            (Path(out) / "output.json").write_text('{"anything": 1}')   # not schema-valid
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(m.subprocess, "run", fake_run)

    persona = _tiny_container_persona(tmp_path)      # has an output_schema requiring "summary"
    res = await container_run(persona, "subj", tmp_path / "ws")
    assert res.success                                # ran ok...
    assert res.deliverable == {"anything": 1}         # ...and returned the raw output (no schema gate)
    assert res.output_path is not None


@pytest.mark.asyncio
async def test_container_run_nonzero_exit_is_failure(tmp_path, monkeypatch):
    import janus.core.validation.container_smoke as m
    from janus.core.validation.container_smoke import container_run
    monkeypatch.setattr(m, "docker_available", lambda: True)

    def fake_run(cmd, *a, **k):
        rc = 0 if cmd[:2] == ["docker", "build"] else 1
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="boom")
    monkeypatch.setattr(m.subprocess, "run", fake_run)

    res = await container_run(_tiny_container_persona(tmp_path), "x", tmp_path / "ws")
    assert res.success is False
    assert res.error and "boom" in res.error


@pytest.mark.asyncio
async def test_container_run_without_docker_fails_cleanly(tmp_path, monkeypatch):
    import janus.core.validation.container_smoke as m
    from janus.core.validation.container_smoke import container_run
    monkeypatch.setattr(m, "docker_available", lambda: False)
    res = await container_run(_tiny_container_persona(tmp_path), "x", tmp_path / "ws")
    assert res.success is False and res.deliverable is None and res.error


@pytest.mark.asyncio
async def test_container_run_streams_lines_to_on_line(tmp_path, monkeypatch):
    import janus.core.validation.container_smoke as m
    from janus.core.validation.container_smoke import container_run
    monkeypatch.setattr(m, "docker_available", lambda: True)
    # build ok (batch subprocess.run); run streams via create_subprocess_exec
    def fake_build(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(m.subprocess, "run", fake_build)

    class FakeStdout:
        def __init__(self, lines): self._lines = [l.encode() for l in lines]
        def __aiter__(self): return self
        async def __anext__(self):
            if not self._lines:
                raise StopAsyncIteration
            return self._lines.pop(0)
    class FakeProc:
        def __init__(self, out, workdir):
            self.stdout = FakeStdout(out); self.returncode = 0; self._workdir = workdir
        async def wait(self):
            (self._workdir / "out").mkdir(parents=True, exist_ok=True)
            (self._workdir / "out" / "output.json").write_text('{"summary": "ok"}')
            return 0
        def kill(self): pass
    async def fake_exec(*cmd, **k):
        # the run cmd contains the -v mount source (…/out)
        out = [c for c in cmd if ":/agent/runs/" in str(c)][0].split(":")[0]
        from pathlib import Path
        return FakeProc(["[state] running\n", "[tool:start] bash\n", "[state] completed\n"],
                        Path(out).parent)
    monkeypatch.setattr(m.asyncio, "create_subprocess_exec", fake_exec)

    persona = _tiny_container_persona(tmp_path)
    seen = []
    res = await container_run(persona, "x", tmp_path / "ws", on_line=seen.append)
    assert "[tool:start] bash" in seen                 # streamed each line
    assert seen[0].startswith("building")               # build status line first
    assert res.success and res.deliverable == {"summary": "ok"}


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
@pytest.mark.asyncio
async def test_container_smoke_run_builds_and_reads_output(tmp_path):
    """End-to-end: a trivial containerized persona builds and runs in Docker.

    No LLM provider is configured in this environment, so we can't assert a
    valid deliverable (the agent exits before calling emit_output). But we can
    prove the most valuable thing for free: the tool-baked Dockerfile (apt
    package + declared tool) actually builds, and that a run was attempted.
    """
    persona = _tiny_container_persona(tmp_path)
    result = await container_smoke_run(persona, "x", tmp_path / "ws", timeout=180)

    # container_smoke.py only ever emits a check named "image_build" on the
    # build-*failure* path (it returns early with just that one check); on
    # success it proceeds straight to "run_completed". So the presence of
    # "run_completed" is itself proof the tool-baked Dockerfile (apt package
    # + declared tool) built successfully — and "image_build" being absent
    # confirms we didn't take the failure shortcut. A broken Dockerfile
    # (e.g. an empty apt-package continuation) would fail here instead.
    by = {c.name: c for c in result.checks}
    assert "image_build" not in by, (
        f"tool image failed to build: {by.get('image_build')}"
    )
    assert "run_completed" in by  # run attempted (ok may be False: no provider)


@pytest.mark.asyncio
async def test_container_run_forwards_dotenv_provider_config(tmp_path, monkeypatch):
    """A `.env`-only provider config must reach the container.

    The container sees only what we pass with `-e`. Reading `os.environ` alone
    silently dropped a `.env` configuration — the setup path README documents —
    so the containerized demo died with "No provider configured" while
    host-side runs worked. Regression guard."""
    import janus.core.validation.container_smoke as m
    from janus.core.config import JanusConfig

    monkeypatch.setattr(m, "docker_available", lambda: True)
    for key in m._PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        m, "load_config",
        lambda **kw: JanusConfig(openrouter_model="vendor/model", openrouter_api_key="sk-test"),
    )

    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    persona = _tiny_container_persona(tmp_path)
    await container_smoke_run(persona, "subj", tmp_path / "work")

    run_cmd = next(c for c in calls if len(c) > 1 and c[1] == "run")
    assert "OPENROUTER_MODEL=vendor/model" in run_cmd
    assert "OPENROUTER_API_KEY=sk-test" in run_cmd


@pytest.mark.asyncio
async def test_container_smoke_is_retryable_after_a_failed_attempt(tmp_path, monkeypatch):
    """A failed containerized validate must be retryable.

    The first attempt leaves `_export/` behind; `export_agent` refuses a
    non-empty destination, so the retry raised FileExistsError as an unhandled
    traceback. That retry is the recovery path from a config error, so the
    crash landed on exactly the user who was already stuck."""
    import janus.core.validation.container_smoke as m

    monkeypatch.setattr(m, "docker_available", lambda: True)

    def fake_run(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    persona = _tiny_container_persona(tmp_path)
    work = tmp_path / "work"

    first = await container_smoke_run(persona, "subj", work)
    assert (work / "_export").exists(), "first attempt should have exported"

    second = await container_smoke_run(persona, "subj", work)   # must not raise
    assert second.checks, "retry must produce a result, not crash"
    assert isinstance(first.passed, bool) and isinstance(second.passed, bool)
