import subprocess
import sys
from pathlib import Path

from janus.core.persona import Persona
from janus.factory.export import export_agent

FIXTURE = Path(__file__).parent.parent / "fixtures" / "personas" / "echo_brief"


def test_exported_repo_loads_vendored_persona_standalone(tmp_path):
    dest = export_agent(Persona.load(FIXTURE), tmp_path / "agent", git_init=False)
    # Run a fresh interpreter with cwd + PYTHONPATH pointed ONLY at the exported repo,
    # so `import janus` and the persona load resolve against the VENDORED copy.
    script = (
        "from pathlib import Path;"
        "from janus.core.persona import Persona;"
        "p = Persona.load(Path('persona'));"
        "print('NAME:' + p.name);"
        "print('TOOLS:' + ','.join(sorted(p.registry.names())))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=dest, env={"PYTHONPATH": str(dest), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NAME:echo_brief" in proc.stdout
    assert "echo_note" in proc.stdout        # the persona's custom tool
    assert "emit_output" in proc.stdout      # auto-added (output schema present)


def test_exported_agent_runs_headless_without_provider(tmp_path):
    """The generated agent.py executes end-to-end via the vendored interface.

    With no provider credentials and no TTY (subprocess), launch() selects the
    headless path; build_backend_for_persona raises NotImplementedError, which
    the runner catches -> friendly message + exit 1. This guards the generated
    runner's import + launch wiring against silent signature drift.
    """
    dest = export_agent(Persona.load(FIXTURE), tmp_path / "agent", git_init=False)
    proc = subprocess.run(
        [sys.executable, "agent.py", "some subject"],
        cwd=dest,
        env={"PYTHONPATH": str(dest), "PATH": "/usr/bin:/bin"},  # no ANTHROPIC_API_KEY etc.
        capture_output=True,
        text=True,
    )
    # Runner caught NotImplementedError from the provider selector and exited 1
    # with the friendly guidance (NOT a traceback).
    assert proc.returncode == 1, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "Configure a provider" in proc.stderr
    assert "Traceback" not in proc.stderr
