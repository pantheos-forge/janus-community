from pathlib import Path

import pytest

from janus.core.persona import Persona
from janus.factory.export import export_agent

FIXTURE = Path(__file__).parent.parent / "fixtures" / "personas" / "echo_brief"


def test_export_creates_self_contained_structure(tmp_path):
    dest = export_agent(Persona.load(FIXTURE), tmp_path / "agent", git_init=False)
    # vendored janus (wholesale)
    assert (dest / "janus" / "core" / "persona.py").is_file()
    assert (dest / "janus" / "core" / "backends" / "generic.py").is_file()
    assert (dest / "janus" / "core" / "validation" / "harness.py").is_file()
    assert not (dest / "janus" / "core" / "__pycache__").exists()  # caches excluded
    # persona copied
    assert (dest / "persona" / "manifest.toml").is_file()
    assert (dest / "persona" / "tools.py").is_file()
    assert (dest / "persona" / "output_schema.json").is_file()
    # wrapper files
    assert (dest / "agent.py").is_file()
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "README.md").is_file()
    assert (dest / "Dockerfile").is_file()
    assert (dest / ".env.example").is_file()
    assert (dest / ".gitignore").is_file()
    assert (dest / "tests" / "test_smoke.py").is_file()


def test_export_uses_agent_name_override(tmp_path):
    dest = export_agent(
        Persona.load(FIXTURE), tmp_path / "agent", agent_name="Solar Scout", git_init=False
    )
    assert 'name = "solar-scout"' in (dest / "pyproject.toml").read_text()


def test_export_git_init_creates_repo(tmp_path):
    dest = export_agent(Persona.load(FIXTURE), tmp_path / "agent", git_init=True)
    assert (dest / ".git").is_dir()


def test_export_refuses_nonempty_destination(tmp_path):
    persona = Persona.load(FIXTURE)
    dest = tmp_path / "agent"
    export_agent(persona, dest, git_init=False)
    with pytest.raises(FileExistsError):
        export_agent(persona, dest, git_init=False)


def test_export_force_replaces_destination(tmp_path):
    persona = Persona.load(FIXTURE)
    dest = tmp_path / "agent"
    export_agent(persona, dest, git_init=False)
    (dest / "stale_file.txt").write_text("left over from a previous export")
    out = export_agent(persona, dest, git_init=False, force=True)
    assert out == dest
    assert not (dest / "stale_file.txt").exists()   # old tree wiped, not overlaid
    assert (dest / "agent.py").exists()


def test_write_runtime_and_wrappers_stamps_janus_and_wrappers(tmp_path):
    from janus.factory.export import write_runtime_and_wrappers

    persona = Persona.load(FIXTURE)
    dest = tmp_path / "agent"
    dest.mkdir()
    write_runtime_and_wrappers(persona, dest, "echo_brief")

    assert (dest / "janus" / "__init__.py").exists()      # package vendored
    for wrapper in ("agent.py", "pyproject.toml", "README.md", "Dockerfile",
                    ".env.example", ".gitignore", "tests/test_smoke.py"):
        assert (dest / wrapper).exists(), wrapper
    assert not (dest / "persona").exists()                 # persona NOT its job


def test_write_runtime_and_wrappers_propagates_deletions(tmp_path):
    """A file that no longer exists in source janus/ must not survive a re-stamp
    (a plain copytree would leave the stale file behind)."""
    from janus.factory.export import write_runtime_and_wrappers

    persona = Persona.load(FIXTURE)
    dest = tmp_path / "agent"
    dest.mkdir()
    write_runtime_and_wrappers(persona, dest, "echo_brief")

    stale = dest / "janus" / "_removed_in_main.py"
    stale.write_text("# a module that no longer exists upstream\n")
    write_runtime_and_wrappers(persona, dest, "echo_brief")   # re-stamp
    assert not stale.exists()


def _minimal_container_persona(d):
    """Create a minimal containerized persona directory with manifest, prompt, and container.toml."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.toml").write_text(
        '[persona]\nname = "toolagent"\n\n[prompt]\nfile = "prompt.md"\n\n[task]\ntemplate = "Do: {subject}"\n\n[tools]\nbuiltins = ["bash"]\n'
    )
    (d / "prompt.md").write_text("A minimal tool agent.\n")
    (d / "container.toml").write_text(
        '[install]\napt = ["ripgrep"]\n\n[[tool]]\nname = "rg"\ndescription = "search with ripgrep"\n'
    )
    return d


def test_containerized_persona_exports_ubuntu_dockerfile_and_compose(tmp_path):
    from janus.factory.export import write_runtime_and_wrappers

    d = tmp_path / "toolagent"
    (d / "persona").mkdir(parents=True)
    persona_dir = _minimal_container_persona(d / "persona")
    persona = Persona.load(persona_dir)
    dest = tmp_path / "out"
    dest.mkdir()
    write_runtime_and_wrappers(persona, dest, "toolagent")
    assert "FROM ubuntu:24.04" in (dest / "Dockerfile").read_text()
    assert (dest / "docker-compose.yml").exists()
