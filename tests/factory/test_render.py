from pathlib import Path

from janus.core.persona import Persona
from janus.factory.render import (
    render_agent_runner,
    render_compose,
    render_dockerfile,
    render_env_example,
    render_gitignore,
    render_pyproject,
    render_readme,
    render_smoke_test,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "personas" / "echo_brief"


def test_pyproject_has_name_and_deps():
    out = render_pyproject("Market Research")
    assert 'name = "market-research"' in out          # sanitized
    assert "jsonschema" in out and "httpx" in out and "pydantic" in out
    assert 'packages = ["janus"]' in out
    assert "claude-agent-sdk" in out                  # optional extra


def test_pyproject_sanitizes_various_names():
    assert 'name = "market-research"' in render_pyproject("Market Research")
    assert 'name = "solar-scout"' in render_pyproject("Solar Scout")


def test_pyproject_falls_back_to_default_name_when_all_non_alnum():
    out = render_pyproject("!!!")
    assert 'name = "janus-agent"' in out


def test_agent_runner_loads_persona_and_isolates_sessions():
    out = render_agent_runner()
    assert "Persona.load" in out
    assert "build_backend_for_persona" in out
    assert '.janus" / "sessions"' in out or '.janus/sessions' in out  # repo-local session isolation
    assert "Configure a provider" in out  # friendly error when no provider is configured


def test_readme_describes_the_agent():
    out = render_readme(Persona.load(FIXTURE), "echo-brief")
    assert "echo-brief" in out
    assert "echoes" in out.lower() or "brief" in out.lower()   # from persona.description
    assert "python agent.py" in out


def test_readme_plain_agent_documents_local_run_and_tui():
    """A plain (non-containerized) agent has no docker-compose.yml, so its
    README must show the venv / plain-`docker run` paths, not `docker compose`,
    plus the TUI-on-a-TTY / headless-otherwise note."""
    out = render_readme(Persona.load(FIXTURE), "echo-brief")
    # No compose RUN path for a plain agent (it may still be named in a
    # "compose is only for containerized agents" aside — that is intended).
    assert "docker compose run" not in out
    assert "docker compose build" not in out
    assert "python -m venv" in out              # local virtualenv path
    assert "docker run" in out                  # plain docker path
    assert "TUI" in out and "headless" in out   # run-mode note


def test_readme_containerized_documents_compose_and_tui():
    """A containerized agent ships docker-compose.yml, so its README leads with
    the compose build/run path (which allocates a TTY → TUI)."""
    import dataclasses

    from janus.core.container import ContainerSpec, ToolEntry
    persona = dataclasses.replace(
        Persona.load(FIXTURE),
        container=ContainerSpec(apt=["ripgrep"], tools=[ToolEntry("rg", "search")]),
    )
    out = render_readme(persona, "tool-agent")
    assert "docker compose build" in out
    assert "docker compose run" in out
    assert "TUI" in out and "headless" in out


def test_gitignore_excludes_local_state():
    out = render_gitignore()
    assert ".janus" in out and "runs/" in out and ".env" in out


def test_dockerfile_is_base_python():
    out = render_dockerfile()
    assert "python:3.13-slim" in out and "agent.py" in out


def test_env_example_lists_providers():
    out = render_env_example()
    assert "ANTHROPIC_API_KEY" in out and "OPENROUTER_API_KEY" in out


def test_smoke_test_loads_persona():
    out = render_smoke_test()
    assert "Persona.load" in out


def test_runner_uses_interface_launch():
    src = render_agent_runner()
    assert "from janus.interface import launch" in src
    assert "launch(" in src
    import ast

    ast.parse(src)  # still valid Python


def test_pyproject_installs_textual_as_base_dep():
    """TUI-by-default: textual is a base dependency, so every install path
    (Docker `pip install -e .`, local install, README) gets the TUI. launch()
    still degrades to the headless renderer when stdout is not a TTY."""
    src = render_pyproject("demo-agent")
    base, sep, _extras = src.partition("[project.optional-dependencies]")
    assert sep, "pyproject should still declare an optional-dependencies section"
    assert "textual" in base, "textual must be a base dependency, not an extra"


def test_render_dockerfile_none_is_the_slim_default():
    df = render_dockerfile()
    assert "python:3.13-slim" in df
    assert "ubuntu" not in df.lower()


def test_render_dockerfile_containerized_uses_ubuntu_and_tool_layers():
    from janus.core.container import ContainerSpec, ToolEntry
    spec = ContainerSpec(apt=["ripgrep"], pip=["radon"],
                         go=["github.com/boyter/scc/v3@latest"],
                         dockerfile_append="RUN echo hi",
                         tools=[ToolEntry("rg", "search")])
    df = render_dockerfile(spec)
    assert "FROM ubuntu:24.04" in df
    assert "ripgrep" in df
    assert "go install" in df and "boyter/scc" in df
    assert "radon" in df
    assert "RUN echo hi" in df
    assert "agent.py" in df


def test_render_compose_passes_bare_env_names_not_empty_defaults():
    """Live-capstone bug: `- USE_CLAUDE_AGENT_SDK=${USE_CLAUDE_AGENT_SDK:-}` sends
    an empty string when the var is unset, and pydantic's bool parser rejects ""
    with a ValidationError that crashes the container at load_config().

    Bare `- VARNAME` tells compose to forward the var ONLY when the host has it
    set, so an unset boolean is omitted rather than sent as "".
    """
    out = render_compose("toolbox")
    assert "image: janus-agent-toolbox:latest" in out
    # No `${VAR:-}` empty-default expansion anywhere — that is exactly what broke.
    assert ":-}" not in out
    assert "=${" not in out
    # Every provider var is still named, in bare pass-through form.
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
                "DS4_URL", "DS4_MODEL", "LOCAL_MODEL", "USE_CLAUDE_AGENT_SDK",
                "JANUS_AUTH_MODE"):
        assert f"      - {var}\n" in out


def test_render_dockerfile_go_stanza_purges_toolchain_in_same_layer():
    """The Go toolchain (~450MB) + build/module caches (often 1GB+) are needed
    only to `go install` the tools; the image should keep just the compiled
    binaries. Layers are additive, so the purge MUST live in the same RUN as the
    install — a later RUN can't reclaim an earlier layer."""
    from janus.core.container import ContainerSpec
    df = render_dockerfile(ContainerSpec(go=["github.com/anchore/syft/cmd/syft@latest"]))
    # locate the single RUN block that installs the toolchain
    blocks = ("\n" + df).split("\nRUN ")
    go_block = next(b for b in blocks if "golang-go" in b)
    assert "go install" in go_block                      # still installs the tool
    assert "apt-get purge -y golang-go" in go_block       # toolchain removed…
    assert "go-build" in go_block                         # …and the build cache…
    assert "/root/go" in go_block                         # …and the module cache — all in ONE layer


def test_render_dockerfile_apt_only_omits_go_and_pip_stanzas():
    from janus.core.container import ContainerSpec
    df = render_dockerfile(ContainerSpec(apt=["ripgrep"]))
    assert "go install" not in df
    assert "pip install --no-cache-dir --break-system-packages r" not in df  # no pip pkgs line


def test_render_dockerfile_empty_apt_with_pip_is_valid():
    from janus.core.container import ContainerSpec
    df = render_dockerfile(ContainerSpec(pip=["radon"]))
    # Base packages line must end with backslash so && rm -rf is a continuation, not standalone
    assert "ca-certificates git \\\n" in df
    # Verify radon is included
    assert "radon" in df
    # Verify there's no broken RUN line (every non-final RUN line in the apt block ends with \)
    lines = df.split("\n")
    # Find the apt RUN block and verify continuations are valid
    apt_run_idx = None
    for i, line in enumerate(lines):
        if line.startswith("RUN apt-get update"):
            apt_run_idx = i
            break
    assert apt_run_idx is not None
    # Verify the block continues properly until rm -rf
    for i in range(apt_run_idx, len(lines)):
        if "rm -rf /var/lib/apt/lists" in lines[i]:
            # Found the end of the apt block
            break
        # Every line before the final rm -rf should end with backslash
        if i > apt_run_idx and not lines[i].startswith("    &&"):
            msg = f"Line {i} should end with backslash: {lines[i]}"
            assert lines[i].rstrip().endswith("\\"), msg
