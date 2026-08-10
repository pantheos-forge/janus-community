# Janus — an engine for building specialized AI agents.
# Copyright (C) 2026 Pantheos Forge
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. This program is distributed WITHOUT ANY WARRANTY;
# see the GNU AGPL <https://www.gnu.org/licenses/> for details.
#
# A persona exception applies — see LICENSE-EXCEPTION.

"""Pure renderers for the wrapper files a generated Janus agent repo needs.

Every function here returns a plain ``str`` — no filesystem access. Task 3's
``export_agent`` is responsible for writing these strings to disk inside a
generated agent's directory tree.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from janus.core.container import ContainerSpec
    from janus.core.persona import Persona


def _sanitize_project_name(agent_name: str) -> str:
    """Lowercase ``agent_name`` and collapse any run of non-``[a-z0-9]`` chars to ``-``.

    Leading/trailing ``-`` are stripped so the result is a clean PEP 508-ish
    project name (e.g. ``"Market Research"`` -> ``"market-research"``).
    """
    lowered = agent_name.lower()
    collapsed = re.sub(r"[^a-z0-9]+", "-", lowered)
    return collapsed.strip("-") or "janus-agent"


def render_pyproject(agent_name: str) -> str:
    """Render the generated agent's ``pyproject.toml``."""
    project_name = _sanitize_project_name(agent_name)
    return f'''[project]
name = "{project_name}"
version = "0.1.0"
description = "Generated Janus agent: {agent_name}"
requires-python = ">=3.12,<4.0"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27",
    "jsonschema>=4.0",
    "textual>=0.60",
]

[project.optional-dependencies]
claude-sdk = [
    "claude-agent-sdk>=0.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["janus"]
'''


def render_agent_runner() -> str:
    """Render the generated agent's ``agent.py`` entry point, verbatim."""
    return '''"""Entry point for this generated Janus agent. Run: python agent.py "<subject>"."""

from __future__ import annotations

import sys
from pathlib import Path

from janus.core.backends.select import build_backend_for_persona
from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.persona import Persona
from janus.core.session import SessionStore
from janus.interface import launch

ROOT = Path(__file__).parent
PERSONA_DIR = ROOT / "persona"


def main(subject: str) -> int:
    persona = Persona.load(PERSONA_DIR)
    workdir = ROOT / "runs" / persona.name
    persona.prepare_workspace(workdir)
    config = load_config(persona=persona.name, working_directory=workdir)
    try:
        backend = build_backend_for_persona(config, persona)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Configure a provider (see .env.example): set ANTHROPIC_API_KEY, or "
              "openrouter_model / local_model / ds4_url.", file=sys.stderr)
        return 1
    sessions = SessionStore(sessions_dir=ROOT / ".janus" / "sessions")
    controller = AgentController(config, backend=backend, session_store=sessions)
    launch(controller, persona.build_task(subject), title=persona.name, banner=persona.banner)
    deliverable = workdir / "output.json"
    if deliverable.exists():
        print(f"deliverable: {deliverable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(" ".join(sys.argv[1:]) or "(no subject provided)"))
'''


def render_readme(persona: Persona, agent_name: str) -> str:
    """Render the generated agent's ``README.md``.

    The *Running* section adapts to the agent's shape: a containerized agent
    (one with a ``container.toml``) ships a ``docker-compose.yml`` and is run
    with ``docker compose``; a plain agent has no compose file and runs from a
    local virtualenv or a plain ``docker run``. Either way the Textual TUI
    appears on a TTY and falls back to a headless log stream otherwise.
    """
    image = "janus-agent-" + _sanitize_project_name(agent_name)
    tui_note = (
        "When the output is piped or there is no TTY, it falls back to a headless "
        f"log stream. On success the deliverable is written to `runs/{persona.name}/output.json`."
    )

    if persona.container is not None:
        running = f'''## Running

This is a **containerized** agent: it ships a Docker image with its
command-line tools baked in, orchestrated by `docker-compose.yml`.

```bash
docker compose build                       # one time (and after tool changes)
docker compose run --rm agent "<subject>"
```

`docker compose run` allocates a TTY, so the TUI launches by default. Provider
credentials come from your **exported** shell environment (compose forwards
only the variables it finds set — see Configuration). {tui_note}'''
    else:
        running = f'''## Running

**Local (virtualenv):**

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python agent.py "<subject>"
```

**Plain Docker:**

```bash
docker build -t {image} .
docker run --rm -it -v "$PWD/runs:/agent/runs" \\
  -e OPENROUTER_API_KEY -e OPENROUTER_MODEL {image} "<subject>"
```

`docker run -it` allocates a TTY, so the TUI launches by default. {tui_note}

> This is a plain agent — it has no `docker-compose.yml`. `docker compose` is
> only for containerized agents (those with a `container.toml`).'''

    return f'''# {agent_name}

{persona.description}

- **Domain:** {persona.domain}

{running}

## Configuration

Copy `.env.example` to `.env` and fill in the provider credentials you need
(see that file for details). For a container run, **export** those variables
into your shell instead — the container receives only the provider variables
you have set.
'''


def render_dockerfile(container: ContainerSpec | None = None) -> str:
    """Render the generated agent's ``Dockerfile``.

    ``None`` → the slim Python runtime (non-container agents). A ``ContainerSpec``
    → an Ubuntu image with the declared apt/go/pip tool layers baked in.
    """
    if container is None:
        return '''FROM python:3.13-slim

WORKDIR /agent

COPY . .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "agent.py"]
'''

    lines = [
        "FROM ubuntu:24.04",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y \\",
        "    python3 python3-pip python3-venv python3-dev build-essential "
        "ca-certificates git \\",
    ]
    if container.apt:
        lines.append("    " + " ".join(container.apt) + " \\")
    lines.append("    && rm -rf /var/lib/apt/lists/*")
    if container.go:
        lines.append("RUN apt-get update && apt-get install -y golang-go "
                     "&& rm -rf /var/lib/apt/lists/* \\")
        installs = " && ".join(f"GOBIN=/usr/local/bin go install {m}" for m in container.go)
        lines.append("    && " + installs + " \\")
        # Keep only the compiled binaries in /usr/local/bin. The Go toolchain
        # (~450MB) + build/module caches (often 1GB+) are build-time only, and
        # because layers are additive they must be purged in THIS same RUN — a
        # later RUN would leave them in an earlier layer.
        lines.append("    && apt-get purge -y golang-go && apt-get autoremove -y \\")
        lines.append("    && rm -rf /root/go /root/.cache/go-build /var/lib/apt/lists/*")
    if container.pip:
        lines.append("RUN pip install --no-cache-dir --break-system-packages "
                     + " ".join(container.pip))
    if container.dockerfile_append.strip():
        lines.append(container.dockerfile_append.strip())
    lines += [
        "WORKDIR /agent",
        "COPY . .",
        "RUN pip install --no-cache-dir --break-system-packages -e .",
        'ENTRYPOINT ["python3", "agent.py"]',
    ]
    return "\n".join(lines) + "\n"


def render_compose(agent_name: str) -> str:
    """Render a ``docker-compose.yml`` for a containerized agent.

    ``docker compose run --rm agent "<subject>"`` builds the tool image, mounts
    the workspace, and passes provider creds from the environment.
    """
    # Bare `- VARNAME` (no `=`) forwards each var ONLY when the host env has it
    # set. The `${VAR:-}` form instead sends an empty string for unset vars, and
    # an empty USE_CLAUDE_AGENT_SDK crashes pydantic's bool parser at load_config
    # inside the container (a real capstone failure). Passing only set vars keeps
    # provider precedence honest, too.
    return f'''services:
  agent:
    build: .
    image: janus-agent-{agent_name}:latest
    volumes:
      - ./runs:/agent/runs
    environment:
      - ANTHROPIC_API_KEY
      - OPENROUTER_API_KEY
      - OPENROUTER_MODEL
      - DS4_URL
      - DS4_MODEL
      - LOCAL_MODEL
      - USE_CLAUDE_AGENT_SDK
      - JANUS_AUTH_MODE
'''


def render_env_example() -> str:
    """Render the generated agent's ``.env.example``."""
    return '''# Provider credentials — set whichever provider(s) this agent uses.

# Anthropic API (used by the "anthropic_api" backend and the Claude Agent SDK)
# ANTHROPIC_API_KEY=

# OpenRouter (used by the "openrouter" backend)
# OPENROUTER_API_KEY=

# For a local model (e.g. Ollama) or a DS4 server, no API key is needed here —
# instead set `local_model` / `ds4_url` in this agent's config (see janus.core.config)
# or the corresponding environment variables it reads.
'''


def render_gitignore() -> str:
    """Render the generated agent's ``.gitignore``."""
    return '''__pycache__/
*.py[cod]
.venv/
.janus/
runs/
*.egg-info/
.env
'''


def render_smoke_test() -> str:
    """Render the generated agent's ``tests/test_smoke.py``.

    Asserts the vendored persona composes without needing a provider: it loads,
    its tool registry is non-empty, and (if declared) its output schema is a dict.
    """
    return '''"""Smoke test: the vendored persona composes without needing a live provider."""

from pathlib import Path

from janus.core.persona import Persona

ROOT = Path(__file__).parent.parent
PERSONA_DIR = ROOT / "persona"


def test_persona_loads_and_composes():
    persona = Persona.load(PERSONA_DIR)
    assert persona.registry.names()
    if persona.output_schema is not None:
        assert isinstance(persona.output_schema, dict)
'''
