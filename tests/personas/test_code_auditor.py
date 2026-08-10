import os
from pathlib import Path

import pytest

from janus.core.persona import Persona
from janus.core.validation.container_smoke import docker_available

PERSONA_DIR = Path(__file__).resolve().parents[2] / "personas" / "code_auditor"


def test_loads_as_a_containerized_agent_with_bash():
    p = Persona.load(PERSONA_DIR)
    assert p.container is not None
    assert "bash" in p.registry.names()
    assert p.container.apt == ["ripgrep", "git"]


def test_tool_inventory_is_in_the_prompt():
    p = Persona.load(PERSONA_DIR)
    assert "Tools available in your environment" in p.system_prompt
    assert "`scc`" in p.system_prompt and "`rg`" in p.system_prompt


def test_exports_an_ubuntu_tool_image(tmp_path):
    from janus.factory.export import export_agent

    dest = export_agent(Persona.load(PERSONA_DIR), tmp_path / "agent", git_init=False)
    assert "FROM ubuntu:24.04" in (dest / "Dockerfile").read_text()
    assert (dest / "docker-compose.yml").exists()
    assert "ripgrep" in (dest / "Dockerfile").read_text()


@pytest.mark.skipif(
    not docker_available()
    or not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_MODEL")),
    reason="needs Docker + a provider",
)
@pytest.mark.asyncio
async def test_code_auditor_validates_in_container():
    from janus.core.validation.container_smoke import container_smoke_run

    p = Persona.load(PERSONA_DIR)
    # small pinned public repo as the subject
    res = await container_smoke_run(p, "https://github.com/pallets/click", Path("/tmp/ca-val"))
    assert res.passed, [c.detail for c in res.checks]
