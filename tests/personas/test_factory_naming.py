import asyncio
from pathlib import Path

from janus.core.persona import Persona
from janus.core.tools.registry import ToolContext
from janus.fleet.registry import FleetRegistry

_FACTORY = Path("personas/factory")


def _factory_registry():
    return Persona.load(_FACTORY).registry


def _dispatch(name, args, fleet_dir, tmp):
    reg = _factory_registry()
    # Construct ToolContext with cwd + extra per its dataclass (extra carries fleet_dir).
    ctx = ToolContext(cwd=tmp, extra={"fleet_dir": str(fleet_dir)})
    return asyncio.run(reg.dispatch(name, args, ctx))


def test_list_fleet_agents_is_registered_on_the_factory():
    assert _factory_registry().get("list_fleet_agents") is not None


def test_list_fleet_agents_lists_assigned_names(tmp_path):
    fleet = tmp_path / "fleet"; fleet.mkdir()
    reg = FleetRegistry(fleet)
    reg.register("themis", domain="law", description="d", source="factory",
                 path=str(fleet / "themis"), clock=lambda: "2026-07-25T00:00:00")
    reg.register("hermes", domain="commerce", description="d", source="factory",
                 path=str(fleet / "hermes"), clock=lambda: "2026-07-25T00:00:00")
    out = _dispatch("list_fleet_agents", {}, fleet, tmp_path)
    assert "themis" in out and "hermes" in out
    assert "reuse" in out.lower()  # instructs the model not to reuse


def test_list_fleet_agents_empty_fleet_says_free(tmp_path):
    fleet = tmp_path / "fleet"; fleet.mkdir()
    out = _dispatch("list_fleet_agents", {}, fleet, tmp_path)
    assert "no agents" in out.lower() or "free" in out.lower()


def test_prompt_mandates_mythological_naming_and_dedup():
    text = (_FACTORY / "prompt.md").read_text().lower()
    assert "mytholog" in text                        # naming theme present
    assert "list_fleet_agents" in text               # dedup awareness wired
    assert "conceptual" in text or "concept" in text  # concept-matched
