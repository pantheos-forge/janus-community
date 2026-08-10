import json

import pytest

from janus.core.tools.registry import ToolContext
from tests.personas.factory_samples import (
    FAILING_VERDICT,
    GOOD_DELIVERABLE,
    GOOD_MANIFEST,
    GOOD_NAME,
    GOOD_PROMPT,
    GOOD_RUBRIC,
    GOOD_SCHEMA,
    PASSING_VERDICT,
    get_tool,
    make_fake_factories,
    scaffold_good,
)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path)


_CONTAINER_TOML = '''[install]
apt = ["ripgrep"]

[[tool]]
name = "rg"
description = "ripgrep search"
usage = "rg -n pat ."
'''

# A manifest that declares bash (containerized personas require it).
_CONTAINER_MANIFEST = '''[persona]
name = "toolmaker"
description = "uses a cli tool"
domain = "demo"
[prompt]
file = "prompt.md"
[tools]
builtins = ["bash", "read_file"]
[task]
template = "Do: {subject}"
[output]
schema_file = "output_schema.json"
[validation]
rubric_file = "rubric.toml"
'''


def test_scaffold_writes_and_validates_container_toml(factory_tools, ctx, tmp_path):
    result = scaffold_good(
        factory_tools, ctx,
        name="toolmaker",
        manifest_toml=_CONTAINER_MANIFEST,
        prompt_md="You use rg.",
        container_toml=_CONTAINER_TOML,
    )
    assert not result.startswith("Scaffold rejected"), result
    cpath = tmp_path / "build" / "toolmaker" / "container.toml"
    assert cpath.exists()
    from janus.core.persona import Persona
    persona = Persona.load(tmp_path / "build" / "toolmaker")
    assert persona.container is not None
    assert persona.container.apt == ["ripgrep"]


def test_scaffold_container_toml_without_bash_is_rejected(factory_tools, ctx, tmp_path):
    manifest_no_bash = _CONTAINER_MANIFEST.replace('["bash", "read_file"]', '["read_file"]')
    result = scaffold_good(
        factory_tools, ctx,
        name="toolmaker", manifest_toml=manifest_no_bash,
        prompt_md="x", container_toml=_CONTAINER_TOML,
    )
    assert result.startswith("Scaffold rejected")
    assert "bash" in result
    assert not (tmp_path / "build" / "toolmaker" / "container.toml").exists()


def test_scaffold_without_container_toml_writes_none(factory_tools, ctx, tmp_path):
    result = scaffold_good(factory_tools, ctx)   # no container_toml
    assert not result.startswith("Scaffold rejected"), result
    assert not (tmp_path / "build" / GOOD_NAME / "container.toml").exists()


def test_rescaffold_drops_container_toml_with_warning(factory_tools, ctx, tmp_path):
    """Scaffold WITH container.toml, then re-scaffold WITHOUT it; verify the warning."""
    # First scaffold: with container.toml
    result1 = scaffold_good(
        factory_tools, ctx,
        name="toolmaker",
        manifest_toml=_CONTAINER_MANIFEST,
        prompt_md="You use rg.",
        container_toml=_CONTAINER_TOML,
    )
    assert not result1.startswith("Scaffold rejected"), result1
    cpath = tmp_path / "build" / "toolmaker" / "container.toml"
    assert cpath.exists(), "container.toml should exist after first scaffold"

    # Re-scaffold: same manifest (still declares bash) but NO container_toml
    result2 = scaffold_good(
        factory_tools, ctx,
        name="toolmaker",
        manifest_toml=_CONTAINER_MANIFEST,
        prompt_md="You use rg.",
        # no container_toml this time
    )
    assert not result2.startswith("Scaffold rejected"), result2
    # Verify container.toml is gone
    assert not cpath.exists(), "container.toml should be removed after re-scaffold without it"
    # Verify warning is in the message
    assert "removed the previous container.toml" in result2
    assert "builtin-only" in result2


def test_scaffold_accepts_a_good_quartet(factory_tools, ctx, tmp_path):
    result = scaffold_good(factory_tools, ctx)
    assert result.startswith("Scaffolded persona")
    build = tmp_path / "build" / GOOD_NAME
    for fname in ("manifest.toml", "prompt.md", "output_schema.json", "rubric.toml"):
        assert (build / fname).exists()


def test_scaffold_rejects_a_bad_name(factory_tools, ctx):
    result = scaffold_good(factory_tools, ctx, name="Haiku Scout!")
    assert result.startswith("Error:")


def test_scaffold_rejects_unparseable_manifest(factory_tools, ctx, tmp_path):
    result = scaffold_good(factory_tools, ctx, manifest_toml="this is [not toml")
    assert "Scaffold rejected" in result and "manifest.toml" in result
    assert not (tmp_path / "build" / GOOD_NAME).exists()   # nothing half-written


def _renamed_manifest():
    from tests.personas.factory_samples import GOOD_MANIFEST
    return GOOD_MANIFEST.replace('name = "haiku_scout"', 'name = "other_name"')


def test_scaffold_rejects_name_mismatch(factory_tools, ctx):
    bad = scaffold_good(factory_tools, ctx, manifest_toml=_renamed_manifest())
    assert "Scaffold rejected" in bad and "name" in bad


def test_scaffold_rejects_template_without_subject(factory_tools, ctx):
    from tests.personas.factory_samples import GOOD_MANIFEST
    bad_manifest = GOOD_MANIFEST.replace(
        'template = "Write a haiku about: {subject}"',
        'template = "Write a haiku."',
    )
    result = scaffold_good(factory_tools, ctx, manifest_toml=bad_manifest)
    assert "Scaffold rejected" in result and "{subject}" in result


def test_scaffold_rejects_custom_tools(factory_tools, ctx):
    from tests.personas.factory_samples import GOOD_MANIFEST
    bad_manifest = GOOD_MANIFEST.replace(
        "[tools]\nbuiltins = []",
        '[tools]\nbuiltins = []\ncustom = "tools:TOOLS"',
    )
    result = scaffold_good(factory_tools, ctx, manifest_toml=bad_manifest)
    assert "Scaffold rejected" in result and "declarative" in result


def test_scaffold_rejects_unknown_builtin(factory_tools, ctx):
    from tests.personas.factory_samples import GOOD_MANIFEST
    bad_manifest = GOOD_MANIFEST.replace("builtins = []", 'builtins = ["web_search"]')
    result = scaffold_good(factory_tools, ctx, manifest_toml=bad_manifest)
    assert "Scaffold rejected" in result and "web_search" in result


def test_scaffold_rejects_invalid_json_schema(factory_tools, ctx):
    result = scaffold_good(
        factory_tools, ctx, output_schema_json='{"type": "objekt"}'
    )
    assert "Scaffold rejected" in result and "output_schema.json" in result


def test_scaffold_rejects_unparseable_schema_json(factory_tools, ctx):
    result = scaffold_good(factory_tools, ctx, output_schema_json="{not json")
    assert "Scaffold rejected" in result and "output_schema.json" in result


def test_scaffold_rejects_non_object_schema_json(factory_tools, ctx):
    result = scaffold_good(factory_tools, ctx, output_schema_json="5")
    assert "Scaffold rejected" in result and "output_schema.json" in result


def test_scaffold_rejects_non_string_template(factory_tools, ctx):
    from tests.personas.factory_samples import GOOD_MANIFEST
    bad_manifest = GOOD_MANIFEST.replace(
        'template = "Write a haiku about: {subject}"',
        "template = 5",
    )
    result = scaffold_good(factory_tools, ctx, manifest_toml=bad_manifest)
    assert "Scaffold rejected" in result and "template" in result


def test_scaffold_rejects_non_list_builtins(factory_tools, ctx):
    from tests.personas.factory_samples import GOOD_MANIFEST
    bad_manifest = GOOD_MANIFEST.replace('builtins = []', 'builtins = "web_fetch"')
    result = scaffold_good(factory_tools, ctx, manifest_toml=bad_manifest)
    assert "Scaffold rejected" in result and "builtins" in result


def test_scaffold_rejects_a_rubric_without_criteria(factory_tools, ctx, tmp_path):
    result = scaffold_good(
        factory_tools, ctx, rubric_toml='tasks = ["autumn rain"]\npass_threshold = 0.7\n'
    )
    assert "Scaffold rejected" in result and "rubric" in result.lower()
    assert not (tmp_path / "build" / GOOD_NAME).exists()


def test_rescaffold_overwrites(factory_tools, ctx, tmp_path):
    scaffold_good(factory_tools, ctx)
    from tests.personas.factory_samples import GOOD_PROMPT
    result = scaffold_good(factory_tools, ctx, prompt_md=GOOD_PROMPT + "Cite your imagery.\n")
    assert result.startswith("Scaffolded persona")
    assert "Cite your imagery." in (tmp_path / "build" / GOOD_NAME / "prompt.md").read_text()


def _fake_one_attempt(factory_tools, ctx):
    state = factory_tools._state_dir(ctx, GOOD_NAME)
    state.mkdir(parents=True, exist_ok=True)
    (state / "attempts.json").write_text(json.dumps({"attempts": [
        {"passed": False, "smoke_passed": True, "judge_passed": False,
         "scores": {"form": 0.4}, "feedback": "not haiku-like"}]}))


def test_rubric_frozen_after_first_validation(factory_tools, ctx):
    scaffold_good(factory_tools, ctx)
    _fake_one_attempt(factory_tools, ctx)
    loosened = GOOD_RUBRIC.replace("pass_threshold = 0.7", "pass_threshold = 0.1")
    result = scaffold_good(factory_tools, ctx, rubric_toml=loosened)
    assert result.startswith("Error:") and "frozen" in result


def test_rescaffold_with_identical_rubric_is_allowed_after_validation(factory_tools, ctx):
    scaffold_good(factory_tools, ctx)
    _fake_one_attempt(factory_tools, ctx)
    result = scaffold_good(factory_tools, ctx)   # same rubric text -> fine
    assert result.startswith("Scaffolded persona")


def _ctx_with_fakes(tmp_path, deliverable, verdict):
    c = ToolContext(cwd=tmp_path)
    c.extra.update(make_fake_factories(deliverable, verdict))
    return c


@pytest.mark.asyncio
async def test_validate_persona_passes_and_records_the_attempt(factory_tools, tmp_path):
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, PASSING_VERDICT)
    scaffold_good(factory_tools, ctx)
    result = await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    data = json.loads(result)
    assert data["passed"] is True
    assert data["attempt"] == 1 and data["attempts_remaining"] == 2
    assert data["smoke"]["passed"] is True
    assert data["judge"]["scores"] == {"form": 0.9}
    attempts = factory_tools._load_attempts(factory_tools._state_dir(ctx, GOOD_NAME))
    assert len(attempts) == 1 and attempts[0]["passed"] is True


@pytest.mark.asyncio
async def test_validate_persona_surfaces_judge_failure(factory_tools, tmp_path):
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, FAILING_VERDICT)
    scaffold_good(factory_tools, ctx)
    result = await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    data = json.loads(result)
    assert data["passed"] is False
    assert data["judge"]["passed"] is False
    assert "prose" in data["judge"]["feedback"]


@pytest.mark.asyncio
async def test_validate_persona_rejects_bad_name(factory_tools, tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    result = await get_tool(factory_tools, "validate_persona").handler(ctx, name="../evil")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_validate_persona_requires_a_scaffold(factory_tools, tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    result = await get_tool(factory_tools, "validate_persona").handler(ctx, name="ghost")
    assert result.startswith("Error:") and "scaffold_persona" in result


@pytest.mark.asyncio
async def test_validate_persona_enforces_the_attempt_budget(factory_tools, tmp_path):
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, FAILING_VERDICT)
    scaffold_good(factory_tools, ctx)
    handler = get_tool(factory_tools, "validate_persona").handler
    for _ in range(3):
        json.loads(await handler(ctx, name=GOOD_NAME))   # three real (failing) attempts
    fourth = await handler(ctx, name=GOOD_NAME)
    assert fourth.startswith("Error:") and "budget exhausted" in fourth
    attempts = factory_tools._load_attempts(factory_tools._state_dir(ctx, GOOD_NAME))
    assert len(attempts) == 3   # the refused call consumed nothing


@pytest.mark.asyncio
async def test_infrastructure_error_consumes_no_attempt(factory_tools, tmp_path):
    ctx = ToolContext(cwd=tmp_path)

    def exploding_factory(persona, wd):
        raise RuntimeError("provider is down")

    ctx.extra["make_agent_backend"] = exploding_factory
    scaffold_good(factory_tools, ctx)
    result = await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    assert result.startswith("Infrastructure error")
    assert factory_tools._load_attempts(factory_tools._state_dir(ctx, GOOD_NAME)) == []


@pytest.mark.asyncio
async def test_validate_persona_defaults_to_production_factories(
    factory_tools, tmp_path, monkeypatch
):
    def marker_factory(persona, wd):
        raise RuntimeError("PRODUCTION-FACTORY-MARKER")

    monkeypatch.setattr(factory_tools, "make_production_agent_backend", marker_factory)
    ctx = ToolContext(cwd=tmp_path)   # no ctx.extra overrides
    scaffold_good(factory_tools, ctx)
    result = await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    assert "PRODUCTION-FACTORY-MARKER" in result


def test_export_persona_rejects_bad_name(factory_tools, tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    result = get_tool(factory_tools, "export_persona").handler(ctx, name="../evil")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_export_refused_without_any_validation(factory_tools, tmp_path):
    ctx = ToolContext(cwd=tmp_path)
    scaffold_good(factory_tools, ctx)
    result = get_tool(factory_tools, "export_persona").handler(ctx, name=GOOD_NAME)
    assert result.startswith("Error: cannot export")


@pytest.mark.asyncio
async def test_export_refused_when_last_validation_failed(factory_tools, tmp_path):
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, FAILING_VERDICT)
    scaffold_good(factory_tools, ctx)
    await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    result = get_tool(factory_tools, "export_persona").handler(ctx, name=GOOD_NAME)
    assert result.startswith("Error: cannot export")


@pytest.mark.asyncio
async def test_export_succeeds_after_a_passing_validation(factory_tools, tmp_path):
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, PASSING_VERDICT)
    scaffold_good(factory_tools, ctx)
    await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    result = get_tool(factory_tools, "export_persona").handler(
        ctx, name=GOOD_NAME, dest=str(tmp_path / "exports" / GOOD_NAME)
    )
    assert result.startswith("Exported")
    dest = tmp_path / "exports" / GOOD_NAME
    assert (dest / "agent.py").exists()
    assert (dest / "janus" / "__init__.py").exists()
    assert (dest / "persona" / "manifest.toml").exists()
    # validation state must NOT leak into the exported repo
    assert not (dest / "persona" / "attempts.json").exists()
    assert not (dest / "persona" / "validation").exists()


@pytest.mark.asyncio
async def test_export_honors_a_relative_dest(factory_tools, tmp_path):
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, PASSING_VERDICT)
    scaffold_good(factory_tools, ctx)
    await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    result = get_tool(factory_tools, "export_persona").handler(
        ctx, name=GOOD_NAME, dest="out/my_agent"
    )
    assert result.startswith("Exported")
    assert (tmp_path / "out" / "my_agent" / "agent.py").exists()


def test_rubric_rejection_includes_the_format_contract(factory_tools, ctx):
    """Live-capstone finding: a bare 'must declare at least one task' error left
    the factory guessing TOML shapes 15+ times. The rejection must teach the
    format: top-level tasks array + [[criteria]] blocks."""
    result = scaffold_good(
        factory_tools, ctx,
        rubric_toml='[rubric]\ntasks = ["misplaced under a table"]\npass_threshold = 0.7\n',
    )
    assert "Scaffold rejected" in result
    assert "tasks = [" in result          # shows the top-level array shape
    assert "[[criteria]]" in result       # shows the criteria block shape


@pytest.mark.asyncio
async def test_export_registers_into_the_fleet_by_default(factory_tools, tmp_path):
    from janus.fleet.registry import FleetRegistry

    fleet = tmp_path / "fleet"
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, PASSING_VERDICT)
    ctx.extra["fleet_dir"] = str(fleet)
    scaffold_good(factory_tools, ctx)
    await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    result = get_tool(factory_tools, "export_persona").handler(ctx, name=GOOD_NAME)
    assert result.startswith("Exported")

    # landed in the fleet home and got registered with its scores
    assert (fleet / GOOD_NAME / "agent.py").exists()
    reg = FleetRegistry(fleet)
    agent = reg.get(GOOD_NAME)
    assert agent is not None and agent["source"] == "factory"
    assert agent["validation_history"] and agent["validation_history"][-1]["passed"] is True


@pytest.mark.asyncio
async def test_explicit_dest_does_not_register(factory_tools, tmp_path):
    from janus.fleet.registry import FleetRegistry

    fleet = tmp_path / "fleet"
    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, PASSING_VERDICT)
    ctx.extra["fleet_dir"] = str(fleet)
    scaffold_good(factory_tools, ctx)
    await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    out = tmp_path / "somewhere_else"
    result = get_tool(factory_tools, "export_persona").handler(
        ctx, name=GOOD_NAME, dest=str(out))
    assert result.startswith("Exported")
    assert (out / "agent.py").exists()
    assert FleetRegistry(fleet).get(GOOD_NAME) is None  # not registered


@pytest.mark.asyncio
async def test_export_persona_refuses_to_overwrite_a_fleet_agent(factory_tools, tmp_path):
    """CRITICAL: export_persona's default dest is fleet_dir/<name> — the same path
    as an existing fleet agent's repo. export_agent(force=True) rmtrees the dest,
    so calling export_persona (instead of export_improved_persona) on an already-
    registered name would destroy that agent's git history. Must be refused."""
    import subprocess

    from janus.fleet.registry import FleetRegistry

    fleet = tmp_path / "fleet"
    agent_dir = fleet / GOOD_NAME
    persona_dir = agent_dir / "persona"
    persona_dir.mkdir(parents=True)
    (persona_dir / "manifest.toml").write_text(GOOD_MANIFEST)
    (persona_dir / "prompt.md").write_text(GOOD_PROMPT)
    (persona_dir / "output_schema.json").write_text(GOOD_SCHEMA)
    (persona_dir / "rubric.toml").write_text(GOOD_RUBRIC)
    # A sentinel + a git history the rmtree-on-overwrite would destroy.
    sentinel = agent_dir / "sentinel.txt"
    sentinel.write_text("do not destroy me")
    subprocess.run(["git", "init", "-q"], cwd=agent_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=agent_dir, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "init"], cwd=agent_dir, check=True)
    FleetRegistry(fleet).register(GOOD_NAME, domain="poetry", description="d",
                                  source="factory", path=str(agent_dir))

    ctx = _ctx_with_fakes(tmp_path, GOOD_DELIVERABLE, PASSING_VERDICT)
    ctx.extra["fleet_dir"] = str(fleet)
    scaffold_good(factory_tools, ctx)
    await get_tool(factory_tools, "validate_persona").handler(ctx, name=GOOD_NAME)
    result = get_tool(factory_tools, "export_persona").handler(ctx, name=GOOD_NAME)

    assert result.startswith("Error:")
    assert "export_improved_persona" in result
    # The agent's existing repo must be completely untouched.
    assert sentinel.exists() and sentinel.read_text() == "do not destroy me"
    log = subprocess.run(["git", "log", "--oneline"], cwd=agent_dir,
                         capture_output=True, text=True).stdout
    assert "init" in log


def test_load_fleet_persona_pulls_files_and_history(factory_tools, tmp_path):
    from janus.fleet.registry import FleetRegistry

    # Arrange: a registered fleet agent with the four persona files + history.
    fleet = tmp_path / "fleet"
    agent_dir = fleet / GOOD_NAME / "persona"
    agent_dir.mkdir(parents=True)
    (agent_dir / "manifest.toml").write_text(GOOD_MANIFEST)
    (agent_dir / "prompt.md").write_text(GOOD_PROMPT)
    (agent_dir / "output_schema.json").write_text(GOOD_SCHEMA)
    (agent_dir / "rubric.toml").write_text(GOOD_RUBRIC)
    reg = FleetRegistry(fleet)
    reg.register(GOOD_NAME, domain="poetry", description="d", source="factory",
                 path=str(fleet / GOOD_NAME))
    reg.append_validation(GOOD_NAME, scores={"form": 0.8}, passed=True, note="factory export")

    # Act
    ctx = ToolContext(cwd=tmp_path / "ws")
    ctx.extra["fleet_dir"] = str(fleet)
    result = get_tool(factory_tools, "load_fleet_persona").handler(ctx, name=GOOD_NAME)

    # Assert: files landed in the factory build dir, fresh state, history returned.
    assert result.startswith("Loaded fleet persona")
    build = tmp_path / "ws" / "build" / GOOD_NAME
    for fn in ("manifest.toml", "prompt.md", "output_schema.json", "rubric.toml"):
        assert (build / fn).exists()
    assert not (tmp_path / "ws" / "build" / ".state" / GOOD_NAME).exists()
    assert '"form": 0.8' in result  # validation history surfaced to the factory


def test_load_fleet_persona_gives_a_fresh_attempt_budget(factory_tools, tmp_path):
    from janus.fleet.registry import FleetRegistry

    fleet = tmp_path / "fleet"
    agent_dir = fleet / GOOD_NAME / "persona"
    agent_dir.mkdir(parents=True)
    (agent_dir / "manifest.toml").write_text(GOOD_MANIFEST)
    (agent_dir / "prompt.md").write_text(GOOD_PROMPT)
    (agent_dir / "output_schema.json").write_text(GOOD_SCHEMA)
    (agent_dir / "rubric.toml").write_text(GOOD_RUBRIC)
    FleetRegistry(fleet).register(GOOD_NAME, domain="poetry", description="d",
                                  source="factory", path=str(fleet / GOOD_NAME))

    ctx = ToolContext(cwd=tmp_path / "ws")
    ctx.extra["fleet_dir"] = str(fleet)
    # Pre-seed stale attempt state that load must clear.
    stale = factory_tools._state_dir(ctx, GOOD_NAME)
    stale.mkdir(parents=True)
    (stale / "attempts.json").write_text('{"attempts": [{"passed": false}]}')

    get_tool(factory_tools, "load_fleet_persona").handler(ctx, name=GOOD_NAME)
    assert factory_tools._load_attempts(factory_tools._state_dir(ctx, GOOD_NAME)) == []


def test_load_fleet_persona_unknown_agent_errors(factory_tools, tmp_path):
    ctx = ToolContext(cwd=tmp_path / "ws")
    ctx.extra["fleet_dir"] = str(tmp_path / "fleet")
    result = get_tool(factory_tools, "load_fleet_persona").handler(ctx, name="ghost")
    assert result.startswith("Error:") and "ghost" in result


def test_load_fleet_persona_malformed_agent_leaves_build_untouched(factory_tools, tmp_path):
    """Stage-then-promote: a fleet agent with all four files present but a
    malformed one must NOT clobber an in-progress build/<name> or wipe its
    attempts before failing."""
    from janus.fleet.registry import FleetRegistry

    fleet = tmp_path / "fleet"
    agent_dir = fleet / GOOD_NAME / "persona"
    agent_dir.mkdir(parents=True)
    (agent_dir / "manifest.toml").write_text("this is [not valid toml")   # corrupt
    (agent_dir / "prompt.md").write_text(GOOD_PROMPT)
    (agent_dir / "output_schema.json").write_text(GOOD_SCHEMA)
    (agent_dir / "rubric.toml").write_text(GOOD_RUBRIC)
    FleetRegistry(fleet).register(GOOD_NAME, domain="poetry", description="d",
                                  source="factory", path=str(fleet / GOOD_NAME))

    ctx = ToolContext(cwd=tmp_path / "ws")
    ctx.extra["fleet_dir"] = str(fleet)

    # Pre-existing in-progress build + attempts that must survive a failed load.
    build_dir = factory_tools._build_dir(ctx, GOOD_NAME)
    build_dir.mkdir(parents=True)
    (build_dir / "sentinel.txt").write_text("in progress")
    state_dir = factory_tools._state_dir(ctx, GOOD_NAME)
    state_dir.mkdir(parents=True)
    (state_dir / "attempts.json").write_text('{"attempts": [{"passed": false}]}')

    result = get_tool(factory_tools, "load_fleet_persona").handler(ctx, name=GOOD_NAME)

    assert result.startswith("Error:")
    assert (build_dir / "sentinel.txt").read_text() == "in progress"  # untouched
    assert factory_tools._load_attempts(state_dir) == [{"passed": False}]  # untouched


@pytest.mark.asyncio
async def test_export_improved_persona_commits_and_records(factory_tools, tmp_path):
    import subprocess

    from janus.fleet.registry import FleetRegistry
    from tests.personas.factory_samples import get_tool

    # Arrange a fleet agent that is a git repo, registered, with the four files
    fleet = tmp_path / "fleet"
    agent = fleet / GOOD_NAME
    persona = agent / "persona"
    persona.mkdir(parents=True)
    (persona / "manifest.toml").write_text(GOOD_MANIFEST)
    (persona / "prompt.md").write_text(GOOD_PROMPT)
    (persona / "output_schema.json").write_text(GOOD_SCHEMA)
    (persona / "rubric.toml").write_text(GOOD_RUBRIC)
    subprocess.run(["git", "init", "-q"], cwd=agent, check=True)
    subprocess.run(["git", "add", "-A"], cwd=agent, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "init"], cwd=agent, check=True)
    reg = FleetRegistry(fleet)
    reg.register(GOOD_NAME, domain="poetry", description="d", source="factory",
                 path=str(agent))

    # Simulate a completed improvement campaign: build dir + a passing attempt.
    ctx = ToolContext(cwd=tmp_path / "ws")
    ctx.extra["fleet_dir"] = str(fleet)
    build = factory_tools._build_dir(ctx, GOOD_NAME)
    build.mkdir(parents=True)
    (build / "manifest.toml").write_text(GOOD_MANIFEST)
    (build / "prompt.md").write_text(GOOD_PROMPT + "\nTightened.\n")
    (build / "output_schema.json").write_text(GOOD_SCHEMA)
    (build / "rubric.toml").write_text(GOOD_RUBRIC)
    state = factory_tools._state_dir(ctx, GOOD_NAME)
    state.mkdir(parents=True)
    (state / "attempts.json").write_text(
        '{"attempts": [{"passed": true, "scores": {"form": 0.95}}]}')

    result = get_tool(factory_tools, "export_improved_persona").handler(
        ctx, name=GOOD_NAME, summary="tighten the form")
    assert result.startswith("Improved")
    # committed into the agent's own repo, history preserved
    log = subprocess.run(["git", "log", "--oneline"], cwd=agent,
                         capture_output=True, text=True).stdout
    assert "improve: tighten the form" in log
    assert "init" in log
    assert "Tightened." in (persona / "prompt.md").read_text()
    # registry got the new scores with an improve note
    hist = reg.get(GOOD_NAME)["validation_history"]
    assert hist and hist[-1]["scores"] == {"form": 0.95}
    assert hist[-1]["note"].startswith("improve:")


@pytest.mark.asyncio
async def test_export_improved_refused_without_passing_validation(factory_tools, tmp_path):
    from tests.personas.factory_samples import get_tool

    ctx = ToolContext(cwd=tmp_path / "ws")
    ctx.extra["fleet_dir"] = str(tmp_path / "fleet")
    build = factory_tools._build_dir(ctx, GOOD_NAME)
    build.mkdir(parents=True)
    (build / "manifest.toml").write_text(GOOD_MANIFEST)
    (build / "prompt.md").write_text(GOOD_PROMPT)
    (build / "output_schema.json").write_text(GOOD_SCHEMA)
    (build / "rubric.toml").write_text(GOOD_RUBRIC)
    # no attempts recorded -> refused
    result = get_tool(factory_tools, "export_improved_persona").handler(
        ctx, name=GOOD_NAME, summary="x")
    assert result.startswith("Error:") and "did not pass" in result


@pytest.mark.asyncio
async def test_check_docker_reports_availability(tmp_path, monkeypatch):
    import janus.core.validation.container_smoke as cs
    from janus.core.persona import Persona
    from janus.core.tools.registry import ToolContext
    from tests.personas.conftest import FACTORY_DIR

    persona = Persona.load(FACTORY_DIR)
    ctx = ToolContext(cwd=str(tmp_path), extra={})

    monkeypatch.setattr(cs, "docker_available", lambda: True)
    ok = await persona.registry.dispatch("check_docker", {}, ctx)
    assert "available" in ok.lower() and "not available" not in ok.lower()

    monkeypatch.setattr(cs, "docker_available", lambda: False)
    no = await persona.registry.dispatch("check_docker", {}, ctx)
    assert "not available" in no.lower()
