from pathlib import Path

import jsonschema
import pytest

from janus.core.persona import Persona

PERSONA_DIR = Path(__file__).parent.parent.parent / "personas" / "factory"


@pytest.fixture
def persona():
    return Persona.load(PERSONA_DIR)


def test_factory_loads_with_expected_identity(persona):
    assert persona.name == "factory"
    assert "agent" in persona.description.lower()


def test_registry_is_exactly_the_designed_toolset(persona):
    names = set(persona.registry.names())
    assert names == {
        "ask_user",
        "check_docker",
        "emit_output",
        "export_improved_persona",
        "export_persona",
        "glob",
        "list_fleet_agents",
        "load_fleet_persona",
        "read_file",
        "scaffold_persona",
        "set_persona_banner",
        "update_plan",
        "validate_persona",
        "web_fetch",
    }


def test_factory_has_no_write_or_shell_access(persona):
    names = set(persona.registry.names())
    assert "bash" not in names
    assert "write_file" not in names
    assert "edit_file" not in names


def test_task_template_carries_the_request(persona):
    task = persona.build_task("an agent that reviews legal contracts")
    assert "legal contracts" in task


def test_workspace_subdirs(persona):
    assert set(persona.workspace_subdirs) == {"build", "exports"}


_EXPORTED_REPORT = {
    "status": "exported",
    "agent": {"name": "haiku_scout", "domain": "poetry",
              "description": "Writes a haiku about a subject."},
    "attempts": [{"smoke_passed": True, "judge_passed": True,
                  "scores": {"form": 0.9}, "feedback_digest": "reads like a haiku",
                  "changes_made": "initial version"}],
    "export_path": "exports/haiku_scout",
    "how_to_run": 'python agent.py "<subject>"',
}

_FAILED_REPORT = {
    "status": "failed",
    "agent": {"name": "haiku_scout", "domain": "poetry",
              "description": "Writes a haiku about a subject."},
    "attempts": [{"smoke_passed": True, "judge_passed": False,
                  "scores": {"form": 0.4}, "feedback_digest": "not haiku-like",
                  "changes_made": "tightened the prompt's form instructions"}],
    "diagnosis": "The judge wants strict 5-7-5; the model cannot count syllables reliably.",
}


def test_report_schema_accepts_an_exported_report(persona):
    jsonschema.validate(_EXPORTED_REPORT, persona.output_schema)


def test_report_schema_accepts_a_failed_report(persona):
    jsonschema.validate(_FAILED_REPORT, persona.output_schema)


def test_report_schema_rejects_exported_without_path(persona):
    bad = {k: v for k, v in _EXPORTED_REPORT.items() if k != "export_path"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, persona.output_schema)


def test_report_schema_rejects_failed_without_diagnosis(persona):
    bad = {k: v for k, v in _FAILED_REPORT.items() if k != "diagnosis"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, persona.output_schema)


def test_prompt_encodes_the_flow_and_the_gate(persona):
    prompt = persona.system_prompt.lower()
    for needle in ("clarify", "research", "spec", "approval", "ask_user", "scaffold_persona",
                   "validate_persona", "export_persona", "load_fleet_persona", "export_improved_persona",
                   "emit_output", "frozen", "choices"):
        assert needle in prompt, f"prompt.md is missing {needle!r}"


def test_prompt_teaches_the_rubric_file_format(persona):
    """Live-capstone finding: without the concrete TOML contract the factory
    guessed rubric formats 15+ times. The prompt must show it."""
    prompt = persona.system_prompt
    assert "tasks = [" in prompt
    assert "[[criteria]]" in prompt
    assert "pass_threshold" in prompt
    assert 'file = "prompt.md"' in prompt   # manifest example present too


def test_prompt_encodes_the_containerization_branch(persona):
    p = persona.system_prompt.lower()
    assert "check_docker" in p                       # pre-check
    assert "container.toml" in p                     # the mechanism
    assert "builtin-only" in p or "builtin only" in p  # the fallback
    # research-grounded package names
    assert "package name" in p or "apt" in p
    # tool list approved at the gate
    assert "container" in p and ("approve" in p or "ask_user" in p or "gate" in p)
    # every inventoried tool must actually be installed (not just described),
    # and non-apt Go/binary tools install via go install / a release binary
    assert "every tool" in p
    assert "go install" in p
    # scanners: prefer a PINNED release binary / version tag over @latest — a
    # real capstone bug shipped gitleaks@latest whose ruleset silently found
    # nothing even on known secrets.
    assert "pinned" in p
    # and treat a scanner that finds nothing on a known-positive fixture as a
    # red flag, not a clean result.
    assert "red flag" in p


def test_prompt_covers_banner_art_flow(persona):
    text = persona.system_prompt
    assert "set_persona_banner" in text
    # sourcing + license constraints are stated where the model can see them
    assert "Commons" in text
    # the soft cap that prevents an art retry-spiral is present
    assert "3 source images" in text
    # write timing: art is written only after validation passes
    assert "write=true" in text


def test_prompt_covers_containerizing_an_existing_agent(persona):
    """Dashboard Containerize action: the factory must know how to add a
    container to an ALREADY-EXPORTED agent without rewriting its prompt/schema."""
    p = persona.system_prompt.lower()
    assert "containerize" in p                     # the branch exists
    assert "load_fleet_persona" in p               # start from the existing agent
    assert "export_improved_persona" in p          # in-place, git-preserving export
    # re-pass the existing declarative files unchanged; only add the container + bash
    assert "verbatim" in p
    assert "bash" in p
    # A distinctive phrase that appears ONLY in the new "Containerizing an
    # EXISTING agent" bullet (occurs exactly once in the prompt) — so this
    # assertion fails if that specific bullet is removed, unlike the substrings
    # above which pre-exist elsewhere in the prompt.
    assert "containerize request" in p
