from pathlib import Path

from janus.core.persona import Persona

FIXTURE = Path(__file__).parent.parent / "fixtures" / "personas" / "echo_brief"
MINIMAL_FIXTURE = Path(__file__).parent.parent / "fixtures" / "personas" / "minimal"


def test_load_parses_manifest_and_prompt():
    p = Persona.load(FIXTURE)
    assert p.name == "echo_brief"
    assert p.domain == "testing"
    assert "test persona" in p.system_prompt.lower()
    assert p.provider_model == "claude-sonnet-5"
    assert p.workspace_subdirs == ["notes"]
    assert p.rubric_path == FIXTURE / "rubric.toml"


def test_registry_composes_builtins_custom_and_emit_output():
    p = Persona.load(FIXTURE)
    names = set(p.registry.names())
    assert {"bash", "write_file", "update_plan"} <= names   # builtins
    assert "echo_note" in names                              # custom
    assert "emit_output" in names                            # auto-added (output schema present)


def test_build_task_fills_template():
    p = Persona.load(FIXTURE)
    assert p.build_task("EV charging") == "Produce a brief about: EV charging"


def test_prepare_workspace_creates_subdirs(tmp_path):
    p = Persona.load(FIXTURE)
    wd = p.prepare_workspace(tmp_path / "wd")
    assert (wd / "notes").is_dir()


def test_output_schema_loaded():
    p = Persona.load(FIXTURE)
    assert p.output_schema["required"] == ["summary"]


def test_minimal_persona_with_sparse_manifest_loads_with_no_tools():
    """A manifest with no [tools]/[output]/[workspace]/[provider]/[validation] must
    load successfully and default to an EMPTY registry — no builtins (esp. no bash),
    no custom tools, and no emit_output (no output schema declared)."""
    p = Persona.load(MINIMAL_FIXTURE)
    assert p.registry.names() == []
    assert "bash" not in p.registry.names()
    assert p.output_schema is None
    assert p.workspace_subdirs == []
    assert p.provider_model is None
    assert p.rubric_path is None


# Task 5: Descriptive load errors
import pytest

_MANIFEST = '''
[persona]
name = "p"
[prompt]
file = "prompt.md"
[task]
template = "Do: {subject}"
[output]
schema_file = "output_schema.json"
'''


def _write(tmp_path, manifest, schema='{"type":"object"}'):
    (tmp_path / "prompt.md").write_text("sys")
    (tmp_path / "output_schema.json").write_text(schema)
    (tmp_path / "manifest.toml").write_text(manifest)
    return tmp_path


def test_missing_name_is_descriptive(tmp_path):
    _write(tmp_path, _MANIFEST.replace('name = "p"', ""))
    with pytest.raises(ValueError, match=r"\[persona\].name"):
        Persona.load(tmp_path)


def test_missing_task_template_is_descriptive(tmp_path):
    _write(tmp_path, _MANIFEST.replace('template = "Do: {subject}"', ""))
    with pytest.raises(ValueError, match=r"\[task\].template"):
        Persona.load(tmp_path)


def test_task_template_must_reference_subject(tmp_path):
    _write(tmp_path, _MANIFEST.replace("Do: {subject}", "Do the thing"))
    with pytest.raises(ValueError, match=r"\{subject\}"):
        Persona.load(tmp_path)


def test_malformed_output_schema_rejected_at_load(tmp_path):
    _write(tmp_path, _MANIFEST, schema='{"type": "not-a-real-type"}')
    with pytest.raises(ValueError, match=r"valid JSON Schema"):
        Persona.load(tmp_path)


def test_output_filename_defaults_and_overrides(tmp_path):
    _write(tmp_path, _MANIFEST)
    assert Persona.load(tmp_path).output_filename == "output.json"
    _write(tmp_path, _MANIFEST + '\nfilename = "brief.json"\n')  # under [output]
    assert Persona.load(tmp_path).output_filename == "brief.json"
