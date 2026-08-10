import pytest

from janus.core.persona import Persona

_MANIFEST = '''
[persona]
name = "toolagent"
description = "uses tools"
domain = "demo"
[prompt]
file = "prompt.md"
[tools]
builtins = {builtins}
[task]
template = "Do: {{subject}}"
'''


def _make(tmp_path, *, builtins='["bash", "read_file"]', container=True):
    d = tmp_path / "toolagent"
    d.mkdir()
    (d / "manifest.toml").write_text(_MANIFEST.format(builtins=builtins))
    (d / "prompt.md").write_text("You are a tool agent.")
    if container:
        (d / "container.toml").write_text(
            '[install]\napt = ["ripgrep"]\n[[tool]]\nname = "rg"\ndescription = "search"\n')
    return d


def test_persona_exposes_container_and_inventory_in_prompt(tmp_path):
    p = Persona.load(_make(tmp_path))
    assert p.container is not None
    assert p.container.apt == ["ripgrep"]
    assert "Tools available in your environment" in p.system_prompt
    assert "`rg` — search" in p.system_prompt


def test_container_without_bash_builtin_raises(tmp_path):
    d = _make(tmp_path, builtins='["read_file"]')  # no bash
    with pytest.raises(ValueError, match="bash"):
        Persona.load(d)


def test_non_container_persona_has_no_container(tmp_path):
    d = _make(tmp_path, container=False)
    p = Persona.load(d)
    assert p.container is None
    assert "Tools available in your environment" not in p.system_prompt
