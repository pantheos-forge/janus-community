import pytest

from janus.core.container import ContainerSpec, ToolEntry


def _write(tmp_path, body):
    p = tmp_path / "container.toml"
    p.write_text(body)
    return p


def test_load_parses_install_and_tools(tmp_path):
    p = _write(tmp_path, '''
[install]
apt = ["ripgrep", "git"]
go = ["github.com/boyter/scc/v3@latest"]

[[tool]]
name = "rg"
description = "ripgrep search"
usage = "rg -n pat ."

[[tool]]
name = "scc"
description = "code counter"
''')
    spec = ContainerSpec.load(p)
    assert spec.apt == ["ripgrep", "git"]
    assert spec.go == ["github.com/boyter/scc/v3@latest"]
    assert spec.pip == []
    assert spec.dockerfile_append == ""
    assert spec.tools == [
        ToolEntry("rg", "ripgrep search", "rg -n pat ."),
        ToolEntry("scc", "code counter", ""),
    ]


def test_inventory_text_lists_tools_with_usage(tmp_path):
    p = _write(tmp_path, '''
[[tool]]
name = "rg"
description = "ripgrep search"
usage = "rg -n pat ."
''')
    text = ContainerSpec.load(p).inventory_text()
    assert "Tools available in your environment" in text
    assert "`rg` — ripgrep search" in text
    assert "rg -n pat ." in text


def test_load_rejects_non_string_install_entry(tmp_path):
    p = _write(tmp_path, '[install]\napt = ["ok", 3]\n')
    with pytest.raises(ValueError):
        ContainerSpec.load(p)


def test_load_rejects_tool_missing_name_or_description(tmp_path):
    p = _write(tmp_path, '[[tool]]\nname = "rg"\n')  # no description
    with pytest.raises(ValueError):
        ContainerSpec.load(p)


def test_load_rejects_non_table_tool_entry(tmp_path):
    p = _write(tmp_path, 'tool = ["not a table"]\n')
    with pytest.raises(ValueError):
        ContainerSpec.load(p)
