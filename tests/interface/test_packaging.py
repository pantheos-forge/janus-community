import ast
import importlib
from pathlib import Path

import janus.interface


def _imports_textual(module_path: Path) -> bool:
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "textual" for a in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "textual":
            return True
    return False


def test_interface_package_imports_without_textual_at_top_level():
    # These modules are imported in headless/CLI paths and must never pull textual.
    root = Path(janus.interface.__file__).parent
    assert not _imports_textual(root / "__init__.py")
    assert not _imports_textual(root / "theme.py")
    assert not _imports_textual(root / "headless.py")
    assert not _imports_textual(root / "cli.py")


def test_theme_exposes_semantic_hex_palette():
    theme = importlib.import_module("janus.interface.theme")
    for name in ("PRIMARY", "SUCCESS", "WARNING", "ERROR", "TEXT_PRIMARY"):
        value = getattr(theme, name)
        assert isinstance(value, str) and value.startswith("#") and len(value) in (4, 7)


def test_stylesheet_is_generic_not_pentest():
    styles = (Path(janus.interface.__file__).parent / "styles.tcss").read_text()
    assert "#activity_feed" in styles
    assert "#status_bar" in styles
    for pentest in ("header_shells", "shell-new", "TerminalWidget", "ShellScreen"):
        assert pentest not in styles, f"pentest selector leaked: {pentest}"
