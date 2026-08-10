import importlib.util
from pathlib import Path

import pytest

FACTORY_DIR = Path(__file__).resolve().parent.parent.parent / "personas" / "factory"


@pytest.fixture
def factory_tools():
    """Load personas/factory/tools.py fresh, the same way Persona.load does."""
    spec = importlib.util.spec_from_file_location(
        "factory_tools_under_test", FACTORY_DIR / "tools.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
