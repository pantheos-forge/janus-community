import json

from janus.interface.components.build_panel import FACTORY_TOOLS, BuildPanel


def test_factory_tool_names_are_pinned_to_the_real_factory():
    """The panel's watched names are a convention with the factory persona —
    this test makes it a contract."""
    import importlib.util
    from pathlib import Path

    tools_py = Path(__file__).parent.parent.parent / "personas" / "factory" / "tools.py"
    spec = importlib.util.spec_from_file_location("factory_tools_pin", tools_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(FACTORY_TOOLS) <= {t.name for t in mod.TOOLS}


def test_panel_state_machine_from_tool_events():
    panel = BuildPanel()
    assert panel.active is False                     # hidden until factory-shaped

    panel.observe_tool("scaffold_persona", "completed", "Scaffolded persona 'x' ...")
    assert panel.active is True
    assert panel.phases["scaffold"] == "done"

    panel.observe_tool("validate_persona", "running", None)
    assert panel.phases["validate"] == "active"

    verdict = json.dumps({
        "attempt": 1, "attempts_remaining": 2, "passed": False,
        "smoke": {"passed": True, "checks": []},
        "judge": {"passed": False,
                  "scores": {"coverage": 0.6, "sourcing": 0.5},
                  "feedback": "thin sourcing"},
    })
    panel.observe_tool("validate_persona", "completed", verdict)
    assert panel.attempt == 1 and panel.attempts_remaining == 2
    assert panel.scores == {"coverage": 0.6, "sourcing": 0.5}
    assert panel.phases["validate"] == "failed"

    panel.observe_tool("validate_persona", "completed", json.dumps({
        "attempt": 2, "attempts_remaining": 1, "passed": True,
        "smoke": {"passed": True, "checks": []},
        "judge": {"passed": True, "scores": {"coverage": 0.9, "sourcing": 0.85},
                  "feedback": "good"},
    }))
    assert panel.phases["validate"] == "done"

    panel.observe_tool("export_persona", "completed", "Exported 'x' to /tmp/out. Run it...")
    assert panel.phases["export"] == "done"
    assert "/tmp/out" in (panel.export_path or "")


def test_panel_ignores_non_factory_tools_and_bad_json():
    panel = BuildPanel()
    panel.observe_tool("web_fetch", "completed", "<html>")
    assert panel.active is False
    panel.observe_tool("validate_persona", "completed", "not json at all")
    assert panel.active is True                      # phase tracked...
    assert panel.scores == {}                        # ...scores degrade gracefully


def test_infrastructure_error_marks_validate_failed_not_done():
    """validate_persona's infra path returns a prose "Infrastructure error
    ..." string (no attempt consumed), not the "Error..." string the budget
    -exhausted path uses. Both must render as a failed phase -- never the
    done glyph a checkmark would wrongly suggest."""
    panel = BuildPanel()
    panel.observe_tool(
        "validate_persona",
        "completed",
        "Infrastructure error during validation (no attempt consumed): boom",
    )
    assert panel.phases["validate"] == "failed"
    assert panel.scores == {}
