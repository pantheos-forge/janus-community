# tests/core/validation/test_rubric.py
from pathlib import Path

import pytest

from janus.core.validation.rubric import Criterion, Rubric

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "personas" / "echo_brief" / "rubric.toml"


def test_load_parses_tasks_criteria_and_gate():
    r = Rubric.load(FIXTURE)
    assert r.tasks == ["renewable energy storage", "EV charging in Europe"]
    assert r.pass_threshold == 0.7
    assert r.mode == "all"
    assert r.criteria == [
        Criterion("coverage", "Covers the key aspects of the topic"),
        Criterion("clarity", "The brief is clear and well-structured"),
    ]


def test_defaults(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text('tasks = ["t"]\n[[criteria]]\nname = "c"\ndescription = "d"\n')
    r = Rubric.load(p)
    assert r.pass_threshold == 0.7
    assert r.mode == "all"


def test_invalid_mode_raises(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text('tasks = ["t"]\nmode = "median"\n[[criteria]]\nname = "c"\ndescription = "d"\n')
    with pytest.raises(ValueError, match="mode"):
        Rubric.load(p)


def test_empty_tasks_or_criteria_raises(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text('tasks = []\n[[criteria]]\nname = "c"\ndescription = "d"\n')
    with pytest.raises(ValueError):
        Rubric.load(p)


def test_directly_constructed_rubric_with_empty_criteria_raises():
    """__post_init__ guards a hand-built Rubric too, not just Rubric.load."""
    with pytest.raises(ValueError):
        Rubric(tasks=["t"], criteria=[], pass_threshold=0.7, mode="all")
