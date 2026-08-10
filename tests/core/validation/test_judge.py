import json
from pathlib import Path

import pytest
from janus.core.backends.generic import GenericBackend
from janus.core.validation.rubric import Criterion, Rubric
from janus.core.validation.judge import (
    JudgeResult, aggregate_and_gate, build_judge_task, judge_one, judge_registry, verdict_schema,
)

RUBRIC = Rubric(
    tasks=["t"],
    criteria=[Criterion("coverage", "covers it"), Criterion("clarity", "clear")],
    pass_threshold=0.7, mode="all",
)


def test_verdict_schema_has_a_score_per_criterion():
    s = verdict_schema(RUBRIC)
    props = s["properties"]["scores"]["properties"]
    assert set(props) == {"coverage", "clarity"}
    assert s["properties"]["scores"]["required"] == ["coverage", "clarity"]


def test_build_judge_task_embeds_criteria_and_deliverable():
    task = build_judge_task(RUBRIC, "solar", {"summary": "hi"})
    assert "coverage" in task and "clarity" in task and "solar" in task and "summary" in task


def test_gate_all_mode_requires_every_criterion():
    r = aggregate_and_gate(RUBRIC, [{"scores": {"coverage": 0.9, "clarity": 0.5}, "feedback": "f"}])
    assert r.scores == {"coverage": 0.9, "clarity": 0.5}
    assert r.passed is False   # clarity 0.5 < 0.7, mode "all"


def test_gate_all_mode_passes_when_all_meet_threshold():
    r = aggregate_and_gate(RUBRIC, [{"scores": {"coverage": 0.8, "clarity": 0.75}, "feedback": "f"}])
    assert r.passed is True


def test_gate_mean_mode():
    rubric = Rubric(tasks=["t"], criteria=RUBRIC.criteria, pass_threshold=0.7, mode="mean")
    r = aggregate_and_gate(rubric, [{"scores": {"coverage": 0.9, "clarity": 0.5}, "feedback": "f"}])
    assert r.passed is True     # mean 0.7 >= 0.7


def test_gate_fails_closed_on_no_verdict():
    """judge_one's fallback ({"scores": {}, "feedback": "<no verdict>"}) when no
    verdict file is produced must fail validation, not pass it: missing scores
    aggregate to 0.0, which is below any sane pass_threshold. This locks the
    safety-critical fail-closed property of the judge gate."""
    r = aggregate_and_gate(RUBRIC, [{"scores": {}, "feedback": "<no verdict>"}])
    assert r.passed is False
    assert r.scores == {"coverage": 0.0, "clarity": 0.0}


def test_gate_averages_across_tasks():
    r = aggregate_and_gate(RUBRIC, [
        {"scores": {"coverage": 1.0, "clarity": 0.8}, "feedback": "a"},
        {"scores": {"coverage": 0.6, "clarity": 0.8}, "feedback": "b"},
    ])
    assert r.scores["coverage"] == 0.8   # (1.0 + 0.6)/2
    assert "a" in r.feedback and "b" in r.feedback


class _VerdictBackend(GenericBackend):
    def __init__(self, *a, verdict, **k):
        super().__init__(*a, **k)
        self._verdict = verdict
        self._n = 0

    async def _chat_completion(self):
        self._n += 1
        if self._n == 1:
            return {"message": {"content": "judging", "tool_calls": [
                {"id": "c1", "function": {"name": "emit_output", "arguments": json.dumps(self._verdict)}}]}}
        return {"message": {"content": "done"}}

    def _tool_result_message(self, tc, result):
        return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}


@pytest.mark.asyncio
async def test_judge_one_captures_verdict(tmp_path):
    verdict = {"scores": {"coverage": 0.9, "clarity": 0.8}, "feedback": "solid"}
    wd = tmp_path / "j"
    wd.mkdir()
    backend = _VerdictBackend(working_directory=wd, system_prompt="judge",
                              model="m", registry=judge_registry(RUBRIC), verdict=verdict)
    out = await judge_one(RUBRIC, "solar", {"summary": "hi"}, backend, wd)
    assert out == verdict
