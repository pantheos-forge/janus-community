# Janus — an engine for building specialized AI agents.
# Copyright (C) 2026 Pantheos Forge
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. This program is distributed WITHOUT ANY WARRANTY;
# see the GNU AGPL <https://www.gnu.org/licenses/> for details.
#
# A persona exception applies — see LICENSE-EXCEPTION.

"""LLM-judge — verdict schema, run, aggregation, and pass gate.

The judge scores a persona's deliverable against a :class:`Rubric`'s criteria.
It dogfoods the Janus agent stack: it is itself a minimal agent whose only
tool is a dynamically-built ``emit_output`` (see
``janus.core.tools.output.make_emit_output_tool``), schema-forced to a
verdict shape of one 0-1 score per criterion plus one paragraph of feedback.

Running the judge once per rubric task yields one verdict dict each;
:func:`aggregate_and_gate` averages each criterion's score across those
verdicts and applies the rubric's pass gate (``"all"`` criteria must clear
``pass_threshold``, or the ``"mean"`` of criteria must clear it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from janus.core.config import load_config
from janus.core.controller import AgentController
from janus.core.events import EventBus
from janus.core.session import SessionStore
from janus.core.tools.output import make_emit_output_tool
from janus.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from janus.core.backend import AgentBackend
    from janus.core.validation.rubric import Rubric

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator. You will be given a subject, a "
    "deliverable produced for that subject, and a numbered list of criteria. "
    "For each criterion, assign a score from 0 to 1 reflecting how well the "
    "deliverable satisfies it (0 = not at all, 1 = fully). Then write one "
    "paragraph of feedback explaining your scores. Do not be lenient: only "
    "award high scores when the deliverable clearly earns them. Once you "
    "have decided on scores and feedback, call emit_output exactly once with "
    "the verdict."
)


def verdict_schema(rubric: Rubric) -> dict[str, Any]:
    """Build the JSON Schema for a verdict: one 0-1 score per criterion + feedback."""
    names = [c.name for c in rubric.criteria]
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    name: {"type": "number", "minimum": 0, "maximum": 1} for name in names
                },
                "required": names,
                "additionalProperties": False,
            },
            "feedback": {"type": "string"},
        },
        "required": ["scores", "feedback"],
        "additionalProperties": False,
    }


def judge_registry(rubric: Rubric) -> ToolRegistry:
    """A registry containing only the schema-forced ``emit_output`` verdict tool."""
    registry = ToolRegistry()
    registry.register(make_emit_output_tool(verdict_schema(rubric)))
    return registry


def build_judge_task(rubric: Rubric, subject: str, deliverable: dict[str, Any]) -> str:
    """The judge's task prompt: subject, deliverable JSON, and numbered criteria."""
    criteria_lines = [
        f"{i}. {c.name}: {c.description}" for i, c in enumerate(rubric.criteria, start=1)
    ]
    criteria_block = "\n".join(criteria_lines)
    deliverable_json = json.dumps(deliverable, indent=2)
    return (
        f"Subject:\n{subject}\n\n"
        f"Deliverable:\n{deliverable_json}\n\n"
        f"Criteria:\n{criteria_block}\n\n"
        "Score each criterion from 0 to 1 and provide one paragraph of "
        "feedback, then call emit_output with the verdict."
    )


async def judge_one(
    rubric: Rubric,
    subject: str,
    deliverable: dict[str, Any],
    judge_backend: AgentBackend,
    working_directory: Path,
    *,
    session_store: SessionStore | None = None,
    events: EventBus | None = None,
) -> dict[str, Any]:
    """Run the judge once and read back the verdict it wrote to ``output.json``.

    ``judge_backend`` is expected to already be configured with
    ``judge_registry(rubric)`` and ``cwd=working_directory``; the caller is
    responsible for creating ``working_directory``.
    """
    controller = AgentController(
        load_config(),
        backend=judge_backend,
        session_store=session_store or SessionStore(sessions_dir=working_directory / ".sessions"),
        events=events,
    )
    await controller.run(build_judge_task(rubric, subject, deliverable))

    output_path = working_directory / "output.json"
    if not output_path.exists():
        return {"scores": {}, "feedback": "<no verdict>"}
    return json.loads(output_path.read_text())


@dataclass
class JudgeResult:
    passed: bool
    scores: dict[str, float]
    feedback: str
    per_task: list[dict[str, Any]]


def aggregate_and_gate(rubric: Rubric, per_task_verdicts: list[dict[str, Any]]) -> JudgeResult:
    """Average each criterion's score across ``per_task_verdicts`` and apply the gate."""
    aggregated: dict[str, float] = {}
    for criterion in rubric.criteria:
        raw_scores = [
            float(verdict.get("scores", {}).get(criterion.name, 0.0))
            for verdict in per_task_verdicts
        ]
        aggregated[criterion.name] = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0

    if rubric.mode == "mean":
        passed = (sum(aggregated.values()) / len(aggregated)) >= rubric.pass_threshold
    else:
        passed = all(score >= rubric.pass_threshold for score in aggregated.values())

    feedback = "\n\n".join(verdict.get("feedback", "") for verdict in per_task_verdicts)

    return JudgeResult(
        passed=passed,
        scores=aggregated,
        feedback=feedback,
        per_task=per_task_verdicts,
    )
