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

"""``validate()`` — ties smoke and judge into a single :class:`ValidationReport`.

Smoke gates the judge: the judge only runs (and its backend is only built) if
the smoke run passes. Both backends are built via injected factories so
production can wrap ``build_backend`` while tests inject scripted fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from janus.core.events import EventBus
from janus.core.validation.judge import JudgeResult, aggregate_and_gate, judge_one
from janus.core.validation.smoke import SmokeCheck, SmokeResult, smoke_run

if TYPE_CHECKING:
    from janus.core.backend import AgentBackend
    from janus.core.persona import Persona
    from janus.core.session import SessionStore
    from janus.core.validation.rubric import Rubric

MakeAgentBackend = Callable[["Persona", Path], "AgentBackend"]
MakeJudgeBackend = Callable[["Rubric", Path], "AgentBackend"]
"""Factory for the judge's backend.

CONTRACT: a production ``make_judge_backend`` MUST construct the backend with
``janus.core.validation.judge.JUDGE_SYSTEM_PROMPT`` as its ``system_prompt``
and ``janus.core.validation.judge.judge_registry(rubric)`` as its tool
registry. Without the strict-evaluator system prompt, the judge silently
runs prompt-less and its verdicts are not framed as a rigorous evaluation.
"""


@dataclass
class ValidationReport:
    smoke: SmokeResult
    judge: JudgeResult | None

    @property
    def passed(self) -> bool:
        return self.smoke.passed and (self.judge is not None and self.judge.passed)


async def validate(
    persona: Persona,
    rubric: Rubric,
    make_agent_backend: MakeAgentBackend,
    make_judge_backend: MakeJudgeBackend,
    working_root: Path,
    *,
    session_store: SessionStore | None = None,
) -> ValidationReport:
    """Run smoke on every rubric task; if all pass, judge the deliverable.

    ``make_judge_backend`` builds the judge's :class:`AgentBackend`. The
    production factory MUST build that backend with
    ``janus.core.validation.judge.JUDGE_SYSTEM_PROMPT`` as its
    ``system_prompt`` and ``judge_registry(rubric)`` as its tool registry —
    that is the contract for a correctly-framed, strict-evaluator judge. A
    factory that omits ``JUDGE_SYSTEM_PROMPT`` ships a prompt-less judge
    silently, with no test or runtime error to catch the omission.
    """
    bus = EventBus()  # private: nested validation runs must not leak onto the global bus

    smoke_results: list[SmokeResult] = []
    for i, task in enumerate(rubric.tasks):
        ws = working_root / f"task{i}" / "smoke"
        if persona.container is not None:
            # Containerized agents run their tools inside their image; the
            # in-process backend factory is not used for the smoke phase.
            from janus.core.validation.container_smoke import container_smoke_run
            result = await container_smoke_run(persona, task, ws)
        else:
            backend = make_agent_backend(persona, ws)
            result = await smoke_run(
                persona, backend, task, ws, session_store=session_store, events=bus
            )
        smoke_results.append(result)

    smoke = SmokeResult(
        passed=all(s.passed for s in smoke_results),
        checks=[
            SmokeCheck(f"task{i}:{c.name}", c.ok, c.detail)
            for i, s in enumerate(smoke_results)
            for c in s.checks
        ],
        deliverable=smoke_results[0].deliverable if smoke_results else None,
    )
    if not smoke.passed:
        return ValidationReport(smoke, None)

    verdicts = []
    for i, task in enumerate(rubric.tasks):
        jws = working_root / f"task{i}" / "judge"
        jws.mkdir(parents=True, exist_ok=True)
        jbackend = make_judge_backend(rubric, jws)
        verdict = await judge_one(
            rubric,
            task,
            smoke_results[i].deliverable,
            jbackend,
            jws,
            session_store=session_store,
            events=bus,
        )
        verdicts.append(verdict)
    judge = aggregate_and_gate(rubric, verdicts)
    return ValidationReport(smoke, judge)
