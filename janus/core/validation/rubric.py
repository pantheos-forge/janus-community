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

"""Declarative validation ``Rubric`` model + TOML loader.

A rubric describes how a persona's deliverable is judged: the list of subject
``tasks`` it should be exercised against, the ``criteria`` a judge checks the
deliverable against, and the pass gate (``pass_threshold`` combined with
``mode``, either requiring ``"all"`` criteria to pass or the ``"mean"`` score
to clear the threshold).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_VALID_MODES = {"all", "mean"}


@dataclass
class Criterion:
    name: str
    description: str


@dataclass
class Rubric:
    tasks: list[str]
    criteria: list[Criterion]
    pass_threshold: float = 0.7
    mode: str = "all"

    def __post_init__(self) -> None:
        """Guard invariants for every ``Rubric``, however it is constructed.

        These same guards used to live only in :meth:`load`, which meant a
        directly-constructed ``Rubric(...)`` (e.g. built in code rather than
        parsed from TOML) could carry an empty ``criteria`` list or an invalid
        ``mode`` and blow up later — a ``ZeroDivisionError`` in
        ``aggregate_and_gate``'s ``"mean"`` mode, or a silent fall-through for
        a typo'd mode. Centralizing them here makes every ``Rubric`` instance
        self-validating.
        """
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode {self.mode!r}: must be one of {sorted(_VALID_MODES)}")
        if not self.tasks:
            raise ValueError("Rubric must declare at least one task")
        if not self.criteria:
            raise ValueError("Rubric must declare at least one criterion")

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        tasks = data.get("tasks", [])
        criteria = [Criterion(c["name"], c["description"]) for c in data.get("criteria", [])]
        pass_threshold = data.get("pass_threshold", 0.7)
        mode = data.get("mode", "all")

        return cls(
            tasks=tasks,
            criteria=criteria,
            pass_threshold=pass_threshold,
            mode=mode,
        )
