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

"""Declarative container tool spec for a persona (``container.toml``).

Its presence marks a persona as *containerized*: export bakes ``[install]``
packages into an Ubuntu image and the ``[[tool]]`` inventory is appended to the
system prompt so the agent knows what's on PATH. Invocation is via the ``bash``
builtin — no per-tool Python code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolEntry:
    name: str
    description: str
    usage: str = ""


@dataclass(frozen=True)
class ContainerSpec:
    apt: list[str] = field(default_factory=list)
    pip: list[str] = field(default_factory=list)
    go: list[str] = field(default_factory=list)
    dockerfile_append: str = ""
    tools: list[ToolEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> ContainerSpec:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        install = data.get("install", {})

        def _str_list(key: str) -> list[str]:
            vals = install.get(key, [])
            if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                raise ValueError(f"container.toml [install].{key} must be a list of strings")
            return vals

        append = install.get("dockerfile_append", "")
        if not isinstance(append, str):
            raise ValueError("container.toml [install].dockerfile_append must be a string")

        tools: list[ToolEntry] = []
        for entry in data.get("tool", []):
            if not isinstance(entry, dict):
                raise ValueError("container.toml [[tool]] entry must be a table")
            name = entry.get("name")
            description = entry.get("description")
            if not isinstance(name, str) or not name:
                raise ValueError("container.toml [[tool]] entry missing a non-empty 'name'")
            if not isinstance(description, str) or not description:
                raise ValueError(f"container.toml [[tool]] {name!r} missing a 'description'")
            usage = entry.get("usage", "")
            if not isinstance(usage, str):
                raise ValueError(f"container.toml [[tool]] {name!r} 'usage' must be a string")
            tools.append(ToolEntry(name, description, usage))

        return cls(
            apt=_str_list("apt"), pip=_str_list("pip"), go=_str_list("go"),
            dockerfile_append=append, tools=tools,
        )

    def inventory_text(self) -> str:
        lines = [
            "## Tools available in your environment",
            "",
            "You run inside a container with these command-line tools on PATH. "
            "Invoke them with the bash tool as needed.",
            "",
        ]
        for t in self.tools:
            lines.append(f"- `{t.name}` — {t.description}")
            if t.usage:
                lines.append(f"    e.g. {t.usage}")
        return "\n".join(lines)
