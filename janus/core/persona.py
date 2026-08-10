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

"""Declarative Persona model + loader — Janus's central abstraction.

A persona is a self-describing directory: ``manifest.toml`` (identity + wiring),
``prompt.md`` (system prompt) or a prompt builder, an optional ``tools.py`` (custom
``ToolSpec``s), and an optional ``output_schema.json`` (JSON Schema deliverable).
``Persona.load(directory)`` reads the manifest and assembles a ready-to-run
``Persona``: a composed ``ToolRegistry`` (builtins + custom + auto-added
``emit_output`` when an output schema is declared), the resolved system prompt,
and the task/workspace/provider/validation wiring.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from janus.core.container import ContainerSpec
from janus.core.tools.builtins import builtin_registry
from janus.core.tools.output import make_emit_output_tool
from janus.core.tools.registry import ToolRegistry, ToolSpec


def _load_attr(directory: Path, module_colon_attr: str) -> Any:
    """Import ``module:attr`` from a ``.py`` file inside ``directory`` and return ``attr``.

    Uses ``importlib.util.spec_from_file_location`` (rather than a package import) so
    a persona directory's helper modules load without needing to be on ``sys.path``.
    The module built via ``module_from_spec``/``exec_module`` is never inserted into
    ``sys.modules``, so it is executed fresh on every ``Persona.load`` call and two
    personas' same-named modules (e.g. ``tools.py``) never collide there; the
    per-directory unique name given to ``spec_from_file_location`` only sets the
    loaded module's ``__name__`` (useful for readable tracebacks), it isn't needed to
    prevent a collision that can't occur.
    """
    module_name, _, attr_name = module_colon_attr.partition(":")
    unique_name = f"_persona_{directory.name}_{module_name}"
    spec = importlib.util.spec_from_file_location(unique_name, directory / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name!r} from {directory}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr_name)


_BANNER_MIN_ROWS = 8
_BANNER_MAX_ROWS = 17
_BANNER_MAX_COLS = 70


def banner_errors(text: str) -> list[str]:
    """Validate braille banner art; return a list of violations (empty = valid).

    Shared by ``Persona.load`` (raises on violations) and the factory's
    ``set_persona_banner`` tool (returns them as error strings).
    """
    errors: list[str] = []
    lines = text.split("\n")
    if not (_BANNER_MIN_ROWS <= len(lines) <= _BANNER_MAX_ROWS):
        errors.append(
            f"must be {_BANNER_MIN_ROWS}-{_BANNER_MAX_ROWS} rows, got {len(lines)}")
    if any(len(line) > _BANNER_MAX_COLS for line in lines):
        errors.append(f"rows must be at most {_BANNER_MAX_COLS} columns")
    if any(not (0x2800 <= ord(ch) <= 0x28FF) for line in lines for ch in line):
        errors.append("must contain only braille characters (U+2800-U+28FF)")
    return errors


@dataclass
class Persona:
    name: str
    description: str
    domain: str
    system_prompt: str
    task_template: str
    registry: ToolRegistry
    output_schema: dict[str, Any] | None
    output_filename: str
    workspace_subdirs: list[str]
    provider_model: str | None
    rubric_path: Path | None
    directory: Path
    banner: str | None = None
    container: ContainerSpec | None = None

    @classmethod
    def load(cls, directory: str | Path) -> Persona:
        directory = Path(directory)
        with open(directory / "manifest.toml", "rb") as f:
            manifest = tomllib.load(f)

        persona_section = manifest.get("persona", {})
        if "name" not in persona_section:
            raise ValueError("Persona manifest missing required [persona].name")
        name = persona_section["name"]
        description = persona_section.get("description", "")
        domain = persona_section.get("domain", "")

        prompt_section = manifest.get("prompt", {})
        if "file" in prompt_section:
            system_prompt = (directory / prompt_section["file"]).read_text()
        elif "builder" in prompt_section:
            builder = _load_attr(directory, prompt_section["builder"])
            system_prompt = builder()
        else:
            raise ValueError(f"Persona {name!r} manifest has no [prompt].file or [prompt].builder")

        tools_section = manifest.get("tools", {})
        reg = builtin_registry(tools_section.get("builtins", []))
        if "custom" in tools_section:
            custom_specs: list[ToolSpec] = _load_attr(directory, tools_section["custom"])
            for spec in custom_specs:
                reg.register(spec)

        output_section = manifest.get("output", {})
        output_filename = output_section.get("filename", "output.json")
        output_schema: dict[str, Any] | None = None
        if "schema_file" in output_section:
            schema_path = directory / output_section["schema_file"]
            output_schema = json.loads(schema_path.read_text())
            try:
                Draft202012Validator.check_schema(output_schema)
            except Exception as e:
                raise ValueError(
                    f"Persona {name!r} output schema is not a valid JSON Schema: {e}") from e

        if output_schema is not None:
            reg.register(make_emit_output_tool(output_schema, output_path=output_filename))

        task_section = manifest.get("task", {})
        if "template" not in task_section:
            raise ValueError(f"Persona {name!r} manifest missing required [task].template")
        task_template = task_section["template"]
        if "{subject}" not in task_template:
            raise ValueError(
                f"Persona {name!r} [task].template must contain the '{{subject}}' placeholder")

        workspace_section = manifest.get("workspace", {})
        workspace_subdirs = workspace_section.get("subdirs", [])

        provider_section = manifest.get("provider", {})
        provider_model = provider_section.get("model")

        validation_section = manifest.get("validation", {})
        rubric_file = validation_section.get("rubric_file")
        rubric_path = directory / rubric_file if rubric_file else None

        banner: str | None = None
        banner_path = directory / "banner.txt"
        if banner_path.exists():
            banner = banner_path.read_text().rstrip("\n")
            errs = banner_errors(banner)
            if errs:
                raise ValueError(
                    f"Persona {name!r} banner.txt invalid: " + "; ".join(errs))

        container: ContainerSpec | None = None
        container_path = directory / "container.toml"
        if container_path.exists():
            container = ContainerSpec.load(container_path)
            if "bash" not in tools_section.get("builtins", []):
                raise ValueError(
                    f"Persona {name!r} has container.toml but does not declare the "
                    "'bash' builtin — a containerized persona must declare `bash` "
                    "to run its tools ([tools].builtins).")
            system_prompt = system_prompt.rstrip() + "\n\n" + container.inventory_text() + "\n"

        return cls(
            name=name,
            description=description,
            domain=domain,
            system_prompt=system_prompt,
            task_template=task_template,
            registry=reg,
            output_schema=output_schema,
            output_filename=output_filename,
            workspace_subdirs=workspace_subdirs,
            provider_model=provider_model,
            rubric_path=rubric_path,
            directory=directory,
            banner=banner,
            container=container,
        )

    def build_task(self, subject: str) -> str:
        return self.task_template.format(subject=subject)

    def prepare_workspace(self, working_directory: str | Path) -> Path:
        working_directory = Path(working_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
        for subdir in self.workspace_subdirs:
            (working_directory / subdir).mkdir(parents=True, exist_ok=True)
        return working_directory
