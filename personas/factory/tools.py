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

"""Factory tools — the code-enforced invariants of the factory persona.

Three tools wrap the Cycle-1 libraries: `scaffold_persona` (integrity-checked
writes + rubric freeze), `validate_persona` (harness + 3-attempt budget), and
`export_persona` (refuses without a passing validation). The LLM drives the
conversation; these tools make malformed personas, budget overruns,
rubric-gaming, and unvalidated exports impossible regardless of what it does.

Workspace layout (relative to the factory's working directory):
  build/<name>/          - exactly the four generated persona files
  build/.state/<name>/   - attempts.json + validation/attempt{N}/ workspaces
                           (kept out of build/<name>/ so exports stay clean)
  exports/<name>/        - default export destination
"""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from pathlib import Path

import jsonschema

from janus.core.config import load_config
from janus.core.persona import Persona
from janus.core.tools.builtins import BUILTINS
from janus.core.tools.registry import ToolContext, tool
from janus.core.validation.harness import validate
from janus.core.validation.production import (
    make_production_agent_backend,
    make_production_judge_backend,
)
from janus.core.validation.rubric import Rubric
from janus.factory import export_agent
from janus.fleet.improve import ImproveError, sync_and_commit
from janus.fleet.registry import FleetRegistry

MAX_VALIDATION_ATTEMPTS = 3

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_GENERATED_FILES = ("manifest.toml", "prompt.md", "output_schema.json", "rubric.toml")


def _build_dir(ctx: ToolContext, name: str) -> Path:
    return Path(ctx.cwd) / "build" / name


def _state_dir(ctx: ToolContext, name: str) -> Path:
    return Path(ctx.cwd) / "build" / ".state" / name


def _load_attempts(state_dir: Path) -> list[dict]:
    path = state_dir / "attempts.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["attempts"]


def _save_attempt(state_dir: Path, record: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    attempts = _load_attempts(state_dir)
    attempts.append(record)
    (state_dir / "attempts.json").write_text(json.dumps({"attempts": attempts}, indent=2))


def _check_manifest(name: str, manifest: dict) -> list[str]:
    """Structural requirements on a generated manifest; one message per problem."""
    errors: list[str] = []
    persona = manifest.get("persona", {})
    if persona.get("name") != name:
        errors.append(
            f"manifest.toml: [persona].name must be {name!r} (got {persona.get('name')!r})"
        )
    if not persona.get("description"):
        errors.append("manifest.toml: [persona].description is required")
    if manifest.get("prompt", {}).get("file") != "prompt.md":
        errors.append('manifest.toml: [prompt].file must be "prompt.md"')
    template = manifest.get("task", {}).get("template", "")
    if not isinstance(template, str):
        errors.append("manifest.toml: [task].template must be a string containing {subject}")
    elif "{subject}" not in template:
        errors.append("manifest.toml: [task].template must contain the {subject} placeholder")
    if manifest.get("output", {}).get("schema_file") != "output_schema.json":
        errors.append('manifest.toml: [output].schema_file must be "output_schema.json"')
    if manifest.get("validation", {}).get("rubric_file") != "rubric.toml":
        errors.append('manifest.toml: [validation].rubric_file must be "rubric.toml"')
    tools_section = manifest.get("tools", {})
    if "custom" in tools_section:
        errors.append(
            "manifest.toml: generated personas must be declarative — no [tools].custom"
        )
    builtins = tools_section.get("builtins", [])
    if not isinstance(builtins, list):
        errors.append("manifest.toml: [tools].builtins must be a list of builtin tool names")
    else:
        unknown = set(builtins) - set(BUILTINS)
        if unknown:
            errors.append(
                f"manifest.toml: unknown builtins {sorted(unknown)}; "
                f"available: {sorted(BUILTINS)}"
            )
    return errors


@tool(
    "scaffold_persona",
    "Write (or overwrite) a generated persona's four declarative files "
    "(manifest.toml, prompt.md, output_schema.json, rubric.toml) into build/<name>/. "
    "Every artifact is integrity-checked before anything is kept: the manifest must "
    "parse with the required layout, the output schema must itself be a valid JSON "
    "Schema, the rubric must load, and the assembled directory must load as a "
    "Persona. After validate_persona has run, the rubric is frozen and cannot change.",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Persona name, lowercase_snake_case; must match [persona].name",
            },
            "manifest_toml": {"type": "string", "description": "Full manifest.toml content"},
            "prompt_md": {"type": "string", "description": "Full prompt.md content"},
            "output_schema_json": {
                "type": "string",
                "description": "Full output_schema.json content (a JSON Schema)",
            },
            "rubric_toml": {"type": "string", "description": "Full rubric.toml content"},
            "container_toml": {
                "type": "string",
                "description": (
                    "Optional container.toml — declares domain CLI tools "
                    "([install] apt/pip/go) to bake into this agent's Docker image "
                    "and a [[tool]] inventory for the prompt. Include ONLY for a "
                    "containerized agent, and the manifest MUST declare the bash "
                    "builtin so the agent can run them."
                ),
            },
        },
        "required": ["name", "manifest_toml", "prompt_md", "output_schema_json", "rubric_toml"],
    },
)
def _scaffold_persona(
    ctx: ToolContext,
    name: str = "",
    manifest_toml: str = "",
    prompt_md: str = "",
    output_schema_json: str = "",
    rubric_toml: str = "",
    container_toml: str = "",
) -> str:
    if not _NAME_RE.fullmatch(name):
        return (
            "Error: name must be lowercase_snake_case (start with a letter; "
            f"letters, digits, underscores only) — got {name!r}"
        )

    errors: list[str] = []

    try:
        manifest = tomllib.loads(manifest_toml)
    except tomllib.TOMLDecodeError as e:
        errors.append(f"manifest.toml: TOML parse error: {e}")
        manifest = None
    if manifest is not None:
        errors.extend(_check_manifest(name, manifest))

    try:
        schema = json.loads(output_schema_json)
        if not isinstance(schema, dict):
            errors.append("output_schema.json: must be a JSON object (a JSON Schema)")
        else:
            jsonschema.validators.validator_for(schema).check_schema(schema)
    except json.JSONDecodeError as e:
        errors.append(f"output_schema.json: JSON parse error: {e}")
    except jsonschema.SchemaError as e:
        errors.append(f"output_schema.json: not a valid JSON Schema: {e.message}")
    except TypeError as e:
        errors.append(f"output_schema.json: not a valid JSON Schema: {e}")

    if not prompt_md.strip():
        errors.append("prompt.md: must not be empty")

    if errors:
        return "Scaffold rejected:\n- " + "\n- ".join(errors)

    build_dir = _build_dir(ctx, name)

    # Rubric freeze: once validation has consumed an attempt, the grading
    # contract is fixed — fixes must improve the persona, not its judge.
    if _load_attempts(_state_dir(ctx, name)):
        existing_rubric = build_dir / "rubric.toml"
        if existing_rubric.exists() and existing_rubric.read_text() != rubric_toml:
            return (
                f"Error: the rubric for {name!r} is frozen — validation has already "
                "run against it. Fix the persona (prompt, schema, manifest), not the "
                "rubric. If you believe the rubric itself is defective, stop and say "
                "so in your final report instead."
            )

    tmp = Path(ctx.cwd) / "build" / f".tmp_{name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "manifest.toml").write_text(manifest_toml)
    (tmp / "prompt.md").write_text(prompt_md)
    (tmp / "output_schema.json").write_text(output_schema_json)
    (tmp / "rubric.toml").write_text(rubric_toml)
    if container_toml.strip():
        (tmp / "container.toml").write_text(container_toml)

    try:
        Rubric.load(tmp / "rubric.toml")
    except (tomllib.TOMLDecodeError, ValueError, KeyError, TypeError) as e:
        shutil.rmtree(tmp)
        return (
            f"Scaffold rejected:\n- rubric.toml: {e}\n"
            "Required format — tasks is a TOP-LEVEL array of strings and each "
            "criterion is a [[criteria]] block:\n"
            'tasks = [\n  "first task/subject...",\n  "second task/subject...",\n]\n'
            "pass_threshold = 0.7\n"
            'mode = "all"  # or "mean"\n\n'
            "[[criteria]]\n"
            'name = "coverage"\n'
            'description = "What the judge should check."'
        )

    try:
        Persona.load(tmp)
    except Exception as e:
        shutil.rmtree(tmp)
        return f"Scaffold rejected:\n- the assembled persona does not load: {e}"

    build_dir.mkdir(parents=True, exist_ok=True)
    for fname in _GENERATED_FILES:
        shutil.copy2(tmp / fname, build_dir / fname)
    # container.toml is optional: promote it when present, and remove a stale one
    # when this scaffold did not include it (keep build/ matching what was validated).
    if (tmp / "container.toml").exists():
        shutil.copy2(tmp / "container.toml", build_dir / "container.toml")
        written = _GENERATED_FILES + ("container.toml",)
        dropped_container = False
    else:
        dropped_container = (build_dir / "container.toml").exists()
        (build_dir / "container.toml").unlink(missing_ok=True)
        written = _GENERATED_FILES
    shutil.rmtree(tmp)
    message = (
        f"Scaffolded persona {name!r} at build/{name} "
        f"({', '.join(written)}). It loads cleanly; run validate_persona next."
    )
    if dropped_container:
        message += (
            " — NOTE: removed the previous container.toml, so this agent is now "
            "builtin-only; re-pass container_toml if that was unintended."
        )
    return message


@tool(
    "validate_persona",
    "Run the full validation harness (smoke run + LLM judge, every rubric task) on a "
    "scaffolded persona. Returns the structured report: per-check smoke results, "
    "per-criterion judge scores, and the judge's written feedback. Each completed "
    f"verdict consumes one of {MAX_VALIDATION_ATTEMPTS} attempts; infrastructure "
    "errors consume nothing.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the scaffolded persona"},
        },
        "required": ["name"],
    },
)
async def _validate_persona(ctx: ToolContext, name: str = "") -> str:
    if not _NAME_RE.fullmatch(name):
        return (
            "Error: name must be lowercase_snake_case (start with a letter; "
            f"letters, digits, underscores only) — got {name!r}"
        )

    build_dir = _build_dir(ctx, name)
    if not (build_dir / "manifest.toml").exists():
        return f"Error: no scaffolded persona named {name!r} — call scaffold_persona first."

    state_dir = _state_dir(ctx, name)
    attempts = _load_attempts(state_dir)
    if len(attempts) >= MAX_VALIDATION_ATTEMPTS:
        return (
            f"Error: validation budget exhausted ({MAX_VALIDATION_ATTEMPTS} attempts "
            "used). Do not scaffold or validate again. Produce your final report now "
            "via emit_output with status='failed' and your best diagnosis."
        )

    try:
        persona = Persona.load(build_dir)
        rubric = Rubric.load(persona.rubric_path)
    except Exception as e:
        return f"Error: the scaffolded persona no longer loads: {e}"

    make_agent = ctx.extra.get("make_agent_backend", make_production_agent_backend)
    make_judge = ctx.extra.get("make_judge_backend", make_production_judge_backend)

    attempt_no = len(attempts) + 1
    working_root = state_dir / "validation" / f"attempt{attempt_no}"
    try:
        report = await validate(persona, rubric, make_agent, make_judge, working_root)
    except Exception as e:
        return (
            f"Infrastructure error during validation (no attempt consumed): {e}. "
            "You may retry once; if it recurs, stop and tell the user."
        )

    _save_attempt(
        state_dir,
        {
            "passed": report.passed,
            "smoke_passed": report.smoke.passed,
            "judge_passed": report.judge.passed if report.judge else False,
            "scores": report.judge.scores if report.judge else {},
            "feedback": report.judge.feedback if report.judge else "",
        },
    )
    return json.dumps(
        {
            "attempt": attempt_no,
            "attempts_remaining": MAX_VALIDATION_ATTEMPTS - attempt_no,
            "passed": report.passed,
            "smoke": {
                "passed": report.smoke.passed,
                "checks": [
                    {"name": c.name, "ok": c.ok, "detail": c.detail}
                    for c in report.smoke.checks
                ],
            },
            "judge": None
            if report.judge is None
            else {
                "passed": report.judge.passed,
                "scores": report.judge.scores,
                "feedback": report.judge.feedback,
            },
        },
        indent=2,
    )


@tool(
    "export_persona",
    "Export a validated persona as a self-contained vendored agent repo. Refused "
    "unless the most recent validate_persona run for this name passed. Default "
    "destination: exports/<name> in the workspace.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the scaffolded persona"},
            "dest": {
                "type": "string",
                "description": "Optional destination directory (default exports/<name>)",
            },
        },
        "required": ["name"],
    },
)
def _export_persona(ctx: ToolContext, name: str = "", dest: str = "") -> str:
    if not _NAME_RE.fullmatch(name):
        return (
            "Error: name must be lowercase_snake_case (start with a letter; "
            f"letters, digits, underscores only) — got {name!r}"
        )

    build_dir = _build_dir(ctx, name)
    if not (build_dir / "manifest.toml").exists():
        return f"Error: no scaffolded persona named {name!r} — call scaffold_persona first."

    state_dir = _state_dir(ctx, name)
    attempts = _load_attempts(state_dir)
    if not attempts or not attempts[-1].get("passed"):
        return (
            "Error: cannot export — the most recent validation did not pass "
            "(or none has run). Run validate_persona until it passes first."
        )

    try:
        persona = Persona.load(build_dir)
    except Exception as e:
        return f"Error: the scaffolded persona no longer loads: {e}"

    registering = not dest.strip()
    fleet_dir = (ctx.extra.get("fleet_dir") or str(load_config().fleet_dir)) if registering else ""
    dest_path = Path(fleet_dir) / name if registering else Path(dest)

    if not dest_path.is_absolute():
        dest_path = Path(ctx.cwd) / dest_path

    if registering:
        existing = FleetRegistry(fleet_dir).get(name)
        if existing is not None:
            return (
                f"Error: {name!r} already exists in the fleet. To change an existing "
                "agent use export_improved_persona (it preserves the agent's git "
                "history); export_persona is for NEW agents only."
            )

    try:
        out = export_agent(persona, dest_path, force=True)
    except Exception as e:
        return f"Error: export failed: {e}"

    if registering:
        try:
            reg = FleetRegistry(fleet_dir)
            reg.register(name, domain=persona.domain, description=persona.description,
                         source="factory", path=str(out))
            last = attempts[-1]
            reg.append_validation(name, scores=last.get("scores", {}),
                                  passed=bool(last.get("passed")), note="factory export")
        except Exception as e:
            return (
                f"Exported {name!r} to {out}, but fleet registration failed: {e}. "
                "You can register it manually with `janus fleet adopt`."
            )

    return (
        f"Exported {name!r} to {out}"
        + (" and registered it in the fleet." if registering else ".")
    )


@tool(
    "load_fleet_persona",
    "Load an EXISTING fleet agent into the factory workspace to improve it. "
    "Copies the agent's four declarative files into build/<name>/ with a fresh "
    "3-attempt validation budget and returns its recorded validation history so "
    "you can anchor improvements against its current scores.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the fleet agent to load"},
        },
        "required": ["name"],
    },
)
def _load_fleet_persona(ctx: ToolContext, name: str = "") -> str:
    if not _NAME_RE.fullmatch(name):
        return (
            "Error: name must be lowercase_snake_case (start with a letter; "
            f"letters, digits, underscores only) — got {name!r}"
        )
    fleet_dir = ctx.extra.get("fleet_dir") or str(load_config().fleet_dir)
    reg = FleetRegistry(fleet_dir)
    try:
        agent = reg.get(name)
    except Exception as e:
        return f"Error: cannot read the fleet registry: {e}"
    if agent is None:
        return f"Error: no fleet agent named {name!r} — check `janus fleet list`."

    src = Path(agent["path"]) / "persona"
    missing = [
        f for f in _GENERATED_FILES if not (src / f).exists()
    ]
    if missing:
        return f"Error: fleet agent {name!r} is missing persona files: {missing}"

    # Stage-then-promote: validate a temp copy before touching build/<name> or
    # its .state, so a malformed fleet agent can't clobber an in-progress build.
    tmp = Path(ctx.cwd) / "build" / f".tmp_load_{name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in _GENERATED_FILES:
        shutil.copy2(src / f, tmp / f)

    try:
        Persona.load(tmp)
    except Exception as e:
        shutil.rmtree(tmp)
        return f"Error: fleet agent {name!r} does not load: {e}"

    # Promote the validated files into the build dir.
    build_dir = _build_dir(ctx, name)
    build_dir.mkdir(parents=True, exist_ok=True)
    for f in _GENERATED_FILES:
        shutil.copy2(tmp / f, build_dir / f)
    shutil.rmtree(tmp)

    # Fresh attempt budget for the improvement campaign.
    state_dir = _state_dir(ctx, name)
    if state_dir.exists():
        shutil.rmtree(state_dir)

    history = agent.get("validation_history") or []
    return (
        f"Loaded fleet persona {name!r} into build/{name} with a fresh "
        f"{MAX_VALIDATION_ATTEMPTS}-attempt budget. Its validation history:\n"
        + json.dumps(history, indent=2)
        + "\nBaseline-validate it first, then diagnose and tighten."
    )


@tool(
    "export_improved_persona",
    "Commit an improved persona back into its existing fleet agent repo. "
    "Refused unless the most recent validate_persona run passed. Updates the "
    "agent's four files in place, commits 'improve: <summary>' in the agent's "
    "own git history (rollback = git revert), and records the new scores.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the fleet agent being improved"},
            "summary": {"type": "string", "description": "One-line summary of the change"},
        },
        "required": ["name", "summary"],
    },
)
def _export_improved_persona(ctx: ToolContext, name: str = "", summary: str = "") -> str:
    if not _NAME_RE.fullmatch(name):
        return (
            "Error: name must be lowercase_snake_case (start with a letter; "
            f"letters, digits, underscores only) — got {name!r}"
        )
    build_dir = _build_dir(ctx, name)
    if not (build_dir / "manifest.toml").exists():
        return f"Error: no persona named {name!r} in the workspace — call load_fleet_persona first."

    attempts = _load_attempts(_state_dir(ctx, name))
    if not attempts or not attempts[-1].get("passed"):
        return (
            "Error: cannot export the improvement — the most recent validation "
            "did not pass (or none has run). Validate until it passes first."
        )

    fleet_dir = ctx.extra.get("fleet_dir") or str(load_config().fleet_dir)
    reg = FleetRegistry(fleet_dir)
    agent = reg.get(name)
    if agent is None:
        return f"Error: {name!r} is not a registered fleet agent."
    agent_dir = Path(agent["path"])

    summary = summary.strip() or "persona improvement"
    try:
        sha = sync_and_commit(build_dir, agent_dir, summary=summary)
    except ImproveError as e:
        return f"Error: could not commit the improvement: {e}"

    last = attempts[-1]
    try:
        reg.append_validation(
            name, scores=last.get("scores", {}), passed=bool(last.get("passed")),
            note=f"improve: {summary}",
        )
    except Exception as e:
        return (
            f"Improved {name!r} (commit {sha}), but recording scores failed: {e}."
        )

    return (
        f"Improved {name!r}: committed {sha} in its fleet repo and recorded the "
        f"new scores. Rollback with `git revert {sha}` in {agent_dir}."
    )


@tool(
    "list_fleet_agents",
    "List the agent names already assigned in the fleet. Call this BEFORE "
    "proposing candidate names for a NEW agent so you never reuse an existing "
    "name.",
    {"type": "object", "properties": {}},
)
def _list_fleet_agents(ctx: ToolContext) -> str:
    fleet_dir = ctx.extra.get("fleet_dir") or str(load_config().fleet_dir)
    try:
        names = sorted(FleetRegistry(fleet_dir).agents().keys())
    except Exception as e:
        return (
            f"Note: could not read the fleet registry ({e}). Proceed with a "
            "mythological name; the export step will reject any collision."
        )
    if not names:
        return "No agents are assigned yet — any name is free."
    return "Assigned agent names (do not reuse): " + ", ".join(names)


@tool(
    "set_persona_banner",
    "Preview or set custom braille splash art for the persona being built, "
    "converted from a Wikimedia Commons image (Public domain or CC0 ONLY — "
    "other licenses are refused). With write=false (default) the rendered "
    "braille is returned for you to judge and iterate on (crop/invert/"
    "threshold). With write=true it writes banner.txt + banner_source.txt "
    "into the build directory (requires a scaffolded persona).",
    {
        "type": "object",
        "properties": {
            "commons_file": {
                "type": "string",
                "description": (
                    "Commons file title, e.g. "
                    "'File:Janus, from Illustrium imagines, 1517.svg'"
                ),
            },
            "rows": {"type": "integer", "description": "banner height in rows, 8-17 (default 16)"},
            "crop": {
                "type": "array", "items": {"type": "number"},
                "minItems": 4, "maxItems": 4,
                "description": "fractional crop box [left, top, right, bottom], each 0.0-1.0",
            },
            "invert": {"type": "boolean", "description": "dots = light pixels instead of dark"},
            "threshold": {"type": "integer", "description": "1-254, default 128"},
            "write": {
                "type": "boolean",
                "description": "write into the build dir instead of previewing",
            },
            "name": {"type": "string", "description": "persona name (required when write=true)"},
        },
        "required": ["commons_file"],
    },
)
def _set_persona_banner(
    ctx: ToolContext,
    commons_file: str = "",
    rows: int = 16,
    crop: list | None = None,
    invert: bool = False,
    threshold: int = 128,
    write: bool = False,
    name: str = "",
) -> str:
    from janus.core.persona import banner_errors
    from janus.factory.banner import (
        BannerError,
        convert_to_braille,
        default_fetcher,
        fetch_commons_image,
    )

    if not isinstance(commons_file, str) or not commons_file.startswith("File:"):
        return ("Error: commons_file must be a Wikimedia Commons title "
                "starting with 'File:' (not a URL).")
    fetcher = ctx.extra.get("banner_fetcher", default_fetcher)
    try:
        image_bytes, license_name = fetch_commons_image(fetcher, commons_file)
        art = convert_to_braille(
            image_bytes,
            rows=rows,
            crop=crop,
            invert=bool(invert),
            threshold=max(1, min(254, int(threshold))),
        )
    except BannerError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: banner generation failed: {e}"
    errs = banner_errors(art)
    if errs:
        return "Error: generated art failed validation: " + "; ".join(errs)
    if not write:
        return f"Preview ({license_name}):\n{art}"
    if not name:
        return "Error: name is required when write=true."
    build_dir = _build_dir(ctx, name)
    if not (build_dir / "manifest.toml").exists():
        return (f"Error: no scaffolded persona at build/{name} — call "
                "scaffold_persona first, then set the banner.")
    from datetime import date

    (build_dir / "banner.txt").write_text(art + "\n")
    (build_dir / "banner_source.txt").write_text(
        f"{commons_file} · {license_name} · Wikimedia Commons · "
        f"{date.today().isoformat()}\n")
    return f"Banner written to build/{name}/banner.txt ({license_name})."


@tool(
    "check_docker",
    "Report whether Docker is available here. Call this BEFORE proposing a "
    "containerized agent: a containerized agent can only be validated and run "
    "if Docker is present. If it is not, build a builtin-only agent instead (or "
    "ask the user whether to proceed without in-container validation).",
    {"type": "object", "properties": {}},
)
def _check_docker(ctx: ToolContext) -> str:
    from janus.core.validation.container_smoke import docker_available

    if docker_available():
        return "Docker is available — containerized agents can be built and validated."
    return (
        "Docker is NOT available — a containerized agent cannot be validated or run "
        "in this environment. Build a builtin-only agent, or ask the user whether to "
        "proceed without in-container validation."
    )


TOOLS = [
    _scaffold_persona,
    _validate_persona,
    _export_persona,
    _load_fleet_persona,
    _export_improved_persona,
    _list_fleet_agents,
    _set_persona_banner,
    _check_docker,
]
