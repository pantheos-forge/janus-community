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

"""Container-mode smoke run for a containerized persona.

Builds the agent's tool image, runs the agent inside it with the workspace
mounted, and reads back ``output.json`` — batch, no live event stream. Docker is
required; when it is unavailable the run fails with a readable check rather than
crashing. The LLM judge remains host-side (unchanged).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jsonschema

from janus.core.config import load_config
from janus.core.validation.smoke import SmokeCheck, SmokeResult

if TYPE_CHECKING:
    from janus.core.persona import Persona

_PROVIDER_ENV = (
    "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "DS4_URL", "DS4_MODEL", "LOCAL_MODEL", "USE_CLAUDE_AGENT_SDK", "JANUS_AUTH_MODE",
)

# Provider env var -> JanusConfig field, for the .env fallback below.
# JANUS_AUTH_MODE is deliberately absent: it has no config field, so it stays
# process-env only.
_PROVIDER_CONFIG_FIELD = {
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "OPENROUTER_MODEL": "openrouter_model",
    "DS4_URL": "ds4_url",
    "DS4_MODEL": "ds4_model",
    "LOCAL_MODEL": "local_model",
    "USE_CLAUDE_AGENT_SDK": "use_claude_agent_sdk",
}


def _provider_env() -> dict[str, str]:
    """Provider variables to hand the container: process env first, then config.

    The container inherits nothing — it sees only what we pass with ``-e``.
    Reading ``os.environ`` alone silently drops a ``.env``-only configuration,
    which is the setup path the README documents, so a containerized agent died
    with "No provider configured" while host-side runs worked fine.
    """
    values = {k: v for k in _PROVIDER_ENV if (v := os.environ.get(k))}
    missing = [k for k in _PROVIDER_ENV if k not in values]
    if not missing:
        return values
    try:
        config = load_config()
    except Exception:  # noqa: BLE001 - config is best-effort here; never block a run
        return values
    for key in missing:
        field = _PROVIDER_CONFIG_FIELD.get(key)
        if field is None:
            continue
        value = getattr(config, field, None)
        if isinstance(value, bool):
            if value:
                values[key] = "true"
        elif value:
            values[key] = str(value)
    return values


def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


@dataclass
class _ContainerExec:
    docker_ok: bool
    build_ok: bool
    ran_ok: bool
    detail: str
    returncode: int | None
    out_dir: Path
    output_path: Path | None
    deliverable: dict | None
    deliverable_error: str


@dataclass
class ContainerRunResult:
    success: bool
    deliverable: dict | None
    output_path: Path | None
    returncode: int | None
    error: str | None


async def _export_build_run(
    persona: Persona, subject: str, working_directory: Path, *, timeout: int, tag: str,
    on_line: Callable[[str], None] | None = None,
) -> _ContainerExec:
    # Timeout applies to BOTH the image build and the in-container run. The
    # default is generous because slow-tool agents (OSINT/security CLI sweeps,
    # e.g. wide multi-site sweeps) legitimately need it — a 900s cap failed a real
    # capstone run despite a healthy container + schema-valid deliverable.
    # JANUS_CONTAINER_SMOKE_TIMEOUT overrides it globally (env-based to match this
    # module's provider-env convention, so no plumbing through validate() and
    # every caller). A malformed override keeps the default rather than crashing
    # the fail-closed validation.
    try:
        timeout = int(os.environ.get("JANUS_CONTAINER_SMOKE_TIMEOUT") or timeout)
    except ValueError:
        pass

    # Resolve to an absolute path: docker treats a RELATIVE `-v` source as a
    # named volume (whose name may not contain slashes), so a relative workspace
    # — as the factory's validate loop passes — is rejected with "invalid
    # characters for a local volume name". An absolute source is a bind mount.
    working_directory = Path(working_directory).resolve()
    working_directory.mkdir(parents=True, exist_ok=True)
    out_dir = working_directory / "out"
    if not docker_available():
        return _ContainerExec(False, False, False,
            "Docker is required to run a containerized agent (docker daemon "
            "unreachable).", None, out_dir, None, None, "")

    from janus.factory.export import export_agent  # lazy: avoid core->factory cycle
    export_dir = working_directory / "_export"
    # Off-thread every blocking step (export copytree + docker build/run) so the
    # caller's event loop stays live — in the dashboard the agent controller
    # shares the Textual loop, and a blocking multi-minute docker build/run froze
    # the TUI (capstone finding). asyncio.to_thread preserves subprocess.run's own
    # timeout + TimeoutExpired propagation, so the "never raises" contract holds.
    # force=True so a retry after a failed attempt works: the previous run left
    # _export/ behind, and export_agent refuses a non-empty destination. That
    # retry is the recovery path from a config error, so refusing it crashed
    # exactly the user who was already stuck.
    await asyncio.to_thread(export_agent, persona, export_dir, git_init=False, force=True)

    if on_line is not None:
        on_line("building image (first run may take a few minutes)…")
    try:
        build = await asyncio.to_thread(
            subprocess.run, ["docker", "build", "-t", tag, str(export_dir)],
            capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _ContainerExec(True, False, False, f"docker build failed: {e}",
                              None, out_dir, None, None, "")
    if build.returncode != 0:
        return _ContainerExec(True, False, False, (build.stderr or build.stdout)[-2000:],
                              None, out_dir, None, None, "")

    out_dir.mkdir(exist_ok=True)
    # PYTHONUNBUFFERED=1 so the in-container agent flushes stdout line-by-line:
    # `docker run` has no TTY, and Python block-buffers a non-TTY stdout, so
    # without this the headless event lines never reach the streaming reader
    # until the buffer flushes at exit (the live log would stay empty mid-run).
    env_args: list[str] = ["-e", "PYTHONUNBUFFERED=1"]
    for k, v in _provider_env().items():
        env_args += ["-e", f"{k}={v}"]
    cmd = ["docker", "run", "--rm", *env_args,
           "-v", f"{out_dir}:/agent/runs/{persona.name}", tag, subject]
    if on_line is None:
        try:
            run = await asyncio.to_thread(subprocess.run, cmd,
                                          capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e:
            return _ContainerExec(True, True, False, f"docker run failed: {e}",
                                  None, out_dir, None, None, "")
        returncode: int | None = run.returncode
        detail = "" if returncode == 0 else (run.stderr or run.stdout)[-2000:]
    else:
        returncode, detail = await _stream_docker_run(cmd, on_line, timeout)
        if returncode is None:
            return _ContainerExec(True, True, False, detail, None, out_dir, None, None, "")

    ran_ok = returncode == 0
    output_path = out_dir / persona.output_filename
    deliverable: dict | None = None
    deliverable_error = ""
    if not output_path.exists():
        deliverable_error = f"{output_path} does not exist"
    else:
        try:
            deliverable = json.loads(output_path.read_text())
        except json.JSONDecodeError as e:
            deliverable_error = str(e)
    return _ContainerExec(True, True, ran_ok, detail, returncode, out_dir,
                          output_path if output_path.exists() else None,
                          deliverable, deliverable_error)


async def _stream_docker_run(
    cmd: list[str], on_line: Callable[[str], None], timeout: int
) -> tuple[int | None, str]:
    """Run `cmd`, forwarding each stdout line to on_line. Returns
    (returncode|None, detail). None returncode = timeout/launch failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    except OSError as e:
        return None, f"docker run failed: {e}"
    tail: list[str] = []

    async def _pump() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip("\n")
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            on_line(line)
        await proc.wait()

    try:
        await asyncio.wait_for(_pump(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return None, f"docker run timed out after {timeout}s"
    rc = proc.returncode
    return rc, ("" if rc == 0 else "\n".join(tail)[-2000:])


async def container_run(
    persona: Persona, subject: str, working_directory: Path, *, timeout: int = 1800,
    on_line: Callable[[str], None] | None = None,
) -> ContainerRunResult:
    """Run a containerized agent inside its image on ``subject`` and return the
    deliverable. Batch/one-shot; never raises (Docker-absent/build/run failures
    become a result with ``success=False``). Unlike smoke, does NOT schema-gate.

    When ``on_line`` is given, streams each container stdout line to it (plus a
    build-status line) instead of running fully batched."""
    ex = await _export_build_run(persona, subject, working_directory,
                                 timeout=timeout, tag=f"janus-agent-{persona.name}:latest",
                                 on_line=on_line)
    if not ex.docker_ok or not ex.build_ok:
        return ContainerRunResult(False, None, None, None, ex.detail)
    if not ex.ran_ok:
        error = ex.detail or f"agent exited with code {ex.returncode}"
    elif ex.deliverable is None:
        error = ex.deliverable_error or "no output.json produced"
    else:
        error = None
    return ContainerRunResult(
        success=ex.ran_ok and ex.deliverable is not None,
        deliverable=ex.deliverable, output_path=ex.output_path,
        returncode=ex.returncode, error=error)


async def container_smoke_run(
    persona: Persona, subject: str, working_directory: Path, *, timeout: int = 1800,
) -> SmokeResult:
    ex = await _export_build_run(persona, subject, working_directory,
                                 timeout=timeout, tag=f"janus-validate-{persona.name}:latest")
    if not ex.docker_ok:
        return SmokeResult(False, [SmokeCheck("docker_available", False, ex.detail)])
    if not ex.build_ok:
        return SmokeResult(False, [SmokeCheck("image_build", False, ex.detail)])
    checks = [SmokeCheck("run_completed", ex.ran_ok,
                         "Run completed." if ex.ran_ok else ex.detail)]
    deliverable = None
    if persona.output_schema is not None:
        if ex.deliverable is None:
            checks.append(SmokeCheck("deliverable_valid", False,
                                     ex.deliverable_error or "no deliverable"))
        else:
            try:
                jsonschema.validate(instance=ex.deliverable, schema=persona.output_schema)
            except jsonschema.ValidationError as e:
                checks.append(SmokeCheck("deliverable_valid", False, str(e)))
            else:
                checks.append(SmokeCheck("deliverable_valid", True,
                                         "Deliverable is schema-valid."))
                deliverable = ex.deliverable
    return SmokeResult(passed=all(c.ok for c in checks), checks=checks, deliverable=deliverable)
