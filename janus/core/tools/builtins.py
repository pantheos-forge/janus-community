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

"""Domain-free builtin tools, ported from the upstream proprietary agent.

Each tool is a free-function handler wrapped as a ToolSpec, bound at call time
to a ToolContext (cwd + optional output emitter) instead of a backend instance.

Extracted from the upstream proprietary agent (read-only reference);
domain-flavored phrasing has been genericized, ``report_finding`` (domain-specific) has been dropped
entirely, and ``self._cwd`` has been replaced with ``ctx.cwd`` throughout. ``update_plan``
no longer mutates a pinned message in a backend's own history (there is no such state on
``ToolContext``); it validates and formats the checklist and returns a confirmation string,
with actually pinning it in context left to the generic loop (a later plan).
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import re
import shlex
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from janus.core.tools.registry import ToolContext, ToolRegistry, ToolSpec, tool

COMMAND_TIMEOUT = 300


def _shq(s: str) -> str:
    """Shell-quote a single argument for safe interpolation into a bash command."""
    return shlex.quote(s)


# web_fetch / glob tool bounds.
_WEB_FETCH_MAX_CHARS = 16000
_GLOB_MAX_RESULTS = 200

_HTML_MARKER_RE = re.compile(r"<!doctype html|<html[\s>]|<body[\s>]", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

_HTTP_TIMEOUT = 12
_USER_AGENT = "Janus/1.0"


def _ssl_context() -> ssl.SSLContext | None:
    """A verifying SSL context, preferring certifi's CA bundle when available.

    Some hosts leave Python without a discoverable system CA bundle, so plain
    ``urlopen`` fails every HTTPS verify. Use certifi's bundle when present; return
    ``None`` to let urllib fall back to its default context otherwise.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _http_get(url: str, headers: dict[str, str]) -> str | None:
    """Fetch ``url`` and return the decoded body, or ``None`` on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT, context=_ssl_context()) as resp:
            body: str = resp.read().decode("utf-8", errors="replace")
            return body
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _html_to_text(body: str) -> str:
    """Extract readable text from an HTML page; return non-HTML bodies unchanged.

    web_fetch is used to read docs (HTML) but also source, JSON, and raw HTTP
    responses — blindly stripping ``<...>`` would mangle code (``a < b``, generics).
    So only apply extraction when the body actually looks like an HTML document;
    otherwise return it verbatim.
    """
    if not _HTML_MARKER_RE.search(body):
        return body
    txt = _SCRIPT_STYLE_RE.sub(" ", body)
    txt = _TAG_RE.sub(" ", txt)
    txt = html.unescape(txt)
    lines = [re.sub(r"[ \t]{2,}", " ", ln.strip()) for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


async def _run_bash(ctx: ToolContext, command: str) -> str:
    """Execute a bash command in ``ctx.cwd`` and return its combined stdout/stderr."""
    if not command.strip():
        return "Error: empty command"
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=ctx.cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
        output = stdout.decode("utf-8", errors="replace")
        max_len = 16000
        if len(output) > max_len:
            output = output[:max_len] + f"\n\n[... truncated, {len(output)} total chars]"
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        return output or "(no output)"
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return f"Command timed out after {COMMAND_TIMEOUT}s: {command[:100]}"
    except Exception as e:
        return f"Error executing command: {e}"


@tool(
    "bash",
    "Execute a bash command and return its combined stdout/stderr output.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
        },
        "required": ["command"],
    },
)
async def _bash(ctx: ToolContext, command: str = "") -> str:
    return await _run_bash(ctx, command)


@tool(
    "write_file",
    "Write content to a file (path is relative to the working directory).",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
)
async def _write_file(ctx: ToolContext, path: str = "", content: str = "") -> str:
    effective_path = path.strip() if path and path.strip() else "output.txt"
    try:
        filepath = Path(ctx.cwd) / effective_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        note = f" (path defaulted to {effective_path!r})" if not path.strip() else ""
        return f"Written {len(content)} bytes to {effective_path}{note}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(
    "read_file",
    "Read the contents of a file.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
        },
        "required": ["path"],
    },
)
def _read_file(ctx: ToolContext, path: str = "") -> str:
    try:
        filepath = Path(path)
        if not filepath.is_absolute():
            filepath = Path(ctx.cwd) / path
        content = filepath.read_text()
        if len(content) > 16000:
            content = content[:16000] + f"\n\n[... truncated, {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool(
    "edit_file",
    "Replace an exact, unique substring in a file. Use for surgical edits instead of "
    "rewriting the whole file with write_file.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to working dir)"},
            "old_string": {
                "type": "string",
                "description": "Exact text to replace (must occur exactly once)",
            },
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
    },
)
def _edit_file(ctx: ToolContext, path: str = "", old_string: str = "", new_string: str = "") -> str:
    if not path.strip():
        return "Error: path is required"
    if not old_string:
        return "Error: old_string is required"
    try:
        fp = Path(ctx.cwd) / path.strip()
        text = fp.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"
    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1:
        return f"Error: old_string is not unique in {path} (found {count} times); add more context"
    try:
        fp.write_text(text.replace(old_string, new_string, 1))
    except Exception as e:
        return f"Error writing {path}: {e}"
    return f"Edited {path} (1 replacement)"


@tool(
    "grep",
    "Search file contents for a regex pattern under the working directory. Returns "
    "matching file:line:text. Prefer this over ad-hoc `bash grep` for navigation.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: working dir)",
            },
            "glob": {"type": "string", "description": "Optional filename glob filter, e.g. '*.py'"},
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive match (default false)",
            },
        },
        "required": ["pattern"],
    },
)
async def _grep(
    ctx: ToolContext,
    pattern: str = "",
    path: str = "",
    glob: str = "",
    ignore_case: bool = False,
) -> str:
    """Regex content search via ripgrep (falls back to grep -r), bounded output."""
    if not pattern.strip():
        return "Error: empty pattern"
    target = path.strip() or "."
    g = glob.strip()
    rg = ["rg", "--line-number", "--no-heading", "--color", "never", "--max-count", "200"]
    if ignore_case:
        rg.append("-i")
    if g:
        rg += ["--glob", g]
    rg += ["-e", pattern, target]
    # grep fallback must honor the glob filter too. GNU grep maps an include glob to
    # --include and an rg-style '!'-prefixed exclusion to --exclude.
    grep = ["grep", "-rn"]
    if ignore_case:
        grep.append("-i")
    if g:
        grep.append(("--exclude=" + g[1:]) if g.startswith("!") else ("--include=" + g))
    grep += ["-e", pattern, target]
    out = await _run_bash(
        ctx,
        "command -v rg >/dev/null 2>&1 && "
        + " ".join(_shq(a) for a in rg)
        + " || "
        + " ".join(_shq(a) for a in grep),
    )
    return out.strip() or "(no matches)"


@tool(
    "glob",
    "List files under the working directory matching a glob pattern, e.g. '*.conf' (top "
    "level) or '**/*.py' (recursive — use ** to recurse). Returns matching paths, "
    "bounded. Prefer this over `bash find/ls` for locating files by name.",
    {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, relative to the working dir. Use ** to recurse.",
            },
        },
        "required": ["pattern"],
    },
)
def _glob(ctx: ToolContext, pattern: str = "") -> str:
    """List files under the working directory matching ``pattern`` (``**`` recurses)."""
    pat = pattern.strip()
    if not pat:
        return "Error: pattern is required"
    if pat.startswith("/"):
        return "Error: pattern must be relative to the working directory"
    base = Path(ctx.cwd)
    try:
        matches = sorted(p for p in base.glob(pat) if p.is_file())
    except (ValueError, OSError) as e:
        return f"Error: invalid glob pattern: {e}"
    if not matches:
        return "(no files match)"
    rels = [str(p.relative_to(base)) for p in matches]
    shown = rels[:_GLOB_MAX_RESULTS]
    result = "\n".join(shown)
    if len(rels) > _GLOB_MAX_RESULTS:
        result += f"\n[... {len(rels) - _GLOB_MAX_RESULTS} more; refine the pattern]"
    return result


@tool(
    "web_fetch",
    "Fetch a URL over HTTP(S) and return its text content (HTML pages have tags "
    "stripped; source/JSON/raw responses are returned verbatim; bounded). Use to read "
    "docs or an HTTP response body without shelling out to curl/wget.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http:// or https:// URL to fetch"},
        },
        "required": ["url"],
    },
)
def _web_fetch(ctx: ToolContext, url: str = "") -> str:
    """Fetch an HTTP(S) URL and return its text (HTML stripped, bounded).

    Synchronous (network-blocking) — the loop offloads it to a thread so it never stalls
    the event loop. Only http/https; other schemes (file://, gopher://) are rejected.
    """
    url = url.strip()
    if not url:
        return "Error: url is required"
    if any(c.isspace() or ord(c) < 0x20 for c in url):
        # A model-emitted URL with an internal space raised
        # http.client.InvalidURL deep in urlopen (live-capstone crash);
        # reject it here with a message the model can act on.
        return (
            "Error: URL contains whitespace or control characters — "
            f"check the address and retry: {url!r}"
        )
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return f"Error: only http/https URLs are supported (got {scheme or 'no scheme'})"
    body = _http_get(url, {})
    if body is None:
        return f"Error: failed to fetch {url}"
    text = _html_to_text(body)
    if len(text) > _WEB_FETCH_MAX_CHARS:
        text = text[:_WEB_FETCH_MAX_CHARS] + f"\n\n[... truncated, {len(text)} total chars]"
    return text or "(empty response)"


_NO_USER_AVAILABLE = (
    "No user is available in this run; proceed with reasonable "
    "assumptions and complete the task."
)


@tool(
    "ask_user",
    "Ask the human user a question and wait for their reply. The reply is "
    "returned as this tool's result. Use for clarifications and approvals; "
    "if no user is available in this run, a note is returned instead — "
    "proceed with reasonable assumptions.",
    {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask the user"},
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional quick-reply choices to offer the user "
                               "(free-text reply is always also accepted)",
            },
        },
        "required": ["question"],
    },
)
async def _ask_user(ctx: ToolContext, question: str = "",
                    choices: list[str] | None = None) -> str:
    if not question.strip():
        return "Error: question is required"
    if ctx.await_user_reply is None:
        return _NO_USER_AVAILABLE
    return await ctx.await_user_reply(question, choices)


_STATUS_MARK: dict[str, str] = {"done": "[x]", "in_progress": "[~]", "pending": "[ ]"}


@tool(
    "update_plan",
    "Record or update your objective checklist. Call this EARLY with your plan and "
    "update it as you make progress. Your current plan stays pinned in context so you "
    "don't lose track of the objective.",
    {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done"],
                        },
                    },
                    "required": ["step", "status"],
                },
            },
        },
        "required": ["plan"],
    },
)
def _update_plan(ctx: ToolContext, plan: list[Any] | None = None) -> str:
    """Validate and format the objective checklist; return a pinned-checklist confirmation.

    Unlike the source (which mutated a single pinned message in the backend's own message
    history by identity), this handler has no message-history state to mutate. It formats
    the checklist and stashes it at ``ctx.extra["plan"]`` for the caller to pin into context
    however it sees fit — actually pinning it is the generic loop's responsibility (a later
    plan). This just validates the shape, formats it, and returns the confirmation string.
    """
    if not isinstance(plan, list) or not plan:
        return "Error: plan must be a non-empty list of {step, status} items"
    lines = ["CURRENT PLAN (keep this updated as you work):"]
    for item in plan:
        if not isinstance(item, dict):
            continue
        mark = _STATUS_MARK.get(str(item.get("status", "pending")), "[ ]")
        lines.append(f"  {mark} {str(item.get('step', '')).strip()}")
    ctx.extra["plan"] = "\n".join(lines)
    return "Plan recorded/updated and pinned to context."


BASH = _bash
WRITE_FILE = _write_file
READ_FILE = _read_file
EDIT_FILE = _edit_file
GREP = _grep
GLOB = _glob
WEB_FETCH = _web_fetch
ASK_USER = _ask_user
UPDATE_PLAN = _update_plan

BUILTINS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        BASH,
        WRITE_FILE,
        READ_FILE,
        EDIT_FILE,
        GREP,
        GLOB,
        WEB_FETCH,
        ASK_USER,
        UPDATE_PLAN,
    ]
}


def builtin_registry(names: list[str] | None = None) -> ToolRegistry:
    selected = names if names is not None else list(BUILTINS)
    reg = ToolRegistry()
    for name in selected:
        reg.register(BUILTINS[name])  # KeyError on unknown name — intentional
    return reg
