# The Janus Factory

The factory is a Janus persona that builds other Janus personas. It runs in
the ordinary runtime (TUI or headless) and drives the conversational build
flow: clarify -> research -> propose spec -> your approval -> scaffold ->
validate -> fix (max 3 attempts) -> export.

Its three custom tools enforce the invariants in code, whatever the model
does: `scaffold_persona` rejects malformed artifacts and freezes the rubric
once validation has run; `validate_persona` runs the real smoke+judge harness
and hard-caps the fix budget at 3 attempts; `export_persona` refuses unless
the latest validation passed.

## Running the factory (live)

Requires a configured provider (see `.env.example`): a direct `ANTHROPIC_API_KEY`,
OpenRouter (`OPENROUTER_API_KEY` + `OPENROUTER_MODEL`), or Claude Code
(`USE_CLAUDE_AGENT_SDK=true`, using your `claude` CLI login).

    janus run --task "an agent that reviews legal contracts for risky clauses"

`run` defaults `--persona` to `factory`, so no `--persona` is needed to build an
agent — pass `--persona <name>` to run any other agent instead.

The factory converses through the ask_user tool. When it asks a question the
runtime enters awaiting-input: the TUI shows a question panel above the input
(with quick-reply chips when the factory offers choices — click one or press
its number, or just type); headless-TTY prints a numbered menu (reply with
the number or full text). The spec gate arrives as a question with
[Approve the spec] / [Request changes] chips. During the build, the right-side
Build panel tracks Scaffold → Validate (attempt count and per-criterion
judge scores, live) → Export. Interactive `ask_user` works on every provider
backend, including Claude Code; only piped/keyless runs degrade to
autonomous-with-assumptions (ask_user fails open).

New agents are named after **mythological deities or creatures** chosen for
conceptual fit to the request (e.g. a contract-risk reviewer → `themis`). The
factory calls `list_fleet_agents` first so it never reuses an assigned name,
and the export step also refuses a name already in the fleet.

Factory-built agents may also include `persona/banner.txt` — custom braille
splash art the factory sourced as public-domain or CC0 imagery from Wikimedia
Commons and converted via `set_persona_banner`, with provenance recorded
alongside in `banner_source.txt`. Agents without one simply show the default
Janus banner. Art conversion requires `pip install -e '.[art]'` in the
environment where the factory runs; if that extra isn't installed, banner art
is skipped gracefully and the build proceeds unaffected.

Artifacts land in the run workspace (`runs/factory/`):

    build/<name>/            the generated persona (4 declarative files)
    build/.state/<name>/     attempts.json + per-attempt validation workspaces
    exports/<name>/          the exported, self-contained agent repo
    output.json              the factory's build report (win or lose)

### Building containerized agents

Some agents need a domain CLI tool (a scanner, a data CLI, a media/codec
tool) beyond the builtins — the factory builds these as *containerized*
agents: a `container.toml` alongside the persona describes an `[install]`
list (apt/pip/go) and a `[[tool]]` entry per tool, and `validate_persona`
builds + runs a Docker image to smoke-test them for real.

This needs two things in the environment where the factory itself runs:

- **Docker available** (the factory calls `check_docker` first and falls
  back to a builtin-only design if it isn't).
- **Provider credentials exported into the shell environment, not only in
  `.env`.** The in-container validation run passes credentials to the
  container via `-e` from `os.environ`, so `.env`-only credentials never
  reach the container and the in-container run will fail to authenticate.

  Example:

      export OPENROUTER_API_KEY=sk-or-...
      export OPENROUTER_MODEL=some/model
      .venv/bin/python -m janus run --persona factory \
          --task "an agent that scans a repo for secrets with ripgrep"

## The Cycle-2 capstone

Definition of done (spec 2026-07-24): in a live session, the factory builds
an agent for a domain we have NOT hand-built, and the exported repo's
validation passed at or above threshold before export. Pick the domain on the
day; good candidates are ones with a clear deliverable shape
(contract-review, grant-writing, incident-postmortem). Evidence = the build
report plus the exported repo.

Failure is a valid outcome: if the rubric cannot be met in 3 attempts, the
factory exports nothing and reports real scores and a diagnosis. That
honesty is load-bearing — do not "fix" it.

## The Cycle-3B capstone

Same bar, new experience: in a real terminal, run

    .venv/bin/python -m janus run --persona factory \
        --task "an agent that <a domain we have not built>"

and drive the whole build through the UI — clarify in the question panel,
approve the spec with a chip, watch validation attempts and scores in the
Build panel, and finish with the exported repo + build report. Success =
exported repo's validation passed at threshold, session conducted entirely
in the UI (no driver scripts).

## Hermetic tests

    .venv/bin/python -m pytest tests/personas/ -v

`test_factory_tools.py` covers each invariant; `test_factory_persona.py` the
persona structure and report schema; `test_factory_e2e.py` drives the whole
flow with scripted fake backends (no key, no network).

## The fleet

Factory exports now land in the fleet home (`~/janus-agents/`, configurable via
`JANUS_FLEET_DIR`) and are registered automatically. Manage them with:

    janus fleet list                     # every agent + last validation
    janus fleet status <name>            # details + full validation history
    janus fleet run <name> "<subject>"   # run it (TUI/headless)
    janus fleet validate <name>          # re-validate; records drift
    janus fleet adopt <path>             # import an existing exported repo
    janus fleet sync [<name>]            # re-vendor the current runtime into agents

Agents you exported before the fleet existed can be imported with `adopt`.

### The dashboard

    janus dashboard          # or just `janus fleet` in a terminal

opens an interactive table of your agents. Live runs need a configured provider
(`ANTHROPIC_API_KEY`, or OpenRouter via `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`) —
without one, sessions immediately show `error`. Columns:

- **NAME** / **DOMAIN** — the agent and its subject area.
- **RUNTIME** — whether the agent's vendored `janus/` runtime matches the current
  source: `current`, `stale(N)` (N files behind — press `s` to sync), or
  `unsynced` (persona but no vendored runtime yet).
- **LAST VALIDATION** — the most recent verdict, compact: `PASS 2026-07-25 μ0.84`,
  where `μ` is the mean across the rubric's criteria. Press `d` for the full
  per-criterion breakdown.
- **SESSION** — any live session's state: `running`, `improving`, `validating`,
  `validated` / `failed` (last validation outcome), `queued`, `error`, plus an
  `[awaiting]` badge when the agent is asking you a question.

From the table:

- `r` run an agent on a subject      - `v` validate it (rubric harness → scores modal)
- `i` improve it (complaint → live factory session)  - `a` adopt an exported repo
- `s` sync its vendored runtime to the current source · `c` containerize a plain agent (see below)
- `n` rename an agent · `x` remove it from the fleet (deregister; files kept, re-adopt to restore)
- `d` show the selected agent's full validation scores (also Enter on an idle validated agent)
- Enter open the selected agent's live run/improve session; on a validated
  agent with no live session it re-shows the last scores instead
- `Escape` return to the table from a session · `Ctrl+Q` quit · `Ctrl+P` command palette

`v` runs the smoke + judge rubric harness in the background (the SESSION column
shows `validating`) and pops a modal with the pass/fail and per-criterion scores;
the result is also recorded to the agent's validation history. `i` spawns the
factory persona as a live, watchable session that diagnoses, tightens, re-validates,
and exports an improved version.

**Inside a session** (Enter on a running `run`/`improve` agent) you get the same
layout as a single factory run: the event feed on the left, the build panel on the
right, and — when the agent asks a question — a question panel above the input.
Answer by clicking a quick-reply chip, pressing its number, or just typing a
free-text reply; `Escape` drops you back to the table (the session keeps running).
`a` prompts for the path to an exported agent repo (a directory with
`persona/manifest.toml`, e.g. under `runs/factory/exports/`) and registers it into
the fleet.

Sessions run concurrently in-process (up to `JANUS_FLEET_MAX_CONCURRENT`, default
3; the rest queue) — switching between the table and a session never disturbs a
running agent, and one session crashing never affects the others.

### Containerize an existing agent (dashboard `c`)

Select a plain agent and press `c`. You'll be asked what tools it needs; the
factory researches exact package names, shows the install list at its spec-gate
for your approval, authors a `container.toml` + adds the `bash` builtin,
validates the agent **in-container**, and the dashboard auto-syncs its Docker
wrappers so it's immediately `docker compose build`-able. Requires Docker
running and a provider configured. Only applies to plain agents — an
already-containerized agent is refused (use Improve to change its tools).

### Improving an agent in place

    janus fleet improve <name> "what's wrong with its output"

runs the factory in improve mode: it loads the agent, baseline-validates it,
diagnoses your complaint against fresh judge feedback, tightens the prompt or
schema (the rubric is frozen — it's the contract), re-validates, and commits
the change in the agent's own git history (`improve: <summary>`). Rollback is
`git revert <sha>` in the agent's fleet directory. The build report shows each
attempt's per-criterion scores versus the baseline so any regression is visible.

### Propagating core fixes to the fleet

Each exported agent carries its own vendored copy of the `janus/` runtime, so a
fix you land in Janus core does not reach agents exported earlier. `janus fleet
sync` re-vendors the current runtime (the `janus/` package plus the generated
wrapper files) into your fleet agents and commits it in each agent's own git
history. The persona — its identity and any `improve` history — and `.env` are
never touched.

    janus fleet sync --dry-run           # preview what would change, write nothing
    janus fleet sync                     # sync every registered agent
    janus fleet sync <name>              # sync just one

Staleness is judged by content, not a version string, so an agent whose runtime
already matches is reported `current` with no commit. Each updated agent gets a
`sync: vendored runtime → <sha>` commit (rollback with `git revert`), and the
registry records the SHA it was synced to. A dirty agent repo is skipped (commit
or stash first, then re-sync); an agent that was only ever partially exported
(persona but no vendored runtime) is materialized into a complete standalone repo.
