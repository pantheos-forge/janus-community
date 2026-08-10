# market-research — Janus proof agent

The hand-authored persona that proves the Janus template end-to-end: it can be
run, validated, and exported, and it produces a structured, sourced market brief.

## Toolset

Builtins only — `web_fetch`, `write_file`, `read_file`, `update_plan` — plus the
auto-added `emit_output`. No shell, no custom search tool.

## Run it (live)

From the repo root, with a provider configured (`export ANTHROPIC_API_KEY=...`):

```bash
# One-off run with the runtime TUI (or headless if not a TTY):
janus run --persona market_research --task "the market for at-home coffee equipment in the United States"

# The Cycle-1 capstone — run + validate against the rubric, print the report:
janus validate --persona market_research
```

`janus validate` runs the agent on `rubric.tasks[0]`, judges the deliverable on
**coverage / sourcing / structure** (pass threshold 0.7, all criteria), and exits 0
on a pass. The deliverable brief is written to the run's `output.json`.

## Automated capstone test

`tests/personas/test_market_research_live.py` runs the same live proof and asserts
a passing `ValidationReport`. It is skipped unless `ANTHROPIC_API_KEY` is set:

```bash
ANTHROPIC_API_KEY=... .venv/bin/python -m pytest tests/personas/test_market_research_live.py -v -s
```

The captured report is printed and written to the test's tmp dir.

## Export it (standalone repo)

```bash
janus export --persona market_research --dest /path/to/market-research-agent
```

Produces a self-contained git repo with the Janus core vendored in; `python agent.py
"<topic>"` runs it. Install the TUI extra there with `pip install -e '.[tui]'`.
