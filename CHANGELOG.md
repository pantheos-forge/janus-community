# Changelog

All notable changes to Janus are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This changelog is **recent-forward**: it starts on 2026-07-28. For earlier
history, see the git log and the design specs under
`docs/superpowers/specs/`.

## [Unreleased]

### Fixed

- **`janus fleet remove --purge` now works on containerized agents** — a
  containerized agent's `runs/` holds root-owned files (the container runs as
  root and writes into the `./runs` bind mount), so the host user could not
  delete them: `--purge` deregistered the agent but silently left its directory
  on disk. It now falls back to a root-capable delete inside a container.
  Without Docker it degrades to the previous behaviour rather than failing.

- **Containerized agents now read `.env`** — a container run built its
  environment from the process environment only, so a provider configured in
  `.env` (the setup path the README documents) never reached the container and
  the agent exited with `No provider configured`, while host-side runs worked.
  Provider values now fall back to the loaded config. (`f321f7f`)
- **A failed containerized validate can be retried** — the `_export/` directory
  left by a failed attempt made the next one raise `FileExistsError` as an
  unhandled traceback, so the natural recovery (fix your config, run it again)
  crashed. (`f321f7f`)
- **`janus validate` explains a smoke failure** — it printed a bare
  `smoke: FAIL` and discarded the detail it had already captured; failed checks
  now report their name and detail. (`f321f7f`)

### Added

- **Janus braille-art logo (SVG)** — a scalable, font-independent SVG of the TUI
  splash medallion (the janiform Janus coin), decoded from the splash `_BANNER`
  into a dot-matrix of circles and themed to the amber/obsidian palette. Lives
  under `docs/marketing/`. (`128f4ca`)
- **Containerized agents run in-container, streamed live** — `run` a
  containerized agent (dashboard `r`, `janus fleet run`) inside its Docker image
  instead of in-process on the host, so its baked-in CLI tools are actually
  available. The run **streams live** — a scrolling dashboard log screen, and to
  the terminal for the CLI (read-only; no interactive prompts). Plain agents keep
  the live in-process run; routing is automatic on the agent's `container.toml`.
  (`3414ce7`, `5f28da6`)
- **Remove a fleet agent** — `janus fleet remove <name>` deregisters it (files
  kept on disk, reversible via `janus fleet adopt`); `--purge --yes` also deletes
  the directory + git history. A dashboard `x` action is **deregister-only** (with
  a confirm dialog + live-session guard) — the destructive purge is never
  reachable from the UI. (`926a0c2`)
- **Rename a fleet agent** — `janus fleet rename <old> <new>` and a dashboard `n`
  action: moves the directory, rewrites the manifest name, swaps the registry key
  (preserving validation history + synced-to), re-renders name-derived wrappers,
  and commits in the agent's repo. Rollback-safe (restores file contents on any
  failure); refuses a taken/invalid name or a dirty repo; the dashboard refuses
  an agent with a live session. (`080d882`)
- **Dashboard "Containerize" action (`c`)** — convert a plain fleet agent into a
  containerized one via an interactive factory session: research + spec-gate the
  tool list, author `container.toml` + add the `bash` builtin, validate the
  agent in-container, then auto-sync its Docker wrappers. Reuses existing factory
  tools (no new tool / no registry change). Fail-closed: nothing is exported
  unless validation passes. (`a9dd548`)
- **`JANUS_CONTAINER_SMOKE_TIMEOUT`** environment override for the in-container
  validation timeout, documented in `.env.example`. (`9d6bace`)
- **TUI by default in exported agents** — `textual` is now a base dependency of a
  generated agent, so every agent runs the Textual UI on a TTY (and still falls
  back to the headless renderer when there is no TTY). (`704e748`)

### Changed

- **Dashboard confirm dialog has clickable buttons** — the yes/no `ConfirmModal`
  (used by the deregister action) now renders real Yes/No buttons you can click,
  in addition to the `y`/`n`/`esc` keys. (`de206be`)
- **Containerized-agent images are smaller** — the Go toolchain + build/module
  caches are purged in the same layer as `go install`, keeping only the compiled
  tool binaries (saves ~1–2 GB for agents using Go-installed tools like
  syft/grype; realized on the agent's next `docker compose build`). (`d98953d`)
- **Generated agent `README.md` adapts to the agent's shape** — containerized
  agents document the `docker compose build/run` path; plain agents document a
  local virtualenv or plain `docker run`, with a note that `docker compose` is
  only for containerized agents. (`eefc6b1`)
- **In-container validation default timeout raised 900 → 1800 seconds** — a 900s
  cap failed slow-tool agents (e.g. OSINT CLI sweeps) even when the container was
  healthy and the deliverable schema-valid. (`9d6bace`)
- **`fleet sync` staleness detection is container-aware** — it renders the Ubuntu
  tool Dockerfile + `docker-compose.yml` for a containerized persona instead of
  the slim default, so containerized agents no longer read as perpetually stale
  and compose drift is visible. (`0e4a0d3`)
- **Factory README dashboard reference** updated with the `RUNTIME` staleness
  column and the `s` (sync) / `c` (containerize) key actions. (`c21801a`)

### Fixed

- **Containerized-run live log stayed empty mid-run** — the in-container
  agent's stdout was block-buffered (no TTY), so streamed lines only
  appeared at the end. The run now sets `PYTHONUNBUFFERED=1` so output
  flushes live. (`c8138a5`)
- **In-container validation no longer freezes the dashboard TUI** — the blocking
  `docker build` / `docker run` (and the export copytree) now run off the event
  loop via `asyncio.to_thread`, so the SessionView stays responsive during the
  multi-minute build. (`9d6bace`)
- **Exported `docker-compose.yml` empty-env-var crash** — compose now passes only
  set provider variables (bare `- NAME` form) instead of `- NAME=${NAME:-}`,
  which expanded an unset boolean to an empty string and crashed pydantic's bool
  parser inside the container. Config also coerces an empty-string boolean to its
  default (defense in depth). (`4311b32`)
- **Factory pins security scanners instead of `go install …@latest`** — an
  unversioned `gitleaks` build shipped a ruleset that silently detected zero
  secrets. The factory now installs scanners from pinned release binaries / tags,
  and validation guidance expects a known-positive fixture (a scanner that finds
  nothing on a planted secret is a tool-integration failure, not a clean result).
  (`4311b32`)

[Unreleased]: https://github.com/pantheos-forge/janus-community
