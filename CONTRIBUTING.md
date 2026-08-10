# Contributing to Janus

Thanks for your interest! A few things to know:

## How this repository is maintained

Janus is developed in an internal repository and synced here. Your pull request
is very welcome — we apply accepted changes upstream with your authorship
preserved, and they appear here on the next sync. That means a merged
contribution may show up as part of a later sync commit rather than as your
original commit; the authorship credit travels with it.

## Developer Certificate of Origin (DCO)

We use the [DCO](https://developercertificate.org/) instead of a CLA. Sign off
every commit:

    git commit -s -m "your message"

The `-s` adds a `Signed-off-by:` line certifying you wrote the change (or have the
right to submit it) under the project's **AGPL-3.0** license. All contributions
are licensed under AGPL-3.0.

## Dev setup

Requires Python ≥ 3.12. Use a virtualenv:

    python -m venv .venv
    .venv/bin/pip install -e '.[dev]'

## Before you open a PR

    .venv/bin/python -m ruff check janus/ personas/
    .venv/bin/python -m pytest -q

Keep `janus/` and `personas/` ruff-clean. Keep the core **domain-agnostic** —
don't bake single-domain concepts into `janus/`.
