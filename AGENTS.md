# timeline-testbed

A synthetic, known-answer testbed for one question: can a small local LLM find abuse evidence that
exists only in an activity timeline, where rules and feature models can't? Read this file, SPEC.md
and STATUS.md first. Read DECISIONS.md and `context/` as needed.

## Invariants

- **Two stores.** The inference store and the ground-truth store are separate files.
  - Only `testbed/eval/` opens a ground-truth store.
  - `testbed/world/` writes labels only through the vendored write-only sink.
  - A test enforces both.
- **`vendor/` is read-only.** It holds EntitlementShield at `fd5deca`. To adapt something, copy it into
  `testbed/`.
- **The generator spec freezes at the end of M1**, with its hash in DECISIONS.md. A later change needs
  the owner and a DECISIONS line, and it voids results from earlier worlds.
- **Gate thresholds in `prereg/` are fixed before their scoring run.** They don't change after a
  result is seen.
- **Confirmation worlds (`runs/confirm-*`)** are opened only by the M6 scorer, after the M6 freeze.
- **No paid APIs, hosted models or GPU rentals.** Package installs and model downloads need the
  owner's go in that turn.
- **Git:** never commit or push; the owner commits. No destructive git, and no deletions outside
  `runs/`, without the owner's go.
- **Secret-guard blocks are final.**

## Standing permissions (owner: confirm or edit before the first session)

- `make test`, `make lint` and `make typecheck`, at any time. They run offline and write only to
  temporary directories and caches.
- Generating development worlds and running the label-free pipeline into `runs/` (gitignored): persist,
  detect, units, render, features, and zero-shot or probe inference on local models already present.
- Reading development-world labels in `testbed/eval/`, for development, health checks and gates.
- Training runs (R3) need the owner's go per run, with M3's estimate of how long they'll take.

## Before each experiment

Write these down in STATUS.md or the prereg file:

1. The decision each possible outcome would change.
2. The number of independent groups per class in the population.
3. That a full-size smoke test of the route passed.

If item 1 has no answer, or item 2 is under 20 abusive groups, the run is a smoke test: a five-line
note and no write-up.

## Process budget

- **STATUS.md:** at most 80 lines covering the goal, state, next step, open owner decisions and the
  last check results. It's not a diary.
- **DECISIONS.md:** one line per decision (date, who, what, link).
- **Reports:** one page per milestone in `reports/`. Prereg files are one page each, in `prereg/`.
- **Cross-vendor review** (Codex, via the adversarial-review or orca-review skill) happens only at the
  M1 freeze and at the M6 pre-registration: one round, plus one check of the fixes. Everything else
  gets a self-review of the diff, plus tests.
- **Pivots:** changing direction needs a short pivot note in DECISIONS.md first, saying why the old
  question is answered or can't be answered, and why the new one serves the goal better.
- **Tests:** every analysis function that produces a reported number has a known-answer test.
- **No new agent tooling** (orchestration, memory, hooks) during the project.
- **Delegation** to Orca workers: only on request.
- **Data stays under `runs/`**, which is gitignored, so no index or commit picks it up.

## Commands (create them in M0)

- `make test` runs pytest, `make lint` runs ruff, and `make typecheck` runs mypy, over `testbed/` and
  `tests/`.
- The vendored test suite isn't part of `make test`. M0 records its baseline once.

## Verification

- Quote the actual summary line and exit code of every check you report.
- Report failures and skips as they are.
- Update STATUS.md before stopping.
