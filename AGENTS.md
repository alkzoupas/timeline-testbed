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
  - Exception (owner, 2026-10-22), for M8 scoring only: Claude Haiku 5.5 (`claude-haiku-5-5`)
    through the owner's Claude subscription. It runs as headless Claude Code with
    `CLAUDE_CONFIG_DIR=~/.claude-haiku-scorer` and `--safe-mode --tools "" --strict-mcp-config
    --no-session-persistence`, from an empty temp directory outside the repo whose path names
    nothing. No API keys and no other hosted model. Haiku sees only the frozen prompt, the renders
    and the pre-registered examples, plus what Claude Code adds to every call and no flag removes:
    an environment block (working directory, platform, date, model) and a note with the account
    identifier. A scoring step isn't new agent tooling. Never read, list or print anything inside
    `~/.claude-haiku-scorer`.
- **Git:** never commit or push; the owner commits. No destructive git, and no deletions outside
  `runs/`, without the owner's go.
- **Secret-guard blocks are final.**

## Standing permissions (confirmed by the owner, 2026-10-21)

- `make test`, `make lint` and `make typecheck`, at any time. They run offline and write only to
  temporary directories and caches.
- Generating development worlds and running the label-free pipeline into `runs/` (gitignored): persist,
  detect, units, render, features, and zero-shot or probe inference on local models already present.
- Reading development-world labels in `testbed/eval/`, for development, health checks and gates.
- Running the existing `~/venvs/es-p6-train` read-only (MLX inference and tokenizer counts), with no
  package changes.
- Training runs (R3) need the owner's go per run, with M3's estimate of how long they'll take.

## Delegation and reviews

- **Coordinator:** the main session, Opus 5.5 (`claude-opus-5-5`) at xhigh effort. It plans, writes
  worker specs, checks every worker diff and triages reviews. Only the coordinator edits STATUS.md,
  DECISIONS.md, `prereg/` and `reports/`.
- **Delegation starts at M1.** The coordinator does the KICKOFF spec review and M0 itself.
- **Implementers:** supervised Orca Claude workers on Claude Sonnet 5.5:
  `orca orchestration worker-start --agent claude --model claude-sonnet-5-5 --effort high`.
- **Groundwork:** Claude Haiku 5.5 (`--model claude-haiku-5-5 --effort medium`), only for rudimentary
  mechanical work: scaffolding, boilerplate, renames, lint fixes. Never logic that SPEC.md defines.
- **Reviewer:** a supervised Orca Codex worker on GPT-6.1 Sol (`--agent codex --model gpt-6.1-sol
  --effort high`), through the orca-review and adversarial-review skills.
- **Check each launch:** report the model and effort shown in the receipt's `launch.effective`, not
  the requested values.
- **Never delegated:** gate thresholds and `prereg/`, the generator freeze and its hash, the
  ground-truth boundary test, the M6 scorer, anything that opens `runs/confirm-*`, and package
  installs or model downloads (the owner's go has to come in the coordinator's turn).
- **Each worker task** names the files it may edit, the SPEC sections it implements and its
  acceptance checks. Workers edit only those files, don't commit, don't edit coordinator-owned files
  and don't start workers of their own.
- **Acceptance:** a worker's report isn't evidence. The coordinator reviews the diff and runs
  `make test`, `make lint` and `make typecheck` itself before accepting it.
- **Review points (three),** each one cross-vendor round plus one check of the fixes:
  1. the M1 freeze: the generator, the health checks and the ground-truth boundary test;
  2. M4, before scoring: the G2 scorer and `prereg/M4.md`;
  3. the M6 pre-registration: `prereg/M6.md`.
- **If Orca is unavailable,** stop and report it. Don't substitute a hidden subagent.

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
- **Cross-vendor review** happens only at the three review points in "Delegation and reviews": one
  round each, plus one check of the fixes. Everything else gets a self-review of the diff, plus tests.
- **Pivots:** changing direction needs a short pivot note in DECISIONS.md first, saying why the old
  question is answered or can't be answered, and why the new one serves the goal better.
- **Tests:** every analysis function that produces a reported number has a known-answer test.
- **No new agent tooling** (orchestration, memory, hooks) during the project. Orca, used as
  "Delegation and reviews" describes, is existing tooling.
- **Delegation** to Orca workers: as "Delegation and reviews" describes; anything beyond that only on
  request.
- **Data stays under `runs/`**, which is gitignored, so no index or commit picks it up.

## Commands (create them in M0)

- `make test` runs pytest, `make lint` runs ruff, and `make typecheck` runs mypy, over `testbed/` and
  `tests/`.
- The vendored test suite isn't part of `make test`. M0 records its baseline once.

## Verification

- Quote the actual summary line and exit code of every check you report.
- Report failures and skips as they are.
- Update STATUS.md before stopping.

<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->
