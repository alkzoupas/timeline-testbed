# Kickoff prompt: M8 from step 4, the Claude Haiku pilot (updated 2026-10-22)

Paste into a fresh chat in `~/code/timeline-testbed`:
"Read `context/m8-haiku-kickoff.md` and pick up from there."

---

You are the coordinator for timeline-testbed (AGENTS.md "Delegation and reviews"). Before acting,
read:
- AGENTS.md and STATUS.md;
- `reports/M8.md` (the smoke) and `reports/M4-probe.md`;
- `context/lessons.md`, especially lessons 7, 11 and 12.

This file is a plan, not a state file: STATUS.md is the record. If `git status` still shows the
2026-10-22 M8 edits uncommitted (AGENTS.md, DECISIONS.md, STATUS.md, `reports/M8.md` and this file),
remind the owner to commit them first.

## Where things stand

- **G2 failed** (2026-10-22). The R2 probe sees the mechanisms it was trained on, but it doesn't
  transfer to held-out families (`reports/RESULT.md`).
- **M8's question** (the DECISIONS pivot): can Claude Haiku 5.5, zero-shot and with held-out
  few-shot, rank timeline-only abuse above its legitimate twins on families it wasn't shown? Haiku
  is a hosted reference point, and the pilot can't rescue G2.
- **Steps 1–3 are done.** That covers the pivot and the AGENTS.md exception, the owner's login to
  `~/.claude-haiku-scorer` (Pro plan), and a 50-call smoke. Its write-up is in `reports/M8.md`, and
  its logs and scratch scripts are in `runs/m8-smoke/`.
- **The smoke's results.** Isolation holds, and every call succeeded and parsed. One call takes
  about 11 s; five at once give about 5 s per call. Thinking is on by default. **Repeatability is
  poor** (retest Pearson 0.22 on 10 units), so each unit is scored *k* times.

## The owner's decisions (2026-10-22)

- Two variants: zero-shot, and held-out few-shot. No in-family few-shot.
- **k = 3 draws** per unit and variant. A unit's score is the mean of its draws.
- Pilot first; the full run is decided after it.
- Claude Code's environment block and account note are accepted as constant context. The calls run
  from a neutral temp directory.
- **The go covers steps 1–5 below**, including the 1,920-call pilot once `prereg/M8.md` is written.
  Pasting this prompt starts them. Stop and ask the owner if:
  - the end-to-end check (step 3) fails;
  - usage limits stop the run (report how far it got);
  - anything would change the prereg after it's written.

## The route (validated by the smoke; reference: `runs/m8-smoke/smoke.py`)

- **Command.** `claude -p --model claude-haiku-5-5 --safe-mode --tools "" --strict-mcp-config
  --no-session-persistence --system-prompt <SYSTEM> --output-format json --json-schema <SCHEMA>`.
  The user message goes on stdin. Don't pass `--effort` (default; thinking on).
- **Environment.** Use only an explicit allowlist (HOME, PATH, USER, LOGNAME, SHELL, LANG, TMPDIR,
  TERM) plus `CLAUDE_CONFIG_DIR=~/.claude-haiku-scorer`. Never inherit the full environment: an API
  key would bypass the subscription. The secret guard blocks `env` dumps; don't route around it.
- **cwd.** One empty `tempfile.mkdtemp()` directory under the system temp dir, so its path names
  nothing. Check it's empty before the run.
- **Preflight.** `claude auth status`, under that environment, must report `loggedIn`,
  `authMethod: claude.ai`, and the scorer directory as `configDirectory`. Record `claude --version`.
  Every pilot call must run on one version: refuse a change mid-run.
- **Output.** The answer is `structured_output.probability` (an integer, 0–100) in the stdout JSON.
  Scored calls take 2 turns. Log the whole stdout JSON plus wall time, return code and the end of
  stderr.
- **Never read, list or print anything inside `~/.claude-haiku-scorer`.**

## Prompt `m8-v1` (freeze it before the pilot)

- It's the smoke's draft, verbatim from `runs/m8-smoke/smoke.py`: `SYSTEM`, `UNIT_INTRO`,
  `VERDICTS`, `SCHEMA` and `few_shot_user`. The system text's sha256 is `0f6af5ef…`.
- It reuses `m4-v1`'s `SETTING` and `QUESTION` from `testbed/models/prompt.py`. It adds one bridge
  sentence and changes the answer format to a 0–100 probability.
- The coordinator writes it into `testbed/models/haiku_prompt.py`, with `PROMPT_VERSION = "m8-v1"`
  and a pinned sha256 over all of its parts. A test recomputes the hash. Workers don't edit this
  file.

## The experiment

- **Pilot world: `runs/m1-dev-s0.75-seed1`, T families only (pairs 2, 3, 4, 6).** All 160 abusive
  units, plus 160 legitimate units (40 per T pair, seeded). 320 units × 2 variants × 3 draws gives
  1,920 calls, about 3 h at five at once.
- **Few-shot, held-out.** Six fixed examples from the tuning world `runs/m4-tune-s0.75-seed301`:
  one abusive and one legitimate unit from each pair on the other side, chosen by a seeded rule and
  in a fixed, interleaved order. Side A (tests pairs 4–6) sees pairs 1–3; side B (tests pairs 1–3)
  sees pairs 4–6 (`r0.TEST_PAIRS`).
- **Labels stay in `testbed/eval/`.** Choosing examples and the subset reads labels, so that code
  lives there. It gives the driver a job file of call keys and user messages with no label fields.
- **Call order.** All 1,920 calls run in one seeded random order, with units, variants and draws
  interleaved, so drift over time can't line up with class.
- **Metric: G2's primary on the pilot units.** That's the per-side T-family AUC, mean of sides A
  and B, computed on the mean of the 3 draws. Compare against F64 logistic and trees on the same
  units with a paired group bootstrap, as G2 does (`testbed/eval/g2.py`, `r0.py`). The F64 scores
  are in `runs/m4-g2/scores-s0.75.jsonl`, rows with `world` = seed1.
- **Go-bar, fixed in `prereg/M8.md` before the run:** either variant's lower bound > 0.5.
  - Decide there between the 2.5th percentile and the 1.25th (Bonferroni over the two variants, as
    G2 did). The power table in `reports/M8.md` assumed the 2.5th.
  - Decide there how unparsed draws count. Proposed: retry a draw twice; score a unit as the mean of
    its parsed draws, and as 50 if none parsed.
- **Reported, not gating:** draw-1 AUC alone, retest reliability over the 3 draws, per-pair AUCs,
  the parse rate, and the Haiku − F64 differences.
- **AGENTS.md "Before each experiment"**, in the prereg:
  1. the pilot decides whether the full M8 runs;
  2. 160 abusive groups (at least 20);
  3. the smoke plus the end-to-end check.
- **Confirmation.** A positive pilot is confirmed on `m2-dev-s0.75-seed2` plus fresh seeds, not on
  seed1, which is one of G2's two s = 0.75 gate worlds. The s = 0 control (`m1-dev-s0-seed1`/`2`)
  comes with the full run.

## Order of work

1. **Prompt freeze** (coordinator): `testbed/models/haiku_prompt.py` and its hash test. Then
   `make test`, `make lint` and `make typecheck`.
2. **Driver and scorer**, by an Orca worker. Launch it with `orca orchestration worker-start --agent
   claude --model claude-sonnet-5-5 --effort high` and report the receipt's `launch.effective`.
   - Allowed files: `testbed/models/haiku_run.py`, `testbed/eval/haiku.py`,
     `tests/test_haiku_run.py` and `tests/test_haiku_eval.py`.
   - `haiku.py select` picks the examples and the subset (seeded) and writes the job file, with its
     sha256, to `runs/m8-pilot/`.
   - `haiku.py score` reads the raw outputs, the labels and the F64 scores, and writes
     `report.json`. It reuses r0 and g2.
   - `haiku_run.py` uses the route above and is resumable: one raw JSON per call key (unit, variant,
     draw). Like `r2_embed.py`, it binds the prompt hash, the job-file sha256, the route flags and
     the CLI version. It runs five calls at once and stops cleanly on a limit or API error; a rerun
     resumes.
   - Known-answer tests, with `claude` mocked. The ground-truth boundary test must still pass.
   - Acceptance: the coordinator reviews the diff and runs `make test`, `make lint` and
     `make typecheck` itself.
3. **End-to-end check** (lesson 7): the real driver on the tuning world, 4 units × 2 variants ×
   3 draws (24 calls), then `score` on it as a dry run.
4. **`prereg/M8.md`** (coordinator, one page): the prompt hash, the example unit ids, the job-file
   sha256, *k* = 3, the go-bar, the bootstrap (resamples and seed), the unparsed rule, the three
   "Before each experiment" items, and the confirmation rule. M8 isn't one of the review points, so
   there's no cross-vendor review unless the owner asks.
5. **The pilot**: 1,920 calls into `runs/m8-pilot/`, then `score`, once. Update `reports/M8.md`
   (one page) and STATUS.md, and add a DECISIONS line with the verdict.

## Guardrails

- The pilot can't rescue G2. A positive result is a new finding, confirmed on worlds the pilot
  didn't touch.
- Don't open the `runs/confirm-*` worlds. Only `testbed/eval/` reads labels. No commits.
- Haiku must never see the repo, the generator or the labels: the calls run outside the repo, with
  `--tools ""` and safe mode.
