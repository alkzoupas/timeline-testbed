# Kickoff prompt: M8, a Claude Haiku pilot (written 2026-10-22)

Paste into a fresh chat in `~/code/timeline-testbed`:
"Read `context/m8-haiku-kickoff.md` and pick up from there."

---

You are the coordinator for timeline-testbed (AGENTS.md "Delegation and reviews"). Read AGENTS.md,
STATUS.md, `reports/M4-probe.md`, `reports/RESULT.md` and `context/lessons.md` (especially lessons
7, 8 and 12) before acting. This file is a plan, not a state file: STATUS.md stays the record.

## Where things stand

- **G2 failed** (2026-10-22, scored once). At *s* = 0.75 the held-out R2 probe scores 9B 0.453 and
  4B 0.486, against F64 logistic 0.462 and trees 0.517. The *s* = 0 control is at chance, so the
  LLM line stopped. M7 is drafted in `reports/RESULT.md` and `reports/RESULT-auc-vs-s.svg`.
- Reported only: in-family R2 scores 0.725 and 0.706, seeing pairs 3 and 6 but not 2 and 4. The
  probe sees mechanisms it was trained on and doesn't transfer to unseen ones.
- If `git status` still shows the M4/M7 files uncommitted, remind the owner to commit them first.

## The owner's decisions so far (2026-10-22)

- Next question: can **Claude Haiku 5.5** (`claude-haiku-5-5`), scored through the owner's
  **Claude subscription via headless Claude Code (no API, no API key)**, rank timeline-only abuse
  above its legitimate twins on families it wasn't shown?
- Variants: **zero-shot**, and **held-out few-shot** (labelled examples from the other side's
  families only). In-family few-shot is not run.
- **Pilot first**; the full run is decided afterwards.
- **A separate Claude Code config directory**, `~/.claude-haiku-scorer`, holding only the owner's
  login.
- **Not yet given:** the go for step 1 below (the pivot note and the AGENTS.md exception). Ask for it.

## Facts checked (Claude Code 2.1.308)

- These flags exist: `--safe-mode` (no CLAUDE.md, hooks, plugins, MCP servers or skills; auth still
  works), `--tools ""` (no tools), `--strict-mcp-config`, `--no-session-persistence` (only with
  `-p`), `--system-prompt`, `--json-schema`, `--output-format json`, `--model`, `--effort`.
- `--bare` would isolate more, but it only accepts `ANTHROPIC_API_KEY`. It never reads
  subscription OAuth, so it's out.
- The binary honours `CLAUDE_CONFIG_DIR`. `claude auth login` and `claude auth status` exist.
- Unknown: whether a login under `CLAUDE_CONFIG_DIR` shares or overwrites the main macOS keychain
  entry. Check `claude auth status` (main) after the owner logs in.
- Unknown: whether Haiku runs with extended thinking by default, and what it costs. Claude Code
  can't set the temperature, so scores are random draws and need a repeatability check.
- Never read, list or print anything inside `~/.claude-haiku-scorer`; it holds credentials. The
  owner sets up subscription terms and usage limits; don't assert them.

## The isolated route (one call per unit)

```
cd <empty temp dir outside the repo>
CLAUDE_CONFIG_DIR=~/.claude-haiku-scorer claude -p --model claude-haiku-5-5 \
  --safe-mode --tools "" --strict-mcp-config --no-session-persistence \
  --system-prompt "<frozen m8-v1 system text>" \
  --json-schema '{"type":"object","properties":{"probability":{"type":"integer","minimum":0,"maximum":100}},"required":["probability"]}' \
  --output-format json "<user prompt: optional examples + the unit's render>"
```

Log the raw JSON of every call: the answer, the model the response reports, usage and duration.

## The experiment

- **Prompt `m8-v1`:** `m4-v1`'s neutral setting and question (`testbed/models/prompt.py`, no family
  or mechanism named), with only the answer format changed to a 0–100 probability that the
  timelines come from one operator. Freeze it by sha256 before the pilot.
- **Few-shot, held-out:** 6 fixed labelled examples from the tuning world
  `runs/m4-tune-s0.75-seed301`, one abusive and one legitimate unit from each training family of the
  other side, chosen by a seeded rule and in a fixed, interleaved order. Side A (tests pairs 4–6)
  sees examples from pairs 1–3; side B (tests 1–3) sees 4–6 (`r0.TEST_PAIRS`). Choosing examples
  reads labels, so that code must live in `testbed/eval/` (the two-store invariant).
- **Pilot:** world `m1-dev-s0.75-seed1`, T families only (pairs 2, 3, 4, 6). All 160 abusive units
  plus 160 legitimate units (40 per T family, seeded) gives 320 units × 2 variants = 640 calls.
  Metric: G2's primary (per-side T-family AUC, mean of A and B), with a paired group bootstrap
  against F64 logistic and trees on the same units, using the F64 scores in
  `runs/m4-g2/scores-s0.75.jsonl`.
- **AGENTS.md "Before each experiment":** (1) the pilot decides whether to run the full M8 (both
  *s* = 0.75 gate worlds plus the *s* = 0 control). Proposed go-bar: either variant's 95% lower
  bound > 0.5, fixed in writing before the pilot. (2) 160 abusive groups (≥ 20). (3) The smoke test
  below.
- **Render sizes, for budgeting:** tuning renders max 2,005, mean 1,314 and p90 1,705 tokens on the
  Qwen tokenizer; prompt overhead 165. Few-shot adds about 8–9k tokens per call.

## Order of work

1. **Pivot note and rule exception (owner's go first).** In DECISIONS.md, one line: G2 answered the
   old question; this one tests the general "one operator" sense the probe lacked, with a stronger
   model and no extra spend. In AGENTS.md "Invariants", an exception for Claude Haiku 5.5 through the
   owner's subscription only, via the isolated route above, for M8 scoring only; no API keys and no
   other hosted model. Note that a scoring step isn't "new agent tooling".
2. **Owner logs in** in a separate terminal: `CLAUDE_CONFIG_DIR=~/.claude-haiku-scorer claude auth
   login`, then `claude auth status` (main) to confirm the main login survived.
3. **Feasibility smoke, about 30 tuning-world units, by hand, before any repo code (lesson 7):**
   - isolation: ask Haiku to list every file, tool, memory or instruction it can see; diff
     `~/.claude/projects`, `~/.claude/handoffs` and the Hindsight spool before and after;
   - per call: seconds and tokens, parse rate, distinct scores, the reported model ID, whether
     thinking is on;
   - repeatability: re-score about 10 units;
   - a few-shot-sized call (examples carrying placeholder labels, for size and timing only);
   - record results in STATUS.md, and size the pilot to the measured limits.
4. **Driver and scorer**, by an Orca Sonnet 5.5 worker at high effort (report `launch.effective`):
   - `testbed/models/haiku_run.py`: resumable, raw JSON per unit, binds prompt, render and route
     hashes like `r2_embed.py`, runs units in seeded random order;
   - `testbed/eval/haiku.py`: picks examples, builds the subset and scores it, reusing r0/g2;
   - known-answer tests, with `claude` mocked in tests.
   The coordinator reviews the diff and runs `make test`, `make lint` and `make typecheck` itself.
5. **`prereg/M8.md`** (one page): the frozen prompt hash, the examples, the subset, the go-bar.
   Then the pilot, scored once, and a short note in `reports/`.

## Guardrails

- It can't rescue G2. A positive pilot is a new finding, confirmed on gate pools or fresh worlds.
- Don't open the `runs/confirm-*` worlds. Only `testbed/eval/` reads labels. No commits.
- Haiku must never see the repo, the generator or the labels. The calls run outside the repo, with
  `--tools ""` and safe mode.
