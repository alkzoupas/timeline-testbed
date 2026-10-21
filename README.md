# timeline-testbed: handoff package

Created on 2026-10-21 from the EntitlementShield postmortem (option A): an owned, mechanistic testbed
and a model ladder, in a fresh project.

## What's in it

| Path | What |
|---|---|
| `SPEC.md` | The spec the implementer builds from |
| `AGENTS.md` | Project rules, kept lean. Check "Standing permissions" before you start |
| `KICKOFF.md` | The first prompt for a new Claude Code (Opus) session |
| `STATUS.md`, `DECISIONS.md` | The state file and the decision log, seeded |
| `context/` | Lessons (one page), prior results and environment, and a map of the reusable vendored code |
| `vendor/entitlement-shield/` | The old repo's code at `fd5deca` (backend, config, scripts), read-only. `VENDORED.md` there lists what's included, and `VENDORED.sha256` has the file hashes |

There's no world data and there are no labels. Tier 4 and its sealed split stay in
`~/code/entitlement-shield`.

## Start

1. Unzip: `unzip ~/code/timeline-testbed-handoff.zip -d ~/code` creates `~/code/timeline-testbed`.
2. Read "Standing permissions" in AGENTS.md and edit anything you don't want.
3. Run `cd ~/code/timeline-testbed && git init`, then make the first commit yourself.
4. Open Claude Code in that folder and paste the quoted prompt from KICKOFF.md.

The first session reviews the spec and asks you questions before writing any code.
