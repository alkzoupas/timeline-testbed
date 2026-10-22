# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-21)

- **M0 done** and committed (`reports/M0.md`).
- **M1 done; generator v1 frozen** (`reports/M1.md`; hash in DECISIONS.md). Not yet committed.
  - Every §5.6 row passes on the final generator: standard s = 0 seeds 1–2, s = 0.75 seed 1, s = 1
    seeds 1–2; generator-only sample (seed 201); channel world (seed 101). Positive control
    0.915–1.0; max render 1,699 tokens; linkage 1,200/1,200, 0 mixed; parity 0/120.
  - Channel tags (frozen, `MEASURED_TAGS`): pair 1 M, pair 5 F, pairs 2, 3, 4, 6 T. F64-w90 sees
    pairs 1, 5 and partly 3 (trees 0.83); pairs 2, 4, 6 stay ≈ 0.5.
  - Review round 1 (`reviews/round-1/`): five findings fixed, one cross-vendor check of the fixes,
    two follow-up boundary-scan gaps fixed by the coordinator (`TRIAGE.md`).
  - Worlds from before the manifest fix sit in `runs/superseded-round1/` (identical world bytes);
    safe to delete when convenient.
- **G1 accounting done (owner decision, DECISIONS):** `world_health` tests each world's null-rule
  rows as one Bonferroni family (T = 163 at s = 0, 157 otherwise); the generator-only sample stays
  per-row and decisive. The five M1 worlds pass under it; a fair world fails 2/100 (nuisance plus
  negative control pooled). Implemented by the coordinator (owner's go; the Orca handle was stale).
- **Next: M2** (R0 across the sweep; gate G1). Orca note: this session's handle is stale
  (`no_active_sender_terminal`); resume the coordinator inside a live Orca terminal before
  delegating again.

## Open owner decisions

- None.
- FYI: an untracked `opencode.json` (graft's MCP config) is in the repo root; not ours, left in place.

## Files changed in M1 (uncommitted)

- New: `testbed/world/{DESIGN.md,families.py,generator.py}`, `testbed/pipeline/{timeline,render,
  tokens,parity,run_world}.py`, `testbed/features/{f12,f64}.py`, `testbed/eval/{labels,metrics,cv,
  oracle,health}.py`, their tests, `tests/test_ground_truth_boundary.py`, `reviews/round-1/`,
  `reports/M1.md`.
- Edited: `testbed/eval/stores.py` (guarded `verify_pair`), `DECISIONS.md`, `STATUS.md`.

## Last checks (2026-10-21, after the world-family change)

- `make test`: `338 passed in 118.64s (0:01:58)`, exit 0. `make lint`: `All checks passed!`, exit 0.
  `make typecheck`: `Success: no issues found in 39 source files`, exit 0.
  `ruff format --check testbed tests`: `39 files already formatted`, exit 0.
- Regeneration: every `run_world`, `health channel` and `health gensample` exit 0; `health world`
  re-run on the five worlds under the world family, exit 0 each.
