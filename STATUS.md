# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-21)

- **M0, M1 done** and committed (`f40cdd1`; `reports/M0.md`, `reports/M1.md`). Generator v1 frozen
  (hash in DECISIONS.md); channel tags frozen as `MEASURED_TAGS` (pair 1 M, pair 5 F, 2/3/4/6 T).
- **M2 done; G1 passed** (`reports/M2-baselines.md`; `prereg/M2.md` written before any R0 score).
  Not yet committed.
  - Ten sweep worlds (`runs/m1-dev-*`, `runs/m2-dev-*`; s ∈ {0, .25, .5, .75, 1} × seeds 1–2) pass
    health; channel re-check `runs/m2-channel/`; R0 `runs/m2-r0/`; gate `runs/m2-g1/g1.json`.
  - Held-out primary (T families) at s = 0.75: detector 0.497, F12 0.488, F64 logistic 0.462,
    F64 trees 0.517; oracle 0.90. No R0 rung sees T at any s; F64 sees pairs 1 and 5.
  - Same-seed worlds share base draws across s (paired sweep); see the report's caveat.
  - Orca run `run_c5fdcd8d67af`: worker task `task_baa19b981487` / dispatch `ctx_3163e6755741`
    (Sonnet 5.5 at high, effective) succeeded and was accepted; its terminal is `retained`
    (`user_takeover`), with 0 reclaimable.
- **Next: M3** (feasibility: render lengths, forward-pass and LoRA throughput; pick the probe and R3
  models; any download needs the owner's go). M4 also needs the R2 tuning world (s = 0.75, own seed).
- Worlds from before the M1 manifest fix sit in `runs/superseded-round1/`; safe to delete.

## Open owner decisions

- None.
- FYI: an untracked `opencode.json` (graft's MCP config) is in the repo root; not ours, left in place.

## Files changed in M2 (uncommitted)

- New: `prereg/M2.md`, `reports/M2-baselines.md`, `reports/M2-auc-vs-s.svg`,
  `testbed/eval/{r0,figure,g1}.py`, `tests/test_{r0,figure,g1}.py`.
- Edited: `testbed/eval/cv.py` (`fit_predict`), `testbed/eval/metrics.py` (recall at FPR, weighted
  bootstrap AUC, intervals), `tests/test_metrics.py`, `STATUS.md`, `DECISIONS.md`.

## Last checks (2026-10-21, M2 final)

- `make test`: `383 passed in 131.74s (0:02:11)`, exit 0. `make lint`: `All checks passed!`, exit 0.
  `make typecheck`: `Success: no issues found in 45 source files`, exit 0.
  `ruff format --check testbed tests`: `45 files already formatted`, exit 0.
- Five new `run_world` + `health world` runs exit 0 each; `health channel` exit 0; `r0` on the sweep
  exit 0 (58.1 s); `g1` exit 0, `G1 passed: health=ok oracle_positive_s1=ok r0_null_s0=ok
  channel_tags=ok`.
