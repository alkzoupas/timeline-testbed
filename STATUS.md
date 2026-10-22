# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-21)

- **M0, M1 done** and committed (`f40cdd1`; `reports/M0.md`, `reports/M1.md`). Generator v1 frozen
  (hash in DECISIONS.md); channel tags frozen as `MEASURED_TAGS` (pair 1 M, pair 5 F, 2/3/4/6 T).
- **M2 done; G1 passed** (`reports/M2-baselines.md`; `prereg/M2.md` written before any R0 score).
  Committed (`274bfcf`).
  - Ten sweep worlds (`runs/m1-dev-*`, `runs/m2-dev-*`; s ∈ {0, .25, .5, .75, 1} × seeds 1–2) pass
    health; channel re-check `runs/m2-channel/`; R0 `runs/m2-r0/`; gate `runs/m2-g1/g1.json`.
  - Held-out primary (T families) at s = 0.75: detector 0.497, F12 0.488, F64 logistic 0.462,
    F64 trees 0.517; oracle 0.90. No R0 rung sees T at any s; F64 sees pairs 1 and 5.
  - Same-seed worlds share base draws across s (paired sweep); see the report's caveat.
  - Orca run `run_c5fdcd8d67af`: worker task `task_baa19b981487` / dispatch `ctx_3163e6755741`
    (Sonnet 5.5 at high, effective) succeeded and was accepted; its terminal is `retained`
    (`user_takeover`), with 0 reclaimable.
- **M3 done** (`reports/M3-feasibility.md`; outputs `runs/m3-feasibility/{9b,4b}/`).
  - Lengths: render plus draft prompt mean 1,006, max 1,761; none over 2,048; the 4B and 9B
    tokenizers agree on every unit.
  - R2 forward pass: 9B 390 tok/s (0.86 h per world); 4B 806 tok/s (0.42 h per world).
  - LoRA, batch 1: 9B 37.8 / 18.0 tok/s (median / longest), 4B 43.1 / 29.6; epoch per fold
    8.9 h vs 7.8 h. **R3 on the 4B**; one epoch fits the 12 h cap, two don't.
  - Scorer gap explained: bf16 logits (steps of 0.125); M4/M5 score with a float32 head.
  - Orca run `run_0dee52b4bf59`: worker `task_2b26b2ef7176` / `ctx_1580512f507d` (Sonnet 5.5 at
    high, effective) built `testbed/models/{mlx_child,local_llm,feasibility}.py`; accepted;
    terminal `retained` (`user_takeover`), 0 reclaimable.
- **Next: M4** (R1 and R2; gate G2). First: generate the R2 tuning world (*s* = 0.75, own seed),
  freeze the prompt and the float32 scoring head, write `prereg/M4.md`; cross-vendor review of
  the G2 scorer and `prereg/M4.md` before scoring.
- Worlds from before the M1 manifest fix sit in `runs/superseded-round1/`; safe to delete.

## Open owner decisions

- None.
- FYI: an untracked `opencode.json` (graft's MCP config) is in the repo root; not ours, left in place.

## Files changed in M3 (uncommitted)

- New: `testbed/models/{mlx_child,local_llm,feasibility}.py`, `tests/test_{local_llm,feasibility}.py`,
  `reports/M3-feasibility.md`.
- Edited: `pyproject.toml` (mypy ignores mlx imports), `STATUS.md`, `DECISIONS.md`.

## Last checks (2026-10-21, M3 worker acceptance)

- `make test`: `420 passed in 140.36s (0:02:20)`, exit 0 (the tiny-model MLX test ran, no skips).
  `make lint`: `All checks passed!`, exit 0. `make typecheck`: `Success: no issues found in 50
  source files`, exit 0. `ruff format --check testbed tests`: `50 files already formatted`, exit 0.
