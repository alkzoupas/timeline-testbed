# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-21)

- SPEC v0.2 adopts the M0 spec review (`reports/M0-spec-review.md`). The owner's decisions are in
  DECISIONS.md, and AGENTS.md's standing permissions are confirmed.
- **M0 is done: every exit criterion is met** (`reports/M0.md`). `.venv` runs Python 3.14.5, with
  pins in `requirements.txt` and the vendored code (`fd5deca`) on the path through a `.pth`.
  `testbed/` holds `world/persist.py`, `pipeline/{detect,units,timing,render_vendored}.py` and
  `eval/stores.py`. `tests/` has 4 files.
- Detector runtime: 0.0042 s per cluster-cycle on `dev` (3.6 s detect, 5.5 s pipeline). A full
  SPEC §5.7 world extrapolates to about 20 s, inside the 30-minute budget.
- Vendored suite baseline: `2 failed, 676 passed, 36 skipped`. The 2 failures are environmental
  (no git checkout; `docs/decisions.md` not vendored), and the 36 skips need `data/tier2`.

## Next

- The owner commits M0. Then M1 (the generator; delegation starts here, per AGENTS.md). The M1
  freeze review also covers `pipeline/detect.py` and `pipeline/units.py` (DECISIONS, 2026-10-21).

## Open owner decisions

- None. (The HF-cache `tokenizer.json` is now allowlisted in the secret guard; see DECISIONS.)

## Last checks (2026-10-21)

- `make test`: `11 passed in 0.90s`, exit 0 (includes the `min` end-to-end smoke test, the
  manifest check and memoised-vs-plain detection equality).
- `make lint`: `All checks passed!`, exit 0. `make typecheck`: `Success: no issues found in 16
  source files`, exit 0.
- `make timing`: exit 0 (`runs/m0-timing/{min,dev}-seed1/timing.json`).
- `make vendored-tests`: `2 failed, 676 passed, 36 skipped in 184.65s (0:03:04)`, exit 2 (this is
  the baseline).
- `make vendor-check`: exit 0, no mismatches.
