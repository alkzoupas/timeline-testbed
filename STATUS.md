# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-30)

- **M0, M1 done** and committed (`f40cdd1`; `reports/M0.md`, `reports/M1.md`). Generator v1 frozen
  (hash in DECISIONS.md); channel tags frozen as `MEASURED_TAGS` (pair 1 M, pair 5 F, 2/3/4/6 T).
- **M2 done; G1 passed** (`reports/M2-baselines.md`, `274bfcf`). Held-out primary (T families) at
  s = 0.75: detector 0.497, F12 0.488, F64 logistic 0.462, F64 trees 0.517; oracle 0.90.
- **M3 done** (`reports/M3-feasibility.md`, `eb9cb5e`). R2 forward pass 9B 0.86 h / 4B 0.42 h per
  world; R3 on the 4B (one epoch fits 12 h); score R1/R3 with a float32 head (bf16 logit steps).
- **M4 done; G2 FAILED**, scored once on 2026-10-22 at 08:26 (`runs/m4-g2/`, `reports/M4-probe.md`;
  inputs at `df1e99a`). Held-out primary at s = 0.75: R2 9B 0.453, 4B 0.486, against F64 logistic
  0.462 and trees 0.517. No (a) lower bound is above 0 (best −0.031); both models meet (b). Per SPEC
  §7.1 the LLM line stops: no M5 or M6.
  - With the owner's go: R2 at s = 0.5, scored with the frozen choices (`runs/m4-r2-s0.5/`), gives
    held-out 9B 0.442 and 4B 0.470. R1 and Nimble were not run.
  - Reported, not gating: in-family at s = 0.75, R2 scores 0.725 and 0.706 against trees 0.517;
    pairs 3 and 6 are seen, pairs 2 and 4 are not, and s = 0 is at chance. This goes into M7 as a
    bounded finding, with no pivot (owner).
- **M7 done**; M4 and M7 committed by the owner (`ccef012`). `reports/RESULT.md` (one page) and
  `reports/RESULT-auc-vs-s.svg`. Figure code
  `testbed/eval/result.py` and `tests/test_result.py`, by Orca worker `ctx_c589d43dcb59` (run
  `run_47321d740e8a`, launch.effective claude-sonnet-5-5 at high). I reviewed the diff and changed
  the docstring path; the checks below are my own runs. The worker was released.
- **M8 done; the Claude Haiku 5.5 pilot is NO-GO** (`reports/M8.md`; prereg `ef48f91`, code
  `070a94d`; `runs/m8-pilot/`). 1,920 calls on seed1's T units via your subscription, 59 min, all
  parsed first try, no errors, CLI 2.1.308 throughout. Held-out primary: zero-shot 0.561 (bound
  0.487), few-shot 0.553 (0.479); F64 trees 0.517. M8 stops; no full run. Reported, not claimed:
  pair 3 alone stands out (few-shot 0.84 [0.74, 0.92], zero-shot 0.68; trees 0.56), post hoc, on
  a gate world, with no *s* = 0 control. Retest reliability 0.49–0.59 (the smoke's 0.22 was low).
  - Driver and scorer by Orca workers `ctx_c8516f5bd037` and `ctx_b6e0640cc6dc` (run
    `run_392d9c02ec97`, launch.effective claude-sonnet-5-5, high); their terminals are yours.

## Next

- **M9: confirm pair 3** (owner's go 2026-10-30; DECISIONS pivot note; `prereg/M9.md` drafted).
  Haiku few-shot on pair 3, fresh worlds (two at *s* = 0.75, one *s* = 0 control), 1,200 calls.
  - [x] AGENTS.md exception extended to M9; pivot + design lines in DECISIONS.md.
  - [x] `testbed/eval/haiku_m9.py` + tests by Orca worker `ctx_30246f64d05e` (run
    `run_874a9ea5f56d`, launch.effective claude-sonnet-5-5, high); reviewed, accepted, released.
  - [x] 24-call end-to-end check passed (`runs/m9-check/`); **frozen** (prereg `d20f7bbd…`).
  - [x] Seeds (`secrets`): s = 0.75 `1759212321`, `1841874529`; s = 0 `98816701`. All three
    worlds pass health; `select --mode confirm` exit 0, 1,200 calls (jobs `5d7d5cfe…`).
  - [ ] Driver running (started 2026-10-30, `runs/m9-logs/driver.log`); then score once.
- The `runs/confirm-*` worlds stay unopened.

## Open owner decisions

- None now. M8 itself is closed as is (owner, `1fd10df`).

## Files changed since `1fd10df` (uncommitted)

- `reports/RESULT.md` (an M8 addendum), `STATUS.md`, `AGENTS.md` (exception to M9),
  `DECISIONS.md` (four lines), `prereg/M9.md`, `testbed/eval/haiku_m9.py`,
  `tests/test_haiku_m9.py` (new). Deleted (gitignored, owner's go 2026-10-30):
  `runs/superseded-round1/`, `runs/m4-embed-superseded/`, `runs/m4-tune-superseded/`.

## Last checks (2026-10-30)

- M9 code: `make test` `803 passed in 159.60s`, lint `All checks passed!`, typecheck `Success: no
  issues found in 73 source files`, format `73 files already formatted`; exit 0 each.
- M9 check: driver exit 0 (`done ok 24, done unparsed 0, remaining 0, errors this run 0`); score
  exit 0 (`M9 CHECK (not a result)`, `8154f5b1…`), matched by hand.

## Earlier checks (2026-10-22; details in `reports/M8.md`, `reports/M4-probe.md`)

- M8 pilot: driver exit 0 (`done ok 1920`, no errors); `score` exit 0, `M8 pilot NO-GO`
  (`d9888902…`). G2: exit 0, `G2 FAIL` (`g2.json` `b23f922c…`).
