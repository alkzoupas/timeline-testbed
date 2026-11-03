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

## M9 (done)

- **M9: confirm pair 3** (owner's go 2026-10-30; DECISIONS pivot note; `prereg/M9.md` drafted).
  Haiku few-shot on pair 3, fresh worlds (two at *s* = 0.75, one *s* = 0 control), 1,200 calls.
  - [x] AGENTS.md exception extended to M9; pivot + design lines in DECISIONS.md.
  - [x] `testbed/eval/haiku_m9.py` + tests by Orca worker `ctx_30246f64d05e` (run
    `run_874a9ea5f56d`, launch.effective claude-sonnet-5-5, high); reviewed, accepted, released.
  - [x] 24-call end-to-end check passed (`runs/m9-check/`); **frozen** (prereg `d20f7bbd…`).
  - [x] Seeds (`secrets`): s = 0.75 `1759212321`, `1841874529`; s = 0 `98816701`. All three
    worlds pass health; `select --mode confirm` exit 0, 1,200 calls (jobs `5d7d5cfe…`).
  - [x] Driver: 759 calls, then a 429 at 12:29 PDT (exit 3); resumed 14:19 on the owner's go, finished
    `done ok 1200, done unparsed 0, remaining 0, errors this run 0`. CLI 2.1.314 throughout.
  - [x] **Scored once: M9 CONFIRMED** (`runs/m9-confirm/scored/`, `report.json` `f533434d…`).
    Few-shot 0.758 [0.679, 0.832]; minus F64 logistic lower 0.129, minus trees lower 0.136 (F64
    0.511 / 0.502); s = 0 control 0.571 [0.445, 0.697]. 80 abusive + 80 legitimate groups.
    Zero-shot 0.577 (not gating). AUCs recomputed by hand from `scores.jsonl`: match.
  - [x] `reports/M9.md`; RESULT.md headline + M9 addendum (as the prereg specified); DECISIONS line.

## Closed and published (2026-11-02)

- **The project is closed and published** (owner) as the public GitHub repo
  `alkzoupas/timeline-testbed`; `runs/confirm-*` never opened. README.md rewritten for outside
  readers, LICENSE (MIT, as EntitlementShield), `.claude/` gitignored, the 429 quote in
  `reports/M9.md` shortened. No result, prereg, frozen code or `runs/` changed.
  `reviews/` keeps its machine paths (recorded sha256s); KICKOFF.md kept. Open owner decisions: none.

## Last checks (2026-11-02)

- `make test` `803 passed in 179.91s`; `make lint` `All checks passed!`; `make typecheck` `Success:
  no issues found in 73 source files`; `ruff format --check testbed tests` `73 files already
  formatted` (no `make format` target); `make vendor-check`; exit 0 each.
- README commands rerun in a temp dir: `run_world` seed 301 byte-identical to
  `runs/m4-tune-s0.75-seed301`; `health world` exit 0; `r0` exit 0; `haiku_m9 select --mode check`
  exit 0, `jobs.jsonl` identical to `runs/m9-check/`. No Haiku calls were made.

## Earlier checks (2026-10-22 and 10-30; details in `reports/M8.md`, `M9.md`, `M4-probe.md`)

- M9 (10-30): `make test` `803 passed in 159.60s`; check driver exit 0 (`done ok 24`); check score
  exit 0 (`M9 CHECK (not a result)`, `8154f5b1…`).
- M8 pilot: driver exit 0 (`done ok 1920`, no errors); `score` exit 0, `M8 pilot NO-GO`
  (`d9888902…`). G2: exit 0, `G2 FAIL` (`g2.json` `b23f922c…`).
