# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-22)

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
- Worlds from before the M1 manifest fix sit in `runs/superseded-round1/`; safe to delete. The
  pre-fix embeddings are kept in `runs/m4-embed-superseded/`.

## Next

- **M8, a Claude Haiku 5.5 pilot** (owner, 2026-10-22), via the owner's subscription; zero-shot plus
  held-out few-shot, pilot first. The plan is in `context/m8-haiku-kickoff.md`; results go to
  `reports/M8.md`.
  - Steps 1–3 done (smoke, 50 calls, `runs/m8-smoke/`; retest Pearson 0.22, so *k* = 3 draws).
    Owner's go covers steps 1–5, through the 1,920-call pilot (`context/m8-haiku-kickoff.md`).
  - Step 1 done: prompt `m8-v1` frozen in `testbed/models/haiku_prompt.py` (sha256 `028db033…`,
    system `0f6af5ef…` = the smoke's), with the route constants; committed by the owner (`0b84bd1`).
  - Step 2: Orca run `run_392d9c02ec97`. Worker `ctx_c8516f5bd037` (claude-sonnet-5-5, high) wrote
    `testbed/eval/haiku.py`, `testbed/models/haiku_run.py` and their tests. My review asked for
    fixes F1–F5 (one pinned CLI binary, `api_error_status`, an unparsed-streak stop, route check in
    the scorer, token sums); fix worker `ctx_b6e0640cc6dc` (launch.effective claude-sonnet-5-5,
    high) did them. I reviewed them, rewrapped one `haiku.py` docstring paragraph and ran the
    checks below: **accepted**, ready to commit. Both worker terminals are yours (`user_takeover`).
  - Then, after you commit step 2: step 3, the 24-call end-to-end check; step 4, `prereg/M8.md`
    (draft ready; go-bar at the 1.25th percentile, Bonferroni over the two variants as G2);
    step 5, the pilot.
- The `runs/confirm-*` worlds stay unopened.

## Open owner decisions

- M8: none open.
- FYI: a Claude Code background session (pid 78316, daemon-forked, started 07:28, idle since 08:24)
  has this repo as its cwd. It's not the scorer; left running. The "Sonnet session" I flagged
  earlier was the M7 Orca worker's ended transcript; there was no process to kill.
- Optional: delete `runs/superseded-round1/` and `runs/m4-*-superseded/` (gitignored data; your go
  needed).

## Files changed since `0b84bd1` (uncommitted)

- `STATUS.md`. Step 2, accepted: `testbed/eval/haiku.py`, `testbed/models/haiku_run.py`,
  `tests/test_haiku_eval.py`, `tests/test_haiku_run.py`.

## Last checks (2026-10-22)

- Step 2: `make test` `763 passed in 170.51s (0:02:50)`; after the rewrap, haiku + boundary tests
  `184 passed in 14.62s`, lint `All checks passed!`, typecheck `Success: no issues found in 71
  source files`, `ruff format --check` `71 files already formatted`; exit 0 each.
- M8 smoke: `smoke.py run` exit 0 (iso 2 + iso2 1 + 47 calls, every rc 0); `summarize.py` exit 0.
- Figure: `python -m testbed.eval.result --r0 runs/m2-r0/r0.json --r2 runs/m4-g2/report.json
  runs/m4-r2-s0.5/r2.json --out reports/RESULT-auc-vs-s.svg`, exit 0. Baselines matched R0 at every
  R2 s, and the printed points match RESULT.md's table.
- R2 s = 0.5: `r2 score`, exit 0 in 11.2 s; `r2.json` sha256 `00ac6cb8…`. Embeddings are bound to
  child `ae565057…` and prompt `0cfe33a7…`.
- G2, 08:26: `python -m testbed.eval.g2 … --out runs/m4-g2`, exit 0 in 23 s: `G2 FAIL:
  a_beats_f64_4b=no b_null_s0_4b=ok a_beats_f64_9b=no b_null_s0_9b=ok`; `g2.json` `b23f922c…`.
