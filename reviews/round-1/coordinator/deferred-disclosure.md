# Round 1 deferred material (coordinator disclosure, 2026-10-21)

Coordinator-owned. Disclosed only after the reviewer's initial report was preserved.

- Initial-report snapshot: `reviews/round-1/coordinator/initial-snapshot-codex-gpt-6.1-sol-adversarial.md`
  (read-only), sha256 `21237146a0a852a87638c5143939060c31a39b33931c5664f1024146dc1ee03e`. Equal to
  `reviews/round-1/codex-gpt-6.1-sol-adversarial.md` at the time of the snapshot.
- Candidate as checked by the coordinator at disclosure: HEAD `6e14f7d5fc297b9e0eb30fb3c06759426b037284`;
  manifest block sha256 `2dd6ad936216deb8abbb5b139c4b085ef55e29ca5b2ec48c027b1f74a6ff9037`, 251 entries,
  every entry matches in mode and hash except `STATUS.md` (coordinator-owned, edited after pinning; it
  is not part of the reviewed code). No file outside the manifest, `graft/` and `reviews/`.
- This file is the complete deferred material: (1) the coordinator's M1 decisions as recorded in
  STATUS.md, (2) DESIGN.md's M1 notes, (3) the implementers' completion summaries. They are author
  claims, not evidence.

## 1. STATUS.md, "Decisions made by the coordinator in M1 (to DECISIONS.md at the freeze)" (verbatim)

- Null rule: two-sided test of AUC = 0.5 at alpha/tests, exact null-variance screen then Monte Carlo
  permutation (DeLong failed fair generators in 11–17% of seeds; this rule in ~4–6%).
- Group sizes balanced per family (`k = 2 + i mod 5`): a chance size imbalance failed seed 1.
- Channel tags are the measured ones (SPEC §5.4): pair 1 M, pair 5 F.
- Render `raw-v1`: day header lines and letter aliases (≈ 10–11 tokens per event); budget ≤ 1,792
  tokens, keeping 256 of the 2,048 for the prompt.
- F-* parity: rebuild a store from the parsed render and re-score with the vendored detector.

## 2. DESIGN.md, the M1 notes (verbatim, candidate line numbers)

These passages are in the candidate you already reviewed. They are listed here as the author's
rationale for the M1 changes.

```text
  15  - Group size *k* is uniform on {2..6} for every family, by a balanced design: group *i* of a
  16    family has `k = 2 + i mod 5`. So every reported world holds exactly as many groups of each
  17    size per family (its counts are multiples of 5). *(M1 change: independent draws let a world's
  18    40 abusive groups differ in size from their twins by chance, and every count that follows size
  19    inherited it. On seed 1, pair 1 had group-size AUC 0.37 and failed the render-length rule at
  20    s = 0.)*
   ...
  77  **Collisions (M1 implementation notes):**
  78  - Pair 3: on a day when several accounts have a session, each draws a distinct free hour.
  79  - Pair 1: the rescale is a linear warp of session start instants, anchored at the claim. Warped
  80    sessions may leave the chronotype hour, and never share a (day, clock hour).
  81  
   ...
  85  **Tuned in M1 (constants in `families.py` and `generator.py`).** This table overrides the
  86  numbers above. Measured on standard worlds, seeds 1–2, at s = 0 and 1: max render 1,677 tokens.
  87  
  88  | Constant | Above | Now | Why |
  89  |---|---|---|---|
  90  | `LOGIN_RATE` | 4 | 2.0 | event budget |
  91  | `LAB_ATTEND_P` (pair 2) | 0.75 | 0.6 | event budget. Co-attended meetings per account pair are ~4, not 6, and the s = 1 oracle AUC is still ~0.97 |
  92  | `OFFICE_DAY_P` (pair 3) | 0.15 | 0.07 | event budget |
  93  | `FAMILY_DAY_P` (pair 4) | 0.2 | 0.10 | event budget. 0.07 left the s = 1 oracle AUC at ~0.94, too near the 0.90 control |
  94  | `SHARED_DAY_P` (pair 6) | 0.12 | 0.055 | event budget |
  95  | `NEWCOMER_CYCLE_SESSIONS` (pair 5) | 1–3 | 1–2 | event budget |
  96  
  97  **Further M1 implementation notes** (the generator worker's resolutions, accepted on review):
  98  - T-pair sessions are eligible by clock hour: the whole hour follows the cycle's claim and ends
  99    inside the cycle. A mechanism that moves events within the hour can't change which sessions
 100    exist.
 101  - Pair 1's first cycle has 4–5 sessions of 3–8 credits, summing to 25. Events at or after
 102    `WORLD_END` are dropped, which can only happen to the last account.
 103  - Pair 5:
 104    - The main account paces its sessions (daily probability `min(1, 20 / days left)`). It reaches
 105      25 in every cycle with about two weeks left.
 106    - A rule-following newcomer keeps the twin's session count and event count per session, and
 107      copies the main account's amounts in order. Where the main account has too few session days
 108      left, the rest are its own sessions at its own hour.
 109    - Its sessions share the main account's (day, clock hour) by design: the exception to the
 110      session rule above. Pair 5's oracle compares exactly that.
 111  - Linkage is observed at `max(created_at, W0)` + uniform{0..2880} min. The vendored clustering
 112    drops observations older than 365 days, and an aged main account would otherwise leave its
 113    group's cluster (coordinator fix on review).
 114  - A cycle whose claim would fall at or after `min(cycle end, WORLD_END)` keeps its row (granted
 115    25, consumed 0) with no claim event. The cycle-end case covers an aged account whose
 116    overlapping cycle ends minutes after W0. Without this rule its claim landed in the next cycle,
 117    which then had two claims (found in M1).
 118  
 119  **Measured channel tags (SPEC §5.4, M1, frozen as `MEASURED_TAGS`).** Pairs 2, 3, 4 and 6 measured
 120  T: every one-sided upper bound ≤ 0.57. Pair 1, designed F, measured M (best F12 AUC 0.69): its
 121  twin also takes turns, so F-CYC-2 and F-CYC-3 fire in both. Pair 5, designed M, measured F (0.91,
 122  through F-CYC-3). The headings below keep the design tags.
 123  
   ...
 165  - **Twin.** Each account has its own hour `h_a ∈ 6..23`, at least 3 h from every other member's.
 166    The set is drawn uniformly from the valid sets, then shuffled over accounts. (8..22 fits only
 167    five accounts at a 3 h gap; changed in M1, before the freeze.)
```

## 3. Implementer completion summaries, Orca run `run_5467cf47b477` (verbatim `worker_done` bodies)

Implementers: Orca Claude workers on Sonnet 5.5 at high effort. Listed in dispatch order. Each body
is the worker's own claim; the coordinator's acceptance checks are not part of this disclosure.

### M1 task W1: the testbed world generator (generator v1)

- Task `task_dc3f544e2190`, outcome `succeeded`, message `msg_8f802d4ed2c4`, completed 2026-10-21T20:11:05.435Z.
- Subject: W1 done: generator v1 (families, generator, tests)
- Files modified: `testbed/world/families.py`, `testbed/world/generator.py`, `tests/test_generator.py`

> Implemented testbed/world/families.py, generator.py and tests/test_generator.py per DESIGN.md plus the three addenda (manifest tier/reported, event budget, small scale): pytest 65 passed (exit 0), ruff check/format --check and mypy clean (exit 0); standard world generates in 0.5 s; six-account groups at s=1 seed 1 have mean 94.4, p99 120.2, max 124 in-window events. Budget tuning (login mean 2.0 and five (tune) activity probabilities, listed in the report) means pair 2 misses its target (mean co-attended meetings per pair 3.98, not >= 6) and pair 4's oracle AUC at s=1 is only ~0.94, so those two need your call; all other DESIGN resolutions (pair 4 hours 6..23 per your answer, pair 3 distinct hours, pair 1 linear warp, hour-level T eligibility, pair 5 follower fallback) are in the report. DESIGN.md needs the (tune) and pair-4 updates; I did not run make test or touch any coordinator file.

### M1 task W2: the raw-event render, its parser, and token counts

- Task `task_a6686a82c59e`, outcome `succeeded`, message `msg_3bcfd63341bb`, completed 2026-10-21T19:49:50.027Z.
- Subject: W2 render/parser/tokens done; over token budget at 160 events
- Files modified: `testbed/pipeline/render.py`, `testbed/pipeline/tokens.py`, `tests/test_render.py`

> Created testbed/pipeline/render.py (raw-v1 render, exact parse, render_unit), testbed/pipeline/tokens.py (out-of-process Qwen3.5-9B count via es-p6-train, loading only the cached tokenizer file) and tests/test_render.py; pytest 38 passed (exit 0), ruff check, ruff format --check and mypy clean (exit 0), nothing else edited. TOKENS: golden example 368 total (header 163, 14 event lines ~14.6 each); synthetic 6-account/160-event/3-shared timeline (throwaway script) 2537 total = 221 header + 2315 event lines = 14.5 tok/event line, so it is OVER the 1792 budget by ~745; only about 108 events fit at 6 accounts. Qwen splits every digit and a space before a digit costs an extra token; measured savings if you want to trade legibility: HHMM without colon -1.0 per line, dropping the d prefix -1.0 per line (both: 12.5/line, 1995 + 221 header, still over), so activity must be sized to roughly 100 events per 6-account group. I kept the spec format d12 14:03 A3 use 2 with a 2-line fixed event-type legend (about 45 tokens, droppable if the prompt carries it) and signal tokens device/payment/recovery/network/behavior (behavioral_template renamed to avoid the banned word template).

### M1 task W3: F12 and F64-w90 features

- Task `task_a239b458342a`, outcome `succeeded`, message `msg_ac96aa01d5b6`, completed 2026-10-21T19:47:56.410Z.
- Subject: W3 done: F12 and F64-w90
- Files modified: `testbed/features/f12.py`, `testbed/features/f64.py`, `tests/test_features.py`

> Created testbed/features/f12.py (F_IDS, f12), testbed/features/f64.py (FEATURES, f64, pure function of UnitTimeline plus the F12 values) and tests/test_features.py (9 known-answer tests: all 64 values on a hand-built 3-account unit, single-claim and no-claim edge cases, f12 incl. refusals, name checks). Checks: pytest '9 passed in 0.10s' exit 0; ruff check 'All checks passed!' exit 0; ruff format --check '3 files already formatted' exit 0; mypy 'Success: no issues found in 3 source files' exit 0. w90 choices: age_days_* = whole days from created_at to window start, floored at 0; joined_in_cycle_frac = share created in [window_start, window_end); hist_claims_* and hist_claim_cycles_* = 0.0 (kept, no pre-window claims); history_span_days = whole days from earliest created_at to window start, 0 if none precede; claim_day_mean/span/gaps use in-window claims offset from the window start with window length (2160 h) as the gap fallback; acct_claim_gap_median_d_mean and signup_to_first_claim_h_median use in-window claims only (so latency for a pre-window account is long); active_days counts distinct UTC dates (equal to the vendored month-day key for a window under a year). Nothing skipped; I ran no make targets, per the rules.

### M1 task W4: AUC, intervals, the null rule, and cross-validated scores

- Task `task_1eabe6a9a84b`, outcome `succeeded`, message `msg_69c3cf4f751f`, completed 2026-10-21T19:48:24.351Z.
- Subject: W4 done: metrics.py, cv.py, test_metrics.py
- Files modified: `testbed/eval/metrics.py`, `testbed/eval/cv.py`, `tests/test_metrics.py`

> Created testbed/eval/metrics.py (auc, delong_var, null_rule, vectorised bootstrap_auc, auc_interval, auc_upper_bound) and testbed/eval/cv.py (models, group_folds, oof_scores), with 26 known-answer tests in tests/test_metrics.py. Checks: pytest '26 passed in 5.88s' exit 0; ruff check 'All checks passed!' exit 0; ruff format --check '3 files already formatted' exit 0; mypy 'Success: no issues found in 3 source files' exit 0; bootstrap of 1,000+4,000 scores x 2,000 resamples takes 0.52s (multinomial counts on the sorted unique grid, no pair loop). Deviations: no scipy (mypy has no scipy stubs and pyproject is outside my files), so the normal quantile is stdlib NormalDist and AUC uses searchsorted placements; I added a public group_folds helper that oof_scores calls so the one-test-fold-per-group property is testable; I did not run make test per the rules, and the delong test hand value is 1/32 (my first 1/128 was an arithmetic slip, fixed).

### M1 task W5: the six oracle statistics

- Task `task_c1160819f82d`, outcome `succeeded`, message `msg_093aae3b4080`, completed 2026-10-21T19:51:17.460Z.
- Subject: W5 oracle statistics done
- Files modified: `testbed/eval/oracle.py`, `tests/test_oracle.py`

> Created testbed/eval/oracle.py (oracle(pair, tl) -> float|None, PAIRS, six private per-pair functions, higher=more abusive; pairs 1,4,6 negated) and tests/test_oracle.py (known-answer per pair plus empty/no-amount/non-consume timelines never raise). Checks: pytest '44 passed in 0.11s' exit 0; ruff check 'All checks passed!' exit 0; ruff format --check '2 files already formatted' exit 0; mypy 'Success: no issues found in 2 source files' exit 0 (own files only, no make test). Readings I chose where DESIGN is ambiguous: pair 2 equal starts across accounts count as co-attended with neither first; pair 4 account mean is over all its active days (not only shared days), amount per event = day total/event count; pair 5 newcomer without consume sessions is skipped, 'same day and hour' = main has a consume session in the same (UTC date, hour), bonus window is 0..24 h inclusive; pair 6 only sessions with >=2 accounts count; pair 1 uses cycle 0 only and earliest claim; pair 3 gap is in whole seconds; unknown pair number raises ValueError.

### M1 task W2b: compact the raw-event render to fit the token budget (follow-up to W2)

- Task `task_13375b2b1ece`, outcome `succeeded`, message `msg_7fd9441aa8ff`, completed 2026-10-21T19:53:27.076Z.
- Subject: W2b render compaction done
- Files modified: `testbed/pipeline/render.py`, `tests/test_render.py`

> Implemented raw-v1 compaction in render.py: a d<day> header line per run of same-day events (events stay one line each, in tl.events order, never merged; an out-of-order day gets a new header), HH:MM kept, letter aliases A1->A..A26->Z in accounts/shared/events, ValueError above 26 accounts, parser carries the day forward and rejects events before a header, empty or repeated headers; docstring, golden, persisted-world test updated, plus new tests (no A<digit> in any render, header coverage, out-of-order days, 26/27 accounts, parser refusals). Tokens (Qwen3.5 count_tokens, 6 accounts, 3 shared values; header incl. Events line = 214 tok): golden 3-acct/14 events 11.07 tok/event line (header 168, total 323); synthetic 160 events spread over 90 days 10.93 tok/line (total 1963), max events in 1792 = 145; clustered 4/day over 40 days 9.82-9.89 tok/line (total 1786-1796 across two random draws), max = 163 (so ~160 is right at the edge of 1,792). Checks: pytest tests/test_render.py '45 passed in 0.80s' exit 0; ruff check 'All checks passed!' exit 0; ruff format --check '2 files already formatted' exit 0; mypy 'Success: no issues found in 2 source files' exit 0; not run: make test (other workers' files), and tokens.py untouched; measurement script is in my scratchpad, not the repo.

### M1 task W6: the world runner and the F-* information-parity check

- Task `task_26377b5821a3`, outcome `succeeded`, message `msg_da28d1eedb63`, completed 2026-10-21T20:20:52.171Z.
- Subject: W6 done: run_world + F-* parity
- Files modified: `testbed/pipeline/run_world.py`, `testbed/pipeline/parity.py`, `tests/test_run_world.py`, `tests/test_parity.py`

> Created testbed/pipeline/run_world.py, parity.py, tests/test_run_world.py, tests/test_parity.py. Checks: pytest 15 passed in 53.75s (exit 0, none skipped; tokenizer venv present); tests/test_ground_truth_boundary.py 11 passed (exit 0); ruff check 'All checks passed!' (0); ruff format --check '4 files already formatted' (0); mypy 'Success: no issues found in 4 source files' (0). Parity: on the small world (120 units, all 12 families, s=1) rescore_from_render(render) equals the unit's f12 exactly for all 120, and f64(parse(text), f12) equals f64(load_timeline(...), f12). One deviation from the brief: parity observes each shared value at max(created_at, window_start), not created_at, because the vendored clustering drops observations older than 365 days and pair 5's aged main accounts are 200-400 days old (same fix the generator uses); I did not run the literal created_at variant, so the 365-day reason is from reading clustering.py, not measured. run_world also takes an optional keyword token_counter (default count_tokens) so tests can stub it. Real run: python -m testbed.pipeline.run_world --seed 1 --s 1.0 --scale standard --out runs/m1-dev-s1-seed1 exit 0. Seconds: generate 1.134, persist 3.392, detect 87.123, units 1.286, features 0.448, tokens 0.875, total 94.258. 1200 units (1200 clusters, 13686 cluster-cycles), 4733 accounts, 79516 events; tokens max 1677, mean 927.76; 0 units exceed 1,792 tokens; detection_sha256 caea14d0a691c20e7005614b4be66583e95a7fe0a90456c7f8dd08fc95ad59ac; render raw-v1. Nothing left open.

### M1 task W7: world health checks, the generator-only sample, and the channel check

- Task `task_c4d882b24432`, outcome `succeeded`, message `msg_f24a564611ed`, completed 2026-10-21T20:26:19.879Z.
- Subject: W7 health checks done; gensample seed 201 passes; null rule too loose at 40+160
- Files modified: `testbed/eval/health.py`, `tests/test_health.py`

> Built testbed/eval/health.py (nuisance, all §5.6 rows, gensample_health, channel_check, CLI) and tests/test_health.py. Checks: pytest 48 passed in 53.54s (exit 0); ruff check 'All checks passed!' (0); ruff format --check '2 files already formatted' (0); mypy 'Success: no issues found in 2 source files' (0); boundary test 11 passed. gensample seed 201, n=2000 (runs/m1-gensample-seed201/health.json): all rows PASS: nuisance_balance s0 and s1 (148 tests each, max |AUC-0.5| 0.0225), negative_control (AUC 0.499-0.505, all CIs contain 0.5), positive_control AUC 0.987/0.974/0.994/0.967/1.000/0.955 for pairs 1-6; oracle None counts (s0): pair2 16+20, pair3 5+12, pair4 10+7, pair6 67+68, pair1 and pair5 none. FINDING for your decision: on fair s=0 generator draws at 40+160 groups the DeLong-based null rule fails often: nuisance_balance failed in 10 of 40 seeds and negative_control in 5 of 40 (SPEC says about 5%), mostly sparse or discrete variables; my scratch script is in the session scratchpad (fpr.py). Choices to confirm: render_visibility also requires differs == 0; parity compares with abs tol 1e-9; twins drawn once and reused at both s in gensample; render_length/negative_control use tests = number of pairs present (6 on standard); channel_check and world CLI verified only on small-scale test worlds, not on real standard or channel worlds (world_health on a small s=1 world exits 1 because nuisance_balance failed by chance at 4+16).

### M1 task W8: a calibrated null rule (replace the DeLong interval)

- Task `task_c48a64c9b508`, outcome `succeeded`, message `msg_e7df9385d104`, completed 2026-10-21T20:41:38.142Z.
- Subject: W8 null rule done (1 unrelated world-test failure)
- Files modified: `testbed/eval/metrics.py`, `tests/test_metrics.py`, `testbed/eval/health.py`, `tests/test_health.py`

> Replaced the DeLong null_rule with null_variance + permutation_p + a screened permutation null_rule (NullRule: auc,z,p,alpha_adj,method,passed; constants NULL_SEED, SCREEN_MARGIN, PERM_FLOOR, PERM_FACTOR), and health.null_row now records auc,z,p,alpha_adj,method,passed,n_pos,n_neg per item (no lo/hi), row alpha_adj, with a per-item seed from sha256(NULL_SEED:key). Own checks: ruff check exit 0 'All checks passed!'; ruff format --check exit 0 '4 files already formatted'; mypy exit 0 'Success: no issues found in 4 source files'; pytest tests/test_metrics.py tests/test_health.py exit 0 '83 passed, 1 deselected' with test_world_health_at_s1_has_the_s1_rows deselected because it now fails (render_visibility row, which uses plain auc, not the null rule) after another worker edited testbed/world/generator.py and families.py at 13:37-13:38 (it passed, 84 passed, before that); full run was 1 failed, 83 passed. Fair-seed calibration (seeds 1000-1099, s=0, 40+160/pair, throwaway script in scratchpad calib.py, exit 0, 39 s): nuisance row (148 tests) failed 6/100 [1011 pair3/has_behavior; 1016 pair3/verified_before_first_claim_frac; 1019 pair4/edge_recovery_contact; 1027 pair2/has_network; 1082 pair6/edge_recovery_contact; 1087 pair2/domain_disposable]; negative-control row (6 tests) failed 8/100 [1018 pair1; 1026 pair6; 1034 pair3; 1042 pair5; 1043 pair6; 1079 pair4; 1089 pair4; 1091 pair2+pair6] - above 5% but ~1.4 SD at n=100, pairs 6 and 4 recur, so worth a larger check for non-exchangeability in the oracle statistic. gensample_health(seed=201,n=2000,out=runs/m1-gensample-seed201) exit 0, passed True, all rows pass (nuisance s0/s1 148 tests, negative_control 6, none failed), 79.4 s (~20 s generation, the rest is permutation tests at N=4000; may have overlapped the concurrent generator edit). Finding: the suggested 'sparse binary, 4 ones all in positives' example does not reproduce (DeLong is conservative there and the exact test fails it, p~0.0014), so the test uses 8 ones all in the negatives (tests=6): DeLong fails, exact p~0.2 passes.

### M1 task W9: balanced group sizes in generator v1

- Task `task_b8379d0e9091`, outcome `succeeded`, message `msg_8496e1d4a813`, completed 2026-10-21T20:42:52.825Z.
- Subject: W9 done: balanced group sizes
- Files modified: `testbed/world/families.py`, `testbed/world/generator.py`, `tests/test_generator.py`

> Every family process now takes k (abusive (base, mech, s, k), twin (base, k), validated 2..6) and draw_group sets k = 2 + i % 5; RNG stream names and other draw order are unchanged. Checks: pytest tests/test_generator.py tests/test_parity.py tests/test_run_world.py = '82 passed in 62.97s' exit 0; ruff check 'All checks passed!' exit 0; ruff format --check '3 files already formatted' exit 0; mypy 'Success: no issues found in 3 source files' exit 0. Standard world seed 1 (s=1): 240 six-account groups, max 131 and p99 120.6 in-window events per group (seed 2: max 136, p99 125.2); generation 0.4 s (world_groups only). Also, with your approval, fixed latent cycle_frames bug (claim past cycle end gave a cycle two claims; hit by adds_relative #89), added known-answer case; DESIGN.md needs the note. New tests: balanced counts per family (8/32 per size), 2 + i % 5 sizes, k rejection. Files changed: testbed/world/families.py, testbed/world/generator.py, tests/test_generator.py. Not run: make test (per rules), other test files.
