# Round 1 triage (coordinator, 2026-10-21)

Review: `reviews/round-1/codex-gpt-6.1-sol-adversarial.md` (final sha256 `164dfc23…c141`), Codex
GPT-6.1 Sol at high (receipt `launch.effective`), Orca run `run_4ce13161fdd5`, dispatch
`ctx_cf045cb28308`. Verdict FIX-THEN-SHIP, five findings, all new (first round).

## Process record

- Candidate: base `6e14f7d`, manifest block sha256 `2dd6ad93…9037`, 251 entries. Re-hashed by the
  coordinator at disclosure and again after `worker_done`: every entry matches except `STATUS.md`
  (coordinator-owned, edited after pinning). No new files outside `graft/` and `reviews/`. HEAD unchanged.
- The first Codex turn stopped on a usage limit. The same dispatch later resumed (terminal taken over by
  the owner), wrote its initial report and asked at the checkpoint. No relaunch was needed.
- Initial report snapshotted before disclosure: `reviews/round-1/coordinator/initial-snapshot-codex-gpt-6.1-sol-adversarial.md`,
  sha256 `21237146…e03e` (read-only). Deferred material disclosed after the snapshot:
  `reviews/round-1/coordinator/deferred-disclosure.md`, sha256 `23a3c268…af7` (STATUS's M1 decisions,
  DESIGN's M1 notes, the ten W1–W9 completion summaries). Reconciliation: five confirmed, none added,
  changed or withdrawn.
- Guard: the reviewer's only write in the candidate was its report. Its `graft map` call refreshed
  `graft/` (excluded from the candidate); disclosed by the reviewer at the time.
- Reviewer's checks: `make test` 286 passed, 1 warning (exit 0); lint and typecheck exit 0.

## Findings

| # | Finding | Valid? (coordinator check) | Action | Owner of the fix |
|---|---|---|---|---|
| 1 | P1. Pair 3's single group template changes the distribution of total volume: per-account total is `N·(a+b)` (variance 2.5·N²) against the twin's sum of N draws (2.5·N). SPEC §5.1(3) matches "total activity volume". | **Valid as a SPEC conflict.** Checked `families.py:521–529`: the code implements SPEC §5.2 exactly ("the template session is one draw from the twin's per-session distribution"). Means match; the dispersion doesn't. Any cross-account identity of amounts changes the dispersion of group sums, and pair 4's SPEC mechanism does the same by design (its oracle is that variance ratio). F12 is unaffected (amounts stay under the rate limits and the cycle grant); F64 sees it partly (trees 0.83). | **Owner decision (taken: keep; see Resolution).** Recommended: keep the SPEC mechanism. Record in DECISIONS that §5.1(3) matches nuisance distributions *other than* the mechanism's own effect, so the dispersion of aggregates that pair 3's template or pair 4's coupling creates is part of the mechanism. Report F64's partial view of pair 3 as a result. Alternative: per-session-index templates (each account's own total then matches; group totals still don't). That changes the generator and the pair 3 oracle. | Owner, then coordinator |
| 2 | P1. `tests/test_ground_truth_boundary.py` drops `ImportFrom.level`, so `from ..eval.labels import open_labels` gets past the guard. | **Valid.** Confirmed at lines 61–70 and 117–126. Coordinator found a second hole: `from testbed import eval` records only `testbed`. | Resolve relative imports against the module's package; check fully qualified targets (`testbed.eval` itself is not shared); also check dotted attribute chains and import-name literals. Add negative fixtures (relative, package, star, attribute, dynamic) and positive ones (`metrics`, `cv`). | Coordinator (never delegated) |
| 3 | P2. `testbed/eval/stores.py:verify_pair` opens a ground-truth store without the confirmation seal. | **Valid.** Pre-existing (M0), exposed by the M1 seal. Callers: `tests/test_smoke_min.py`, `tests/test_generator.py`. No real confirmation store exists or was opened. | Add one guard in `labels.py` (seal check plus missing-store refusal), called by `open_labels` and `verify_pair`. Add tests: `verify_pair` refuses an unsealed `confirm-*` pair, and every function that builds a ground-truth engine calls the guard first (`persist` is exempt: it refuses existing stores). `ground_truth_db`/`_url` stay in `persist.py`, `labels.py` and `stores.py`. | Coordinator (confirm-* seal) |
| 4 | P2. `null_rule` is a permutation test of exchangeability, not an AUC confidence interval. The reviewer's counterexample (40 positives at ±1, 160 negatives at 0; population AUC 0.5) rejects in ~27% of samples. | **Valid on the description; the rule stays.** For every §5.6 null-rule row, the null that should hold is equality of distributions (§5.1(3) "drawn from the same distribution"; *s* = 0 is the twin's own process), and the permutation test tests exactly that, more strictly than an AUC CI would. But the docstring claims it is a test-inverted AUC interval, and its calibration line (z 6%, permutation 4%) disagrees with W8's reported 6/100 (nuisance) and 8/100 (negative control). | Rewrite the `null_rule` docstring to say what it tests. Add a known-answer test pinning that semantics (the counterexample rejects at K = 24, passes at K = 20). Re-measure the fair-generator false-failure rate per row on the current generator (100 seeds, s = 0, 40 + 160 per pair); the coordinator re-runs it. The DECISIONS line records the rule as an exchangeability test that replaces SPEC's CI wording. | Sonnet worker (code, test, measurement); coordinator (DECISIONS) |
| 5 | P2. `world_manifest` writes the design tags (`CHANNEL_TAGS`: pair 1 F, pair 5 M), not the frozen `MEASURED_TAGS` (pair 1 M, pair 5 F). The test pins the design tags. | **Valid.** `MEASURED_TAGS` has no consumer. Manifests go to the ground-truth receipt (the inference-receipt test already checks that). | Manifest `channel_tags` = `MEASURED_TAGS`; add `designed_channel_tags` = `CHANNEL_TAGS`. The channel record gains `frozen` and `matches_frozen`; its pass rule is unchanged. Update the tests. A generator change: regenerate the M1 worlds (step 6). DESIGN.md note by the coordinator. | Sonnet worker; coordinator (DESIGN.md) |

## Resolution (2026-10-21)

- **1. Owner decision: keep the mechanism and record it.** No generator change. DESIGN.md's pair 3 gets a "Volume"
  note; DECISIONS gets the line at the freeze; the M1 report states F64's partial view of pair 3.
- **2. Fixed (coordinator).** `tests/test_ground_truth_boundary.py` resolves relative imports and checks
  full targets (imports, attribute chains with aliases expanded, dotted `testbed.` literals). Ground-truth
  imports are allowed only in `labels.py`/`stores.py` (and persist's two names). 26 new specimen and
  resolution tests (31 new boundary tests in all with item 3's).
- **3. Fixed (coordinator).** `labels.guard_stores` (seal plus missing-store refusal) runs first in
  `open_labels` and `stores.verify_pair`. New tests: `verify_pair` sealed and missing-store refusal,
  `persist` refuses an existing store, and a static check that every ground-truth engine is built
  after `guard_stores` (it finds exactly the two openers). Mutation check: with the guard call
  removed from `verify_pair`, 3 boundary tests fail.
- **4. Fixed (Sonnet worker W10, Orca run `run_f4c68e6c4880`, `launch.effective` claude-sonnet-5-5 at high).**
  The docstring now says exchangeability test, not AUC interval. A known-answer test checks the exact
  enumeration p (0.005092447416579459) and the K = 24 fail / K = 20 pass. Calibration re-measured on the
  current generator (seeds 1000–1099, s = 0, 40 + 160 per pair). Re-run by the coordinator: nuisance row
  (148 tests) fails **2/100**, negative-control row (6 tests) fails **4/100** (exit 0, 29 s).
  Coordinator wording fix to the test docstring.
- **5. Fixed (W10).** Manifest `channel_tags` = `MEASURED_TAGS`, plus `designed_channel_tags`; the channel record
  gains `frozen`/`matches_frozen`. The DESIGN.md note is the coordinator's. The generator changed, so the M1
  worlds get regenerated.
- W10 diff review: every one of the six files reverts by the transcript's edits to its candidate hash,
  so the change is exactly what the transcript shows (`ruff --fix` only re-sorted an import).
- Checks after all fixes: `make test` `318 passed in 115.82s`, `make lint` `All checks passed!`,
  `make typecheck` `Success: no issues found in 39 source files`, `ruff format --check` `39 files
  already formatted`, each exit 0.
- Fix candidate: `reviews/round-1/coordinator/fixes.diff` (sha256 `560a7b33…78f`, all ten changed files
  against round-1 candidate copies verified by hash) and `fix-candidate-manifest.txt` (sha256 `cc1c8782…25d`,
  251 files, none new).

## Check of the fixes (the one cross-vendor check)

`reviews/round-1/codex-gpt-6.1-sol-fixcheck.md` (sha256 `aa363096…ea2`), Orca `run_3d8f267c3eb4`,
Codex GPT-6.1 Sol at high (effective). Verdict FIX-THEN-SHIP. Findings 1, 3, 4 and 5 resolved; finding 2
partially. Two new findings, both valid and both in the boundary test (coordinator-owned):

| # | Finding | Valid? | Action (coordinator) |
|---|---|---|---|
| F1 | P1. `import_module("..eval.labels", __package__)` passes every boundary predicate | **Valid.** Reproduced: no violation before the fix. | `import_module` with a literal name resolves via `importlib.util.resolve_name`, against a literal package or the module's own package; a non-literal name or package, and any `__import__`, fail closed. 11 new specimens (7 caught, 2 allowed, 2 ground-truth imports). |
| F2 | P2. An aliased factory (`... import make_ground_truth_engine as make_gt`) escapes the guard scan | **Valid.** The item 3 note above ("every ground-truth engine") overstated that test's coverage. | `_guard_report` resolves calls through import aliases (names and module aliases); the factory may only be called (no rebinding, passing, module-level build); the guard must be the real `testbed.eval.labels.guard_stores`. 8 new specimens. |

Both of the reviewer's reproductions now fail the scan. Replayed in memory: F1's module gives
`{'testbed.eval.labels'}`; F2's specimen appended to the real `stores.py` gives "unguarded builds an
engine without guard_stores first"; the real `stores.py` stays clean, with `verify_pair` guarded.
`make test` `337 passed in 118.48s`, `make lint` `All checks passed!`, `make typecheck` `Success: no
issues found in 39 source files`, format `39 files already formatted`, each exit 0. Per AGENTS.md's
budget (one check of the fixes) these two fixes get coordinator self-review and tests, not another
cross-vendor round. Remaining limit, as the reviewer notes: the scan reads ordinary spellings, not
constructed strings or reflection.

## Next

The generator changed (manifest tags), so regenerate the M1 worlds, re-run health, the channel check
and the generator-only sample, and freeze.
