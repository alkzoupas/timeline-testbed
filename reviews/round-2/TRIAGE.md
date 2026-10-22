# Round 2 triage (coordinator, 2026-10-21)

Review: `reviews/round-2/codex-gpt-6.1-sol-adversarial.md` (final sha256 `ae567d31…166a`), Codex
GPT-6.1 Sol at high (receipt `launch.effective`), Orca run `run_e8503c4e45c3`, task
`task_672446a5d9b9`, dispatch `ctx_d1fba5dceaf7`. Review point 2 (M4, before scoring: G2 scorer and
`prereg/M4.md`). Verdict FIX-THEN-SHIP, one finding (P2), new.

## Process record

- Candidate: base `eb9cb5e`, 289 entries, manifest sha256 `5beb5804…3e88` (`CANDIDATE.md`).
  Re-hashed by the coordinator at the checkpoint and after `worker_done`: all 289 match; the only new
  files are the report and the two coordinator files in `coordinator/`. HEAD unchanged. The permitted
  scratch dir was never created.
- Initial report snapshotted before disclosure: `coordinator/initial-snapshot-codex-gpt-6.1-sol-adversarial.md`,
  sha256 `98c31c52…3287` (read-only); the final report's Initial assessment section is byte-identical.
  Disclosed after the snapshot: `coordinator/deferred-disclosure.md` (sha256 `642d33de…c747`: both
  implementer reports and specs, coordinator notes), `STATUS.md`, `reviews/round-1/`. Reconciliation:
  one confirmed; none added, changed or withdrawn.
- Exposure (reviewer's record): an injected previous-session handoff (round 1's fix-check verdict)
  and DECISIONS.md's M4 lines before the initial assessment; other startup context unknown. Not a
  fully blind review.
- Reviewer's checks: `make test` `549 passed in 163.58s (0:02:43)`, `make lint` `All checks passed!`,
  `make typecheck` `Success: no issues found in 63 source files`, each exit 0.
- Terminal released (transcript archived); no reclaimable worker left on the run.

## Finding

| # | Finding | Valid? (coordinator check) | Action | Owner of the fix |
|---|---|---|---|---|
| 1 | P2. `r2_embed` reuses a finished chunk after checking only its ids and array structure, never the saved rows (prompt and render text) or the code that produced it; `run` then stamps current provenance on the merged sidecar and `r2.load_features` (hence G2) accepts it. The restart test pins the gap (`tests/test_r2_embed.py:157–166` reuses a chunk made from empty renders). | **Valid**, at `r2_embed.py:63–71` and `:99–106`. **Live here:** the 9B tuning embedding reused chunks 0–2 (computed 20:51–21:34) after the reboot; `prompt.py` (last edit 20:58:32) and `mlx_child.py` (21:00:36) were edited while they ran, and each chunk starts a fresh child. The prompt-text hash was checked at start; the child code was never bound. Same class (reviewer's context note): `r1_run` resume checks structure only; Nimble binds its rows but not its child code. | Bind each chunk to the child script's sha256, the exact rows bytes, the model and the layers, else recompute; record child, rows and npz sha256 in the merged sidecar from verified chunks; `load_features` checks them against the current world and child. The same binding for R1 and Nimble. Then recompute both tuning embeddings (old arrays kept for comparison), re-run `r2 tune`, re-freeze by a dated amendment (no gate world scored). | Sonnet worker (Orca); coordinator acceptance, recompute and re-freeze; then one fix check (Codex). |

## Resolution

- 23:14: embed chain stopped by the coordinator; the gate 9B run of `m1-dev-s0.75-seed1` stopped at
  chunk 1 and will be recomputed under the fix.
- **Fixed (Sonnet worker, Orca run `run_18b9ec4140c4`, task `task_028f1b6a967a`, dispatch
  `ctx_24b952fe7cf2`, `launch.effective` claude-sonnet-5-5 at high).** Both children report
  `child_sha256` (own source at startup) and `rows_sha256` (input bytes as read); `local_llm`
  (`rows_text`, `child_sha256`, `check_child_identity`) and `nimble_run` refuse a result naming
  another child or other rows. `r2_embed`, `r1_run` and `nimble_run` reuse a chunk only on matching
  rows, child, snapshot (and R2 layers), else recompute; chunks disagreeing on the child are
  refused. Merged records carry `child_sha256`, `rows_sha256` (R2 also `npz_sha256`);
  `r2.load_features`, `r2.embedding_provenance` and `r1.load_r1` refuse mismatches and missing
  fields. 13 files changed, all in the allowed set (manifest diff); `tests/test_g2.py` unchanged.
- Coordinator acceptance: code read for every touched function; `make test` `600 passed in
  132.39s (0:02:12)`, `make lint` `All checks passed!`, `make typecheck` `Success: no issues found
  in 63 source files`, `ruff format --check` `63 files already formatted`, each exit 0. Own
  mutation check (rows-hash comparison disabled in `local_llm.check_child_identity`): 7 tests fail
  across `test_r2_embed`, `test_local_llm`, `test_r1`; file restored to sha256 `15e49a7e…236c`.
- The worker edited through Python heredocs, not the Edit tool, and the files are untracked, so no
  unified diff exists; `coordinator/fix-edits.md` renders every scripted replacement from its
  session transcript as diff hunks (three slice-based scripts verbatim), with
  `coordinator/fix-files.txt` (pre → post sha256 per file).
- `mlx_child.py` sha256 is now `ae565057…6980`; every embedding binds to it, so any later edit to
  that file voids them all.
- 23:34: chain restarted under the fix (both tuning worlds first, then the gate and s = 0.5 worlds).
## Fix check (2026-10-21)

`reviews/round-2/codex-gpt-6.1-sol-fixcheck.md` (sha256 `83f6f7c2…0ca0`), Codex GPT-6.1 Sol at high
(receipt `launch.effective`), run `run_e8503c4e45c3`, dispatch `ctx_66b982210c5b`. Candidate
`coordinator/fix-candidate-manifest.txt` (`cbb9116c…f58d`); all 13 fix files matched; STATUS.md the
only drift (coordinator-owned, confirmed to the reviewer). Two escalations answered: the STATUS
drift, and `make test`'s `selftest-tiny` (a tiny random-weight child inside the authorized suite).
One CPU-only reproducer authorized at the reviewer's request (synthetic files under
`/private/tmp/tt-review-round2/`, `5/5 incorrect acceptances reproduced`, exit 0). Guard: only the
report was added. Reviewer's checks: `make test` `600 passed in 149.47s`, lint and typecheck exit 0.

| # | Finding | Valid? (coordinator check) | Action | Owner |
|---|---|---|---|---|
| 1 (PARTIAL) | P2. A chunk's `.npz` is not bound to its `.json`: the child writes the npz, then the json (`mlx_child.py:565–572`), so an overwrite killed between them, or a replaced npz, leaves a valid json beside other vectors, which `_load_part` and the merge certify. | **Valid** (write order confirmed). Needs an interrupted run with different inputs into the same out dir; none happened here (the live recompute started from empty chunk dirs). | Parent-side, so `mlx_child.py` (and the live recompute) stay untouched: remove a chunk's stale files before recomputing it; after a verified child run write a commit marker with the sha256 of its npz and json; reuse only a chunk whose marker matches both. Tests: npz-only replacement, interrupted pair, marker missing or wrong. | Sonnet worker; coordinator acceptance (self-review and tests; no further cross-vendor round). |
| 2 | P2. `r1.load_r1` for Nimble never checks `rows_sha256` (stale, wrong or missing are accepted). | **Valid**; the coordinator's fix spec asked only for the child check there. Reported-only path; not G2. | Rebuild Nimble's `id/context` rows from current renders (respecting `limit`) and require their hash; mismatch, missing, changed-render and limited-prefix tests. | Same worker. |

- **Both fixed (Sonnet worker, run `run_18b9ec4140c4`, task `task_ea91c2482c96`, dispatch
  `ctx_af13d713627d`, `launch.effective` claude-sonnet-5-5 at high).** `r2_embed` removes a chunk's
  npz, json, marker and temp file before its child runs; after `check_chunk` and
  `check_embeddings` pass it writes `chunkNNN.done.json` (npz and json sha256, tag; temp file then
  `os.replace`); `_load_part` reuses only on a matching marker. `r1.check_nimble_provenance`
  requires `rows_sha256` over the current renders (the `limit` prefix; a malformed `limit` is
  refused). Exactly the four allowed files changed. `mlx_child.py` untouched (sha256 still
  `ae565057…6980`).
- Coordinator acceptance: `diff -u` against pre-fix copies taken before dispatch, read in full;
  `make test` `617 passed in 142.97s (0:02:22)`, `make lint` `All checks passed!`, `make
  typecheck` `Success: no issues found in 63 source files`, `ruff format --check` `63 files
  already formatted`, each exit 0. Own mutation check (the `_clear_chunk` call removed): 17
  `test_r2_embed` tests fail; restored to sha256 `b854265f…93ba`. Worker's mutation checks (npz
  comparison, Nimble rows comparison) failed their tests as intended.
- No further cross-vendor round (AGENTS.md: one round plus one check of the fixes).
- 2026-10-22: both tuning embeddings recomputed under the fix (9B 00:22, 4B 00:48; 0 chunks
  reused); bit-identical to `runs/m4-embed-superseded/` in every array and 300-row chunk, so the
  reused 9B chunks were not stale. `r2 tune` re-run: identical grid, selections and `tune.md`;
  only `tune.json`'s embedding block changed. Re-frozen in `g2.TUNE_SHA256` and `prereg/M4.md`;
  `make test` `617 passed in 141.38s (0:02:21)`, lint, typecheck and format clean, each exit 0.
  Round 2 closed.
