# Adversarial review: M4 G2 (Codex / GPT-6.1 Sol, code)

Verdict: **FIX-THEN-SHIP**

## Scope and context

- Base verified: `eb9cb5ef75dd7f3c32474606d3b1cb546dc4f3a6` (HEAD).
- Candidate: working tree defined by `reviews/round-2/CANDIDATE.md`. Recomputed the newline-terminated 289-line manifest hash: `5beb5804185e05ac583fa45c19f8a7be62519e0a7afefba3050a9940f2343e88`. All 289 file hashes and filesystem modes matched, both at entry and after the checks. Hashing deferred files did not disclose their contents. Ignored data, caches, `graft/`, and round-2 review artifacts are outside that manifest.
- Requirements read: AGENTS.md, SPEC.md (including §§6.2–6.4, 7, 7.1, 9), DECISIONS.md, `prereg/M4.md`, adversarial-review and `references/reviewer-inputs.md`, and the live Orca task/dispatch restrictions.
- Startup exposure: an automatically injected previous-session handoff disclosed the M1 fix-check verdict, “FIX-THEN-SHIP,” two remaining boundary gaps, and earlier check summaries. DECISIONS.md, an expressly required input, also disclosed the M1 disposition and coordinator choices for M4. The candidate packet and source docstrings describe intended behavior. No implementer completion report, STATUS.md contents, or round-1 review contents were read before this initial assessment. Other hidden startup/recall exposure is **unknown**; this is not a fully blind review. The assigned reviewer name is used above; I did not inspect a launch receipt to independently verify effective model/effort.
- Ownership: only this report is authored. No source fixes, shared checkpoint edits, commits, installations, new workers, GPU/model executions, or gate-data scoring.

## Initial assessment

### Findings (most severe first)

1. **[P2] Resuming embeddings can certify stale chunks as the frozen input.** — `testbed/models/r2_embed.py:63–70`, `:99–106`, `:116–125`; downstream acceptance at `testbed/eval/r2.py:65–73` and `:115–128`.

   Failure scenario: an interrupted extraction leaves a finished chunk under `<out>/<model>/<world>.parts/`. Retry with the same unit IDs but different render text or prompt (for example an earlier input copy with the same world basename, or pre-freeze work left in that output location). `_load_part` never reads or checks the saved child JSON or `chunkNNN.rows.jsonl` against the current input. Its array check verifies IDs, keys, shape/dtype, and finiteness only. The reused vector rows therefore survive even though they were computed from another prompt/render. `run` subsequently writes **current** world checksum, prompt hash, and model revision into the merged sidecar; none of those fields is derived from the reused chunks. `r2.load_features`, and thus G2's dry-run and scoring path, accepts that sidecar. F64 reproduction does not detect a change confined to LLM input. The single gate can consequently evaluate a different representation than the frozen probe and report either verdict without an input mismatch.

   Evidence is static and also visible in the existing synthetic restart test: `tests/test_r2_embed.py:157–166` preloads the first chunk with empty render strings, while `_world` at `:87–93` writes `render 0` / `render 1` for those same IDs; the test expects that chunk to be reused. Its fake embeddings depend only on IDs (`:112–116`), so it cannot detect this semantic mismatch. The observed full suite passes this test. This is not a claim that the current background extraction contains stale data; its files were deliberately not inspected.

   Fix: bind each resumable chunk to the exact serialized input rows, prompt identity, model revision, extraction configuration, and world identity before reuse; reject or recompute a mismatching chunk. The existing saved `.rows.jsonl` can supply a direct input comparison. Preserve verified provenance when merging rather than stamping it afresh, and add a known-answer restart test whose vectors depend on render text and whose changed-text/same-ID case must be rejected or recomputed. A completed embedding's sidecar should also bind its vector artifact and render input so a later load cannot pair stale arrays with current metadata.

### Coverage and non-findings

- Read the complete G2 and R2 implementations and their primary tests; inspected the R0 refactor/diff and known-answer tests, shared primary comparison/bootstrap code, metric kernels, CV fits and folds, and the label-opening/assignment path. Read `r2_embed.py`, its tests, the frozen prompt, the local child interface/array validation, and the MLX prompt/span/forward/pooling/embed path. Inspected the local-model changes and relevant fake-child tests for regressions.
- The numerical gate agrees with the explicit prereg: mean of the two T-only held-out side AUCs; worlds pooled before fitting; two fresh fold-local scaler/logistic fits; 2,000 common group resamples within (side, class); paired differences against **both** F64 models; strict lower bound > 0 at the 1.25th percentile; each model's two-sided 95% negative-control interval; and the stated PASS/LEAK/FAIL truth table. R1 has no rescue path in G2.
- Read-only inspection of both tuning JSONs confirms 1,200 units, tuning seed 301 world identity, both frozen selections and their full hashes (`c6786c0da48a0409216d314586b291730d70770ee5587f294bfb4ea0f6378c6b`, `68b88ca5ac1c0daa130d081d4e34358cd6411784f7d7f3f0f2013246a519f75b`), and zero recorded max-iteration hits across their grids. This checks saved evidence, not a rerun of tuning.
- Read `runs/m2-r0/r0.json` (sha256 `4f3f2ddc122fa384d6cc83a5840cf6489df098ac7b12b81c11e74b99250fa5a3`): its four F64 held-out primaries exactly equal `g2.M2_F64`. This is an artifact comparison, not recomputation on gate labels. Read only the tuning/gate `manifest.json` metadata to verify scale, seed, s, generator version and distinct checksums.
- Reviewed R1's evaluation wiring and restart/provenance code, Nimble's restart/output provenance and scoring-summary code, and their graph/API surfaces as context only. Their GPU paths and complete implementation correctness were not audited. R1 has a similar ID-only restart acceptance pattern; it is not a separate G2 finding.
- The separate tuning-world interpretation of nested CV, different training population size, two-model correction, 9B C at the fixed grid edge, and negative-control policy are explicit prereg/DECISIONS choices; I did not silently replace them with another design.

### Checks

All commands ran in `/Users/alkzoupas/code/timeline-testbed` under the task's offline-check authority. Test-generated artifacts were confined to permitted temp directories/caches; no custom model or gate execution was performed.

| Command | Exact observed summary / output | Exit |
|---|---|---:|
| `make test` | `======================= 549 passed in 163.58s (0:02:43) ========================` | 0 |
| `make lint` | `All checks passed!` | 0 |
| `make typecheck` | `Success: no issues found in 63 source files` | 0 |
| `git diff --check eb9cb5e` | No output | 0 |

No test failures or skips were reported. Passing checks do not exercise the stale-content failure above: the resume fake deliberately ignores input text.

## Not checked

- Did not run G2 (including dry-run), R2 tune/score, any model/GPU route, generate a world, compute any gate/s=0.5 label-derived value, inspect `runs/m4-embed/`, touch the background GPU job, or open confirmation paths. No custom executable reproducer was run because task authority names only the three offline suites and forbids the extraction route; the finding rests on the exact code/test path above.
- The actual gate embedding contents and their provenance are unverified. G2's full-input dry-run remains pending in `prereg/M4.md:90`; its successful result must be recorded before scoring, as preregistered. This review is not that experiment-readiness check.
- No new false-positive calibration experiment was performed. The prereg's bootstrap error-rate language is treated as nominal inference; a passing unit suite does not empirically establish those rates.
- Deferred reconciliation has not yet occurred. The initial assessment above must be preserved before disclosure.

## Reconciliation

Final verdict: **FIX-THEN-SHIP**. The initial assessment above is retained unchanged; its pending-disclosure statements describe the checkpoint and are superseded here.

### Disclosure record

- The coordinator's blocking-ask reply accepted the checkpoint and reconfirmed the unchanged 289-entry candidate, with only this report added outside the manifest. The initial report is preserved read-only at `reviews/round-2/coordinator/initial-snapshot-codex-gpt-6.1-sol-adversarial.md`, sha256 `98c31c52a22be8709ebf0985918b5df16a30bc2d046ce5312c1bf9d7af733287`. I independently hashed it and verified byte identity with my report before reconciliation.
- Read `reviews/round-2/coordinator/deferred-disclosure.md`, independently verified sha256 `642d33de307343cf33d64a562f71deb61e2a8a8d8c7367fa361d5813f5eac747`: the two implementer completion summaries/specifications and coordinator acceptance/restart notes. Read STATUS.md, both round-1 review reports, TRIAGE.md and historical disclosure/snapshot context; inspected historical candidate/fix-manifest identities and the scope of the historical fixes diff. This was reconciliation context, not a new audit of all M1 fixes or a rerun of the earlier experiments.
- Historical fix artifacts retain the disclosed hashes: `reviews/round-1/coordinator/fixes.diff` = `560a7b338b9749d4e53624440289c5ef4cec04f2bfc174052884cbdf16b8b78f`; `fix-candidate-manifest.txt` = `cc1c878262d02b28cb9ca851fe8ebe7a6ac47a142060b95ea4a01821f31fc25d`. Earlier verdicts, calibration claims and launch/model claims are author/reviewer history, not substituted for this review's source evidence or observed checks.

### Finding dispositions

| Initial finding | Disposition | Reason after disclosure |
|---|---|---|
| 1. Stale embedding chunks certified as current input | **Confirmed, P2 unchanged** | The coordinator notes and STATUS describe a SIGKILL/reboot and resumption of three 9B tuning chunks, making restart handling operationally relevant. They do not state that the resumed rows actually differed; no corrupt tuning or gate artifact is asserted. The source still compares only array IDs/structure and stamps current metadata afterward. Neither the saved tuning-result hashes, passing suites, nor the implementation specs provide a content check for the reused chunk. The same-ID/different-render test path identified before disclosure remains sufficient evidence of the gap. |

No findings were added, changed or withdrawn after disclosure. Prior M1 findings concern generator semantics, null-test descriptions and store/import boundaries; they do not resolve or duplicate this new chunk-content issue. The disclosed dotted-filename fix is present in the reviewed loader and regression tests, so it is not reopened. The disclosed max-iteration recording policy agrees with the implementation and does not create a new finding. R1's accepted cached/full numerical difference remains context-only and cannot rescue G2.

The three required suites retain their exact observed exit-0 results above. Reconciliation changes only this report, so they were not rerun. Candidate hashing was repeated before completion; all 289 content hashes and modes still matched the frozen manifest. The coordinator must triage finding 1 and verify the accepted resolution before the one G2 scoring run; the preregistered full-input dry-run remains separate pending work, outside this worker's authority.
