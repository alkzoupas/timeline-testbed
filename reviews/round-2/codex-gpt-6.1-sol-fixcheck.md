# Adversarial review: round-2 provenance fix check (Codex / GPT-6.1 Sol, code)

Verdict: **FIX-THEN-SHIP**

## Scope and context

- **Base:** HEAD `eb9cb5ef75dd7f3c32474606d3b1cb546dc4f3a6`. **Pinned fix candidate:** `coordinator/fix-candidate-manifest.txt`, 289 entries, SHA256 `cbb9116cbb91457e72ecfe9d7afb103b3f713c7d940f4a9443eefbf92184f58d`. Recomputed the manifest digest and checked every entry's bytes/mode; all source/test entries match, including all 13 after-hashes in `coordinator/fix-files.txt`. STATUS.md alone differs: current SHA256 `8375ccc4cc68aa54c544a429ed4599508334f8c63406c6f8d674938db3a42679`. Coordinator message `msg_7e6e45ce86ed` confirms its post-pin state/check updates and authorizes this scope exception. The fix manifest differs from the original round-2 candidate in the 13 named files plus coordinator-owned STATUS.md. Source identity was rechecked after the suites and reproducer.
- **Inputs read:** AGENTS.md; SPEC.md; STATUS.md; Makefile; adversarial-review skill and `references/reviewer-inputs.md`; Orca skill/runtime guides; finding 1 and reconciliation in `codex-gpt-6.1-sol-adversarial.md`; TRIAGE.md; CANDIDATE.md identity/manifest; the complete fix-edits.md change record and fix-files.txt; the eight changed source files and five changed test files, focusing on every recorded edit and its surrounding functions; unchanged `testbed/eval/g2.py:159–263` and the R2 pooling path. Used graft for source locations and exhaustive `read_rows` callers. The change record was inspected, never executed; current `check_source` compares rows, so its recorded temporary mutation is absent.
- **Exposure:** informed fix check by design; prior finding, triage, implementation edit record and STATUS were available. An injected previous-session handoff disclosed the preceding review's P2 and check results. Other startup/recall exposure is **unknown**. No claim of blind independence or independently verified effective launch model/effort; the title uses the assigned reviewer identity. No deferred material or checkpoint applies.
- **Ownership:** authored only this report in the repo; scratch reproducer/data under `/private/tmp/tt-review-round2/`; permitted caches and automatic graft refresh. No source fixes, shared checkpoint edits, commits, installs or nested workers.

## Findings (most severe first)

1. **[P2, remaining part of finding 1] A resumed chunk's NPZ is still not bound to its verified JSON.** — `testbed/models/r2_embed.py:101–103`; publication at `testbed/models/mlx_child.py:565–572`; merge/recertification at `testbed/models/r2_embed.py:148–166`.

   Failure scenario: chunk JSON/NPZ for input A exists; retry under different text or child B correctly rejects A's metadata and begins recomputing. The child overwrites the same NPZ before writing its JSON. Kill it between those writes, then restore input/child A and resume. A's JSON passes the new child/rows/snapshot/layers checks, while B's finite arrays with the same IDs/keys/shapes pass `check_embeddings`. No per-chunk vector digest links these files. The merge then records A's rows/child and hashes the already mismatched merged vectors. `r2.load_features` accepts all three hashes, so G2's `_features` → `pooled_features` → `load_features` path accepts B's vectors as A. The same gap accepts an independently replaced chunk NPZ without changing its JSON.

   **Evidence:** the authorized CPU-only reproducer created precisely a matching A JSON / different-vector NPZ pair. `_load_part` accepted it, and `load_features` accepted value `99.0` after synthetic publication of the runner's exact merged-sidecar fields. Publication was reproduced directly; neither the runner nor G2 was executed. This proves the integrity gap, not that any live artifact is corrupt.

   **Fix:** include the chunk NPZ digest in the child's completed metadata and verify it before reuse; publish a complete pair safely and invalidate the old completion record before overwriting. Add a restart test that replaces only the NPZ, preserving valid JSON, and verifies refusal/recomputation. The final NPZ hash alone cannot detect a mismatch already merged.

2. **[P2, incomplete Nimble analogue] Nimble evaluation ignores its recorded rows hash.** — `testbed/eval/r1.py:55–79`, `:101–104`.

   Failure scenario: score renders A, retain the run's world checksum/schema/adapter/current-child metadata, then edit renders under the same IDs to B. Unlike the local-model branch, the Nimble branch never reconstructs current rows or checks `rows_sha256`; stale scores are accepted. An incorrect or absent field is accepted too. This is a remaining analogue of the accepted content-binding fix, not a regression introduced by its new child check. R1/Nimble do not feed G2.

   **Evidence:** the CPU reproducer independently exercised stale, incorrect and missing `rows_sha256`; all three `r1.load_r1(..., "nimble", ...)` calls returned `[5.0]`. The existing passing Nimble-loader fixture itself omits this field (`tests/test_r1.py:569–583`), so current tests explicitly permit the omission.

   **Fix:** reconstruct Nimble's serialized `id/context` rows from current renders, respecting `limit`, and require their hash in `check_nimble_provenance`/`load_r1`. Add mismatch, missing-field, changed-render and limited-prefix loader cases, parallel to the local-model cases.

## Finding dispositions and coverage

| Target | Disposition | Evidence |
|---|---|---|
| Original finding 1 / R2 as triaged | **PARTIAL** | `check_source`/`check_chunk` now bind JSON to exact prompt/render row bytes, current child, snapshot path and layers; absent/mismatching fields cause recomputation. Mixed child hashes are refused. Merged `child_sha256` comes from checked chunks; `rows_sha256` hashes the concatenated input each chunk was checked against; final `npz_sha256` binds the written merged file. `load_features` requires and compares the new rows/child/NPZ fields. The unchecked JSON-to-chunk-NPZ link remains finding 1 above. |
| Local R1 (`r1_run`, `r1.load_r1`) | **RESOLVED for the triaged resume/input binding** | `_load_part` checks rows/child/snapshot plus score structure; fresh chunks get the same identity check; mixed children are refused; the sidecar carries checked child identity and full/prefix rows hash. The loader requires matching current rows/child through `r2.check_provenance`, retaining the float32-head check. |
| Nimble child/runner/evaluation analogue | **PARTIAL** | The child hashes bytes actually read and its startup source; runner checks child, exact rows, schema/adapter/settings, agreement across chunks and with count, and records rows/child hashes. Evaluation adds the child check but omits rows verification (finding 2). |

G2's unchanged dry-run (`g2.py:193–208`) and scoring (`:211–263`) both call `_features` (`:186–187`), which delegates to the checked R2 pooling/loader path. Ordinary final-sidecar rows/child/NPZ mismatches now fail closed there; neither caller bypasses the loader. The mixed chunk-pair path above is the supported remaining route by which stale/mismatched vectors can reach G2. `embedding_provenance` requires all listed fields but is an extractor, not itself a hash verifier; tune loads features first. No additional supported regressions were found in the scoped edits. All changed child `read_rows` call sites unpack the new tuple correctly.

## Tests and verification

Commands ran from `/Users/alkzoupas/code/timeline-testbed`. The suites were started unchanged under the task's explicit authority.

| Command | Exact observed summary | Exit |
|---|---|---:|
| `make test` | `======================= 600 passed in 149.47s (0:02:29) ========================` | 0 |
| `make lint` | `All checks passed!` | 0 |
| `make typecheck` | `Success: no issues found in 63 source files` | 0 |
| `.venv/bin/python /private/tmp/tt-review-round2/fixcheck_repro.py` | `5/5 incorrect acceptances reproduced (CPU-only synthetic helper calls)` | 0 |

No suite failures or skips were reported. The additional reproducer was explicitly authorized by the coordinator's blocking-ask reply; it invokes only `_load_part`, `load_features` and `load_r1` on synthetic files, with model/child identity helpers patched. Script SHA256: `e07b999ce2ef1dcff617b8fa520f70dfc9fd40a446700853edb3f8d2a01ac19c`. Exact output:

```text
R2 mixed chunk JSON=A / NPZ=B: ACCEPTED, mean_final=99.0
R2 merged sidecar for A / vectors B: ACCEPTED by load_features, value=99.0
Nimble stale rows_sha256: ACCEPTED by load_r1, score=5.0
Nimble mismatch rows_sha256: ACCEPTED by load_r1, score=5.0
Nimble missing rows_sha256: ACCEPTED by load_r1, score=5.0
5/5 incorrect acceptances reproduced (CPU-only synthetic helper calls)
```

The new tests **do discriminate the ordinary broken behaviors**: R2's fake vectors now depend on render text (`test_r2_embed.py:89–91`), changed-text/same-ID resumption asserts recomputation and correct values (`:220–227`), identity/layers/snapshot/missing-field cases assert recomputation, and mixed children assert refusal. Loader tests cover new-field mismatch/omission, changed renders and replacement of a **final** NPZ (`test_r2.py:200–248`). Local R1 and Nimble resume tests assert which chunks are recomputed, so their ID-dependent fake scores do not defeat those call-count assertions. Independent serialization/hash expectations and child-read tests strengthen the checks. However, none replaces only a **chunk** NPZ while retaining its valid JSON, and Nimble-loader tests cover child identity without requiring rows identity. The passing 600-test suite therefore misses both supported gaps. No mutation testing was performed by this reviewer; triage's earlier mutation results are coordinator evidence only.

## Not checked

- No live embeddings, superseded embeddings, gate labels, s = 0.5 labels or confirmation paths were opened. Outside the authorized test suite, no world generation, G2/dry-run, R2 tune/score or extraction/scoring runner was invoked. No live numerical result, recomputation, tuning comparison or re-freeze is certified here.
- `make test` includes `tests/test_local_llm.py:422`, which spawned `mlx_child.py selftest-tiny` on a tiny randomly initialized model. Discovered during execution and reported immediately as `msg_477a9557eb28`; coordinator acknowledgment `msg_1f59ab0e9891` confirmed this integration test is included in the authorized suite and uses no downloaded weights. No separate GPU/model command was launched by this reviewer. The background job was not touched; its health is the coordinator's observation, not independently verified here.
- The reproducer simulates an interrupted write's resulting file pair; it does not kill a real child or invoke G2. The interrupted overwrite sequence is established by source write order. Existing trust assumptions about immutable snapshot contents, installed dependencies, coordinated filesystem writers and final R1/Nimble score JSONL files were not independently audited.
