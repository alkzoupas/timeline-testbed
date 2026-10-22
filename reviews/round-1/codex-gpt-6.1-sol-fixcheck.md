# Adversarial review: M1 round-1 check of fixes (Codex / GPT-6.1 Sol, code)

Verdict: FIX-THEN-SHIP

## Scope and context

- Base: observed HEAD `6e14f7d5fc297b9e0eb30fb3c06759426b037284`. Candidate: working tree pinned by `reviews/round-1/coordinator/fix-candidate-manifest.txt`, SHA256 `cc1c878262d02b28cb9ca851fe8ebe7a6ac47a142060b95ea4a01821f31fc25d`. Verified all 251 entries' format and all 250 in-scope entries' content hashes and modes, both before review and after checks. `STATUS.md` is the excluded entry. No in-scope mismatch or candidate change was observed.
- Fixes: `reviews/round-1/coordinator/fixes.diff`, SHA256 `560a7b338b9749d4e53624440289c5ef4cec04f2bfc174052884cbdf16b8b78f`, independently verified before review and after checks. Reviewed its ten changed files, relevant persistence/family code and callers, SPEC §§5, 6.1–6.2, 9–10, AGENTS invariants, DECISIONS, the original review and TRIAGE. This is the single check of fixes, not a fresh full review; no staged disclosure.
- Startup exposure: visible system/developer instructions, project/global AGENTS, environment, the live dispatch, Graft starting points and an auto-generated previous-session handoff summarizing the original review's verdict and top finding. Original review, triage and author rationale were intentionally available. STATUS was read as instructed, but was not treated as a review target. No manual memory recall or session-history search was performed. Hidden hook/runtime exposure and actual launch receipt are **unknown**; this is an informed fix-check, not a blind review.
- Scope incident: the first `graft ask` reported an automatic refresh of one excluded graph file. No candidate source or coordinator-owned file was edited. The only authored artifact is this report; authorized checks wrote their normal temporary files/caches.

## Per-finding dispositions

| Round-1 finding | Disposition | Evidence and limits |
|---|---|---|
| 1. Pair 3 volume distribution | **Resolved within the agreed fix scope** | The owner accepted the mechanism. `testbed/world/DESIGN.md:162–165` accurately distinguishes preserved mean from changed dispersion, gives conditional variance `2.5·N²` versus `2.5·N`, and states the F12/F64 consequence. A single template is still drawn and reused at `testbed/world/families.py:521–529`; the note describes that implementation. At intermediate strength the displayed formula concerns the replayed sessions, as explicitly stated. No re-argument of the owner decision. The DECISIONS line/M1 report are scheduled for the freeze, not yet delivered by this candidate. |
| 2. Relative-import boundary bypass | **Partially resolved** | `_resolve` now uses package and relative level (`tests/test_ground_truth_boundary.py:87–114`), and ordinary relative `from` imports, package imports and dotted aliases have negative fixtures (`:270–290`). However, relative `import_module` remains an ordinary spelling that all boundary predicates permit; finding F1 below. |
| 3. Unguarded `verify_pair` opener | **Resolved for the actual opener** | `guard_stores` checks the seal and both paths (`testbed/eval/labels.py:79–86`), preceding both engines in `open_labels` (`:92–94`) and `verify_pair` (`testbed/eval/stores.py:19–21`). Missing-store and seal tests cover the latter (`tests/test_ground_truth_boundary.py:370–373`, `:434–445`). The `persist` exemption is sound for the current sequential fresh-store route: both paths are refused if existing before either engine is created, and labels go through the sink (`testbed/world/persist.py:49–61`; refusal test at `tests/test_ground_truth_boundary.py:376–379`). The new universal enforcement test has a separate alias gap, F2 below. |
| 4. Null rule versus AUC interval | **Resolved for description and known-answer coverage** | `testbed/eval/metrics.py:121–131` now identifies exchangeability, the AUC statistic, screening and Monte Carlo evaluation, explicitly disclaims an AUC CI, and permits rejection under unequal distributions with population AUC 0.5. The hypergeometric enumeration at `tests/test_metrics.py:233–248` correctly weights assignments of the pooled +1/−1/0 scores; `:251–271` pins exact p `0.005092447416579459`, a toleranced Monte Carlo estimate and rejection at K=24, and passing AUC 0.5 at K=20. Passing does not establish distribution equality; this remains an AUC-statistic permutation test, not an omnibus certification. No new statistical algorithm was introduced. Calibration reruns and the planned DECISIONS line are not independently established by this fix-check. |
| 5. Manifest channel tags | **Resolved** | `testbed/world/generator.py:385–386` emits measured tags under `channel_tags`, preserving designed tags separately; the regression test checks both (`tests/test_generator.py:157–159`). `testbed/eval/health.py:864–868` retains the original design comparison and adds the frozen comparison, checked at `tests/test_health.py:709–714`. The §5.4 pass rule is unchanged (`testbed/eval/health.py:907–914`): T-designed families must measure T, while F/M measurements are recorded. Thus this metadata fix does not silently add a different gate. Regenerated full-size worlds remain the coordinator's next step. |

## Findings (most severe first)

1. **F1 — P1: relative dynamic imports still bypass the label boundary.** `tests/test_ground_truth_boundary.py:136–153`, especially `:152`. Failure scenario: a module in `testbed/pipeline/` uses the following normal, literal relative import:

   ```python
   from importlib import import_module
   from testbed.world.persist import Stores

   def leak(path):
       return import_module("..eval.labels", __package__).open_labels(Stores.under(path)).label
   ```

   In that package, Python resolves the module to `testbed.eval.labels`. `_targets` only includes string literals beginning with `testbed.`, and the `open_labels` chain is rooted in a call rather than a name. Consequently `_eval_violations` and `_gt_imports` both return empty sets; the specimen has no sealed literals, forbidden store attributes or label-surface attributes either. Label-free pipeline/model code can therefore read development labels while the required boundary checks pass. This is the relative counterpart of the absolute `import_module` spelling already covered at `:286`, not built-string/getattr obfuscation. A read-only AST reproduction using the candidate's actual helpers confirmed the empty detections, with `importlib.util.resolve_name` confirming the destination; the absolute-string control is detected. **Fix:** resolve literal `import_module` calls against their literal package or the scanned module's `__package__`, expanding imported aliases; apply the same eval allowlist to the resolved target and add relative dynamic-import negative fixtures. Alternatively, fail closed on unresolved dynamic imports outside approved modules. No actual candidate pipeline leak was found.

2. **F2 — P2: the new guard-enforcement scan misses an ordinary imported factory alias.** `tests/test_ground_truth_boundary.py:174–177`, `:230–245`. Failure scenario: `labels.py` or `stores.py` adds:

   ```python
   from app.db.ground_truth_store import make_ground_truth_engine as make_gt

   def unguarded(stores):
       return make_gt(stores.ground_truth_url)
   ```

   Those modules are allowed the import and path access. `_call_name` returns `make_gt`, so the engine call is absent from `engines` and from the per-function check; the two existing guarded functions still satisfy `GUARDED_OPENERS`. The additional opener can open a confirmation engine without the guard, yet `test_every_ground_truth_engine_is_guarded` passes. A read-only reproduction appended this specimen only to an in-memory AST and ran the actual enforcement function against the otherwise unchanged module trees; it passed. **Fix:** resolve call targets through the import alias map before recognizing both the engine and guard, including module-qualified aliases, and add a negative specimen for an aliased unguarded factory. Existing `open_labels` and `verify_pair` are correctly ordered; this finding is about the new test's claimed coverage, not a remaining runtime defect in those two functions.

## Triage accuracy

The stated implementations of the runtime guard, volume note, null-rule specimen and channel metadata agree with source. TRIAGE's claim that the new static test enforces *every* ground-truth opener overstates its coverage (`reviews/round-1/TRIAGE.md:43–44`); F2 demonstrates an ordinary alias it misses. Its boundary-fix description is accurate for static `from` imports and absolute dotted strings, but does not establish closure of ordinary relative dynamic imports (F1). The reported 100-seed calibration outcomes, worker launch receipts, mutation-test run and earlier check timings are coordinator evidence, not independently rerun evidence here. DECISIONS still lacks the promised freeze entries, consistently with the triage's scheduled freeze work; that is not treated as a newly discovered defect.

## Verification

The three authorized commands ran from `/Users/alkzoupas/code/timeline-testbed`; actual process exits were read directly.

| Command | Exact observed summary | Exit |
|---|---|---|
| `make test` | `======================= 318 passed in 118.58s (0:01:58) ========================` | 0 |
| `make lint` | `All checks passed!` | 0 |
| `make typecheck` | `Success: no issues found in 39 source files` | 0 |

No failures, skips or warnings were reported by this suite run. Read-only manifest/hash checks and corrected AST reproductions also exited 0; final identity output was `Final identity check: 250 in-scope manifest entries match hashes and modes; both pinned artifact hashes match`. An initial auxiliary AST harness failed before executing the specimens because a synthesized import node lacked `lineno`; adding `ast.fix_missing_locations` corrected the harness. No source was changed for any reproduction.

## Not checked

- No direct ground-truth database access, world runs, health CLI, calibration reruns, model use, installs, downloads, network access, format command or vendored-suite rerun. The expressly authorized `make test` itself uses temporary fixture stores. Findings use source/AST inspection, not real confirmation data.
- No fresh review of unchanged M1 mechanisms, downstream M2–M6 scoring, once-only confirmation policy, concurrency/TOCTOU safety, or exhaustive hostile Python obfuscation. The current sequential `persist` exemption is not an atomic-create guarantee against concurrent writers.
- No general constant propagation, constructed strings or reflective `getattr` attack. Rejecting unresolved dynamic imports would cheaply close much of that surface without building a Python interpreter into the test, but this review's concrete bypass needs only literal relative import syntax.
- The proposed freeze entries, regenerated worlds, resulting health/channel/sample evidence and final generator hash remain coordinator work. The report's verdict is confined to these fixes and the two demonstrated enforcement gaps.
