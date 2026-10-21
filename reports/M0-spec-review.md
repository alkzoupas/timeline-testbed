# M0 spec review (2026-10-21, coordinator)

This reviews SPEC v0.1 against the vendored code at `fd5deca`. Paths are under
`vendor/entitlement-shield/`. AUC noise figures use the normal approximation at chance.

## (a) The claim, and what would falsify it

**The claim.** Take the families where the only thing separating one operator from several real
people is cross-account timing, which the detector's 12 features can't see by construction. On
families it never trained on, an LLM that reads a fixed render (probed or fine-tuned) ranks the
abusive groups above their twins better than F64 does, at a stated *s*, with a paired
group-bootstrap 95% interval that excludes 0. It's a claim about the method only.

**What falsifies it.** A valid testbed (G1 passes, and the oracle computed from the render text
beats F64 at *s* = 0.75) where R2 − F64 on held-out T families at *s* = 0.75 has a lower bound
≤ 0, and R3 doesn't clear F64 either. An in-family win without a held-out win falsifies Q3. A G1
failure falsifies nothing; it voids the testbed.

## (b) Problems, ranked

1. **The health checks and G1 fail on noise.** At 40 vs 160 groups, the AUC's SE at chance is
   0.051. The per-family band of 0.45–0.55 passes 67% of the time, so all 12 negative controls
   (6 families × 2 seeds) pass about 1% of the time. Nuisance balance runs about 48 tests at
   ≤ 0.60, so expect one false failure per world. G1's world-level band (SE 0.021) fails about 15%
   of the time by chance. *Fix:* check nuisance and *s* = 0 on large generator-only samples (no
   detector); give the per-world checks a Bonferroni 99% CI.
2. **Pair 6 is F.** "Never active on the same day" is exactly F-CYC-2
   (`backend/app/services/features.py:505`), and failing banding's concurrency check raises
   F-CYC-1 (`banding.py:123`). *Fix:* re-tag it F, or redesign it as T (for example, shared
   sessions whose accounts' events never interleave).
3. **T is three families, and pair 3 is at risk.** Fold B tests only pair 3.
   - F64 keeps the per-account mean and max (`scripts/p6_4_features.py:77`), which exposes a
     script that fixes counts.
   - Same-hour sessions collapse banding's diurnal histograms.
   - Its *s* = 0 ("full jitter") isn't the twin.

   *Fix:* make each T mechanism a coupling of its twin, and test that coupling with the
   following rules:
   - Hold every F12 input's distribution fixed: day sets, shared days, hourly and daily totals,
     168-bin histograms, claim, exhaustion and signup times, verification order, logins, domain,
     age and linkage.
   - Change only cross-account order or dependence. Pair 2 assigns the twin's own slots in a
     fixed order; pair 4 applies a rank copula to the amounts.
   - Each session follows the mechanism with probability *s*, and otherwise follows the twin.
4. **The vendored renders erase pairs 2 and 3.** v2 and v3 keep one row per (day, account, event
   type): the first time and the day's total (`scripts/p6_recall_pilot.py:228`). At about 11 tokens
   per raw event (unmeasured, see (c)), 2,048 tokens hold about 150 events, roughly 8 per
   account-cycle. Render length also tracks event count. *Fix:* a new raw-event render with no
   derived statistics. Count its tokens at the largest group, check visibility at *s* = 0.75 too,
   add render length to the nuisance checks, and make F64 recomputable from the text.
5. **The primary metric is undefined.** It doesn't say which negatives (T twins or all legitimate
   groups), whether the Fold A and Fold B models are pooled, or how *s* compares across families.
   *Fix:* per fold, score T abusive groups against their own twins, take the mean of the two
   folds, and bootstrap within folds. Also plot against the measured oracle AUC.
6. **G2 and G3 can pass against the wrong baseline.** G2 omits the F64 trees. The world for layer
   choice, the probe's C and the G2 worlds aren't fixed. R3 may run on a different model from R2.
   *Fix:* require a lower bound > 0 against both F64 models. Choose layer and C by nested CV on a
   tuning world that G2 never scores. Score G2 on both *s* = 0.75 seeds, and run R2 on R3's model.
7. **The channel tags don't partition, and the check is weak.** Breaking "same distribution" moves
   some mean almost every time, so M is empty. A single logistic at n = 40/160 lets a true AUC of
   0.58 pass 65% of the time, and the spec doesn't say in-sample or CV. *Fix:* CV over 2 pooled
   *s* = 1 seeds. Tag T if the 95% upper bound is ≤ 0.60 for the F12 logistic, the F12 trees and
   the detector score; F if near the oracle; M otherwise.
8. **The unit's window is ambiguous.** There's one payload per cluster-cycle, and cycles anchor
   per account (`generator.py:497`, `features.py:760`), so the "latest cycle" may be partial, which
   makes F-CYC-1 depend on cycle phase. `unit_features` scores a single cycle
   (`p6_4_features.py:149`). *Fix:* one record per cluster; build the render and F64 ("F64-w90",
   with known-answer tests) from the store over the same 90 days; fix cycle anchoring in the spec.
9. **The nuisance list misses fields that leak.** The render and F64 both expose context-only
   signals (`behavioral_template`, `network_group`; `p6_4_features.py:62`), and F-VER-1 reads
   claim-before-verify order. Endpoint tier, logins, event mix and event count aren't listed.
   *Fix:* match and check every persisted field except the mechanism.
10. **The scope and the LLM controls are thin.** Q3 rests on 2 folds × 3 families, and the
    bootstrap is conditional on those families. G1 can't test the LLM rungs at *s* = 0. F64 has no
    cross-account timing, so "beats F64" may mean only that F64 never looked there. *Fix:* claim
    "on these k families", report per family, and add one R2 run at *s* = 0 to M4 (about 2 h).

## (c) What may not work on this machine

- **Memory.** M4 Pro, 48 GiB, wired limit 40 GiB. The 9B LoRA at 2k used 34 GiB, so a longer
  render, batch > 1 or a concurrent job won't fit. *Check:* in M3, one step at the maximum render
  length, logging peak memory.
- **The secret guard blocks `tokenizer.json`.** It refused my token count, and I didn't route
  around it. Render sizing and R1–R3 need the tokenizer. *Owner:* allowlist `tokenizer*.json` in
  the HF cache, or run the counts yourself.
- **Writes into `vendor/`.** Imports write `__pycache__` and `pip install -e` writes `*.egg-info`.
  The tree is clean now (`shasum -c` exits 0). *Fix:* install only the dependencies, put the
  vendored code on the path with a `.pth`, set `PYTHONPYCACHEPREFIX`, run pytest with
  `-p no:cacheprovider`, then rerun `shasum -c`.
- **Detector runtime.** An unmemoised unit takes 1.9 s at Tier 4
  (`scripts/tier4_arms_run.py:150`), and per-account cycles multiply the units. *Check (M0):* time
  `min`, then `dev` with memoisation, and report seconds per unit.
- **Python.** Only 3.14.5 and uv 0.12.1 are installed. The vendored dependencies include fastapi,
  anthropic and openai: unused, and no calls are made. *Check:* `uv pip install --dry-run` before
  the go.
- **MLX.** `~/venvs/es-p6-train` (3.12, mlx-lm 0.32.0) can run R2/R3 out of process with no
  install. MTPLX can't return hidden states, and its 300 tok/s was measured at 18k tokens.
  *Check:* an M3 script on 10 renders at 2k.
- **Disk.** 407 GiB free. Only Qwen3.5-9B-4bit and Llama-3.2-1B are cached.

## (d) SPEC §12

1. Keep 6 × (40 + 160) at 20% if M0's timing fits, with fix 1. If pair 6 becomes T, use folds
   {1,2,3} | {4,5,6}, which gives 2 T families per side.
2. Keep the *s* grid. Make *s* = 0.75 the sole primary (it's G2's) and 0.5 secondary.
3. R2 on the 9B and on R3's model. R3 on Qwen 3.5 at about 4B; approve the download at M3, on
   measured numbers.
4. Confirm the permissions, and add the M0 `.venv` install, read-only `es-p6-train` and the
   tokenizer allowlist.
5. Tier 4 transfer: no by default. Revisit at M6 if G3 passes (ceiling 0.67, with misses driven
   by fragmentation).
6. Owner family: yes, as a separate module, so v1 regenerates byte-identically (otherwise the
   freeze voids earlier worlds). It gets its own dev-world health check and is reported apart from
   the primary.
