# Round 2 deferred disclosure (coordinator-owned)

Supplied after the initial assessment was snapshotted. Author narrative, not evidence.

## 1. Deferred files in the repo

- `STATUS.md` (coordinator's state file; M4 section).
- `reviews/round-1/`: the M1-freeze review (`codex-gpt-6.1-sol-adversarial.md`), its fix check
  (`codex-gpt-6.1-sol-fixcheck.md`), `TRIAGE.md`, and `coordinator/` evidence.

## 2. Implementer completion reports (Orca run `run_18b9ec4140c4`)

Both implementers were Claude Sonnet 5.5 at high effort (effective). The coordinator reviewed
both diffs and ran `make test`, `make lint`, `make typecheck` itself before accepting.

### task_173b86534537 (r0 refactor, r2.py, r1.py, R1 float32 head)

> M4 worker: r0 refactor, R2 probe, R1 float32 head done
>
> Done: r0.py refactor (R0Unit.unit, primary_comparisons/primary_boot, evaluate(rungs, extra_differences)); r2.py + r1.py + r1_run.py + mlx_child 'zeroshot' (float32 head) + local_llm wrappers, all with synthetic-data tests. make test 502 passed, lint/typecheck/format all exit 0; Part 1 r0.json sha256 equals runs/m2-r0/r0.json (4f3f2ddc...fa3). Real-model agreement (9b, 10 units, float32 head): max max_abs_diff 0.0682 nats, max |expected-expected_full| 0.0351, 10 distinct expected values in 10 rows, 0 pairs on the 0.125 bf16 grid; the f32 head did NOT bring cached vs full under 1e-3 (bf16 hidden states differ by path) - smoke of 20 units: 144 tok/s, 7.7 s/unit (GPU shared). Left: coordinator re-runs acceptance, decides on the 0.07-nat path gap before the R1 scored run; details in my final chat message.

### task_c9f8fa979fc6 (Nimble R1 runner)

> Nimble R1 runner done; count + 20-unit smoke reported
>
> Added nimble_child.py, nimble_run.py, r1.py --model nimble and tests (test_nimble_run.py, test_r1.py); make test 547 passed (exit 0), make lint, make typecheck and ruff format --check all exit 0. Count-only over 1,200 tuning renders: max 2005, mean 1313.8, p90 1705.1 prompt tokens, 0 exceed 2048 (0 exceed 8192). Smoke (20 units, runs/m4-nimble-dev/smoke): 6.5 s/unit steady (6.4 mean), peak 18.0 GiB, 20 distinct expected values (argmax 10 for 19 of 20 rows, 1 for one); no AUC computed. Your SIGKILL notice arrived while the smoke was already running; I ran no real model after it, and I apologise for the memory collision.

## 3. Coordinator notes after acceptance

- The coordinator wrote `prereg/M4.md`, `testbed/eval/g2.py`, `tests/test_g2.py`,
  `testbed/models/prompt.py` and `testbed/models/r2_embed.py` (+ test) itself.
- After acceptance the coordinator found that `r2.load_features` and `r1.load_r1` built paths
  with `Path.with_suffix`, which cuts dotted world names (`m1-dev-s0.75-seed1` ->
  `m1-dev-s0.json`); `r2 tune` failed closed with FileNotFoundError. Fixed with
  `Path(f"{stem}.json")` and regression tests using dotted names (`tests/test_r2.py`,
  `tests/test_r1.py`). Checks after the fix: `549 passed in 160.50s`; lint, typecheck and format
  clean (exit 0 each).
- Tuning results were seen by the coordinator before freezing (allowed: the tuning world is the
  selection world). The selections follow the pre-registered rule; 9B's C = 10 is the grid edge
  and was kept.
- The machine rebooted mid-M4 (about 21:49); the 9B tuning embedding resumed from three
  re-checked chunks. A Nimble smoke run had overlapped the 9B embed child, which was SIGKILLed.

## 4. Implementation specs given to the implementers (verbatim)

### task_173b86534537

# M4 worker task: R2 probe fitting, R1 zero-shot with a float32 head, report plumbing

Repo: /Users/alkzoupas/code/timeline-testbed (work in place, current worktree). Read AGENTS.md, then
SPEC.md §6.3, §6.4, §7 (R1, R2), §9, and `reports/M3-feasibility.md` ("Scorer agreement"). Use
`graft` to find code (`graft skeleton <file>`, `graft ask "..." --source`).

## Hard rules

- Edit ONLY the files listed under "Files you may edit". Do not commit. Do not edit STATUS.md,
  DECISIONS.md, prereg/, reports/, testbed/world/, testbed/eval/{labels,stores,g1,health}.py,
  tests/test_ground_truth_boundary.py, testbed/models/{prompt,r2_embed}.py, or vendor/.
  Start no workers of your own.
- **No scoring of development worlds.** Do NOT compute any AUC, report or tuning result on any
  real world under runs/ (not the tuning world, not any `m1-dev-*` / `m2-dev-*` world). Those
  runs belong to the coordinator after the pre-registration. Test everything on synthetic data.
- A GPU job is already running (`runs/m4-logs/embed-chain.log`, ~9 h of forward passes). You may
  run the real models ONLY for: the R1 agreement check below (10 units) and a <= 20-unit smoke of
  the `r1_run` CLI, both on `runs/m4-tune-s0.75-seed301` renders (label-free), writing under
  `runs/m4-r1-dev/`. No full-world runs. Never touch `runs/m4-embed/`.
- Labels reach model code only as arguments from `testbed/eval/` (SPEC §9;
  `tests/test_ground_truth_boundary.py` enforces it: outside `testbed/eval/`, only
  `testbed.eval.metrics` and `testbed.eval.cv` may be imported).
- No package installs. MLX code runs only in `~/venvs/es-p6-train` through
  `testbed/models/local_llm.py` (read-only venv).
- Match the surrounding style (docstrings, named seeds, `from __future__ import annotations`,
  typed). Python 3.14 in `.venv`; `mlx_child.py` stays 3.12-compatible.

## Files you may edit

- `testbed/eval/r0.py` (refactor, no behaviour change)
- `testbed/eval/r2.py` (new), `testbed/eval/r1.py` (new)
- `testbed/models/mlx_child.py`, `testbed/models/local_llm.py`, `testbed/models/r1_run.py` (new)
- `tests/test_r0.py`, `tests/test_r2.py` (new), `tests/test_r1.py` (new), `tests/test_local_llm.py`

## Part 1: r0.py refactor (no behaviour change)

1. `R0Unit` gets a last field `unit: str = ""`; `load_world` sets it to the unit key (the
   `units.jsonl` "unit", which is also the `renders.jsonl` "unit" and the embedding id).
2. Factor the primary-metric block of `evaluate` into two public functions, used by `evaluate`:
   - `primary_comparisons(units) -> dict[str, _Comparison]` (sides "A", "B"; the T units of each
     side; weights from one `stratified_weights` draw over (side, class) strata with `BOOT_SEED`,
     exactly as now);
   - `primary_boot(score, comps) -> tuple[float, NDArray]`: (mean over sides of the side AUCs,
     mean over sides of the `weighted_auc` resamples).
3. `evaluate(units, scores, *, rungs=RUNGS, extra_differences=())`: the expected score keys are
   `rungs x REGIMES`; "oracle" keeps its special handling only if present. Each
   `(a, b)` in `extra_differences` adds `report["differences"][regime][f"{a}_minus_{b}"]` (same
   form as the existing differences, paired on the same resamples). With the defaults the output
   must be byte-identical to today's.
4. Acceptance for Part 1: run
   `.venv/bin/python -m testbed.eval.r0 --worlds runs/m1-dev-s*-seed* runs/m2-dev-s*-seed* --out <a new dir under $TMPDIR>`
   (this re-scores R0 only, already published in M2; allowed) and show that its `r0.json` sha256
   equals `runs/m2-r0/r0.json`'s. Do not write into runs/.

## Part 2: `testbed/eval/r2.py`, the R2 probe (SPEC §7 R2)

Embeddings exist (or will) at `<embed_dir>/<model>/<world>.npz` + `.json`, written by
`testbed/models/r2_embed.py` (read it): arrays `mean_16`, `last_16`, `mean_final`, `last_final`
(float32, rows in `renders.jsonl` order) and `ids`; the json holds `prompt_sha256`,
`model_revision`, `world_checksum_sha256`, `layers`.

Constants: `CANDIDATES = ("mean_final", "mean_16", "last_final")` (SPEC's three; `last_16` is not
a candidate); `C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)`; `MODELS = ("9b", "4b")`;
`MAX_ITER = 10_000`.

Functions (pure where possible; known-answer tests for each):
- `load_features(embed_dir, model, world_dir, units, feature) -> NDArray[float64]`: rows aligned
  with `units` (those of that world, matched by `R0Unit.unit`). Raises if: the json's
  `prompt_sha256` != `testbed.models.prompt.PROMPT_SHA256`; `world_checksum_sha256` != the
  world's `manifest.json` checksum; `model_revision` != `local_llm.model_revision(model)`; any
  unit id is missing; ids are duplicated; values are non-finite. (A pooled caller concatenates
  per world in `units` order; give a helper `pooled_features(embed_dir, model, worlds, units,
  feature)` that does this for units spanning several worlds, keyed by `R0Unit.world`.)
- `fit_scores(x_train, y_train, x_test, C) -> NDArray[float64]`: a fresh
  `make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER))` (L2, lbfgs, the
  sklearn defaults otherwise) fitted on float64 training rows; returns `decision_function` on the
  test rows (log-odds; avoids saturation ties). Also report whether lbfgs hit `MAX_ITER` (return
  it, or record it; don't fail).
- `held_out_scores(x, y, pairs, C)`: as `r0._held_out` (side A = pairs 4-6 scored by the model
  fitted on pairs 1-3, and the reverse), with `fit_scores`.
- `in_family_scores(x, y, groups, strata, C)`: `cv.group_folds(strata, groups, folds=r0.FOLDS,
  seed=r0.CV_SEED)` (the same folds as R0's in-family regime), `fit_scores` per fold.
- `primary_auc(units, score) -> float`: `r0.primary_boot(score, r0.primary_comparisons(units))[0]`.
- `tune(world_dir, embed_dir, model) -> dict`: for every candidate x C, the held-out primary and
  the in-family primary on that single world, plus per-family held-out AUCs; and the selection:
  **the highest held-out primary; ties broken by the smaller C, then by CANDIDATES order**.
  Output dict: grid rows, `selected: {"feature", "C"}`, n units, embedding provenance.
- `score_rungs(worlds, embed_dir, choices) -> (units, scores)`: pooled units of the worlds
  (`r0.load_world` per world, pooled and sorted by group as `r0.run` does), `r0.score_pool`
  scores, plus for each model in `choices` (`{model: (feature, C)}`) the rung `f"r2_{model}"` in
  both regimes.
- CLI: `python -m testbed.eval.r2 tune --world W --embed DIR --model M --out DIR` (writes
  `tune.json` and `tune.md`, a table of the grid) and
  `python -m testbed.eval.r2 score --worlds W [W ...] --embed DIR --choice 9b:mean_final:1e-3
  [--choice ...] --out DIR` (worlds must share one s; writes `r2.json` = `r0.evaluate(units,
  scores, rungs=RUNGS + r2 rungs, extra_differences=[(r2, "f64_logistic"), (r2, "f64_trees")
  for each r2 rung])` plus `scores.jsonl` in `r0._write_scores` style). Both refuse a non-empty
  `--out`.
- Tests on synthetic units/embeddings (write small npz/json fixtures in tmp_path; monkeypatch
  `local_llm.model_revision`): alignment by unit id across two worlds; every provenance refusal;
  held-out uses only the other side's rows (a planted signal only in side B's training rows must
  not reach side A's scores... design a clear known-answer case); tie-breaking of `tune`'s
  selection; a separable planted feature gives AUC 1.0 and pure noise gives ~0.5.

## Part 3: R1 zero-shot, float32 head (SPEC §7 R1; M3 "Scorer agreement")

The prompt is frozen in `testbed/models/prompt.py` (`child_rows`); it asks for `RISK_LEVEL: <n>`.
- `mlx_child.py`: a `zeroshot` command. Per row: one prefill with a cache, then each level
  candidate (`candidate_sets(tokenizer)["level"]`, the 10 terminated continuations
  `RISK_LEVEL: n<|im_end|>`) on a cloned cache, as `cached_logprobs` does, **but with the output
  head computed in float32**: cast the hidden states to float32 before `head(...)` and assert the
  logits' dtype is float32. Write per row: `id`, `n_prompt`, the 10 log-probs, `mass` (sum of
  the 10 probabilities), the normalized distribution, `expected` (sum n * p_n over the
  normalized 10) and `argmax`. Score = `expected`. Keep `cached_logprobs` / `cmd_score`
  unchanged (M3 used them); add a float32 variant instead (e.g. a `head_dtype` argument
  defaulting to the current behaviour). Option `--check-full` also runs the plain full forward
  pass per candidate with the float32 head and records per-row `max_abs_diff` over candidates
  whose normalized p >= 1e-3, and `expected_full`.
- Extend `selftest-tiny` (or add a test path) so the float32-head cached and full paths are
  checked on the tiny model (agreement within 1e-5 nats), and that float32-head logits are
  float32.
- `local_llm.py`: a typed `zeroshot(model_key, rows, check_full=False, **kw)` wrapper checking
  ids, finiteness, 10 levels per row.
- `testbed/models/r1_run.py`: `python -m testbed.models.r1_run --world W --model 9b --out DIR
  [--limit N] [--check-full]`: renders only, `prompt.child_rows`, the same length check as
  `r2_embed.check_fits`, chunked and resumable like `r2_embed` (reuse its helpers by import;
  don't edit it); writes `<out>/<model>/<world>.jsonl` (one row per unit) and `.json`
  (provenance: prompt sha256, model revision, world checksum, head dtype "float32", agreement
  summary if `--check-full`). Label-free.
- `testbed/eval/r1.py`: `load_r1(r1_dir, model, world_dir, units) -> NDArray` aligned by unit
  id with provenance checks like `load_features`; and a CLI
  `python -m testbed.eval.r1 --worlds W [W ...] --r1 DIR --model 9b --out DIR` writing
  `r1.json` = `r0.evaluate(...)` with the rung `r1_9b` (no fitting: same scores in both
  regimes, like the detector) and differences against F64 logistic and trees. Tests on synthetic
  data only.
- **Agreement check (real model, allowed):** run `r1_run --check-full --limit 10` on
  `runs/m4-tune-s0.75-seed301` for `9b` into `runs/m4-r1-dev/agree/`, and report: max
  `max_abs_diff`, max |expected - expected_full|, the number of distinct `expected` values in
  the 10 rows, and whether any normalized p sits on a bf16 grid (log-odds differences that are
  exact multiples of 0.125). Then a <= 20-unit smoke without `--check-full` into
  `runs/m4-r1-dev/smoke/` with its tok/s and seconds per unit. Report numbers; don't tune
  anything.

## Acceptance (the coordinator re-runs all of these)

- `make test`, `make lint`, `make typecheck` and `.venv/bin/python -m ruff format --check
  testbed tests` all exit 0; quote their summary lines.
- Part 1's r0.json sha256 equality.
- The agreement numbers above.
- In `worker_done`: the list of files changed, and anything in this spec you could not do or
  resolved locally (state each local resolution).


### task_c9f8fa979fc6

# M4 follow-up task: Nimble zero-shot runner (R1's second model; reported, not gating)

Repo: /Users/alkzoupas/code/timeline-testbed (current worktree). Same hard rules as your previous
task: edit only the files listed below; no commits; no STATUS/DECISIONS/prereg/reports edits; no
nested workers; no package installs; **no AUC or report on any real world** (the coordinator
scores). The embedding chain is still using the GPU: real-model runs are limited to the
count-only pass and a <= 20-unit smoke on `runs/m4-tune-s0.75-seed301`, under `runs/m4-nimble-dev/`.

## Goal

SPEC §7 R1 names "Nimble (typed score)". The predecessor's runner is
`vendor/entitlement-shield/scripts/p6_3_nimble_run.py` (read it fully; `vendor/` is read-only, so
copy what you need). Run Bespoke-Nimble-9B (merged model, the runner's arm "b" only) over our
`raw-v1` renders with the coordinator-frozen schema `testbed.models.prompt.NIMBLE_SCHEMA`
(sha256 pinned as `NIMBLE_SCHEMA_SHA256`; do not edit `prompt.py`).

## Files you may edit

- `testbed/models/nimble_child.py` (new): runs in `~/code/nimble/.venv-mlx/bin/python -I`, imports
  nothing from `testbed`, like `mlx_child.py`. Pins exactly as the vendored runner: the Nimble
  repo at `~/code/nimble` HEAD `4edb0703a2e75d0170c3a45645a1961f4d1a2867`, the adapter revision
  `bd792f44…` and `adapter_sha256` (via the merged model's `READY.json` and the adapter
  snapshot), the contract from the adapter's `schema_config.json` (`validate_contract`). Commands:
  `count` (the prepared prompt token count per row, no model loaded) and `score` (the repo's
  `ParallelScorer.evaluate`, mode `parallel`; softmax at T = 1.0 over the 10 choices; per row the
  10 probabilities, `expected` = sum n * p_n, `argmax`, token count, seconds). The context is the
  render text alone, at full length; a row that doesn't fit `MAX_INPUT_TOKENS` fails closed.
- `testbed/models/nimble_run.py` (new, `.venv` side): like `testbed/models/r1_run.py` (reuse its
  and `r2_embed`'s helpers by import, don't edit them): renders only, chunked and resumable,
  child out of process with `HF_HUB_OFFLINE=1`; writes `<out>/nimble/<world>.jsonl` (rows with
  `id`, `n_prompt`, `p`, `expected`, `argmax`) and `<world>.json` (provenance:
  `nimble_schema_sha256`, adapter revision and sha256, Nimble repo commit, world checksum,
  token stats, throughput, `limit`). `--count-only` writes only the token counts.
- `testbed/eval/r1.py`: `load_r1` / CLI accept `--model nimble`, checking `nimble_schema_sha256`
  == `prompt.NIMBLE_SCHEMA_SHA256` and the pinned adapter revision instead of the prompt hash and
  `local_llm.model_revision`; rung name `r1_nimble`. Keep the 9B path unchanged.
- `tests/test_nimble_run.py` (new), `tests/test_r1.py`: synthetic tests (fake child), every
  provenance refusal, alignment, resume.

## Real-model steps (allowed) and what to report

1. `--count-only` over all 1,200 tuning-world renders: max, mean and p90 prompt tokens. SPEC §6.2
   caps a rung's input at 2,048 tokens including the prompt; report how many exceed 2,048 (don't
   truncate or change anything; the coordinator decides).
2. A 20-unit smoke `score` into `runs/m4-nimble-dev/smoke/`: seconds per unit, peak memory, and
   the number of distinct `expected` values. No AUC.

## Acceptance (the coordinator re-runs these)

`make test`, `make lint`, `make typecheck`, `ruff format --check testbed tests` all exit 0 (quote
the summary lines); the two real-model numbers above; files changed; local resolutions stated.

