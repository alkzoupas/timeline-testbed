# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-21)

- **M0, M1 done** and committed (`f40cdd1`; `reports/M0.md`, `reports/M1.md`). Generator v1 frozen
  (hash in DECISIONS.md); channel tags frozen as `MEASURED_TAGS` (pair 1 M, pair 5 F, 2/3/4/6 T).
- **M2 done; G1 passed** (`reports/M2-baselines.md`, `274bfcf`). Held-out primary (T families) at
  s = 0.75: detector 0.497, F12 0.488, F64 logistic 0.462, F64 trees 0.517; oracle 0.90.
- **M3 done** (`reports/M3-feasibility.md`, `eb9cb5e`). R2 forward pass 9B 0.86 h / 4B 0.42 h per
  world; R3 on the 4B (one epoch fits 12 h); score R1/R3 with a float32 head (bf16 logit steps).
- **M4 in progress** (R1, R2, gate G2). Done so far:
  - Tuning world `runs/m4-tune-s0.75-seed301` (seed 301, s = 0.75): health passes, max render 1,632.
  - Prompt frozen: `testbed/models/prompt.py` (`m4-v1`, sha256 `0cfe33a7…`), 165 tokens overhead,
    longest prompt 1,864 + 8 answer tokens.
  - Embed driver `testbed/models/r2_embed.py` (layers 16 and final; chunked, resumable). Chain:
    tune, s0.75 ×2, s0 ×2, s0.5 ×2 on 9b then 4b (`runs/m4-logs/embed-chain.log`, outputs
    `runs/m4-embed/<model>/<world>.npz`). 21:41 the 9B tune child was SIGKILLed (exit -9) while
    the Nimble smoke shared memory; the machine rebooted ~21:49 (cause not found; no panic
    report). Restarted 22:22, strictly one model on the GPU at a time; 9B tune chunks 0–2 reused.
  - `prereg/M4.md` (selection rule, G2 rows (a)/(b), PASS/LEAK/FAIL) and `testbed/eval/g2.py`,
    `tests/test_g2.py`; R2 choices frozen there and in `g2.FROZEN` / `TUNE_SHA256`.
  - Orca run `run_18b9ec4140c4`, both tasks completed (Sonnet 5.5 at high, effective):
    `task_173b86534537` (r0 refactor, `testbed/eval/{r2,r1}.py`, R1 float32 head, `r1_run.py`;
    accepted, `make test` 502 passed, r0.json reproduces M2) and `task_c9f8fa979fc6` (Nimble R1
    runner: `nimble_{child,run}.py`, `r1.py --model nimble`; diff reviewed, numbers re-read from
    `runs/m4-nimble-dev/`: tuning renders max 2,005 / mean 1,314 / p90 1,705 tokens, 0 over
    2,048; smoke 6.5 s/unit, peak 18.0 GiB, 20 distinct of 20).
  - Fixed (coordinator): `r2.load_features` and `r1.load_r1` built paths with `with_suffix`,
    which cuts dotted world names (`…s0.75-seed1` → `…s0.json`; failed closed); regression tests
    with dotted names in `tests/test_r2.py`, `tests/test_r1.py`.
  - `r2 tune` (`runs/m4-tune/<model>/`): 9B `mean_16` C = 10, held-out 0.526 (grid 0.461–0.526,
    in-family ≤ 0.730); 4B `mean_16` C = 0.001, held-out 0.546 (grid 0.462–0.546, in-family
    ≤ 0.727). Near chance held-out on the tuning world; frozen anyway (prereg amendment).
  - Review round 2 (review point 2; Codex GPT-6.1 Sol at high, effective): FIX-THEN-SHIP, one P2
    (`reviews/round-2/TRIAGE.md`): R2 embed resume reused chunks unbound to their rows or child
    code. Live: the 9B tune reused chunks made while `prompt.py`/`mlx_child.py` were edited.
  - Fix accepted (Sonnet worker `ctx_24b952fe7cf2`): chunks and sidecars bound to rows, child and
    npz sha256; loaders refuse mismatches; same for R1 and Nimble. `mlx_child.py` sha256
    `ae565057…6980`: **any edit to it voids every embedding.** Old embeddings kept in
    `runs/m4-embed-superseded/`. Chain restarted 23:34 (tune worlds first).
  - Fix check (Codex, effective gpt-6.1-sol at high): PARTIAL, two P2 left (a chunk's npz not
    bound to its json; Nimble load ignores `rows_sha256`). Both fixed parent-side by worker
    `ctx_af13d713627d` (Sonnet 5.5 at high, effective) and accepted; `mlx_child.py` untouched.
  - Tune embeddings recomputed under the fix: bit-identical to the superseded ones; `r2 tune`
    re-run, same selections; `tune.json` hashes re-frozen (9B `9476e5df…`, 4B `e66035cb…`).
    Round 2 closed. 2026-10-22: all eight gate embeddings done by 05:48 (child `ae565057…`);
    s0.5 ×2 done; `CHAIN END 08:18:45`, 14 of 14, no FAIL. `g2 --dry-run` exit 0, in prereg.
- **Next:** score G2 once (awaiting the owner's go after the 2026-10-22 review); R2 s = 0.5; R1
  9B and Nimble (reported); `reports/M4-probe.md`.
- Worlds from before the M1 manifest fix sit in `runs/superseded-round1/`; safe to delete.

## Open owner decisions

- None blocking. FYI (coordinator calls in `prereg/M4.md`, owner may override before scoring):
  G2 counts a pass on either model (9B or 4B) with a Bonferroni-adjusted bound (1.25th percentile);
  SPEC's "nested CV" is realized as selection on the separate tuning world.
- FYI: an untracked `opencode.json` (graft's MCP config) is in the repo root; not ours, left in place.

## Files changed in M4 (uncommitted)

- Coordinator: `testbed/models/{prompt,r2_embed}.py`, `tests/test_r2_embed.py`, `prereg/M4.md`,
  `testbed/eval/g2.py`, `tests/test_g2.py`, `reviews/round-2/`; DECISIONS, STATUS. Workers
  (accepted): `testbed/eval/{r0,r2,r1}.py`, `testbed/models/{mlx_child,local_llm,r1_run,
  nimble_child,nimble_run}.py`, their tests. The round-2 fix worker edits a subset of these.

## Last checks (2026-10-22)

- 2026-10-22 07:30, before G2: `make test` `617 passed in 160.97s (0:02:40)`, exit 0; lint,
  typecheck (63 files), format clean, exit 0 each.
- M4 after the re-freeze: `make test` `617 passed in 141.38s (0:02:21)`, exit 0;
  `make lint` `All checks passed!`, exit 0; `make typecheck` `Success: no issues found in 63
  source files`, exit 0; `ruff format --check testbed tests` `63 files already
  formatted`, exit 0.
- M3 acceptance: `make test` `420 passed in 140.36s (0:02:20)`; lint, typecheck, format clean.
