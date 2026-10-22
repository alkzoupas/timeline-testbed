# M4: R2 probe; gate G2 (2026-10-22)

SPEC §7.1 G2, as pre-registered in `prereg/M4.md` (frozen choices, amendments and dry run all dated
before scoring). Generator `tt-gen-v1`, render `raw-v1`, prompt `m4-v1`. **G2 failed**
(`runs/m4-g2/g2.json`, sha256 `b23f922c…`; `python -m testbed.eval.g2` exit 0, 23 s, scored once).
Per SPEC §7.1 the LLM line stops: no M5 (R3, G3), no M6; M7 writes the result up. R1 can't rescue it.

## G2

R2 = StandardScaler + L2 logistic on `mean_16` (layer 16, mean over the render span), frozen on the
tuning world (seed 301): 9B C = 10, 4B C = 0.001. Pools: 2,400 units per *s* (two worlds, 0 mixed);
T families 320 abusive / 1,280 legitimate groups. Paired bootstrap, 2,000 resamples, seed 20261033.
F64 reproduced M2 exactly (0.462 / 0.517 at *s* = 0.75); 0 lbfgs fits hit `max_iter`.

| Row (prereg) | 9B | 4B |
|---|---|---|
| R2 primary, held-out, *s* = 0.75 | 0.453 | 0.486 |
| (a) R2 − F64 logistic: point, bound (1.25th pct) | −0.009, −0.065 | +0.024, −0.031 |
| (a) R2 − F64 trees: point, bound (1.25th pct) | −0.064, −0.121 | −0.031, −0.085 |
| (a) met (both bounds > 0) | no | no |
| (b) R2 at *s* = 0, [95%] contains 0.5 | 0.472 [0.439, 0.507], yes | 0.469 [0.434, 0.505], yes |

Verdict: no model meets (a), so **FAIL**. Both meet (b): the pipeline shows no signal at *s* = 0.

## Primary metric (SPEC §6.4): T-family AUC, mean of sides A and B, [95%]

| Held-out | s = 0 | s = 0.75 |
|---|---|---|
| detector | 0.497 [0.46, 0.53] | 0.497 [0.46, 0.53] |
| F64-w90 logistic | 0.468 [0.43, 0.50] | 0.462 [0.43, 0.50] |
| F64-w90 trees | 0.524 [0.49, 0.56] | 0.517 [0.48, 0.55] |
| R2 9B | 0.472 [0.44, 0.51] | 0.453 [0.42, 0.49] |
| R2 4B | 0.469 [0.43, 0.50] | 0.486 [0.45, 0.52] |
| oracle bound | 0.49 | 0.90 |

Held-out recall at FPR 5%, *s* = 0.75: 9B 0.050, 4B 0.044 (random 0.050). At *s* = 0.75 the 9B is
below F64 trees: −0.064 [−0.114, −0.012].

## Reported, not gating: the in-family regime

Trained on the same family (R0's in-family folds, same frozen feature and C, not tuned for this):

| In-family, T families | primary *s* = 0 | primary *s* = 0.75 | pair 2 | pair 3 | pair 4 | pair 6 |
|---|---|---|---|---|---|---|
| F64-w90 trees | 0.533 [0.50, 0.57] | 0.517 [0.48, 0.55] | 0.54 | 0.58 | 0.50 | 0.45 |
| R2 9B | 0.462 [0.43, 0.50] | 0.725 [0.69, 0.76] | 0.56 | 0.94 | 0.47 | 0.82 |
| R2 4B | 0.458 [0.42, 0.49] | 0.706 [0.67, 0.74] | 0.55 | 0.94 | 0.40 | 0.85 |
| oracle bound (*s* = 0.75) | | 0.90 | 0.85 | 0.99 | 0.90 | 0.86 |

Pair columns are at *s* = 0.75. R2 − F64 trees in-family at *s* = 0.75: 9B +0.208 [0.16, 0.25],
4B +0.189 [0.14, 0.23]. Recall at FPR 5%: 9B 0.29, 4B 0.35 (trees 0.05). On the M and F families
in-family, R2 matches F64 trees (pair 1 0.94 vs 0.94; pair 5 0.88 vs 0.91).

## What it shows

- The embeddings carry timeline-only evidence for two of the four T families. Trained on that
  family, R2 reaches 0.94 on pair 3 and 0.82–0.85 on pair 6, near their oracles (0.99, 0.86), while
  F64 trees stays near chance (0.58, 0.45). The *s* = 0 control is at chance in-family as well, so
  this isn't a leak.
- That evidence doesn't transfer. Trained on the other side's families, R2 is near or below chance on
  every T family (held-out 0.41–0.55), and the 9B is worse than F64 trees. SPEC §2's claim is about
  held-out families, so G2 fails on the question it was built to ask.
- Pairs 2 and 4 aren't seen even in-family (0.40–0.56, oracles 0.85 and 0.90).
- Below-chance values at *s* = 0 (R2 in-family 4B, upper bound 0.492; F64 logistic held-out 0.468)
  repeat M2's caveat: the sweep is paired across *s*, so chance structure repeats. A signal would sit above 0.5.
- The in-family result is reported only. These are development worlds already seen, so it can't
  become a claim without a new pre-registration and fresh worlds (a pivot, DECISIONS first).

## *s* = 0.5 (secondary, run after the FAIL with the owner's go)

`r2 score`, same frozen choices, on `m2-dev-s0.5-seed1` + `seed2` (`runs/m4-r2-s0.5/r2.json`, sha256
`00ac6cb8…`; exit 0, 11 s; F64 reproduces M2's *s* = 0.5 column). Held-out: 9B 0.442 [0.41, 0.48],
4B 0.470 [0.43, 0.51]; F64 logistic 0.465, trees 0.534; R2 − trees 9B −0.093 [−0.14, −0.04], 4B
−0.065 [−0.12, −0.01]. In-family: 9B 0.625 [0.59, 0.66], 4B 0.612 [0.58, 0.65], with pair 3 at
0.75–0.79 and pair 6 at 0.67–0.76 (oracles 0.98, 0.76). Weaker in-family than at *s* = 0.75, still
no held-out transfer.

R1 (9B zero-shot) and Nimble were not run (the owner's call, 2026-10-22): no decision depends on them
after the FAIL, and they would take hours of GPU time.

## Checks

The code is unchanged from `df1e99a` (tree clean before the run). Re-run after scoring: `make test`
`617 passed in 144.00s (0:02:24)`; `make lint` `All checks passed!`; `make typecheck` `Success: no
issues found in 63 source files`; `ruff format --check` `63 files already formatted`; exit 0 each. Before the
run, I checked `mlx_child.py` sha256 `ae565057…6980`, the prompt hash `0cfe33a7…` and both `tune.json`
sha256 against the frozen values. Outputs: `runs/m4-g2/{g2,report}.json` (`report.json` sha256
`2bec71b0…`), `scores-s0.0.jsonl`, `scores-s0.75.jsonl`; log `runs/m4-logs/g2.txt`.
