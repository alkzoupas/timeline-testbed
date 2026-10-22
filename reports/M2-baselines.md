# M2: R0 baselines across the sweep; gate G1 (2026-10-21)

SPEC §10 M2, §7.1 G1, as pre-registered in `prereg/M2.md` before any R0 score. Generator `tt-gen-v1`.
**G1 passed** (`runs/m2-g1/g1.json`, sha256 `083fd7a8…`; `python -m testbed.eval.g1` exit 0, scored once).
Built by a supervised Orca worker (Claude Sonnet 5.5 at high, per `launch.effective`; `r0.py`,
`figure.py`, metric helpers); the coordinator wrote the pre-registration and the gate (`g1.py`) and
reviewed the diff.

## G1

| Row (prereg) | Result |
|---|---|
| (a) health | All ten sweep worlds (s ∈ {0, .25, .5, .75, 1} × seeds 1, 2) pass every §5.6 row; generator-only sample (seed 201) passes |
| (b) oracle ≥ 0.90 at s = 1, pooled | 0.981, 0.978, 1.0, 0.965, 1.0, 0.943 (pairs 1–6) |
| (c) R0 null rule at s = 0, T = 62 | all pass at 0.05 / 62; smallest p 0.042 (F64 trees, in-family, pooled) |
| (d) channel tags | channel check re-run (`runs/m2-channel/`): measured = frozen (1 M, 5 F, 2/3/4/6 T) |

Groups per *s* (two worlds): 480 abusive / 1,920 legitimate; T families 320 / 1,280. Smoke test at full
size (pooled s = 1, R0 route end to end) passed before scoring. All worlds: 1,200 units, 0 mixed, max
render 1,699 tokens.

## Primary metric (SPEC §6.4): T-family AUC, mean of sides A and B, [95% group bootstrap]

Held-out regime (figure: `reports/M2-auc-vs-s.svg`; full tables: `runs/m2-r0/curves.md`):

| Rung | s = 0 | s = 0.25 | s = 0.5 | s = 0.75 | s = 1 |
|---|---|---|---|---|---|
| detector | 0.497 [0.46, 0.53] | 0.497 [0.46, 0.53] | 0.497 [0.46, 0.53] | 0.497 [0.46, 0.53] | 0.497 [0.46, 0.53] |
| F12 logistic | 0.492 [0.46, 0.53] | 0.487 [0.45, 0.52] | 0.492 [0.46, 0.53] | 0.488 [0.45, 0.52] | 0.490 [0.45, 0.52] |
| F64-w90 logistic | 0.468 [0.43, 0.50] | 0.467 [0.43, 0.50] | 0.465 [0.43, 0.50] | 0.462 [0.43, 0.50] | 0.455 [0.42, 0.49] |
| F64-w90 trees | 0.524 [0.49, 0.56] | 0.527 [0.49, 0.56] | 0.534 [0.50, 0.57] | 0.517 [0.48, 0.55] | 0.511 [0.47, 0.55] |
| oracle bound (mean of families) | 0.49 | 0.67 | 0.80 | 0.90 | 0.97 |

In-family, the same picture: every rung 0.475–0.541; F64 trees' interval clears 0.5 only at s = 1
([0.51, 0.58]; pair 3's volume spread, as at M1). Random is 0.497 [0.46, 0.53].

Per family at s = 0.75 (AUC, family against its twin):

| s = 0.75 | 1 (M) | 2 (T) | 3 (T) | 4 (T) | 5 (F) | 6 (T) |
|---|---|---|---|---|---|---|
| detector | 0.58 | 0.47 | 0.47 | 0.52 | 0.55 | 0.52 |
| F12 logistic, in-family / held-out | 0.65 / 0.61 | 0.47 / 0.48 | 0.51 / 0.47 | 0.53 / 0.53 | 0.82 / 0.84 | 0.47 / 0.49 |
| F64-w90 logistic, in-family / held-out | 0.86 / 0.74 | 0.53 / 0.46 | 0.47 / 0.42 | 0.48 / 0.50 | 0.89 / 0.70 | 0.48 / 0.47 |
| F64-w90 trees, in-family / held-out | 0.94 / 0.65 | 0.54 / 0.53 | 0.58 / 0.54 | 0.50 / 0.50 | 0.91 / 0.63 | 0.45 / 0.50 |
| oracle bound | 0.91 | 0.85 | 0.99 | 0.90 | 0.96 | 0.86 |

## What it shows

- The testbed does what it was built for. The T evidence is present (oracle 0.67 → 0.97 over *s*),
  and no R0 rung sees it at any *s*; F64 does see the F and M families (pairs 1 and 5).
- At s = 0.75 the G2 bar is low: held-out F64 logistic 0.462, trees 0.517. Trees − logistic, paired:
  +0.055 [0.011, 0.099]. Stronger F64 − detector: +0.020 [−0.029, 0.070].
- The oracle bound is under 0.90 for pairs 2 and 6 at s = 0.75 (0.85, 0.86). That's reported, not a
  gate; the bound for the primary is 0.90.
- Recall at FPR 5% on the held-out primary at s = 0.75 is ≤ 0.053 for every rung.

## Caveat: the sweep is paired across *s*

A world's base draws depend on (version, seed, family, index), not on *s* (`generator.py:224,246`),
so worlds with the same seed differ only in the mechanism. Random and detector scores are therefore
identical at every *s*. Any chance fluctuation also repeats at every *s*. Held-out F64 logistic is
below chance on the T families: 0.468 at s = 0, where its interval [0.433, 0.501] contains 0.5. Its
upper bound then falls just under 0.5 at s = 0.5, 0.75 and 1 (0.499, 0.497, 0.490). Most of that is
the shared draws (the s = 0 value is almost the same); the mechanism may add a small negative transfer
on top. A below-chance baseline doesn't let R2 pass G2 at chance, because G2 also needs R2 to beat
trees (0.517). Read each *s* on its own interval, and don't treat the five points as independent
replicates.

## Checks

`make test` `383 passed in 145.03s (0:02:25)`; `make lint` `All checks passed!`; `make typecheck`
`Success: no issues found in 45 source files`; `ruff format --check` `45 files already formatted`.
Each exited 0. `r0` on the sweep: 58.1 s, exit 0 (`runs/m2-r0/r0.json` sha256 `4f3f2ddc…`).

## Carried to M3 / M4

- R2 needs its own tuning world (SPEC §7: s = 0.75, its own seed; no gate scores it). Not generated yet.
- The G2 scorer should reuse `r0.evaluate`'s primary and paired-bootstrap code, and gets the M4
  cross-vendor review.
- From M1: the boundary scan reads ordinary spellings, not constructed strings or reflection.
