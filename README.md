# timeline-testbed

A synthetic, known-answer testbed for one question: when the only evidence that separates abuse
from legitimate use is in an account group's activity timeline, can a small local LLM find it
where a rules detector and feature-based models can't? The testbed is a world generator that
creates six abuse/legitimate twin pairs. In each pair, one operator runs several accounts
(abusive), and its twin is several real people sharing resources (legitimate). Each pair's
mechanism is declared in advance, along with its strength *s* and the channel its evidence
lives in: F (visible to the detector's 12 features), T (timeline-only) or M (mixed). The answer
is mostly no. On held-out timeline-only families, a linear probe on Qwen 3.5 9B and 4B adds
nothing over a 64-feature model. The one confirmed exception is a hosted model: Claude Haiku 5.5,
few-shot, ranks pair 3 (the "template operator") at AUC 0.758 [0.679, 0.832] on fresh worlds,
where both 64-feature models are at chance.

The full write-up is one page: [`reports/RESULT.md`](reports/RESULT.md). The pair 3 confirmation is
in [`reports/M9.md`](reports/M9.md).

## Results

AUC with 95% bootstrap intervals, where the reports give them. 0.5 is chance.

**Held-out timeline-only families** (pairs 2, 3, 4, 6), *s* = 0.75. Each model is scored only on
families it wasn't trained on (mean of two sides). Source: `reports/RESULT.md`.

| Model | AUC |
|---|---|
| Oracle bound (the generator's own label-free statistic) | 0.90 |
| Frozen detector (the predecessor's rules) | 0.497 [0.46, 0.53] |
| 64 features (F64), logistic regression | 0.462 |
| 64 features (F64), boosted trees | 0.517 [0.48, 0.55] |
| R2 linear probe on Qwen 3.5 9B (4-bit MLX) | 0.453 [0.42, 0.49] |
| R2 linear probe on Qwen 3.5 4B (4-bit MLX) | 0.486 [0.45, 0.52] |

**Pair 3 only, on three fresh worlds** (M9, pre-registered). Two worlds at *s* = 0.75 hold 80
abusive and 80 legitimate groups. Source: `reports/M9.md`.

| Model | AUC |
|---|---|
| Claude Haiku 5.5, few-shot (gating; examples from pairs 4–6 only) | 0.758 [0.679, 0.832] |
| Claude Haiku 5.5, zero-shot (reported, not gating) | 0.577 [0.485, 0.665] |
| F64 logistic / boosted trees (held-out) | 0.511 [0.425, 0.597] / 0.502 [0.414, 0.596] |
| Control: Haiku few-shot on an *s* = 0 world (no signal; 40 + 40 groups) | 0.571 [0.445, 0.697] |

Few-shot beats F64 logistic by +0.246 [0.129, 0.357] and trees by +0.256 [0.136, 0.364]. An
earlier Haiku pilot (M8) across all four held-out timeline-only families was a pre-registered
NO-GO: zero-shot 0.561 and few-shot 0.553. So Haiku doesn't have a general sense of "one operator";
it reads this one mechanism.

## The template operator and its twin

Pair 3's abusive group is one person running several accounts from a script. Each session in each
account replays the same template: the same two purchase amounts and the same gap between them.
The legitimate twin is a small office team on a shared card. They keep a similar schedule, but each
person's amounts and gaps are their own draws from the same distribution.

Taken alone, each abusive account looks normal. Its amounts, gaps, session times and session counts
come from exactly the twin's distributions, and they stay under every rate limit. The abuse exists
only *between* accounts: the sessions match each other across accounts. The 64 features are counts,
rates, ages and gap statistics, summarised per account and over the group. None of them compares
one account's session with another's, so the match is invisible to them. (The template does widen
the spread of total volume, which F64 could in principle pick up. Held-out, it scores 0.50–0.51.)
A model that reads the rendered timeline can see the match directly.

## Method and integrity

- **Pre-registration.** Every scored experiment has a one-page prereg in [`prereg/`](prereg/),
  with its thresholds fixed before scoring. Each is recorded in DECISIONS.md, with hashes where they
  were frozen.
- **Kill gates.** G1 (`reports/M2-baselines.md`) checked that the testbed is valid. Every world
  passed its health checks, the oracle reached ≥ 0.90 at *s* = 1, every baseline was at chance at
  *s* = 0, and the channel tags held. It passed. G2 (`reports/M4-probe.md`) required the LLM probe
  to beat both F64 models held-out. It failed, so the fine-tuning and confirmation milestones (M5,
  M6) were not run.
- **Sealed worlds.** M6 was to be scored once on sealed `runs/confirm-*` worlds. It never ran, so
  those worlds were never opened (or generated). M8 and M9 were later pivots, each with its own
  prereg, and M9 ran on worlds drawn fresh after its prereg was frozen.
- **Label isolation.** Labels go to a separate store. Only `testbed/eval/` can open it, and a test
  enforces this (`tests/test_ground_truth_boundary.py`).
- **Adversarial reviews.** At two of the three review points, a second model (OpenAI Codex) reviewed
  the code and preregs adversarially. The reports, triage and fixes are in [`reviews/`](reviews/).
  The project was built by AI coding agents under the rules in `AGENTS.md`.
- **Decision log.** [`DECISIONS.md`](DECISIONS.md) has one line per decision, including each pivot
  and why.

## Limits

- The data are synthetic. The mechanisms are declared and the claims are method-level only. Nothing
  here says how real abuse looks or how often it happens.
- One abuse family is confirmed (pair 3, at *s* = 0.75). The *s* = 0 control is small (40 + 40
  groups), so it rules out only a large artifact.
- The positive result comes from a hosted model (Claude Haiku 5.5), not the small local LLM the
  question asks about. The probe's held-out null stands.
- The local-model line used one model series (4-bit MLX), one prompt and one probe, with two worlds
  per *s*. Fine-tuning (R3) and zero-shot on the local models (R1) were never scored.
- Nothing tested here says which features of the rendered timeline carry the pair 3 signal.

## Repository map

| Path | What |
|---|---|
| `SPEC.md` | The full design: the question, the generator, the six families, the model ladder, the gates |
| `reports/` | One page per milestone (M0–M4, M8, M9) and `RESULT.md`, the final write-up |
| `prereg/` | The pre-registrations (M2, M4, M8, M9) |
| `testbed/` | The code: `world/` (generator), `pipeline/` (detect, units, render), `features/`, `models/`, `eval/` |
| `tests/` | pytest suite, including known-answer tests for every reported metric |
| `vendor/entitlement-shield/` | A read-only snapshot of [EntitlementShield](https://github.com/alkzoupas/entitlement-shield) at `fd5deca`; `VENDORED.md` lists what's included |
| `reviews/` | The cross-vendor review rounds: reports, triage, fix diffs |
| `context/` | Background: lessons from the predecessor, its prior results, a reuse map, the M8 kickoff |
| `STATUS.md`, `DECISIONS.md` | The final state and the decision log |
| `AGENTS.md` | The rules the AI agents worked under |

`runs/`, which holds the worlds, label stores and model outputs, is not committed. Review files keep
the machine paths they were written with.

## Reproducing

**Requirements.** macOS on Apple silicon (that's what it was built on), Python 3.14 and
[uv](https://github.com/astral-sh/uv). Generating a world also counts render tokens with the
Qwen 3.5 9B tokenizer. That step runs out of process in a separate venv at `~/venvs/es-p6-train`
(which has `tokenizers`), and reads the cached `tokenizer.json` of
`mlx-community/Qwen3.5-9B-MLX-4bit` from the Hugging Face cache, offline
(`testbed/pipeline/tokens.py`). The R2 probe needs MLX and the models in that same venv
(`testbed/models/mlx_child.py`).

```sh
make venv            # create .venv from requirements.txt with uv
make test            # pytest
make lint            # ruff
make typecheck       # mypy
make vendor-check    # verify vendor/ against VENDORED.sha256
```

**Generate a world and check it.** World names follow `<name>-s<s>-seed<seed>`. Generation is
deterministic: the same seed and *s* give byte-identical files (about 2 minutes on an M4 Pro).

```sh
.venv/bin/python -m testbed.pipeline.run_world --seed 1 --s 0.75 --scale standard \
    --out runs/m1-dev-s0.75-seed1
.venv/bin/python -m testbed.eval.health world runs/m1-dev-s0.75-seed1   # writes health.json
```

**Baselines (R0).** `RESULT.md`'s baseline rows pool two worlds per *s* (seeds 1 and 2, at
*s* ∈ {0, 0.25, 0.5, 0.75, 1}):

```sh
.venv/bin/python -m testbed.eval.r0 --worlds runs/m1-dev-s0.75-seed1 runs/m2-dev-s0.75-seed2 \
    --out runs/r0-s0.75
```

**M9 (Haiku on pair 3).** The few-shot examples come from the tuning world
`runs/m4-tune-s0.75-seed301` (seed 301, *s* = 0.75). The scored worlds use M9's seeds: `1759212321`
and `1841874529` at *s* = 0.75, and `98816701` at *s* = 0. Generate each one, then run health on
it, as above. Then:

```sh
.venv/bin/python -m testbed.eval.haiku_m9 select --mode confirm --runs runs \
    --worlds m9-s0.75-seed1759212321 m9-s0.75-seed1841874529 --control m9-s0-seed98816701 \
    --out runs/m9-confirm
.venv/bin/python -m testbed.models.haiku_run --jobs runs/m9-confirm/jobs.jsonl \
    --out runs/m9-confirm/raw
.venv/bin/python -m testbed.eval.haiku_m9 score --job runs/m9-confirm --raw runs/m9-confirm/raw \
    --runs runs --out runs/m9-confirm/scored
```

`haiku_run` makes the 1,200 Haiku calls through headless Claude Code, not an API key. It needs the
Claude Code CLI, logged in with a Claude subscription under the config dir
`~/.claude-haiku-scorer`. The route and its isolation flags are in `testbed/models/haiku_prompt.py`
and `AGENTS.md`. Haiku's outputs are sampled, so a rerun won't reproduce the scores exactly. `score`
refuses to overwrite an existing `--out`.

## Relation to EntitlementShield

This testbed came out of the postmortem of
[EntitlementShield](https://github.com/alkzoupas/entitlement-shield), an abuse detector for a
fictional free-credit product. For eight weeks it tried to show that an LLM second stage adds
value, and every result was exploratory or null. The main reason was that none of its evaluation
worlds could answer the question (SPEC.md §1, `context/lessons.md`). This project built a world
generator where the answer is known, and reuses the predecessor's frozen detector, features and
label store, which are vendored read-only.

## License

MIT, the same as EntitlementShield. See [`LICENSE`](LICENSE).
