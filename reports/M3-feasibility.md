# M3: feasibility on this machine (2026-10-21)

**Machine:** Apple M4 Pro, 48 GiB; mlx 0.32.3, mlx-lm 0.32.0 (`~/venvs/es-p6-train`, read-only).
**Models:**
- `9b`: `mlx-community/Qwen3.5-9B-MLX-4bit` @ `938d891`; 32 layers, hidden size 4,096.
- `4b`: `mlx-community/Qwen3.5-4B-MLX-4bit` @ `32f3e8e`; 32 layers, hidden size 2,560. Downloaded
  with the owner's go.

**Run:** `python -m testbed.models.feasibility --world runs/m2-dev-s0.75-seed2 --model {9b,4b}`, exit
0 each. Outputs are in `runs/m3-feasibility/{9b,4b}/summary.{json,md}`. Code:
`testbed/models/{mlx_child,local_llm,feasibility}.py`.

**Inputs and limits:**
- Renders only; no labels are read.
- The prompt is a DRAFT; M4 freezes the real one.
- The LoRA bench trains on a fixed dummy answer ("A") and saves no adapter.
- Batch size 1, except where marked. Nothing was tuned for speed.

## Results

| Measure | 9B | 4B |
|---|---|---|
| Render tokens, 10 sweep worlds (9B tokenizer; n = 12,000) | mean 943, p90 1,338, max 1,699 | same tokenizer: identical counts and spans on all 1,200 units of this world |
| Render plus draft prompt (this world) | mean 1,006, p90 1,407, max 1,761; 0 of 1,200 over 2,048 | identical |
| Forward pass (R2 extractor; 100 units after one warm-up) | **390 tok/s**, peak 5.6 GiB | **806 tok/s**, peak 3.2 GiB |
| R2 embeds, one world (1.21M tokens) | **0.86 h** | **0.42 h** |
| LoRA step at median length (1,006 tokens) | 37.8 tok/s, 26.7 s/step, 15.3 GiB | 43.1 tok/s, 23.4 s/step, 12.5 GiB |
| LoRA step at p90 length (1,407 tokens) | 31.9 tok/s, 20.9 GiB | 36.2 tok/s, 18.1 GiB |
| LoRA step at the longest prompt (1,761 tokens) | 18.0 tok/s, 26.9 GiB | 29.6 tok/s, 24.0 GiB |
| LoRA step at the median, batch size 4 | 20.5 tok/s, 44.1 GiB (slower) | 17.9 tok/s, 40.8 GiB (slower) |
| LoRA epoch per held-out fold (600 units × 2 pooled worlds, 1.21M tokens) | **8.9 h**; 10.5 h at the p90 rate | **7.8 h**; 9.3 h at the p90 rate |
| Scoring one unit, cached (one prefill plus 12 candidate branches) | 3.5 s | 1.8 s |

Training-path checks: every step's loss and gradients were finite. LoRA B starts at zero, so 248 of
496 gradient tensors are non-zero at step 0; all 496 are non-zero from step 1 on.

## Scorer agreement (SPEC §11's open issue)

On 10 units per model, the cached scorer (one prefill, then a copied cache per candidate) was
compared with a plain full forward pass of `prompt + candidate`.

- **A/B:** 18 of 20 rows agree within 4e-4 nats. One row per model differs by 0.06 nats (9B) and
  0.09 nats (4B).
- **Levels 1–10:** they differ by up to 0.71 nats. Every gap of that size is on a candidate with a
  probability of about 1e-18, because this prompt asks for A or B. These are tail numbers; no score
  depends on them.
- **Cause: the logits are rounded to bf16.** Every P(A | A or B) is a sigmoid of an exact multiple
  of 0.125:
  - 9B: 0.269, 0.348, 0.407, 0.438 and 0.5 (gaps −1.0, −0.625, −0.375, −0.25 and 0);
  - 4B: 0.5 up to 0.778, gaps 0 to 1.25 in steps of 0.125.

  At logit magnitudes of 16–32, bf16's step is 0.125. The two paths process inputs of different
  lengths and land on neighbouring bf16 values, and half a step is about 0.06 nats.
- On the tiny random model in float32, the two paths agree within 2e-6 nats, so the cache copy is
  correct. The predecessor's 0.12-nat gap was very likely the same effect.

**Consequence:**
- A score taken from bf16 logits is coarse: P(A) moves in steps of 0.125 in log-odds, which creates
  many ties and blunts the AUC of R1 and R3.
- **Action (M4/M5):** compute the output head in float32 for scoring, then repeat this agreement
  check before any scored run. R2 is unaffected: it reads hidden states, not logits.

## Decisions and projections

- **Probe (R2):** both models, as the owner decided.
  - 5 worlds (the tuning world, two worlds at *s* = 0.75 and two at *s* = 0): about 4.3 h on the 9B
    plus 2.1 h on the 4B.
  - All 11 worlds (adding the other *s*): about 9.5 h plus 4.6 h.
- **R3 model: the 4B.** Its LoRA step is only 1.1–1.6× faster than the 9B's, not "several times".
  - The cost of a step doesn't scale with parameter count. Most likely it's set by the gated-delta
    recurrence and gradient checkpointing (the predecessor's §3), which isn't measured here.
  - The 4B uses about 3 GiB less at every length (2.8–2.9 GiB), and its R2 and scoring passes run
    twice as fast.
- **R3 budget against SPEC §7's ≤ 12 h per run:**
  - One epoch over both pooled worlds fits on either model: 7.8–9.3 h on the 4B.
  - Two epochs don't fit (15.6–18.6 h on the 4B).
  - Options for `prereg/M5.md`: one epoch; train on one world (about 3.9 h per epoch); or LoRA on the
    top layers only. The last option is unmeasured; measure it in M5's first step before choosing.
- **Batch size 1.** Batch 4 at equal lengths was slower and used 41–44 GiB.
- **R1 (9B, one prefill per unit):** about 0.9 h per world, about 1.8 h for SPEC's two worlds.
  Nimble wasn't measured.

## Limits

- One world, one machine and one MLX version. The LoRA rate comes from 3 steps per length bucket
  after one warm-up.
- The epoch projection applies the median-length rate to every unit, with the p90 rate as the
  pessimistic bound.
