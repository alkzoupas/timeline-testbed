# Prior results and environment (from EntitlementShield, as of 2026-10-21)

Sources are paths in `~/code/entitlement-shield`. Every number is open-scope and exploratory; none of
them is a claim.

## What the predecessor measured

| Item | Result | Source |
|---|---|---|
| Dev world (its own generator), tuning split | P = R = 1.0, FPR 0. Circular | `docs/FINDINGS.md` §1 |
| Tier 4 (GPT-authored; 200 clusters; 22 abusive source groups): the detector at threshold 0.60 | Recall 0.70 (142 of 202 units); 643 of 1,051 legitimate units flagged; precision 0.18 against a 0.16 base rate | `docs/p6.0-recall-headroom.md` |
| Tier 4 misses | 60 units from 4 source groups, 53 of them from 2. 34 are single-account fragments | `docs/p6.0-recall-headroom.md`, `docs/p6.1c-fp-anatomy.md` |
| Prompted Qwen 3.5 9B on 492 allowed units | AUC 0.38. The risk level tracks the number of accounts in view; AUC within strata 0.47–0.55 | `docs/p6.1-pilot-results.md`, `docs/p6.1c-fp-anatomy.md` |
| Nimble-9B zero-shot, allowed multi-account stratum | AUC 0.580 (base) and 0.495 (with the adapter) | `docs/p6.3-nimble-probe.md` |
| Supervised ceiling on Tier 4: 64 features, source-group CV | Logistic 0.670, trees 0.672, detector 0.580. In the allowed multi-account pool: logistic 0.789, against Nimble's 0.50–0.58 | `docs/p6.4-learnability-ceiling.md` |
| Tier 5 attempts | Gemini (no executed files); codex-01 (a perfect shortcut, then a ceiling of 0.586); codex-02 (one uniform label shift) | `docs/tier5-intake-*.md` |

## Local environment (recorded 2026-10-16; paths re-checked 2026-10-21; re-verify in M0 and M3)

**Machine:** Apple Silicon Mac with MLX. Disk had about 450 GiB free on 2026-10-16. Check memory with
`sysctl hw.memsize`.

**Present (checked 2026-10-21):**
- `mlx-community/Qwen3.5-9B-MLX-4bit` in the Hugging Face cache (revision `938d891` per the old
  STATUS).
- The merged Nimble model at
  `~/code/nimble/.cache/models/nimble-9b-bd792f44ec8e265be861bfcdf4e05967ffe0e858/` (base
  `Qwen/Qwen3.5-9B` @ `c202236` plus the adapter `bespokelabs/Bespoke-Nimble-9B` @ `bd792f4`).
- The training venv `~/venvs/es-p6-train`: mlx 0.32.3, mlx-lm 0.32.0, mlx-lm-lora 5.8.6.
- The Nimble venvs: `~/code/nimble/.venv-mlx` (mlx 0.32.2, mlx-lm 0.31.3, transformers 5.17.0) and
  `~/code/nimble/.cache/venvs/nimble` (torch, transformers, peft, for merging). The Nimble repo is at
  `~/code/nimble` @ `4edb070`.

**MTPLX:** a local OpenAI-compatible server.
- `mtplx-qwen35-9b-optimized-speed` worked for the P6.1 pilot.
- A 27B model returned HTTP 507 (out of memory) on most samples under load.

**Measured (P6.2 Step 0, `docs/p6.2-step0-feasibility.md`):**
- LoRA steps at 1k tokens (37 tok/s) and 2k tokens (≈ 20 tok/s, 34 GiB); failure at 4k tokens and
  above.
- Scoring an 18k-token prompt took 56 s.
- "10" tokenizes as "1" and "0".
- The cached scorer differs from a full forward pass by 0.12 nats at level 10 (open).

**Nimble (P6.3):** an 8,192-token hard cap; trained on prompts of 2,048 tokens or fewer; about 3 s and
about 13 s per unit at 1.2k and 5.9k tokens; peak memory 20.9 GiB.

**Policy:** no paid APIs and no rentals. Downloads and installs need the owner's go.
