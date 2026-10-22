"""M3 feasibility bench (SPEC §10, §11): render lengths, forward-pass and LoRA throughput.

    python -m testbed.models.feasibility --world runs/m2-dev-s0.75-seed2 --model 9b --out DIR

Four steps, each writing its child JSON into DIR, then `summary.json` and `summary.md`:
`count` over every render, `embed` on a seeded sample plus the longest unit, `score` on 10 units,
and `lora-bench` at three length buckets (batch 1) and at the median bucket (batch 4).

Inputs are renders only (`renders.jsonl`, rows `{"unit", "text"}`). No labels are read, and the
LoRA bench trains on a fixed dummy answer. The prompt below is a DRAFT for M3 only: M4 freezes
the real one.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from testbed.models import local_llm

# DRAFT (M3 only). M4 freezes the real prompt.
DRAFT_BEFORE = (
    "Below is the activity timeline of one cluster of linked accounts on a platform that gives "
    "new accounts free credits.\n\n"
)
DRAFT_AFTER = (
    "\n\nDoes this timeline show one operator abusing the credits through several accounts (A), "
    "or ordinary independent users (B)? Answer with one letter."
)

TOKEN_LIMIT = 2048  # SPEC §6.2: render plus prompt
TRAIN_FRACTION = 0.5  # one held-out fold trains on 3 of 6 pairs: 600 of 1,200 units per world
EMBED_SAMPLE = 100
SCORE_UNITS = 10
BENCH_STEPS, BENCH_WARMUP = 3, 1
BATCH_UNITS = 4


# --------------------------------------------------------------------------- pure helpers


def check_out_dir(out: Path) -> None:
    """Refuse a directory that already holds anything."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"refusing to write into a non-empty {out}")


def load_rows(world: Path) -> list[dict[str, str]]:
    """Child rows from `<world>/renders.jsonl`: renders only, no labels."""
    rows = []
    for line in (world / "renders.jsonl").read_text().splitlines():
        rec = json.loads(line)
        rows.append(
            {
                "id": rec["unit"],
                "before": DRAFT_BEFORE,
                "render": rec["text"],
                "after": DRAFT_AFTER,
            }
        )
    return rows


def describe(values: Sequence[float]) -> dict[str, float]:
    """n, mean, p50, p90, p99, max and sum (percentiles interpolate linearly, as numpy)."""
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "sum": float(x.sum()),
    }


def nearest_units(n_prompt: Mapping[str, int], target: float, k: int = 1) -> list[str]:
    """The `k` unit ids whose `n_prompt` is nearest `target`; ties go to the smaller id."""
    return sorted(n_prompt, key=lambda u: (abs(n_prompt[u] - target), u))[:k]


def sample_with_longest(n_prompt: Mapping[str, int], n_sample: int, seed: int) -> list[str]:
    """`n_sample` seeded units (never the longest), then the longest one last."""
    longest = max(sorted(n_prompt), key=lambda u: n_prompt[u])
    others = sorted(u for u in n_prompt if u != longest)
    return random.Random(seed).sample(others, n_sample) + [longest]


def embed_layers(n_layers: int) -> str:
    """Quarter, half and three-quarter depth, then the final state: `8,16,24,final` at 32."""
    return f"{n_layers // 4},{n_layers // 2},{3 * n_layers // 4},final"


def embed_hours(total_tokens: float, tok_per_s: float) -> float:
    """Hours for one world's embeds: every unit's prompt tokens at the steady forward rate."""
    return total_tokens / tok_per_s / 3600


def tokens_per_epoch(
    sum_n_prompt: float, n_units: int, answer_tokens: int, train_fraction: float, n_worlds: int
) -> float:
    """Training tokens per epoch: (prompt tokens + answer tokens) of each unit, times the share
    trained on, times the number of pooled worlds."""
    return (sum_n_prompt + n_units * answer_tokens) * train_fraction * n_worlds


def lora_epoch_hours(tokens_per_epoch: float, tok_per_s: float) -> float:
    return tokens_per_epoch / tok_per_s / 3600


def bench_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Steady-state numbers of one lora-bench result (warm-up steps are reported apart)."""
    steps = result["steps"]
    steady = [s for s in steps if not s["warmup"]]
    ok = (
        result["error"] is None
        and bool(steady)
        and all(s["loss_finite"] and s["grads_finite"] for s in steps)
    )
    tokens = sum(s["tokens"] for s in steady)
    seconds = sum(s["seconds"] for s in steady)
    return {
        "batch_size": result["batch_size"],
        "n_prompt": result["n_prompt"],
        "ok": ok,
        "error": result["error"],
        "warmup_seconds": [s["seconds"] for s in steps if s["warmup"]],
        "steady_seconds": [s["seconds"] for s in steady],
        "steady_tokens": tokens,
        "tokens_per_s": tokens / seconds if seconds and steady else None,
        "peak_memory_gib": max((s["peak_memory_gib"] for s in steps), default=None),
        "loss": [s["loss"] for s in steps],
        "grad_tensors_nonzero": [s["grad_tensors_nonzero"] for s in steps],
    }


def best_batch(benches: Mapping[str, Mapping[str, Any]]) -> tuple[str, float] | None:
    """The median-bucket bench with the highest steady tok/s among those that ran cleanly."""
    ran = [
        (name, b["tokens_per_s"])
        for name, b in benches.items()
        if name.startswith("median") and b["ok"] and b["tokens_per_s"]
    ]
    return max(ran, key=lambda x: x[1]) if ran else None


# --------------------------------------------------------------------------- runner


def machine() -> dict[str, str]:
    def sysctl(key: str) -> str:
        done = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        return done.stdout.strip()

    return {
        "hw.memsize": sysctl("hw.memsize"),
        "machdep.cpu.brand_string": sysctl("machdep.cpu.brand_string"),
    }


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(world: Path, model: str, out: Path, seed: int, train_worlds: int) -> dict[str, Any]:
    check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(world)
    by_id = {r["id"]: r for r in rows}

    log(f"[1/4] count: {len(rows)} renders")
    counted = local_llm.count(model, rows, work_dir=out, tag="count")
    n_prompt = {r["id"]: r["n_prompt"] for r in counted["rows"]}
    render_tokens = [r["render_tokens"] for r in counted["rows"]]
    count = {
        "n_prompt": describe(list(n_prompt.values())),
        "render_tokens": describe(render_tokens),
        "over_limit": sum(v > TOKEN_LIMIT for v in n_prompt.values()),
        "limit": TOKEN_LIMIT,
        "template": counted["template"],
    }

    layers = embed_layers(n_layers(model))
    ids = sample_with_longest(n_prompt, EMBED_SAMPLE, seed)
    log(f"[2/4] embed: {len(ids)} units, layers {layers}")
    emb, _ = local_llm.embed(model, [by_id[i] for i in ids], layers, out, tag="embed")

    score_ids = sample_with_longest(n_prompt, SCORE_UNITS - 1, seed)
    log(f"[3/4] score: {len(score_ids)} units")
    scored = local_llm.score(model, [by_id[i] for i in score_ids], work_dir=out, tag="score")

    median, p90 = count["n_prompt"]["p50"], count["n_prompt"]["p90"]
    longest = max(sorted(n_prompt), key=lambda u: n_prompt[u])
    plans = {
        "median-b1": ([nearest_units(n_prompt, median)[0]], 1),
        "p90-b1": ([nearest_units(n_prompt, p90)[0]], 1),
        "longest-b1": ([longest], 1),
        "median-b4": (nearest_units(n_prompt, median, BATCH_UNITS), BATCH_UNITS),
    }
    benches: dict[str, dict[str, Any]] = {}
    for name, (units, batch) in plans.items():
        log(f"[4/4] lora-bench {name}: {units}")
        res = local_llm.lora_bench(
            model,
            [by_id[u] for u in units],
            BENCH_STEPS,
            BENCH_WARMUP,
            batch,
            work_dir=out,
            tag=f"lora-{name}",
        )
        benches[name] = {**bench_summary(res), "units": units}

    answer_tokens = len(count["template"]["ab"]["A"]["ids"])  # the dummy target "A" + <|im_end|>
    total = count["n_prompt"]["sum"]
    steady = emb["steady_tokens_per_s"]
    epoch_tokens = tokens_per_epoch(total, len(rows), answer_tokens, TRAIN_FRACTION, train_worlds)
    best = best_batch(benches)
    p90_rate = benches["p90-b1"]["tokens_per_s"] if benches["p90-b1"]["ok"] else None
    projections = {
        "embed_hours_per_world": embed_hours(total, steady) if steady else None,
        "embed_total_tokens": total,
        "embed_steady_tokens_per_s": steady,
        "lora_tokens_per_epoch": epoch_tokens,
        "lora_train_fraction": TRAIN_FRACTION,
        "lora_train_worlds": train_worlds,
        "lora_answer_tokens": answer_tokens,
        "lora_best_median_bench": best[0] if best else None,
        "lora_epoch_hours_median": lora_epoch_hours(epoch_tokens, best[1]) if best else None,
        "lora_epoch_hours_p90_bound": lora_epoch_hours(epoch_tokens, p90_rate)
        if p90_rate
        else None,
    }

    summary = {
        "world": str(world),
        "model": model,
        "model_revision": local_llm.model_revision(model),
        "mlx": emb["mlx"],
        "mlx_lm": emb["mlx_lm"],
        "machine": machine(),
        "seed": seed,
        "draft_prompt": {"before": DRAFT_BEFORE, "after": DRAFT_AFTER, "frozen": False},
        "count": count,
        "embed": {
            "layers": layers,
            "units": len(ids),
            "first_row_seconds": emb["first_row_seconds"],
            "steady_tokens_per_s": steady,
            "peak_memory_gib": emb["peak_memory_gib"],
            "wall_seconds": emb["wall_seconds"],
        },
        "score": {
            "units": len(score_ids),
            "max_abs_diff_nats": scored["max_abs_diff"],
            "mean_seconds_cached": float(np.mean([r["seconds_cached"] for r in scored["rows"]])),
            "mean_seconds_full": float(np.mean([r["seconds_full"] for r in scored["rows"]])),
            "min_answer_mass_ab": min(r["ab"]["mass_cached"] for r in scored["rows"]),
            "peak_memory_gib": scored["peak_memory_gib"],
        },
        "lora": benches,
        "projections": projections,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "summary.md").write_text(render_markdown(summary))
    return summary


def n_layers(model: str) -> int:
    config = json.loads((local_llm.model_dir(model) / "config.json").read_text())
    return int(config["text_config"]["num_hidden_layers"])


def _fmt(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{digits}f}"


def render_markdown(s: Mapping[str, Any]) -> str:
    c, e, sc, p = s["count"], s["embed"], s["score"], s["projections"]
    t = c["n_prompt"]
    lines = [
        f"# M3 feasibility: {s['model']} ({s['model_revision'][:12]})",
        "",
        f"World `{s['world']}`. mlx {s['mlx']}, mlx-lm {s['mlx_lm']}. "
        f"Machine: {s['machine']['machdep.cpu.brand_string']}, {s['machine']['hw.memsize']} bytes. "
        "The prompt is a DRAFT (M3 only).",
        "",
        "## 1. Lengths (tokens)",
        "",
        "| | mean | p50 | p90 | p99 | max | sum |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, d in (("prompt + render", t), ("render only", c["render_tokens"])):
        lines.append(
            f"| {label} | {d['mean']:.0f} | {d['p50']:.0f} | {d['p90']:.0f} | {d['p99']:.0f} "
            f"| {d['max']:.0f} | {d['sum']:,.0f} |"
        )
    lines += [
        "",
        f"Prompts over {c['limit']} tokens: {c['over_limit']} of {t['n']}.",
        "",
        "## 2. Forward pass (R2 extractor)",
        "",
        f"{e['units']} units, layers {e['layers']}. First row {_fmt(e['first_row_seconds'])} s "
        f"(warm-up). Steady **{_fmt(e['steady_tokens_per_s'], 1)} tok/s**, peak "
        f"{_fmt(e['peak_memory_gib'])} GiB.",
        "",
        "## 3. Scorer agreement",
        "",
        f"{sc['units']} units. Max |cached - full| in nats: "
        f"A/B {sc['max_abs_diff_nats']['ab']:.4f}, "
        f"levels 1-10 {sc['max_abs_diff_nats']['level']:.4f}. Mean seconds per unit: cached "
        f"{_fmt(sc['mean_seconds_cached'])}, full {_fmt(sc['mean_seconds_full'])}. "
        f"Lowest A/B answer mass {sc['min_answer_mass_ab']:.4f}.",
        "",
        "## 4. LoRA step (rank 16, dummy target, batch size B)",
        "",
        "| bench | B | prompt tokens | steady s/step | tok/s | peak GiB | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, b in s["lora"].items():
        status = "ok" if b["ok"] else (b["error"] or {}).get("error", "failed")
        lines.append(
            f"| {name} | {b['batch_size']} | {b['n_prompt']} "
            f"| {', '.join(_fmt(x, 1) for x in b['steady_seconds'])} "
            f"| {_fmt(b['tokens_per_s'], 1)} "
            f"| {_fmt(b['peak_memory_gib'])} | {status} |"
        )
    lines += [
        "",
        "## 5. Projections",
        "",
        f"- Embeds for one world: {p['embed_total_tokens']:,.0f} tokens at "
        f"{_fmt(p['embed_steady_tokens_per_s'], 1)} tok/s = "
        f"**{_fmt(p['embed_hours_per_world'])} h**.",
        f"- LoRA epoch: {p['lora_tokens_per_epoch']:,.0f} tokens "
        f"({p['lora_train_fraction']} of each of {p['lora_train_worlds']} worlds, with "
        f"{p['lora_answer_tokens']} answer tokens per unit). At the best median-bucket rate "
        f"({p['lora_best_median_bench']}): **{_fmt(p['lora_epoch_hours_median'])} h**; "
        f"pessimistic (p90 bucket): {_fmt(p['lora_epoch_hours_p90_bound'])} h.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(local_llm.MODELS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-worlds", type=int, default=2)
    args = parser.parse_args(argv)
    run(args.world, args.model, args.out, args.seed, args.train_worlds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
