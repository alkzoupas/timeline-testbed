"""R1 zero-shot scoring for one world (SPEC §7 R1): label-free forward passes.

    python -m testbed.models.r1_run --world runs/m1-dev-s0.75-seed1 --model 9b --out runs/m4-r1 \\
        [--limit N] [--check-full]

Reads only `<world>/renders.jsonl` and `<world>/manifest.json`. Each render is wrapped in the
frozen prompt (`prompt.py`) and checked to fit SPEC §6.2's 2,048 tokens, as `r2_embed` does. The
MLX child's `zeroshot` then scores the ten `RISK_LEVEL: n` answers with the output head in
float32 (M3 found bf16 logits tie P(answer) on a 0.125 log-odds grid). The score is
`expected` = sum n * p_n over the normalized ten. Rows run in chunks of `CHUNK`, so a crash loses
at most one chunk: a rerun reuses every chunk already written and checked.

`--check-full` also scores each answer with a plain full forward pass, for the agreement check
(`agreement_summary`); it multiplies the cost by about 11. `--limit N` scores only the first N
renders, for smoke runs; the output then holds N rows and records the limit.

Writes `<out>/<model>/<world>.jsonl` (one row per unit, in render order) and `<world>.json`
(provenance, throughput and the agreement summary). The chunk directory `<world>.parts/` is
removed once the merged files are written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from testbed.models import local_llm, prompt, r2_embed

CHUNK = 100  # a zero-shot row costs ~3.5 s, so a chunk is ~6 min of work
P_FLOOR = 1e-3  # the agreement check ignores candidates below this normalized probability
GRID = 0.125  # bf16's step for logit magnitudes of 16-32
GRID_TOL = 1e-5  # float32 head noise at those magnitudes is ~2e-6
ROW_FIELDS = ("id", "n_prompt", "logprobs", "mass", "p", "expected", "argmax")
CHECK_FIELDS = ("logprobs_full", "expected_full", "max_abs_diff")


def on_grid_pairs(p: Sequence[float]) -> int:
    """The number of candidate pairs, among those with p >= `P_FLOOR`, whose log-odds difference
    is within `GRID_TOL` of a multiple of `GRID` (0.125). Bf16 logits put every pair there."""
    logp = [float(np.log(x)) for x in p if x >= P_FLOOR]
    count = 0
    for a, b in combinations(logp, 2):
        d = abs(a - b) / GRID
        if abs(d - round(d)) * GRID <= GRID_TOL:
            count += 1
    return count


def agreement_summary(rows: Sequence[Mapping[str, Any]], check_full: bool) -> dict[str, Any]:
    """What the M3 follow-up needs: how many distinct scores there are, whether any normalized p
    sits on the bf16 grid, and (with `check_full`) the cached-versus-full agreement."""
    if not rows:
        raise ValueError("no rows")
    grid = [on_grid_pairs(r["p"]) for r in rows]
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_distinct_expected": len({float(r["expected"]) for r in rows}),
        "p_floor": P_FLOOR,
        "grid": GRID,
        "grid_tol": GRID_TOL,
        "rows_with_grid_pairs": sum(1 for g in grid if g),
        "grid_pairs": sum(grid),
        "min_mass": min(float(r["mass"]) for r in rows),
    }
    if check_full:
        summary["max_abs_diff"] = max(float(r["max_abs_diff"]) for r in rows)
        summary["max_expected_diff"] = max(
            abs(float(r["expected"]) - float(r["expected_full"])) for r in rows
        )
    return summary


def _load_part(
    work: Path, tag: str, rows: Sequence[Mapping[str, str]], check_full: bool, model: str
) -> dict[str, Any] | None:
    """A finished chunk's child result, or None if it is missing, fails the structural checks or
    was not made from these rows, this child script and this model snapshot."""
    path = work / f"{tag}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
        r2_embed.check_source(result, rows, model, f"zeroshot {tag}")
        local_llm.check_zeroshot(result, rows, check_full)
    except RuntimeError, ValueError, KeyError, TypeError, OSError:
        return None
    return dict(result)


def run(
    world: Path, model: str, out: Path, *, limit: int | None = None, check_full: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    world = world.resolve()
    manifest = json.loads((world / "manifest.json").read_text())
    renders = [json.loads(x) for x in (world / "renders.jsonl").read_text().splitlines() if x]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        renders = renders[:limit]
    rows = prompt.child_rows(renders)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{world}: duplicate unit ids in renders.jsonl")
    if prompt.prompt_sha256() != prompt.PROMPT_SHA256:
        raise RuntimeError("prompt.py text differs from PROMPT_SHA256")

    dest = out / model
    final_rows, final_json = dest / f"{world.name}.jsonl", dest / f"{world.name}.json"
    if final_rows.exists() or final_json.exists():
        raise FileExistsError(f"{final_rows} or its json already exists")
    work = dest / f"{world.name}.parts"
    work.mkdir(parents=True, exist_ok=True)

    longest = r2_embed.check_fits(local_llm.count(model, rows)["rows"])

    results: list[dict[str, Any]] = []
    reused = 0
    for i, (a, b) in enumerate(r2_embed.chunk_ranges(len(rows), CHUNK)):
        tag = f"chunk{i:03d}"
        result = _load_part(work, tag, rows[a:b], check_full, model)
        if result is None:
            result = local_llm.zeroshot(model, rows[a:b], check_full, work_dir=work, tag=tag)
            r2_embed.check_source(result, rows[a:b], model, f"zeroshot {tag}")
        else:
            reused += 1
        if results and result["child_sha256"] != results[0]["child_sha256"]:
            raise RuntimeError(f"{tag}: made by a different child script than chunk000")
        results.append(result)
    per_row = [r for res in results for r in res["rows"]]
    if [r["id"] for r in per_row] != ids:
        raise RuntimeError("merged ids differ from the render order")

    steady_tokens = sum(res["steady_tokens"] for res in results)
    steady_seconds = sum(res["steady_seconds"] for res in results)
    fields = ROW_FIELDS + (CHECK_FIELDS if check_full else ())
    record = {
        "world": world.name,
        "world_checksum_sha256": manifest["checksum_sha256"],
        "generator_version": manifest["generator_version"],
        "model": model,
        "model_revision": local_llm.model_revision(model),
        "prompt_version": prompt.PROMPT_VERSION,
        "prompt_sha256": prompt.PROMPT_SHA256,
        "child_sha256": results[0]["child_sha256"],  # every chunk agrees (checked above)
        "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
        "head_dtype": "float32",
        "limit": limit,
        "check_full": check_full,
        "n_rows": len(rows),
        "n_prompt": {"max": longest, "mean": float(np.mean([r["n_prompt"] for r in per_row]))},
        "chunks": len(results),
        "chunks_reused": reused,
        "peak_memory_gib": max(res["peak_memory_gib"] for res in results),
        "steady_tokens_per_s": steady_tokens / steady_seconds if steady_seconds else None,
        "steady_seconds_per_unit": steady_seconds / max(len(per_row) - len(results), 1),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "agreement": agreement_summary(per_row, check_full),
    }
    final_rows.write_text("".join(json.dumps({k: r[k] for k in fields}) + "\n" for r in per_row))
    final_json.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    shutil.rmtree(work)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(local_llm.MODELS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-full", action="store_true")
    args = parser.parse_args(argv)
    record = run(args.world, args.model, args.out, limit=args.limit, check_full=args.check_full)
    rate = record["steady_tokens_per_s"]
    print(
        f"{record['world']} {record['model']}: {record['n_rows']} rows, "
        f"{'n/a' if rate is None else f'{rate:.0f}'} tok/s, {record['wall_seconds']:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
