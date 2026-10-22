"""Run one world through the label-free pipeline (SPEC §4, §5.7, §6.1, §6.2).

generate -> persist -> detect -> units -> render / F12 / F64 -> token counts. Everything is
written under `out`, which must be new or empty. Label-free: labels travel only inside the
generated world into the write-only sink in `persist.py`; nothing here reads them.

Files under `out`: the two stores, `manifest.json` (the world manifest), `units.jsonl` (one row
per unit, sorted by unit key), `renders.jsonl` and `run.json` (timings, counts, token stats).

Usage: .venv/bin/python -m testbed.pipeline.run_world --seed 1 --s 1.0 --scale standard \
    --out runs/m1-dev-s1-seed1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from app.services.generator import WORLD_END

from testbed.features.f12 import f12
from testbed.features.f64 import f64
from testbed.pipeline.detect import detect
from testbed.pipeline.render import RENDER_VERSION, render
from testbed.pipeline.timeline import load_timeline
from testbed.pipeline.tokens import count_tokens
from testbed.pipeline.units import load_units
from testbed.world.generator import WorldSpec, generate_world, world_manifest
from testbed.world.persist import Stores, persist

TOKEN_BUDGET = 1792  # SPEC §6.2: the render plus the prompt must fit in 2,048 tokens

TokenCounter = Callable[[Sequence[str]], list[int]]


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write_lines(path: Path, rows: Sequence[Any]) -> None:
    path.write_text("".join(_dump(row) + "\n" for row in rows))


def run_world(
    spec: WorldSpec, out: Path, *, token_counter: TokenCounter = count_tokens
) -> dict[str, Any]:
    """Run `spec` into `out` and return the `run.json` record. Refuses a non-empty `out`.

    `token_counter` is the Qwen tokenizer unless a test swaps it for a stub."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    stores = Stores.under(out)
    marks = [time.monotonic()]

    def lap() -> None:
        marks.append(time.monotonic())

    world = generate_world(spec)
    manifest = world_manifest(world, spec)
    lap()
    persist(world, manifest, stores)
    lap()
    detection = detect(stores.inference_url, run_id=out.name)
    lap()
    units = load_units(stores.inference_db, WORLD_END)
    lap()

    rows: list[dict[str, Any]] = []
    renders: list[dict[str, str]] = []
    with closing(sqlite3.connect(f"file:{stores.inference_db}?mode=ro", uri=True)) as conn:
        for unit in units:
            timeline = load_timeline(conn, unit.member_account_ids)
            text = render(timeline)
            f12_values = f12(unit)
            rows.append(
                {
                    "unit": unit.snapshot_id,
                    "members": list(unit.member_account_ids),
                    "detector_score": unit.detector_score,
                    "f12": f12_values,
                    "f64": f64(timeline, f12_values),
                    "render_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            renders.append({"unit": unit.snapshot_id, "text": text})
    lap()

    tokens = token_counter([r["text"] for r in renders])
    if len(tokens) != len(rows):
        raise RuntimeError("the token counter returned the wrong number of counts")
    for row, count in zip(rows, tokens, strict=True):
        row["tokens"] = count
    lap()

    (out / "manifest.json").write_text(_dump(manifest) + "\n")
    _write_lines(out / "units.jsonl", rows)
    _write_lines(out / "renders.jsonl", renders)
    names = ("generate", "persist", "detect", "units", "features", "tokens")
    seconds = {n: round(b - a, 3) for n, a, b in zip(names, marks[:-1], marks[1:], strict=True)}
    seconds["total"] = round(marks[-1] - marks[0], 3)
    record: dict[str, Any] = {
        "seed": spec.seed,
        "s": spec.s,
        "scale": spec.scale,
        "seconds": seconds,
        "counts": {
            "units": len(rows),
            "clusters": detection.clusters,
            "cluster_cycles": detection.cluster_cycles,
            "accounts": manifest["counts"]["accounts"],
            "events": manifest["counts"]["events"],
        },
        "detection_sha256": detection.detection_sha256,
        "render_version": RENDER_VERSION,
        "max_tokens": max(tokens, default=0),
        "mean_tokens": round(sum(tokens) / len(tokens), 2) if tokens else 0.0,
        "token_budget": TOKEN_BUDGET,
        "units_over_budget": sum(1 for t in tokens if t > TOKEN_BUDGET),
    }
    (out / "run.json").write_text(_dump(record) + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--s", type=float, required=True)
    parser.add_argument("--scale", choices=("standard", "channel"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = run_world(WorldSpec(args.seed, args.s, args.scale), args.out)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
