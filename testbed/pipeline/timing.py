"""M0: time generate, persist, detect and units on a vendored preset (SPEC §5.7, §10).

Label-free. Writes the stores and `timing.json` under `<out>/<preset>-seed<seed>/`, and refuses
to reuse a directory that already exists.

Usage: .venv/bin/python -m testbed.pipeline.timing --preset dev --out runs/m0-timing
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from app.services.generator import PRESETS, WORLD_END, build_manifest, generate_world

from testbed.pipeline.detect import detect
from testbed.pipeline.units import load_units
from testbed.world.persist import Stores, persist


def time_preset(preset: str, seed: int, out: Path) -> dict[str, Any]:
    world_dir = out / f"{preset}-seed{seed}"
    world_dir.mkdir(parents=True, exist_ok=False)
    stores = Stores.under(world_dir)

    started = time.monotonic()
    world = generate_world(seed, preset)
    manifest = build_manifest(world, seed, preset)
    generated = time.monotonic()
    persist(world, manifest, stores)
    persisted = time.monotonic()
    detection = detect(stores.inference_url, run_id=f"m0-timing-{preset}-{seed}")
    detected = time.monotonic()
    units = load_units(stores.inference_db, WORLD_END)
    done = time.monotonic()

    record: dict[str, Any] = {
        "preset": preset,
        "seed": seed,
        "accounts": len(world.accounts),
        "events": len(world.events),
        "linkage_observations": len(world.linkage),
        "clusters": detection.clusters,
        "cluster_cycles": detection.cluster_cycles,
        "units": len(units),
        "seconds": {
            "generate": round(generated - started, 3),
            "persist": round(persisted - generated, 3),
            "detect": round(detected - persisted, 3),
            "units": round(done - detected, 3),
            "total": round(done - started, 3),
        },
        "detect_seconds_per_cluster_cycle": round(
            (detected - persisted) / max(1, detection.cluster_cycles), 4
        ),
        "detection_sha256": detection.detection_sha256,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    (world_dir / "timing.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(time_preset(args.preset, args.seed, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
