"""Gate G1 (SPEC §7.1), exactly as pre-registered in `prereg/M2.md`. Coordinator-owned.

G1 passes only if (a) every sweep world's health and the generator-only sample pass, (b) the oracle
reaches `POSITIVE_MIN` per family on the pooled s = 1 worlds, (c) every R0 rung meets the null rule
on the pooled s = 0 worlds, as one Bonferroni family of `G1_TESTS` items, and (d) the channel check
passes with the frozen tags.

CLI::

    python -m testbed.eval.g1 --sweep runs/m1-dev-s0-seed1 ... \
        --gensample runs/m1-gensample-seed201 --channel runs/m2-channel/channel.json \
        --out runs/m2-r0/g1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from testbed.eval.health import null_row, positive_row
from testbed.eval.r0 import FITTED, REGIMES, R0Unit, load_world, score_pool
from testbed.world.families import MEASURED_TAGS, PAIRS
from testbed.world.generator import GENERATOR_VERSION

Row = dict[str, Any]
Items = dict[str, tuple[np.ndarray, np.ndarray]]

S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
WORLDS_PER_S = 2
G1_TESTS = 62  # prereg/M2.md (c): 2 x 7 unfitted + 3 x 2 x 7 fitted + 6 oracle
POSITIVE_MIN = 0.90  # SPEC §7.1, prereg/M2.md (b)
UNFITTED = ("random", "detector")  # one score per unit, the same in both regimes


def _comparisons(units: Sequence[R0Unit], score: np.ndarray) -> Items:
    """Each family against its own twin (`family:<pair>`); all abusive against all legitimate."""
    pos = np.array([u.positive for u in units])
    pair = np.array([u.pair for u in units])
    out: Items = {}
    for p in sorted(PAIRS):
        mask = pair == p
        out[f"family:{p}"] = (score[mask & pos], score[mask & ~pos])
    out["pooled"] = (score[pos], score[~pos])
    return out


def null_items(units: Sequence[R0Unit], scores: Mapping[tuple[str, str], np.ndarray]) -> Items:
    """The 62 items of prereg/M2.md (c), keyed `<rung>[|<regime>]|<comparison>`."""
    items: Items = {}
    for rung in UNFITTED:
        if not np.array_equal(scores[(rung, "in_family")], scores[(rung, "held_out")]):
            raise ValueError(f"{rung} differs between regimes; it must not be fitted")
        for key, value in _comparisons(units, scores[(rung, "in_family")]).items():
            items[f"{rung}|{key}"] = value
    for rung in FITTED:
        for regime in REGIMES:
            for key, value in _comparisons(units, scores[(rung, regime)]).items():
                items[f"{rung}|{regime}|{key}"] = value
    for key, value in _comparisons(units, scores[("oracle", "in_family")]).items():
        if key != "pooled":  # the oracle's statistics are on different scales per pair
            items[f"oracle|{key}"] = value
    if len(items) != G1_TESTS:
        raise ValueError(f"{len(items)} null items, pre-registered {G1_TESTS}")
    return items


def positive_items(units: Sequence[R0Unit], scores: Mapping[tuple[str, str], np.ndarray]) -> Items:
    """prereg/M2.md (b): the oracle per family against its twin."""
    items = _comparisons(units, scores[("oracle", "in_family")])
    return {f"oracle|{k}": v for k, v in items.items() if k != "pooled"}


def _pool(worlds: Sequence[Path], s: float) -> list[R0Unit]:
    units: list[R0Unit] = []
    for world in worlds:
        world_s, loaded, mixed = load_world(world)
        if world_s != s:
            raise ValueError(f"{world}: s = {world_s}, expected {s}")
        if mixed:
            raise ValueError(f"{world}: {mixed} mixed units")
        units.extend(loaded)
    return sorted(units, key=lambda u: u.group)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def health_row(sweep: Sequence[Path], gensample: Path) -> Row:
    """prereg/M2.md (a): each sweep world's `health.json` passed and describes the current world;
    the generator-only sample passed on this generator version."""
    results: dict[str, Any] = {}
    for world in sweep:
        report = _json(world / "health.json")
        current = report["manifest"] == _json(world / "manifest.json")
        results[world.name] = {"passed": bool(report["passed"]) and current, "current": current}
    sample = _json(gensample / "health.json")
    results[gensample.name] = {
        "passed": bool(sample["passed"]) and sample["generator_version"] == GENERATOR_VERSION
    }
    failed = sorted(k for k, r in results.items() if not r["passed"])
    return {"name": "health", "passed": not failed, "failed": failed, "results": results}


def channel_row(channel: Path) -> Row:
    """prereg/M2.md (d): the channel check passed and measured the frozen tags."""
    report = _json(channel)
    measured = {int(p): t for p, t in report["measured_tags"].items()}
    frozen = measured == MEASURED_TAGS
    return {
        "name": "channel_tags",
        "passed": bool(report["passed"]) and frozen,
        "measured": measured,
        "frozen": frozen,
    }


def sweep_by_s(sweep: Sequence[Path]) -> dict[float, list[Path]]:
    """The sweep's worlds by s; it must be the full grid, `WORLDS_PER_S` standard worlds each."""
    by_s: dict[float, list[Path]] = {}
    for world in sweep:
        manifest = _json(world / "manifest.json")
        if manifest["scale"] != "standard" or manifest["generator_version"] != GENERATOR_VERSION:
            raise ValueError(f"{world}: not a standard {GENERATOR_VERSION} world")
        by_s.setdefault(float(manifest["s"]), []).append(world)
    counts = Counter({s: len(w) for s, w in by_s.items()})
    if sorted(by_s) != list(S_GRID) or set(counts.values()) != {WORLDS_PER_S}:
        raise ValueError(f"sweep must be {WORLDS_PER_S} worlds at each s in {S_GRID}: {counts}")
    return by_s


def gate(sweep: Sequence[Path], gensample: Path, channel: Path) -> Row:
    """G1 as pre-registered; every row must pass."""
    by_s = sweep_by_s(sweep)
    rows = [health_row(sweep, gensample)]
    s1 = _pool(by_s[1.0], 1.0)
    rows.append(
        positive_row("oracle_positive_s1", positive_items(s1, score_pool(s1)), minimum=POSITIVE_MIN)
    )
    s0 = _pool(by_s[0.0], 0.0)
    rows.append(null_row("r0_null_s0", null_items(s0, score_pool(s0)), tests=G1_TESTS))
    rows.append(channel_row(channel))
    return {
        "gate": "G1",
        "prereg": "prereg/M2.md",
        "worlds": {str(s): [w.name for w in ws] for s, ws in sorted(by_s.items())},
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--gensample", type=Path, required=True)
    parser.add_argument("--channel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; G1 is scored once")
    report = gate([p.resolve() for p in args.sweep], args.gensample.resolve(), args.channel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    marks = " ".join(f"{r['name']}={'ok' if r['passed'] else 'FAIL'}" for r in report["rows"])
    print(f"G1 {'passed' if report['passed'] else 'FAILED'}: {marks}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
