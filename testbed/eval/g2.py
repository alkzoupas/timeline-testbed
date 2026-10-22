"""Gate G2 (SPEC §7.1), exactly as pre-registered in `prereg/M4.md`. Coordinator-owned.

G2 asks whether the R2 probe beats both F64-w90 models on the primary metric (SPEC §6.4: held-out
regime, T families, mean of sides A and B) at s = 0.75, and shows no signal at s = 0. For each
model in `FROZEN`, with its frozen feature and C:

- (a) R2 - F64 logistic and R2 - F64 trees, paired on R0's primary resamples, both have a lower
  bound above 0 at `GATE_LEVEL` (two-sided 97.5%: SPEC's 95%, Bonferroni-adjusted for two models);
- (b) R2's primary interval at `NULL_LEVEL` on the pooled s = 0 worlds contains 0.5.

Verdict: PASS if some model meets (a) and every model meets (b); LEAK if some model meets (a) and
some model fails (b); FAIL if no model meets (a).

`FROZEN` holds the tuning world's choices (`r2 tune`, prereg amendment) and `TUNE_SHA256` the
sha256 of each `tune.json`; the scorer checks both. F64 scores are recomputed with `score_pool` and
must reproduce M2's published values (`M2_F64`), or the scorer stops before any R2 number exists.

CLI::

    python -m testbed.eval.g2 --runs runs --embed runs/m4-embed --tune runs/m4-tune \
        --out runs/m4-g2 [--dry-run]

`--dry-run` loads and checks every input (worlds, tune files, embeddings, the F64 reproduction)
and fits no probe and computes no R2 number. Without it, G2 is scored once: `--out` must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from testbed.eval import r2
from testbed.eval.metrics import percentile_interval
from testbed.eval.r0 import (
    RUNGS,
    R0Unit,
    _write_scores,
    evaluate,
    load_world,
    primary_boot,
    primary_comparisons,
    score_pool,
)

Row = dict[str, Any]
Boot = tuple[float, NDArray[np.float64]]  # (point, paired primary resamples)

GATE_WORLDS: dict[float, tuple[str, ...]] = {
    0.75: ("m1-dev-s0.75-seed1", "m2-dev-s0.75-seed2"),
    0.0: ("m1-dev-s0-seed1", "m1-dev-s0-seed2"),
}
F64_RUNGS = ("f64_logistic", "f64_trees")
GATE_LEVEL = 1 - 0.05 / 2  # two-sided 97.5%: lower bound = the 1.25th percentile
NULL_LEVEL = 0.95
# M2's published held-out primaries (runs/m2-r0/r0.json, sha256 4f3f2ddc...).
M2_F64: dict[float, dict[str, float]] = {
    0.75: {"f64_logistic": 0.461552734375, "f64_trees": 0.5165673828125},
    0.0: {"f64_logistic": 0.46761230468750004, "f64_trees": 0.5243701171875},
}
M2_TOLERANCE = 1e-12
# model -> (feature, C), frozen on the tuning world before scoring (prereg/M4.md amendment).
FROZEN: dict[str, tuple[str, float]] = {"9b": ("mean_16", 10.0), "4b": ("mean_16", 0.001)}
TUNE_SHA256: dict[str, str] = {
    "9b": "9476e5dfa8cd80610b99ae7a3098f78abcf8787bbe154c096f0de201f1592bca",
    "4b": "e66035cb2e4aceb7c04e374e3e02b1edf356089724e8017ef6e6feea228a610e",
}


# --- Pure core ------------------------------------------------------------------------------------


def lower_bound(boot: NDArray[np.float64], level: float = GATE_LEVEL) -> float:
    """The lower end of the two-sided percentile interval of `boot` at `level`."""
    return percentile_interval(boot, level)[0]


def verdict(beats: Mapping[str, bool], null_ok: Mapping[str, bool]) -> str:
    """PASS, LEAK or FAIL from each model's (a) and (b) (prereg/M4.md)."""
    if set(beats) != set(null_ok) or not beats:
        raise ValueError("(a) and (b) must cover the same, non-empty set of models")
    if not any(beats.values()):
        return "FAIL"
    if not all(null_ok.values()):
        return "LEAK"
    return "PASS"


def decide(
    f64: Mapping[float, Mapping[str, Boot]], probe: Mapping[float, Mapping[str, Boot]]
) -> Row:
    """The G2 rows and verdict from the held-out primary (point, resamples) of every rung.

    `f64[s][rung]` and `probe[s][model]` for s in 0.75 and 0; every resample array of one s
    comes from the same `primary_comparisons` weights, so differences are paired."""
    models = sorted(probe[0.75])
    if sorted(probe[0.0]) != models:
        raise ValueError("the s = 0 run must cover the same models")
    rows: list[Row] = []
    beats: dict[str, bool] = {}
    null_ok: dict[str, bool] = {}
    for m in models:
        point, boot = probe[0.75][m]
        diffs = {}
        for rung in F64_RUNGS:
            f_point, f_boot = f64[0.75][rung]
            bound = lower_bound(boot - f_boot)
            diffs[rung] = {"point": point - f_point, "lower_bound": bound, "passed": bound > 0}
        beats[m] = all(d["passed"] for d in diffs.values())
        rows.append(
            {"name": f"a_beats_f64_{m}", "r2_primary": point, "minus": diffs, "passed": beats[m]}
        )
        z_point, z_boot = probe[0.0][m]
        lo, hi = percentile_interval(z_boot, NULL_LEVEL)
        null_ok[m] = lo <= 0.5 <= hi
        rows.append(
            {"name": f"b_null_s0_{m}", "auc": z_point, "lo": lo, "hi": hi, "passed": null_ok[m]}
        )
    return {
        "rows": rows,
        "f64": {str(s): {r: f64[s][r][0] for r in F64_RUNGS} for s in sorted(f64)},
        "verdict": verdict(beats, null_ok),
    }


# --- Inputs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pool:
    s: float
    worlds: list[Path]
    units: list[R0Unit]
    mixed: dict[str, int]


def load_pool(runs: Path, s: float) -> Pool:
    """The gate worlds of `s`, pooled and sorted by group, as `r0.run` pools them."""
    worlds = [(runs / name).resolve() for name in GATE_WORLDS[s]]
    units: list[R0Unit] = []
    mixed: dict[str, int] = {}
    for world in worlds:
        world_s, world_units, n_mixed = load_world(world)
        if world_s != s:
            raise ValueError(f"{world.name}: s = {world_s}, expected {s}")
        units.extend(world_units)
        mixed[world.name] = n_mixed
    return Pool(s, worlds, sorted(units, key=lambda u: u.group), mixed)


def check_tune(tune: Path) -> dict[str, str]:
    """Each model's `tune.json` matches `TUNE_SHA256` and selected `FROZEN`'s choice."""
    if not FROZEN or set(FROZEN) != set(TUNE_SHA256):
        raise RuntimeError("FROZEN and TUNE_SHA256 must be set, for the same models")
    out = {}
    for m, (feature, c) in sorted(FROZEN.items()):
        path = tune / m / "tune.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != TUNE_SHA256[m]:
            raise RuntimeError(f"{path}: sha256 {digest} != the frozen {TUNE_SHA256[m]}")
        selected = json.loads(path.read_text())["selected"]
        if (selected["feature"], float(selected["C"])) != (feature, c):
            raise RuntimeError(f"{path}: selected {selected} != FROZEN {(feature, c)}")
        out[m] = digest
    return out


def f64_boots(pool: Pool, scores: Mapping[tuple[str, str], NDArray[np.float64]]) -> dict[str, Boot]:
    """F64's held-out primary on the pool; raises unless it reproduces M2."""
    comps = primary_comparisons(pool.units)
    out = {rung: primary_boot(scores[(rung, "held_out")], comps) for rung in F64_RUNGS}
    for rung, (point, _) in out.items():
        if abs(point - M2_F64[pool.s][rung]) > M2_TOLERANCE:
            raise RuntimeError(f"s = {pool.s} {rung}: {point!r} does not reproduce M2")
    return out


def _features(embed: Path, m: str, pool: Pool) -> NDArray[np.float64]:
    return r2.pooled_features(embed, m, pool.worlds, pool.units, FROZEN[m][0])


# --- Run ------------------------------------------------------------------------------------------


def dry_run(runs: Path, embed: Path, tune: Path) -> Row:
    """Every input loaded and checked; no probe is fitted and no R2 number computed."""
    digests = check_tune(tune)
    out: Row = {"tune_sha256": digests, "pools": {}}
    for s in GATE_WORLDS:
        pool = load_pool(runs, s)
        f64_boots(pool, score_pool(pool.units))
        shapes = {m: list(_features(embed, m, pool).shape) for m in sorted(FROZEN)}
        out["pools"][str(s)] = {
            "worlds": [w.name for w in pool.worlds],
            "units": len(pool.units),
            "mixed": pool.mixed,
            "feature_shapes": shapes,
            "f64_reproduces_m2": True,
        }
    return out


def score(runs: Path, embed: Path, tune: Path) -> tuple[Row, Row, dict[float, Any]]:
    """(gate, full report by s, units and scores by s for `scores.jsonl`)."""
    digests = check_tune(tune)
    f64: dict[float, dict[str, Boot]] = {}
    probe: dict[float, dict[str, Boot]] = {}
    report: Row = {}
    scored: dict[float, Any] = {}
    hits: dict[str, dict[str, int]] = {}  # lbfgs fits that stopped at MAX_ITER (recorded only)
    for s in GATE_WORLDS:
        pool = load_pool(runs, s)
        scores = dict(score_pool(pool.units))
        f64[s] = f64_boots(pool, scores)
        comps = primary_comparisons(pool.units)
        y = np.array([int(u.positive) for u in pool.units], dtype=np.int64)
        pairs = np.array([u.pair for u in pool.units], dtype=np.int64)
        groups = [u.group for u in pool.units]
        strata = r2.strata_of(pool.units)
        probe[s] = {}
        for m, (_, c) in sorted(FROZEN.items()):
            x = _features(embed, m, pool)
            held: list[bool] = []
            held_out = r2.held_out_scores(x, y, pairs, c, held)
            probe[s][m] = primary_boot(held_out, comps)
            scores[(r2.rung_name(m), "held_out")] = held_out
            infam: list[bool] = []
            scores[(r2.rung_name(m), "in_family")] = r2.in_family_scores(
                x, y, groups, strata, c, infam
            )
            hits[f"{s}/{m}"] = {"held_out": sum(held), "in_family": sum(infam)}
        extra = tuple(r2.rung_name(m) for m in sorted(FROZEN))
        report[str(s)] = {
            "s": s,
            "worlds": [w.name for w in pool.worlds],
            "mixed": pool.mixed,
            **evaluate(
                pool.units,
                scores,
                rungs=RUNGS + extra,
                extra_differences=[(e, f) for e in extra for f in F64_RUNGS],
            ),
        }
        scored[s] = (pool.units, scores, RUNGS + extra)
    gate = {
        "gate": "G2",
        "prereg": "prereg/M4.md",
        "frozen": {m: {"feature": f, "C": c} for m, (f, c) in sorted(FROZEN.items())},
        "tune_sha256": digests,
        "worlds": {str(s): list(names) for s, names in GATE_WORLDS.items()},
        "levels": {"gate": GATE_LEVEL, "null": NULL_LEVEL},
        "max_iter_hits": hits,
        **decide(f64, probe),
    }
    return gate, report, scored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--embed", type=Path, required=True)
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args.runs, args.embed, args.tune), indent=2, sort_keys=True))
        return 0
    if args.out.exists():
        parser.error(f"{args.out} exists; G2 is scored once")
    gate, report, scored = score(args.runs, args.embed, args.tune)
    args.out.mkdir(parents=True)
    for name, obj in (("g2.json", gate), ("report.json", report)):
        text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)
        (args.out / name).write_text(text + "\n")
    for s, (units, scores, rungs) in scored.items():
        _write_scores(args.out / f"scores-s{s}.jsonl", {s: (units, scores)}, rungs)
    marks = " ".join(f"{r['name']}={'ok' if r['passed'] else 'no'}" for r in gate["rows"])
    print(f"G2 {gate['verdict']}: {marks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
