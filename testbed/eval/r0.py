"""R0, the baselines (SPEC §7): random, detector score, F12 logistic, F64-w90 logistic and trees,
and the oracle bound, scored in both regimes (SPEC §6.3) and reported as SPEC §6.4 asks.

Labels come only from `open_labels`, as in `health._channel_units`. `score_pool` and `evaluate`
are pure functions of `R0Unit` lists, so known-answer tests drive them without a store.

Conventions:

- Positive class = abusive family; negative = its legitimate twin. One unit per group.
- Held-out regime: side A is pairs 4-6 and side B is pairs 1-3 (`TEST_PAIRS`). A side is scored by
  the model fitted on the other side's pairs, so every unit gets exactly one held-out score.
- Primary (SPEC §6.4): per side, the AUC of the side's T abusive units against the twins of the
  same pairs (T by `MEASURED_TAGS`); the mean of the two sides.
- The oracle statistic only ranks within a pair (its fill is per pair). So only `family:<pair>` is
  a real AUC for it; every pooled item reports the mean of the family AUCs it covers.
- Intervals are 95% percentile intervals of `N_BOOT` group-bootstrap resamples. Every comparison
  draws its weights once, from `BOOT_SEED`, and all rungs and regimes share them, so differences
  are paired.

CLI::

    python -m testbed.eval.r0 --worlds DIR [DIR ...] --out DIR [--scales standard]

Worlds that share an `s` are pooled. Writes `r0.json`, `scores.jsonl`, `curves.md` and
`auc-vs-s.svg` into a new or empty `--out`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from testbed.eval import cv
from testbed.eval.figure import line_chart_svg
from testbed.eval.labels import assign_units, open_labels
from testbed.eval.metrics import (
    auc,
    percentile_interval,
    recall_at_fpr,
    stratified_weights,
    weighted_auc,
)
from testbed.eval.oracle import oracle
from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES as F64_NAMES
from testbed.pipeline.timeline import load_timeline
from testbed.world.families import MEASURED_TAGS, PAIRS
from testbed.world.persist import Stores

RUNGS = ("random", "detector", "f12_logistic", "f64_logistic", "f64_trees", "oracle")
FITTED = ("f12_logistic", "f64_logistic", "f64_trees")
REGIMES = ("in_family", "held_out")
TEST_PAIRS = {"A": (4, 5, 6), "B": (1, 2, 3)}  # fold A trains on 1,2,3 and tests on 4,5,6

# Named seeds (AGENTS.md), distinct from health's (20261021-23) and the vendored CV seed.
CV_SEED = 20261031  # fold assignment of the in-family regime
RANDOM_SEED = 20261032  # the random rung
BOOT_SEED = 20261033  # the group bootstrap

N_BOOT = 2000
FOLDS = 5
FPRS = (0.05, 0.10)
ABUSIVE = "abusive"
LEGITIMATE = "legitimate"
# rung -> (feature set, `cv.models()` key)
FITTED_MODELS = {
    "f12_logistic": ("f12", "logistic_l2"),
    "f64_logistic": ("f64", "logistic_l2"),
    "f64_trees": ("f64", "hist_gradient_boosting"),
}
PAIR_OF_FAMILY = {family: pair for pair, names in PAIRS.items() for family in names}
CHANNELS = ("T", "F", "M")

Scores = Mapping[tuple[str, str], NDArray[np.float64]]


@dataclass(frozen=True)
class R0Unit:
    world: str  # world directory name
    group: str  # f"{world}:{group_id}", unique across pooled worlds
    family: str
    pair: int
    positive: bool  # label == "abusive"
    f12: tuple[float, ...]  # F_IDS order
    f64: tuple[float, ...]  # F64_NAMES order
    detector: float
    oracle: float | None  # undefined for some timelines (testbed.eval.oracle)
    unit: str = ""  # the units.jsonl key, which is also the renders.jsonl key and the embedding id


def side_of(pair: int) -> str:
    """The test side whose held-out fold scores `pair`: A for pairs 4-6, B for pairs 1-3."""
    if pair in TEST_PAIRS["A"]:
        return "A"
    if pair in TEST_PAIRS["B"]:
        return "B"
    raise ValueError(f"unknown pair {pair!r}")


def _other(side: str) -> str:
    return "B" if side == "A" else "A"


def t_pairs(side: str) -> tuple[int, ...]:
    """The side's test pairs whose channel is T (`MEASURED_TAGS`)."""
    return tuple(p for p in TEST_PAIRS[side] if MEASURED_TAGS[p] == "T")


def _vector(
    value: Mapping[str, float] | Sequence[float], names: Sequence[str]
) -> tuple[float, ...]:
    """A units.jsonl feature entry as floats in `names` order (a dict, or an ordered list)."""
    if isinstance(value, Mapping):
        return tuple(float(value[name]) for name in names)
    if len(value) != len(names):
        raise ValueError(f"expected {len(names)} features, got {len(value)}")
    return tuple(float(v) for v in value)


# --- Loading --------------------------------------------------------------------------------------


def load_world(
    world: Path, *, scales: Sequence[str] = ("standard",)
) -> tuple[float, list[R0Unit], int]:
    """(s, units sorted by group, mixed-unit count) for one world directory.

    Raises if the world's scale isn't in `scales`, if a unit's label disagrees with its family,
    or if one group came out as two units (SPEC §5.5 expects one)."""
    world = world.resolve()
    manifest = json.loads((world / "manifest.json").read_text())
    if manifest["scale"] not in scales:
        raise ValueError(f"{world}: scale {manifest['scale']!r} is not one of {list(scales)}")
    stores = Stores.under(world)
    labels = open_labels(stores)
    rows = [json.loads(line) for line in (world / "units.jsonl").read_text().splitlines() if line]
    by_unit = {r["unit"]: r for r in rows}
    assigned = assign_units(((r["unit"], r["members"]) for r in rows), labels.group_of)
    units: list[R0Unit] = []
    with closing(sqlite3.connect(f"file:{stores.inference_db}?mode=ro", uri=True)) as conn:
        for unit, group in assigned.group_of_unit.items():
            family = labels.family[group]
            pair = PAIR_OF_FAMILY[family]
            positive = labels.label[group] == ABUSIVE
            if positive != (family == PAIRS[pair][0]):
                raise ValueError(f"group {group}: label {labels.label[group]} vs family {family}")
            row = by_unit[unit]
            units.append(
                R0Unit(
                    world.name,
                    f"{world.name}:{group}",
                    family,
                    pair,
                    positive,
                    _vector(row["f12"], F_IDS),
                    _vector(row["f64"], F64_NAMES),
                    float(row["detector_score"]),
                    oracle(pair, load_timeline(conn, row["members"])),
                    unit,
                )
            )
    units.sort(key=lambda u: u.group)
    groups = [u.group for u in units]
    if len(set(groups)) != len(groups):
        raise ValueError(f"{world}: a group came out as more than one unit")
    return float(manifest["s"]), units, len(assigned.mixed)


# --- Scores ---------------------------------------------------------------------------------------


def _oracle_scores(units: Sequence[R0Unit]) -> NDArray[np.float64]:
    """The oracle statistic, each undefined value one below its pair's smallest defined value
    (`health.fill_undefined`, applied per pair), or 0 if the pair has none defined."""
    out = np.zeros(len(units))
    for pair in sorted({u.pair for u in units}):
        index = [i for i, u in enumerate(units) if u.pair == pair]
        defined = [v for i in index if (v := units[i].oracle) is not None]
        fill = min(defined) - 1.0 if defined else 0.0
        for i in index:
            value = units[i].oracle
            out[i] = fill if value is None else value
    return out


def _held_out(
    x: NDArray[np.float64], y: NDArray[np.int64], pairs: NDArray[np.int64], model: str
) -> NDArray[np.float64]:
    """Each unit's score from the fold that tests it: fitted on the other side's pairs."""
    out = np.full(y.size, np.nan)
    for side in TEST_PAIRS:
        test = np.isin(pairs, TEST_PAIRS[side])
        train = np.isin(pairs, TEST_PAIRS[_other(side)])
        out[test] = cv.fit_predict(x[train], y[train], x[test], model=model)
    if np.isnan(out).any():
        raise ValueError("some units are in neither test side")
    return out


def score_pool(units: Sequence[R0Unit]) -> dict[tuple[str, str], NDArray[np.float64]]:
    """Scores for every (rung, regime) of the pooled units, aligned with `units`."""
    if not units:
        raise ValueError("no units")
    y = np.array([int(u.positive) for u in units], dtype=np.int64)
    pairs = np.array([u.pair for u in units], dtype=np.int64)
    groups = [u.group for u in units]
    strata = [f"{ABUSIVE if u.positive else LEGITIMATE}|{u.family}" for u in units]
    features = {
        "f12": np.array([u.f12 for u in units], dtype=np.float64),
        "f64": np.array([u.f64 for u in units], dtype=np.float64),
    }
    plain = {
        "random": np.random.default_rng(RANDOM_SEED).random(len(units)),
        "detector": np.array([u.detector for u in units], dtype=np.float64),
        "oracle": _oracle_scores(units),
    }
    out: dict[tuple[str, str], NDArray[np.float64]] = {}
    for rung, values in plain.items():
        for regime in REGIMES:
            out[(rung, regime)] = values.copy()
    for rung, (kind, model) in FITTED_MODELS.items():
        x = features[kind]
        out[(rung, "in_family")] = cv.oof_scores(
            x, y, groups=groups, strata=strata, model=model, folds=FOLDS, seed=CV_SEED
        )
        out[(rung, "held_out")] = _held_out(x, y, pairs, model)
    return out


# --- Evaluation -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Comparison:
    """Positive and negative unit indices with their shared bootstrap weights."""

    pos: NDArray[np.intp]
    neg: NDArray[np.intp]
    w_pos: NDArray[np.int64]
    w_neg: NDArray[np.int64]


def _comparison(index: NDArray[np.intp], positive: NDArray[np.bool_], what: str) -> _Comparison:
    """A comparison of the units at `index`, resampled within each class."""
    flags = positive[index]
    if flags.all() or not flags.any():
        raise ValueError(
            f"{what}: needs both classes, got {int(flags.sum())} positive of {flags.size}"
        )
    weights = stratified_weights(np.where(flags, "pos", "neg"), n_boot=N_BOOT, seed=BOOT_SEED)
    return _Comparison(index[flags], index[~flags], weights[:, flags], weights[:, ~flags])


def _recalls(score: NDArray[np.float64], c: _Comparison) -> dict[str, float]:
    return {
        f"recall_fpr{round(f * 100):02d}": recall_at_fpr(score[c.pos], score[c.neg], f)
        for f in FPRS
    }


def _auc_entry(score: NDArray[np.float64], c: _Comparison, *, recall: bool) -> dict[str, Any]:
    boot = weighted_auc(score[c.pos], score[c.neg], c.w_pos, c.w_neg)
    lo, hi = percentile_interval(boot)
    entry: dict[str, Any] = {
        "auc": auc(score[c.pos], score[c.neg]),
        "lo": lo,
        "hi": hi,
        "n_pos": int(c.pos.size),
        "n_neg": int(c.neg.size),
    }
    return {**entry, **_recalls(score, c)} if recall else entry


def _mean_entry(values: Sequence[float], pairs: Sequence[int], c: Mapping[int, _Comparison]) -> Any:
    """The oracle's stand-in for a pooled AUC: the mean of its family AUCs, with no interval."""
    return {
        "mean_family_auc": float(np.mean(values)),
        "families": list(pairs),
        "n_pos": sum(int(c[p].pos.size) for p in pairs),
        "n_neg": sum(int(c[p].neg.size) for p in pairs),
    }


def _difference(a: NDArray[np.float64], b: NDArray[np.float64], point: float) -> dict[str, Any]:
    lo, hi = percentile_interval(a - b)
    return {"point": point, "lo": lo, "hi": hi}


def primary_comparisons(units: Sequence[R0Unit]) -> dict[str, _Comparison]:
    """The primary metric's two comparisons, by side ("A", "B"): the side's T units, with one
    `stratified_weights` draw over (side, class) strata from `BOOT_SEED`.

    Raises if a side lacks a class."""
    pair = np.array([u.pair for u in units], dtype=np.int64)
    positive = np.array([u.positive for u in units], dtype=bool)
    everyone = np.arange(len(units))
    sides = list(TEST_PAIRS)
    t_units = everyone[np.isin(pair, [p for s in sides for p in t_pairs(s)])]
    in_side = np.array([side_of(int(p)) for p in pair[t_units]])
    flags = positive[t_units]
    for side in sides:
        mask = in_side == side
        if not flags[mask].any() or flags[mask].all():
            raise ValueError(f"primary: side {side} needs both classes")
    weights = stratified_weights(
        np.array([f"{s}|{'pos' if f else 'neg'}" for s, f in zip(in_side, flags, strict=True)]),
        n_boot=N_BOOT,
        seed=BOOT_SEED,
    )
    primary: dict[str, _Comparison] = {}
    for side in sides:
        pos_cols = (in_side == side) & flags
        neg_cols = (in_side == side) & ~flags
        primary[side] = _Comparison(
            t_units[pos_cols], t_units[neg_cols], weights[:, pos_cols], weights[:, neg_cols]
        )
    return primary


def primary_boot(
    score: NDArray[np.float64], comps: Mapping[str, _Comparison]
) -> tuple[float, NDArray[np.float64]]:
    """(the mean over sides of the side AUCs, the mean over sides of the `weighted_auc`
    resamples) of one score vector against `primary_comparisons`."""
    side_boot = [weighted_auc(score[c.pos], score[c.neg], c.w_pos, c.w_neg) for c in comps.values()]
    point = float(np.mean([auc(score[c.pos], score[c.neg]) for c in comps.values()]))
    return point, np.mean(side_boot, axis=0)


def evaluate(
    units: Sequence[R0Unit],
    scores: Scores,
    *,
    rungs: Sequence[str] = RUNGS,
    extra_differences: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """The SPEC §6.4 report for one pooled set of units.

    `scores` holds exactly `rungs` x `REGIMES`; `rungs` must include `RUNGS`' fitted rungs and the
    detector, whose differences are always reported. Each `(a, b)` of `extra_differences` adds
    `differences[regime]["a_minus_b"]`, paired on the same resamples. "oracle" is special-cased,
    as it only ranks within a pair. Raises, rather than return NaN, if a comparison lacks a class
    or a pair."""
    expected = {(rung, regime) for rung in rungs for regime in REGIMES}
    if set(scores) != expected:
        raise ValueError(f"scores must hold exactly the rung x regime keys; got {sorted(scores)}")
    if not {"detector", *FITTED} <= set(rungs):
        raise ValueError("rungs must include the detector and the fitted R0 rungs")
    for a, b in extra_differences:
        if "oracle" in (a, b) or not {a, b} <= set(rungs):
            raise ValueError(f"extra difference {a} - {b} needs two real rungs of `rungs`")
    n = len(units)
    for key, values in scores.items():
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"scores for {key} must be {n} finite values")
    pair = np.array([u.pair for u in units], dtype=np.int64)
    positive = np.array([u.positive for u in units], dtype=bool)
    everyone = np.arange(n)

    family = {p: _comparison(everyone[pair == p], positive, f"family:{p}") for p in PAIRS}
    channel = {
        tag: _comparison(
            everyone[np.isin(pair, [p for p in PAIRS if MEASURED_TAGS[p] == tag])],
            positive,
            f"channel:{tag}",
        )
        for tag in CHANNELS
    }
    pooled = _comparison(everyone, positive, "pooled")
    primary = primary_comparisons(units)
    sides = list(primary)

    report: dict[str, Any] = {
        "n_units": n,
        "n_boot": N_BOOT,
        "seeds": {"cv": CV_SEED, "random": RANDOM_SEED, "boot": BOOT_SEED},
        "t_pairs": {side: list(t_pairs(side)) for side in sides},
        "regimes": {},
    }
    boots: dict[tuple[str, str], NDArray[np.float64]] = {}  # primary resamples of every real rung
    for regime in REGIMES:
        report["regimes"][regime] = {}
        for rung in rungs:
            score = scores[(rung, regime)]
            out: dict[str, Any] = {}
            if rung == "oracle":
                fam_auc = {p: _auc_entry(score, family[p], recall=False) for p in PAIRS}
                for p in PAIRS:
                    out[f"family:{p}"] = fam_auc[p]
                by_pair = {p: fam_auc[p]["auc"] for p in PAIRS}
                side_pairs = {s: t_pairs(s) for s in sides}
                all_t = [p for s in sides for p in side_pairs[s]]
                out["primary"] = {
                    **_mean_entry([by_pair[p] for p in all_t], all_t, family),
                    "sides": {
                        s: _mean_entry([by_pair[p] for p in side_pairs[s]], side_pairs[s], family)
                        for s in sides
                    },
                }
                for tag in CHANNELS:
                    covered = [p for p in PAIRS if MEASURED_TAGS[p] == tag]
                    out[f"channel:{tag}"] = _mean_entry(
                        [by_pair[p] for p in covered], covered, family
                    )
                out["pooled"] = _mean_entry([by_pair[p] for p in PAIRS], list(PAIRS), family)
            else:
                _, boot = primary_boot(score, primary)
                boots[(rung, regime)] = boot
                side_entry = {
                    s: {
                        "auc": auc(score[primary[s].pos], score[primary[s].neg]),
                        "n_pos": int(primary[s].pos.size),
                        "n_neg": int(primary[s].neg.size),
                        **_recalls(score, primary[s]),
                    }
                    for s in sides
                }
                lo, hi = percentile_interval(boot)
                out["primary"] = {
                    "auc": float(np.mean([side_entry[s]["auc"] for s in sides])),
                    "lo": lo,
                    "hi": hi,
                    "n_pos": sum(e["n_pos"] for e in side_entry.values()),
                    "n_neg": sum(e["n_neg"] for e in side_entry.values()),
                    **{
                        key: float(np.mean([side_entry[s][key] for s in sides]))
                        for key in side_entry["A"]
                        if key.startswith("recall_")
                    },
                    "sides": side_entry,
                }
                for p in PAIRS:
                    out[f"family:{p}"] = _auc_entry(score, family[p], recall=False)
                for tag in CHANNELS:
                    out[f"channel:{tag}"] = _auc_entry(score, channel[tag], recall=False)
                out["pooled"] = _auc_entry(score, pooled, recall=True)
            report["regimes"][regime][rung] = out

    report["differences"] = {}
    for regime in REGIMES:
        point = {
            r: report["regimes"][regime][r]["primary"]["auc"]
            for r in (*FITTED, *rungs)
            if r != "oracle"
        }
        n_counts = {
            "n_pos": sum(int(c.pos.size) for c in primary.values()),
            "n_neg": sum(int(c.neg.size) for c in primary.values()),
        }
        stronger = max(("f64_logistic", "f64_trees"), key=lambda r: point[r])
        report["differences"][regime] = {
            "f64_trees_minus_f64_logistic": {
                **_difference(
                    boots[("f64_trees", regime)],
                    boots[("f64_logistic", regime)],
                    point["f64_trees"] - point["f64_logistic"],
                ),
                **n_counts,
            },
            "stronger_f64_minus_detector": {
                "f64": stronger,
                **_difference(
                    boots[(stronger, regime)],
                    boots[("detector", regime)],
                    point[stronger] - point["detector"],
                ),
                **n_counts,
            },
        }
        for a, b in extra_differences:
            report["differences"][regime][f"{a}_minus_{b}"] = {
                **_difference(boots[(a, regime)], boots[(b, regime)], point[a] - point[b]),
                **n_counts,
            }
    return report


# --- Run ------------------------------------------------------------------------------------------


def _s_key(s: float) -> str:
    return str(float(s))


def _write_scores(
    path: Path,
    by_s: Mapping[float, tuple[Sequence[R0Unit], Scores]],
    rungs: Sequence[str] = RUNGS,
) -> None:
    with path.open("w") as f:
        for s, (units, scores) in sorted(by_s.items()):
            for i, u in enumerate(units):
                row = {
                    "s": s,
                    "world": u.world,
                    "group": u.group,
                    "pair": u.pair,
                    "family": u.family,
                    "positive": u.positive,
                    "side": side_of(u.pair),
                    "scores": {
                        rung: {regime: float(scores[(rung, regime)][i]) for regime in REGIMES}
                        for rung in rungs
                    },
                }
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _cell(entry: Mapping[str, Any]) -> str:
    if "mean_family_auc" in entry:
        return f"{entry['mean_family_auc']:.3f} (mean of families)"
    return f"{entry['auc']:.3f} [{entry['lo']:.3f}, {entry['hi']:.3f}]"


def _point(entry: Mapping[str, Any]) -> float:
    """An entry's AUC, or the oracle's mean of family AUCs."""
    return float(entry["mean_family_auc"] if "mean_family_auc" in entry else entry["auc"])


def curves_markdown(results: Mapping[str, Mapping[str, Any]]) -> str:
    """The primary metric per rung x regime against s, then per-family AUCs per rung at each s."""
    keys = sorted(results, key=float)
    lines = [
        "# R0 curves",
        "",
        "AUC with its 95% group-bootstrap interval. The oracle row is the mean of its",
        "per-family AUCs. Random is 0.5.",
    ]
    for regime in REGIMES:
        lines += [
            "",
            f"## Primary metric, {regime} (mean over sides A and B of the T-family AUC)",
            "",
        ]
        lines.append("| rung | " + " | ".join(f"s = {k}" for k in keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for rung in RUNGS:
            cells = [_cell(results[k]["regimes"][regime][rung]["primary"]) for k in keys]
            lines.append(f"| {rung} | " + " | ".join(cells) + " |")
    for k in keys:
        for regime in REGIMES:
            lines += ["", f"## Per-family AUC, s = {k}, {regime}", ""]
            lines.append("| rung | " + " | ".join(f"pair {p}" for p in PAIRS) + " |")
            lines.append("|---|" + "---|" * len(PAIRS))
            for rung in RUNGS:
                row = results[k]["regimes"][regime][rung]
                cells = [f"{row[f'family:{p}']['auc']:.3f}" for p in PAIRS]
                lines.append(f"| {rung} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Pairs: " + "; ".join(f"{p} = {a} vs {b}" for p, (a, b) in PAIRS.items()) + ".")
    return "\n".join(lines) + "\n"


def curves_svg(results: Mapping[str, Mapping[str, Any]]) -> str:
    """The held-out primary metric against s, one line per rung."""
    series: dict[str, list[tuple[float, float, float | None, float | None]]] = {}
    for k in sorted(results, key=float):
        for rung in RUNGS:
            entry = results[k]["regimes"]["held_out"][rung]["primary"]
            if rung == "oracle":
                point = (float(k), entry["mean_family_auc"], None, None)
                series.setdefault("oracle (mean of families)", []).append(point)
            else:
                series.setdefault(rung, []).append(
                    (float(k), entry["auc"], entry["lo"], entry["hi"])
                )
    return line_chart_svg(
        series,
        title="R0: T-family AUC against s, held-out regime",
        x_label="signal strength s",
        y_label="AUC (mean of sides A and B)",
    )


def run(
    worlds: Sequence[Path], out: Path, *, scales: Sequence[str] = ("standard",)
) -> dict[str, Any]:
    """Score and report each `s` over its pooled worlds; write the four files into `out`."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    names = [w.resolve().name for w in worlds]
    if len(set(names)) != len(names):
        raise ValueError(f"world directory names must be unique: {names}")
    pooled: dict[float, list[R0Unit]] = defaultdict(list)
    members: dict[float, list[str]] = defaultdict(list)
    mixed: dict[float, dict[str, int]] = defaultdict(dict)
    for world in worlds:
        s, units, n_mixed = load_world(world, scales=scales)
        pooled[s].extend(units)
        members[s].append(world.resolve().name)
        mixed[s][world.resolve().name] = n_mixed
    scored: dict[float, tuple[list[R0Unit], Scores]] = {}
    results: dict[str, dict[str, Any]] = {}
    for s in sorted(pooled):
        units = sorted(pooled[s], key=lambda u: u.group)
        scores = score_pool(units)
        scored[s] = (units, scores)
        results[_s_key(s)] = {
            "s": s,
            "worlds": members[s],
            "mixed": mixed[s],
            **evaluate(units, scores),
        }
    out.mkdir(parents=True, exist_ok=True)
    report = {"scales": list(scales), "by_s": results}
    (out / "r0.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_scores(out / "scores.jsonl", scored)
    (out / "curves.md").write_text(curves_markdown(results))
    (out / "auc-vs-s.svg").write_text(curves_svg(results))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--worlds", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=["standard"])
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = run(args.worlds, args.out, scales=args.scales)
    print(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")
    for k, result in report["by_s"].items():
        for regime in REGIMES:
            rungs = result["regimes"][regime]
            cells = ", ".join(f"{rung} {_point(rungs[rung]['primary']):.3f}" for rung in RUNGS)
            print(f"s = {k} {regime} primary: {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
