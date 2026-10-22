"""M8 Haiku pilot: the job builder (`select`) and scorer (`score`) (context/m8-haiku-kickoff.md).

    python -m testbed.eval.haiku select --mode pilot|check --runs runs --out runs/m8-pilot
    python -m testbed.eval.haiku score --job runs/m8-pilot --raw runs/m8-pilot/raw \
        --runs runs --out runs/m8-pilot/scored

`select` reads labels, so it lives here. It picks the six few-shot examples per side from the
tuning world and the scored units from the pilot world (mode `pilot`: every abusive T unit and 40
seeded legitimate ones per T pair), or four tuning-world units for the 24-call end-to-end check
(mode `check`). It writes `jobs.jsonl`, one row per call in one seeded random order with no label
field, which the label-free driver (`testbed.models.haiku_run`) reads, and `select.json` (the
choices, with labels, and the job file's sha256).

`score` checks the driver's raw output against the job (hashes, the route and model, one CLI
version that didn't change during the run, completeness, each attempt re-classified with the
driver's `classify`), turns draws into unit scores (the mean of the parsed draws, 50 if none
parsed), and reports G2's primary metric (SPEC §6.4: held-out regime, T families, mean of sides A
and B) for each variant, paired with F64 logistic and trees on G2's resamples
(`r0.primary_comparisons`, `N_BOOT`, `BOOT_SEED`). GO means some variant's lower bound at
`GO_LEVEL` (g2's 1.25th percentile, Bonferroni over the two variants) is above 0.5. Draw-1 AUC,
per-pair AUCs, retest reliability, the parse rate and the token sums (all four usage fields, input
split by cache) are reported, not gating. The metric core is a pure function of `R0Unit` lists and
score arrays. `--out` must not exist: the pilot is scored once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from testbed.eval import g2, r0
from testbed.eval.metrics import percentile_interval
from testbed.eval.r0 import R0Unit
from testbed.models import haiku_prompt as hp
from testbed.models import haiku_run as hr
from testbed.world.families import PAIRS

PILOT_WORLD = "m1-dev-s0.75-seed1"
TUNE_WORLD = "m4-tune-s0.75-seed301"  # the few-shot examples and the check units come from here
K = 3  # draws per unit and variant (the owner's decision, 2026-10-22)
VARIANTS = ("zero_shot", "few_shot")
N_LEGIT_PER_PAIR = 40
# Named seeds (AGENTS.md), distinct from health's (20261021-23), r0's (20261031-33) and R2's.
EXAMPLE_SEED = 20261081
SUBSET_SEED = 20261082
ORDER_SEED = 20261083
CHECK_SEED = 20261084
GO_LEVEL = g2.GATE_LEVEL  # the lower bound is the 1.25th percentile: Bonferroni over two variants
GO_THRESHOLD = 0.5
F64_POOL_PILOT = g2.GATE_WORLDS[0.75]  # the pilot world is one of G2's two s = 0.75 gate worlds
UNPARSED_SCORE = 50.0  # a unit with no parsed draw (a coin flip on the 0-100 scale)
MODES = ("pilot", "check")
DRAW_PAIRS = ((0, 1), (0, 2), (1, 2))  # the draw pairs of the reliability block
PRESENTATION = ((0, True), (1, False), (2, True), (0, False), (1, True), (2, False))
# (index into the side's example pairs, abusive): the fixed, interleaved order of the examples
F64_RUNGS = g2.F64_RUNGS
# usage.input_tokens is only the uncached part of the input (~2 per call); the rest is in the cache
# fields, so the token sums report all of them.
INPUT_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
USAGE_FIELDS = (*INPUT_FIELDS, "output_tokens")


class ScoreRefusal(Exception):
    """An input check failed before any metric: exit 2."""


# --- Selection ------------------------------------------------------------------------------------


def other_side(side: str) -> str:
    return "B" if side == "A" else "A"


def pick_examples(tune_units: Sequence[R0Unit]) -> dict[str, list[R0Unit]]:
    """Each side's six examples, in presentation order.

    Side S's example pairs are the other side's test pairs (so a pair is never shown on the side
    that tests it). One abusive and one legitimate unit per pair, M and F pairs included, drawn
    from candidates sorted by unit id with one `EXAMPLE_SEED` stream, pair ascending and abusive
    before legitimate."""
    rng = np.random.default_rng(EXAMPLE_SEED)
    picked: dict[tuple[int, bool], R0Unit] = {}
    for pair in sorted(PAIRS):
        for positive in (True, False):
            candidates = sorted(
                (u for u in tune_units if u.pair == pair and u.positive == positive),
                key=lambda u: u.unit,
            )
            if not candidates:
                raise ValueError(f"pair {pair}: no {'abusive' if positive else 'legitimate'} unit")
            picked[(pair, positive)] = candidates[int(rng.integers(len(candidates)))]
    out: dict[str, list[R0Unit]] = {}
    for side in r0.TEST_PAIRS:
        pairs = r0.TEST_PAIRS[other_side(side)]
        out[side] = [picked[(pairs[i], positive)] for i, positive in PRESENTATION]
    return out


def pilot_units(units: Sequence[R0Unit]) -> list[R0Unit]:
    """Every abusive unit of the T pairs, and `N_LEGIT_PER_PAIR` seeded legitimate ones per T pair
    (candidates sorted by unit id, T pairs ascending), sorted by unit id."""
    t_pairs = sorted(p for side in r0.TEST_PAIRS for p in r0.t_pairs(side))
    chosen = [u for u in units if u.pair in t_pairs and u.positive]
    rng = np.random.default_rng(SUBSET_SEED)
    for pair in t_pairs:
        candidates = sorted((u for u in units if u.pair == pair and not u.positive), key=_unit_id)
        index = rng.choice(len(candidates), N_LEGIT_PER_PAIR, replace=False)
        chosen += [candidates[i] for i in sorted(int(i) for i in index)]
    return sorted(chosen, key=_unit_id)


def check_units(tune_units: Sequence[R0Unit], exclude: set[str]) -> list[R0Unit]:
    """The end-to-end check's four units: per side (A, then B), one abusive and one legitimate
    unit of that side's T pairs, never one of `exclude` (the example units)."""
    rng = np.random.default_rng(CHECK_SEED)
    chosen = []
    for side in r0.TEST_PAIRS:
        for positive in (True, False):
            candidates = sorted(
                (
                    u
                    for u in tune_units
                    if u.pair in r0.t_pairs(side)
                    and u.positive == positive
                    and u.unit not in exclude
                ),
                key=_unit_id,
            )
            if not candidates:
                raise ValueError(f"side {side}: no candidate for the check")
            chosen.append(candidates[int(rng.integers(len(candidates)))])
    return chosen


def _unit_id(unit: R0Unit) -> str:
    return unit.unit


def build_jobs(
    units: Sequence[R0Unit],
    examples: Mapping[str, Sequence[R0Unit]],
    renders: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Every unit x variant x draw as a job row, in one `ORDER_SEED` permutation of the keys' sorted
    order. `renders[world][unit]` is a render's text; examples come from `TUNE_WORLD`."""
    shots = {
        side: [(renders[e.world][e.unit], e.positive) for e in examples[side]] for side in examples
    }
    rows: list[dict[str, Any]] = []
    for u in units:
        render = renders[u.world][u.unit]
        users = {
            "zero_shot": hp.zero_shot_user(render),
            "few_shot": hp.few_shot_user(shots[r0.side_of(u.pair)], render),
        }
        for variant in VARIANTS:
            for draw in range(1, K + 1):
                key = f"{u.unit}.{variant}.d{draw}"
                if not hr.KEY_PATTERN.fullmatch(key):
                    raise ValueError(f"key {key!r} is not filename-safe")
                rows.append(
                    {
                        "key": key,
                        "world": u.world,
                        "unit": u.unit,
                        "variant": variant,
                        "draw": draw,
                        "prompt_sha256": hp.PROMPT_SHA256,
                        "user": users[variant],
                        "user_sha256": hr.sha256_text(users[variant]),
                    }
                )
    rows.sort(key=lambda r: r["key"])
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("duplicate job keys")
    order = np.random.default_rng(ORDER_SEED).permutation(len(rows))
    return [rows[int(i)] for i in order]


def class_of(unit: R0Unit) -> str:
    return r0.ABUSIVE if unit.positive else r0.LEGITIMATE


def build_selection(
    mode: str,
    tune_units: Sequence[R0Unit],
    pilot_world_units: Sequence[R0Unit],
    renders: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(job rows, select.json without its jobs_sha256), from loaded worlds. No I/O."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    examples = pick_examples(tune_units)
    example_ids = {e.unit for side in examples.values() for e in side}
    if mode == "pilot":
        units = pilot_units(pilot_world_units)
        world, f64_pool = PILOT_WORLD, list(F64_POOL_PILOT)
    else:
        units = check_units(tune_units, example_ids)
        world, f64_pool = TUNE_WORLD, [TUNE_WORLD]
    rows = build_jobs(units, examples, renders)
    per_pair: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for u in units:
        per_pair[str(u.pair)][class_of(u)] += 1
    meta = {
        "mode": mode,
        "prompt_version": hp.PROMPT_VERSION,
        "prompt_sha256": hp.PROMPT_SHA256,
        "k": K,
        "variants": list(VARIANTS),
        "seeds": {
            "example": EXAMPLE_SEED,
            "subset": SUBSET_SEED,
            "order": ORDER_SEED,
            "check": CHECK_SEED,
        },
        "world": world,
        "tune_world": TUNE_WORLD,
        "f64_pool": f64_pool,
        "n_calls": len(rows),
        "counts": {
            "units_per_pair": {p: dict(c) for p, c in sorted(per_pair.items())},
            "examples_per_side": {side: len(shots) for side, shots in examples.items()},
        },
        "examples": {
            side: [{"unit": e.unit, "pair": e.pair, "class": class_of(e)} for e in shots]
            for side, shots in examples.items()
        },
        "units": sorted(u.unit for u in units),
    }
    return rows, meta


def load_renders(world: Path) -> dict[str, str]:
    rows = [json.loads(x) for x in (world / "renders.jsonl").read_text().splitlines() if x.strip()]
    return {r["unit"]: r["text"] for r in rows}


def select(mode: str, runs: Path, out: Path) -> dict[str, Any]:
    """Load the worlds, choose, and write `jobs.jsonl` and `select.json` into a new/empty `out`."""
    if hp.prompt_sha256() != hp.PROMPT_SHA256:
        raise RuntimeError("haiku_prompt.py text differs from PROMPT_SHA256")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    _, tune_units, _ = r0.load_world(runs / TUNE_WORLD)
    pilot_world_units: list[R0Unit] = []
    if mode == "pilot":
        _, pilot_world_units, _ = r0.load_world(runs / PILOT_WORLD)
    renders = {TUNE_WORLD: load_renders(runs / TUNE_WORLD)}
    if mode == "pilot":
        renders[PILOT_WORLD] = load_renders(runs / PILOT_WORLD)
    rows, meta = build_selection(mode, tune_units, pilot_world_units, renders)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    meta["jobs_sha256"] = hr.sha256_text(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.jsonl").write_text(text)
    (out / "select.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


# --- Unit scores ----------------------------------------------------------------------------------


def unit_scores(
    draws: Sequence[Sequence[int | None]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(the mean of each unit's parsed draws, draw 1 alone); `UNPARSED_SCORE` where none parsed
    (resp. draw 1 did not). `draws[i][j]` is draw j + 1 of unit i, None if it ended unparsed."""
    mean = np.array(
        [
            float(np.mean(parsed))
            if (parsed := [d for d in row if d is not None])
            else UNPARSED_SCORE
            for row in draws
        ]
    )
    first = np.array([UNPARSED_SCORE if row[0] is None else float(row[0]) for row in draws])
    return mean, first


# --- Metric core ----------------------------------------------------------------------------------


def _entry(point: float, boot: NDArray[np.float64]) -> dict[str, float]:
    lo, hi = percentile_interval(boot)
    return {"point": point, "lo": lo, "hi": hi}


def pair_aucs(units: Sequence[R0Unit], score: NDArray[np.float64]) -> dict[str, Any]:
    """The AUC of each T pair present among `units`, with `r0._auc_entry`'s 95% interval; null with
    a reason where the pair lacks a class."""
    pair = np.array([u.pair for u in units], dtype=np.int64)
    positive = np.array([u.positive for u in units], dtype=bool)
    out: dict[str, Any] = {}
    for p in sorted(p for side in r0.TEST_PAIRS for p in r0.t_pairs(side)):
        index = np.flatnonzero(pair == p)
        if index.size == 0:
            continue
        flags = positive[index]
        if flags.all() or not flags.any():
            out[str(p)] = {"auc": None, "reason": f"{int(flags.sum())} abusive of {flags.size}"}
        else:
            comparison = r0._comparison(index, positive, f"pair {p}")
            out[str(p)] = r0._auc_entry(score, comparison, recall=False)
    return out


def _ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Midranks (ties share the mean rank)."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    return np.asarray(((last - counts + 1 + last) / 2)[inverse], dtype=np.float64)


def pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> float | None:
    """Pearson r, or None for fewer than 3 points or a constant input."""
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float | None:
    return pearson(_ranks(a), _ranks(b)) if a.size >= 3 else None


def spearman_brown(r: float | None, k: int = K) -> float | None:
    """The reliability of the mean of `k` draws from the single-draw correlation `r`."""
    if r is None or 1 + (k - 1) * r == 0:
        return None
    return k * r / (1 + (k - 1) * r)


def _mean(values: Sequence[float | None]) -> float | None:
    return None if any(v is None for v in values) else float(np.mean(values))  # type: ignore[arg-type]


def reliability(draws: NDArray[np.float64]) -> dict[str, Any]:
    """Retest agreement between draws over the units with every draw parsed. `draws` is
    (units, K) with NaN where a draw is unparsed."""
    full = draws[~np.isnan(draws).any(axis=1)]
    pairs: dict[str, Any] = {}
    for i, j in DRAW_PAIRS:
        pairs[f"{i + 1}-{j + 1}"] = {
            "pearson": pearson(full[:, i], full[:, j]),
            "spearman": spearman(full[:, i], full[:, j]),
        }
    mean_pearson = _mean([v["pearson"] for v in pairs.values()])
    return {
        "n_units": int(full.shape[0]),
        "pairs": pairs,
        "mean_pearson": mean_pearson,
        "mean_spearman": _mean([v["spearman"] for v in pairs.values()]),
        "spearman_brown_k": K,
        "spearman_brown": spearman_brown(mean_pearson, K),
    }


def parse_rate(draw_statuses: Sequence[Sequence[Sequence[str]]]) -> dict[str, int]:
    """Counts for one variant. `draw_statuses[unit][draw]` is a draw's attempt statuses in order.
    A draw is "ok first try" if its ok attempt had no unparsed one before it ("error" attempts are
    infrastructure retries and don't count), "ok after retry" otherwise, else unparsed."""
    out = {"draws": 0, "ok_first_try": 0, "ok_after_retry": 0, "unparsed": 0, "units_no_parsed": 0,
           "error_attempts": 0}  # fmt: skip
    for unit in draw_statuses:
        parsed = 0
        for statuses in unit:
            out["draws"] += 1
            out["error_attempts"] += statuses.count(hr.ERROR)
            if hr.OK in statuses:
                parsed += 1
                before = statuses[: statuses.index(hr.OK)]
                out["ok_after_retry" if hr.UNPARSED in before else "ok_first_try"] += 1
            else:
                out["unparsed"] += 1
        out["units_no_parsed"] += parsed == 0
    return out


def evaluate(
    units: Sequence[R0Unit],
    scores: Mapping[str, NDArray[np.float64]],
    draw1: Mapping[str, NDArray[np.float64]],
    f64: Mapping[str, NDArray[np.float64]],
) -> dict[str, Any]:
    """The primary-metric report. `scores` and `draw1` hold each variant's unit scores and `f64`
    each F64 rung's held-out scores, all aligned with `units`. Everything is paired through one
    `r0.primary_comparisons`, so differences use the same resamples."""
    comps = r0.primary_comparisons(units)
    f64_boot = {rung: r0.primary_boot(f64[rung], comps) for rung in F64_RUNGS}
    primary: dict[str, Any] = {}
    minus: dict[str, Any] = {}
    first: dict[str, Any] = {}
    bounds: dict[str, float] = {}
    for variant, score in scores.items():
        point, boot = r0.primary_boot(score, comps)
        bounds[variant] = g2.lower_bound(boot, GO_LEVEL)
        primary[variant] = {**_entry(point, boot), "lower_bound": bounds[variant]}
        minus[variant] = {
            rung: r0._difference(boot, f_boot, point - f_point)
            for rung, (f_point, f_boot) in f64_boot.items()
        }
        first[variant] = _entry(*r0.primary_boot(draw1[variant], comps))
    go = any(bound > GO_THRESHOLD for bound in bounds.values())
    return {
        "n_units": len(units),
        "n_boot": r0.N_BOOT,
        "boot_seed": r0.BOOT_SEED,
        "primary": primary,
        "go": {
            "level": GO_LEVEL,
            "threshold": GO_THRESHOLD,
            "lower_bounds": bounds,
            "verdict": "GO" if go else "NO-GO",
        },
        "f64": {rung: _entry(point, boot) for rung, (point, boot) in f64_boot.items()},
        "haiku_minus_f64": minus,
        "draw1_primary": first,
        "per_pair_auc": {
            **{v: pair_aucs(units, s) for v, s in scores.items()},
            **{rung: pair_aucs(units, f64[rung]) for rung in F64_RUNGS},
        },
    }


# --- Raw output -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    key: str
    n: int
    status: str
    probability: int | None
    wall_s: float
    cli_version: str
    models: tuple[str, ...]
    usage: Mapping[str, int]  # `USAGE_FIELDS` of the raw's top-level usage; 0 where absent


@dataclass(frozen=True)
class Raw:
    rows: list[dict[str, Any]]
    select: dict[str, Any]
    run: dict[str, Any]
    attempts: dict[str, list[Attempt]]  # key -> committed attempts in order


def usage_of(stdout: str) -> dict[str, int]:
    """`USAGE_FIELDS` of an attempt's top-level `usage` (0 for a missing or non-integer field)."""
    try:
        usage = json.loads(stdout)["usage"]
    except ValueError, KeyError, TypeError:
        usage = None
    if not isinstance(usage, dict):
        usage = {}
    return {
        f: v if isinstance(v := usage.get(f), int) and not isinstance(v, bool) else 0
        for f in USAGE_FIELDS
    }


def load_raw(job: Path, raw: Path) -> Raw:
    """The job and the driver's output, after every pre-metric check; raises `ScoreRefusal`."""
    jobs_bytes = (job / "jobs.jsonl").read_bytes()
    jobs_sha256 = hashlib.sha256(jobs_bytes).hexdigest()
    selected = json.loads((job / "select.json").read_text())
    run = json.loads((raw / "run.json").read_text())
    if not selected["jobs_sha256"] == jobs_sha256 == run["jobs_sha256"]:
        raise ScoreRefusal("jobs.jsonl sha256 differs between the file, select.json and run.json")
    if not run["prompt_sha256"] == hp.PROMPT_SHA256 == selected["prompt_sha256"]:
        raise ScoreRefusal("prompt_sha256 differs between run.json, select.json and haiku_prompt")
    if run.get("route") != list(hp.ROUTE_FLAGS) or run.get("model") != hp.MODEL:
        raise ScoreRefusal("run.json route or model differs from haiku_prompt's")
    if "cli_version_after" in run and run["cli_version_after"] != run["cli_version"]:
        raise ScoreRefusal(
            f"CLI version changed during the run: {run['cli_version']!r} -> "
            f"{run['cli_version_after']!r}"
        )
    rows = [json.loads(x) for x in jobs_bytes.decode().splitlines() if x.strip()]
    wanted = {r["key"]: r for r in rows}
    found: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted((raw / "calls").iterdir()):
        m = hr.ATTEMPT_FILE.match(path.name)
        if m and m["meta"] and m["key"] in wanted:
            found[m["key"]][int(m["n"])] = path
    attempts: dict[str, list[Attempt]] = {}
    for key, row in wanted.items():
        attempts[key] = []
        for n, meta_path in sorted(found[key].items()):
            meta = json.loads(meta_path.read_text())
            raw_path = meta_path.with_name(f"{key}.a{n}.json")
            if not raw_path.exists():
                raise ScoreRefusal(f"{key} a{n}: meta without its raw file")
            stdout = raw_path.read_bytes().decode()
            if (meta["key"], meta["attempt"]) != (key, n):
                raise ScoreRefusal(f"{key} a{n}: meta names {meta['key']} a{meta['attempt']}")
            if (meta["status"], meta["probability"]) != hr.classify(meta["rc"], stdout):
                raise ScoreRefusal(f"{key} a{n}: meta status disagrees with classify on the raw")
            if meta["user_sha256"] != row["user_sha256"]:
                raise ScoreRefusal(f"{key} a{n}: user_sha256 differs from the job row")
            attempts[key].append(
                Attempt(
                    key,
                    n,
                    meta["status"],
                    meta["probability"],
                    float(meta["wall_s"]),
                    meta["cli_version"],
                    tuple(meta["models"]),
                    usage_of(stdout),
                )
            )
    open_keys = [k for k, a in attempts.items() if not hr.is_done([x.status for x in a])]
    if open_keys:
        started = sum(bool(attempts[k]) for k in open_keys)
        raise ScoreRefusal(
            f"{len(open_keys)} of {len(rows)} keys are not DONE ({started} started, "
            f"{len(open_keys) - started} never called), e.g. {open_keys[:3]}"
        )
    versions = {a.cli_version for attempt_list in attempts.values() for a in attempt_list}
    if len(versions) != 1 or versions != {run["cli_version"]}:
        raise ScoreRefusal(f"CLI versions {sorted(versions)} vs run.json {run['cli_version']!r}")
    return Raw(rows, selected, run, attempts)


def draw_outcome(attempts: Sequence[Attempt]) -> int | None:
    """A DONE draw's probability: its "ok" attempt's, or None if it ended unparsed."""
    for a in attempts:
        if a.status == hr.OK:
            return a.probability
    return None


# --- Score ----------------------------------------------------------------------------------------


def load_f64(
    runs: Path, pool_names: Sequence[str]
) -> tuple[list[R0Unit], dict[str, NDArray[np.float64]]]:
    """(the pooled units sorted by group, each F64 rung's held-out scores) for the worlds of
    `pool_names`. The G2 gate pool must reproduce M2, or `g2.f64_boots` raises."""
    units: list[R0Unit] = []
    worlds, mixed = [], {}
    for name in pool_names:
        s, world_units, n_mixed = r0.load_world(runs / name)
        units += world_units
        worlds.append((runs / name).resolve())
        mixed[name] = n_mixed
        if list(pool_names) == list(F64_POOL_PILOT) and s != 0.75:
            raise ScoreRefusal(f"{name}: s = {s}, expected 0.75")
    units.sort(key=lambda u: u.group)
    scores = r0.score_pool(units)
    if list(pool_names) == list(F64_POOL_PILOT):
        g2.f64_boots(g2.Pool(0.75, worlds, units, mixed), scores)
    return units, {rung: scores[(rung, "held_out")] for rung in F64_RUNGS}


def calls_block(raw: Raw) -> dict[str, Any]:
    every = [a for attempt_list in raw.attempts.values() for a in attempt_list]
    walls = [a.wall_s for a in every]
    return {
        "attempts": len(every),
        "wall_s_p50": statistics.median(walls),
        "wall_s_max": max(walls),
        "cli_version": raw.run["cli_version"],
        "models": sorted({m for a in every for m in a.models}),
        **{f: sum(a.usage[f] for a in every) for f in USAGE_FIELDS},
        "input_tokens_total": sum(a.usage[f] for a in every for f in INPUT_FIELDS),
    }


def score(job: Path, raw_dir: Path, runs: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(report, scores.jsonl rows) of a finished run. Raises `ScoreRefusal` on any failed check."""
    raw = load_raw(job, raw_dir)
    pool, f64_all = load_f64(runs, raw.select["f64_pool"])
    job_units = {(r["world"], r["unit"]) for r in raw.rows}
    if job_units != {(raw.select["world"], u) for u in raw.select["units"]}:
        raise ScoreRefusal("the job's units differ from select.json's scored units")
    keep = [i for i, u in enumerate(pool) if (u.world, u.unit) in job_units]
    missing = job_units - {(pool[i].world, pool[i].unit) for i in keep}
    if missing:
        raise ScoreRefusal(
            f"{len(missing)} job units are not in the F64 pool, e.g. {sorted(missing)[:3]}"
        )
    units = [pool[i] for i in keep]
    f64 = {rung: f64_all[rung][keep] for rung in F64_RUNGS}

    key_of = {(r["world"], r["unit"], r["variant"], r["draw"]): r["key"] for r in raw.rows}
    draws: dict[str, list[list[int | None]]] = {}
    statuses: dict[str, list[list[list[str]]]] = {}
    for variant in VARIANTS:
        keys = [[key_of[(u.world, u.unit, variant, d)] for d in range(1, K + 1)] for u in units]
        draws[variant] = [[draw_outcome(raw.attempts[k]) for k in row] for row in keys]
        statuses[variant] = [[[a.status for a in raw.attempts[k]] for k in row] for row in keys]
    means, firsts = {}, {}
    for variant in VARIANTS:
        means[variant], firsts[variant] = unit_scores(draws[variant])
    report = evaluate(units, means, firsts, f64)
    report["reliability"] = {
        v: reliability(np.array([[np.nan if d is None else d for d in row] for row in draws[v]]))
        for v in VARIANTS
    }
    report["parse_rate"] = {v: parse_rate(statuses[v]) for v in VARIANTS}
    report["calls"] = calls_block(raw)
    report["binding"] = {
        "mode": raw.select["mode"],
        "prompt_version": raw.select["prompt_version"],
        "prompt_sha256": raw.select["prompt_sha256"],
        "jobs_sha256": raw.select["jobs_sha256"],
        "k": K,
        "f64_pool": raw.select["f64_pool"],
        "unparsed_score": UNPARSED_SCORE,
    }
    rows = []
    for i, u in enumerate(units):
        row: dict[str, Any] = {
            "world": u.world,
            "unit": u.unit,
            "group": u.group,
            "family": u.family,
            "pair": u.pair,
            "positive": u.positive,
        }
        for variant in VARIANTS:
            row[variant] = {
                "draws": ["unparsed" if d is None else d for d in draws[variant][i]],
                "score": float(means[variant][i]),
                "draw1_score": float(firsts[variant][i]),
            }
        row.update({rung: float(f64[rung][i]) for rung in F64_RUNGS})
        rows.append(row)
    return report, rows


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select", help="choose units and examples; write the job file")
    pick.add_argument("--mode", choices=MODES, required=True)
    pick.add_argument("--runs", type=Path, required=True)
    pick.add_argument("--out", type=Path, required=True)
    grade = sub.add_parser("score", help="score a finished driver run, once")
    grade.add_argument("--job", type=Path, required=True)
    grade.add_argument("--raw", type=Path, required=True)
    grade.add_argument("--runs", type=Path, required=True)
    grade.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        try:
            meta = select(args.mode, args.runs, args.out)
        except FileExistsError as e:
            parser.error(str(e))
        print(f"jobs.jsonl sha256 {meta['jobs_sha256']}")
        counts = meta["counts"]
        print(f"mode {meta['mode']}: {meta['n_calls']} calls")
        print(f"units per pair and class {counts['units_per_pair']}")
        print(f"examples per side {meta['counts']['examples_per_side']}")
        return 0
    if args.out.exists():
        parser.error(f"{args.out} exists; the pilot is scored once")
    try:
        report, rows = score(args.job, args.raw, args.runs)
    except ScoreRefusal as e:
        print(f"refused: {e}", file=sys.stderr)
        return hr.EXIT_REFUSED
    args.out.mkdir(parents=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    (args.out / "report.json").write_text(text + "\n")
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    go = report["go"]
    print(f"M8 pilot {go['verdict']}")
    bounds = " ".join(f"{v} {b:.4f}" for v, b in go["lower_bounds"].items())
    print(f"lower bounds at level {go['level']:.3f} (GO above {go['threshold']}): {bounds}")
    if report["binding"]["mode"] != "pilot":
        print(f"mode {report['binding']['mode']}: a dry run of the route, not a pilot result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
