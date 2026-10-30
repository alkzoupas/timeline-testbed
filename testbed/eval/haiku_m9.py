"""M9 confirmation of pair 3 with Haiku: the job builder (`select`) and scorer (`score`)
(prereg/M9.md).

    python -m testbed.eval.haiku_m9 select --mode check --runs runs --out runs/m9-check
    python -m testbed.eval.haiku_m9 select --mode confirm --runs runs --worlds W1 W2 --control C \
        --out runs/m9-confirm
    python -m testbed.eval.haiku_m9 score --job DIR --raw DIR/raw --runs runs --out DIR/scored

`select` reads labels, so it lives here. It takes the six few-shot examples of side B (pairs 4-6)
from the tuning world, as M8 did, and the scored units from the worlds: every pair 3 abusive unit
and 40 seeded pair 3 legitimate ones per world. Mode `confirm` takes two s = 0.75 worlds (both
variants) and one s = 0 control (few-shot only); mode `check` takes four pair 3 units of the
tuning world (both variants), a dry run of the route. It writes `jobs.jsonl`, one row per call in
one seeded random order across all worlds with no label field, which the label-free driver
(`testbed.models.haiku_run`) reads, and `select.json` (the choices, with labels, the job file's
sha256 and the sha256 of `prereg/M9.md`).

`score` checks the driver's raw output against the job with M8's checks (`haiku.load_raw`), turns
draws into unit scores (`haiku.unit_scores`), and reports pair 3's pooled AUC on the two s = 0.75
worlds for each variant, paired with F64 logistic and trees on one set of within-(world, class)
resamples, the control's few-shot AUC with its own resamples, and the prereg's verdict
(CONFIRMED, NOT CONFIRMED or LEAK; in mode `check`, "CHECK (not a result)"). Zero-shot, per-world
and draw-1 AUCs, retest reliability, the parse rate and the token sums are reported, not gating.
`score` refuses if `prereg/M9.md` changed since `select`; `--out` must not exist: M9 is scored once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from testbed.eval import g2, haiku, r0
from testbed.eval import metrics as mx
from testbed.eval.r0 import R0Unit
from testbed.models import haiku_prompt as hp
from testbed.models import haiku_run as hr

PAIR = 3  # the one pair M9 scores: template operator against office card (side B's test pair)
SIDE = r0.side_of(PAIR)  # "B": the examples shown are pairs 4-6
K = haiku.K
VARIANTS = haiku.VARIANTS
CONTROL_VARIANTS = ("few_shot",)
N_LEGIT = 40
S_SCORED = 0.75
S_CONTROL = 0.0
N_WORLDS = 2
# Named seeds (AGENTS.md), distinct from M8's (20261081-84), health's, r0's and R2's.
SUBSET_SEED = 20261091
ORDER_SEED = 20261092
BOOT_SEED = 20261093
CHECK_SEED = 20261094
N_BOOT = 2000
CHECK_PER_CLASS = 2
MODES = ("confirm", "check")
CHECK_VERDICT = "CHECK (not a result)"
F64_RUNGS = g2.F64_RUNGS
FORBIDDEN_WORLDS = frozenset(
    {haiku.TUNE_WORLD, haiku.PILOT_WORLD, *(w for names in g2.GATE_WORLDS.values() for w in names)}
)
M8_SELECT = "m8-pilot/select.json"  # under `runs`: the examples M9 must reproduce
PREREG_PATH = Path(__file__).resolve().parents[2] / "prereg" / "M9.md"
Entry = dict[str, float]


class SelectRefusal(RuntimeError):
    """A `select` input check failed before any job was written: exit 2."""


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


# --- Selection ------------------------------------------------------------------------------------


def _unit_id(unit: R0Unit) -> str:
    return unit.unit


def class_of(unit: R0Unit) -> str:
    return haiku.class_of(unit)


def world_rng(seed: int, world: str) -> np.random.Generator:
    """A stream from `seed` and the world's name, so the order of `--worlds` can't matter."""
    return np.random.default_rng([seed, int(hashlib.sha256(world.encode()).hexdigest()[:8], 16)])


def pair3_units(world_units: Sequence[R0Unit], world: str) -> list[R0Unit]:
    """Every pair 3 abusive unit and `N_LEGIT` seeded pair 3 legitimate ones (candidates sorted by
    unit id, one stream per world), sorted by unit id."""
    chosen = sorted((u for u in world_units if u.pair == PAIR and u.positive), key=_unit_id)
    if not chosen:
        raise SelectRefusal(f"{world}: no pair {PAIR} abusive unit")
    candidates = sorted((u for u in world_units if u.pair == PAIR and not u.positive), key=_unit_id)
    if len(candidates) < N_LEGIT:
        raise SelectRefusal(
            f"{world}: {len(candidates)} pair {PAIR} legitimate units, need {N_LEGIT}"
        )
    index = world_rng(SUBSET_SEED, world).choice(len(candidates), N_LEGIT, replace=False)
    chosen += [candidates[i] for i in sorted(int(i) for i in index)]
    return sorted(chosen, key=_unit_id)


def check_units(tune_units: Sequence[R0Unit], exclude: set[str]) -> list[R0Unit]:
    """The end-to-end check's four units: `CHECK_PER_CLASS` abusive and as many legitimate pair 3
    units of the tuning world, never one of `exclude` (the example units), sorted by unit id."""
    rng = np.random.default_rng(CHECK_SEED)
    chosen: list[R0Unit] = []
    for positive in (True, False):
        candidates = sorted(
            (
                u
                for u in tune_units
                if u.pair == PAIR and u.positive == positive and u.unit not in exclude
            ),
            key=_unit_id,
        )
        if len(candidates) < CHECK_PER_CLASS:
            raise SelectRefusal(f"check: fewer than {CHECK_PER_CLASS} candidates of one class")
        index = rng.choice(len(candidates), CHECK_PER_CLASS, replace=False)
        chosen += [candidates[i] for i in sorted(int(i) for i in index)]
    return sorted(chosen, key=_unit_id)


def build_jobs(
    cohorts: Sequence[tuple[Sequence[R0Unit], Sequence[str]]],
    examples: Sequence[R0Unit],
    renders: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Every unit x variant x draw as a job row, in one `ORDER_SEED` permutation of the keys' sorted
    order. `cohorts` is (units of one world, that world's variants); `renders[world][unit]` is a
    render's text and `examples` are side B's, in presentation order, from the tuning world."""
    shots = [(renders[e.world][e.unit], e.positive) for e in examples]
    rows: list[dict[str, Any]] = []
    for units, variants in cohorts:
        for u in units:
            render = renders[u.world][u.unit]
            users = {
                "zero_shot": hp.zero_shot_user(render),
                "few_shot": hp.few_shot_user(shots, render),
            }
            for variant in variants:
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


def build_selection(
    mode: str,
    tune_units: Sequence[R0Unit],
    worlds: Mapping[str, tuple[float, Sequence[R0Unit]]],
    control: str | None,
    renders: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(job rows, select.json without its jobs_sha256 and prereg_sha256), from loaded worlds.
    `worlds` maps each world (scored ones in mode `confirm`, then the control) to (s, its units);
    in mode `check` it is the tuning world alone. No I/O."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    examples = haiku.pick_examples(tune_units)[SIDE]
    example_ids = {e.unit for e in examples}
    scored = [w for w in worlds if w != control]
    if mode == "check":
        if control is not None or list(worlds) != [haiku.TUNE_WORLD]:
            raise ValueError("mode check takes the tuning world alone, with no control")
        picked = {haiku.TUNE_WORLD: check_units(worlds[haiku.TUNE_WORLD][1], example_ids)}
    else:
        picked = {w: pair3_units(worlds[w][1], w) for w in worlds}
    variants = {w: list(CONTROL_VARIANTS if w == control else VARIANTS) for w in worlds}
    rows = build_jobs([(picked[w], variants[w]) for w in worlds], examples, renders)
    meta = {
        "mode": mode,
        "prompt_version": hp.PROMPT_VERSION,
        "prompt_sha256": hp.PROMPT_SHA256,
        "k": K,
        "pair": PAIR,
        "variants": variants,
        "seeds": {
            "subset": SUBSET_SEED,
            "order": ORDER_SEED,
            "boot": BOOT_SEED,
            "check": CHECK_SEED,
        },
        "n_boot": N_BOOT,
        "worlds": {w: worlds[w][0] for w in worlds},
        "scored_worlds": scored,
        "control": control,
        "tune_world": haiku.TUNE_WORLD,
        "examples": {
            SIDE: [{"unit": e.unit, "pair": e.pair, "class": class_of(e)} for e in examples]
        },
        "units": {
            w: {
                "counts": {
                    cls: sum(class_of(u) == cls for u in picked[w])
                    for cls in (r0.ABUSIVE, r0.LEGITIMATE)
                },
                "ids": [u.unit for u in picked[w]],
            }
            for w in worlds
        },
        "n_calls": len(rows),
    }
    return rows, meta


def load_worlds(runs: Path, names: Sequence[str]) -> dict[str, tuple[float, list[R0Unit]]]:
    out: dict[str, tuple[float, list[R0Unit]]] = {}
    for name in names:
        s, units, _ = r0.load_world(runs / name)
        out[name] = (s, units)
    return out


def check_world_name(name: str) -> None:
    if not name or name != Path(name).name:
        raise SelectRefusal(f"{name!r} is not a bare world directory name")
    if name.startswith("confirm-"):
        raise SelectRefusal(f"{name}: runs/confirm-* is closed (M6's)")
    if name in FORBIDDEN_WORLDS:
        raise SelectRefusal(f"{name} is the tuning, pilot or a G2 gate world")


def check_health(runs: Path, name: str) -> None:
    path = runs / name / "health.json"
    if not path.is_file():
        raise SelectRefusal(f"{name}: no health.json")
    if json.loads(path.read_text()).get("passed") is not True:
        raise SelectRefusal(f"{name}: health.json does not record a pass")


def check_examples_match_m8(runs: Path, meta: Mapping[str, Any]) -> None:
    """M9's side B examples are M8's, when M8's `select.json` is there to say."""
    path = runs / M8_SELECT
    if path.is_file():
        m8 = json.loads(path.read_text())["examples"][SIDE]
        if m8 != meta["examples"][SIDE]:
            raise SelectRefusal(f"the side {SIDE} examples differ from {path}")


def select(
    mode: str, runs: Path, out: Path, worlds: Sequence[str] = (), control: str | None = None
) -> dict[str, Any]:
    """Check, load, choose, and write `jobs.jsonl` and `select.json` into a new/empty `out`."""
    if hp.prompt_sha256() != hp.PROMPT_SHA256:
        raise SelectRefusal("haiku_prompt.py text differs from PROMPT_SHA256")
    if out.exists() and any(out.iterdir()):
        raise SelectRefusal(f"{out} exists and is not empty")
    if mode not in MODES:
        raise SelectRefusal(f"mode {mode!r} is not one of {MODES}")
    if mode == "check":
        if worlds or control is not None:
            raise SelectRefusal("mode check takes no --worlds or --control")
    else:
        if len(worlds) != N_WORLDS:
            raise SelectRefusal(
                f"mode confirm needs exactly {N_WORLDS} --worlds, got {len(worlds)}"
            )
        if control is None:
            raise SelectRefusal("mode confirm needs --control")
        names = [*worlds, control]
        if len(set(names)) != len(names):
            raise SelectRefusal(f"worlds are not distinct: {names}")
        for name in names:
            check_world_name(name)
            check_health(runs, name)
    tune = load_worlds(runs, [haiku.TUNE_WORLD])[haiku.TUNE_WORLD][1]
    loaded: dict[str, tuple[float, list[R0Unit]]]
    if mode == "check":
        loaded = {haiku.TUNE_WORLD: (S_SCORED, tune)}
    else:
        # sorted, so the job file doesn't depend on the order of --worlds
        loaded = load_worlds(runs, [*sorted(worlds), str(control)])
        for name, (s, _) in loaded.items():
            want = S_CONTROL if name == control else S_SCORED
            if s != want:
                raise SelectRefusal(f"{name}: s = {s}, expected {want}")
    renders = {w: haiku.load_renders(runs / w) for w in {haiku.TUNE_WORLD, *loaded}}
    rows, meta = build_selection(mode, tune, loaded, control, renders)
    check_examples_match_m8(runs, meta)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    meta["jobs_sha256"] = hr.sha256_text(text)
    meta["prereg_sha256"] = prereg_sha256()
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.jsonl").write_text(text)
    (out / "select.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


# --- Metric core ----------------------------------------------------------------------------------


def auc(score: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    """Mann-Whitney AUC of `score` for the units flagged `positive` (ties 1/2)."""
    return mx.auc(score[positive], score[~positive])


def resample_weights(
    worlds: Sequence[str],
    positive: NDArray[np.bool_],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> NDArray[np.int64]:
    """(n_boot, n_units) counts of `n_boot` resamples of the units within (world, class), each
    stratum keeping its size. One matrix serves every score array, so differences are paired."""
    strata = [f"{w}|{'pos' if p else 'neg'}" for w, p in zip(worlds, positive, strict=True)]
    return mx.stratified_weights(strata, n_boot=n_boot, seed=seed)


def boot_auc(
    score: NDArray[np.float64], positive: NDArray[np.bool_], weights: NDArray[np.int64]
) -> NDArray[np.float64]:
    """The AUC of each resample of `weights` (n_boot, n_units)."""
    return mx.weighted_auc(
        score[positive], score[~positive], weights[:, positive], weights[:, ~positive]
    )


def entry(point: float, boot: NDArray[np.float64]) -> Entry:
    """`point` with the 2.5th and 97.5th percentiles of `boot`."""
    lo, hi = mx.percentile_interval(boot)
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def auc_entry(
    score: NDArray[np.float64], positive: NDArray[np.bool_], weights: NDArray[np.int64]
) -> tuple[Entry, NDArray[np.float64]]:
    """(the AUC's entry, its resamples)."""
    boot = boot_auc(score, positive, weights)
    return entry(auc(score, positive), boot), boot


def verdict(
    few_lo: float, diff_lo: Mapping[str, float], control: tuple[float, float] | None
) -> tuple[dict[str, bool | None], str]:
    """(the three criteria, the verdict) as prereg/M9.md states them.

    (a) the few-shot AUC's lower bound is above 0.5; (b) both few-shot minus F64 lower bounds are
    above 0; (c) the control's interval contains 0.5 (None, and not judged, without a control).
    Failing (c) is LEAK whatever (a) and (b) say; else CONFIRMED needs (a) and (b)."""
    a = few_lo > 0.5
    b = all(lo > 0 for lo in diff_lo.values())
    c = None if control is None else control[0] <= 0.5 <= control[1]
    criteria = {
        "a_few_shot_lower_above_half": a,
        "b_beats_both_f64": b,
        "c_control_contains_half": c,
    }
    if c is False:
        return criteria, "LEAK"
    return criteria, "CONFIRMED" if a and b else "NOT CONFIRMED"


def evaluate(
    mode: str,
    worlds: Sequence[str],
    positive: NDArray[np.bool_],
    scores: Mapping[str, NDArray[np.float64]],
    draw1: Mapping[str, NDArray[np.float64]],
    f64: Mapping[str, NDArray[np.float64]],
    control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The report's metrics. The arrays are aligned over the units of the two s = 0.75 worlds
    (`worlds[i]` is unit i's world): `scores` and `draw1` hold each variant's unit scores and `f64`
    each F64 rung's held-out scores. `control`, if any, holds its `positive`, few-shot `score`
    and `draw1` and the reported `f64` arrays, over the control world's units."""
    world_set = sorted(set(worlds))
    weights = resample_weights(worlds, positive)
    primary: dict[str, Entry] = {}
    boots: dict[str, NDArray[np.float64]] = {}
    for name, array in {**scores, **f64}.items():
        e, boots[name] = auc_entry(array, positive, weights)
        primary[name] = e
    minus: dict[str, dict[str, Entry]] = {
        v: {
            rung: entry(primary[v]["point"] - primary[rung]["point"], boots[v] - boots[rung])
            for rung in F64_RUNGS
        }
        for v in scores
    }
    per_world: dict[str, dict[str, Entry]] = {}
    for w in world_set:
        mask = np.array([x == w for x in worlds])
        w_weights = resample_weights([w] * int(mask.sum()), positive[mask])
        per_world[w] = {
            name: auc_entry(array[mask], positive[mask], w_weights)[0]
            for name, array in {**scores, **f64}.items()
        }
    report: dict[str, Any] = {
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "interval": "percentile, 2.5 / 97.5",
        "n_units": {
            w: {
                "abusive": int(sum(x == w and p for x, p in zip(worlds, positive, strict=True))),
                "legitimate": int(
                    sum(x == w and not p for x, p in zip(worlds, positive, strict=True))
                ),
            }
            for w in world_set
        },
        "primary": {v: primary[v] for v in scores},
        "f64": {rung: primary[rung] for rung in F64_RUNGS},
        "haiku_minus_f64": minus,
        "draw1_primary": {v: auc_entry(draw1[v], positive, weights)[0] for v in scores},
        "per_world_auc": per_world,
    }
    control_entry = None
    if control is not None:
        c_pos = control["positive"]
        c_weights = resample_weights(["control"] * len(c_pos), c_pos)
        control_entry = auc_entry(control["score"], c_pos, c_weights)[0]
        report["control"] = {
            "few_shot": control_entry,
            "draw1": auc_entry(control["draw1"], c_pos, c_weights)[0],
            "f64": {  # reported, not gating
                rung: auc_entry(control["f64"][rung], c_pos, c_weights)[0] for rung in F64_RUNGS
            },
            "n_units": {
                "abusive": int(c_pos.sum()),
                "legitimate": int((~c_pos).sum()),
            },
        }
    else:
        report["control"] = None
    criteria, decision = verdict(
        primary["few_shot"]["lo"],
        {rung: minus["few_shot"][rung]["lo"] for rung in F64_RUNGS},
        None if control_entry is None else (control_entry["lo"], control_entry["hi"]),
    )
    report["criteria"] = criteria
    report["verdict"] = CHECK_VERDICT if mode == "check" else decision
    report["verdict_if_scored"] = decision
    return report


# --- Score ----------------------------------------------------------------------------------------


def f64_held_out(units: Sequence[R0Unit]) -> dict[str, NDArray[np.float64]]:
    """Each F64 rung's held-out scores of `units`, fitted on them (pairs 4-6 train, pair 3 tests),
    aligned with `units`. Unlike M8's, this doesn't reproduce a published pool."""
    scores = r0.score_pool(units)
    return {rung: scores[(rung, "held_out")] for rung in F64_RUNGS}


def _expected_rows(select_json: Mapping[str, Any]) -> set[tuple[str, str, str, int]]:
    return {
        (world, unit, variant, draw)
        for world, info in select_json["units"].items()
        for unit in info["ids"]
        for variant in select_json["variants"][world]
        for draw in range(1, select_json["k"] + 1)
    }


def check_select(raw: haiku.Raw) -> dict[str, Any]:
    """The `select.json` structure and the job rows it implies; raises `ScoreRefusal`."""
    sel = raw.select
    if sel.get("mode") not in MODES:
        raise haiku.ScoreRefusal(f"select.json mode {sel.get('mode')!r} is not one of {MODES}")
    scored, control = sel["scored_worlds"], sel["control"]
    if sel["k"] != K or sel["pair"] != PAIR or sel["n_boot"] != N_BOOT:
        raise haiku.ScoreRefusal("select.json k, pair or n_boot differs from haiku_m9's")
    if set(sel["worlds"]) != {*scored, *([control] if control else [])}:
        raise haiku.ScoreRefusal("select.json worlds differ from its scored worlds and control")
    for w in scored:
        if list(sel["variants"][w]) != list(VARIANTS):
            raise haiku.ScoreRefusal(f"{w}: scored worlds run both variants")
    if control is not None and list(sel["variants"][control]) != list(CONTROL_VARIANTS):
        raise haiku.ScoreRefusal(f"{control}: the control runs the few-shot variant alone")
    got = {(r["world"], r["unit"], r["variant"], r["draw"]) for r in raw.rows}
    if got != _expected_rows(sel) or len(raw.rows) != sel["n_calls"]:
        raise haiku.ScoreRefusal("the job's rows differ from select.json's units and variants")
    return sel


def score(job: Path, raw_dir: Path, runs: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(report, scores.jsonl rows) of a finished run. Raises `haiku.ScoreRefusal` on any failed
    check."""
    selected = json.loads((job / "select.json").read_text())
    if selected.get("prereg_sha256") != prereg_sha256():
        raise haiku.ScoreRefusal("prereg/M9.md differs from the version select recorded")
    raw = haiku.load_raw(job, raw_dir)
    sel = check_select(raw)
    mode, scored, control = sel["mode"], sel["scored_worlds"], sel["control"]
    names = [*scored, *([control] if control else [])]
    loaded = load_worlds(runs, names)
    for name, (s, _) in loaded.items():
        want = S_CONTROL if name == control else S_SCORED
        if s != want:
            raise haiku.ScoreRefusal(f"{name}: s = {s}, expected {want}")
    by_world: dict[str, list[R0Unit]] = {}
    pool_units = sorted((u for w in scored for u in loaded[w][1]), key=lambda u: u.group)
    f64_all = f64_held_out(pool_units)
    f64_control: dict[str, NDArray[np.float64]] = {}
    control_pool: list[R0Unit] = []
    if control is not None:
        control_pool = sorted(loaded[control][1], key=lambda u: u.group)
        f64_control = f64_held_out(control_pool)
    keep: dict[str, list[int]] = {}
    for w in names:
        wanted = set(sel["units"][w]["ids"])
        pool = control_pool if w == control else pool_units
        keep[w] = [i for i, u in enumerate(pool) if u.world == w and u.unit in wanted]
        found = {pool[i].unit for i in keep[w]}
        if found != wanted:
            raise haiku.ScoreRefusal(
                f"{w}: {len(wanted - found)} job units are not in the world, "
                f"e.g. {sorted(wanted - found)[:3]}"
            )
        by_world[w] = [pool[i] for i in keep[w]]
        if any(u.pair != PAIR for u in by_world[w]):
            raise haiku.ScoreRefusal(f"{w}: a job unit is not in pair {PAIR}")
        counts = {r0.ABUSIVE: 0, r0.LEGITIMATE: 0}
        for u in by_world[w]:
            counts[class_of(u)] += 1
        if counts != sel["units"][w]["counts"]:
            raise haiku.ScoreRefusal(f"{w}: class counts differ from select.json's")
    scored_units = [u for w in scored for u in by_world[w]]
    f64 = {
        rung: np.array([f64_all[rung][i] for w in scored for i in keep[w]]) for rung in F64_RUNGS
    }
    control_units = by_world[control] if control else []
    control_f64 = (
        {rung: np.array([f64_control[rung][i] for i in keep[control]]) for rung in F64_RUNGS}
        if control
        else {}
    )

    key_of = {(r["world"], r["unit"], r["variant"], r["draw"]): r["key"] for r in raw.rows}
    all_units = scored_units + control_units
    draws: dict[str, list[list[int | None]]] = {}
    statuses: dict[str, list[list[list[str]]]] = {}
    means: dict[str, NDArray[np.float64]] = {}
    firsts: dict[str, NDArray[np.float64]] = {}
    for variant in VARIANTS:
        units_v = [u for u in all_units if variant in sel["variants"][u.world]]
        keys = [[key_of[(u.world, u.unit, variant, d)] for d in range(1, K + 1)] for u in units_v]
        draws[variant] = [[haiku.draw_outcome(raw.attempts[k]) for k in row] for row in keys]
        statuses[variant] = [[[a.status for a in raw.attempts[k]] for k in row] for row in keys]
        means[variant], firsts[variant] = haiku.unit_scores(draws[variant])
    n_scored = len(scored_units)
    positive = np.array([u.positive for u in scored_units], dtype=bool)
    control_arg = None
    if control is not None:
        control_arg = {
            "positive": np.array([u.positive for u in control_units], dtype=bool),
            "score": means["few_shot"][n_scored:],
            "draw1": firsts["few_shot"][n_scored:],
            "f64": control_f64,
        }
    report = evaluate(
        mode,
        [u.world for u in scored_units],
        positive,
        {v: means[v][:n_scored] for v in VARIANTS},
        {v: firsts[v][:n_scored] for v in VARIANTS},
        f64,
        control_arg,
    )
    report["reliability"] = {
        "scope": "every unit with the variant, control included",
        **{
            v: haiku.reliability(
                np.array([[np.nan if d is None else d for d in row] for row in draws[v]])
            )
            for v in VARIANTS
        },
    }
    report["parse_rate"] = {v: haiku.parse_rate(statuses[v]) for v in VARIANTS}
    report["calls"] = haiku.calls_block(raw)
    report["binding"] = {
        "mode": mode,
        "prompt_version": sel["prompt_version"],
        "prompt_sha256": sel["prompt_sha256"],
        "jobs_sha256": sel["jobs_sha256"],
        "prereg_sha256": sel["prereg_sha256"],
        "k": K,
        "worlds": sel["worlds"],
        "scored_worlds": scored,
        "control": control,
        "unparsed_score": haiku.UNPARSED_SCORE,
    }
    rows = []
    for i, u in enumerate(all_units):
        row: dict[str, Any] = {
            "world": u.world,
            "unit": u.unit,
            "group": u.group,
            "family": u.family,
            "pair": u.pair,
            "positive": u.positive,
        }
        for variant in VARIANTS:
            if variant in sel["variants"][u.world]:
                row[variant] = {
                    "draws": ["unparsed" if d is None else d for d in draws[variant][i]],
                    "score": float(means[variant][i]),
                    "draw1_score": float(firsts[variant][i]),
                }
        rungs = control_f64 if u.world == control else f64
        offset = n_scored if u.world == control else 0
        row.update({rung: float(rungs[rung][i - offset]) for rung in F64_RUNGS})
        rows.append(row)
    return report, rows


# --- CLI ------------------------------------------------------------------------------------------


def _fmt(e: Mapping[str, float]) -> str:
    return f"{e['point']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}]"


def summary(report: Mapping[str, Any]) -> list[str]:
    lines = [f"M9 {report['verdict']}"]
    parts = [f"few_shot AUC {_fmt(report['primary']['few_shot'])}"]
    for rung, e in report["haiku_minus_f64"]["few_shot"].items():
        parts.append(f"minus {rung} lower {e['lo']:.4f}")
    if report["control"] is not None:
        parts.append(f"control AUC {_fmt(report['control']['few_shot'])}")
    lines.append("; ".join(parts))
    lines.append("criteria " + " ".join(f"{k}={v}" for k, v in report["criteria"].items()))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select", help="choose units and examples; write the job file")
    pick.add_argument("--mode", choices=MODES, required=True)
    pick.add_argument("--runs", type=Path, required=True)
    pick.add_argument("--out", type=Path, required=True)
    pick.add_argument("--worlds", nargs="+", default=[], help="the two s = 0.75 worlds (confirm)")
    pick.add_argument("--control", help="the s = 0 control world (confirm)")
    grade = sub.add_parser("score", help="score a finished driver run, once")
    grade.add_argument("--job", type=Path, required=True)
    grade.add_argument("--raw", type=Path, required=True)
    grade.add_argument("--runs", type=Path, required=True)
    grade.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        try:
            meta = select(args.mode, args.runs, args.out, args.worlds, args.control)
        except SelectRefusal as e:
            print(f"refused: {e}", file=sys.stderr)
            return hr.EXIT_REFUSED
        print(f"jobs.jsonl sha256 {meta['jobs_sha256']}")
        print(f"prereg/M9.md sha256 {meta['prereg_sha256']}")
        print(f"mode {meta['mode']}: {meta['n_calls']} calls")
        for world, info in meta["units"].items():
            print(f"{world} (s = {meta['worlds'][world]}): {info['counts']}")
        return 0
    if args.out.exists():
        print(f"refused: {args.out} exists; M9 is scored once", file=sys.stderr)
        return hr.EXIT_REFUSED
    try:
        report, rows = score(args.job, args.raw, args.runs)
    except haiku.ScoreRefusal as e:
        print(f"refused: {e}", file=sys.stderr)
        return hr.EXIT_REFUSED
    args.out.mkdir(parents=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    (args.out / "report.json").write_text(text + "\n")
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    print("\n".join(summary(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
