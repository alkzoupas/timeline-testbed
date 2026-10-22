"""R2, the linear probe (SPEC §7 R2): an L2 logistic model on pooled hidden states of the render.

Embeddings come from `testbed.models.r2_embed` (`<embed_dir>/<model>/<world>.npz` + `.json`). A
probe is one feature (`CANDIDATES`) at one L2 strength `C` (`C_GRID`), fitted by a fresh
`StandardScaler` + `LogisticRegression` per fold on float64 rows. Its score is the decision
function (log-odds), which keeps saturated probabilities from tying.

Two stages, as SPEC §7 describes:

- `tune`: on the dedicated tuning world, every candidate x C is scored in the held-out regime and
  in the in-family regime; the selection is the highest held-out primary metric (ties: the smaller
  C, then `CANDIDATES` order). No gate scores that world.
- `score_rungs`: R0's rungs plus `r2_<model>` at the frozen (feature, C) of each model, on the
  pooled worlds of one `s`, in both regimes (`r0.evaluate` reports them, with the differences
  against F64 logistic and trees).

The labels come from `r0.load_world` (`testbed.eval.labels`); embeddings carry none. Every embedding
is checked before use against the frozen prompt, the world's checksum and the model revision, the
sha256 of the world's rows text (prompt and renders as they are now), of the current MLX child
script and of the npz file.

CLI::

    python -m testbed.eval.r2 tune --world W --embed DIR --model 9b --out DIR
    python -m testbed.eval.r2 score --worlds W [W ...] --embed DIR \\
        --choice 9b:mean_final:1e-3 [--choice 4b:mean_16:1e-2] --out DIR

Both refuse a non-empty `--out`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from testbed.eval import cv, r0
from testbed.eval.metrics import auc
from testbed.models import local_llm, prompt
from testbed.world.families import PAIRS

CANDIDATES = ("mean_final", "mean_16", "last_final")  # SPEC §7's three; `last_16` is not one
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MODELS = ("9b", "4b")
MAX_ITER = 10_000


def rung_name(model: str) -> str:
    return f"r2_{model}"


# --- Loading --------------------------------------------------------------------------------------


def world_rows_sha256(world_dir: Path, limit: int | None = None) -> str:
    """The sha256 of the rows text the MLX child reads for `world_dir` (the first `limit` renders
    if given), as `r2_embed` and `r1_run` write it."""
    renders = [json.loads(x) for x in (world_dir / "renders.jsonl").read_text().splitlines() if x]
    rows = prompt.child_rows(renders[:limit] if limit is not None else renders)
    return local_llm.sha256_text(local_llm.rows_text(rows))


def _require(meta: Mapping[str, Any], field: str, world_dir: Path, what: str) -> Any:
    if not meta.get(field):
        raise ValueError(f"{world_dir.name}: {what} has no {field}")
    return meta[field]


def check_provenance(
    meta: Mapping[str, Any], world_dir: Path, model: str, *, limit: int | None = None, what: str
) -> None:
    """Raise unless the run json `meta` matches the frozen prompt, the world, the model, the
    world's rows text (the first `limit` renders for a limited run) and the current MLX child
    script. A missing field is a refusal."""
    if meta.get("prompt_sha256") != prompt.PROMPT_SHA256:
        raise ValueError(f"{world_dir.name}: {what} was made with a different prompt")
    checksum = json.loads((world_dir / "manifest.json").read_text())["checksum_sha256"]
    if meta.get("world_checksum_sha256") != checksum:
        raise ValueError(f"{world_dir.name}: {what} is of a different world (checksum differs)")
    if meta.get("model_revision") != local_llm.model_revision(model):
        raise ValueError(f"{world_dir.name}: {what} is from a different {model} revision")
    if _require(meta, "rows_sha256", world_dir, what) != world_rows_sha256(world_dir, limit):
        raise ValueError(f"{world_dir.name}: {what} rows_sha256 differs from the world's rows now")
    if _require(meta, "child_sha256", world_dir, what) != local_llm.child_sha256():
        raise ValueError(f"{world_dir.name}: {what} was made by a different mlx_child.py")


def check_npz_sha256(meta: Mapping[str, Any], npz: Path, world_dir: Path) -> None:
    """Raise unless the embedding json's `npz_sha256` is the hash of `npz` as it is now."""
    want = _require(meta, "npz_sha256", world_dir, "embedding")
    if hashlib.sha256(npz.read_bytes()).hexdigest() != want:
        raise ValueError(f"{world_dir.name}: embedding npz_sha256 differs from {npz.name}")


PROVENANCE_FIELDS = (
    "prompt_sha256",
    "model_revision",
    "world_checksum_sha256",
    "layers",
    "rows_sha256",
    "child_sha256",
    "npz_sha256",
)


def embedding_provenance(embed_dir: Path, model: str, world_dir: Path) -> dict[str, Any]:
    """The embedding json's fields that identify what it was computed from."""
    world_dir = world_dir.resolve()
    meta = json.loads((embed_dir / model / f"{world_dir.name}.json").read_text())
    for field in PROVENANCE_FIELDS:
        _require(meta, field, world_dir, "embedding")
    return {k: meta[k] for k in PROVENANCE_FIELDS}


def aligned_rows(
    ids: Sequence[str], values: NDArray[Any], wanted: Sequence[str], what: str
) -> NDArray[Any]:
    """`values` rows reordered to `wanted` by id. Raises on duplicate ids, on a wanted id that is
    absent, and on a row count that disagrees with `ids`. Extra ids are ignored: clusters that mix
    groups are embedded but are not units."""
    if len(ids) != values.shape[0]:
        raise ValueError(f"{what}: {len(ids)} ids for {values.shape[0]} rows")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what}: duplicate ids")
    where = {unit: i for i, unit in enumerate(ids)}
    missing = [u for u in wanted if u not in where]
    if missing:
        raise ValueError(f"{what}: {len(missing)} units have no row, e.g. {missing[:3]}")
    return values[[where[u] for u in wanted]]


def load_features(
    embed_dir: Path, model: str, world_dir: Path, units: Sequence[r0.R0Unit], feature: str
) -> NDArray[np.float64]:
    """The `feature` rows of `units` (all of `world_dir`), aligned by `R0Unit.unit`, as float64.

    Raises if the embedding json's prompt hash, world checksum or model revision disagree with
    the frozen prompt, the world's manifest or the local snapshot; if its rows hash differs from
    the world's rows text now, its child hash from the current `mlx_child.py` or its npz hash from
    the npz file; if any of these fields is missing; if a unit has no row, if the ids repeat, or
    if a value is not finite."""
    world_dir = world_dir.resolve()
    foreign = [u.group for u in units if u.world != world_dir.name or not u.unit]
    if foreign:
        raise ValueError(
            f"{world_dir.name}: units of another world or without a unit key: {foreign[:3]}"
        )
    stem = embed_dir / model / world_dir.name  # names hold dots (s0.75), so no `with_suffix`
    meta = json.loads(Path(f"{stem}.json").read_text())
    check_provenance(meta, world_dir, model, what="embedding")
    check_npz_sha256(meta, Path(f"{stem}.npz"), world_dir)
    with np.load(Path(f"{stem}.npz"), allow_pickle=False) as data:
        if feature not in data.files:
            raise ValueError(f"{stem}.npz has no array {feature!r}; it holds {data.files}")
        ids = [str(x) for x in data["ids"]]
        rows = np.asarray(data[feature], dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"{stem}.npz: {feature} has shape {rows.shape}")
    out = aligned_rows(ids, rows, [u.unit for u in units], f"{stem}.npz")
    if not np.isfinite(out).all():
        raise ValueError(f"{stem}.npz: {feature} has non-finite values")
    return np.asarray(out, dtype=np.float64)


def pooled_features(
    embed_dir: Path,
    model: str,
    worlds: Sequence[Path],
    units: Sequence[r0.R0Unit],
    feature: str,
) -> NDArray[np.float64]:
    """`load_features` for units spanning several worlds: rows in `units` order, each world's
    units read from that world's embedding (keyed by `R0Unit.world`)."""
    by_name = {w.resolve().name: w for w in worlds}
    if len(by_name) != len(worlds):
        raise ValueError("world directory names must be unique")
    unknown = {u.world for u in units} - set(by_name)
    if unknown:
        raise ValueError(f"units from worlds not given: {sorted(unknown)}")
    out: NDArray[np.float64] | None = None
    for name, world in by_name.items():
        index = [i for i, u in enumerate(units) if u.world == name]
        if not index:
            continue
        rows = load_features(embed_dir, model, world, [units[i] for i in index], feature)
        if out is None:
            out = np.zeros((len(units), rows.shape[1]))
        if rows.shape[1] != out.shape[1]:
            raise ValueError(f"{name}: width {rows.shape[1]} differs from the other worlds")
        out[index] = rows
    if out is None:
        raise ValueError("no units")
    return out


# --- Fitting --------------------------------------------------------------------------------------


def fit_scores(
    x_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.floating[Any]],
    C: float,
    max_iter_hits: list[bool] | None = None,
) -> NDArray[np.float64]:
    """Log-odds of class 1 on `x_test` from a fresh scaler + L2 logistic model (lbfgs, strength
    `C`) fitted on float64 training rows. If `max_iter_hits` is given, whether lbfgs stopped at
    `MAX_ITER` is appended to it; a failure to converge is recorded, not raised."""
    labels = np.asarray(y_train)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y_train must hold both classes, coded 0/1")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(np.asarray(x_train, dtype=np.float64), labels)
    if max_iter_hits is not None:
        max_iter_hits.append(bool(np.max(pipe[-1].n_iter_) >= MAX_ITER))
    return np.asarray(
        pipe.decision_function(np.asarray(x_test, dtype=np.float64)), dtype=np.float64
    )


def held_out_scores(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    pairs: NDArray[np.int64],
    C: float,
    max_iter_hits: list[bool] | None = None,
) -> NDArray[np.float64]:
    """Each unit's score from the fold that tests it, fitted on the other side's pairs only
    (`r0._held_out` with `fit_scores`)."""
    out = np.full(y.size, np.nan)
    for side in r0.TEST_PAIRS:
        test = np.isin(pairs, r0.TEST_PAIRS[side])
        train = np.isin(pairs, r0.TEST_PAIRS[r0._other(side)])
        out[test] = fit_scores(x[train], y[train], x[test], C, max_iter_hits)
    if np.isnan(out).any():
        raise ValueError("some units are in neither test side")
    return out


def strata_of(units: Sequence[r0.R0Unit]) -> list[str]:
    """The in-family fold strata of `r0.score_pool`: label x family."""
    return [f"{r0.ABUSIVE if u.positive else r0.LEGITIMATE}|{u.family}" for u in units]


def in_family_scores(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    groups: Sequence[str],
    strata: Sequence[str],
    C: float,
    max_iter_hits: list[bool] | None = None,
) -> NDArray[np.float64]:
    """Out-of-fold scores on R0's in-family folds (`cv.group_folds`, `r0.FOLDS`, `r0.CV_SEED`)."""
    out = np.full(y.size, np.nan)
    for train, test in cv.group_folds(strata, groups, folds=r0.FOLDS, seed=r0.CV_SEED):
        out[test] = fit_scores(x[train], y[train], x[test], C, max_iter_hits)
    if np.isnan(out).any():
        raise RuntimeError("some rows were never in a test fold")
    return out


def primary_auc(units: Sequence[r0.R0Unit], score: NDArray[np.float64]) -> float:
    """The primary metric (SPEC §6.4) of one score vector over `units`."""
    return r0.primary_boot(score, r0.primary_comparisons(units))[0]


def _labels(units: Sequence[r0.R0Unit]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    y = np.array([int(u.positive) for u in units], dtype=np.int64)
    pairs = np.array([u.pair for u in units], dtype=np.int64)
    return y, pairs


# --- Tuning ---------------------------------------------------------------------------------------


def select(grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The grid row with the highest held-out primary metric; ties go to the smaller C, then to
    the earlier candidate in `CANDIDATES`. (Values are compared to 12 decimals.)"""
    if not grid:
        raise ValueError("empty grid")
    best = min(
        grid,
        key=lambda row: (
            -round(float(row["held_out_primary"]), 12),
            float(row["C"]),
            CANDIDATES.index(row["feature"]),
        ),
    )
    return {"feature": best["feature"], "C": best["C"]}


def tune(
    world_dir: Path, embed_dir: Path, model: str, *, scales: Sequence[str] = ("standard",)
) -> dict[str, Any]:
    """Every candidate x C on one world: the held-out and in-family primary metrics, the held-out
    AUC of each pair, and the selection (`select`)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {list(MODELS)}")
    world_dir = world_dir.resolve()
    s, units, _ = r0.load_world(world_dir, scales=scales)
    y, pairs = _labels(units)
    groups = [u.group for u in units]
    strata = strata_of(units)
    comps = r0.primary_comparisons(units)
    positive = np.array([u.positive for u in units])
    grid: list[dict[str, Any]] = []
    embedding = embedding_provenance(embed_dir, model, world_dir)
    for feature in CANDIDATES:
        x = load_features(embed_dir, model, world_dir, units, feature)
        for C in C_GRID:
            hits: list[bool] = []
            held = held_out_scores(x, y, pairs, C, hits)
            infam = in_family_scores(x, y, groups, strata, C, hits)
            grid.append(
                {
                    "feature": feature,
                    "C": C,
                    "held_out_primary": r0.primary_boot(held, comps)[0],
                    "in_family_primary": r0.primary_boot(infam, comps)[0],
                    "held_out_family_auc": {
                        str(p): auc(held[(pairs == p) & positive], held[(pairs == p) & ~positive])
                        for p in PAIRS
                    },
                    "max_iter_hits": int(sum(hits)),
                    "fits": len(hits),
                }
            )
    return {
        "world": world_dir.name,
        "s": s,
        "model": model,
        "n_units": len(units),
        "embedding": embedding,
        "candidates": list(CANDIDATES),
        "c_grid": list(C_GRID),
        "max_iter": MAX_ITER,
        "grid": grid,
        "selected": select(grid),
    }


def tune_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# R2 tuning: {result['model']} on {result['world']} (s = {result['s']})",
        "",
        f"{result['n_units']} units. Primary metric (mean over sides A and B of the T-family AUC)",
        "by feature and C; the selection is the highest held-out primary (ties: smaller C, then",
        "candidate order).",
        "",
        "| feature | C | held-out | in-family | "
        + " | ".join(f"pair {p}" for p in PAIRS)
        + " | max_iter hits |",
        "|---|---|---|---|" + "---|" * len(PAIRS) + "---|",
    ]
    for row in result["grid"]:
        pairs = " | ".join(f"{row['held_out_family_auc'][str(p)]:.3f}" for p in PAIRS)
        lines.append(
            f"| {row['feature']} | {row['C']:g} | {row['held_out_primary']:.3f} | "
            f"{row['in_family_primary']:.3f} | {pairs} | {row['max_iter_hits']}/{row['fits']} |"
        )
    chosen = result["selected"]
    lines += ["", f"Selected: `{chosen['feature']}` at C = {chosen['C']:g}.", ""]
    return "\n".join(lines)


# --- Scoring --------------------------------------------------------------------------------------


def load_pool(
    worlds: Sequence[Path], scales: Sequence[str]
) -> tuple[float, list[r0.R0Unit], dict[str, int]]:
    """(the shared s, the pooled units sorted by group as `r0.run` does, mixed counts)."""
    names = [w.resolve().name for w in worlds]
    if len(set(names)) != len(names):
        raise ValueError(f"world directory names must be unique: {names}")
    units: list[r0.R0Unit] = []
    values: set[float] = set()
    mixed: dict[str, int] = {}
    for world in worlds:
        s, part, n_mixed = r0.load_world(world, scales=scales)
        values.add(s)
        units.extend(part)
        mixed[world.resolve().name] = n_mixed
    if len(values) != 1:
        raise ValueError(f"worlds must share one s, got {sorted(values)}")
    return values.pop(), sorted(units, key=lambda u: u.group), mixed


def _score_units(
    units: Sequence[r0.R0Unit],
    worlds: Sequence[Path],
    embed_dir: Path,
    choices: Mapping[str, tuple[str, float]],
) -> tuple[dict[tuple[str, str], NDArray[np.float64]], dict[str, int]]:
    """R0's scores plus `r2_<model>` in both regimes; and each probe's max-iter hit count."""
    scores = r0.score_pool(units)
    y, pairs = _labels(units)
    groups = [u.group for u in units]
    strata = strata_of(units)
    hits_by_rung: dict[str, int] = {}
    for model, (feature, C) in choices.items():
        if model not in MODELS or feature not in CANDIDATES:
            raise ValueError(f"unknown choice {model}:{feature}")
        x = pooled_features(embed_dir, model, worlds, units, feature)
        hits: list[bool] = []
        scores[(rung_name(model), "in_family")] = in_family_scores(x, y, groups, strata, C, hits)
        scores[(rung_name(model), "held_out")] = held_out_scores(x, y, pairs, C, hits)
        hits_by_rung[rung_name(model)] = int(sum(hits))
    return scores, hits_by_rung


def score_rungs(
    worlds: Sequence[Path],
    embed_dir: Path,
    choices: Mapping[str, tuple[str, float]],
    *,
    scales: Sequence[str] = ("standard",),
) -> tuple[list[r0.R0Unit], dict[tuple[str, str], NDArray[np.float64]]]:
    """(pooled units, scores): R0's rungs and, for each `{model: (feature, C)}` of `choices`, the
    rung `r2_<model>` in both regimes. The worlds must share one s."""
    _, units, _ = load_pool(worlds, scales)
    scores, _ = _score_units(units, worlds, embed_dir, choices)
    return units, scores


def parse_choice(text: str) -> tuple[str, str, float]:
    """`9b:mean_final:1e-3` -> ("9b", "mean_final", 0.001); C must be on `C_GRID`."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"--choice wants MODEL:FEATURE:C, got {text!r}")
    model, feature, c_text = parts
    if model not in MODELS:
        raise ValueError(f"--choice {text!r}: model must be one of {list(MODELS)}")
    if feature not in CANDIDATES:
        raise ValueError(f"--choice {text!r}: feature must be one of {list(CANDIDATES)}")
    C = float(c_text)
    if C not in C_GRID:
        raise ValueError(f"--choice {text!r}: C must be one of {list(C_GRID)}")
    return model, feature, C


def run_score(
    worlds: Sequence[Path],
    embed_dir: Path,
    choices: Mapping[str, tuple[str, float]],
    out: Path,
    *,
    scales: Sequence[str] = ("standard",),
) -> dict[str, Any]:
    """Score the pooled worlds, report them with `r0.evaluate`, and write `r2.json` and
    `scores.jsonl` into a new or empty `out`."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    if not choices:
        raise ValueError("no --choice given")
    s, units, mixed = load_pool(worlds, scales)
    scores, hits = _score_units(units, worlds, embed_dir, choices)
    r2_rungs = tuple(rung_name(m) for m in choices)
    rungs = r0.RUNGS + r2_rungs
    differences = [(r, base) for r in r2_rungs for base in ("f64_logistic", "f64_trees")]
    embedding: dict[str, Any] = {}
    for model in choices:
        for world in worlds:
            embedding[f"{model}/{world.resolve().name}"] = embedding_provenance(
                embed_dir, model, world
            )
    report = {
        "s": s,
        "worlds": [w.resolve().name for w in worlds],
        "mixed": mixed,
        "choices": {m: {"feature": f, "C": c} for m, (f, c) in choices.items()},
        "max_iter_hits": hits,
        "embedding": embedding,
        **r0.evaluate(units, scores, rungs=rungs, extra_differences=differences),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "r2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    r0._write_scores(out / "scores.jsonl", {s: (units, scores)}, rungs)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--world", type=Path, required=True)
    t.add_argument("--embed", type=Path, required=True)
    t.add_argument("--model", choices=MODELS, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--scales", nargs="+", default=["standard"])
    sc = sub.add_parser("score")
    sc.add_argument("--worlds", type=Path, nargs="+", required=True)
    sc.add_argument("--embed", type=Path, required=True)
    sc.add_argument("--choice", action="append", required=True)
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--scales", nargs="+", default=["standard"])
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty")
    started = time.perf_counter()
    if args.command == "tune":
        result = tune(args.world, args.embed, args.model, scales=args.scales)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tune.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (args.out / "tune.md").write_text(tune_markdown(result))
        chosen = result["selected"]
        seconds = time.perf_counter() - started
        print(f"selected {chosen['feature']} C={chosen['C']:g} in {seconds:.1f} s")
        return 0
    parsed: dict[str, tuple[str, float]] = {}
    for text in args.choice:
        model, feature, C = parse_choice(text)
        if model in parsed:
            raise ValueError(f"two --choice entries for model {model}")
        parsed[model] = (feature, C)
    report = run_score(args.worlds, args.embed, parsed, args.out, scales=args.scales)
    print(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")
    for regime in r0.REGIMES:
        for model in parsed:
            primary = report["regimes"][regime][rung_name(model)]["primary"]
            print(f"s = {report['s']} {regime} {rung_name(model)}: {primary['auc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
