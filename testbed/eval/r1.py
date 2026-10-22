"""R1, zero-shot (SPEC §7 R1): the local model's `RISK_LEVEL` read as a score, reported with R0.

`testbed.models.r1_run` writes `<r1_dir>/<model>/<world>.jsonl` (one row per unit, with
`expected` = sum n * p_n over the ten normalized answers) and `.json` (provenance). Nothing is
fitted, so the rung `r1_<model>` holds the same scores in both regimes, as the detector does.
It is reported with differences against F64 logistic and trees on the same resamples. R1 is
reported, not gating (SPEC §7).

CLI::

    python -m testbed.eval.r1 --worlds W [W ...] --r1 DIR --model 9b --out DIR

Worlds must share one `s`. Writes `r1.json` and `scores.jsonl` into a new or empty `--out`.

`--model nimble` reads Bespoke-Nimble-9B's typed score (`testbed.models.nimble_run`) as rung
`r1_nimble`: its provenance is the frozen schema hash and the pinned adapter, not the prompt hash.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from testbed.eval import r0, r2
from testbed.models import nimble_child, nimble_run, prompt

NIMBLE = "nimble"
MODELS = (*r2.MODELS, NIMBLE)
# the run json's fields the report carries per world
RUN_FIELDS = {
    "local": ("prompt_sha256", "model_revision", "head_dtype", "agreement"),
    NIMBLE: (
        "nimble_schema_sha256",
        "adapter_revision",
        "adapter_sha256",
        "nimble_commit",
        "n_distinct_expected",
        "n_prompt",
    ),
}


def rung_name(model: str) -> str:
    return f"r1_{model}"


def world_nimble_rows_sha256(world_dir: Path, limit: int | None = None) -> str:
    """The sha256 of the rows text the Nimble child reads for `world_dir` (the first `limit`
    renders if given), as `nimble_run` writes it."""
    renders = [json.loads(x) for x in (world_dir / "renders.jsonl").read_text().splitlines() if x]
    rows = nimble_run.child_rows(renders[:limit] if limit is not None else renders)
    return nimble_run.sha256_text(nimble_run.rows_text(rows))


def check_nimble_provenance(meta: Mapping[str, Any], world_dir: Path) -> None:
    """Raise unless a Nimble run's json matches the frozen schema, the world, the world's rows
    text now (the first `limit` renders for a limited run) and the pinned Nimble commit, adapter
    release and weights, and the fixed scoring settings. A missing `rows_sha256` is a refusal."""
    if meta.get("nimble_schema_sha256") != prompt.NIMBLE_SCHEMA_SHA256:
        raise ValueError(f"{world_dir.name}: Nimble run used a different schema")
    checksum = json.loads((world_dir / "manifest.json").read_text())["checksum_sha256"]
    if meta.get("world_checksum_sha256") != checksum:
        raise ValueError(f"{world_dir.name}: Nimble run is of a different world (checksum differs)")
    limit = meta.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError(f"{world_dir.name}: Nimble run has limit {limit!r}")
    if not meta.get("rows_sha256"):
        raise ValueError(f"{world_dir.name}: Nimble run has no rows_sha256")
    if meta["rows_sha256"] != world_nimble_rows_sha256(world_dir, limit):
        raise ValueError(
            f"{world_dir.name}: Nimble run rows_sha256 differs from the world's rows now"
        )
    if meta.get("adapter_revision") != nimble_child.ADAPTER_REVISION:
        raise ValueError(f"{world_dir.name}: Nimble run is from a different adapter revision")
    if not meta.get("child_sha256"):
        raise ValueError(f"{world_dir.name}: Nimble run has no child_sha256")
    if meta["child_sha256"] != nimble_run.child_sha256():
        raise ValueError(f"{world_dir.name}: Nimble run was made by a different nimble_child.py")
    pins = {
        "adapter_sha256": nimble_child.ADAPTER_SHA256,
        "nimble_commit": nimble_child.NIMBLE_COMMIT,
        "temperature": nimble_child.TEMPERATURE,
        "mode": nimble_child.MODE,
    }
    for key, value in pins.items():
        if meta.get(key) != value:
            raise ValueError(
                f"{world_dir.name}: Nimble run has {key} {meta.get(key)!r}, not {value!r}"
            )


def load_r1(
    r1_dir: Path, model: str, world_dir: Path, units: Sequence[r0.R0Unit]
) -> NDArray[np.float64]:
    """The R1 score (`expected`) of each of `units` (all of `world_dir`), aligned by unit id.

    Raises if the run's prompt hash, world checksum or model revision disagree with the frozen
    prompt, the world's manifest or the local snapshot, if its rows hash differs from the world's
    rows text now (the first `limit` renders for a limited run) or its child hash from the current
    `mlx_child.py`, if the head was not float32, if a unit has no row, if the ids repeat, or if a
    score is not finite. For `nimble` the checks are the schema hash, world checksum, rows hash
    (the same prefix rule), pinned adapter and the current `nimble_child.py` hash instead
    (`check_nimble_provenance`)."""
    world_dir = world_dir.resolve()
    foreign = [u.group for u in units if u.world != world_dir.name or not u.unit]
    if foreign:
        raise ValueError(
            f"{world_dir.name}: units of another world or without a unit key: {foreign[:3]}"
        )
    stem = r1_dir / model / world_dir.name  # names hold dots (s0.75), so no `with_suffix`
    meta = json.loads(Path(f"{stem}.json").read_text())
    if model == NIMBLE:
        check_nimble_provenance(meta, world_dir)
    else:
        r2.check_provenance(meta, world_dir, model, limit=meta.get("limit"), what="R1 run")
        if meta.get("head_dtype") != "float32":
            raise ValueError(f"{world_dir.name}: R1 head dtype is {meta.get('head_dtype')!r}")
    rows = [json.loads(x) for x in Path(f"{stem}.jsonl").read_text().splitlines() if x]
    ids = [str(r["id"]) for r in rows]
    scores = np.array([r["expected"] for r in rows], dtype=np.float64)
    out = r2.aligned_rows(ids, scores, [u.unit for u in units], f"{stem}.jsonl")
    if not np.isfinite(out).all():
        raise ValueError(f"{stem}.jsonl: non-finite scores")
    return np.asarray(out, dtype=np.float64)


def pooled_r1(
    r1_dir: Path,
    model: str,
    worlds: Sequence[Path],
    units: Sequence[r0.R0Unit],
) -> NDArray[np.float64]:
    """`load_r1` for units spanning several worlds: scores in `units` order, each world's units
    read from that world's run (keyed by `R0Unit.world`)."""
    by_name = {w.resolve().name: w for w in worlds}
    if len(by_name) != len(worlds):
        raise ValueError("world directory names must be unique")
    unknown = {u.world for u in units} - set(by_name)
    if unknown:
        raise ValueError(f"units from worlds not given: {sorted(unknown)}")
    out = np.full(len(units), np.nan)
    for name, world in by_name.items():
        index = [i for i, u in enumerate(units) if u.world == name]
        if index:
            out[index] = load_r1(r1_dir, model, world, [units[i] for i in index])
    if np.isnan(out).any():
        raise ValueError("some units got no R1 score")
    return out


def score_r1(
    units: Sequence[r0.R0Unit], worlds: Sequence[Path], r1_dir: Path, model: str
) -> dict[tuple[str, str], NDArray[np.float64]]:
    """R0's scores plus `r1_<model>`, the same vector in both regimes."""
    scores = r0.score_pool(units)
    values = pooled_r1(r1_dir, model, worlds, units)
    for regime in r0.REGIMES:
        scores[(rung_name(model), regime)] = values.copy()
    return scores


def run(
    worlds: Sequence[Path],
    r1_dir: Path,
    model: str,
    out: Path,
    *,
    scales: Sequence[str] = ("standard",),
) -> dict[str, Any]:
    """Score the pooled worlds, report them with `r0.evaluate`, and write `r1.json` and
    `scores.jsonl` into a new or empty `out`."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {list(MODELS)}")
    s, units, mixed = r2.load_pool(worlds, scales)
    scores = score_r1(units, worlds, r1_dir, model)
    rung = rung_name(model)
    rungs = r0.RUNGS + (rung,)
    differences = [(rung, base) for base in ("f64_logistic", "f64_trees")]
    runs: dict[str, Any] = {}
    for world in worlds:
        meta = json.loads((r1_dir / model / f"{world.resolve().name}.json").read_text())
        fields = RUN_FIELDS[NIMBLE if model == NIMBLE else "local"]
        runs[world.resolve().name] = {
            k: meta[k] for k in (*fields, "world_checksum_sha256", "limit")
        }
    values = scores[(rung, "held_out")]
    report = {
        "s": s,
        "worlds": [w.resolve().name for w in worlds],
        "mixed": mixed,
        "model": model,
        "runs": runs,
        "n_distinct_scores": int(np.unique(values).size),
        **r0.evaluate(units, scores, rungs=rungs, extra_differences=differences),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "r1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    r0._write_scores(out / "scores.jsonl", {s: (units, scores)}, rungs)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--worlds", type=Path, nargs="+", required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=["standard"])
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = run(args.worlds, args.r1, args.model, args.out, scales=args.scales)
    print(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")
    for regime in r0.REGIMES:
        primary = report["regimes"][regime][rung_name(args.model)]["primary"]
        print(f"s = {report['s']} {regime} {rung_name(args.model)}: {primary['auc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
