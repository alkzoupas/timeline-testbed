"""R1's second model, Bespoke-Nimble-9B (SPEC §7 R1): label-free typed-score forward passes.

    python -m testbed.models.nimble_run --world runs/m1-dev-s0.75-seed1 --out runs/m4-r1 \\
        [--limit N] [--count-only]

Reads only `<world>/renders.jsonl` and `<world>/manifest.json`. The context is the render text
alone, at full length; the question is the frozen schema `prompt.NIMBLE_SCHEMA`. The Nimble child
(`nimble_child.py`, in the Nimble MLX venv) counts the prompt tokens of every row first, with no
model loaded, and a row over `MAX_INPUT_TOKENS` fails the run before any forward pass. SPEC §6.2's
2,048-token cap is not enforced here: `n_prompt.n_over_2048` records how many rows pass it. Rows
then run in chunks of `CHUNK`, so a crash loses at most one chunk: a rerun reuses every chunk
already written and checked against the pinned Nimble commit, adapter and schema.

`--count-only` writes only `<out>/nimble/<world>.counts.json` (the token counts and their stats).
`--limit N` scores only the first N renders, for smoke runs, and records the limit.

Writes `<out>/nimble/<world>.jsonl` (rows `id`, `n_prompt`, `p`, `expected`, `argmax`, in render
order; `expected` = sum n * p_n over the softmax of the ten candidate logits) and `<world>.json`
(provenance and throughput). The chunk directory `<world>.parts/` is removed once those are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from testbed.models import nimble_child, prompt, r2_embed

VENV_PYTHON = Path.home() / "code" / "nimble" / ".venv-mlx" / "bin" / "python"
CHILD_SCRIPT = Path(__file__).with_name("nimble_child.py")

CHUNK = 100
TOKEN_CAP = r2_embed.TOKEN_LIMIT  # SPEC §6.2: reported (`n_over_2048`), not enforced
ROW_FIELDS = ("id", "n_prompt", "p", "expected", "argmax")
N_LEVELS = len(nimble_child.LEVELS)

Row = Mapping[str, str]  # {"id", "context"}


def child_rows(renders: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Nimble child rows (`{"id", "context"}`) from `renders.jsonl` records (`{"unit", "text"}`)."""
    return [{"id": r["unit"], "context": r["text"]} for r in renders]


def rows_text(rows: Sequence[Row]) -> str:
    return "".join(json.dumps(dict(r)) + "\n" for r in rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def child_sha256() -> str:
    """The sha256 of the Nimble child script as it is now."""
    return hashlib.sha256(CHILD_SCRIPT.read_bytes()).hexdigest()


def schema_text() -> str:
    """The frozen schema as the child reads it; its bytes hash to `NIMBLE_SCHEMA_SHA256`."""
    text = json.dumps(prompt.NIMBLE_SCHEMA, sort_keys=True)
    if sha256_text(text) != prompt.NIMBLE_SCHEMA_SHA256:
        raise RuntimeError("prompt.NIMBLE_SCHEMA differs from NIMBLE_SCHEMA_SHA256")
    return text


def token_stats(n_prompt: Sequence[int]) -> dict[str, Any]:
    """The prompt-length stats SPEC §6.2 needs: max, mean, p90 (linear interpolation), and how
    many prompts exceed the 2,048-token cap and the child's `MAX_INPUT_TOKENS`."""
    if not n_prompt:
        raise ValueError("no prompts")
    x = np.asarray(n_prompt, dtype=np.float64)
    return {
        "max": int(x.max()),
        "mean": float(x.mean()),
        "p90": float(np.percentile(x, 90)),
        "n_over_2048": int((x > TOKEN_CAP).sum()),
        "n_over_max_input": int((x > nimble_child.MAX_INPUT_TOKENS).sum()),
    }


def check_fits(count_rows: Sequence[Mapping[str, Any]]) -> None:
    """Raise if any prompt exceeds the child's `MAX_INPUT_TOKENS` (nothing is truncated)."""
    limit = nimble_child.MAX_INPUT_TOKENS
    over = [r["id"] for r in count_rows if r["n_prompt"] > limit]
    if over:
        raise ValueError(f"{len(over)} prompts exceed {limit} tokens, e.g. {over[:3]}")


def check_provenance(prov: Mapping[str, Any]) -> None:
    """Raise `RuntimeError` unless a child result's provenance names the pinned Nimble commit,
    adapter release and weights, the frozen schema, the fixed scoring settings and this child
    script as it is now."""
    expected = {
        "child_sha256": child_sha256(),
        "nimble_commit": nimble_child.NIMBLE_COMMIT,
        "adapter_repo": nimble_child.ADAPTER_REPO,
        "adapter_revision": nimble_child.ADAPTER_REVISION,
        "adapter_sha256": nimble_child.ADAPTER_SHA256,
        "nimble_schema_sha256": prompt.NIMBLE_SCHEMA_SHA256,
        "max_input_tokens": nimble_child.MAX_INPUT_TOKENS,
        "temperature": nimble_child.TEMPERATURE,
        "mode": nimble_child.MODE,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise RuntimeError(f"nimble child: {key} is {prov.get(key)!r}, expected {value!r}")
    if not prov.get("contract_sha256"):
        raise RuntimeError("nimble child: no contract_sha256")


def _check_common(result: Mapping[str, Any], command: str, rows: Sequence[Row]) -> None:
    if result.get("command") != command:
        raise RuntimeError(f"nimble child: result is of {result.get('command')!r}, not {command}")
    check_provenance(result["provenance"])
    if result.get("rows_sha256") != sha256_text(rows_text(rows)):
        raise RuntimeError(f"nimble {command}: the child read other rows than were written")
    if [r["id"] for r in result["rows"]] != [r["id"] for r in rows]:
        raise RuntimeError(f"nimble {command}: row ids differ from the input")
    for r in result["rows"]:
        if not isinstance(r["n_prompt"], int) or r["n_prompt"] < 1:
            raise RuntimeError(f"nimble {command}: {r['id']} has n_prompt {r['n_prompt']!r}")


def check_score(result: Mapping[str, Any], rows: Sequence[Row]) -> None:
    """Raise `RuntimeError` unless a `score` child result is complete for `rows`: pinned
    provenance, the same rows, the ids in order, ten finite probabilities that sum to 1, and
    `expected` and `argmax` that agree with them."""
    _check_common(result, "score", rows)
    for r in result["rows"]:
        p = r["p"]
        values = [*p, r["expected"], r["seconds"]]
        try:
            finite = bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())
        except TypeError, ValueError:
            finite = False
        if len(p) != N_LEVELS or not finite:
            raise RuntimeError(f"nimble score: {r['id']} does not have {N_LEVELS} finite values")
        if not math.isclose(sum(p), 1.0, abs_tol=nimble_child.SUM_TOLERANCE):
            raise RuntimeError(f"nimble score: {r['id']} probabilities sum to {sum(p)}")
        want = sum(n * x for n, x in enumerate(p, 1))
        if not math.isclose(r["expected"], want, abs_tol=1e-9):
            raise RuntimeError(f"nimble score: {r['id']} expected {r['expected']} != {want}")
        if r["argmax"] != 1 + int(np.argmax(p)):
            raise RuntimeError(f"nimble score: {r['id']} argmax {r['argmax']} disagrees with p")


def run_child(
    command: str,
    rows: Sequence[Row],
    *,
    work_dir: Path | None = None,
    tag: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one child command out of process (`python -I`, `HF_HUB_OFFLINE=1`) and return its
    parsed JSON. Files go to `work_dir` as `<tag>.rows.jsonl`, `<tag>.schema.json` and
    `<tag>.json`, and stay there; without a `work_dir` they go to a temporary directory."""
    if work_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return run_child(command, rows, work_dir=Path(tmp), tag=tag, timeout=timeout)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = tag or command
    rows_file, schema_file = work_dir / f"{stem}.rows.jsonl", work_dir / f"{stem}.schema.json"
    out_json = work_dir / f"{stem}.json"
    rows_file.write_text(rows_text(rows))
    schema_file.write_text(schema_text())
    cmd = [
        str(VENV_PYTHON),
        "-I",
        str(CHILD_SCRIPT),
        command,
        "--in",
        str(rows_file),
        "--schema",
        str(schema_file),
        "--schema-sha256",
        prompt.NIMBLE_SCHEMA_SHA256,
        "--out",
        str(out_json),
    ]
    done = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"HF_HUB_OFFLINE": "1"},
        timeout=timeout,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"nimble_child {command} failed (exit {done.returncode}): {done.stderr[-800:]}"
        )
    result: dict[str, Any] = json.loads(out_json.read_text())
    return result


def count(rows: Sequence[Row], **kw: Any) -> dict[str, Any]:
    """The child's token counts for `rows` (no model loaded), checked."""
    result = run_child("count", rows, **kw)
    _check_common(result, "count", rows)
    return result


def score(rows: Sequence[Row], **kw: Any) -> dict[str, Any]:
    """The child's Nimble scores for `rows`, checked (`check_score`)."""
    result = run_child("score", rows, **kw)
    check_score(result, rows)
    return result


def _load_part(work: Path, tag: str, rows: Sequence[Row]) -> dict[str, Any] | None:
    """A finished chunk's child result, or None if it is missing, or fails the checks: the pinned
    provenance with this child script, and the sha256 of exactly these rows' text."""
    path = work / f"{tag}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
        check_score(result, rows)
    except RuntimeError, ValueError, KeyError, TypeError, OSError:
        return None
    return dict(result)


def _provenance_record(prov: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "nimble_commit",
        "adapter_revision",
        "adapter_sha256",
        "contract_sha256",
        "child_sha256",
    )
    return {
        **{k: prov[k] for k in keys},
        "nimble_schema_sha256": prov["nimble_schema_sha256"],
        "max_input_tokens": prov["max_input_tokens"],
        "temperature": prov["temperature"],
        "mode": prov["mode"],
        "packages": prov["packages"],
    }


def _load_inputs(world: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads((world / "manifest.json").read_text())
    renders = [json.loads(x) for x in (world / "renders.jsonl").read_text().splitlines() if x]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        renders = renders[:limit]
    rows = child_rows(renders)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{world}: duplicate unit ids in renders.jsonl")
    schema_text()  # the frozen schema's hash, before any child runs
    return manifest, rows


def run(
    world: Path, out: Path, *, limit: int | None = None, count_only: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    world = world.resolve()
    manifest, rows = _load_inputs(world, limit)
    ids = [r["id"] for r in rows]
    dest = out / "nimble"
    final_rows, final_json = dest / f"{world.name}.jsonl", dest / f"{world.name}.json"
    counts_json = dest / f"{world.name}.counts.json"
    produced = (counts_json,) if count_only else (final_rows, final_json)
    if any(p.exists() for p in produced):
        raise FileExistsError(f"{produced[0]} or its pair already exists")
    dest.mkdir(parents=True, exist_ok=True)

    counted = count(rows)
    base = {
        "world": world.name,
        "world_checksum_sha256": manifest["checksum_sha256"],
        "generator_version": manifest["generator_version"],
        "model": "nimble",
        **_provenance_record(counted["provenance"]),
        "rows_sha256": counted["rows_sha256"],  # checked against the rows text in `count`
        "limit": limit,
        "n_rows": len(rows),
        "n_prompt": token_stats([r["n_prompt"] for r in counted["rows"]]),
    }
    if count_only:
        record = {
            **base,
            "count_only": True,
            "wall_seconds": round(time.perf_counter() - started, 1),
            "rows": counted["rows"],
        }
        counts_json.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return record
    check_fits(counted["rows"])

    work = dest / f"{world.name}.parts"
    work.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    reused = 0
    for i, (a, b) in enumerate(r2_embed.chunk_ranges(len(rows), CHUNK)):
        tag = f"chunk{i:03d}"
        result = _load_part(work, tag, rows[a:b])
        if result is None:
            result = score(rows[a:b], work_dir=work, tag=tag)
        else:
            reused += 1
        results.append(result)
    per_row = [r for res in results for r in res["rows"]]
    if [r["id"] for r in per_row] != ids:
        raise RuntimeError("merged ids differ from the render order")
    if [r["n_prompt"] for r in per_row] != [r["n_prompt"] for r in counted["rows"]]:
        raise RuntimeError("scored prompt lengths differ from the counted ones")
    provenance = {
        json.dumps(_provenance_record(res["provenance"]), sort_keys=True) for res in results
    }
    if len({res["provenance"]["child_sha256"] for res in results}) != 1:
        raise RuntimeError("chunks were scored by different child scripts")
    if len(provenance) != 1:
        raise RuntimeError("chunks were scored under different packages or settings")
    if results[0]["provenance"]["child_sha256"] != base["child_sha256"]:
        raise RuntimeError("chunks were scored by a different child script than the count")

    steady_tokens = sum(res["steady_tokens"] for res in results)
    steady_seconds = sum(res["steady_seconds"] for res in results)
    record = {
        **base,
        "child_sha256": results[0]["provenance"]["child_sha256"],  # every chunk agrees
        "count_only": False,
        "chunks": len(results),
        "chunks_reused": reused,
        "peak_memory_gib": max(res["peak_memory_gib"] for res in results),
        "steady_tokens_per_s": steady_tokens / steady_seconds if steady_seconds else None,
        "steady_seconds_per_unit": steady_seconds / max(len(per_row) - len(results), 1),
        "mean_seconds_per_unit": sum(r["seconds"] for r in per_row) / len(per_row),
        "n_distinct_expected": len({float(r["expected"]) for r in per_row}),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    final_rows.write_text(
        "".join(json.dumps({k: r[k] for k in ROW_FIELDS}) + "\n" for r in per_row)
    )
    final_json.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    shutil.rmtree(work)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args(argv)
    record = run(args.world, args.out, limit=args.limit, count_only=args.count_only)
    stats = record["n_prompt"]
    print(
        f"{record['world']} nimble: {record['n_rows']} rows, prompt tokens max {stats['max']}, "
        f"mean {stats['mean']:.0f}, p90 {stats['p90']:.0f}, over 2048: {stats['n_over_2048']}, "
        f"{record['wall_seconds']:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
