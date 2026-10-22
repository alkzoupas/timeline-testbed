"""R2 hidden-state extraction for one world (SPEC §7 R2): label-free forward passes.

    python -m testbed.models.r2_embed --world runs/m1-dev-s0.75-seed1 --model 9b --out runs/m4-embed

Reads only `<world>/renders.jsonl` and `<world>/manifest.json`. Each render is wrapped in the
frozen prompt (`prompt.py`); every prompt plus the longest R1 answer must fit SPEC §6.2's 2,048
tokens, checked with the model's own tokenizer before any forward pass. The MLX child's `embed`
then runs in chunks of `CHUNK` rows, so a crash loses at most one chunk: a rerun reuses every
chunk already written and checked. A chunk is reused only if its result names the current child
script, the exact rows text of the chunk, the current model snapshot and the current layers
(`check_chunk`), and its commit marker `chunkNNN.done.json` holds the sha256 of the chunk's npz
and json as they are now; otherwise it is recomputed. The marker is written last, after the
child's result passes its checks, and a chunk's files and marker are removed before its child runs,
so no npz can sit beside the json (or the marker) of another run.

Writes `<out>/<model>/<world>.npz` (float32 arrays `mean_<L>` and `last_<L>` for each layer in
`LAYERS`, plus `ids`, in render order) and `<world>.json` (provenance, throughput, and the sha256 of
the child script, of the rows text and of the npz). The chunk directory `<world>.parts/` is
removed once the merged file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from testbed.models import local_llm, prompt

LAYERS = "16,final"  # half depth (0-based decoder output 16 of 32) and the final normed state
CHUNK = 300
TOKEN_LIMIT = 2048  # SPEC §6.2: the prompt plus the answer
ANSWER_RESERVE = 8  # the longest R1 candidate, `RISK_LEVEL: 10<|im_end|>`, is 8 tokens


def chunk_ranges(n: int, size: int) -> list[tuple[int, int]]:
    """[start, end) row ranges of at most `size` rows covering 0..n."""
    if n < 1 or size < 1:
        raise ValueError(f"need n >= 1 and size >= 1, got {n}, {size}")
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def merge_parts(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate chunk arrays key by key, in chunk order; every chunk must hold the same keys."""
    if not parts:
        raise ValueError("no parts")
    keys = set(parts[0])
    for i, part in enumerate(parts):
        if set(part) != keys:
            raise ValueError(f"part {i} keys {sorted(part)} != {sorted(keys)}")
    return {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}


def check_fits(count_rows: Sequence[Mapping[str, Any]]) -> int:
    """The longest prompt; raise if any prompt plus `ANSWER_RESERVE` exceeds `TOKEN_LIMIT`."""
    over = [r["id"] for r in count_rows if r["n_prompt"] + ANSWER_RESERVE > TOKEN_LIMIT]
    if over:
        raise ValueError(f"{len(over)} prompts exceed {TOKEN_LIMIT} tokens, e.g. {over[:3]}")
    return max(int(r["n_prompt"]) for r in count_rows)


def layer_names() -> list[str]:
    return [x.strip() for x in LAYERS.split(",")]


def check_source(
    result: Mapping[str, Any], rows: Sequence[local_llm.Row], model: str, tag: str
) -> None:
    """Raise `RuntimeError` unless a chunk's child result was made from exactly `rows` (the rows
    text, by hash), by the current child script and from the current `model` snapshot."""
    local_llm.check_child_identity(tag, result, rows)
    if result.get("model_dir") != str(local_llm.model_dir(model)):
        raise RuntimeError(
            f"{tag}: made from {result.get('model_dir')!r}, not the current snapshot"
        )


def check_chunk(
    result: Mapping[str, Any], rows: Sequence[local_llm.Row], model: str, tag: str
) -> None:
    """`check_source`, and the result must be for the current `LAYERS`."""
    check_source(result, rows, model, f"embed {tag}")
    if result.get("layers") != layer_names():
        raise RuntimeError(f"{tag}: made for layers {result.get('layers')!r}, not {layer_names()}")


def _chunk_paths(work: Path, tag: str) -> tuple[Path, Path, Path]:
    """The chunk's npz, json and commit marker."""
    return work / f"{tag}.npz", work / f"{tag}.json", work / f"{tag}.done.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear_chunk(work: Path, tag: str) -> None:
    """Remove the chunk's files and marker, so a child never writes beside a stale file."""
    for path in (*_chunk_paths(work, tag), work / f"{tag}.done.json.tmp"):
        path.unlink(missing_ok=True)


def _commit_chunk(work: Path, tag: str) -> None:
    """Write the chunk's marker, holding the sha256 of its npz and json. It is written last, to a
    temporary file and then renamed, so a half-written marker never exists."""
    npz, js, done = _chunk_paths(work, tag)
    marker = {"tag": tag, "npz_sha256": _file_sha256(npz), "json_sha256": _file_sha256(js)}
    tmp = work / f"{tag}.done.json.tmp"
    tmp.write_text(json.dumps(marker, sort_keys=True) + "\n")
    os.replace(tmp, done)


def _committed(work: Path, tag: str) -> bool:
    """True if the chunk's marker exists and names the sha256 of the npz and json as they are."""
    npz, js, done = _chunk_paths(work, tag)
    try:
        marker = json.loads(done.read_text())
        return bool(
            marker["tag"] == tag
            and marker["npz_sha256"] == _file_sha256(npz)
            and marker["json_sha256"] == _file_sha256(js)
        )
    except ValueError, KeyError, TypeError, OSError:
        return False


def _load_part(
    work: Path, tag: str, rows: Sequence[local_llm.Row], model: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """A finished chunk's (child result, arrays), or None if it is missing, has no matching
    commit marker, fails the structural checks or was not made from these rows, this child, this
    snapshot and these layers."""
    npz, js, _ = _chunk_paths(work, tag)
    if not (npz.exists() and js.exists() and _committed(work, tag)):
        return None
    try:
        result = json.loads(js.read_text())
        check_chunk(result, rows, model, tag)
        arrays = local_llm.check_embeddings(npz, [r["id"] for r in rows], layer_names())
    except RuntimeError, ValueError, KeyError, TypeError, OSError:
        return None
    return dict(result), arrays


def run(world: Path, model: str, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    world = world.resolve()
    manifest = json.loads((world / "manifest.json").read_text())
    renders = [json.loads(x) for x in (world / "renders.jsonl").read_text().splitlines() if x]
    rows = prompt.child_rows(renders)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{world}: duplicate unit ids in renders.jsonl")
    if prompt.prompt_sha256() != prompt.PROMPT_SHA256:
        raise RuntimeError("prompt.py text differs from PROMPT_SHA256")

    dest = out / model
    final_npz, final_json = dest / f"{world.name}.npz", dest / f"{world.name}.json"
    if final_npz.exists() or final_json.exists():
        raise FileExistsError(f"{final_npz} or its json already exists")
    work = dest / f"{world.name}.parts"
    work.mkdir(parents=True, exist_ok=True)

    counted = local_llm.count(model, rows)
    longest = check_fits(counted["rows"])

    parts: list[dict[str, np.ndarray]] = []
    results: list[dict[str, Any]] = []
    reused = 0
    for i, (a, b) in enumerate(chunk_ranges(len(rows), CHUNK)):
        tag = f"chunk{i:03d}"
        loaded = _load_part(work, tag, rows[a:b], model)
        if loaded is None:
            _clear_chunk(work, tag)
            result, _ = local_llm.embed(model, rows[a:b], LAYERS, work, tag=tag)
            check_chunk(result, rows[a:b], model, tag)
            npz = _chunk_paths(work, tag)[0]
            part = local_llm.check_embeddings(npz, [r["id"] for r in rows[a:b]], layer_names())
            _commit_chunk(work, tag)
        else:
            result, part = loaded
            reused += 1
        if results and result["child_sha256"] != results[0]["child_sha256"]:
            raise RuntimeError(f"{tag}: made by a different child script than chunk000")
        parts.append(part)
        results.append(result)
    merged = merge_parts(parts)
    if merged["ids"].tolist() != ids:
        raise RuntimeError("merged ids differ from the render order")
    np.savez(final_npz, **dict[str, Any](merged))

    per_row = [r for res in results for r in res["rows"]]
    steady_tokens = sum(res["steady_tokens"] for res in results)
    steady_seconds = sum(res["steady_seconds"] for res in results)
    record = {
        "world": world.name,
        "world_checksum_sha256": manifest["checksum_sha256"],
        "generator_version": manifest["generator_version"],
        "model": model,
        "model_revision": local_llm.model_revision(model),
        "prompt_version": prompt.PROMPT_VERSION,
        "prompt_sha256": prompt.PROMPT_SHA256,
        "child_sha256": results[0]["child_sha256"],  # every chunk agrees (checked above)
        "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
        "npz_sha256": hashlib.sha256(final_npz.read_bytes()).hexdigest(),
        "layers": results[0]["layers"],
        "hidden_size": results[0]["hidden_size"],
        "n_rows": len(rows),
        "n_prompt": {"max": longest, "mean": float(np.mean([r["n_prompt"] for r in per_row]))},
        "chunks": len(results),
        "chunks_reused": reused,
        "peak_memory_gib": max(res["peak_memory_gib"] for res in results),
        "steady_tokens_per_s": steady_tokens / steady_seconds if steady_seconds else None,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    final_json.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    shutil.rmtree(work)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(local_llm.MODELS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = run(args.world, args.model, args.out)
    print(
        f"{record['world']} {record['model']}: {record['n_rows']} rows, "
        f"{record['steady_tokens_per_s']:.0f} tok/s, {record['wall_seconds']:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
