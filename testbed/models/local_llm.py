"""Parent-side wrapper for the MLX child (`mlx_child.py`), in the style of `pipeline/tokens.py`.

MLX lives only in the existing `es-p6-train` venv; `.venv` never imports it. Each call writes its
inputs to a work dir, runs the child out of process with `python -I` and `HF_HUB_OFFLINE=1`, checks
the exit code and the result shapes, and returns the parsed JSON. Inputs are renders only: no
labels, no database.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VENV_PYTHON = Path.home() / "venvs" / "es-p6-train" / "bin" / "python"
CHILD_SCRIPT = Path(__file__).with_name("mlx_child.py")
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

# key -> (HF cache directory name, pinned revision or revision prefix)
MODELS: dict[str, tuple[str, str]] = {
    "9b": (
        "models--mlx-community--Qwen3.5-9B-MLX-4bit",
        "938d8919941c6e7efd3c7150eff7fe9d12afa631",
    ),
    "4b": ("models--mlx-community--Qwen3.5-4B-MLX-4bit", "32f3e8ecf654"),
}

Row = Mapping[str, str]  # {"id", "before", "render", "after"}


def snapshot_dir(cache_name: str, revision: str, hub: Path | None = None) -> Path:
    """The one snapshot of `cache_name` whose name starts with `revision`; raise otherwise."""
    snapshots = (hub or HF_HUB) / cache_name / "snapshots"
    found = sorted(p for p in snapshots.glob("*") if p.name.startswith(revision))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one snapshot of {cache_name} at revision {revision}*, found {len(found)}"
        )
    return found[0]


def model_dir(key: str, hub: Path | None = None) -> Path:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; expected one of {sorted(MODELS)}")
    return snapshot_dir(*MODELS[key], hub=hub)


def model_revision(key: str) -> str:
    """The full revision of the snapshot in use (the directory name)."""
    return model_dir(key).name


@dataclass(frozen=True)
class ChildRun:
    result: dict[str, Any]
    npz: Path | None  # `embed` only
    work_dir: Path


def run_child(
    command: str,
    *,
    model_key: str | None = None,
    rows: Sequence[Row] | None = None,
    args: Sequence[str] = (),
    work_dir: Path | None = None,
    tag: str | None = None,
    timeout: float | None = None,
) -> ChildRun:
    """Run one child command and return its parsed JSON.

    Files go to `work_dir` as `<tag>.rows.jsonl`, `<tag>.json` and (for `embed`) `<tag>.npz`, and
    stay there. Without a `work_dir` they go to a temporary directory that is removed afterwards.
    """
    if work_dir is None:
        if command == "embed":
            raise ValueError("embed needs a work_dir: the npz would be deleted")
        with tempfile.TemporaryDirectory() as tmp:
            run = run_child(
                command,
                model_key=model_key,
                rows=rows,
                args=args,
                work_dir=Path(tmp),
                tag=tag,
                timeout=timeout,
            )
        return ChildRun(run.result, None, Path(tmp))

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = tag or command
    out_json, out_npz = work_dir / f"{stem}.json", work_dir / f"{stem}.npz"
    cmd = [str(VENV_PYTHON), "-I", str(CHILD_SCRIPT), command]
    if model_key is not None:
        cmd += ["--model-dir", str(model_dir(model_key))]
    if rows is not None:
        rows_file = work_dir / f"{stem}.rows.jsonl"
        rows_file.write_text("".join(json.dumps(dict(r)) + "\n" for r in rows))
        cmd += ["--in", str(rows_file)]
    cmd += list(args) + ["--out", str(out_json)]
    if command == "embed":
        cmd += ["--out-npz", str(out_npz)]
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
            f"mlx_child {command} failed (exit {done.returncode}): {done.stderr[-800:]}"
        )
    result = json.loads(out_json.read_text())
    return ChildRun(result, out_npz if command == "embed" else None, work_dir)


# --------------------------------------------------------------------------- typed commands


def count(model_key: str, rows: Sequence[Row], **kw: Any) -> dict[str, Any]:
    """The child result. `rows` holds `n_prompt`, render `span` and `render_tokens` per input row,
    in order; `template` holds the candidate answers' token ids."""
    result = run_child("count", model_key=model_key, rows=rows, **kw).result
    if [r["id"] for r in result["rows"]] != [r["id"] for r in rows]:
        raise RuntimeError("count: row ids differ from the input")
    return result


def check_embeddings(
    npz_path: Path, ids: Sequence[str], layers: Sequence[str], hidden: int | None = None
) -> dict[str, np.ndarray]:
    """Load an embed npz; check the ids match the input order and every array is finite."""
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    if arrays["ids"].tolist() != list(ids):
        raise RuntimeError("embed: npz ids differ from the input order")
    expected = {f"{kind}_{layer}" for kind in ("mean", "last") for layer in layers} | {"ids"}
    if set(arrays) != expected:
        raise RuntimeError(f"embed: npz keys {sorted(arrays)} != expected {sorted(expected)}")
    for name, value in arrays.items():
        if name == "ids":
            continue
        if value.shape[0] != len(ids) or value.ndim != 2:
            raise RuntimeError(f"embed: {name} has shape {value.shape} for {len(ids)} rows")
        if hidden is not None and value.shape[1] != hidden:
            raise RuntimeError(f"embed: {name} has width {value.shape[1]}, expected {hidden}")
        if value.dtype != np.float32:
            raise RuntimeError(f"embed: {name} is {value.dtype}, expected float32")
        if not np.isfinite(value).all():
            raise RuntimeError(f"embed: {name} has non-finite values")
    return arrays


def embed(
    model_key: str, rows: Sequence[Row], layers: str, work_dir: Path, tag: str = "embed"
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The R2 extractor: (child result, arrays `mean_<L>` / `last_<L>` / `ids`), checked."""
    run = run_child(
        "embed",
        model_key=model_key,
        rows=rows,
        args=["--layers", layers],
        work_dir=work_dir,
        tag=tag,
    )
    if run.npz is None:
        raise RuntimeError("embed produced no npz")
    names = [x.strip() for x in layers.split(",")]
    arrays = check_embeddings(
        run.npz, [r["id"] for r in rows], names, run.result.get("hidden_size")
    )
    if [r["id"] for r in run.result["rows"]] != [r["id"] for r in rows]:
        raise RuntimeError("embed: row ids in the JSON differ from the input")
    return run.result, arrays


def score(model_key: str, rows: Sequence[Row], **kw: Any) -> dict[str, Any]:
    result = run_child("score", model_key=model_key, rows=rows, **kw).result
    if [r["id"] for r in result["rows"]] != [r["id"] for r in rows]:
        raise RuntimeError("score: row ids differ from the input")
    return result


def lora_bench(
    model_key: str,
    rows: Sequence[Row],
    steps: int,
    warmup: int,
    batch_size: int,
    **kw: Any,
) -> dict[str, Any]:
    args = ["--steps", str(steps), "--warmup", str(warmup), "--batch-size", str(batch_size)]
    result = run_child("lora-bench", model_key=model_key, rows=rows, args=args, **kw).result
    done = len(result["steps"])
    if result["error"] is None and done != steps + warmup:
        raise RuntimeError(f"lora-bench: {done} steps reported, expected {steps + warmup}")
    return result


def selftest_tiny(**kw: Any) -> dict[str, Any]:
    return run_child("selftest-tiny", **kw).result
