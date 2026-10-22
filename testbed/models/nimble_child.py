# transformers and the Nimble repo live only in the Nimble MLX venv, not .venv.
# mypy: disable-error-code="import-not-found"
"""Nimble child for R1's second model (SPEC §7 R1): run in the Nimble MLX venv, not `.venv`.

    ~/code/nimble/.venv-mlx/bin/python -I testbed/models/nimble_child.py <command> ...

Commands (each reads `--in` rows and writes a JSON result to `--out`):

- `count`: the Nimble prompt's token count per row. No model is loaded.
- `score`: Bespoke-Nimble-9B (the merged model, the predecessor's arm "b") over each row's context
  with the repo's `ParallelScorer.evaluate`, mode `parallel`. The ten candidate logits are put
  through a softmax at T = 1.0; per row `p`, `expected` = sum n * p_n and `argmax` follow.

This is a copy, trimmed to what R1 needs, of `vendor/entitlement-shield/scripts/p6_3_nimble_run.py`
(vendor/ is read-only). It runs outside the package (`-I`), so it imports nothing from `testbed`
and never opens a database; `.venv` imports it only for the pins and the pure helpers, since
`mlx`, `transformers` and the Nimble repo are imported inside functions. It stays Python
3.12-compatible.

A row is `{"id", "context"}`: the render text alone, at full length. The schema comes as a file
whose bytes must hash to `--schema-sha256` (`prompt.NIMBLE_SCHEMA_SHA256`). A row whose prompt
does not fit `MAX_INPUT_TOKENS` fails the `score` command; nothing is truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

NIMBLE = Path.home() / "code" / "nimble"
NIMBLE_COMMIT = "4edb0703a2e75d0170c3a45645a1961f4d1a2867"
HUB = NIMBLE / ".cache" / "huggingface" / "hub"
ADAPTER_REPO = "bespokelabs/Bespoke-Nimble-9B"
ADAPTER_REVISION = "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
ADAPTER_SHA256 = "29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a"
BASE_REPO = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER = HUB / "models--bespokelabs--Bespoke-Nimble-9B" / "snapshots" / ADAPTER_REVISION
MODEL_DIR = NIMBLE / ".cache" / "models" / f"nimble-9b-{ADAPTER_REVISION}"

MAX_INPUT_TOKENS = 8192  # the predecessor's fit limit; SPEC §6.2's 2,048 is reported, not enforced
UNLIMITED = 1 << 20  # `count` measures the full length whatever it is
TEMPERATURE = 1.0  # shipped, unfitted (temperature_config.json)
MODE = "parallel"
FIELD = "risk_level"
LEVELS = [str(k) for k in range(1, 11)]
SUM_TOLERANCE = 1e-4
MEMORY_LIMIT_GIB = 40  # of 48 GiB unified memory; keeps the machine out of swap


# --------------------------------------------------------------------------- pure helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


CHILD_SHA256 = sha256_file(Path(__file__))  # this source, at startup


def check_schema_text(text: str, sha256: str) -> dict[str, Any]:
    """The schema dict, if `text` hashes to `sha256` and is the one `risk_level` field over the
    ten levels 1-10 in order; `SystemExit` otherwise."""
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        raise SystemExit("schema file differs from --schema-sha256")
    schema = json.loads(text)
    if list(schema) != [FIELD] or schema[FIELD].get("choices") != LEVELS:
        raise SystemExit("schema is not the one risk_level field over the choices 1-10")
    return dict(schema)


def check_ready(ready: dict[str, Any]) -> None:
    """The merged model's `READY.json` must name the pinned adapter."""
    if ready.get("model") != ADAPTER_REPO or ready.get("revision") != ADAPTER_REVISION:
        raise SystemExit("merged model was made from another adapter release")
    if ready.get("adapter_sha256") != ADAPTER_SHA256:
        raise SystemExit("merged model was made from another adapter")


def check_contract_base(contract: dict[str, Any]) -> None:
    if (contract.get("model"), contract.get("revision")) != (BASE_REPO, BASE_REVISION):
        raise SystemExit("adapter contract names another base")


def softmax(logits: Sequence[float], temperature: float = TEMPERATURE) -> list[float]:
    top = max(logits)
    exps = [math.exp((x - top) / temperature) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def summarize(logits: Sequence[float], choices: Sequence[Any]) -> dict[str, Any]:
    """R1's Nimble score from the ten candidate logits, in level order 1-10.

    `p` is the softmax at T = 1.0, `expected` = sum n * p_n and `argmax` the most likely level
    (an int, 1-10; the first on a tie). Raises if the choices are not 1-10 in order, a logit is
    not finite, or `p` does not sum to 1."""
    if [str(c) for c in choices] != LEVELS:
        raise ValueError(f"choices must be the levels {LEVELS}, got {list(choices)}")
    if len(logits) != len(LEVELS) or not all(math.isfinite(x) for x in logits):
        raise ValueError(f"need {len(LEVELS)} finite logits, got {list(logits)}")
    p = softmax(logits)
    if abs(sum(p) - 1) > SUM_TOLERANCE:
        raise ValueError(f"probabilities sum to {sum(p)}")
    best = max(range(len(p)), key=p.__getitem__)
    return {
        "p": p,
        "expected": sum(n * x for n, x in enumerate(p, 1)),
        "argmax": best + 1,
    }


def read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    """The rows and the sha256 of the file's bytes as read."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    for r in rows:
        if set(r) != {"id", "context"}:
            raise SystemExit(f"row keys must be id/context, got {sorted(r)}")
        if not isinstance(r["context"], str) or not r["context"].strip():
            raise SystemExit(f"{r['id']}: the context must be a nonempty string")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate row ids")
    return rows, hashlib.sha256(raw).hexdigest()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n")


# --------------------------------------------------------------------------- Nimble side


def setup() -> tuple[Any, Callable[..., Any], dict[str, Any]]:
    """The adapter's tokenizer, contract prompt builder and provenance, from the pinned repo."""
    head = subprocess.run(
        ["git", "-C", str(NIMBLE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != NIMBLE_COMMIT:
        raise SystemExit(f"Nimble repo at {head}, expected {NIMBLE_COMMIT}")
    sys.path.insert(0, str(NIMBLE))
    from nimble.training.candidate_schema import validate_contract
    from transformers import AutoTokenizer

    contract_path = ADAPTER / "schema_config.json"
    contract = json.loads(contract_path.read_text())
    check_contract_base(contract)
    if sha256_file(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise SystemExit("adapter weights differ from the pinned release")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER))
    prepare = validate_contract(contract, tokenizer)
    provenance = {
        "nimble_commit": NIMBLE_COMMIT,
        "adapter_repo": ADAPTER_REPO,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_sha256": ADAPTER_SHA256,
        "contract_sha256": sha256_file(contract_path),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "temperature": TEMPERATURE,
        "mode": MODE,
        "packages": {p: md.version(p) for p in ("mlx", "mlx-lm", "transformers")},
    }
    return tokenizer, prepare, provenance


def load_schema(args: argparse.Namespace) -> dict[str, Any]:
    text = args.schema.read_text()
    schema = check_schema_text(text, args.schema_sha256)
    sys.path.insert(0, str(NIMBLE))
    from nimble.scoring.serving_schema import parse_schema

    if parse_schema(text) != schema:
        raise SystemExit("schema parsed differently by the Nimble repo")
    return schema


def _provenance(
    provenance: dict[str, Any], args: argparse.Namespace, command: str, t0: float, rows_sha256: str
) -> dict[str, Any]:
    return {
        "command": command,
        "provenance": {
            **provenance,
            "nimble_schema_sha256": args.schema_sha256,
            "child_sha256": CHILD_SHA256,
        },
        "rows_sha256": rows_sha256,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- commands


def cmd_count(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    schema = load_schema(args)
    tokenizer, prepare, provenance = setup()
    out = []
    rows, rows_sha256 = read_rows(args.inp)
    for row in rows:
        prepared = prepare(tokenizer, row["context"], schema, UNLIMITED)
        out.append({"id": row["id"], "n_prompt": len(prepared.full_ids[0])})
    write_json(args.out, {**_provenance(provenance, args, "count", t0, rows_sha256), "rows": out})


def cmd_score(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    rows, rows_sha256 = read_rows(args.inp)
    schema = load_schema(args)
    tokenizer, prepare, provenance = setup()
    check_ready(json.loads((MODEL_DIR / "READY.json").read_text()))

    import mlx.core as mx

    mx.set_memory_limit(MEMORY_LIMIT_GIB * 2**30)
    sys.path.insert(0, str(NIMBLE))
    from nimble.scoring.parallel_scorer import ParallelScorer

    scorer = ParallelScorer(
        model_path=str(MODEL_DIR),
        max_input_tokens=MAX_INPUT_TOKENS,
        temperature=TEMPERATURE,
        model_id=ADAPTER_REPO,
        revision=ADAPTER_REVISION,
    )
    load_seconds = time.perf_counter() - t0
    out, peak = [], 0.0
    for row in rows:
        try:
            prepared = prepare(tokenizer, row["context"], schema, MAX_INPUT_TOKENS)
        except ValueError as error:
            if "Nothing was truncated" in str(error):
                raise SystemExit(f"{row['id']}: {error}") from error
            raise
        n_prompt = len(prepared.full_ids[0])
        logit_rows, stats = scorer.evaluate(prepared, mode=MODE)
        logits = [float(x) for x in logit_rows[0].tolist()]
        entry = summarize(logits, prepared.choices[0])
        peak = max(peak, float(stats["mlx_peak_active_gib"]))
        out.append(
            {"id": row["id"], "n_prompt": n_prompt, **entry, "seconds": stats["total_seconds"]}
        )
        mx.clear_cache()
        print(
            f"{row['id']} tokens={n_prompt} E={entry['expected']:.3f}", file=sys.stderr, flush=True
        )
    rest = out[1:]  # the first row pays the kernel compile
    steady_tokens = sum(r["n_prompt"] for r in rest)
    steady_seconds = sum(r["seconds"] for r in rest)
    write_json(
        args.out,
        {
            **_provenance(provenance, args, "score", t0, rows_sha256),
            "model_dir": str(MODEL_DIR),
            "model_file_sizes": {p.name: p.stat().st_size for p in MODEL_DIR.glob("*.safetensors")},
            "load_seconds": round(load_seconds, 2),
            "peak_memory_gib": round(peak, 3),
            "rows": out,
            "steady_tokens": steady_tokens,
            "steady_seconds": steady_seconds,
            "steady_tokens_per_s": steady_tokens / steady_seconds if steady_seconds else None,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("count", cmd_count), ("score", cmd_score)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--in", dest="inp", type=Path, required=True)
        p.add_argument("--schema", type=Path, required=True)
        p.add_argument("--schema-sha256", required=True)
        p.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
