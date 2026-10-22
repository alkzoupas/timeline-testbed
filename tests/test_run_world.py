"""End-to-end test of the world runner on the test-only small scale (SPEC §4, §6.1, §6.2)."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES
from testbed.pipeline.render import RENDER_VERSION, parse, render
from testbed.pipeline.run_world import main, run_world
from testbed.pipeline.timeline import load_timeline
from testbed.pipeline.tokens import VENV_PYTHON, count_tokens, tokenizer_file
from testbed.world.generator import WorldSpec
from testbed.world.persist import Stores

SPEC = WorldSpec(seed=1, s=1.0, scale="small")
ROW_KEYS = {"unit", "members", "detector_score", "f12", "f64", "tokens", "render_sha256"}


def _tokenizer_present() -> bool:
    if not VENV_PYTHON.exists():
        return False
    try:
        tokenizer_file()
    except FileNotFoundError:
        return False
    return True


HAVE_TOKENIZER = _tokenizer_present()


def _stub_tokens(texts: Sequence[str]) -> list[int]:
    return [len(t) // 4 for t in texts]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("run") / "world"
    counter = count_tokens if HAVE_TOKENIZER else _stub_tokens
    return out, run_world(SPEC, out, token_counter=counter)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_files_and_run_record(run: tuple[Path, dict[str, Any]]) -> None:
    out, record = run
    assert {p.name for p in out.iterdir()} >= {
        "manifest.json",
        "units.jsonl",
        "renders.jsonl",
        "run.json",
        "inference.db",
    }
    assert json.loads((out / "run.json").read_text()) == record
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 1 and manifest["scale"] == "small"
    assert record["render_version"] == RENDER_VERSION
    assert record["counts"]["units"] == 120
    assert record["counts"]["accounts"] == manifest["counts"]["accounts"]
    assert set(record["seconds"]) == {
        "generate",
        "persist",
        "detect",
        "units",
        "features",
        "tokens",
        "total",
    }
    assert len(record["detection_sha256"]) == 64
    tokens = [r["tokens"] for r in _jsonl(out / "units.jsonl")]
    assert record["max_tokens"] == max(tokens)
    assert record["mean_tokens"] == round(sum(tokens) / len(tokens), 2)
    assert record["units_over_budget"] == sum(t > record["token_budget"] for t in tokens)


def test_one_row_per_unit_with_the_schema(run: tuple[Path, dict[str, Any]]) -> None:
    out, record = run
    rows = _jsonl(out / "units.jsonl")
    renders = _jsonl(out / "renders.jsonl")
    assert len(rows) == len(renders) == record["counts"]["units"]
    keys = [r["unit"] for r in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert keys == [r["unit"] for r in renders]
    for row in rows:
        assert set(row) == ROW_KEYS
        assert row["members"] == sorted(row["members"]) and len(row["members"]) >= 2
        assert math.isfinite(row["detector_score"])
        assert list(row["f12"]) == sorted(F_IDS)
        assert all(math.isfinite(v) for v in row["f12"].values())


def test_f64_has_64_finite_values(run: tuple[Path, dict[str, Any]]) -> None:
    out, _ = run
    for row in _jsonl(out / "units.jsonl"):
        assert len(row["f64"]) == 64
        assert set(row["f64"]) == set(FEATURES)
        assert all(math.isfinite(v) for v in row["f64"].values())


def test_render_hash_and_text(run: tuple[Path, dict[str, Any]]) -> None:
    out, _ = run
    stores = Stores.under(out)
    with closing(sqlite3.connect(f"file:{stores.inference_db}?mode=ro", uri=True)) as conn:
        for row, rendered in zip(
            _jsonl(out / "units.jsonl"), _jsonl(out / "renders.jsonl"), strict=True
        ):
            text = rendered["text"]
            assert row["render_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            assert text == render(load_timeline(conn, row["members"]))
            assert render(parse(text)) == text


@pytest.mark.skipif(not HAVE_TOKENIZER, reason="the es-p6-train venv or the tokenizer is missing")
def test_tokens_match_count_tokens(run: tuple[Path, dict[str, Any]]) -> None:
    out, _ = run
    rows = _jsonl(out / "units.jsonl")
    counts = count_tokens([r["text"] for r in _jsonl(out / "renders.jsonl")])
    assert [r["tokens"] for r in rows] == counts


def test_stub_counter_is_used_per_unit(tmp_path: Path) -> None:
    """The counter is called once, in unit order, and a short answer is an error."""
    seen: list[int] = []

    def counter(texts: Sequence[str]) -> list[int]:
        seen.append(len(texts))
        return [7] * len(texts)

    record = run_world(SPEC, tmp_path / "w", token_counter=counter)
    assert seen == [record["counts"]["units"]]
    assert record["max_tokens"] == 7 and record["mean_tokens"] == 7.0
    assert record["units_over_budget"] == 0
    with pytest.raises(RuntimeError, match="wrong number"):
        run_world(SPEC, tmp_path / "w2", token_counter=lambda texts: [1])


def test_rerun_into_the_same_directory_refuses(run: tuple[Path, dict[str, Any]]) -> None:
    out, _ = run
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(FileExistsError):
        run_world(SPEC, out, token_counter=_stub_tokens)
    assert sorted(p.name for p in out.iterdir()) == before


def test_empty_existing_directory_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    record = run_world(SPEC, tmp_path / "empty", token_counter=_stub_tokens)
    assert record["counts"]["units"] == 120


def test_cli_rejects_the_test_only_scale(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--seed", "1", "--s", "1.0", "--scale", "small", "--out", "unused"])
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
