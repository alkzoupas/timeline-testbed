"""Known-answer tests for the R1 run (testbed/models/r1_run.py) and its report
(testbed/eval/r1.py), on synthetic units and fake child results. No model is loaded."""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from testbed.eval import r0, r1
from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES as F64_NAMES
from testbed.models import local_llm, mlx_child, nimble_child, nimble_run, prompt, r1_run
from testbed.world.families import PAIRS

REVISION = "rev-test"
CHILD = "c" * 64  # the "current" mlx_child.py hash
OLD_CHILD = "d" * 64


@pytest.fixture(autouse=True)
def _revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_llm, "model_revision", lambda key: REVISION)
    monkeypatch.setattr(local_llm, "model_dir", lambda key: Path("/models") / f"snap-{key}")
    monkeypatch.setattr(local_llm, "child_sha256", lambda: CHILD)


def make_units(world: str, n_pos: int = 10, n_neg: int = 10) -> list[r0.R0Unit]:
    rng = np.random.default_rng(zlib.crc32(world.encode()))
    units = []
    for pair in PAIRS:
        for k in range(n_pos + n_neg):
            positive = k < n_pos
            units.append(
                r0.R0Unit(
                    world,
                    f"{world}:p{pair}-{k:03d}",
                    PAIRS[pair][0 if positive else 1],
                    pair,
                    positive,
                    tuple(rng.normal(size=len(F_IDS))),
                    tuple(rng.normal(size=len(F64_NAMES))),
                    float(rng.random()),
                    None,
                    f"{world}-u{pair}{k:03d}",
                )
            )
    units.sort(key=lambda u: u.group)
    return units


def make_world(root: Path, name: str, checksum: str = "sum-1") -> Path:
    path = root / name
    path.mkdir(parents=True)
    manifest = {"checksum_sha256": checksum, "s": 0.75, "generator_version": "g1"}
    (path / "manifest.json").write_text(json.dumps(manifest))
    write_renders(path, [f"render {i} of {name}" for i in range(3)])
    return path


def write_renders(world: Path, texts: Sequence[str], unit: str = "r") -> None:
    lines = [json.dumps({"unit": f"{unit}{i}", "text": t}) for i, t in enumerate(texts)]
    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")


def rows_sha256_of(world: Path, limit: int | None = None) -> str:
    """The hash of the rows text `r1_run` would write for `world`, built here by hand."""
    out = ""
    for line in (world / "renders.jsonl").read_text().splitlines()[:limit]:
        r = json.loads(line)
        row = {"id": r["unit"], "before": prompt.BEFORE, "render": r["text"], "after": prompt.AFTER}
        out += json.dumps(row) + "\n"
    return hashlib.sha256(out.encode()).hexdigest()


def level_row(unit: str, expected_peak: int = 3, shift: float = 0.0) -> dict[str, Any]:
    """A child row whose answer mass peaks at `expected_peak`."""
    lps = [-1.0 - 0.7 * abs(n - expected_peak) + shift for n in range(1, 11)]
    return {
        "id": unit,
        "n_prompt": 900,
        "seconds": 3.0,
        "logprobs": lps,
        **mlx_child.level_summary(lps),
    }


def write_r1(
    r1_dir: Path,
    model: str,
    world: Path,
    units: Sequence[r0.R0Unit],
    expected: NDArray[np.float64],
    *,
    reverse: bool = False,
    extra: Sequence[str] = (),
    prompt_sha: str = prompt.PROMPT_SHA256,
    checksum: str = "sum-1",
    revision: str = REVISION,
    head_dtype: str = "float32",
    rows_sha: str | None = None,
    child_sha: str = CHILD,
    drop: Sequence[str] = (),
    limit: int | None = None,
) -> None:
    order = list(range(len(units)))[::-1] if reverse else list(range(len(units)))
    rows = [{"id": units[i].unit, "expected": float(expected[i])} for i in order]
    rows += [{"id": e, "expected": 99.0} for e in extra]
    dest = r1_dir / model
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{world.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "prompt_sha256": prompt_sha,
        "model_revision": revision,
        "world_checksum_sha256": checksum,
        "head_dtype": head_dtype,
        "rows_sha256": rows_sha or rows_sha256_of(world, limit),
        "child_sha256": child_sha,
        "limit": limit,
        "agreement": {"n_distinct_expected": len(set(expected.tolist()))},
    }
    for field in drop:
        del meta[field]
    (dest / f"{world.name}.json").write_text(json.dumps(meta))


# --- Pure helpers ---------------------------------------------------------------------------------


def _softmax(logits: Sequence[float]) -> list[float]:
    z = np.exp(np.array(logits) - max(logits))
    return (z / z.sum()).tolist()


def test_on_grid_pairs_counts_log_odds_differences_that_are_multiples_of_an_eighth() -> None:
    on_grid = _softmax([0.0, 0.125, 0.375, -0.25, 1.0, -0.5, 0.625, 0.0, 0.125, 0.25])
    assert r1_run.on_grid_pairs(on_grid) == 45  # all 10 candidates, all 45 pairs
    off_grid = _softmax(
        [0.0, 0.0731, 0.3219, -0.2013, 1.0583, -0.5917, 0.6347, 0.0211, 0.4409, 0.2917]
    )
    assert r1_run.on_grid_pairs(off_grid) == 0
    # a candidate under the floor is ignored: three on-grid candidates plus one off-grid tail
    tail = _softmax([0.0, 0.125, 0.25, -20.0])
    assert r1_run.on_grid_pairs(tail) == 3
    assert r1_run.on_grid_pairs([1.0]) == 0


def test_agreement_summary_counts_distinct_scores_and_the_worst_gap() -> None:
    rows: list[dict[str, Any]] = [
        {"p": _softmax([0.0, 0.125]), "expected": 4.0, "mass": 0.5},
        {"p": _softmax([0.0, 0.1]), "expected": 4.0, "mass": 0.4},
        {"p": _softmax([0.0, 0.7]), "expected": 5.0, "mass": 0.9},
    ]
    s = r1_run.agreement_summary(rows, check_full=False)
    assert (s["n_rows"], s["n_distinct_expected"]) == (3, 2)
    assert (s["rows_with_grid_pairs"], s["grid_pairs"]) == (1, 1)
    assert s["min_mass"] == 0.4 and "max_abs_diff" not in s
    full = [
        {**r, "expected_full": r["expected"] + d, "max_abs_diff": m}
        for r, d, m in zip(rows, (0.0, -0.25, 0.1), (0.01, 0.07, 0.02), strict=True)
    ]
    s = r1_run.agreement_summary(full, check_full=True)
    assert s["max_abs_diff"] == 0.07 and s["max_expected_diff"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="no rows"):
        r1_run.agreement_summary([], check_full=False)


# --- r1_run ---------------------------------------------------------------------------------------


class FakeChild:
    """Stands in for `local_llm.count` and `local_llm.zeroshot`; writes the chunk json the way
    `run_child` does, and can fail on a chosen call."""

    def __init__(self, fail_on_call: int | None = None, n_prompt: int = 900) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call
        self.n_prompt = n_prompt
        self.child_sha256 = CHILD  # what a chunk made now reports
        self.switch_child_at_call: int | None = None  # edit the child before this zeroshot call

    def count(self, model: str, rows: Sequence[Any], **kw: Any) -> dict[str, Any]:
        return {"rows": [{"id": r["id"], "n_prompt": self.n_prompt} for r in rows]}

    def zeroshot(
        self, model: str, rows: Sequence[Any], check_full: bool = False, **kw: Any
    ) -> dict[str, Any]:
        if self.switch_child_at_call == len(self.calls):
            self.child_sha256 = OLD_CHILD
        self.calls.append([r["id"] for r in rows])
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("child died")
        out = []
        for r in rows:
            row = level_row(r["id"], expected_peak=1 + zlib.crc32(r["id"].encode()) % 10)
            if check_full:
                row |= {
                    "logprobs_full": row["logprobs"],
                    "expected_full": row["expected"] + 0.01,
                    "max_abs_diff": 0.02,
                }
            out.append(row)
        result = {
            "model_dir": str(local_llm.model_dir(model)),
            "child_sha256": self.child_sha256,
            "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
            "head_dtype": "float32",
            "check_full": check_full,
            "rows": out,
            "steady_tokens": 900 * (len(rows) - 1),
            "steady_seconds": 3.0 * (len(rows) - 1),
            "peak_memory_gib": 7.5,
        }
        local_llm.check_zeroshot(result, rows, check_full)
        (Path(kw["work_dir"]) / f"{kw['tag']}.json").write_text(json.dumps(result))
        return result


@pytest.fixture
def small_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    world = make_world(tmp_path, "w1")
    renders = [{"unit": f"u{i}", "text": f"render {i}"} for i in range(5)]
    (world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
    monkeypatch.setattr(r1_run, "CHUNK", 2)
    return world


def install(monkeypatch: pytest.MonkeyPatch, child: FakeChild) -> None:
    monkeypatch.setattr(local_llm, "count", child.count)
    monkeypatch.setattr(local_llm, "zeroshot", child.zeroshot)


def test_run_writes_one_row_per_unit_in_render_order_with_provenance(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeChild()
    install(monkeypatch, child)
    out = tmp_path / "r1"
    record = r1_run.run(small_world, "9b", out)
    assert child.calls == [["u0", "u1"], ["u2", "u3"], ["u4"]]
    rows = [json.loads(x) for x in (out / "9b" / "w1.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [f"u{i}" for i in range(5)]
    assert all(set(r) == set(r1_run.ROW_FIELDS) for r in rows)
    assert all(len(r["logprobs"]) == 10 and len(r["p"]) == 10 for r in rows)
    for r in rows:  # the score is the expectation of the normalized distribution
        assert r["expected"] == pytest.approx(sum(n * p for n, p in enumerate(r["p"], 1)))
    saved = json.loads((out / "9b" / "w1.json").read_text())
    assert saved == record
    assert saved["prompt_sha256"] == prompt.PROMPT_SHA256 and saved["head_dtype"] == "float32"
    assert saved["world_checksum_sha256"] == "sum-1" and saved["model_revision"] == REVISION
    assert saved["limit"] is None and saved["check_full"] is False and saved["n_rows"] == 5
    assert saved["chunks"] == 3 and saved["chunks_reused"] == 0
    assert saved["steady_tokens_per_s"] == pytest.approx(300.0)  # 900 tokens per 3 s
    assert saved["steady_seconds_per_unit"] == pytest.approx(3.0)
    assert saved["agreement"]["n_rows"] == 5 and "max_abs_diff" not in saved["agreement"]
    assert not (out / "9b" / "w1.parts").exists()
    with pytest.raises(FileExistsError):
        r1_run.run(small_world, "9b", out)


def test_run_resumes_from_the_chunks_already_written(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "r1"
    install(monkeypatch, FakeChild(fail_on_call=3))
    with pytest.raises(RuntimeError, match="child died"):
        r1_run.run(small_world, "9b", out)
    assert sorted(p.name for p in (out / "9b" / "w1.parts").glob("*.json")) == [
        "chunk000.json",
        "chunk001.json",
    ]
    assert not (out / "9b" / "w1.jsonl").exists()
    # a damaged chunk is redone, an intact one reused
    (out / "9b" / "w1.parts" / "chunk001.json").write_text("{not json")
    resumed = FakeChild()
    install(monkeypatch, resumed)
    record = r1_run.run(small_world, "9b", out)
    assert resumed.calls == [["u2", "u3"], ["u4"]]
    assert record["chunks_reused"] == 1 and record["n_rows"] == 5


def _crashed_run(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Run until the third zeroshot call dies: chunks 0 and 1 are left in the parts directory."""
    out = tmp_path / "r1"
    install(monkeypatch, FakeChild(fail_on_call=3))
    with pytest.raises(RuntimeError, match="child died"):
        r1_run.run(small_world, "9b", out)
    return out, out / "9b" / "w1.parts"


def test_run_reuses_chunks_made_from_the_same_rows_child_and_snapshot(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = _crashed_run(small_world, tmp_path, monkeypatch)
    resumed = FakeChild()
    install(monkeypatch, resumed)
    record = r1_run.run(small_world, "9b", out)
    assert resumed.calls == [["u4"]] and record["chunks_reused"] == 2


def test_run_redoes_every_chunk_when_the_render_text_changed_under_the_same_ids(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = _crashed_run(small_world, tmp_path, monkeypatch)
    renders = [{"unit": f"u{i}", "text": f"edited {i}"} for i in range(5)]
    (small_world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
    resumed = FakeChild()
    install(monkeypatch, resumed)
    record = r1_run.run(small_world, "9b", out)
    assert resumed.calls == [["u0", "u1"], ["u2", "u3"], ["u4"]] and record["chunks_reused"] == 0
    assert record["rows_sha256"] == rows_sha256_of(small_world)


@pytest.mark.parametrize(
    "stale",
    [
        {"child_sha256": OLD_CHILD},  # mlx_child.py was edited since
        {"model_dir": "/models/other-snapshot"},
        {"rows_sha256": "0" * 64},
    ],
)
def test_run_redoes_a_chunk_that_names_another_child_snapshot_or_rows(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: dict[str, str]
) -> None:
    out, parts = _crashed_run(small_world, tmp_path, monkeypatch)
    saved = json.loads((parts / "chunk000.json").read_text())
    (parts / "chunk000.json").write_text(json.dumps({**saved, **stale}))
    resumed = FakeChild()
    install(monkeypatch, resumed)
    record = r1_run.run(small_world, "9b", out)
    assert resumed.calls == [["u0", "u1"], ["u4"]] and record["chunks_reused"] == 1


@pytest.mark.parametrize("field", ["child_sha256", "rows_sha256", "model_dir"])
def test_run_redoes_a_chunk_that_lacks_a_binding_field(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    out, parts = _crashed_run(small_world, tmp_path, monkeypatch)
    saved = json.loads((parts / "chunk001.json").read_text())
    del saved[field]
    (parts / "chunk001.json").write_text(json.dumps(saved))
    resumed = FakeChild()
    install(monkeypatch, resumed)
    r1_run.run(small_world, "9b", out)
    assert resumed.calls == [["u2", "u3"], ["u4"]]


def test_run_records_the_child_and_rows_hashes(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeChild())
    record = r1_run.run(small_world, "9b", tmp_path / "r1")
    assert record["child_sha256"] == CHILD
    assert record["rows_sha256"] == rows_sha256_of(small_world)
    limited = r1_run.run(small_world, "9b", tmp_path / "r2", limit=3)
    assert limited["rows_sha256"] == rows_sha256_of(small_world, 3)


def test_run_refuses_chunks_made_by_different_children(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeChild()
    child.switch_child_at_call = 1  # edited while the run is going
    monkeypatch.setattr(local_llm, "child_sha256", lambda: child.child_sha256)
    install(monkeypatch, child)
    with pytest.raises(RuntimeError, match="different child script"):
        r1_run.run(small_world, "9b", tmp_path / "r1")
    assert not (tmp_path / "r1" / "9b" / "w1.json").exists()


def test_run_limit_and_check_full(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeChild())
    record = r1_run.run(small_world, "9b", tmp_path / "r1", limit=3, check_full=True)
    assert record["n_rows"] == 3 and record["limit"] == 3 and record["check_full"] is True
    assert record["agreement"]["max_abs_diff"] == 0.02
    assert record["agreement"]["max_expected_diff"] == pytest.approx(0.01)
    rows = [json.loads(x) for x in (tmp_path / "r1" / "9b" / "w1.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(
        set(r) == set(r1_run.ROW_FIELDS + r1_run.CHECK_FIELDS) for r in rows
    )
    with pytest.raises(ValueError, match="at least 1"):
        r1_run.run(small_world, "9b", tmp_path / "r2", limit=0)


def test_run_refuses_a_prompt_over_the_token_limit(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeChild(n_prompt=2048 - 8 + 1)
    install(monkeypatch, child)
    with pytest.raises(ValueError, match="exceed 2048"):
        r1_run.run(small_world, "9b", tmp_path / "r1")
    assert child.calls == []  # before any forward pass
    ok = FakeChild(n_prompt=2048 - 8)
    install(monkeypatch, ok)
    assert r1_run.run(small_world, "9b", tmp_path / "r1")["n_prompt"]["max"] == 2040


# --- load_r1 --------------------------------------------------------------------------------------


def test_load_r1_aligns_by_unit_id_and_pools_worlds(tmp_path: Path) -> None:
    w1, w2 = make_world(tmp_path, "w1"), make_world(tmp_path, "w2")
    u1, u2 = make_units("w1"), make_units("w2")
    e1, e2 = np.linspace(1, 10, len(u1)), np.linspace(10, 1, len(u2))
    write_r1(tmp_path / "r", "9b", w1, u1, e1, reverse=True, extra=["mixed-cluster"])
    write_r1(tmp_path / "r", "9b", w2, u2, e2)
    np.testing.assert_array_equal(r1.load_r1(tmp_path / "r", "9b", w1, u1), e1)
    pooled = (u1 + u2)[::-1][::3] + (u1 + u2)[::-1][1::3] + (u1 + u2)[::-1][2::3]
    by_unit = dict(zip([u.unit for u in u1 + u2], np.concatenate([e1, e2]), strict=True))
    got = r1.pooled_r1(tmp_path / "r", "9b", [w1, w2], pooled)
    np.testing.assert_array_equal(got, [by_unit[u.unit] for u in pooled])
    with pytest.raises(ValueError, match="worlds not given"):
        r1.pooled_r1(tmp_path / "r", "9b", [w1], pooled)


def test_load_r1_reads_a_world_whose_name_holds_a_dot(tmp_path: Path) -> None:
    # real world names carry s (m2-dev-s0.5-seed1); `with_suffix` would read m2-dev-s0.json
    world = make_world(tmp_path, "m2-dev-s0.5-seed1")
    units = make_units("m2-dev-s0.5-seed1")
    expected = np.linspace(1, 10, len(units))
    write_r1(tmp_path / "r", "9b", world, units, expected)
    np.testing.assert_array_equal(r1.load_r1(tmp_path / "r", "9b", world, units), expected)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"prompt_sha": "0" * 64}, "different prompt"),
        ({"checksum": "other"}, "different world"),
        ({"revision": "rev-old"}, "different 9b revision"),
        ({"head_dtype": "bfloat16"}, "head dtype"),
        ({"rows_sha": "0" * 64}, "rows_sha256 differs"),
        ({"child_sha": "0" * 64}, "different mlx_child.py"),
        ({"drop": ["rows_sha256"]}, "no rows_sha256"),
        ({"drop": ["child_sha256"]}, "no child_sha256"),
    ],
)
def test_load_r1_refuses_each_provenance_mismatch(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_r1(tmp_path / "r", "9b", world, units, np.ones(len(units)), **override)
    with pytest.raises(ValueError, match=message):
        r1.load_r1(tmp_path / "r", "9b", world, units)


def test_load_r1_refuses_rows_changed_after_the_run(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_r1(tmp_path / "r", "9b", world, units, np.ones(len(units)))
    r1.load_r1(tmp_path / "r", "9b", world, units)  # fine as run
    write_renders(world, ["render 0 of w1", "edited", "render 2 of w1"])  # same ids, other text
    with pytest.raises(ValueError, match="w1: R1 run rows_sha256 differs"):
        r1.load_r1(tmp_path / "r", "9b", world, units)


def test_load_r1_checks_a_limited_runs_rows_against_the_first_renders(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")[:2]
    write_r1(tmp_path / "r", "9b", world, units, np.ones(2), limit=2)
    r1.load_r1(tmp_path / "r", "9b", world, units)
    write_r1(
        tmp_path / "r", "9b", world, units, np.ones(2), limit=2, rows_sha=rows_sha256_of(world)
    )
    with pytest.raises(ValueError, match="rows_sha256 differs"):  # the hash of all 3 renders
        r1.load_r1(tmp_path / "r", "9b", world, units)


def test_load_r1_refuses_missing_duplicate_and_non_finite_scores(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    ones = np.ones(len(units))
    write_r1(tmp_path / "r", "9b", world, units[:-1], ones[:-1])
    with pytest.raises(ValueError, match="1 units have no row"):
        r1.load_r1(tmp_path / "r", "9b", world, units)
    write_r1(tmp_path / "r", "9b", world, units, ones, extra=[units[2].unit])
    with pytest.raises(ValueError, match="duplicate ids"):
        r1.load_r1(tmp_path / "r", "9b", world, units)
    bad = ones.copy()
    bad[4] = np.nan
    write_r1(tmp_path / "r", "9b", world, units, bad)
    with pytest.raises(ValueError, match="non-finite"):
        r1.load_r1(tmp_path / "r", "9b", world, units)
    with pytest.raises(ValueError, match="another world"):
        r1.load_r1(tmp_path / "r", "9b", world, make_units("w2")[:2])


# --- The report -----------------------------------------------------------------------------------


@pytest.fixture
def two_worlds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], Path]:
    worlds = [make_world(tmp_path, "w1"), make_world(tmp_path, "w2")]
    by_name = {w.name: make_units(w.name) for w in worlds}
    monkeypatch.setattr(
        r0, "load_world", lambda w, scales=("standard",): (0.75, list(by_name[Path(w).name]), 0)
    )
    rng = np.random.default_rng(3)
    for w in worlds:
        units = by_name[w.name]
        positive = np.array([u.positive for u in units])
        # abusive units sit at 7-9, their twins at 1-3 (and a few ties): separable
        expected = np.where(positive, 8.0, 2.0) + np.round(rng.random(len(units)), 1)
        write_r1(tmp_path / "r", "9b", w, units, expected)
    return worlds, tmp_path / "r"


def test_the_report_adds_one_r1_rung_with_the_same_scores_in_both_regimes(
    two_worlds: tuple[list[Path], Path], tmp_path: Path
) -> None:
    worlds, r1_dir = two_worlds
    out = tmp_path / "out"
    report = r1.run(worlds, r1_dir, "9b", out)
    assert json.loads((out / "r1.json").read_text()) == report
    assert report["s"] == 0.75 and report["worlds"] == ["w1", "w2"] and report["n_units"] == 240
    assert report["n_distinct_scores"] <= 22  # ties are real and are reported
    for regime in r0.REGIMES:
        assert report["regimes"][regime]["r1_9b"]["primary"]["auc"] == 1.0
        diffs = report["differences"][regime]
        assert {"r1_9b_minus_f64_logistic", "r1_9b_minus_f64_trees"} <= set(diffs)
        gap = diffs["r1_9b_minus_f64_trees"]
        trees = report["regimes"][regime]["f64_trees"]["primary"]["auc"]
        assert gap["point"] == pytest.approx(1.0 - trees)
        assert (gap["n_pos"], gap["n_neg"]) == (80, 80)
    assert report["regimes"]["held_out"]["r1_9b"] == report["regimes"]["in_family"]["r1_9b"]
    assert set(report["runs"]) == {"w1", "w2"}
    assert report["runs"]["w1"]["head_dtype"] == "float32"
    rows = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 240
    assert all(r["scores"]["r1_9b"]["held_out"] == r["scores"]["r1_9b"]["in_family"] for r in rows)
    with pytest.raises(FileExistsError):
        r1.run(worlds, r1_dir, "9b", out)
    with pytest.raises(ValueError, match="unknown model"):
        r1.run(worlds, r1_dir, "70b", tmp_path / "other")


def test_a_score_unrelated_to_the_label_is_at_chance(
    two_worlds: tuple[list[Path], Path], tmp_path: Path
) -> None:
    worlds, r1_dir = two_worlds
    rng = np.random.default_rng(11)
    for w in worlds:
        units = make_units(w.name)
        write_r1(r1_dir, "9b", w, units, 1 + 9 * rng.random(len(units)))
    report = r1.run(worlds, r1_dir, "9b", tmp_path / "out")
    primary = report["regimes"]["held_out"]["r1_9b"]["primary"]
    assert primary["auc"] == pytest.approx(0.5, abs=0.1)
    assert primary["lo"] < 0.5 < primary["hi"]
    assert r1.main(["--worlds", *map(str, worlds), "--r1", str(r1_dir), "--model", "9b",
                    "--out", str(tmp_path / "cli")]) == 0  # fmt: skip
    assert (tmp_path / "cli" / "r1.json").exists()


# --- Nimble (`--model nimble`) --------------------------------------------------------------------


def nimble_rows_sha256_of(world: Path, limit: int | None = None) -> str:
    """The hash of the rows text `nimble_run` would write for `world`, built here by hand."""
    out = ""
    for line in (world / "renders.jsonl").read_text().splitlines()[:limit]:
        r = json.loads(line)
        out += json.dumps({"id": r["unit"], "context": r["text"]}) + "\n"
    return hashlib.sha256(out.encode()).hexdigest()


def nimble_meta(expected: NDArray[np.float64], world: Path, **override: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "nimble_schema_sha256": prompt.NIMBLE_SCHEMA_SHA256,
        "world_checksum_sha256": "sum-1",
        "adapter_revision": nimble_child.ADAPTER_REVISION,
        "adapter_sha256": nimble_child.ADAPTER_SHA256,
        "nimble_commit": nimble_child.NIMBLE_COMMIT,
        "temperature": nimble_child.TEMPERATURE,
        "mode": nimble_child.MODE,
        "child_sha256": nimble_run.child_sha256(),
        "rows_sha256": override.get("rows_sha256")
        or nimble_rows_sha256_of(world, override.get("limit")),
        "limit": None,
        "n_distinct_expected": len(set(expected.tolist())),
        "n_prompt": {"max": 900, "mean": 900.0},
    }
    return meta | override


def write_nimble(
    r1_dir: Path,
    world: Path,
    units: Sequence[r0.R0Unit],
    expected: NDArray[np.float64],
    *,
    reverse: bool = False,
    extra: Sequence[str] = (),
    **override: Any,
) -> None:
    order = list(range(len(units)))[::-1] if reverse else list(range(len(units)))
    rows = [{"id": units[i].unit, "expected": float(expected[i])} for i in order]
    rows += [{"id": e, "expected": 99.0} for e in extra]
    dest = r1_dir / "nimble"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{world.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (dest / f"{world.name}.json").write_text(json.dumps(nimble_meta(expected, world, **override)))


def test_the_rung_is_named_for_the_model_and_the_cli_knows_nimble() -> None:
    assert r1.rung_name("nimble") == "r1_nimble" and "nimble" in r1.MODELS
    assert r1.MODELS[:2] == ("9b", "4b")  # the local-model path is unchanged


def test_load_r1_nimble_aligns_by_unit_id_and_pools_worlds(tmp_path: Path) -> None:
    w1, w2 = make_world(tmp_path, "w1"), make_world(tmp_path, "w2")
    u1, u2 = make_units("w1"), make_units("w2")
    e1, e2 = np.linspace(1, 10, len(u1)), np.linspace(10, 1, len(u2))
    write_nimble(tmp_path / "r", w1, u1, e1, reverse=True, extra=["mixed-cluster"])
    write_nimble(tmp_path / "r", w2, u2, e2)
    np.testing.assert_array_equal(r1.load_r1(tmp_path / "r", "nimble", w1, u1), e1)
    pooled = (u1 + u2)[::-1]
    by_unit = dict(zip([u.unit for u in u1 + u2], np.concatenate([e1, e2]), strict=True))
    got = r1.pooled_r1(tmp_path / "r", "nimble", [w1, w2], pooled)
    np.testing.assert_array_equal(got, [by_unit[u.unit] for u in pooled])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"nimble_schema_sha256": "0" * 64}, "different schema"),
        ({"world_checksum_sha256": "other"}, "different world"),
        ({"adapter_revision": "0" * 40}, "different adapter revision"),
        ({"adapter_sha256": "0" * 64}, "adapter_sha256"),
        ({"nimble_commit": "0" * 40}, "nimble_commit"),
        ({"temperature": 0.5}, "temperature"),
        ({"mode": "independent"}, "mode"),
        ({"child_sha256": "0" * 64}, "different nimble_child.py"),
        ({"child_sha256": None}, "no child_sha256"),
        ({"child_sha256": ""}, "no child_sha256"),
    ],
)
def test_load_r1_nimble_refuses_each_provenance_mismatch(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_nimble(tmp_path / "r", world, units, np.ones(len(units)), **override)
    with pytest.raises(ValueError, match=message):
        r1.load_r1(tmp_path / "r", "nimble", world, units)


def test_load_r1_nimble_refuses_stale_wrong_and_missing_rows_hashes(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    ones = np.ones(len(units))
    write_nimble(tmp_path / "r", world, units, ones)
    r1.load_r1(tmp_path / "r", "nimble", world, units)  # the matching run loads
    write_nimble(tmp_path / "r", world, units, ones, rows_sha256="0" * 64)
    with pytest.raises(ValueError, match="rows_sha256 differs"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    for missing in (None, ""):
        write_nimble(tmp_path / "r", world, units, ones, rows_sha256=missing)
        with pytest.raises(ValueError, match="no rows_sha256"):
            r1.load_r1(tmp_path / "r", "nimble", world, units)
    meta_path = tmp_path / "r" / "nimble" / "w1.json"
    meta = json.loads(meta_path.read_text())
    del meta["rows_sha256"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="no rows_sha256"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    # scored on the renders as they were, which were then edited under the same ids
    write_nimble(tmp_path / "r", world, units, ones)
    write_renders(world, [f"edited {i}" for i in range(3)])
    with pytest.raises(ValueError, match="rows_sha256 differs"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)


def test_load_r1_nimble_checks_the_prefix_of_a_limited_run(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    ones = np.ones(len(units))
    write_nimble(tmp_path / "r", world, units, ones, limit=2)
    r1.load_r1(tmp_path / "r", "nimble", world, units)  # the hash of the first two renders
    write_nimble(
        tmp_path / "r", world, units, ones, limit=2, rows_sha256=nimble_rows_sha256_of(world)
    )
    with pytest.raises(ValueError, match="rows_sha256 differs"):  # the full file's hash is not it
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    write_nimble(tmp_path / "r", world, units, ones, limit=2)
    lines = (world / "renders.jsonl").read_text().splitlines()
    lines[2] = json.dumps({"unit": "r2", "text": "edited beyond the limit"})
    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")
    r1.load_r1(tmp_path / "r", "nimble", world, units)  # renders past the limit do not matter
    lines[1] = json.dumps({"unit": "r1", "text": "edited inside the limit"})
    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="rows_sha256 differs"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)


@pytest.mark.parametrize("limit", [0, -1, "2", True, 1.5])
def test_load_r1_nimble_refuses_a_malformed_limit(tmp_path: Path, limit: Any) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_nimble(tmp_path / "r", world, units, np.ones(len(units)), limit=limit, rows_sha256="x")
    with pytest.raises(ValueError, match="limit"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)


def test_load_r1_nimble_does_not_accept_the_local_models_provenance(tmp_path: Path) -> None:
    """A 9B-shaped run json (prompt hash, model revision, head dtype) is not a Nimble run."""
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_r1(tmp_path / "r", "nimble", world, units, np.ones(len(units)))
    with pytest.raises(ValueError, match="different schema"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    # and a Nimble run json is not a 9B run
    write_nimble(tmp_path / "r", world, units, np.ones(len(units)))
    (tmp_path / "r" / "9b").mkdir()
    for suffix in (".json", ".jsonl"):
        src = tmp_path / "r" / "nimble" / f"w1{suffix}"
        (tmp_path / "r" / "9b" / f"w1{suffix}").write_text(src.read_text())
    with pytest.raises(ValueError, match="different prompt"):
        r1.load_r1(tmp_path / "r", "9b", world, units)


def test_load_r1_nimble_refuses_missing_duplicate_and_non_finite_scores(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    ones = np.ones(len(units))
    write_nimble(tmp_path / "r", world, units[:-1], ones[:-1])
    with pytest.raises(ValueError, match="1 units have no row"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    write_nimble(tmp_path / "r", world, units, ones, extra=[units[2].unit])
    with pytest.raises(ValueError, match="duplicate ids"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    bad = ones.copy()
    bad[4] = np.nan
    write_nimble(tmp_path / "r", world, units, bad)
    with pytest.raises(ValueError, match="non-finite"):
        r1.load_r1(tmp_path / "r", "nimble", world, units)
    with pytest.raises(ValueError, match="another world"):
        r1.load_r1(tmp_path / "r", "nimble", world, make_units("w2")[:2])


def test_the_nimble_report_adds_the_r1_nimble_rung(
    two_worlds: tuple[list[Path], Path], tmp_path: Path
) -> None:
    worlds, r1_dir = two_worlds
    for w in worlds:
        units = make_units(w.name)
        positive = np.array([u.positive for u in units])
        shift = 0.5 * (w.name == "w2")  # distinct scores across the two worlds
        expected = np.where(positive, 8.0, 2.0) + shift + np.arange(len(units)) * 1e-3
        write_nimble(r1_dir, w, units, expected)
    out = tmp_path / "out"
    report = r1.run(worlds, r1_dir, "nimble", out)
    assert json.loads((out / "r1.json").read_text()) == report
    assert report["model"] == "nimble" and report["n_units"] == 240
    for regime in r0.REGIMES:
        assert report["regimes"][regime]["r1_nimble"]["primary"]["auc"] == 1.0
        assert {"r1_nimble_minus_f64_logistic", "r1_nimble_minus_f64_trees"} <= set(
            report["differences"][regime]
        )
    assert report["n_distinct_scores"] == 240
    run = report["runs"]["w1"]
    assert run["nimble_schema_sha256"] == prompt.NIMBLE_SCHEMA_SHA256
    assert run["adapter_revision"] == nimble_child.ADAPTER_REVISION
    assert run["world_checksum_sha256"] == "sum-1" and run["limit"] is None
    assert "prompt_sha256" not in run  # the 9B path's fields are not claimed for Nimble
    rows = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines()]
    assert all("r1_nimble" in r["scores"] for r in rows) and "r1_9b" not in rows[0]["scores"]
    assert r1.main(["--worlds", *map(str, worlds), "--r1", str(r1_dir), "--model", "nimble",
                    "--out", str(tmp_path / "cli")]) == 0  # fmt: skip
    # the 9B report from the same worlds (the fixture wrote their 9B runs) keeps its own fields
    nine = r1.run(worlds, r1_dir, "9b", tmp_path / "nine")
    assert set(nine["runs"]["w1"]) == {
        "prompt_sha256",
        "model_revision",
        "head_dtype",
        "agreement",
        "world_checksum_sha256",
        "limit",
    }
