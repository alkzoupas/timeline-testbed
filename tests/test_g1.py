"""Known-answer tests for gate G1 (`testbed/eval/g1.py`, prereg/M2.md)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from testbed.eval.g1 import (
    G1_TESTS,
    channel_row,
    health_row,
    null_items,
    positive_items,
    sweep_by_s,
)
from testbed.eval.r0 import FITTED, REGIMES, RUNGS, R0Unit
from testbed.world.families import MEASURED_TAGS, PAIRS
from testbed.world.generator import GENERATOR_VERSION


def units(per_class: int = 4) -> list[R0Unit]:
    out = []
    for pair, (family, twin) in sorted(PAIRS.items()):
        for positive, name in ((True, family), (False, twin)):
            for i in range(per_class):
                out.append(
                    R0Unit(
                        "w", f"w:{pair}{name}{i}", name, pair, positive, (0.0,), (0.0,), 0.0, 1.0
                    )
                )
    return sorted(out, key=lambda u: u.group)


def scores(us: list[R0Unit], separate: str | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Uninformative scores (rank = index), except `separate` (a rung) scores the label."""
    base = np.arange(len(us), dtype=float)
    label = np.array([u.positive for u in us], dtype=float)
    return {
        (rung, regime): (label if rung == separate else base.copy())
        for rung in RUNGS
        for regime in REGIMES
    }


def test_null_items_are_the_preregistered_62() -> None:
    us = units()
    items = null_items(us, scores(us))
    assert len(items) == G1_TESTS == 62
    assert sum(k.startswith(("random|", "detector|")) for k in items) == 14
    assert sum(k.split("|")[0] in FITTED for k in items) == 42
    assert sum(k.startswith("oracle|") for k in items) == 6
    assert "oracle|pooled" not in items
    pos, neg = items["f64_trees|held_out|family:4"]
    assert pos.size == neg.size == 4


def test_null_items_refuse_an_unfitted_rung_that_differs_by_regime() -> None:
    us = units()
    s = scores(us)
    s[("detector", "held_out")] = s[("detector", "held_out")] + 1.0
    with pytest.raises(ValueError, match="detector differs"):
        null_items(us, s)


def test_null_items_split_a_family_from_its_own_twin() -> None:
    us = units()
    pos, neg = null_items(us, scores(us, separate="f12_logistic"))[
        "f12_logistic|in_family|family:2"
    ]
    assert set(pos) == {1.0} and set(neg) == {0.0}


def test_positive_items_are_the_six_oracle_families() -> None:
    us = units()
    assert sorted(positive_items(us, scores(us))) == [f"oracle|family:{p}" for p in sorted(PAIRS)]


def write_world(root: Path, name: str, s: float, *, scale: str = "standard") -> Path:
    world = root / name
    world.mkdir()
    manifest = {"s": s, "scale": scale, "generator_version": GENERATOR_VERSION}
    (world / "manifest.json").write_text(json.dumps(manifest))
    (world / "health.json").write_text(json.dumps({"passed": True, "manifest": manifest}))
    return world


def test_sweep_by_s_needs_the_full_grid_twice(tmp_path: Path) -> None:
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    worlds = [write_world(tmp_path, f"s{s}-{i}", s) for s in grid for i in (1, 2)]
    assert {s: len(w) for s, w in sweep_by_s(worlds).items()} == dict.fromkeys(grid, 2)
    with pytest.raises(ValueError, match="sweep must be"):
        sweep_by_s(worlds[:-1])
    with pytest.raises(ValueError, match="not a standard"):
        sweep_by_s([*worlds, write_world(tmp_path, "small", 0.5, scale="small")])


def test_health_row_fails_a_stale_or_failed_report(tmp_path: Path) -> None:
    good = write_world(tmp_path, "good", 0.0)
    stale = write_world(tmp_path, "stale", 0.5)
    (stale / "manifest.json").write_text(json.dumps({"s": 0.5, "changed": True}))
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "health.json").write_text(
        json.dumps({"passed": True, "generator_version": GENERATOR_VERSION})
    )
    row = health_row([good, stale], sample)
    assert not row["passed"] and row["failed"] == ["stale"]
    assert health_row([good], sample)["passed"]
    (sample / "health.json").write_text(json.dumps({"passed": True, "generator_version": "old"}))
    assert health_row([good], sample)["failed"] == ["sample"]


def test_channel_row_needs_the_frozen_tags(tmp_path: Path) -> None:
    path = tmp_path / "channel.json"
    tags = {str(p): t for p, t in MEASURED_TAGS.items()}
    path.write_text(json.dumps({"passed": True, "measured_tags": tags}))
    assert channel_row(path)["passed"]
    path.write_text(json.dumps({"passed": True, "measured_tags": {**tags, "3": "M"}}))
    assert not channel_row(path)["passed"]
