"""Information parity of the F-* values and F64 (SPEC §5.6, §6.2).

A small world (one set of groups from each of the 12 families, s = 1) runs through the pipeline.
For every unit, F12 recomputed from the render text alone must equal the detector's F12 exactly,
and F64 of the parsed render must equal F64 of the store's timeline.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest
from app.services.generator import WORLD_END

from testbed.features.f12 import F_IDS, f12
from testbed.features.f64 import FEATURES, f64
from testbed.pipeline.parity import _world, rescore_from_render
from testbed.pipeline.render import parse, render
from testbed.pipeline.run_world import run_world
from testbed.pipeline.timeline import (
    WINDOW_START,
    Account,
    Event,
    Shared,
    UnitTimeline,
    load_timeline,
)
from testbed.pipeline.units import load_units
from testbed.world.generator import WorldSpec
from testbed.world.persist import Stores

W0 = WINDOW_START


def _no_tokens(texts: Sequence[str]) -> list[int]:
    return [0] * len(texts)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("parity") / "w"
    run_world(WorldSpec(1, 1.0, "small"), out, token_counter=_no_tokens)
    return out


def test_world_covers_all_twelve_families(world: Path) -> None:
    manifest = json.loads((world / "manifest.json").read_text())
    assert len(manifest["families"]) == 12


def test_f12_recomputed_from_the_render_equals_the_detectors(world: Path) -> None:
    rows = [json.loads(line) for line in (world / "units.jsonl").read_text().splitlines()]
    texts = {
        r["unit"]: r["text"]
        for r in map(json.loads, (world / "renders.jsonl").read_text().splitlines())
    }
    assert len(rows) == 120
    mismatches = [
        row["unit"] for row in rows if rescore_from_render(texts[row["unit"]]) != row["f12"]
    ]
    assert mismatches == []


def test_f64_of_the_parsed_render_equals_f64_of_the_store(world: Path) -> None:
    stores = Stores.under(world)
    units = load_units(stores.inference_db, WORLD_END)
    assert len(units) == 120
    with closing(sqlite3.connect(f"file:{stores.inference_db}?mode=ro", uri=True)) as conn:
        for unit in units:
            timeline = load_timeline(conn, unit.member_account_ids)
            values = f12(unit)
            from_render = f64(parse(render(timeline)), values)
            assert from_render == f64(timeline, values)
            assert list(from_render) == list(FEATURES)


def _timeline(accounts: list[Account], shared: list[Shared], events: list[Event]) -> UnitTimeline:
    return UnitTimeline(W0, WORLD_END, tuple(accounts), tuple(shared), tuple(events))


def _pair(shared: list[Shared]) -> UnitTimeline:
    created = W0 + timedelta(days=1, hours=9)
    return _timeline(
        [
            Account("A1", created, "DISPOSABLE", "NONE"),
            Account("A2", created + timedelta(days=1), "DISPOSABLE", "NONE"),
        ],
        shared,
        [
            Event(created, "A1", "SIGNUP", None),
            Event(created + timedelta(hours=1), "A1", "CLAIM_CREDITS", 25),
            Event(created + timedelta(days=1), "A2", "SIGNUP", None),
        ],
    )


def test_rescore_known_answer() -> None:
    """Two disposable accounts sharing one HIGH device: F-DOM-1 counts both, F-DEV-2 fires, and
    F-DEV-1 (which needs a strong shared device) does not."""
    values = rescore_from_render(render(_pair([Shared("STABLE_DEVICE", "HIGH", ("A1", "A2"))])))
    assert list(values) == list(F_IDS)
    assert values["F-DOM-1"] == 2.0
    assert values["F-DEV-2"] == 1.0
    assert values["F-DEV-1"] == 0.0
    assert values["F-PAY-1"] == 0.0


def test_rescore_refuses_a_render_that_is_not_one_unit() -> None:
    with pytest.raises(ValueError, match="expected one unit"):
        rescore_from_render(render(_pair([])))  # two accounts with no shared value


def test_rebuilt_cycles_follow_the_cycle_rule() -> None:
    """Created 45 days before the window: cycles 1..4 overlap it. The last one is partial."""
    created = W0 - timedelta(days=45, hours=-3)
    consume = [
        Event(W0 + timedelta(days=2), "A1", "CONSUME_CREDITS", 4),
        Event(W0 + timedelta(days=3), "A1", "CONSUME_CREDITS", 3),
        Event(W0 + timedelta(days=20), "A1", "CONSUME_CREDITS", 5),
    ]
    tl = _timeline([Account("A1", created, "CONSUMER", "EMAIL")], [], consume)
    cycles = _world(tl).cycles
    assert [c.cycle_id for c in cycles] == [f"acct-p01-c{i:02d}" for i in (1, 2, 3, 4)]
    assert [c.cycle_start for c in cycles] == [
        created + timedelta(days=30 * i) for i in (1, 2, 3, 4)
    ]
    assert [c.credits_granted for c in cycles] == [25] * 4
    assert [c.credits_consumed for c in cycles] == [7, 5, 0, 0]
    assert [c.credits_expired for c in cycles] == [18, 20, 25, 0]
