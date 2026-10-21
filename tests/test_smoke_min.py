"""M0 end-to-end smoke test on the vendored `min` preset (SPEC §10).

generate → persist to temporary stores → verify the pair → detect → units → render one unit.
The memoised detector must match the plain one byte for byte.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from app.services.generator import (
    WORLD_END,
    GeneratedWorld,
    build_manifest,
    canonical_bytes,
    generate_world,
)

from testbed.eval.stores import verify_pair
from testbed.pipeline.detect import Detection, detect
from testbed.pipeline.render_vendored import render_unit
from testbed.pipeline.units import Unit, load_units
from testbed.world.persist import Stores, persist

SEED, PRESET = 1, "min"


@dataclass(frozen=True)
class Run:
    world: GeneratedWorld
    stores: Stores
    detection: Detection
    units: list[Unit]


def _persisted(directory: Path) -> tuple[GeneratedWorld, Stores]:
    world = generate_world(SEED, PRESET)
    stores = Stores.under(directory)
    persist(world, build_manifest(world, SEED, PRESET), stores)
    return world, stores


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    world, stores = _persisted(tmp_path_factory.mktemp("min-memo"))
    detection = detect(stores.inference_url, run_id="smoke-memo")
    return Run(world, stores, detection, load_units(stores.inference_db, WORLD_END))


def test_generation_is_deterministic() -> None:
    assert canonical_bytes(generate_world(SEED, PRESET)) == canonical_bytes(
        generate_world(SEED, PRESET)
    )


def test_stores_are_a_verified_pair(run: Run) -> None:
    verified = verify_pair(run.stores)
    assert verified.world_id


def test_persist_refuses_existing_stores(run: Run) -> None:
    with pytest.raises(FileExistsError):
        persist(run.world, build_manifest(run.world, SEED, PRESET), run.stores)


def test_one_unit_per_detected_cluster(run: Run) -> None:
    assert run.detection.clusters > 0
    assert len(run.units) == run.detection.clusters
    assert sum(len(u.payloads) for u in run.units) == run.detection.cluster_cycles
    members = [m for u in run.units for m in u.member_account_ids]
    assert len(members) == len(set(members)), "an account sits in two units"


def test_memoised_detection_matches_plain(run: Run, tmp_path: Path) -> None:
    _, stores = _persisted(tmp_path)
    plain = detect(stores.inference_url, run_id="smoke-memo", memoize=False)
    assert plain.detection_sha256 == run.detection.detection_sha256


def test_render_one_unit(run: Run) -> None:
    unit = max(run.units, key=lambda u: (len(u.member_account_ids), u.snapshot_id))
    with closing(sqlite3.connect(f"file:{run.stores.inference_db}?mode=ro", uri=True)) as conn:
        text = render_unit(conn, unit.latest, render="v3")
    assert text.startswith("Credit cycle: ")
    assert f"Accounts in cluster: {len(unit.member_account_ids)}." in text
    assert "History, one line per day" in text
    assert "acct-" not in text, "the render leaks raw account ids"
    families = {label.scenario_family for label in run.world.labels}
    assert not any(f in text for f in families), "the render names a scenario family"
