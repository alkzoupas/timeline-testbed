"""Write a generated world to a fresh inference store and a separate ground-truth store.

This is the wiring layer, as in the vendored `scripts/seed.py`. It creates the ground-truth
engine only to back the vendored write-only sink and the world receipts; it never reads ground
truth. Reading labels happens only in `testbed/eval/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.schema import create_all_schemas
from app.db.world_provenance import record_world_receipts
from app.services.generator import GeneratedWorld, persist_world
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Stores:
    """One world's two sqlite stores, always addressed by explicit paths.

    The vendored engines fall back to URLs relative to the working directory when given
    None, so nothing here ever passes None."""

    inference_db: Path
    ground_truth_db: Path

    @classmethod
    def under(cls, directory: Path) -> Stores:
        directory = directory.resolve()
        return cls(directory / "inference.db", directory / "ground_truth.db")

    @property
    def inference_url(self) -> str:
        return f"sqlite:///{self.inference_db}"

    @property
    def ground_truth_url(self) -> str:
        return f"sqlite:///{self.ground_truth_db}"


def persist(world: GeneratedWorld, manifest: dict[str, Any], stores: Stores) -> None:
    """Persist `world` into two new stores and record the dev-tier world receipts."""
    for path in (stores.inference_db, stores.ground_truth_db):
        if path.exists():
            raise FileExistsError(f"store already exists: {path}")
    inf_engine = make_inference_engine(stores.inference_url)
    gt_engine = make_ground_truth_engine(stores.ground_truth_url)
    try:
        create_all_schemas(inf_engine, gt_engine)
        with (
            Session(inf_engine, future=True) as inf,
            Session(gt_engine, future=True) as gt,
        ):
            persist_world(world, inf, make_ground_truth_sink(gt))
            record_world_receipts(inf, gt, manifest, tier="dev")
            inf.commit()
            gt.commit()
    finally:
        inf_engine.dispose()
        gt_engine.dispose()
