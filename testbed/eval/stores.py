"""Ground-truth store access for evaluation (SPEC §9). Only `testbed/eval/` imports this."""

from __future__ import annotations

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import VerifiedWorld, verify_world_pair
from sqlalchemy.orm import Session

from testbed.world.persist import Stores


def verify_pair(stores: Stores) -> VerifiedWorld:
    """Prove the two stores are one world's matched pair (the vendored receipt check)."""
    inf_engine = make_inference_engine(stores.inference_url)
    gt_engine = make_ground_truth_engine(stores.ground_truth_url)
    try:
        with (
            Session(inf_engine, future=True) as inf,
            Session(gt_engine, future=True) as gt,
        ):
            return verify_world_pair(inf, gt)
    finally:
        inf_engine.dispose()
        gt_engine.dispose()
