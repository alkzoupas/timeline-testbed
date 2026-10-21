"""Run the frozen vendored detector over a persisted inference store (SPEC §4)."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import app.services.features as features_mod
from app.db.engine import make_inference_engine
from app.services.detector import detection_bytes, run_detector
from app.services.generator import WORLD_END
from sqlalchemy.orm import Session


@contextmanager
def memoized_signal_values() -> Iterator[None]:
    """Share one global resource-sharing map per `(session, as_of)`.

    Copied from the vendored `scripts/tier4_arms_run.py` (`memoized_signal_values`), where it is
    result-identical by construction: `features._load_ctx` rebuilds the map for every
    cluster-cycle, the map is a pure function of the linkage table and `as_of`, and the
    detector never writes linkage rows. `tests/test_smoke_min.py` pins the equivalence here. The
    vendored module is patched at runtime, never edited."""
    original: Callable[[Session, datetime], Any] = features_mod.shared_signal_values
    cache: dict[tuple[int, datetime], Any] = {}

    def cached(session: Session, as_of: datetime) -> Any:
        key = (id(session), as_of)
        if key not in cache:
            cache[key] = original(session, as_of)
        return cache[key]

    features_mod.shared_signal_values = cached
    try:
        yield
    finally:
        features_mod.shared_signal_values = original  # type: ignore[assignment]


@dataclass(frozen=True)
class Detection:
    run_id: str
    as_of: datetime
    clusters: int
    cluster_cycles: int
    seconds: float
    detection_sha256: str  # sha256 of the vendored canonical `detection_bytes`


def detect(
    inference_url: str, *, run_id: str, as_of: datetime = WORLD_END, memoize: bool = True
) -> Detection:
    """Run `run_detector` once over the store at `inference_url` and commit its outputs."""
    engine = make_inference_engine(inference_url)
    started = time.monotonic()
    try:
        with (
            memoized_signal_values() if memoize else nullcontext(),
            Session(engine, future=True) as inf,
        ):
            result = run_detector(inf, as_of=as_of, run_id=run_id)
            inf.commit()
    finally:
        engine.dispose()
    return Detection(
        run_id=run_id,
        as_of=result.as_of,
        clusters=result.clusters,
        cluster_cycles=result.cluster_cycles,
        seconds=round(time.monotonic() - started, 3),
        detection_sha256=hashlib.sha256(detection_bytes(result.detections)).hexdigest(),
    )
