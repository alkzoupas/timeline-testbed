"""Units: one per detector cluster, built from its detection-evidence payloads (SPEC §6.1).

Label-free: reads only the inference store's `detection_evidence` table.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Unit:
    snapshot_id: str
    member_account_ids: tuple[str, ...]  # sorted
    detector_score: float  # max risk_score over the cluster's cycle units
    latest: dict[str, Any]  # the payload with the latest window_start
    payloads: tuple[dict[str, Any], ...]  # every cycle payload, ordered by window_start


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def units_from_payloads(payloads: Iterable[dict[str, Any]], as_of: datetime) -> list[Unit]:
    """Group cycle payloads into one unit per cluster snapshot, sorted by snapshot id.

    Only payloads computed at `as_of` count. A cluster whose payloads disagree on membership
    is an error, never silently merged."""
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        if _ts(payload["as_of"]) == as_of:
            by_snapshot[payload["cluster_snapshot_id"]].append(payload)
    units = []
    for snapshot_id in sorted(by_snapshot):
        group = sorted(
            by_snapshot[snapshot_id], key=lambda p: (_ts(p["window_start"]), p["cycle_id"])
        )
        members = {tuple(sorted(p["member_account_ids"])) for p in group}
        if len(members) != 1:
            raise ValueError(f"cluster {snapshot_id}: payloads disagree on membership")
        units.append(
            Unit(
                snapshot_id=snapshot_id,
                member_account_ids=members.pop(),
                detector_score=max(float(p["risk_score"]) for p in group),
                latest=group[-1],
                payloads=tuple(group),
            )
        )
    return units


def load_units(inference_db: Path, as_of: datetime) -> list[Unit]:
    """Units from a detected inference store, opened read-only."""
    with closing(sqlite3.connect(f"file:{inference_db.resolve()}?mode=ro", uri=True)) as conn:
        rows = conn.execute("select payload_json from detection_evidence").fetchall()
    return units_from_payloads((json.loads(raw) for (raw,) in rows), as_of)
