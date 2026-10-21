"""Known-answer tests for unit construction (SPEC §6.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from testbed.pipeline.units import units_from_payloads

AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


def _payload(
    snapshot: str, start: str, risk: float, members: list[str], as_of: datetime = AS_OF
) -> dict[str, Any]:
    return {
        "cluster_snapshot_id": snapshot,
        "cycle_id": f"{snapshot}::cyc-{start}",
        "as_of": as_of.isoformat(),
        "window_start": f"{start}T00:00:00+00:00",
        "risk_score": risk,
        "member_account_ids": members,
    }


def test_one_unit_per_cluster_with_max_score_and_latest_payload() -> None:
    payloads = [
        _payload("snap-b", "2026-04-02", 0.20, ["acct-2", "acct-1"]),
        _payload("snap-b", "2026-05-02", 0.70, ["acct-1", "acct-2"]),
        _payload("snap-b", "2026-03-03", 0.40, ["acct-1", "acct-2"]),
        _payload("snap-a", "2026-05-10", 0.10, ["acct-9"]),
        # A different detector time never joins the unit.
        _payload(
            "snap-b", "2026-05-20", 0.99, ["acct-1", "acct-2"], datetime(2026, 5, 1, tzinfo=UTC)
        ),
    ]
    units = units_from_payloads(payloads, AS_OF)

    assert [u.snapshot_id for u in units] == ["snap-a", "snap-b"]
    b = units[1]
    assert b.member_account_ids == ("acct-1", "acct-2")
    assert b.detector_score == 0.70
    assert b.latest["window_start"].startswith("2026-05-02")
    starts = [p["window_start"][:10] for p in b.payloads]
    assert starts == ["2026-03-03", "2026-04-02", "2026-05-02"]
    assert units[0].detector_score == 0.10


def test_membership_disagreement_is_an_error() -> None:
    payloads = [
        _payload("snap-a", "2026-04-02", 0.2, ["acct-1", "acct-2"]),
        _payload("snap-a", "2026-05-02", 0.2, ["acct-1"]),
    ]
    with pytest.raises(ValueError, match="disagree on membership"):
        units_from_payloads(payloads, AS_OF)
