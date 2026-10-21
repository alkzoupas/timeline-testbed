"""The vendored detector's frozen feature manifest regenerates to its recorded hash (SPEC §4)."""

from __future__ import annotations

from app.services.feature_manifest import verified_manifest_hash

FROZEN = "fe21a7b052bea913fcbb880037a85c022d0e2be9d855a426bd449f60172a2021"


def test_vendored_manifest_is_the_frozen_one() -> None:
    assert verified_manifest_hash() == FROZEN
