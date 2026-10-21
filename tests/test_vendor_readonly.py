"""vendor/ is read-only (AGENTS.md): every file matches VENDORED.sha256 and nothing was added."""

from __future__ import annotations

import hashlib
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "entitlement-shield"
BOOKKEEPING = {"VENDORED.md", "VENDORED.sha256"}


def _listed() -> dict[str, str]:
    listed = {}
    for line in (VENDOR / "VENDORED.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name.removeprefix("*").removeprefix("./")] = digest
    return listed


def test_vendored_files_match_their_hashes() -> None:
    listed = _listed()
    assert listed, "VENDORED.sha256 is empty"
    changed = [
        name
        for name, digest in listed.items()
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != digest
    ]
    assert changed == []


def test_nothing_was_added_to_vendor() -> None:
    present = {
        str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*") if p.is_file() or p.is_symlink()
    }
    added = sorted(present - set(_listed()) - BOOKKEEPING)
    assert added == [], f"files added under vendor/ (e.g. __pycache__, *.egg-info): {added[:5]}"
