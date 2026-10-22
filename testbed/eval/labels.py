"""The only ground-truth reader (SPEC §9). Coordinator-owned; never delegated (AGENTS.md).

It verifies the store pair before any label read, and it refuses a confirmation world
(`runs/confirm-*`, SPEC §8) unless `prereg/M6.md` exists and every hash in its `seal` block matches
the file on disk. Every ground-truth opener calls `guard_stores` first: `open_labels` here and
`stores.verify_pair` (a boundary test enforces it). Labels leave this package only as arguments to
label-free code.

The seal block in `prereg/M6.md` is a fenced code block tagged `seal`, one `<sha256>  <path>` line
per frozen artifact, paths relative to the repo root (the `shasum -a 256` output format).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.db.engine import make_inference_engine
from app.db.ground_truth_reader import load_labels, load_scenario_families, load_split_groups
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from sqlalchemy.orm import Session

from testbed.world.persist import Stores

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRM_PREFIX = "confirm-"
_SEAL_BLOCK = re.compile(r"^```seal\n(.*?)^```", re.MULTILINE | re.DOTALL)
_SEAL_LINE = re.compile(r"^([0-9a-f]{64}) {2}(\S.*)$")


class SealError(PermissionError):
    """A confirmation world's labels were requested without a valid M6 seal."""


@dataclass(frozen=True)
class WorldLabels:
    world_id: str
    group_of: dict[str, str]  # account id -> group id
    label: dict[str, str]  # group id -> "abusive" | "legitimate"
    family: dict[str, str]  # group id -> scenario family


@dataclass(frozen=True)
class UnitAssignment:
    group_of_unit: dict[str, str]  # unit key -> its single group
    mixed: tuple[str, ...]  # unit keys whose members span groups (excluded, SPEC §6.1)


def is_confirmation_store(path: Path) -> bool:
    """True if any directory on the resolved path is a `confirm-*` world."""
    return any(part.startswith(CONFIRM_PREFIX) for part in path.resolve().parts)


def check_seal(prereg: Path, repo_root: Path) -> None:
    """Raise `SealError` unless `prereg` exists, has a non-empty seal block, and every listed
    file exists under `repo_root` with the recorded sha256."""
    if not prereg.is_file():
        raise SealError(f"{prereg} does not exist: confirmation labels stay sealed")
    block = _SEAL_BLOCK.search(prereg.read_text())
    lines = [ln for ln in (block.group(1).splitlines() if block else []) if ln.strip()]
    if not lines:
        raise SealError(f"{prereg} has no seal block")
    for line in lines:
        match = _SEAL_LINE.match(line)
        if match is None:
            raise SealError(f"malformed seal line: {line!r}")
        digest, rel = match.groups()
        target = (repo_root / rel).resolve()
        if not target.is_relative_to(repo_root.resolve()) or not target.is_file():
            raise SealError(f"sealed artifact missing: {rel}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise SealError(f"sealed artifact changed since the M6 freeze: {rel}")


def guard_stores(stores: Stores, *, repo_root: Path = REPO_ROOT) -> None:
    """Run before opening a ground-truth store: check the seal for a confirmation world, and
    refuse a missing store."""
    if is_confirmation_store(stores.ground_truth_db) or is_confirmation_store(stores.inference_db):
        check_seal(repo_root / "prereg" / "M6.md", repo_root)
    for path in (stores.inference_db, stores.ground_truth_db):
        if not path.is_file():  # never let the vendored engine create an empty store
            raise FileNotFoundError(path)


def open_labels(stores: Stores, *, repo_root: Path = REPO_ROOT) -> WorldLabels:
    """Guard the stores, verify the pair, then read every group's label and family and the
    account-to-group map."""
    guard_stores(stores, repo_root=repo_root)
    inf_engine = make_inference_engine(stores.inference_url)
    gt_engine = make_ground_truth_engine(stores.ground_truth_url)
    try:
        with (
            Session(inf_engine, future=True) as inf,
            Session(gt_engine, future=True) as gt,
        ):
            verified = verify_world_pair(inf, gt)
            group_of = load_split_groups(gt)
            groups = set(group_of.values())
            labels = {g: str(v.value) for g, v in load_labels(gt, groups).items()}
            families = load_scenario_families(gt, groups)
    finally:
        inf_engine.dispose()
        gt_engine.dispose()
    if set(labels) != groups or set(families) != groups:
        raise ValueError("labels and split groups disagree")
    return WorldLabels(verified.world_id, group_of, labels, families)


def assign_units(
    units: Iterable[tuple[str, Sequence[str]]], group_of: dict[str, str]
) -> UnitAssignment:
    """Map each (unit key, member ids) to its group; a unit spanning groups is mixed."""
    assigned: dict[str, str] = {}
    mixed: list[str] = []
    for key, members in units:
        groups = {group_of[m] for m in members}
        if len(groups) == 1:
            assigned[key] = groups.pop()
        else:
            mixed.append(key)
    return UnitAssignment(assigned, tuple(mixed))
