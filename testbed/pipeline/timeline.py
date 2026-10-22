"""One unit's label-free timeline over the shared window (SPEC §6.2). Coordinator-owned contract.

The render (`render.py`) prints a `UnitTimeline`, and its parser returns one. F64-w90 and the
oracle statistics are functions of a `UnitTimeline`. So when the render round-trips exactly, every
input they read is in the render text (information parity, SPEC §5.6).

Read from the inference store, never from detection payloads. Aliases follow the vendored render:
`A1..An` over the sorted member ids. Ids and signal hashes never leave this module.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.services.generator import WORLD_END

WINDOW_DAYS = 90
WINDOW_START = WORLD_END - timedelta(days=WINDOW_DAYS)  # 2026-03-03T00:00Z, a Tuesday


def parse_ts(value: str) -> datetime:
    """SQLite keeps naive UTC text (`YYYY-MM-DD HH:MM:SS.ffffff`). Return it as aware UTC."""
    parsed = datetime.fromisoformat(value.replace(" ", "T"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _store_ts(value: datetime) -> str:
    """The SQLite text form of an aware UTC instant, for range comparisons in SQL."""
    return f"{value.astimezone(UTC):%Y-%m-%d %H:%M:%S.%f}"


@dataclass(frozen=True)
class Account:
    alias: str  # A1..An
    created_at: datetime  # aware UTC; may precede the window
    email_domain_type: str  # vendored enum name, e.g. "CONSUMER"
    verification_level: str  # e.g. "PHONE"


@dataclass(frozen=True)
class Shared:
    """One linkage value observed on these members up to `window_end`. Edge-forming and
    context-only signals alike; a value held by one member is kept too."""

    signal_type: str  # e.g. "STABLE_DEVICE"
    reliability: str  # e.g. "HIGH"; one reliability per value (the loader refuses mixed)
    aliases: tuple[str, ...]  # sorted by alias_key


@dataclass(frozen=True)
class Event:
    at: datetime  # aware UTC
    alias: str
    event_type: str  # e.g. "CONSUME_CREDITS"
    amount: int | None


@dataclass(frozen=True)
class UnitTimeline:
    window_start: datetime
    window_end: datetime
    accounts: tuple[Account, ...]  # alias order
    shared: tuple[Shared, ...]  # sorted by (signal_type, alias keys, reliability)
    events: tuple[Event, ...]  # window_start <= at < window_end, store order (see load_timeline)


def alias_key(alias: str) -> int:
    """Sort key for aliases: A2 before A10."""
    return int(alias[1:])


def aliases(member_account_ids: Sequence[str]) -> dict[str, str]:
    """Account id -> alias, `A1..An` over the sorted ids (as the vendored render)."""
    return {m: f"A{i + 1}" for i, m in enumerate(sorted(member_account_ids))}


def load_timeline(
    conn: sqlite3.Connection,
    member_account_ids: Sequence[str],
    *,
    window_start: datetime = WINDOW_START,
    window_end: datetime = WORLD_END,
) -> UnitTimeline:
    """The members' accounts, linkage values and in-window events, from an inference store.

    Events are ordered by (timestamp, account id, event id), so ties between accounts at one
    instant have a fixed order. Appeals are excluded (the generator emits none)."""
    alias = aliases(member_account_ids)
    members = sorted(alias)
    marks = ",".join("?" * len(members))
    rows = conn.execute(
        f"select account_id, created_at, email_domain_type, verification_level "
        f"from accounts where account_id in ({marks})",
        members,
    ).fetchall()
    if len(rows) != len(members):
        raise ValueError("a member account is missing from the store")
    by_alias = {
        alias[a]: Account(alias[a], parse_ts(created), domain, level)
        for a, created, domain, level in rows
    }
    accounts = tuple(by_alias[alias[m]] for m in members)

    holders: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for account_id, signal, value, reliability in conn.execute(
        f"select account_id, signal_type, signal_value_hash, reliability "
        f"from linkage_observations where account_id in ({marks}) and timestamp < ?",
        [*members, _store_ts(window_end)],
    ):
        holders[(signal, value)][reliability].add(alias[account_id])
    shared = []
    for (signal, _value), by_rel in holders.items():
        if len(by_rel) != 1:
            raise ValueError(f"one {signal} value carries mixed reliabilities")
        ((reliability, held),) = by_rel.items()
        shared.append(Shared(signal, reliability, tuple(sorted(held, key=alias_key))))
    shared.sort(key=lambda s: (s.signal_type, [alias_key(a) for a in s.aliases], s.reliability))

    events = tuple(
        Event(parse_ts(ts), alias[account_id], event_type, amount)
        for account_id, ts, event_type, amount in conn.execute(
            f"select account_id, timestamp, event_type, credits_amount from account_events "
            f"where account_id in ({marks}) and timestamp >= ? and timestamp < ? "
            f"and event_type != 'APPEAL' order by timestamp, account_id, event_id",
            [*members, _store_ts(window_start), _store_ts(window_end)],
        )
    )
    return UnitTimeline(window_start, window_end, accounts, tuple(shared), events)
