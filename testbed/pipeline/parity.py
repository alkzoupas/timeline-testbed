"""Information parity of the 12 F-* values (SPEC §5.6, §6.2): recompute F12 from the render text.

`rescore_from_render(text)` parses a render, rebuilds a small world from it, runs the vendored
detector over that world and returns the F12 raw values of the unit with the latest
`window_start`. If these equal the unit's F12 from the full store, the render carries everything
the F-* values read.

Why the render is enough. Every F-* raw value is a function of:

- the in-window events of the unit's accounts;
- the accounts' email domain type, verification level and `created_at`;
- the cycle anchors (the cycles are derived from `created_at`, and so is their consumption);
- the linkage holders and the reliability of each shared value.

The render prints all of these. The generator puts no event before the window start except an
aged account's signup (DESIGN.md), and no F-* reads that outside F-CYC-3's 60-day signup window,
which the rebuilt signup event covers, because the render keeps `created_at` for every account.

Choices the render doesn't dictate:

- Account ids `acct-p01`, `acct-p02`, ... sort in alias order, so the rebuilt unit's aliases
  match the render's.
- Each holder observes each shared value at `max(created_at, window_start)`. The vendored
  clustering drops observations older than 365 days, which a literal `created_at` would do to a
  year-old account (the generator avoids this the same way).
- A value's hash is a fixed hash of its render line and position, so equal lines stay distinct
  values.
- Event ids number each account's events in render order, so ties at one instant keep that order.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from app.models.enums import (
    EmailDomainType,
    EndpointCostTier,
    EventType,
    Reliability,
    SignalType,
    VerificationLevel,
)
from app.services.generator import (
    GenAccount,
    GenCycle,
    GeneratedWorld,
    GenEvent,
    GenLinkage,
)

from testbed.features.f12 import f12
from testbed.pipeline.detect import detect
from testbed.pipeline.render import parse
from testbed.pipeline.timeline import UnitTimeline
from testbed.pipeline.units import load_units
from testbed.world.families import CYCLE_DAYS, GRANT
from testbed.world.persist import Stores, persist


def _account_id(alias: str) -> str:
    return f"acct-p{int(alias[1:]):02d}"


def _value_hash(position: int, signal: str, reliability: str, holders: str) -> str:
    line = f"{position}|{signal}|{reliability}|{holders}"
    return hashlib.sha256(line.encode()).hexdigest()[:32]


def _world(tl: UnitTimeline) -> GeneratedWorld:
    """The timeline as a vendored world, with empty labels, splits and fixtures."""
    accounts = [
        GenAccount(
            _account_id(a.alias),
            a.created_at,
            EmailDomainType[a.email_domain_type],
            VerificationLevel[a.verification_level],
        )
        for a in tl.accounts
    ]

    events: list[GenEvent] = []
    counts: dict[str, int] = defaultdict(int)
    for e in tl.events:
        acct = _account_id(e.alias)
        kind = EventType[e.event_type]
        tier = EndpointCostTier.HIGH if kind is EventType.CONSUME_CREDITS else None
        events.append(GenEvent(f"{acct}-e{counts[acct]:05d}", acct, e.at, kind, e.amount, tier))
        counts[acct] += 1

    linkage: list[GenLinkage] = []
    observed: dict[str, int] = defaultdict(int)
    created = {a.alias: a.created_at for a in tl.accounts}
    for position, s in enumerate(tl.shared):
        value = _value_hash(position, s.signal_type, s.reliability, " ".join(s.aliases))
        for alias in s.aliases:
            acct = _account_id(alias)
            linkage.append(
                GenLinkage(
                    f"{acct}-l{observed[acct]}",
                    acct,
                    max(created[alias], tl.window_start),
                    SignalType[s.signal_type],
                    value,
                    Reliability[s.reliability],
                )
            )
            observed[acct] += 1

    cycles: list[GenCycle] = []
    length = timedelta(days=CYCLE_DAYS)
    for a in tl.accounts:
        acct = _account_id(a.alias)
        index = 0 if a.created_at >= tl.window_start else (tl.window_start - a.created_at) // length
        while (start := a.created_at + length * index) < tl.window_end:
            end = start + length
            consumed = sum(
                e.amount or 0
                for e in tl.events
                if e.alias == a.alias and e.event_type == "CONSUME_CREDITS" and start <= e.at < end
            )
            expired = GRANT - consumed if end <= tl.window_end else 0
            cycles.append(
                GenCycle(f"{acct}-c{index:02d}", acct, start, end, GRANT, consumed, expired)
            )
            index += 1

    return GeneratedWorld(
        accounts=accounts,
        events=events,
        linkage=linkage,
        cycles=cycles,
        labels=[],
        split_groups=[],
        fixtures=[],
    )


def rescore_from_render(text: str) -> dict[str, float]:
    """F12 raw values of the latest-window unit of the world rebuilt from `text`.

    Raises `ValueError` unless the rebuilt world yields exactly one unit."""
    tl = parse(text)
    with tempfile.TemporaryDirectory(prefix="parity-") as tmp:
        stores = Stores.under(Path(tmp))
        persist(_world(tl), {"tier": "dev"}, stores)
        detect(stores.inference_url, run_id="parity", as_of=tl.window_end)
        units = load_units(stores.inference_db, tl.window_end)
    if len(units) != 1:
        raise ValueError(f"expected one unit from the render, got {len(units)}")
    return f12(units[0])
