"""The six oracle statistics (SPEC §5.2): each family's mechanism, measured on one timeline.

Label-free: a statistic reads only a `UnitTimeline`, so it can be recomputed from the render's
parse (SPEC §5.6, render visibility) or from the store alike. `oracle(pair, tl)` returns a score
where higher means more abusive (statistics where lower is more abusive are negated), or None when
the statistic is undefined for the timeline. The definitions follow `testbed/world/DESIGN.md`.

Shared rules: only `CONSUME_CREDITS` events define sessions and amounts. A session is the consume
events of one group in one (UTC date, clock hour); pair 3 uses one account's events instead. Events
are taken in time order, with ties in timeline order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from itertools import combinations
from statistics import mean, median

from testbed.pipeline.timeline import Account, Event, UnitTimeline, alias_key

PAIRS = (1, 2, 3, 4, 5, 6)

CONSUME = "CONSUME_CREDITS"
CLAIM = "CLAIM_CREDITS"
CYCLE = timedelta(days=30)
CYCLE_CAP = 25  # credits granted per cycle; consuming this many exhausts it
EXHAUSTION_WINDOW = timedelta(hours=24)  # pair 5: how soon after an exhaustion a newcomer starts

Session = tuple[date, int]  # (UTC date, clock hour)


def oracle(pair: int, tl: UnitTimeline) -> float | None:
    """The pair's oracle statistic as a higher-is-more-abusive score, or None if undefined."""
    try:
        statistic = _STATISTICS[pair]
    except KeyError:
        raise ValueError(f"unknown pair {pair!r}; expected one of {PAIRS}") from None
    return statistic(tl)


def _consumes(tl: UnitTimeline) -> list[Event]:
    """Consume events in time order (the sort is stable, so ties keep timeline order)."""
    return sorted((e for e in tl.events if e.event_type == CONSUME), key=lambda e: e.at)


def _session_key(at: datetime) -> Session:
    utc = at.astimezone(UTC)
    return utc.date(), utc.hour


def _sessions(events: Iterable[Event]) -> dict[Session, list[Event]]:
    """Time-ordered events grouped by (UTC date, clock hour)."""
    sessions: dict[Session, list[Event]] = defaultdict(list)
    for event in events:
        sessions[_session_key(event.at)].append(event)
    return sessions


def _by_alias(events: Iterable[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.alias].append(event)
    return grouped


def _creation_order(accounts: Sequence[Account]) -> list[Account]:
    return sorted(accounts, key=lambda a: (a.created_at, alias_key(a.alias)))


def _exhaustions(events: Iterable[Event], created_at: datetime) -> dict[int, datetime]:
    """Cycle index -> the consume event at which the account's cumulative consumption in that
    cycle first reaches 25. Cycles anchor at `created_at`; `events` are one account's consumes."""
    used: dict[int, int] = defaultdict(int)
    exhausted: dict[int, datetime] = {}
    for event in events:
        cycle = (event.at - created_at) // CYCLE
        if cycle < 0 or cycle in exhausted:
            continue
        used[cycle] += event.amount or 0
        if used[cycle] >= CYCLE_CAP:
            exhausted[cycle] = event.at
    return exhausted


def _pair1(tl: UnitTimeline) -> float | None:
    """Median over consecutive accounts (by creation, ties by alias) of the absolute gap in hours
    between j's exhaustion in its first cycle and j+1's first claim, skipping any j without an
    exhaustion or a next claim. Lower is more abusive, so the score is its negation."""
    consumes = _by_alias(_consumes(tl))
    first_claim: dict[str, datetime] = {}
    for event in tl.events:
        if event.event_type == CLAIM:
            first_claim[event.alias] = min(event.at, first_claim.get(event.alias, event.at))
    ordered = _creation_order(tl.accounts)
    gaps = []
    for current, following in zip(ordered, ordered[1:], strict=False):
        exhaustion = _exhaustions(consumes.get(current.alias, ()), current.created_at).get(0)
        claim = first_claim.get(following.alias)
        if exhaustion is not None and claim is not None:
            gaps.append(abs((claim - exhaustion).total_seconds()) / 3600)
    return -float(median(gaps)) if gaps else None


def _pair2(tl: UnitTimeline) -> float | None:
    """Start-order concordance: over account pairs co-attending at least 2 sessions,
    sum |#a first - #b first| / sum n_ab. An account's start is its earliest consume event in the
    session; equal starts count as co-attended with neither first. Higher is more abusive."""
    firsts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # a first, b first, n
    for events in _sessions(_consumes(tl)).values():
        starts: dict[str, datetime] = {}
        for event in events:
            starts.setdefault(event.alias, event.at)
        for a, b in combinations(sorted(starts, key=alias_key), 2):
            tally = firsts[(a, b)]
            tally[2] += 1
            if starts[a] < starts[b]:
                tally[0] += 1
            elif starts[b] < starts[a]:
                tally[1] += 1
    kept = [tally for tally in firsts.values() if tally[2] >= 2]
    total = sum(n for _, _, n in kept)
    if total == 0:
        return None
    return sum(abs(a_first - b_first) for a_first, b_first, _ in kept) / total


def _pair3(tl: UnitTimeline) -> float | None:
    """Template similarity: among each account's sessions with exactly 2 consume events (an
    account's events in one date and hour), the fraction of cross-account session pairs whose
    (amount 1, amount 2, gap in seconds) are identical. Higher is more abusive."""
    keys: dict[tuple[int, int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_account: dict[str, int] = defaultdict(int)
    for alias, events in _by_alias(_consumes(tl)).items():
        for session in _sessions(events).values():
            if len(session) != 2:
                continue
            first, second = session
            key = (
                first.amount or 0,
                second.amount or 0,
                int((second.at - first.at).total_seconds()),
            )
            keys[key][alias] += 1
            per_account[alias] += 1
    total = sum(per_account.values())
    pairs = (total * total - sum(n * n for n in per_account.values())) // 2
    if pairs == 0:
        return None
    matches = 0
    for by_account in keys.values():
        size = sum(by_account.values())
        matches += (size * size - sum(n * n for n in by_account.values())) // 2
    return matches / pairs


def _pair4(tl: UnitTimeline) -> float | None:
    """Budget coupling: x is an account's amount per consume event on a day, z is x minus the
    account's mean x over its active days. Over days with at least 2 active accounts, the
    statistic is sum_d (sum_a z)^2 / sum_d sum_a z^2. Lower is more abusive, so the score is its
    negation; None when no day has 2 active accounts or every z is 0."""
    daily: dict[str, dict[date, list[int]]] = defaultdict(lambda: defaultdict(list))
    for event in _consumes(tl):
        daily[event.alias][event.at.astimezone(UTC).date()].append(event.amount or 0)
    x = {
        alias: {day: sum(amounts) / len(amounts) for day, amounts in days.items()}
        for alias, days in daily.items()
    }
    deviations: dict[date, list[float]] = defaultdict(list)
    for days in x.values():
        centre = mean(days.values())
        for day, value in days.items():
            deviations[day].append(value - centre)
    shared = [z for z in deviations.values() if len(z) >= 2]
    denominator = sum(v * v for z in shared for v in z)
    if denominator == 0:
        return None
    return 0.0 - sum(sum(z) ** 2 for z in shared) / denominator


def _pair5(tl: UnitTimeline) -> float | None:
    """Copy of the main (earliest-created) account by each newcomer: the newcomer's mean over its
    sessions of whether the main account also has a consume session in the same date and hour,
    plus 1 if its creation is 0-24 h after a main exhaustion; then the mean over newcomers. A
    newcomer without consume sessions is skipped. Higher is more abusive."""
    ordered = _creation_order(tl.accounts)
    if len(ordered) < 2:
        return None
    main, newcomers = ordered[0], ordered[1:]
    consumes = _by_alias(_consumes(tl))
    main_sessions = set(_sessions(consumes.get(main.alias, ())))
    main_exhaustions = list(_exhaustions(consumes.get(main.alias, ()), main.created_at).values())
    scores = []
    for newcomer in newcomers:
        sessions = list(_sessions(consumes.get(newcomer.alias, ())))
        if not sessions:
            continue
        overlap = sum(session in main_sessions for session in sessions) / len(sessions)
        follows = any(
            timedelta(0) <= newcomer.created_at - exhaustion <= EXHAUSTION_WINDOW
            for exhaustion in main_exhaustions
        )
        scores.append(overlap + (1 if follows else 0))
    return float(mean(scores)) if scores else None


def _pair6(tl: UnitTimeline) -> float | None:
    """Mean over shared sessions (at least 2 accounts with consume events in one date and hour)
    of the account switches among the session's consume events, in time order, minus (accounts - 1).
    Lower is more abusive, so the score is its negation."""
    excess = []
    for events in _sessions(_consumes(tl)).values():
        order = [event.alias for event in events]
        accounts = set(order)
        if len(accounts) < 2:
            continue
        switches = sum(a != b for a, b in zip(order, order[1:], strict=False))
        excess.append(switches - (len(accounts) - 1))
    return 0.0 - float(mean(excess)) if excess else None


_STATISTICS: dict[int, Callable[[UnitTimeline], float | None]] = {
    1: _pair1,
    2: _pair2,
    3: _pair3,
    4: _pair4,
    5: _pair5,
    6: _pair6,
}
