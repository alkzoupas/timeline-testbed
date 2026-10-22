"""The 12 family processes (6 twin pairs) of generator v1 (DESIGN.md "Families", SPEC §5.2).

Each process is a pure function of its RNG streams and its group size `k` (2..6, set by the
generator's balanced design, never drawn here). A twin takes `(base, k)`. An abusive family takes
`(base, mech, s, k)`, draws every twin number from `base` in the twin's order, and uses `mech`
only on top of that: at s = 0 it returns exactly its twin's `Plan`. A mechanism draws from `mech`
whether or not it applies (`u < s`), so worlds at different s share every draw.

A process returns a `Plan`: creation times, chronotype hours, claim delays and the consume events.
The rest of an account (signup, claims, verification, logins, linkage, cycle rows, ids) belongs to
`generator.py`, which reads the claims from the same `cycle_frames` the processes use here.

Local resolutions of DESIGN.md, all reported to the coordinator:
- Pair 4's hours are drawn from 6..23 (the coordinator's resolution: 8..22 can't hold six
  accounts 3 h apart).
- Pair 3's accounts pick distinct hours on a shared weekday, so no two sessions share a
  (day, clock hour).
- Pair 1's rescale is a linear warp of the first-cycle session start instants, anchored at the
  claim, so the last session can land exactly 60..600 min before the next turn. Warped sessions
  leave the chronotype hour and never share a (day, hour).
- T-pair sessions are eligible by clock hour (the whole hour follows the claim and ends inside the
  cycle), so moving events within the hour can't change which sessions exist.
- A pair-5 newcomer that follows the rule keeps the twin's event count per session; where the main
  account has fewer session days left than the twin draws (it stops at the cap), the rest are its
  own sessions at its own hour.
- The pair-5 main account paces its sessions to the days left in the cycle and reaches the 25 cap
  whenever it has about two weeks; a shorter partial cycle may fall short.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from app.services.generator import WORLD_END

# --- Frame constants (DESIGN.md "Frame") ----------------------------------------------------------

WINDOW_DAYS = 90
W0 = WORLD_END - timedelta(days=WINDOW_DAYS)  # 2026-03-03T00:00Z, a Tuesday
W0_WEEKDAY = W0.weekday()  # Monday is 0
CYCLE_DAYS = 30
GRANT = 25
DAY_MIN = 1440
GROUP_SIZE = (2, 6)
CLAIM_DELAY_MIN = (30, 480)  # minutes after the later of the cycle start and W0
CLAIM_SLOTS = 4  # claim delays drawn per account: a 90-day window holds at most four cycles
T_CREATE_DAYS = 10  # T-pair accounts are created in [W0, W0 + 10 d)
CHRONO_HOURS = (8, 22)  # an account's own hour, where a family doesn't set it
CAP_EVENTS = 6  # a T account-cycle holds at most 6 consume events, each of at most 4 credits
MAX_EVENT_CREDITS = 4
SESSION_SPAN_MIN = 56  # a session's start plus its gap always ends inside its clock hour

# --- Family names, pairs and channel tags ---------------------------------------------------------

ROTATOR = "rotator"
HOUSEHOLD = "household_device"
PARALLEL_FARM = "parallel_farm"
CLASSROOM = "classroom_lab"
TEMPLATE_OPERATOR = "template_operator"
OFFICE_CARD = "office_card"
SINGLE_BUDGET = "single_budget"
HEAVY_FAMILY = "heavy_family"
AGED_EXPANSION = "aged_expansion"
ADDS_RELATIVE = "adds_relative"
SERIAL_OPERATOR = "serial_operator"
ROOMMATES = "roommates_card"

PAIRS: dict[int, tuple[str, str]] = {
    1: (ROTATOR, HOUSEHOLD),
    2: (PARALLEL_FARM, CLASSROOM),
    3: (TEMPLATE_OPERATOR, OFFICE_CARD),
    4: (SINGLE_BUDGET, HEAVY_FAMILY),
    5: (AGED_EXPANSION, ADDS_RELATIVE),
    6: (SERIAL_OPERATOR, ROOMMATES),
}
CHANNEL_TAGS: dict[int, str] = {1: "F", 2: "T", 3: "T", 4: "T", 5: "M", 6: "T"}  # as designed
# SPEC §5.4: the tags measured at M1 and frozen. F and M are what the check measured (pair 1: best
# F12 AUC 0.69, M; pair 5: 0.91, F); every T-designed pair measured T. Reports use these.
MEASURED_TAGS: dict[int, str] = {1: "M", 2: "T", 3: "T", 4: "T", 5: "F", 6: "T"}
T_PAIRS = (2, 3, 4, 6)

# --- Pair 1 ---------------------------------------------------------------------------------------

TURN_FIRST_DAYS = 10  # T_1 in W0 + [0, 10) d
TURN_GAP_DAYS = (6, 14)  # T_{j+1} = T_j + [6, 14) d
FIRST_CYCLE_SPAN_DAYS = (4, 20)  # X_j: session days lie within X_j days after the claim
FIRST_CYCLE_SESSIONS = (4, 5)  # drawn count, capped by the span, so 25 fits at <= 8 per session
SESSION_CREDITS = (3, 8)  # first-cycle session totals, summing to 25
LATER_CYCLE_SESSIONS = (0, 2)
HOUSEHOLD_START = (0, 40)  # session start minute
HOUSEHOLD_GAP = (3, 15)  # minutes between a session's two events
EXHAUSTION_LEAD_MIN = (60, 600)  # rescaled exhaustion lands this long before T_{j+1}

# --- Pair 2 ---------------------------------------------------------------------------------------

LAB_HOURS = (9, 16)
LAB_ATTEND_P = 0.6  # (tune) per account per meeting; DESIGN 0.75, cut for the event budget
LAB_OFFSETS = 21  # distinct start offsets from {0..20} min
LAB_GAP = (3, 15)

# --- Pair 3 ---------------------------------------------------------------------------------------

OFFICE_DAY_P = 0.07  # (tune) per account per weekday; DESIGN 0.15, cut for the event budget
OFFICE_HOURS = (9, 17)
OFFICE_START = (0, 25)
OFFICE_GAP = (2, 30)

# --- Pair 4 ---------------------------------------------------------------------------------------

BUDGET_HOURS = (6, 23)  # widened from 8..22: six accounts 3 h apart need a 16-hour span
BUDGET_HOUR_SEP = 3
FAMILY_DAY_P = 0.10  # (tune) per day; DESIGN 0.2. 0.07 gave oracle AUC ~0.94 at s = 1
BUDGET_ACTIVE_P = 0.8
BUDGET_START = (0, 30)
BUDGET_GAP = (2, 25)

# --- Pair 5 ---------------------------------------------------------------------------------------

MAIN_AGE_DAYS = (200, 400)
MAIN_PACE_DAYS = 20  # the main account's daily session probability is min(1, 20 / days left)
NEWCOMER_FROM_DAYS = 5  # newcomer created in [W0 + 5 d, WORLD_END - 20 d)
NEWCOMER_UNTIL_DAYS = 20
NEWCOMER_CYCLE_SESSIONS = (1, 2)  # (tune) DESIGN 1-3, cut for the event budget
EXPANSION_START = (0, 30)
EXPANSION_GAP = (2, 25)
FOLLOW_DELAY_MIN = (60, 1440)  # a rule newcomer is created this long after a main exhaustion

# --- Pair 6 ---------------------------------------------------------------------------------------

SHARED_DAY_P = 0.055  # (tune) per day; DESIGN 0.12, cut for the event budget
SHARED_HOURS = (18, 22)
SHARED_JOIN_P = 0.7
SHARED_MINUTES = 56  # event minutes are distinct, from {0..55}
SHARED_EVENTS = (2, 3)  # a joiner's event count; amounts are 1..4 for 2, 1..2 for 3


def minutes(n: int) -> timedelta:
    return timedelta(minutes=n)


def check_size(k: int) -> None:
    if not GROUP_SIZE[0] <= k <= GROUP_SIZE[1]:
        raise ValueError(f"group size must lie in {GROUP_SIZE}, got {k}")


def midnight(t: datetime) -> datetime:
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


def day_at(day: int, hour: int) -> datetime:
    """The start of clock hour `hour` on window day `day` (day 0 is W0)."""
    return W0 + timedelta(days=day, hours=hour)


def is_weekday(day: int) -> bool:
    return (W0_WEEKDAY + day) % 7 < 5


# --- Cycles and claims ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleFrame:
    """One entitlement cycle of an account that overlaps the window."""

    index: int
    start: datetime
    end: datetime
    claim: datetime | None  # None when the claim would fall at or after the cycle's limit

    @property
    def limit(self) -> datetime:
        return min(self.end, WORLD_END)


def draw_claim_offsets(rng: random.Random) -> tuple[int, ...]:
    """Claim delays (minutes) for the account's in-window cycles, in order. Always CLAIM_SLOTS."""
    return tuple(rng.randint(*CLAIM_DELAY_MIN) for _ in range(CLAIM_SLOTS))


def cycle_frames(created: datetime, offsets: Sequence[int]) -> tuple[CycleFrame, ...]:
    """Cycles anchored at `created`: every cycle that starts before WORLD_END and ends after W0.

    The claim is at the later of the cycle start and W0, plus the account's delay for that cycle.
    A claim at or after the cycle's end or WORLD_END is None: an aged account's cycle that ends
    minutes after W0 would otherwise put its claim in the next cycle."""
    c = 0 if created >= W0 else (W0 - created) // timedelta(days=CYCLE_DAYS)
    frames: list[CycleFrame] = []
    while (start := created + timedelta(days=CYCLE_DAYS * c)) < WORLD_END:
        end = start + timedelta(days=CYCLE_DAYS)
        claim = max(start, W0) + minutes(offsets[len(frames)])
        frames.append(CycleFrame(c, start, end, claim if claim < min(end, WORLD_END) else None))
        c += 1
    return tuple(frames)


def frame_at(frames: Sequence[CycleFrame], t: datetime) -> CycleFrame | None:
    for f in frames:
        if f.start <= t < f.end:
            return f
    return None


# --- The plan a process returns -------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """A group's account facts and consume events; `generator.py` adds everything else."""

    created: tuple[datetime, ...]
    chrono: tuple[int, ...]  # each account's hour, for its logins
    offsets: tuple[tuple[int, ...], ...]  # claim delays
    consume: tuple[tuple[tuple[datetime, int], ...], ...]  # (time, credits), by time

    @property
    def size(self) -> int:
        return len(self.created)


@dataclass(frozen=True)
class _Session:
    """Consume events at `at + offsets[i]` minutes. A session fits inside one clock hour."""

    acct: int
    at: datetime
    offsets: tuple[int, ...]
    amounts: tuple[int, ...]
    tag: int = 0  # the meeting or day the session belongs to

    def times(self) -> list[datetime]:
        return [self.at + minutes(o) for o in self.offsets]

    @property
    def first(self) -> datetime:
        return self.at + minutes(self.offsets[0])


def _admit(
    sessions: Sequence[_Session],
    frames: Sequence[Sequence[CycleFrame]],
    cap: int | None = CAP_EVENTS,
) -> list[_Session]:
    """Keep the sessions whose whole clock hour follows their cycle's claim and ends before the
    cycle does, then drop an account-cycle's latest sessions until it holds at most `cap` events.

    Eligibility is by clock hour (`at` is the hour's start), so a mechanism that moves events
    within the hour can't make a kept session ineligible."""
    by_cycle: dict[tuple[int, int], list[_Session]] = defaultdict(list)
    for sess in sorted(sessions, key=lambda x: (x.first, x.acct)):
        cycle = frame_at(frames[sess.acct], sess.first)
        if cycle is None or cycle.claim is None:
            continue
        if sess.at < cycle.claim or sess.at + timedelta(hours=1) > cycle.limit:
            continue
        by_cycle[(sess.acct, cycle.index)].append(sess)
    kept: list[_Session] = []
    for group in by_cycle.values():
        events = 0
        for sess in group:
            events += len(sess.amounts)
            if cap is not None and events > cap:
                break
            kept.append(sess)
    return sorted(kept, key=lambda x: (x.first, x.acct))


def _consume(sessions: Sequence[_Session], k: int) -> tuple[tuple[tuple[datetime, int], ...], ...]:
    rows: list[list[tuple[datetime, int]]] = [[] for _ in range(k)]
    for sess in sessions:
        rows[sess.acct].extend(zip(sess.times(), sess.amounts, strict=True))
    return tuple(tuple(sorted(r, key=lambda e: e[0])) for r in rows)


def _hour_days(lo: datetime, hi: datetime, hour: int) -> list[datetime]:
    """Midnights of the days on which a session at `hour` fits in [lo, hi)."""
    out: list[datetime] = []
    d = midnight(lo)
    while d + timedelta(hours=hour, minutes=SESSION_SPAN_MIN) < hi:
        if d + timedelta(hours=hour) >= lo:
            out.append(d)
        d += timedelta(days=1)
    return out


def _created_t(rng: random.Random) -> datetime:
    return W0 + minutes(rng.randrange(T_CREATE_DAYS * DAY_MIN))


# --- Pair 1: rotator vs household_device (F) ------------------------------------------------------


def _split_credits(rng: random.Random, n: int) -> list[int]:
    """25 credits in `n` sessions of 3..8 each, drawn so every prefix leaves a feasible rest."""
    lo_c, hi_c = SESSION_CREDITS
    parts: list[int] = []
    left = GRANT
    for i in range(n - 1):
        more = n - i - 1
        part = rng.randint(max(lo_c, left - hi_c * more), min(hi_c, left - lo_c * more))
        parts.append(part)
        left -= part
    parts.append(left)
    return parts


def _household_session(rng: random.Random, at: datetime, total: int) -> _Session:
    """One session of `total` credits (1..8) in one or two events, at its first event's time."""
    start = rng.randint(*HOUSEHOLD_START)
    gap = rng.randint(*HOUSEHOLD_GAP)
    two = total > MAX_EVENT_CREDITS or (total >= 2 and rng.randint(1, 2) == 2)
    if not two:
        return _Session(0, at + minutes(start), (0,), (total,))
    first = rng.randint(max(1, total - MAX_EVENT_CREDITS), min(MAX_EVENT_CREDITS, total - 1))
    return _Session(0, at + minutes(start), (0, gap), (first, total - first))


def _rotation(base: random.Random, mech: random.Random | None, s: float, k: int) -> Plan:
    check_size(k)
    turn = W0 + minutes(base.randrange(TURN_FIRST_DAYS * DAY_MIN))
    created: list[datetime] = []
    for j in range(k):
        if j:
            turn += minutes(base.randrange(TURN_GAP_DAYS[0] * DAY_MIN, TURN_GAP_DAYS[1] * DAY_MIN))
        created.append(turn)

    chrono: list[int] = []
    offsets: list[tuple[int, ...]] = []
    frames: list[tuple[CycleFrame, ...]] = []
    first: list[list[_Session]] = []
    later: list[list[_Session]] = []
    for j in range(k):
        chrono.append(base.randint(*CHRONO_HOURS))
        offsets.append(draw_claim_offsets(base))
        frames.append(cycle_frames(created[j], offsets[j]))
        hour = timedelta(hours=chrono[j])
        claim = frames[j][0].claim
        assert claim is not None  # the first cycle starts at least 10 days before WORLD_END

        span = base.randint(FIRST_CYCLE_SPAN_DAYS[0], FIRST_CYCLE_SPAN_DAYS[1] - 1)
        n = min(base.randint(*FIRST_CYCLE_SESSIONS), span)
        days = sorted(base.sample(range(1, span + 1), n))
        totals = _split_credits(base, n)
        first.append(
            [
                replace(
                    _household_session(base, midnight(claim) + timedelta(days=o) + hour, total),
                    acct=j,
                )
                for o, total in zip(days, totals, strict=True)
            ]
        )

        extra: list[_Session] = []
        for cycle in frames[j][1:]:
            count = base.randint(*LATER_CYCLE_SESSIONS)
            if cycle.claim is None:
                continue
            options = _hour_days(cycle.claim, cycle.limit, chrono[j])
            for d in sorted(base.sample(options, min(count, len(options)))):
                total = base.randint(1, SESSION_CREDITS[1])
                extra.append(replace(_household_session(base, d + hour, total), acct=j))
        later.append(extra)

    if mech is not None:
        for j in range(k - 1):
            u = mech.random()
            lead = mech.randint(*EXHAUSTION_LEAD_MIN)
            if u >= s:
                continue
            claim = frames[j][0].claim
            assert claim is not None
            last = first[j][-1]
            den = (last.at - claim) // minutes(1)
            num = (created[j + 1] - minutes(lead) - minutes(last.offsets[-1]) - claim) // minutes(1)
            first[j] = [
                replace(
                    x, at=claim + minutes(((x.at - claim) // minutes(1) * num + den // 2) // den)
                )
                for x in first[j]
            ]

    sessions = [x for j in range(k) for x in first[j] + later[j]]
    rows: list[list[tuple[datetime, int]]] = [[] for _ in range(k)]
    for x in sessions:
        rows[x.acct].extend(
            (t, a) for t, a in zip(x.times(), x.amounts, strict=True) if t < WORLD_END
        )
    return Plan(
        tuple(created),
        tuple(chrono),
        tuple(offsets),
        tuple(tuple(sorted(r, key=lambda e: e[0])) for r in rows),
    )


def household_device(base: random.Random, k: int) -> Plan:
    """Shared-device household (pair 1 twin, channel F).

    Members take turns on one device with independent start times. Turn starts are
    T_1 in W0 + [0, 10) d and T_{j+1} = T_j + [6, 14) d, and account j is created at T_j. In its
    first cycle it consumes exactly 25 credits, in sessions on random days within X_j in [4, 20)
    days of its claim, at its own chronotype hour; the last event exhausts the cycle. Later cycles
    hold 0-2 sessions of at most 8 credits."""
    return _rotation(base, None, 0.0, k)


def rotator(base: random.Random, mech: random.Random, s: float, k: int) -> Plan:
    """Sequential rotator (pair 1 abusive, channel F).

    s is the probability, per hand-off j -> j+1, that account j's first-cycle session days are
    rescaled so that its exhaustion lands uniform{60..600} min before T_{j+1}. Otherwise its span
    is the twin's. The event count, amounts and creation times are always the twin's."""
    return _rotation(base, mech, s, k)


# --- Pair 2: parallel_farm vs classroom_lab (T) ---------------------------------------------------


def _restart(sess: _Session, start: int) -> _Session:
    """The session with its first event at minute `start`, keeping its gaps."""
    return replace(sess, offsets=tuple(start + o - sess.offsets[0] for o in sess.offsets))


def _lab(base: random.Random, mech: random.Random | None, s: float, k: int) -> Plan:
    check_size(k)
    created = [_created_t(base) for _ in range(k)]
    offsets = [draw_claim_offsets(base) for _ in range(k)]
    frames = [cycle_frames(created[j], offsets[j]) for j in range(k)]
    weekday = base.randrange(5)
    hour = base.randint(*LAB_HOURS)
    meetings = [d for d in range(WINDOW_DAYS) if (W0_WEEKDAY + d) % 7 == weekday]

    sessions: list[_Session] = []
    for d in meetings:
        attending = [a for a in range(k) if base.random() < LAB_ATTEND_P]
        starts = base.sample(range(LAB_OFFSETS), len(attending))
        for a, start in zip(attending, starts, strict=True):
            m = base.randint(1, 2)
            gap = base.randint(*LAB_GAP)
            amounts = tuple(base.randint(1, MAX_EVENT_CREDITS) for _ in range(m))
            offs = (start, start + gap)[:m]
            sessions.append(_Session(a, day_at(d, hour), offs, amounts, tag=d))
    admitted = _admit(sessions, frames)

    if mech is not None:
        rank = mech.sample(range(k), k)  # the operator's account order, once per group
        for d in meetings:
            u = mech.random()
            here = [x for x in admitted if x.tag == d]
            if u >= s or len(here) < 2:
                continue
            starts = sorted(x.offsets[0] for x in here)
            moved = {
                x.acct: _restart(x, st)
                for x, st in zip(
                    sorted(here, key=lambda x: rank.index(x.acct)), starts, strict=True
                )
            }
            admitted = [moved[x.acct] if x.tag == d else x for x in admitted]
    return Plan(
        tuple(created),
        tuple([hour] * k),
        tuple(offsets),
        _consume(admitted, k),
    )


def classroom_lab(base: random.Random, k: int) -> Plan:
    """Classroom lab (pair 2 twin, channel T).

    One weekly lab: a weekday and an hour in 9..16, fixed per group. At each meeting each
    eligible account attends with probability LAB_ATTEND_P. An attendee gets a distinct start
    offset from {0..20} min, 1-2 events with a gap of 3-15 min, and amounts of uniform{1..4}."""
    return _lab(base, None, 0.0, k)


def parallel_farm(base: random.Random, mech: random.Random, s: float, k: int) -> Plan:
    """Scripted parallel farm (pair 2 abusive, channel T).

    s is the probability, per meeting, that the attendees' drawn start offsets are reassigned in
    sorted order to the group's fixed operator order. The order is a permutation drawn once per
    group from `mech`. Each account keeps its own events, gaps and amounts."""
    return _lab(base, mech, s, k)


# --- Pair 3: template_operator vs office_card (T) -------------------------------------------------


def _office(base: random.Random, mech: random.Random | None, s: float, k: int) -> Plan:
    check_size(k)
    created = [_created_t(base) for _ in range(k)]
    offsets = [draw_claim_offsets(base) for _ in range(k)]
    chrono = [base.randint(*OFFICE_HOURS) for _ in range(k)]
    frames = [cycle_frames(created[j], offsets[j]) for j in range(k)]

    sessions: list[_Session] = []
    for d in range(WINDOW_DAYS):
        if not is_weekday(d):
            continue
        used: set[int] = set()
        for a in range(k):
            if base.random() >= OFFICE_DAY_P:
                continue
            # no two sessions of a group share a (day, clock hour)
            hour = base.choice(
                [h for h in range(OFFICE_HOURS[0], OFFICE_HOURS[1] + 1) if h not in used]
            )
            used.add(hour)
            start = base.randint(*OFFICE_START)
            gap = base.randint(*OFFICE_GAP)
            amounts = (base.randint(1, MAX_EVENT_CREDITS), base.randint(1, MAX_EVENT_CREDITS))
            sessions.append(_Session(a, day_at(d, hour), (start, start + gap), amounts, tag=d))
    admitted = _admit(sessions, frames)

    if mech is not None:
        template = (mech.randint(1, MAX_EVENT_CREDITS), mech.randint(1, MAX_EVENT_CREDITS))
        t_gap = mech.randint(*OFFICE_GAP)
        out: list[_Session] = []
        for x in sorted(admitted, key=lambda x: (x.acct, x.first)):
            if mech.random() < s:
                x = replace(x, offsets=(x.offsets[0], x.offsets[0] + t_gap), amounts=template)
            out.append(x)
        admitted = out
    return Plan(tuple(created), tuple(chrono), tuple(offsets), _consume(admitted, k))


def office_card(base: random.Random, k: int) -> Plan:
    """Small office team on a shared card (pair 3 twin, channel T).

    Each account, on each weekday with probability OFFICE_DAY_P, has one session at an hour in
    9..17. It has exactly 2 consume events: start uniform{0..25} min, gap uniform{2..30} min,
    amounts uniform{1..4} each. At most 3 sessions per account-cycle."""
    return _office(base, None, 0.0, k)


def template_operator(base: random.Random, mech: random.Random, s: float, k: int) -> Plan:
    """Template operator (pair 3 abusive, channel T).

    s is the probability, per (account, session), that the session's (amount 1, amount 2, gap) is
    replaced by the group's template. The template is one draw from the twin's per-session
    distribution, using `mech`. Session time, start minute and count are the twin's."""
    return _office(base, mech, s, k)


# --- Pair 4: single_budget vs heavy_family (T) ----------------------------------------------------


def _spread_hours(rng: random.Random, k: int) -> list[int]:
    """k hours in BUDGET_HOURS at least BUDGET_HOUR_SEP apart: a uniform valid set, shuffled."""
    lo, hi = BUDGET_HOURS
    gap = BUDGET_HOUR_SEP - 1
    slots = rng.sample(range(lo, hi - gap * (k - 1) + 1), k)
    hours = [h + gap * i for i, h in enumerate(sorted(slots))]
    rng.shuffle(hours)
    return hours


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _exchangeable(eps: Sequence[float], s: float) -> list[float]:
    """Z with unit marginals and correlation -s/(n-1) among the n values (DESIGN.md pair 4)."""
    n = len(eps)
    if n < 2 or s == 0.0:
        return list(eps)
    t = s / (n - 1 + s)
    gamma = 1.0 - math.sqrt(max(0.0, 1.0 - n * t))
    mean = sum(eps) / n
    scale = math.sqrt(1.0 - t)
    return [(e - gamma * mean) / scale for e in eps]


def _budget(base: random.Random, s: float, k: int) -> Plan:
    check_size(k)
    created = [_created_t(base) for _ in range(k)]
    offsets = [draw_claim_offsets(base) for _ in range(k)]
    hours = _spread_hours(base, k)
    frames = [cycle_frames(created[j], offsets[j]) for j in range(k)]

    sessions: list[_Session] = []
    for d in range(WINDOW_DAYS):
        if base.random() >= FAMILY_DAY_P:
            continue
        for a in range(k):
            if base.random() >= BUDGET_ACTIVE_P:
                continue
            m = base.randint(1, 2)
            start = base.randint(*BUDGET_START)
            gap = base.randint(*BUDGET_GAP)
            offs = (start, start + gap)[:m]
            sessions.append(_Session(a, day_at(d, hours[a]), offs, (0,) * m, tag=d))
    eps = [[base.gauss(0.0, 1.0) for _ in range(k)] for _ in range(WINDOW_DAYS)]
    admitted = _admit(sessions, frames)

    by_day: dict[int, list[_Session]] = defaultdict(list)
    for x in admitted:
        by_day[x.tag].append(x)
    final: list[_Session] = []
    for d, active in sorted(by_day.items()):
        z = _exchangeable([eps[d][x.acct] for x in active], s)
        for x, zi in zip(active, z, strict=True):
            level = 1 + min(3, int(4.0 * _phi(zi)))
            final.append(replace(x, amounts=(level,) * len(x.amounts)))
    return Plan(tuple(created), tuple(hours), tuple(offsets), _consume(final, k))


def heavy_family(base: random.Random, k: int) -> Plan:
    """Real heavy-use family (pair 4 twin, channel T).

    Each account has its own hour in 6..23, at least 3 h from every other member's. Each day is a
    family day with probability FAMILY_DAY_P. On a family day each eligible account is active with
    probability 0.8, in one session at its hour: 1-2 events, start uniform{0..30} min, gap
    uniform{2..25} min. Every event of that account-day has amount 1 + floor(4 Phi(Z)), and the
    Z of the twin are independent N(0, 1)."""
    return _budget(base, 0.0, k)


def single_budget(base: random.Random, mech: random.Random, s: float, k: int) -> Plan:
    """Single-budget "family" (pair 4 abusive, channel T).

    s is the strength of a negative exchangeable correlation, rho = -s/(n_d - 1), among the Z of
    the n_d >= 2 accounts active on day d. The Z come from the twin's N(0, 1) draws by a linear
    map (t = s/(n_d - 1 + s), gamma = 1 - sqrt(1 - n_d t), Z = (eps - gamma mean(eps)) /
    sqrt(1 - t)), so each account's marginal is unchanged. `mech` stays unused: the coupling is a
    transform of the twin's own draws."""
    del mech
    return _budget(base, s, k)


# --- Pair 5: aged_expansion vs adds_relative (M) --------------------------------------------------


@dataclass(frozen=True)
class _Slot:
    """A newcomer session's own start minute, gap and the twin's events."""

    start: int
    gap: int
    amounts: tuple[int, ...]


def _place(
    hour: int, row: Sequence[_Slot], picks: Sequence[tuple[datetime, tuple[int, ...] | None]]
) -> list[_Session]:
    """Sessions on the picked days: a slot's own start, gap and event count, and its own amounts
    unless the pick carries amounts to copy (copied in order, the slot's own where it has more)."""
    out: list[_Session] = []
    for slot, (day, copy) in zip(row, picks, strict=False):
        n = len(slot.amounts)
        amounts = slot.amounts if copy is None else (*copy, *slot.amounts[len(copy) :])[:n]
        out.append(
            _Session(
                0, day + timedelta(hours=hour), (slot.start, slot.start + slot.gap)[:n], amounts
            )
        )
    return out


def _twin_sessions(
    base: random.Random, frames: Sequence[CycleFrame], hour: int, slots: Sequence[Sequence[_Slot]]
) -> list[_Session]:
    out: list[_Session] = []
    for cycle, row in zip(frames, slots, strict=False):
        if cycle.claim is None:
            continue
        options = _hour_days(cycle.claim, cycle.limit, hour)
        days = sorted(base.sample(options, min(len(row), len(options))))
        out += _place(hour, row, [(d, None) for d in days])
    return out


def _follow_sessions(
    mech: random.Random,
    main: Sequence[_Session],
    main_hour: int,
    frames: Sequence[CycleFrame],
    hour: int,
    slots: Sequence[Sequence[_Slot]],
) -> list[_Session]:
    """Sessions on the main account's session days, at its hour, with its amounts.

    The session count per cycle is still the twin's. Where the main account has fewer session
    days left in the cycle than the twin draws (it stops at the cap), the rest are the newcomer's
    own sessions on other days, at its own hour."""
    out: list[_Session] = []
    for cycle, row in zip(frames, slots, strict=False):
        if cycle.claim is None:
            continue
        options = [
            x
            for x in main
            if x.at >= cycle.claim and x.at + minutes(SESSION_SPAN_MIN) < cycle.limit
        ]
        chosen = sorted(mech.sample(range(len(options)), min(len(row), len(options))))
        picks = [(midnight(options[i].at), options[i].amounts) for i in chosen]
        out += _place(main_hour, row[: len(picks)], picks)
        taken = {d for d, _ in picks}
        own = [d for d in _hour_days(cycle.claim, cycle.limit, hour) if d not in taken]
        days = sorted(mech.sample(own, min(len(row) - len(picks), len(own))))
        out += _place(hour, row[len(picks) :], [(d, None) for d in days])
    return out


def _expansion(base: random.Random, mech: random.Random | None, s: float, k: int) -> Plan:
    check_size(k)
    age = base.randrange(MAIN_AGE_DAYS[0] * DAY_MIN, MAIN_AGE_DAYS[1] * DAY_MIN)
    main_created = WORLD_END - minutes(age)
    main_hour = base.randint(*CHRONO_HOURS)
    main_offsets = draw_claim_offsets(base)

    # The main account reaches the 25 cap in every in-window cycle its days allow.
    main_sessions: list[_Session] = []
    exhaustions: list[datetime] = []
    for cycle in cycle_frames(main_created, main_offsets):
        if cycle.claim is None:
            continue
        options = _hour_days(cycle.claim, cycle.limit, main_hour)
        pace = min(1.0, MAIN_PACE_DAYS / max(1, len(options)))
        consumed = 0
        for d in options:
            u = base.random()
            start = base.randint(*EXPANSION_START)
            gap = base.randint(*EXPANSION_GAP)
            wanted = [base.randint(1, MAX_EVENT_CREDITS) for _ in range(base.randint(1, 2))]
            if consumed >= GRANT or u >= pace:
                continue
            kept: list[int] = []
            for amount in wanted:
                kept.append(min(amount, GRANT - consumed))
                consumed += kept[-1]
                if consumed == GRANT:
                    break
            offs = (start, start + gap)[: len(kept)]
            sess = _Session(0, d + timedelta(hours=main_hour), offs, tuple(kept))
            main_sessions.append(sess)
            if consumed == GRANT:
                exhaustions.append(sess.times()[-1])

    created = [main_created]
    chrono = [main_hour]
    offsets = [main_offsets]
    consume = [_consume(main_sessions, 1)[0]]
    lo = W0 + timedelta(days=NEWCOMER_FROM_DAYS)
    hi = WORLD_END - timedelta(days=NEWCOMER_UNTIL_DAYS)
    for _ in range(1, k):
        drawn = lo + minutes(base.randrange((hi - lo) // minutes(1)))
        hour = base.randint(*CHRONO_HOURS)
        delays = draw_claim_offsets(base)
        slots: list[list[_Slot]] = []
        for _cycle in range(3):
            row: list[_Slot] = []
            for _session in range(base.randint(*NEWCOMER_CYCLE_SESSIONS)):
                start = base.randint(*EXPANSION_START)
                gap = base.randint(*EXPANSION_GAP)
                both = (base.randint(1, MAX_EVENT_CREDITS), base.randint(1, MAX_EVENT_CREDITS))
                row.append(_Slot(start, gap, both[: base.randint(1, 2)]))
            slots.append(row)

        new_created = drawn
        sessions = _twin_sessions(base, cycle_frames(drawn, delays), hour, slots)
        if mech is not None:
            u = mech.random()
            lead = mech.randint(*FOLLOW_DELAY_MIN)
            near = [e for e in exhaustions if lo <= e < hi]
            if near:
                after = min(near, key=lambda e: abs(e - drawn)) + minutes(lead)
                follow = _follow_sessions(
                    mech, main_sessions, main_hour, cycle_frames(after, delays), hour, slots
                )
                if u < s:
                    sessions, new_created = follow, after
        created.append(new_created)
        chrono.append(hour)
        offsets.append(delays)
        consume.append(_consume(sessions, 1)[0])
    return Plan(tuple(created), tuple(chrono), tuple(offsets), tuple(consume))


def adds_relative(base: random.Random, k: int) -> Plan:
    """Established user adds a relative (pair 5 twin, channel M).

    The main account is created WORLD_END - uniform[200, 400) d. It reaches the 25 cap in every
    in-window cycle, in sessions at its own hour on its own days. Each of the k - 1 newcomers is
    created at uniform[W0 + 5 d, WORLD_END - 20 d) and has its own hour, days and amounts, and
    NEWCOMER_CYCLE_SESSIONS sessions per cycle."""
    return _expansion(base, None, 0.0, k)


def aged_expansion(base: random.Random, mech: random.Random, s: float, k: int) -> Plan:
    """Established-account expansion (pair 5 abusive, channel M).

    s is the probability, per newcomer, that it follows the rule. If it does: it's created
    uniform{60..1440} min after the main account's exhaustion nearest to its drawn creation time;
    its sessions sit on main-account session days after its creation, at the main account's hour,
    with the main account's session amounts; its session count per cycle is still the twin's
    draw."""
    return _expansion(base, mech, s, k)


# --- Pair 6: serial_operator vs roommates_card (T) ------------------------------------------------


def _roommates(base: random.Random, mech: random.Random | None, s: float, k: int) -> Plan:
    check_size(k)
    created = [_created_t(base) for _ in range(k)]
    offsets = [draw_claim_offsets(base) for _ in range(k)]
    chrono = [base.randint(*SHARED_HOURS) for _ in range(k)]
    frames = [cycle_frames(created[j], offsets[j]) for j in range(k)]

    sessions: list[_Session] = []
    for d in range(WINDOW_DAYS):
        if base.random() >= SHARED_DAY_P:
            continue
        hour = base.randint(*SHARED_HOURS)
        joiners = [a for a in range(k) if base.random() < SHARED_JOIN_P]
        counts = [base.randint(*SHARED_EVENTS) for _ in joiners]
        slots = sorted(base.sample(range(SHARED_MINUTES), sum(counts)))
        owners = [a for a, m in zip(joiners, counts, strict=True) for _ in range(m)]
        base.shuffle(owners)
        for a, m in zip(joiners, counts, strict=True):
            top = MAX_EVENT_CREDITS if m == SHARED_EVENTS[0] else MAX_EVENT_CREDITS // 2
            amounts = tuple(base.randint(1, top) for _ in range(m))
            mine = tuple(t for t, o in zip(slots, owners, strict=True) if o == a)
            sessions.append(_Session(a, day_at(d, hour), mine, amounts, tag=d))
    admitted = _admit(sessions, frames)

    if mech is not None:
        by_day: dict[int, list[_Session]] = defaultdict(list)
        for x in admitted:
            by_day[x.tag].append(x)
        final: list[_Session] = []
        for _day, here in sorted(by_day.items()):
            u = mech.random()
            order = list(range(len(here)))
            mech.shuffle(order)
            if u >= s or len(here) < 2:
                final.extend(here)
                continue
            pool = sorted(t for x in here for t in x.offsets)
            at = 0
            for i in order:
                n = len(here[i].offsets)
                final.append(replace(here[i], offsets=tuple(pool[at : at + n])))
                at += n
        admitted = sorted(final, key=lambda x: (x.first, x.acct))
    return Plan(tuple(created), tuple(chrono), tuple(offsets), _consume(admitted, k))


def roommates_card(base: random.Random, k: int) -> Plan:
    """Roommates sharing a payment card (pair 6 twin, channel T).

    Each day, with probability SHARED_DAY_P, there's a shared session at an hour in 18..22. Each
    eligible account joins with probability 0.7. A joiner gets 2-3 consume events at distinct
    minutes drawn uniformly over {0..55} for the whole session, so the accounts' events interleave.
    Amounts are uniform{1..4} for 2 events and uniform{1..2} for 3."""
    return _roommates(base, None, 0.0, k)


def serial_operator(base: random.Random, mech: random.Random, s: float, k: int) -> Plan:
    """Serial session operator (pair 6 abusive, channel T).

    s is the probability, per shared session (at least 2 joiners), that the session's sorted
    consume minutes are reassigned so each joiner's events are contiguous. The block order is
    random from `mech`, and each joiner keeps its event count and its amounts in order."""
    return _roommates(base, mech, s, k)


# --- Registry -------------------------------------------------------------------------------------

AbusiveProcess = Callable[[random.Random, random.Random, float, int], Plan]
TwinProcess = Callable[[random.Random, int], Plan]

ABUSIVE: dict[str, AbusiveProcess] = {
    ROTATOR: rotator,
    PARALLEL_FARM: parallel_farm,
    TEMPLATE_OPERATOR: template_operator,
    SINGLE_BUDGET: single_budget,
    AGED_EXPANSION: aged_expansion,
    SERIAL_OPERATOR: serial_operator,
}
TWINS: dict[str, TwinProcess] = {
    HOUSEHOLD: household_device,
    CLASSROOM: classroom_lab,
    OFFICE_CARD: office_card,
    HEAVY_FAMILY: heavy_family,
    ADDS_RELATIVE: adds_relative,
    ROOMMATES: roommates_card,
}
FAMILIES = tuple(f for pair in PAIRS.values() for f in pair)
