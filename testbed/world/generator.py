"""Generator v1 (SPEC §5, DESIGN.md): the frame around the 12 family processes.

A group is drawn from two RNG streams seeded by strings, so output depends only on the generator
version, seed, s and scale. The family process (`families.py`) draws the creation times and consume
events from `base` (and its mechanism from `mech`). Everything else comes from `base`, after the
process, in a fixed order: the nuisance sampler, linkage, then each account's verification and
logins. Those draws never read what a mechanism changed except through arithmetic on times, so a
group's base stream is the same at every s.

Scales: `standard` (6 pairs x 40 + 160) and `channel` (pairs 2, 3, 4, 6 x 200 + 800, at s = 1) are
the reported ones. `small` (6 pairs x 4 + 16) is test-only, for end-to-end tests; it is never a
reported world.

Pure and deterministic: no I/O, no wall clock. Labels travel only inside the vendored
`GeneratedWorld`, to the write-only sink in `persist.py`.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.models.enums import (
    EmailDomainType,
    EndpointCostTier,
    EventType,
    GroundTruthLabel,
    Reliability,
    SignalType,
    VerificationLevel,
)
from app.services.generator import (
    WORLD_END,
    GenAccount,
    GenCycle,
    GeneratedWorld,
    GenEvent,
    GenLabel,
    GenLinkage,
    GenSplit,
    LinkageFixture,
    canonical_bytes,
)

from testbed.pipeline.timeline import (
    WINDOW_START,
    Account,
    Event,
    Shared,
    UnitTimeline,
    alias_key,
    aliases,
)
from testbed.world.families import (
    ABUSIVE,
    CHANNEL_TAGS,
    GRANT,
    GROUP_SIZE,
    MEASURED_TAGS,
    PAIRS,
    T_PAIRS,
    TWINS,
    W0,
    Plan,
    cycle_frames,
    frame_at,
    midnight,
    minutes,
)

GENERATOR_VERSION = "tt-gen-v1"

# --- Nuisance sampler (DESIGN.md) -----------------------------------------------------------------

DOMAIN_WEIGHTS = (
    (EmailDomainType.CONSUMER, 0.70),
    (EmailDomainType.EDUCATION, 0.10),
    (EmailDomainType.ENTERPRISE, 0.10),
    (EmailDomainType.DISPOSABLE, 0.10),
)
LEVEL_WEIGHTS = (
    (VerificationLevel.NONE, 0.10),
    (VerificationLevel.EMAIL, 0.40),
    (VerificationLevel.PHONE, 0.40),
    (VerificationLevel.ENHANCED, 0.10),
)
VERIFY_BEFORE_CLAIM_P = 0.8  # (tune) otherwise 1-48 h after the first claim
VERIFY_LATE_MIN = (60, 2880)  # (tune)
VERIFY_FAILED_P = 0.1
LOGIN_RATE = (
    2.0  # (tune) Poisson mean over the account's in-window life; DESIGN 4, cut for the budget
)
LOGIN_HOUR_JITTER = 2  # the account's hour plus or minus 2 h
EDGE_SIGNALS = (SignalType.STABLE_DEVICE, SignalType.PAYMENT_TOKEN, SignalType.RECOVERY_CONTACT)
EDGE_RELIABILITIES = (Reliability.MEDIUM, Reliability.HIGH)
LINK_DELAY_MIN = 2880  # observed at max(created_at, W0) + uniform{0..2880} min
NETWORK_P = 0.4
BEHAVIORAL_P = 0.3

# --- Scales (SPEC §5.7) ---------------------------------------------------------------------------

SCALES: dict[str, tuple[tuple[int, ...], int, int]] = {
    "standard": (tuple(PAIRS), 40, 160),
    "small": (tuple(PAIRS), 4, 16),  # test-only: same code path, never a reported world
    "channel": (T_PAIRS, 200, 800),
}


@dataclass(frozen=True)
class WorldSpec:
    seed: int
    s: float
    scale: str

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ValueError(f"unknown scale {self.scale!r}; choose from {sorted(SCALES)}")
        if not 0.0 <= self.s <= 1.0:
            raise ValueError(f"s must lie in [0, 1], got {self.s}")
        if self.scale == "channel" and self.s != 1.0:
            raise ValueError("the channel scale is generated at s = 1")


@dataclass(frozen=True)
class GroupDraw:
    """One group as the generator drew it: inference rows plus the label and ids (truth)."""

    group_id: str
    family: str
    label: GroundTruthLabel
    accounts: tuple[GenAccount, ...]
    events: tuple[GenEvent, ...]
    linkage: tuple[GenLinkage, ...]
    cycles: tuple[GenCycle, ...]


def _sha(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _pick(rng: random.Random, n: int) -> int:
    """A uniform integer in [0, n) from exactly one draw, so draw counts never depend on n."""
    return min(n - 1, int(rng.random() * n))


def _weighted[T](rng: random.Random, table: Sequence[tuple[T, float]]) -> T:
    u = rng.random()
    total = 0.0
    for item, weight in table:
        total += weight
        if u < total:
            return item
    return table[-1][0]


def _poisson(rng: random.Random, mean: float) -> int:
    u = rng.random()
    p = math.exp(-mean)
    cumulative, n = p, 0
    while u > cumulative and n < 100:
        n += 1
        p *= mean / n
        cumulative += p
    return n


def _verification(
    rng: random.Random, created: datetime, first_claim: datetime, level: VerificationLevel
) -> list[tuple[datetime, EventType]]:
    u_before, u_failed, r_when, r_failed = (rng.random() for _ in range(4))
    if level is VerificationLevel.NONE:
        return []
    lo = max(created, W0)
    if u_before < VERIFY_BEFORE_CLAIM_P:
        span = (first_claim - lo) // minutes(1)
        when = lo + minutes(1 + int(r_when * (span - 1)))
    else:
        late = VERIFY_LATE_MIN
        when = first_claim + minutes(late[0] + int(r_when * (late[1] - late[0] + 1)))
    when = min(when, WORLD_END - minutes(1))
    rows = [(when, EventType.VERIFICATION)]
    if u_failed < VERIFY_FAILED_P:
        rows.append(
            (
                lo + minutes(int(r_failed * ((when - lo) // minutes(1)))),
                EventType.VERIFICATION_FAILED,
            )
        )
    return rows


def _logins(rng: random.Random, created: datetime, hour: int) -> list[datetime]:
    """Poisson logins over the account's in-window life, at its hour plus or minus 2 h."""
    lo = max(created, W0) + minutes(1)
    day0 = midnight(lo)
    days = (WORLD_END - day0).days
    out: list[datetime] = []
    for _ in range(_poisson(rng, LOGIN_RATE)):
        r_day, r_hour, r_minute = rng.random(), rng.random(), rng.random()
        h = (hour + _pick_from(r_hour, 2 * LOGIN_HOUR_JITTER + 1) - LOGIN_HOUR_JITTER) % 24
        t = day0 + minutes(_pick_from(r_day, days) * 1440 + h * 60 + _pick_from(r_minute, 60))
        out.append(min(max(t, lo), WORLD_END - minutes(1)))
    return out


def _pick_from(r: float, n: int) -> int:
    return min(n - 1, int(r * n))


def draw_group(family: str, i: int, seed: int, s: float) -> GroupDraw:
    """Group `i` of `family`. A twin ignores `s`.

    Group size is balanced by design, not drawn: `k = 2 + i % 5`, so groups 0..4 have sizes 2..6
    and the pattern repeats. Every reported scale has per-family counts that are multiples of 5,
    hence exactly `count / 5` groups of each size per family in every world. The test-only `small`
    scale's abusive count is 4 and is not balanced; it is never a reported world."""
    k = GROUP_SIZE[0] + i % 5
    stream = f"{GENERATOR_VERSION}|{seed}|{family}|{i}"
    base = random.Random(f"{stream}|base")
    if family in ABUSIVE:
        plan = ABUSIVE[family](base, random.Random(f"{stream}|mech"), s, k)
        label = GroundTruthLabel.ABUSIVE
    elif family in TWINS:
        plan = TWINS[family](base, k)
        label = GroundTruthLabel.LEGITIMATE
    else:
        raise ValueError(f"unknown family {family!r}")
    return _assemble(family, i, seed, base, plan, label)


def _assemble(
    family: str,
    i: int,
    seed: int,
    base: random.Random,
    plan: Plan,
    label: GroundTruthLabel,
) -> GroupDraw:
    k = plan.size
    group_id = "grp-" + _sha(GENERATOR_VERSION, seed, family, i)[:16]
    ids = [f"acct-{_sha(GENERATOR_VERSION, seed, family, i, j)[:16]}" for j in range(k)]

    domains = [_weighted(base, DOMAIN_WEIGHTS) for _ in range(k)]
    levels = [_weighted(base, LEVEL_WEIGHTS) for _ in range(k)]

    edge_signal = EDGE_SIGNALS[_pick(base, len(EDGE_SIGNALS))]
    edge_reliability = EDGE_RELIABILITIES[_pick(base, len(EDGE_RELIABILITIES))]
    context = [(edge_signal, edge_reliability)]
    if base.random() < NETWORK_P:
        context.append((SignalType.NETWORK_GROUP, Reliability.LOW))
    if base.random() < BEHAVIORAL_P:
        context.append((SignalType.BEHAVIORAL_TEMPLATE, Reliability.LOW))
    values = {sig: _sha(GENERATOR_VERSION, seed, group_id, sig.value)[:32] for sig, _ in context}

    accounts: list[GenAccount] = []
    events: list[GenEvent] = []
    linkage: list[GenLinkage] = []
    cycles: list[GenCycle] = []
    for j in range(k):
        acct, created = ids[j], plan.created[j]
        accounts.append(GenAccount(acct, created, domains[j], levels[j]))

        for n, (sig, reliability) in enumerate(context):
            # From the later of creation and W0: the vendored clustering drops observations older
            # than 365 days, and an aged account would otherwise leave its group's cluster.
            seen = min(
                max(created, W0) + minutes(_pick(base, LINK_DELAY_MIN + 1)), WORLD_END - minutes(1)
            )
            linkage.append(GenLinkage(f"{acct}-l{n}", acct, seen, sig, values[sig], reliability))

        frames = cycle_frames(created, plan.offsets[j])
        first_claim = next(f.claim for f in frames if f.claim is not None)
        rows: list[tuple[datetime, EventType, int | None, EndpointCostTier | None]] = [
            (created, EventType.SIGNUP, None, None)
        ]
        rows += [(f.claim, EventType.CLAIM_CREDITS, GRANT, None) for f in frames if f.claim]
        rows += [
            (t, EventType.CONSUME_CREDITS, a, EndpointCostTier.HIGH) for t, a in plan.consume[j]
        ]
        rows += [
            (t, kind, None, None)
            for t, kind in _verification(base, created, first_claim, levels[j])
        ]
        rows += [(t, EventType.LOGIN, None, None) for t in _logins(base, created, plan.chrono[j])]
        rows.sort(key=lambda r: (r[0], r[1].value))
        events += [
            GenEvent(f"{acct}-e{n:05d}", acct, t, kind, amount, tier)
            for n, (t, kind, amount, tier) in enumerate(rows)
        ]

        consumed: dict[int, int] = defaultdict(int)
        for t, amount in plan.consume[j]:
            f = frame_at(frames, t)
            if f is None or f.claim is None or t < f.claim or t >= WORLD_END:
                raise ValueError(f"consume at {t} lies outside any claimed cycle")
            consumed[f.index] += amount
        for f in frames:
            if consumed[f.index] > GRANT:
                raise ValueError(f"cycle {f.index} consumes {consumed[f.index]} > {GRANT}")
            expired = GRANT - consumed[f.index] if f.end <= WORLD_END else 0
            cycles.append(
                GenCycle(
                    f"{acct}-c{f.index:02d}",
                    acct,
                    f.start,
                    f.end,
                    GRANT,
                    consumed[f.index],
                    expired,
                )
            )
    return GroupDraw(
        group_id, family, label, tuple(accounts), tuple(events), tuple(linkage), tuple(cycles)
    )


def generate_groups(family: str, n: int, seed: int, s: float) -> list[GroupDraw]:
    """The generator-only sample (SPEC §5.6): groups 0..n-1 of `family`, in memory only."""
    return [draw_group(family, i, seed, s) for i in range(n)]


def world_groups(spec: WorldSpec) -> list[GroupDraw]:
    pairs, abusive, twin = SCALES[spec.scale]
    groups: list[GroupDraw] = []
    for pair in pairs:
        abusive_family, twin_family = PAIRS[pair]
        groups += generate_groups(abusive_family, abusive, spec.seed, spec.s)
        groups += generate_groups(twin_family, twin, spec.seed, spec.s)
    return groups


def assemble_world(groups: Sequence[GroupDraw]) -> GeneratedWorld:
    """Groups in the vendored data model, with the vendored row order."""
    accounts: list[GenAccount] = []
    events: list[GenEvent] = []
    linkage: list[GenLinkage] = []
    cycles: list[GenCycle] = []
    labels: list[GenLabel] = []
    splits: list[GenSplit] = []
    fixtures: list[LinkageFixture] = []
    for g in groups:
        accounts += g.accounts
        events += g.events
        linkage += g.linkage
        cycles += g.cycles
        labels.append(GenLabel(g.group_id, g.family, g.label, ""))
        splits += [GenSplit(a.account_id, g.group_id) for a in g.accounts]
        members = sorted(a.account_id for a in g.accounts)
        fixtures.append(LinkageFixture(g.group_id, g.family, [members]))
    return GeneratedWorld(
        accounts=sorted(accounts, key=lambda a: a.account_id),
        events=sorted(events, key=lambda e: (e.account_id, e.timestamp, e.event_id)),
        linkage=sorted(linkage, key=lambda x: x.observation_id),
        cycles=sorted(cycles, key=lambda c: c.cycle_id),
        labels=sorted(labels, key=lambda x: x.cluster_id),
        split_groups=sorted(splits, key=lambda x: x.account_id),
        fixtures=sorted(fixtures, key=lambda f: f.cluster_id),
    )


def generate_world(spec: WorldSpec) -> GeneratedWorld:
    """The world for `spec`. Deterministic and pure."""
    return assemble_world(world_groups(spec))


def world_manifest(world: GeneratedWorld, spec: WorldSpec) -> dict[str, Any]:
    """The manifest `persist` records. Its `world_version` is the generator version alone, so the
    inference receipt never names s."""
    abusive = sum(1 for x in world.labels if x.label is GroundTruthLabel.ABUSIVE)
    return {
        "tier": "dev",
        "reported": False,
        "generator_version": GENERATOR_VERSION,
        "world_version": GENERATOR_VERSION,
        "seed": spec.seed,
        "s": spec.s,
        "scale": spec.scale,
        "families": sorted({x.scenario_family for x in world.labels}),
        "channel_tags": {str(p): MEASURED_TAGS[p] for p in SCALES[spec.scale][0]},
        "designed_channel_tags": {str(p): CHANNEL_TAGS[p] for p in SCALES[spec.scale][0]},
        "counts": {
            "groups": len(world.labels),
            "abusive_groups": abusive,
            "accounts": len(world.accounts),
            "events": len(world.events),
            "linkage_observations": len(world.linkage),
            "cycles": len(world.cycles),
        },
        "checksum_sha256": hashlib.sha256(canonical_bytes(world)).hexdigest(),
    }


def group_timeline(group: GroupDraw) -> UnitTimeline:
    """The group's label-free timeline, equal to `load_timeline` on the persisted world.

    Names follow what the store returns: the vendored enum *names* (`"CONSUME_CREDITS"`)."""
    alias = aliases([a.account_id for a in group.accounts])
    ordered = sorted(group.accounts, key=lambda a: a.account_id)
    accounts = tuple(
        Account(
            alias[a.account_id], a.created_at, a.email_domain_type.name, a.verification_level.name
        )
        for a in ordered
    )
    holders: dict[tuple[SignalType, str], set[str]] = defaultdict(set)
    reliabilities: dict[tuple[SignalType, str], Reliability] = {}
    for x in group.linkage:
        if x.timestamp < WORLD_END:
            holders[(x.signal_type, x.signal_value_hash)].add(alias[x.account_id])
            reliabilities[(x.signal_type, x.signal_value_hash)] = x.reliability
    shared = [
        Shared(sig.name, reliabilities[(sig, value)].name, tuple(sorted(held, key=alias_key)))
        for (sig, value), held in holders.items()
    ]
    shared.sort(key=lambda x: (x.signal_type, [alias_key(a) for a in x.aliases], x.reliability))
    window = sorted(
        (e for e in group.events if WINDOW_START <= e.timestamp < WORLD_END),
        key=lambda e: (e.timestamp, e.account_id, e.event_id),
    )
    events = tuple(
        Event(e.timestamp, alias[e.account_id], e.event_type.name, e.credits_amount) for e in window
    )
    return UnitTimeline(WINDOW_START, WORLD_END, accounts, tuple(shared), events)
