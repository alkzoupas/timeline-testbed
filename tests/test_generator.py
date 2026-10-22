"""Generator v1: determinism, the s = 0 twin, the T couplings, validity, opacity and the timeline
contract (SPEC §5.1, §5.2, §5.5, §5.7; DESIGN.md)."""

from __future__ import annotations

import hashlib
import random
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import pytest
from app.models.enums import EventType, GroundTruthLabel, Reliability, SignalType
from app.services.generator import WORLD_END, GenCycle, GeneratedWorld, GenEvent, canonical_bytes

from testbed.eval.stores import verify_pair
from testbed.pipeline.timeline import load_timeline
from testbed.world.families import (
    ABUSIVE,
    CHANNEL_TAGS,
    FAMILIES,
    GRANT,
    MEASURED_TAGS,
    PAIRS,
    T_PAIRS,
    TWINS,
    W0,
    _exchangeable,
    adds_relative,
    aged_expansion,
    cycle_frames,
    heavy_family,
    parallel_farm,
    rotator,
    serial_operator,
    single_budget,
    template_operator,
)
from testbed.world.generator import (
    GENERATOR_VERSION,
    GroupDraw,
    WorldSpec,
    assemble_world,
    draw_group,
    generate_groups,
    generate_world,
    group_timeline,
    world_manifest,
)
from testbed.world.persist import Stores, persist

SEED = 1
EDGE_SIGNALS = {SignalType.STABLE_DEVICE, SignalType.PAYMENT_TOKEN, SignalType.RECOVERY_CONTACT}
MINUTE = timedelta(minutes=1)


@pytest.fixture(scope="module")
def standard() -> GeneratedWorld:
    return generate_world(WorldSpec(SEED, 1.0, "standard"))


def _k(i: int) -> int:
    """The generator's balanced group size for group `i`, for direct process calls."""
    return 2 + i % 5


def _pair_of(family: str) -> int:
    return next(p for p, names in PAIRS.items() if family in names)


# --- Determinism and scale -----------------------------------------------------------------------


def test_same_spec_same_bytes_and_other_seed_differs(standard: GeneratedWorld) -> None:
    again = generate_world(WorldSpec(SEED, 1.0, "standard"))
    other = generate_world(WorldSpec(SEED + 1, 1.0, "standard"))
    assert canonical_bytes(standard) == canonical_bytes(again)
    assert canonical_bytes(standard) != canonical_bytes(other)


def test_s_changes_the_world() -> None:
    lo = generate_world(WorldSpec(SEED, 0.0, "standard"))
    hi = generate_world(WorldSpec(SEED, 1.0, "standard"))
    assert canonical_bytes(lo) != canonical_bytes(hi)


def test_standard_world_scale_and_prevalence(standard: GeneratedWorld) -> None:
    by_family = Counter(x.scenario_family for x in standard.labels)
    for abusive, twin in PAIRS.values():
        assert by_family[abusive] == 40
        assert by_family[twin] == 160
    labels = Counter(x.label for x in standard.labels)
    assert labels == {GroundTruthLabel.ABUSIVE: 240, GroundTruthLabel.LEGITIMATE: 960}
    assert len(standard.labels) == len(standard.fixtures) == 1200


def test_small_scale_is_the_same_code_path_at_any_s() -> None:
    world = generate_world(WorldSpec(SEED, 0.5, "small"))
    by_family = Counter(x.scenario_family for x in world.labels)
    assert {f: by_family[f] for f in FAMILIES} == {
        f: (4 if f in {a for a, _ in PAIRS.values()} else 16) for f in FAMILIES
    }
    assert len(world.labels) == 120
    # the same group index draws the same group at every scale
    standard_group = draw_group("classroom_lab", 2, SEED, 0.5)
    ids = {a.account_id for a in world.accounts}
    assert {a.account_id for a in standard_group.accounts} <= ids


def test_channel_scale_is_the_four_t_pairs_at_s_one() -> None:
    with pytest.raises(ValueError, match="channel"):
        WorldSpec(SEED, 0.5, "channel")
    with pytest.raises(ValueError, match="scale"):
        WorldSpec(SEED, 1.0, "huge")
    with pytest.raises(ValueError, match="s must"):
        WorldSpec(SEED, 1.5, "standard")
    spec = WorldSpec(SEED, 1.0, "channel")
    assert (spec.seed, spec.s, spec.scale) == (SEED, 1.0, "channel")


def test_group_sizes_are_balanced_by_design(standard: GeneratedWorld) -> None:
    size_of = Counter(x.split_group_id for x in standard.split_groups)
    labels = {x.cluster_id: x.scenario_family for x in standard.labels}
    per_family: dict[str, Counter[int]] = defaultdict(Counter)
    for group_id, family in labels.items():
        per_family[family][size_of[group_id]] += 1
    for abusive, twin in PAIRS.values():
        assert per_family[abusive] == {k: 8 for k in range(2, 7)}  # 40 / 5
        assert per_family[twin] == {k: 32 for k in range(2, 7)}  # 160 / 5
    for family in ("rotator", "classroom_lab"):
        for i in range(12):
            assert len(draw_group(family, i, SEED, 1.0).accounts) == 2 + i % 5


def test_a_process_rejects_a_group_size_outside_two_to_six() -> None:
    for k in (1, 7):
        with pytest.raises(ValueError, match="group size"):
            TWINS["classroom_lab"](random.Random("x"), k)
        with pytest.raises(ValueError, match="group size"):
            ABUSIVE["rotator"](random.Random("x"), random.Random("y"), 1.0, k)


def test_manifest_counts_and_checksum(standard: GeneratedWorld) -> None:
    spec = WorldSpec(SEED, 1.0, "standard")
    manifest = world_manifest(standard, spec)
    assert manifest["generator_version"] == manifest["world_version"] == GENERATOR_VERSION
    assert (manifest["seed"], manifest["s"], manifest["scale"]) == (SEED, 1.0, "standard")
    assert manifest["checksum_sha256"] == hashlib.sha256(canonical_bytes(standard)).hexdigest()
    assert manifest["counts"]["groups"] == 1200
    assert manifest["counts"]["abusive_groups"] == 240
    assert manifest["counts"]["accounts"] == len(standard.accounts)
    assert manifest["families"] == sorted(FAMILIES)
    # SPEC §5.4: the manifest carries the frozen measured tags, and the design tags beside them
    assert manifest["channel_tags"] == {str(p): t for p, t in MEASURED_TAGS.items()}
    assert manifest["designed_channel_tags"] == {str(p): t for p, t in CHANNEL_TAGS.items()}


def test_standard_world_generates_quickly() -> None:
    started = time.perf_counter()
    generate_world(WorldSpec(SEED + 100, 1.0, "standard"))
    assert time.perf_counter() - started < 30


# --- s = 0 is the twin, exactly ------------------------------------------------------------------


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_abusive_process_at_s_zero_is_its_twin(pair: int) -> None:
    abusive, twin = PAIRS[pair]
    for i in range(40):
        want = TWINS[twin](random.Random(f"base-{pair}-{i}"), _k(i))
        for mech_seed in ("a", "b"):
            got = ABUSIVE[abusive](
                random.Random(f"base-{pair}-{i}"), random.Random(mech_seed), 0.0, _k(i)
            )
            assert got == want


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_mechanism_does_something_at_s_one(pair: int) -> None:
    abusive, twin = PAIRS[pair]
    differs = sum(
        ABUSIVE[abusive](random.Random(f"base-{pair}-{i}"), random.Random("m"), 1.0, _k(i))
        != TWINS[twin](random.Random(f"base-{pair}-{i}"), _k(i))
        for i in range(40)
    )
    assert differs >= 30


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_a_twin_ignores_s(pair: int) -> None:
    _, twin = PAIRS[pair]
    assert draw_group(twin, 3, SEED, 0.0) == draw_group(twin, 3, SEED, 1.0)


# --- Known-answer tests for each mechanism --------------------------------------------------------


def _first_cycle_exhaustion(created: datetime, rows: tuple[tuple[datetime, int], ...]) -> datetime:
    total = 0
    for t, amount in rows:
        total += amount
        if total >= GRANT and t < created + timedelta(days=30):
            return t
    raise AssertionError("never exhausts")


def test_pair1_rescale_lands_exhaustion_before_the_next_turn() -> None:
    checked = 0
    for i in range(60):
        twin = TWINS["household_device"](random.Random(f"r{i}"), _k(i))
        got = rotator(random.Random(f"r{i}"), random.Random("m"), 1.0, _k(i))
        assert got.created == twin.created
        assert [[a for _, a in r] for r in got.consume] == [[a for _, a in r] for r in twin.consume]
        for j in range(got.size - 1):
            exhausted = _first_cycle_exhaustion(got.created[j], got.consume[j])
            lead = (got.created[j + 1] - exhausted) // MINUTE
            assert 60 <= lead <= 600
            checked += 1
        last = got.size - 1
        assert got.consume[last] == twin.consume[last]
    assert checked > 100


def test_pair1_first_cycle_is_exactly_25_credits() -> None:
    for i in range(60):
        plan = TWINS["household_device"](random.Random(f"h{i}"), _k(i))
        for j in range(plan.size - 1):
            first = [a for t, a in plan.consume[j] if t < plan.created[j] + timedelta(days=30)]
            assert sum(first) == GRANT


def _pair_order_signs(
    plan_rows: tuple[tuple[tuple[datetime, int], ...], ...],
) -> dict[tuple[int, int], set[int]]:
    starts: dict[tuple[int, int], dict[int, datetime]] = defaultdict(dict)
    for a, rows in enumerate(plan_rows):
        for t, _ in rows:
            slot = (t.toordinal(), t.hour)
            starts[slot].setdefault(a, t)
    signs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for first in starts.values():
        for a, b in combinations(sorted(first), 2):
            signs[(a, b)].add(1 if first[a] < first[b] else -1)
    return signs


def test_pair2_full_mechanism_gives_one_fixed_order() -> None:
    full = [
        _pair_order_signs(
            parallel_farm(random.Random(f"p{i}"), random.Random("m"), 1.0, _k(i)).consume
        )
        for i in range(40)
    ]
    twin = [
        _pair_order_signs(TWINS["classroom_lab"](random.Random(f"p{i}"), _k(i)).consume)
        for i in range(40)
    ]
    assert all(len(v) == 1 for signs in full for v in signs.values())
    assert any(len(v) == 2 for signs in twin for v in signs.values())


def test_pair3_full_mechanism_replays_one_template() -> None:
    for i in range(40):
        plan = template_operator(random.Random(f"t{i}"), random.Random("m"), 1.0, _k(i))
        scripts = set()
        for rows in plan.consume:
            sessions = defaultdict(list)
            for t, a in rows:
                sessions[(t.toordinal(), t.hour)].append((t, a))
            for events in sessions.values():
                (t1, a1), (t2, a2) = events
                scripts.add((a1, a2, (t2 - t1) // MINUTE))
        assert len(scripts) <= 1


def test_pair4_two_active_accounts_split_a_fixed_budget() -> None:
    days = 0
    for i in range(60):
        plan = single_budget(random.Random(f"b{i}"), random.Random("m"), 1.0, _k(i))
        by_day: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        for a, rows in enumerate(plan.consume):
            for t, amount in rows:
                by_day[t.toordinal()][a].add(amount)
        for per in by_day.values():
            assert all(len(v) == 1 for v in per.values())  # one amount per account-day
            if len(per) == 2:
                assert sum(next(iter(v)) for v in per.values()) == 5
                days += 1
    assert days > 50


def test_pair4_exchangeable_construction_known_answers() -> None:
    eps = [0.3, -1.2, 0.7, 1.9]
    assert _exchangeable(eps, 0.0) == eps
    assert _exchangeable([0.5], 1.0) == [0.5]
    assert sum(_exchangeable(eps, 1.0)) == pytest.approx(0.0, abs=1e-12)
    rng = random.Random(0)
    draws = [_exchangeable([rng.gauss(0, 1) for _ in range(4)], 0.6) for _ in range(40000)]
    var = sum(z[0] ** 2 for z in draws) / len(draws)
    cov = sum(z[0] * z[1] for z in draws) / len(draws)
    assert var == pytest.approx(1.0, abs=0.03)
    assert cov / var == pytest.approx(-0.6 / 3, abs=0.03)  # rho = -s / (n - 1)


def test_pair4_hours_are_spread_even_for_six_accounts() -> None:
    sizes: Counter[int] = Counter()
    for i in range(300):
        plan = heavy_family(random.Random(f"s{i}"), _k(i))
        sizes[plan.size] += 1
        assert all(6 <= h <= 23 for h in plan.chrono)
        assert all(abs(a - b) >= 3 for a, b in combinations(plan.chrono, 2))
    assert set(sizes) == {2, 3, 4, 5, 6}


def _main_exhaustions(
    plan_rows: tuple[tuple[datetime, int], ...], created: datetime
) -> list[datetime]:
    totals: dict[int, int] = defaultdict(int)
    out = []
    for t, amount in plan_rows:
        cycle = (t - created) // timedelta(days=30)
        before = totals[cycle]
        totals[cycle] += amount
        if before < GRANT <= totals[cycle]:
            out.append(t)
    return out


def test_pair5_following_newcomer_starts_after_a_main_exhaustion() -> None:
    lo = W0 + timedelta(days=5)
    hi = WORLD_END - timedelta(days=20)
    followed = 0
    for i in range(60):
        twin = adds_relative(random.Random(f"e{i}"), _k(i))
        got = aged_expansion(random.Random(f"e{i}"), random.Random("m"), 1.0, _k(i))
        assert got.consume[0] == twin.consume[0]  # the main account is the twin's
        exhaustions = _main_exhaustions(got.consume[0], got.created[0])
        near = [e for e in exhaustions if lo <= e < hi]
        for j in range(1, got.size):
            if not near:
                assert got.created[j] == twin.created[j]
                continue
            nearest = min(near, key=lambda e: abs(e - twin.created[j]))
            assert 60 <= (got.created[j] - nearest) // MINUTE <= 1440
            followed += 1
    assert followed > 80


def test_pair5_main_account_reaches_the_cap_with_two_weeks_left() -> None:
    reached = total = 0
    for i in range(60):
        plan = adds_relative(random.Random(f"c{i}"), _k(i))
        for cycle in cycle_frames(plan.created[0], plan.offsets[0]):
            if cycle.claim is None or cycle.limit - cycle.claim < timedelta(days=14):
                continue
            total += 1
            spent = sum(a for t, a in plan.consume[0] if cycle.claim <= t < cycle.limit)
            reached += spent == GRANT
    assert total > 100
    assert reached == total


def test_pair6_full_mechanism_makes_blocks() -> None:
    sessions = 0
    for i in range(40):
        plan = serial_operator(random.Random(f"q{i}"), random.Random("m"), 1.0, _k(i))
        owners_by_slot: dict[tuple[int, int], list[tuple[datetime, int]]] = defaultdict(list)
        for a, rows in enumerate(plan.consume):
            for t, _ in rows:
                owners_by_slot[(t.toordinal(), t.hour)].append((t, a))
        for events in owners_by_slot.values():
            owners = [a for _, a in sorted(events)]
            joiners = len(set(owners))
            if joiners >= 2:
                switches = sum(x != y for x, y in zip(owners, owners[1:], strict=False))
                assert switches == joiners - 1
                sessions += 1
    assert sessions > 50


def test_cycle_frames_known_answers() -> None:
    offsets = (60, 120, 180, 240)
    fresh = cycle_frames(W0 + timedelta(days=1), offsets)
    assert [f.index for f in fresh] == [0, 1, 2]
    assert fresh[0].claim == W0 + timedelta(days=1, minutes=60)
    assert fresh[2].end == W0 + timedelta(days=91)
    aged = cycle_frames(W0 - timedelta(days=45), offsets)
    assert [f.index for f in aged] == [1, 2, 3, 4]
    assert aged[0].start == W0 - timedelta(days=15)
    assert aged[0].claim == W0 + timedelta(minutes=60)  # the later of the cycle start and W0
    assert aged[1].claim == W0 + timedelta(days=15, minutes=120)
    late = cycle_frames(W0 + timedelta(days=89, hours=23, minutes=50), offsets)
    assert len(late) == 1
    assert late[0].claim is None  # the claim would fall after WORLD_END
    # a cycle that ends 10 min after W0 can't hold the claim, which would fall at W0 + 60 min
    edge = cycle_frames(W0 - timedelta(days=59, hours=23, minutes=50), offsets)
    assert [f.index for f in edge] == [1, 2, 3, 4]
    assert edge[0].end == W0 + timedelta(minutes=10)
    assert edge[0].claim is None
    assert edge[1].claim == W0 + timedelta(minutes=10 + 120)


# --- T couplings (SPEC §5.1(9)) -------------------------------------------------------------------


def _non_consume(g: GroupDraw) -> list[tuple[str, datetime, EventType, int | None]]:
    return sorted(
        (e.account_id, e.timestamp, e.event_type, e.credits_amount)
        for e in g.events
        if e.event_type is not EventType.CONSUME_CREDITS
    )


def _consume_events(g: GroupDraw) -> list[GenEvent]:
    return [e for e in g.events if e.event_type is EventType.CONSUME_CREDITS]


def _hour_counts(g: GroupDraw) -> Counter[tuple[str, int, int]]:
    return Counter(
        (e.account_id, e.timestamp.toordinal(), e.timestamp.hour) for e in _consume_events(g)
    )


def _grants(g: GroupDraw) -> list[tuple[str, str, datetime, datetime, int]]:
    return sorted(
        (c.cycle_id, c.account_id, c.cycle_start, c.cycle_end, c.credits_granted) for c in g.cycles
    )


@pytest.mark.parametrize("pair", T_PAIRS)
def test_t_pair_couples_the_twin_and_changes_only_the_named_part(pair: int) -> None:
    abusive, _ = PAIRS[pair]
    full = generate_groups(abusive, 60, SEED, 1.0)
    none = generate_groups(abusive, 60, SEED, 0.0)
    changed = 0
    for one, zero in zip(full, none, strict=True):
        assert one.group_id == zero.group_id
        assert one.accounts == zero.accounts
        assert one.linkage == zero.linkage
        assert _grants(one) == _grants(zero)
        assert _non_consume(one) == _non_consume(zero)  # claims, signups, verifications, logins
        assert _hour_counts(one) == _hour_counts(zero)
        days_one = {(e.account_id, e.timestamp.toordinal()) for e in _consume_events(one)}
        days_zero = {(e.account_id, e.timestamp.toordinal()) for e in _consume_events(zero)}
        assert days_one == days_zero
        rows_one = [(e.account_id, e.timestamp, e.credits_amount) for e in _consume_events(one)]
        rows_zero = [(e.account_id, e.timestamp, e.credits_amount) for e in _consume_events(zero)]
        if pair in (2, 6):  # only the minutes move; each account keeps its amounts in order
            assert _amounts_by_account(rows_one) == _amounts_by_account(rows_zero)
            changed += [t for _, t, _ in rows_one] != [t for _, t, _ in rows_zero]
        elif pair == 4:  # only the amounts change
            assert [(a, t) for a, t, _ in rows_one] == [(a, t) for a, t, _ in rows_zero]
            changed += [c for _, _, c in rows_one] != [c for _, _, c in rows_zero]
        else:  # pair 3: amounts, and the second event's minute through the template's gap
            first_one = _session_starts(rows_one)
            assert first_one == _session_starts(rows_zero)
            changed += [c for _, _, c in rows_one] != [c for _, _, c in rows_zero]
    assert changed > len(full) // 2


def _amounts_by_account(
    rows: list[tuple[str, datetime, int | None]],
) -> dict[str, list[int | None]]:
    out: dict[str, list[int | None]] = defaultdict(list)
    for a, _, c in sorted(rows, key=lambda r: (r[0], r[1])):
        out[a].append(c)
    return out


def _session_starts(
    rows: list[tuple[str, datetime, int | None]],
) -> dict[tuple[str, int, int], datetime]:
    starts: dict[tuple[str, int, int], datetime] = {}
    for a, t, _ in sorted(rows, key=lambda r: r[1]):
        starts.setdefault((a, t.toordinal(), t.hour), t)
    return starts


@pytest.mark.parametrize("family", [f for p in T_PAIRS for f in PAIRS[p]])
@pytest.mark.parametrize("s", [0.0, 1.0])
def test_t_accounts_stay_under_the_rate_limits(family: str, s: float) -> None:
    for g in generate_groups(family, 40, SEED, s):
        cycles: dict[str, list[GenCycle]] = {c.account_id: [] for c in g.cycles}
        for c in g.cycles:
            cycles[c.account_id].append(c)
        hours: Counter[tuple[str, int, int]] = Counter()
        days: Counter[tuple[str, int]] = Counter()
        in_cycle: Counter[tuple[str, str]] = Counter()
        credits: Counter[tuple[str, str]] = Counter()
        for e in _consume_events(g):
            assert e.credits_amount is not None
            hours[(e.account_id, e.timestamp.toordinal(), e.timestamp.hour)] += e.credits_amount
            days[(e.account_id, e.timestamp.toordinal())] += e.credits_amount
            (cycle,) = [
                c for c in cycles[e.account_id] if c.cycle_start <= e.timestamp < c.cycle_end
            ]
            in_cycle[(e.account_id, cycle.cycle_id)] += 1
            credits[(e.account_id, cycle.cycle_id)] += e.credits_amount
        assert max(hours.values(), default=0) <= 8
        assert max(days.values(), default=0) <= 20
        assert max(in_cycle.values(), default=0) <= 6
        assert max(credits.values(), default=0) <= 24
        assert all(a.created_at < W0 + timedelta(days=10) for a in g.accounts)


# --- Mechanical validity on a standard world -----------------------------------------------------


def test_timestamps_are_whole_minutes_inside_the_world(standard: GeneratedWorld) -> None:
    stamps = (
        [a.created_at for a in standard.accounts]
        + [e.timestamp for e in standard.events]
        + [x.timestamp for x in standard.linkage]
        + [c.cycle_start for c in standard.cycles]
    )
    assert all(t.second == 0 and t.microsecond == 0 and t.tzinfo is not None for t in stamps)
    assert all(e.timestamp < WORLD_END for e in standard.events)
    assert all(x.timestamp < WORLD_END for x in standard.linkage)
    assert all(a.created_at < WORLD_END for a in standard.accounts)


def test_nothing_precedes_creation_and_only_signup_precedes_the_window(
    standard: GeneratedWorld,
) -> None:
    created = {a.account_id: a.created_at for a in standard.accounts}
    for e in standard.events:
        assert e.timestamp >= created[e.account_id]
        if e.timestamp < W0:
            assert e.event_type is EventType.SIGNUP
    assert all(x.timestamp >= created[x.account_id] for x in standard.linkage)
    # Linkage is observed inside the window: the vendored clustering drops observations older
    # than 365 days, which would split an aged account from its group.
    assert all(x.timestamp >= W0 for x in standard.linkage)
    signups = Counter(e.account_id for e in standard.events if e.event_type is EventType.SIGNUP)
    assert set(signups.values()) == {1}
    assert all(
        e.timestamp == created[e.account_id]
        for e in standard.events
        if e.event_type is EventType.SIGNUP
    )


def test_cycles_match_events(standard: GeneratedWorld) -> None:
    by_account: dict[str, list[GenCycle]] = defaultdict(list)
    for c in standard.cycles:
        by_account[c.account_id].append(c)
    consumed: Counter[str] = Counter()
    claims: dict[str, datetime] = {}
    claim_count: Counter[str] = Counter()
    for e in standard.events:
        if e.event_type is EventType.CLAIM_CREDITS:
            assert e.credits_amount == GRANT
            (cycle,) = [
                c for c in by_account[e.account_id] if c.cycle_start <= e.timestamp < c.cycle_end
            ]
            claims[cycle.cycle_id] = e.timestamp
            claim_count[cycle.cycle_id] += 1
    for e in standard.events:
        if e.event_type is EventType.CONSUME_CREDITS:
            assert e.credits_amount is not None and e.credits_amount >= 1
            (cycle,) = [
                c for c in by_account[e.account_id] if c.cycle_start <= e.timestamp < c.cycle_end
            ]
            assert cycle.cycle_id in claims and e.timestamp >= claims[cycle.cycle_id]
            consumed[cycle.cycle_id] += e.credits_amount
    assert set(claim_count.values()) == {1}
    for c in standard.cycles:
        assert c.credits_granted == GRANT
        assert c.credits_consumed == consumed[c.cycle_id] <= GRANT
        assert c.cycle_end - c.cycle_start == timedelta(days=30)
        assert c.cycle_start < WORLD_END and c.cycle_end > W0
        assert c.credits_expired == (GRANT - c.credits_consumed if c.cycle_end <= WORLD_END else 0)


def test_ids_are_unique_and_events_are_numbered_in_time_order(standard: GeneratedWorld) -> None:
    for rows in (
        [a.account_id for a in standard.accounts],
        [e.event_id for e in standard.events],
        [x.observation_id for x in standard.linkage],
        [c.cycle_id for c in standard.cycles],
    ):
        assert len(rows) == len(set(rows))
    last: dict[str, tuple[datetime, str]] = {}
    for e in standard.events:  # world order is (account, timestamp, event id)
        previous = last.get(e.account_id)
        if previous:
            assert previous[0] <= e.timestamp and previous[1] < e.event_id
        last[e.account_id] = (e.timestamp, e.event_id)
    assert [a.account_id for a in standard.accounts] == sorted(
        a.account_id for a in standard.accounts
    )
    assert [c.cluster_id for c in standard.labels] == sorted(c.cluster_id for c in standard.labels)


def test_groups_have_two_to_six_accounts_and_one_shared_edge_value(
    standard: GeneratedWorld,
) -> None:
    members: dict[str, list[str]] = defaultdict(list)
    for s in standard.split_groups:
        members[s.split_group_id].append(s.account_id)
    assert {len(v) for v in members.values()} == {2, 3, 4, 5, 6}
    holders: dict[str, set[str]] = defaultdict(set)
    kinds: dict[str, set[tuple[SignalType, Reliability]]] = defaultdict(set)
    for x in standard.linkage:
        holders[x.signal_value_hash].add(x.account_id)
        kinds[x.signal_value_hash].add((x.signal_type, x.reliability))
    owner = {a: g for g, accts in members.items() for a in accts}
    values: dict[str, list[str]] = defaultdict(list)
    for value, accounts in holders.items():
        assert len({owner[a] for a in accounts}) == 1  # no value is shared across groups
        assert len(kinds[value]) == 1
        values[owner[next(iter(accounts))]].append(value)
    for group, accts in members.items():
        edge = [v for v in values[group] if next(iter(kinds[v]))[0] in EDGE_SIGNALS]
        assert len(edge) == 1  # exactly one edge-forming value, held by every member
        assert holders[edge[0]] == set(accts)
        assert next(iter(kinds[edge[0]]))[1] in (Reliability.MEDIUM, Reliability.HIGH)
        for v in set(values[group]) - set(edge):  # context-only: LOW, on every member
            assert holders[v] == set(accts)
            assert next(iter(kinds[v]))[1] is Reliability.LOW


def test_account_nuisance_marginals(standard: GeneratedWorld) -> None:
    n = len(standard.accounts)
    domains = Counter(a.email_domain_type.value for a in standard.accounts)
    assert domains["consumer"] / n == pytest.approx(0.70, abs=0.04)
    levels = Counter(a.verification_level.value for a in standard.accounts)
    assert levels["email"] / n == pytest.approx(0.40, abs=0.04)
    assert levels["none"] / n == pytest.approx(0.10, abs=0.03)
    verified = Counter(
        e.account_id for e in standard.events if e.event_type is EventType.VERIFICATION
    )
    assert set(verified.values()) == {1}
    with_level = {a.account_id for a in standard.accounts if a.verification_level.value != "none"}
    assert set(verified) == with_level


# --- Opacity (SPEC §5.1(7)) -----------------------------------------------------------------------


def test_inference_fields_carry_no_family_group_seed_or_s() -> None:
    seed, s = 424242, 0.3125
    world = generate_world(WorldSpec(seed, s, "standard"))
    fields: set[str] = set()
    for a in world.accounts:
        fields |= {a.account_id, a.email_domain_type.value, a.verification_level.value}
    for e in world.events:
        fields |= {e.event_id, e.account_id, e.event_type.value}
        if e.endpoint_cost_tier is not None:
            fields.add(e.endpoint_cost_tier.value)
        fields.add(str(e.credits_amount))
    for x in world.linkage:
        fields |= {x.observation_id, x.account_id, x.signal_value_hash, x.signal_type.value}
        fields.add(x.reliability.value)
    for c in world.cycles:
        fields |= {c.cycle_id, c.account_id}
    haystack = "\n".join(sorted(fields))
    tokens = {*FAMILIES, str(seed), str(s), *(label.cluster_id for label in world.labels)}
    assert not [t for t in tokens if t in haystack]
    manifest = world_manifest(world, WorldSpec(seed, s, "standard"))
    assert manifest["world_version"] == GENERATOR_VERSION


# --- The timeline contract ------------------------------------------------------------------------


def test_group_timeline_equals_load_timeline_on_a_persisted_world(tmp_path: Path) -> None:
    groups = [draw_group(f, i, SEED, 1.0) for f in FAMILIES for i in range(3)]
    world = assemble_world(groups)
    stores = Stores.under(tmp_path)
    manifest = world_manifest(world, WorldSpec(SEED, 1.0, "standard"))
    assert (manifest["tier"], manifest["reported"]) == ("dev", False)
    persist(world, manifest, stores)
    verified = verify_pair(stores)
    assert verified.tier == "dev"
    assert verified.world_version == GENERATOR_VERSION
    with closing(sqlite3.connect(f"file:{stores.inference_db}?mode=ro", uri=True)) as conn:
        for g in groups:
            members = [a.account_id for a in g.accounts]
            assert group_timeline(g) == load_timeline(conn, members)
    sample = group_timeline(groups[0])
    assert sample.events and {e.event_type for e in sample.events} >= {"CLAIM_CREDITS"}
    assert any(e.event_type == "CONSUME_CREDITS" for g in groups for e in group_timeline(g).events)
    assert sample.accounts[0].email_domain_type.isupper()


def test_group_timeline_orders_events_and_aliases_by_sorted_ids() -> None:
    g = draw_group("classroom_lab", 0, SEED, 0.0)
    tl = group_timeline(g)
    ids = sorted(a.account_id for a in g.accounts)
    assert [a.alias for a in tl.accounts] == [f"A{i + 1}" for i in range(len(ids))]
    stamps = [e.at for e in tl.events]
    assert stamps == sorted(stamps)
    assert all(tl.window_start <= t < tl.window_end for t in stamps)
    assert len(tl.events) == sum(1 for e in g.events if e.timestamp >= W0)
