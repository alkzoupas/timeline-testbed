"""Known-answer tests for the six oracle statistics (SPEC §5.2). Each timeline is built by hand
over whole minutes inside the default window, and the expected value is worked out in a comment."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from app.services.generator import WORLD_END

from testbed.eval.oracle import PAIRS, oracle
from testbed.pipeline.timeline import WINDOW_START, Account, Event, UnitTimeline


def at(days: int, hours: int = 0, minutes: int = 0) -> datetime:
    """An instant `days` d, `hours` h, `minutes` min after the window start (a Tuesday, 00:00Z)."""
    return WINDOW_START + timedelta(days=days, hours=hours, minutes=minutes)


def consume(alias: str, when: datetime, amount: int) -> Event:
    return Event(when, alias, "CONSUME_CREDITS", amount)


def claim(alias: str, when: datetime) -> Event:
    return Event(when, alias, "CLAIM_CREDITS", 25)


def timeline(created: dict[str, datetime], events: list[Event]) -> UnitTimeline:
    accounts = tuple(Account(alias, when, "CONSUMER", "EMAIL") for alias, when in created.items())
    return UnitTimeline(WINDOW_START, WORLD_END, accounts, (), tuple(events))


# Pair 1 ---------------------------------------------------------------------------------------


def rotator(*, a2_exhausts: bool) -> UnitTimeline:
    """Four accounts created 2 d apart. A1 exhausts at day 3 10:00, A2 (if it does) at day 5 08:00,
    A3 at day 7 09:00; A4 never exhausts. Claims: A2 12 h after A1's exhaustion, A3 6 h after A2's,
    A4 30 h before A3's (the gap is absolute)."""
    created = {"A1": at(0), "A2": at(2), "A3": at(4), "A4": at(6)}
    events = [
        claim("A1", at(0, 2)),
        consume("A1", at(1, 10), 10),
        consume("A1", at(3, 10), 15),  # 10 + 15 = 25: A1's first-cycle exhaustion
        consume("A1", at(31, 9), 25),  # A1's second cycle exhausts too, but pair 1 uses cycle 0
        claim("A2", at(3, 22)),
        consume("A2", at(4, 8), 20),
        consume("A2", at(5, 8), 5 if a2_exhausts else 3),  # 25, or 23 and no exhaustion
        consume("A2", at(33, 8), 25),  # cycle 1 exhaustion must not rescue A2's first cycle
        claim("A3", at(5, 14)),
        consume("A3", at(7, 9), 25),
        claim("A4", at(6, 3)),
    ]
    return timeline(created, events)


def test_pair1_median_of_absolute_gaps() -> None:
    # gaps: A1->A2 |day 3 22:00 - day 3 10:00| = 12 h; A2->A3 |day 5 14:00 - day 5 08:00| = 6 h;
    # A3->A4 |day 6 03:00 - day 7 09:00| = 30 h. median(12, 6, 30) = 12, negated.
    assert oracle(1, rotator(a2_exhausts=True)) == pytest.approx(-12.0)


def test_pair1_skips_an_account_that_never_exhausts_its_first_cycle() -> None:
    # A2 reaches only 23 in cycle 0, so A2->A3 is skipped: median(12, 30) = 21, negated.
    assert oracle(1, rotator(a2_exhausts=False)) == pytest.approx(-21.0)


def test_pair1_orders_by_creation_not_alias() -> None:
    # The same hand-off with the aliases swapped: A2 is created first, so it is the earlier account.
    # gap |day 3 22:00 - day 3 10:00| = 12 h.
    tl = timeline(
        {"A1": at(2), "A2": at(0)},
        [consume("A2", at(3, 10), 25), claim("A1", at(3, 22))],
    )
    assert oracle(1, tl) == pytest.approx(-12.0)


def test_pair1_none_without_any_exhaustion() -> None:
    tl = timeline(
        {"A1": at(0), "A2": at(2)},
        [claim("A1", at(0, 1)), consume("A1", at(1), 24), claim("A2", at(2, 1))],
    )
    assert oracle(1, tl) is None


def test_pair1_none_for_a_single_account() -> None:
    assert oracle(1, timeline({"A1": at(0)}, [consume("A1", at(1), 25)])) is None


# Pair 2 ---------------------------------------------------------------------------------------


def lab_session(day: int, order: list[str], hour: int = 10) -> list[Event]:
    """One session: accounts start 5 min apart in `order`, each with a second event 3 min after."""
    events = []
    for place, alias in enumerate(order):
        events.append(consume(alias, at(day, hour, 5 * place), 2))
        events.append(consume(alias, at(day, hour, 5 * place + 3), 1))
    return events


LAB = {"A1": at(0), "A2": at(0), "A3": at(0), "A4": at(0)}


def test_pair2_perfectly_ordered_group_is_one() -> None:
    # Every pair co-attends 4 sessions and the same account is always first:
    # 3 pairs x |4 - 0| = 12 over 3 pairs x 4 = 12. A4 attends once, so its pairs are excluded.
    events = [e for day in range(1, 5) for e in lab_session(day, ["A1", "A2", "A3"])]
    events += lab_session(5, ["A4", "A1"])
    # A1/A4 co-attend once only (n < 2), so they add nothing.
    assert oracle(2, timeline(LAB, events)) == pytest.approx(1.0)


def test_pair2_balanced_group_is_zero() -> None:
    # Orders alternate A1 A2 A3 / A3 A2 A1, so each pair is first 2 times each way: |2 - 2| = 0.
    # A4 joins session 1 only; if its n = 1 pairs counted they would push the value above 0.
    events = []
    for day in range(1, 5):
        order = ["A1", "A2", "A3"] if day % 2 else ["A3", "A2", "A1"]
        if day == 1:
            order = ["A4", *order]
        events += lab_session(day, order)
    assert oracle(2, timeline(LAB, events)) == pytest.approx(0.0)


def test_pair2_partial_concordance() -> None:
    # Orders 123, 123, 213: (A1,A2) first 2-1 -> 1; (A1,A3) 3-0 -> 3; (A2,A3) 3-0 -> 3.
    # (1 + 3 + 3) / (3 + 3 + 3) = 7/9.
    events = lab_session(1, ["A1", "A2", "A3"])
    events += lab_session(2, ["A1", "A2", "A3"])
    events += lab_session(3, ["A2", "A1", "A3"])
    assert oracle(2, timeline(LAB, events)) == pytest.approx(7 / 9)


def test_pair2_none_without_a_repeated_pair() -> None:
    events = lab_session(1, ["A1", "A2"]) + lab_session(2, ["A1", "A3"])
    assert oracle(2, timeline(LAB, events)) is None


# Pair 3 ---------------------------------------------------------------------------------------


def two_event_session(alias: str, day: int, first: int, second: int, gap: int) -> list[Event]:
    return [consume(alias, at(day, 10), first), consume(alias, at(day, 10, gap), second)]


def test_pair3_fraction_of_identical_cross_account_pairs() -> None:
    # A1: (2,3,10 min), (1,4,5), (1,4,5); A2: (2,3,10), (4,4,15), (1,1,3). The two identical A1
    # sessions share an account, so they don't count. 3 x 3 = 9 cross pairs, 1 identical: 1/9.
    events = []
    for day, spec in enumerate([(2, 3, 10), (1, 4, 5), (1, 4, 5)], start=1):
        events += two_event_session("A1", day, *spec)
    for day, spec in enumerate([(2, 3, 10), (4, 4, 15), (1, 1, 3)], start=1):
        events += two_event_session("A2", day, *spec)
    # Sessions with 1 or 3 events are ignored, even when their first two events would match.
    events += [consume("A1", at(8, 10), 2)]
    events += [*two_event_session("A2", 9, 2, 3, 10), consume("A2", at(9, 10, 20), 1)]
    tl = timeline({"A1": at(0), "A2": at(0)}, events)
    assert oracle(3, tl) == pytest.approx(1 / 9)


def test_pair3_full_template_is_one() -> None:
    # Both accounts replay (2, 3, 10 min) in 2 sessions each: 4 cross pairs, all identical.
    events = []
    for alias in ("A1", "A2"):
        for day in (1, 2):
            events += two_event_session(alias, day, 2, 3, 10)
    assert oracle(3, timeline({"A1": at(0), "A2": at(0)}, events)) == pytest.approx(1.0)


def test_pair3_gap_is_part_of_the_key() -> None:
    # Same amounts, different gaps (10 vs 11 min): no match, 0 of 1 pair.
    events = two_event_session("A1", 1, 2, 3, 10) + two_event_session("A2", 1, 2, 3, 11)
    assert oracle(3, timeline({"A1": at(0), "A2": at(0)}, events)) == pytest.approx(0.0)


def test_pair3_none_when_one_account_has_all_the_sessions() -> None:
    events = two_event_session("A1", 1, 2, 3, 10) + two_event_session("A1", 2, 2, 3, 10)
    assert oracle(3, timeline({"A1": at(0), "A2": at(0)}, events)) is None


# Pair 4 ---------------------------------------------------------------------------------------


def budget(alias: str, hour: int, amounts: dict[int, int]) -> list[Event]:
    """One session per listed day at `hour`: two events of the day's amount."""
    events = []
    for day, amount in amounts.items():
        events += [consume(alias, at(day, hour), amount), consume(alias, at(day, hour, 10), amount)]
    return events


def test_pair4_opposite_deviations_give_zero() -> None:
    # A: 1,3,1,3 (mean 2) -> z = -1,+1,-1,+1; B: 3,1,3,1 -> z = +1,-1,+1,-1. Each day sums to 0,
    # so the statistic is 0 / 8 = 0 (the score is -0.0 negated, i.e. 0).
    a = budget("A1", 9, {1: 1, 2: 3, 3: 1, 4: 3})
    b = budget("A2", 14, {1: 3, 2: 1, 3: 3, 4: 1})
    assert oracle(4, timeline({"A1": at(0), "A2": at(0)}, a + b)) == pytest.approx(0.0)


def test_pair4_identical_deviations_give_minus_two() -> None:
    # Both accounts have z = -1,+1,-1,+1: each day (sum z)^2 = 4 and sum z^2 = 2, so 16 / 8 = 2,
    # negated.
    a = budget("A1", 9, {1: 1, 2: 3, 3: 1, 4: 3})
    b = budget("A2", 14, {1: 1, 2: 3, 3: 1, 4: 3})
    assert oracle(4, timeline({"A1": at(0), "A2": at(0)}, a + b)) == pytest.approx(-2.0)


def test_pair4_account_mean_covers_all_its_active_days() -> None:
    # A1 is also active alone on day 3 (amount 4): mean(1, 3, 4) = 8/3, z = -5/3, 1/3 on the
    # shared days. A2: 3, 1 (mean 2) -> z = +1, -1.
    # Day 1: (-5/3 + 1)^2 = 4/9, squares 25/9 + 1 = 34/9. Day 2: (1/3 - 1)^2 = 4/9, squares
    # 1/9 + 1 = 10/9. Statistic (8/9) / (44/9) = 2/11, negated. Day 3 has one active account.
    a = budget("A1", 9, {1: 1, 2: 3, 3: 4})
    b = budget("A2", 14, {1: 3, 2: 1})
    assert oracle(4, timeline({"A1": at(0), "A2": at(0)}, a + b)) == pytest.approx(-2 / 11)


def test_pair4_none_with_one_active_account_per_day() -> None:
    a = budget("A1", 9, {1: 1, 2: 3})
    b = budget("A2", 14, {3: 3, 4: 1})
    assert oracle(4, timeline({"A1": at(0), "A2": at(0)}, a + b)) is None


# Pair 5 ---------------------------------------------------------------------------------------


def expansion() -> tuple[dict[str, datetime], list[Event]]:
    """A main account created exactly 10 cycles before the window start, so its cycle 10 is days
    0-29 and cycle 11 is days 30-59. It exhausts cycle 10 at day 5 10:00 and cycle 11 at day 35
    10:00. A2 is created 3 h after the first exhaustion and has 4 sessions, 2 on main's (day, hour):
    day 33 10:00 and day 35 10:00 match; day 34 10:00 (main inactive that day) and day 33 15:00
    (main active that day, other hour) do not."""
    created = {"A1": at(-300), "A2": at(5, 13)}
    events = [
        consume("A1", at(2, 10), 10),
        consume("A1", at(5, 10), 15),  # cycle 10: 10 + 15 = 25
        consume("A1", at(32, 10), 5),
        consume("A1", at(33, 10), 10),
        consume("A1", at(35, 10), 10),  # cycle 11: 5 + 10 + 10 = 25
        consume("A2", at(33, 10), 2),
        consume("A2", at(34, 10), 2),
        consume("A2", at(35, 10), 2),
        consume("A2", at(33, 15), 2),
    ]
    return created, events


def test_pair5_overlap_plus_timing_bonus() -> None:
    # A2: 2 of 4 sessions overlap main's = 0.5; created 3 h after main's exhaustion = +1 -> 1.5.
    created, events = expansion()
    assert oracle(5, timeline(created, events)) == pytest.approx(1.5)


def test_pair5_mean_over_newcomers() -> None:
    # A3 is created at day 20 (15 d after one exhaustion, 15 d before the next) -> no bonus, and
    # its only session (day 36 10:00) is on a day main is inactive -> 0. Mean (1.5 + 0) / 2.
    created, events = expansion()
    created["A3"] = at(20)
    events.append(consume("A3", at(36, 10), 2))
    assert oracle(5, timeline(created, events)) == pytest.approx(0.75)


def test_pair5_bonus_needs_the_newcomer_after_the_exhaustion_within_a_day() -> None:
    # Created 1 h before the exhaustion, and 25 h after it: neither is "within 24 h after".
    # Both newcomers' only session misses main's, so each scores 0.
    created, events = expansion()
    created["A2"] = at(5, 9)
    created["A3"] = at(6, 11)
    events.append(consume("A3", at(40, 10), 2))
    events = [e for e in events if e.alias != "A2"] + [consume("A2", at(40, 11), 2)]
    assert oracle(5, timeline(created, events)) == pytest.approx(0.0)


def test_pair5_none_without_newcomer_sessions() -> None:
    created, events = expansion()
    assert oracle(5, timeline(created, [e for e in events if e.alias == "A1"])) is None


# Pair 6 ---------------------------------------------------------------------------------------


def test_pair6_blocked_and_interleaved_sessions() -> None:
    # Day 1 20:xx: A1 A1 A2 A2 -> 1 switch, excess 1 - (2 - 1) = 0.
    # Day 2 20:xx: A1 A2 A1 A2 -> 3 switches, excess 3 - 1 = 2.
    # Day 3 21:xx has only A1: not a shared session. Mean excess (0 + 2) / 2 = 1, negated.
    events = [
        consume("A1", at(1, 20, 0), 2),
        consume("A1", at(1, 20, 10), 2),
        consume("A2", at(1, 20, 20), 2),
        consume("A2", at(1, 20, 30), 2),
        consume("A1", at(2, 20, 0), 2),
        consume("A2", at(2, 20, 10), 2),
        consume("A1", at(2, 20, 20), 2),
        consume("A2", at(2, 20, 30), 2),
        consume("A1", at(3, 21, 0), 2),
        consume("A1", at(3, 21, 5), 2),
    ]
    assert oracle(6, timeline({"A1": at(0), "A2": at(0)}, events)) == pytest.approx(-1.0)


def test_pair6_three_accounts_use_accounts_minus_one_as_the_minimum() -> None:
    # A1 A2 A3 A1: 3 switches, 3 accounts -> excess 3 - 2 = 1, negated.
    events = [
        consume("A1", at(1, 20, 0), 1),
        consume("A2", at(1, 20, 10), 1),
        consume("A3", at(1, 20, 20), 1),
        consume("A1", at(1, 20, 30), 1),
    ]
    created = {"A1": at(0), "A2": at(0), "A3": at(0)}
    assert oracle(6, timeline(created, events)) == pytest.approx(-1.0)


def test_pair6_none_without_a_shared_session() -> None:
    events = [consume("A1", at(1, 20), 1), consume("A2", at(1, 21), 1)]
    assert oracle(6, timeline({"A1": at(0), "A2": at(0)}, events)) is None


# Every pair -----------------------------------------------------------------------------------


@pytest.mark.parametrize("pair", PAIRS)
def test_empty_timelines_never_raise(pair: int) -> None:
    created = {"A1": at(0), "A2": at(1), "A3": at(2)}
    for tl in (timeline(created, []), timeline({}, [])):
        result = oracle(pair, tl)
        assert result is None or isinstance(result, float)


@pytest.mark.parametrize("pair", PAIRS)
def test_non_consume_events_define_nothing(pair: int) -> None:
    events = [Event(at(1, 10), "A1", "LOGIN", None), Event(at(1, 10, 5), "A2", "SIGNUP", None)]
    result = oracle(pair, timeline({"A1": at(0), "A2": at(0)}, events))
    assert result is None or isinstance(result, float)


@pytest.mark.parametrize("pair", PAIRS)
def test_consume_events_without_amounts_do_not_raise(pair: int) -> None:
    events = [
        Event(at(1, 10), "A1", "CONSUME_CREDITS", None),
        Event(at(1, 10, 5), "A2", "CONSUME_CREDITS", None),
    ]
    result = oracle(pair, timeline({"A1": at(0), "A2": at(0)}, events))
    assert result is None or isinstance(result, float)


def test_unknown_pair_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match="unknown pair"):
        oracle(7, timeline({}, []))


def test_pairs_are_one_to_six() -> None:
    assert PAIRS == (1, 2, 3, 4, 5, 6)
