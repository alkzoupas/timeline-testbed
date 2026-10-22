"""Known-answer tests for the raw-event render and its parser (SPEC §6.2)."""

from __future__ import annotations

import random
import re
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
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
    GenAccount,
    GeneratedWorld,
    GenEvent,
    GenLabel,
    GenLinkage,
    GenSplit,
    build_manifest,
)

from testbed.pipeline.render import RENDER_VERSION, parse, render, render_unit
from testbed.pipeline.timeline import (
    WINDOW_START,
    Account,
    Event,
    Shared,
    UnitTimeline,
    load_timeline,
)
from testbed.pipeline.tokens import VENV_PYTHON, count_tokens, tokenizer_file
from testbed.world.persist import Stores, persist

WINDOW_END = WINDOW_START + timedelta(days=90)

# Words that would name a family, a regime or a label: none may reach the render.
BANNED = (
    "rotator household parallel farm classroom template operator office single budget heavy "
    "aged expansion relative serial roommates family abusive legit"
).split()


def _at(month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


EVENT_NAMES = [
    "SIGNUP",
    "CLAIM_CREDITS",
    "CONSUME_CREDITS",
    "LOGIN",
    "APPEAL",
    "VERIFICATION",
    "VERIFICATION_FAILED",
]


def _golden_timeline() -> UnitTimeline:
    """3 accounts (A1 aged), 4 linkage values (one context-only, one single-holder), and every
    event type with and without an amount, including a 0 and two events in one minute."""
    accounts = (
        Account("A1", _at(2, 10, 9, 12), "CONSUMER", "PHONE"),
        Account("A2", _at(3, 4, 22, 41), "DISPOSABLE", "NONE"),
        Account("A3", _at(3, 3, 0, 0), "ENTERPRISE", "ENHANCED"),
    )
    shared = (
        Shared("BEHAVIORAL_TEMPLATE", "MEDIUM", ("A2", "A3")),
        Shared("NETWORK_GROUP", "LOW", ("A1", "A2", "A3")),
        Shared("PAYMENT_TOKEN", "MEDIUM", ("A3",)),
        Shared("STABLE_DEVICE", "HIGH", ("A1", "A2")),
    )
    events = (
        Event(_at(3, 3, 0, 0), "A3", "SIGNUP", None),
        Event(_at(3, 3, 9, 30), "A1", "LOGIN", None),
        Event(_at(3, 3, 9, 30), "A3", "LOGIN", 5),
        Event(_at(3, 4, 22, 41), "A2", "SIGNUP", 0),
        Event(_at(3, 5, 8, 5), "A2", "CLAIM_CREDITS", 100),
        Event(_at(3, 5, 8, 6), "A1", "CLAIM_CREDITS", None),
        Event(_at(3, 14, 14, 3), "A1", "CONSUME_CREDITS", 2),
        Event(_at(3, 14, 14, 4), "A2", "CONSUME_CREDITS", None),
        Event(_at(3, 20, 3, 15), "A3", "VERIFICATION", None),
        Event(_at(3, 20, 3, 16), "A3", "VERIFICATION", 1),
        Event(_at(4, 10, 12, 0), "A2", "VERIFICATION_FAILED", None),
        Event(_at(4, 10, 12, 1), "A2", "VERIFICATION_FAILED", 3),
        Event(_at(5, 31, 23, 59), "A1", "APPEAL", None),
        Event(_at(5, 31, 23, 59), "A3", "APPEAL", 4),
    )
    return UnitTimeline(WINDOW_START, WINDOW_END, accounts, shared, events)


GOLDEN = """\
Window: starts d1 00:00 (Tuesday 2026-03-03 UTC), ends d91 00:00. Times are UTC.
Event types: signup, claim = claim credits, use = consume credits, login, appeal,
verify = passed verification, verify_fail = failed verification.
Accounts: alias, created, email domain, verification
A d-20 09:12 consumer phone
B d2 22:41 disposable none
C d1 00:00 enterprise enhanced
Shared values: signal, reliability, holders
behavior medium B C
network low A B C
payment medium C
device high A B
Events: d<day> header, then time, account, type, amount
d1
00:00 C signup
09:30 A login
09:30 C login 5
d2
22:41 B signup 0
d3
08:05 B claim 100
08:06 A claim
d12
14:03 A use 2
14:04 B use
d18
03:15 C verify
03:16 C verify 1
d39
12:00 B verify_fail
12:01 B verify_fail 3
d90
23:59 A appeal
23:59 C appeal 4"""


def _assert_label_free(text: str, hashes: tuple[str, ...] = ()) -> None:
    assert "acct-" not in text
    for value in hashes:
        assert value not in text
    words = set(re.findall(r"[a-z]+", text.lower()))
    assert not words & set(BANNED), sorted(words & set(BANNED))
    assert not re.search(r"A[0-9]", text), "an A1-style alias reached the render"


def _random_timeline(rng: random.Random) -> UnitTimeline:
    start = _at(3, 3) + timedelta(minutes=rng.choice([0, 0, 390, 1439]))
    end = start + timedelta(days=rng.randint(1, 120), minutes=rng.randint(0, 2000))
    n = rng.randint(1, 8)
    aliases = [f"A{i + 1}" for i in range(n)]
    accounts = tuple(
        Account(
            a,
            start + timedelta(minutes=rng.randint(-200_000, 100_000)),
            rng.choice(["CONSUMER", "DISPOSABLE", "EDUCATION", "ENTERPRISE"]),
            rng.choice(["NONE", "EMAIL", "PHONE", "ENHANCED"]),
        )
        for a in aliases
    )
    shared = tuple(
        Shared(
            rng.choice(
                [
                    "STABLE_DEVICE",
                    "PAYMENT_TOKEN",
                    "RECOVERY_CONTACT",
                    "NETWORK_GROUP",
                    "BEHAVIORAL_TEMPLATE",
                ]
            ),
            rng.choice(["LOW", "MEDIUM", "HIGH"]),
            tuple(rng.sample(aliases, rng.randint(1, n))),
        )
        for _ in range(rng.randint(0, 5))
    )
    events = tuple(
        Event(
            start + timedelta(minutes=rng.randint(-5000, 150_000)),
            rng.choice(aliases),
            rng.choice(EVENT_NAMES),
            rng.choice([None, 0, 1, 7, 250, -3]),
        )
        for _ in range(rng.randint(0, 60))
    )
    return UnitTimeline(start, end, accounts, shared, events)


def test_render_version() -> None:
    assert RENDER_VERSION == "raw-v1"


def test_render_matches_golden_text() -> None:
    assert render(_golden_timeline()) == GOLDEN


def test_parse_inverts_the_golden_render() -> None:
    tl = _golden_timeline()
    assert parse(GOLDEN) == tl
    assert parse(render(tl)) == tl


def test_golden_render_is_label_free() -> None:
    _assert_label_free(GOLDEN)


@pytest.mark.parametrize("seed", range(25))
def test_round_trip_on_random_whole_minute_timelines(seed: int) -> None:
    tl = _random_timeline(random.Random(seed))
    text = render(tl)
    assert parse(text) == tl
    assert render(parse(text)) == text
    _assert_label_free(text)


def test_empty_sections_round_trip() -> None:
    tl = replace(_golden_timeline(), shared=(), events=())
    assert parse(render(tl)) == tl


@pytest.mark.parametrize(
    "bad",
    [
        _at(3, 5, 8, 5) + timedelta(seconds=1),
        _at(3, 5, 8, 5) + timedelta(microseconds=1),
        _at(3, 5, 8, 5) + timedelta(seconds=59, microseconds=999_999),
    ],
)
def test_render_refuses_sub_minute_timestamps(bad: datetime) -> None:
    tl = _golden_timeline()
    with pytest.raises(ValueError, match="whole minute"):
        render(replace(tl, events=(Event(bad, "A1", "LOGIN", None),)))
    with pytest.raises(ValueError, match="whole minute"):
        render(replace(tl, accounts=(replace(tl.accounts[0], created_at=bad),)))
    with pytest.raises(ValueError, match="whole minute"):
        render(replace(tl, window_end=bad))


def test_render_refuses_a_non_utc_timestamp() -> None:
    tl = _golden_timeline()
    plus_one = datetime(2026, 3, 5, 8, 5, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match="not UTC"):
        render(replace(tl, events=(Event(plus_one, "A1", "LOGIN", None),)))
    with pytest.raises(ValueError, match="not UTC"):
        render(replace(tl, events=(Event(datetime(2026, 3, 5, 8, 5), "A1", "LOGIN", None),)))


def test_render_refuses_unknown_vocabulary() -> None:
    tl = _golden_timeline()
    with pytest.raises(ValueError, match="event type"):
        render(replace(tl, events=(Event(_at(3, 5), "A1", "REFUND", None),)))
    with pytest.raises(ValueError, match="signal type"):
        render(replace(tl, shared=(Shared("SIM_CARD", "HIGH", ("A1",)),)))


def test_parse_refuses_malformed_text() -> None:
    with pytest.raises(ValueError):
        parse(GOLDEN.replace("14:03 A use 2", "14:03 A use two"))
    with pytest.raises(ValueError, match="not a Monday"):
        parse(GOLDEN.replace("Tuesday", "Monday"))
    with pytest.raises(ValueError, match="missing a section"):
        parse(GOLDEN.split("\nShared values")[0])


def test_parse_refuses_an_event_line_before_any_day_header() -> None:
    with pytest.raises(ValueError, match="before any day header"):
        parse(GOLDEN.replace("\nd1\n00:00 C signup", "\n00:00 C signup"))


def test_parse_refuses_a_day_header_with_no_events_or_a_repeated_one() -> None:
    with pytest.raises(ValueError, match="empty or repeated"):
        parse(GOLDEN.replace("\nd2\n22:41", "\nd2\nd2\n22:41"))
    with pytest.raises(ValueError, match="empty or repeated"):
        parse(GOLDEN.replace("\nd2\n22:41", "\nd2\nd3\n22:41"))
    with pytest.raises(ValueError, match="no event line"):
        parse(GOLDEN + "\nd91")
    with pytest.raises(ValueError, match="empty or repeated"):
        parse(GOLDEN.replace("\n23:59 A appeal", "\nd90\n23:59 A appeal"))


def test_day_headers_cover_each_event_day_once_and_keep_event_order() -> None:
    tl = _golden_timeline()
    text = render(tl)
    event_lines = text.split("Events: ")[1].split("\n")[1:]
    headers = [line for line in event_lines if re.fullmatch(r"d-?[0-9]+", line)]
    assert headers == ["d1", "d2", "d3", "d12", "d18", "d39", "d90"]
    assert len(event_lines) == len(headers) + len(tl.events)  # one line per event, none merged


def test_events_out_of_day_order_get_a_header_per_run_and_round_trip() -> None:
    tl = replace(
        _golden_timeline(),
        events=(
            Event(_at(3, 5, 8, 5), "A1", "LOGIN", None),
            Event(_at(3, 3, 9, 0), "A2", "LOGIN", None),
            Event(_at(3, 5, 8, 6), "A3", "LOGIN", None),
            Event(_at(3, 5, 8, 7), "A3", "LOGIN", None),
            Event(_at(3, 2, 23, 59), "A3", "LOGIN", None),  # day 0, before the window
        ),
    )
    text = render(tl)
    assert text.split("\n")[-9:] == [
        "d3",
        "08:05 A login",
        "d1",
        "09:00 B login",
        "d3",
        "08:06 C login",
        "08:07 C login",
        "d0",
        "23:59 C login",
    ]
    assert parse(text) == tl


def test_render_never_contains_a_numbered_alias() -> None:
    assert not re.search(r"A[0-9]", GOLDEN)
    for seed in range(25):
        assert not re.search(r"A[0-9]", render(_random_timeline(random.Random(seed))))


def _accounts(n: int) -> tuple[Account, ...]:
    return tuple(Account(f"A{i}", _at(3, 3), "CONSUMER", "NONE") for i in range(1, n + 1))


def test_twenty_six_accounts_use_every_letter_and_round_trip() -> None:
    accounts = _accounts(26)
    events = tuple(Event(_at(3, 3, 1, i), a.alias, "LOGIN", None) for i, a in enumerate(accounts))
    tl = UnitTimeline(
        WINDOW_START, WINDOW_END, accounts, (Shared("NETWORK_GROUP", "LOW", ("A1", "A26")),), events
    )
    text = render(tl)
    letters = [line.split(" ")[0] for line in text.split("Accounts: ")[1].split("\n")[1:27]]
    assert letters == [chr(ord("A") + i) for i in range(26)]
    assert "network low A Z" in text
    assert parse(text) == tl


def test_render_refuses_more_than_26_accounts() -> None:
    tl = UnitTimeline(WINDOW_START, WINDOW_END, _accounts(27), (), ())
    with pytest.raises(ValueError, match="at most 26"):
        render(tl)


def test_persisted_world_round_trips(tmp_path: Path) -> None:
    hashes = ("hash-device-9f3a", "hash-network-77c1")

    def acct(i: str, created: datetime, domain: EmailDomainType, level: VerificationLevel):
        return GenAccount(f"acct-{i}", created, domain, level)

    def event(n: int, i: str, at: datetime, kind: EventType, amount: int | None) -> GenEvent:
        tier = None if amount is None else EndpointCostTier.LOW
        return GenEvent(f"ev-{n}", f"acct-{i}", at, kind, amount, tier)

    def link(n: int, i: str, kind: SignalType, value: str, rel: Reliability) -> GenLinkage:
        return GenLinkage(f"ob-{n}", f"acct-{i}", _at(3, 5, 1, 0), kind, value, rel)

    world = GeneratedWorld(
        accounts=[
            # Ids sort b < c < d, so the aliases A1..A3 follow that order, not creation order.
            acct("c", _at(3, 4, 22, 41), EmailDomainType.DISPOSABLE, VerificationLevel.NONE),
            acct("b", _at(2, 10, 9, 12), EmailDomainType.CONSUMER, VerificationLevel.PHONE),
            acct("d", _at(3, 3, 0, 0), EmailDomainType.ENTERPRISE, VerificationLevel.ENHANCED),
        ],
        events=[
            event(1, "d", _at(3, 3, 0, 0), EventType.SIGNUP, None),
            event(2, "b", _at(3, 3, 9, 30), EventType.LOGIN, None),
            event(3, "d", _at(3, 3, 9, 30), EventType.LOGIN, None),
            event(4, "c", _at(3, 4, 22, 41), EventType.SIGNUP, None),
            event(5, "c", _at(3, 5, 8, 5), EventType.CLAIM_CREDITS, 100),
            event(6, "b", _at(3, 14, 14, 3), EventType.CONSUME_CREDITS, 2),
            event(7, "d", _at(4, 10, 12, 0), EventType.VERIFICATION_FAILED, None),
            event(8, "c", _at(5, 31, 23, 59), EventType.VERIFICATION, None),
            # Outside the window: before it, and at the window end.
            event(9, "b", _at(3, 2, 23, 59), EventType.LOGIN, None),
            event(10, "b", _at(6, 1, 0, 0), EventType.LOGIN, None),
        ],
        linkage=[
            link(1, "b", SignalType.STABLE_DEVICE, hashes[0], Reliability.HIGH),
            link(2, "c", SignalType.STABLE_DEVICE, hashes[0], Reliability.HIGH),
            link(3, "b", SignalType.NETWORK_GROUP, hashes[1], Reliability.LOW),
            link(4, "c", SignalType.NETWORK_GROUP, hashes[1], Reliability.LOW),
            link(5, "d", SignalType.NETWORK_GROUP, hashes[1], Reliability.LOW),
        ],
        cycles=[],
        labels=[GenLabel("cluster-1", "household", GroundTruthLabel.LEGITIMATE, "")],
        split_groups=[GenSplit(f"acct-{i}", "grp-1") for i in "bcd"],
        fixtures=[],
    )
    stores = Stores.under(tmp_path)
    persist(world, build_manifest(world, 1, "hand"), stores)

    members = ["acct-d", "acct-b", "acct-c"]
    with closing(sqlite3.connect(f"file:{stores.inference_db}?mode=ro", uri=True)) as conn:
        tl = load_timeline(conn, members)
        text = render_unit(conn, members)

    assert text == render(tl)
    assert text == (
        "Window: starts d1 00:00 (Tuesday 2026-03-03 UTC), ends d91 00:00. Times are UTC.\n"
        "Event types: signup, claim = claim credits, use = consume credits, login, appeal,\n"
        "verify = passed verification, verify_fail = failed verification.\n"
        "Accounts: alias, created, email domain, verification\n"
        "A d-20 09:12 consumer phone\n"
        "B d2 22:41 disposable none\n"
        "C d1 00:00 enterprise enhanced\n"
        "Shared values: signal, reliability, holders\n"
        "network low A B C\n"
        "device high A B\n"
        "Events: d<day> header, then time, account, type, amount\n"
        "d1\n"
        "00:00 C signup\n"
        "09:30 A login\n"
        "09:30 C login\n"
        "d2\n"
        "22:41 B signup\n"
        "d3\n"
        "08:05 B claim 100\n"
        "d12\n"
        "14:03 A use 2\n"
        "d39\n"
        "12:00 C verify_fail\n"
        "d90\n"
        "23:59 B verify"
    )
    assert tl.accounts[0].alias == "A1"
    assert parse(text) == tl
    _assert_label_free(text, hashes)


def _tokenizer_available() -> bool:
    try:
        return VENV_PYTHON.exists() and tokenizer_file().exists()
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _tokenizer_available(), reason="es-p6-train venv or tokenizer missing")
def test_count_tokens_is_positive_and_deterministic() -> None:
    texts = ["d12 14:03 A3 use 2", "Accounts: alias, created, email domain, verification"]
    first = count_tokens(texts)
    assert len(first) == 2
    assert all(isinstance(n, int) and n > 0 for n in first)
    assert count_tokens(texts) == first
    assert count_tokens([texts[1]]) == [first[1]]
    # Digits are one token each in this tokenizer: d,1,2,space,1,4,:,0,3,space-A,3,space-use,
    # space,2.
    assert first[0] == 14
    assert count_tokens([]) == []
