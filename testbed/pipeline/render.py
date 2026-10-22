"""The raw-event render of a unit, and its exact parser (SPEC §6.2). Label-free text.

One fixed render per unit, shared by the LLM rungs. It prints every input of the unit's
`UnitTimeline` and nothing derived from it, so `parse(render(tl)) == tl` and anything computed from
the timeline can be recomputed from the text (information parity, SPEC §5.6).

Format (`raw-v1`), an example::

    Window: starts d1 00:00 (Tuesday 2026-03-03 UTC), ends d91 00:00. Times are UTC.
    Event types: signup, claim = claim credits, use = consume credits, login, appeal,
    verify = passed verification, verify_fail = failed verification.
    Accounts: alias, created, email domain, verification
    A d-40 09:12 consumer phone
    B d3 22:41 disposable none
    Shared values: signal, reliability, holders
    device high A B
    network low A B
    Events: d<day> header, then time, account, type, amount
    d3
    22:41 B signup
    d4
    08:00 A claim 100
    d12
    14:03 B use 2
    14:03 A use 2

Why each field is there:

- Window line: day 1 is the first day, with its weekday and date, so weekdays and the clock are
  derivable. Times are UTC. The end is explicit, so the window length is readable.
- Alias: the i-th capital letter stands for the timeline's `A<i>` (`A` for `A1`, `B` for `A2`...),
  in account lines, shared-value holders and events alike. One letter costs fewer tokens than
  `A12`. The parser maps it back, so the timeline keeps its `A1..An` aliases, and a render
  never has more than 26 accounts.
- Account line: the alias is the only name an account has. Creation is a relative day and time,
  which is at or before day 0 for an account older than the window (account age). The email
  domain type and the verification level are the account's static attributes.
- Shared values: one line per linkage value, with its signal type, its reliability and the
  aliases that hold it. Context-only values and values held by one account are kept: the model
  decides what a shared value is worth. Hashes are omitted; the holders carry the same
  information.
- Day header: a `d<day>` line before the first event of each run of events on one relative day.
  It is a compaction of the same per-event information: each event line below it has the same
  day it would have carried, and no event is merged, dropped or aggregated. The parser carries the
  current day forward, so an event line before any header is malformed. A day with no events has
  no header. Events are never reordered, so a day that recurs after another day gets a new header.
- Event line: `HH:MM`, alias, a short event type and the amount when there is one. One line per
  event in store order, signups included, so the order of simultaneous events, timing and rhythm
  across accounts stay visible. Whole minutes only: `render` refuses a timestamp with seconds
  rather than round it.

No account ids, hashes, group ids, family names or derived cross-account statistics appear.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta

from testbed.pipeline.timeline import Account, Event, Shared, UnitTimeline, load_timeline

RENDER_VERSION = "raw-v1"

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Store names (vendored enum member names) <-> short render tokens.
_EVENT_TOKENS = {
    "SIGNUP": "signup",
    "CLAIM_CREDITS": "claim",
    "CONSUME_CREDITS": "use",
    "LOGIN": "login",
    "APPEAL": "appeal",
    "VERIFICATION": "verify",
    "VERIFICATION_FAILED": "verify_fail",
}
_SIGNAL_TOKENS = {
    "STABLE_DEVICE": "device",
    "PAYMENT_TOKEN": "payment",
    "RECOVERY_CONTACT": "recovery",
    "NETWORK_GROUP": "network",
    "BEHAVIORAL_TEMPLATE": "behavior",
}
_EVENT_NAMES = {v: k for k, v in _EVENT_TOKENS.items()}
_SIGNAL_NAMES = {v: k for k, v in _SIGNAL_TOKENS.items()}

_EVENT_LEGEND = (
    "Event types: signup, claim = claim credits, use = consume credits, login, appeal,\n"
    "verify = passed verification, verify_fail = failed verification."
)
_ACCOUNTS_HEAD = "Accounts: alias, created, email domain, verification"
_SHARED_HEAD = "Shared values: signal, reliability, holders"
_EVENTS_HEAD = "Events: d<day> header, then time, account, type, amount"

_ALIAS = r"A[1-9][0-9]*"  # the timeline's alias form
_LETTER = r"[A-Z]"  # its render form
MAX_ACCOUNTS = 26
_DAY = r"d(-?[0-9]+) ([0-9]{2}):([0-9]{2})"
_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_WINDOW_RE = re.compile(
    rf"Window: starts d1 ([0-9]{{2}}):([0-9]{{2}}) \((\w+) ({_DATE}) UTC\), "
    rf"ends {_DAY}\. Times are UTC\."
)
_ACCOUNT_RE = re.compile(rf"({_LETTER}) {_DAY} ([a-z0-9_]+) ([a-z0-9_]+)")
_SHARED_RE = re.compile(rf"([a-z]+) ([a-z0-9_]+) ({_LETTER}(?: {_LETTER})*)")
_DAY_HEADER_RE = re.compile(r"d(-?[0-9]+)")
_EVENT_RE = re.compile(rf"([0-9]{{2}}):([0-9]{{2}}) ({_LETTER}) ([a-z_]+)(?: (-?[0-9]+))?")
_ENUM_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")


def _check_minute(at: datetime) -> datetime:
    if at.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp is not UTC: {at!r}")
    if at.second or at.microsecond:
        raise ValueError(f"timestamp is not a whole minute: {at!r}")
    return at.astimezone(UTC)


def _day_number(at: datetime, day1: date) -> int:
    """The day relative to day 1; zero or negative before the window."""
    return (at.date() - day1).days + 1


def _fmt(at: datetime, day1: date) -> str:
    """`d<day> HH:MM`."""
    at = _check_minute(at)
    return f"d{_day_number(at, day1)} {at:%H:%M}"


def _unfmt(day: str, hour: str, minute: str, day1: date) -> datetime:
    return datetime.combine(day1 + timedelta(days=int(day) - 1), time(int(hour), int(minute)), UTC)


def _lower(name: str) -> str:
    if not _ENUM_NAME_RE.fullmatch(name):
        raise ValueError(f"not an enum member name: {name!r}")
    return name.lower()


def _letter(alias: str) -> str:
    """`A1` -> `A`, `A2` -> `B`..."""
    if not re.fullmatch(_ALIAS, alias):
        raise ValueError(f"not an alias: {alias!r}")
    number = int(alias[1:])
    if number > MAX_ACCOUNTS:
        raise ValueError(f"alias beyond {MAX_ACCOUNTS} accounts: {alias!r}")
    return chr(ord("A") + number - 1)


def _unletter(letter: str) -> str:
    return f"A{ord(letter) - ord('A') + 1}"


def _lookup(table: dict[str, str], name: str, what: str) -> str:
    if name not in table:
        raise ValueError(f"unknown {what}: {name!r}")
    return table[name]


def render(tl: UnitTimeline) -> str:
    """The timeline as text. Raises `ValueError` on a timestamp that isn't a whole UTC minute."""
    if len(tl.accounts) > MAX_ACCOUNTS:
        raise ValueError(f"{len(tl.accounts)} accounts; a render has at most {MAX_ACCOUNTS}")
    start = _check_minute(tl.window_start)
    day1 = start.date()
    lines = [
        f"Window: starts d1 {start:%H:%M} ({_WEEKDAYS[day1.weekday()]} {day1.isoformat()} UTC), "
        f"ends {_fmt(tl.window_end, day1)}. Times are UTC.",
        _EVENT_LEGEND,
        _ACCOUNTS_HEAD,
    ]
    for a in tl.accounts:
        lines.append(
            f"{_letter(a.alias)} {_fmt(a.created_at, day1)} "
            f"{_lower(a.email_domain_type)} {_lower(a.verification_level)}"
        )
    lines.append(_SHARED_HEAD)
    for s in tl.shared:
        signal = _lookup(_SIGNAL_TOKENS, s.signal_type, "signal type")
        holders = " ".join(_letter(a) for a in s.aliases)
        lines.append(f"{signal} {_lower(s.reliability)} {holders}")
    lines.append(_EVENTS_HEAD)
    day: int | None = None
    for e in tl.events:
        kind = _lookup(_EVENT_TOKENS, e.event_type, "event type")
        amount = "" if e.amount is None else f" {e.amount}"
        at = _check_minute(e.at)
        if (e_day := _day_number(at, day1)) != day:
            day = e_day
            lines.append(f"d{day}")
        lines.append(f"{at:%H:%M} {_letter(e.alias)} {kind}{amount}")
    return "\n".join(lines)


def _match(pattern: re.Pattern[str], line: str, what: str) -> re.Match[str]:
    found = pattern.fullmatch(line)
    if found is None:
        raise ValueError(f"malformed {what} line: {line!r}")
    return found


def parse(text: str) -> UnitTimeline:
    """The exact inverse of `render`. Raises `ValueError` on any line it doesn't recognise."""
    lines = text.split("\n")
    legend = _EVENT_LEGEND.split("\n")
    if lines[1 : 1 + len(legend)] != legend:
        raise ValueError("missing or altered event-type legend")
    window = _match(_WINDOW_RE, lines[0], "window")
    hour, minute, weekday, day1_text, end_day, end_hour, end_minute = window.groups()
    day1 = date.fromisoformat(day1_text)
    if _WEEKDAYS[day1.weekday()] != weekday:
        raise ValueError(f"{day1_text} is not a {weekday}")
    window_start = _unfmt("1", hour, minute, day1)
    window_end = _unfmt(end_day, end_hour, end_minute, day1)

    accounts: list[Account] = []
    shared: list[Shared] = []
    events: list[Event] = []
    reader = _EventReader(day1)
    heads: list[tuple[str, Callable[[str], None]]] = [
        (_ACCOUNTS_HEAD, lambda line: accounts.append(_parse_account(line, day1))),
        (_SHARED_HEAD, lambda line: shared.append(_parse_shared(line))),
        (_EVENTS_HEAD, lambda line: reader.read(line, events)),
    ]
    rest = lines[1 + len(legend) :]
    section = -1
    for line in rest:
        if section + 1 < len(heads) and line == heads[section + 1][0]:
            section += 1
        elif section < 0:
            raise ValueError(f"line before the accounts section: {line!r}")
        else:
            heads[section][1](line)
    if section != len(heads) - 1:
        raise ValueError("missing a section")
    reader.finish()
    return UnitTimeline(window_start, window_end, tuple(accounts), tuple(shared), tuple(events))


def _parse_account(line: str, day1: date) -> Account:
    alias, day, hour, minute, domain, level = _match(_ACCOUNT_RE, line, "account").groups()
    return Account(_unletter(alias), _unfmt(day, hour, minute, day1), domain.upper(), level.upper())


def _parse_shared(line: str) -> Shared:
    signal, reliability, holders = _match(_SHARED_RE, line, "shared-value").groups()
    return Shared(
        _lookup(_SIGNAL_NAMES, signal, "signal token"),
        reliability.upper(),
        tuple(_unletter(h) for h in holders.split()),
    )


class _EventReader:
    """The events section: day headers set the day that the event lines below them carry."""

    def __init__(self, day1: date) -> None:
        self._day1 = day1
        self._day: int | None = None
        self._empty_header = False  # a header with no event line after it yet

    def read(self, line: str, events: list[Event]) -> None:
        header = _DAY_HEADER_RE.fullmatch(line)
        if header is not None:
            day = int(header.group(1))
            if self._empty_header or day == self._day:
                raise ValueError(f"empty or repeated day header: {line!r}")
            self._day = day
            self._empty_header = True
            return
        if self._day is None:
            raise ValueError(f"event line before any day header: {line!r}")
        hour, minute, letter, kind, amount = _match(_EVENT_RE, line, "event").groups()
        events.append(
            Event(
                _unfmt(str(self._day), hour, minute, self._day1),
                _unletter(letter),
                _lookup(_EVENT_NAMES, kind, "event token"),
                None if amount is None else int(amount),
            )
        )
        self._empty_header = False

    def finish(self) -> None:
        if self._empty_header:
            raise ValueError("day header with no event line after it")


def render_unit(conn: sqlite3.Connection, member_account_ids: Sequence[str]) -> str:
    """The render of one unit, read from an inference store."""
    return render(load_timeline(conn, member_account_ids))
