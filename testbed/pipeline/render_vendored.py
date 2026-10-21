"""The vendored P6 render, copied for the M0 smoke test only (SPEC §6.2).

Copied unchanged from `vendor/entitlement-shield/scripts/p6_recall_pilot.py` at `fd5deca`:
`render_unit` (lines 137-247), `_history_v3` (250-286), `_ts` (120-123) and the constants at
100-103. SPEC v0.2 rules it out for the LLM rungs: it keeps one row per (day, account, event
type), so within-day order, per-event amounts and gaps are lost. M1 writes the new render.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

RENDERS = ("v1", "v2", "v3")  # v2 fixes negative ages (P6.1c §3.5); v3 compacts history (P6.3)
# v3 short event names; every other event type keeps its own name.
V3_EVENTS = {"claim_credits": "claim", "consume_credits": "consume",
             "verification": "verify_ok", "verification_failed": "verify_fail"}


def _ts(value: str) -> datetime:
    """Payload timestamps are aware; SQLite stores naive UTC. Compare both as UTC."""
    parsed = datetime.fromisoformat(value.replace(" ", "T"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def render_unit(conn: sqlite3.Connection, payload: dict[str, Any], *,
                render: str = "v1", lookback: int | None = None) -> str:
    """Compressed, label-free view of one unit: who, what they share, what they did.

    ``v1`` is what P6.1, P6.1b and P6.1c saw (and what ``p6_fp_anatomy`` parses): an account
    created mid-cycle shows a negative age (P6.1c §3.5). ``v2`` (P6.2) shows such an account
    as joining on day N of the cycle instead; everything else is identical.

    ``v3`` (P6.3) is ``v2`` with the History section compacted losslessly: one line per day,
    the date printed once, short event names, ``Ncr`` for N credits. ``lookback`` (v3 only)
    is the fixed fallback for a token cap: history days more than ``lookback`` days before
    the cycle start are rolled up into per-week, per-account, per-event counts and credit
    sums. ``None`` keeps every day in full. The roll-up adds nothing the lines did not hold.
    """
    if render not in RENDERS:
        raise ValueError(f"render must be one of {RENDERS}")
    if lookback is not None and (render != "v3" or lookback < 0):
        raise ValueError("lookback needs render='v3' and a non-negative number of days")
    members = sorted(payload["member_account_ids"])
    alias = {m: f"A{i + 1}" for i, m in enumerate(members)}
    start, end = _ts(payload["window_start"]), _ts(payload["window_end"])
    as_of = _ts(payload["as_of"])
    marks = ",".join("?" * len(members))
    heading = ("age at cycle start" if render == "v1"
               else "age at cycle start, or the cycle day it joined")
    lines = [
        f"Credit cycle: {start:%Y-%m-%d} to {end:%Y-%m-%d}. Accounts in cluster: {len(members)}.",
        "",
        f"Accounts ({heading}, email domain type, verification level):",
    ]
    accounts = {
        account_id: (created, domain, level)
        for account_id, created, domain, level in conn.execute(
            f"select account_id, created_at, email_domain_type, verification_level "
            f"from accounts where account_id in ({marks})", members,
        )
    }
    for m in members:
        created, domain, level = accounts[m]
        joined = _ts(created)
        if render != "v1" and joined > start:
            when = f"joined on day {(joined - start).days + 1} of this cycle"
        else:
            when = f"{(start - joined).days} days old"
        lines.append(f"- {alias[m]}: {when}, {domain.lower()}, {level.lower()}")

    shared: dict[str, list[list[str]]] = defaultdict(list)
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    cutoff = f"{as_of:%Y-%m-%d %H:%M:%S.%f}"  # SQLite's naive-UTC text format
    for account_id, signal, value in conn.execute(
        f"select account_id, signal_type, signal_value_hash from linkage_observations "
        f"where account_id in ({marks}) and timestamp <= ?", [*members, cutoff],
    ):
        groups[(signal, value)].add(account_id)
    for (signal, _), accts in sorted(groups.items()):
        if len(accts) > 1:
            shared[signal.lower()].append(sorted(alias[a] for a in accts))
    lines += ["", "Shared resources (accounts observed using the same value):"]
    if not shared:
        lines.append("- none recorded")
    for signal, sets in sorted(shared.items()):
        uniq = sorted({" ".join(s) for s in sets})
        lines.append(f"- {signal}: " + "; ".join(f"[{u}]" for u in uniq))

    events = [e for e in payload["timeline"] if e["event_type"] != "appeal"]
    per: dict[str, Counter[str]] = defaultdict(Counter)
    credits: dict[str, Counter[str]] = defaultdict(Counter)
    hours: dict[str, set[int]] = defaultdict(set)
    days: dict[str, set[str]] = defaultdict(set)
    for e in events:
        t = _ts(e["timestamp"])
        if not start <= t < end:
            continue
        a = alias[e["account_id"]]
        per[a][e["event_type"]] += 1
        credits[a][e["event_type"]] += e["credits_amount"] or 0
        hours[a].add(t.hour)
        days[a].add(f"{t:%m-%d}")
    lines += ["", "Activity inside this cycle, per account:"]
    for a in sorted(alias.values(), key=lambda s: int(s[1:])):
        c = per[a]
        lines.append(
            f"- {a}: claims {c['claim_credits']} ({credits[a]['claim_credits']} credits), "
            f"consumed {credits[a]['consume_credits']} credits in {c['consume_credits']} uses, "
            f"logins {c['login']}, verifications ok {c['verification']} / failed "
            f"{c['verification_failed']}, active days {len(days[a])}, "
            f"distinct active hours {len(hours[a])}"
        )

    # History: one line per day per account and event type, so logins and small
    # consumes do not drown the claims, signups and verifications.
    daily: dict[tuple[str, str, str], list[int]] = {}
    first_time: dict[tuple[str, str, str], str] = {}
    for e in events:
        t = _ts(e["timestamp"])
        k = (f"{t:%Y-%m-%d}", alias[e["account_id"]], e["event_type"])
        n, total = daily.get(k, [0, 0])
        daily[k] = [n + 1, total + (e["credits_amount"] or 0)]
        first_time.setdefault(k, f"{t:%H:%M}")
    ordered = sorted(
        daily.items(), key=lambda kv: (kv[0][0], first_time[kv[0]], kv[0][1], kv[0][2])
    )
    if render == "v3":
        return "\n".join(lines + _history_v3(ordered, first_time, start, lookback))
    lines += ["", "History (date, first time, account, event, count, credits; "
              f"lines before {start:%Y-%m-%d} are earlier context):"]
    for (day, a, kind), (n, total) in ordered:
        extra = f" x{n}" if n > 1 else ""
        extra += f" {total} credits" if total else ""
        lines.append(f"{day} {first_time[(day, a, kind)]} {a} {kind}{extra}")
    return "\n".join(lines)


def _history_v3(ordered: list[tuple[tuple[str, str, str], list[int]]],
                first_time: dict[tuple[str, str, str], str], start: datetime,
                lookback: int | None) -> list[str]:
    """v3 History: v2's rows grouped by day; older pre-cycle days optionally rolled up by week."""
    cut = "" if lookback is None else f"{start.date() - timedelta(days=lookback)}"
    weekly: dict[tuple[str, str, str], list[int]] = {}
    days: dict[str, list[str]] = {}
    for (day, a, kind), (n, total) in ordered:
        name = V3_EVENTS.get(kind, kind)
        if day < cut:
            iso = datetime.strptime(day, "%Y-%m-%d").isocalendar()
            k = (f"{iso.year}-W{iso.week:02d}", a, name)
            c, t = weekly.get(k, [0, 0])
            weekly[k] = [c + n, t + total]
            continue
        extra = f" x{n}" if n > 1 else ""
        extra += f" {total}cr" if total else ""
        days.setdefault(day, []).append(f"{a} {name} {first_time[(day, a, kind)]}{extra}")
    out = ["", "History, one line per day (date: account event first-time, then xN when it "
           "happened N times that day and Ncr for N credits; the year is shown when it "
           f"changes; days before {start:%Y-%m-%d} are earlier context):"]
    if weekly:
        out.append(f"Days before {cut} are summarised per ISO week ({len({k[0] for k in weekly})} "
                   "weeks): account event xcount, credits.")
        week_lines: dict[str, list[str]] = {}
        for (week, a, name), (n, total) in sorted(
            weekly.items(), key=lambda kv: (kv[0][0], int(kv[0][1][1:]), kv[0][2])
        ):
            week_lines.setdefault(week, []).append(
                f"{a} {name} x{n}" + (f" {total}cr" if total else ""))
        out += [f"week {week}: " + "; ".join(items) for week, items in week_lines.items()]
    year = ""
    for day, items in days.items():
        label = day if day[:4] != year else day[5:]
        year = day[:4]
        out.append(f"{label}: " + "; ".join(items))
    return out
