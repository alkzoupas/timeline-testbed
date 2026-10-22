"""F64-w90: the predecessor's 64 label-free features over the 90-day window (SPEC §6.2).

Adapted from `p6_4_features.py` (`FEATURES`, `unit_features`, vendored at fd5deca). The vendored
code read one 30-day cycle plus the earlier cycles' timeline; this reads a `UnitTimeline`, whose
window is `[tl.window_start, tl.window_end)`, and the 12 F-* values. Nothing else, so every
non-F-* feature is recomputable from the render text (information parity, SPEC §5.6). Every
vendored formula is kept; the window replaces "the cycle" and the timeline holds in-window events
only. What the vendored code tied to "before the cycle" means here:

- Age: `age_days_*` is whole days from `created_at` to the window start, floored at 0.
  `joined_in_cycle_frac` (name kept) is the fraction of accounts created inside the window.
- History: `hist_claims_*` and `hist_claim_cycles_*` count claims before the window. The timeline
  holds none, so they are 0.0. They are kept so the names and the count match the predecessor's.
  `history_span_days` is whole days from the earliest account `created_at` to the window start,
  0 if none precedes it.
- Claim timing: `claim_day_mean`, `claim_span_h` and the gap features use the window's claims,
  measured from the window start. A gap-free unit falls back to the window length in hours.
- Per-account claim features (`acct_claim_gap_median_d_mean`, `signup_to_first_claim_h_median`)
  use the account's claims inside the window; the latency runs from `created_at` to the first
  of them, so an account created before the window has a long latency.
- F-* features are the F12 values under the vendored lowercase names.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping

from testbed.features.f12 import F_IDS
from testbed.pipeline.timeline import UnitTimeline

DOMAINS = ("CONSUMER", "DISPOSABLE", "EDUCATION", "ENTERPRISE")
LEVELS = ("NONE", "EMAIL", "PHONE", "ENHANCED")
# Linkage signal type -> feature-name stem ("group" is kept out of feature names).
SIGNALS = {
    "BEHAVIORAL_TEMPLATE": "behavior",
    "NETWORK_GROUP": "network",
    "PAYMENT_TOKEN": "payment",
    "RECOVERY_CONTACT": "recovery",
    "STABLE_DEVICE": "device",
}
ACTIVITY = (
    "claims",
    "claim_credits",
    "consume_credits",
    "consume_uses",
    "logins",
    "verif_ok",
    "verif_failed",
    "active_days",
    "active_hours",
)
MISSING = -1.0  # fill for a temporal summary with no defined value


def _f12_name(feature_id: str) -> str:
    return feature_id.lower().replace("-", "_")


FEATURES: tuple[str, ...] = (
    "n_accounts",
    "age_days_mean",
    "age_days_max",
    "age_days_min",
    "joined_in_cycle_frac",
    *(f"domain_{d.lower()}_frac" for d in DOMAINS),
    *(f"verification_{v.lower()}_frac" for v in LEVELS),
    *(f"shared_values_{s}" for s in SIGNALS.values()),
    *(f"shared_accounts_frac_{s}" for s in SIGNALS.values()),
    *(f"{a}_{stat}" for a in ACTIVITY for stat in ("mean", "max")),
    "claim_day_mean",
    "claim_span_h",
    "claim_gap_median_h",
    "claim_gap_min_h",
    "hist_claims_mean",
    "hist_claims_max",
    "hist_claim_cycles_mean",
    "hist_claim_cycles_max",
    "acct_claim_gap_median_d_mean",
    "signup_to_first_claim_h_median",
    "history_span_days",
    *(_f12_name(f) for f in F_IDS),
)


def _mean_max(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), max(values)


def f64(tl: UnitTimeline, f12_values: Mapping[str, float]) -> dict[str, float]:
    """The 64 features of one unit, in `FEATURES` order. A pure function of its arguments."""
    if set(f12_values) != set(F_IDS):
        raise ValueError(f"F-* set changed: {sorted(f12_values)}")
    start, end = tl.window_start, tl.window_end
    accounts = tl.accounts
    n = len(accounts)
    if n == 0:
        raise ValueError("a unit needs at least one account")
    f: dict[str, float] = {"n_accounts": float(n)}

    ages = [max(0, (start - a.created_at).days) for a in accounts]
    f["age_days_mean"], f["age_days_max"] = _mean_max([float(x) for x in ages])
    f["age_days_min"] = float(min(ages))
    f["joined_in_cycle_frac"] = sum(start <= a.created_at < end for a in accounts) / n
    domains = Counter(a.email_domain_type for a in accounts)
    levels = Counter(a.verification_level for a in accounts)
    if set(domains) - set(DOMAINS) or set(levels) - set(LEVELS):
        raise ValueError(f"unexpected domain or verification value: {domains} {levels}")
    for d in DOMAINS:
        f[f"domain_{d.lower()}_frac"] = domains[d] / n
    for v in LEVELS:
        f[f"verification_{v.lower()}_frac"] = levels[v] / n

    if {s.signal_type for s in tl.shared} - set(SIGNALS):
        raise ValueError("unexpected signal type")
    for signal, stem in SIGNALS.items():
        multi = [
            set(s.aliases) for s in tl.shared if s.signal_type == signal and len(s.aliases) > 1
        ]
        f[f"shared_values_{stem}"] = float(len(multi))
        f[f"shared_accounts_frac_{stem}"] = len(set().union(*multi)) / n

    known = {a.alias for a in accounts}
    act: dict[str, Counter[str]] = {a: Counter() for a in known}
    hours: dict[str, set[int]] = defaultdict(set)
    days: dict[str, set[object]] = defaultdict(set)
    window_claims: list[float] = []  # seconds since the window start
    claims_at: dict[str, list[float]] = defaultdict(list)  # epoch seconds
    for e in tl.events:
        if e.alias not in known:
            raise ValueError(f"event for an unknown account {e.alias}")
        if not start <= e.at < end:
            raise ValueError("event outside the window")
        c = act[e.alias]
        credits = e.amount or 0
        if e.event_type == "CLAIM_CREDITS":
            c["claims"] += 1
            c["claim_credits"] += credits
            window_claims.append((e.at - start).total_seconds())
            claims_at[e.alias].append(e.at.timestamp())
        elif e.event_type == "CONSUME_CREDITS":
            c["consume_uses"] += 1
            c["consume_credits"] += credits
        elif e.event_type == "LOGIN":
            c["logins"] += 1
        elif e.event_type == "VERIFICATION":
            c["verif_ok"] += 1
        elif e.event_type == "VERIFICATION_FAILED":
            c["verif_failed"] += 1
        hours[e.alias].add(e.at.hour)
        days[e.alias].add(e.at.date())  # the window is under a year, so a date is a month-day
    for alias in act:
        act[alias]["active_days"], act[alias]["active_hours"] = len(days[alias]), len(hours[alias])
    for name in ACTIVITY:
        f[f"{name}_mean"], f[f"{name}_max"] = _mean_max(
            [float(act[a.alias][name]) for a in accounts]
        )

    window_h = (end - start).total_seconds() / 3600
    window_claims.sort()
    gaps = [(b - a) / 3600 for a, b in zip(window_claims, window_claims[1:], strict=False)]
    f["claim_day_mean"] = (
        sum(window_claims) / len(window_claims) / 86400 if window_claims else MISSING
    )
    f["claim_span_h"] = (window_claims[-1] - window_claims[0]) / 3600 if window_claims else MISSING
    f["claim_gap_median_h"] = statistics.median(gaps) if gaps else window_h
    f["claim_gap_min_h"] = min(gaps) if gaps else window_h

    # The timeline holds no claims before the window (see the module docstring).
    f["hist_claims_mean"], f["hist_claims_max"] = 0.0, 0.0
    f["hist_claim_cycles_mean"], f["hist_claim_cycles_max"] = 0.0, 0.0
    acct_gaps = [
        statistics.median([(b - a) / 86400 for a, b in zip(ts, ts[1:], strict=False)])
        for ts in (sorted(claims_at[a.alias]) for a in accounts)
        if len(ts) > 1
    ]
    f["acct_claim_gap_median_d_mean"] = sum(acct_gaps) / len(acct_gaps) if acct_gaps else MISSING
    latency = [
        (min(claims_at[a.alias]) - a.created_at.timestamp()) / 3600
        for a in accounts
        if claims_at[a.alias]
    ]
    f["signup_to_first_claim_h_median"] = statistics.median(latency) if latency else MISSING
    earliest = min(a.created_at for a in accounts)
    f["history_span_days"] = float(max(0, (start - earliest).days))

    for fid in F_IDS:
        f[_f12_name(fid)] = float(f12_values[fid])
    if set(f) != set(FEATURES) or len(f) != len(FEATURES):
        raise ValueError("computed features drifted from FEATURES")
    if not all(math.isfinite(v) for v in f.values()):
        raise ValueError("non-finite feature")
    return {name: f[name] for name in FEATURES}
