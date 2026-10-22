"""Known-answer tests for F12 and F64-w90 (SPEC §6.2).

Every expected number below was worked out by hand from the timelines in this file."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import pytest

from testbed.features.f12 import F_IDS, f12
from testbed.features.f64 import FEATURES, f64
from testbed.pipeline.timeline import Account, Event, Shared, UnitTimeline
from testbed.pipeline.units import Unit

# The vendored p6_4_features.FORBIDDEN, copied (the script hard-codes Tier 4 paths).
FORBIDDEN = re.compile(r"label|scenario|group|family|split|truth|abus|legit|ambig")

START = datetime(2026, 3, 3, tzinfo=UTC)
END = datetime(2026, 6, 1, tzinfo=UTC)
WINDOW_H = 2160.0  # 90 days

# Distinct values, so a swapped column shows. F-AGE-1 -> 0.1 ... F-VER-1 -> 1.2.
F12_VALUES = {fid: round((i + 1) / 10, 1) for i, fid in enumerate(F_IDS)}


def at(day: int, hour: int, month: int = 3) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=UTC)


def ev(when: datetime, alias: str, kind: str, amount: int | None = None) -> Event:
    return Event(when, alias, kind, amount)


def timeline(
    accounts: tuple[Account, ...], shared: tuple[Shared, ...], events: tuple[Event, ...]
) -> UnitTimeline:
    return UnitTimeline(START, END, accounts, shared, events)


def three_account_timeline() -> UnitTimeline:
    accounts = (
        # Created 29.5 days before the window starts.
        Account("A1", datetime(2026, 2, 1, 12, tzinfo=UTC), "CONSUMER", "EMAIL"),
        Account("A2", at(10, 6), "DISPOSABLE", "NONE"),
        Account("A3", at(20, 18), "DISPOSABLE", "PHONE"),
    )
    shared = (
        Shared("NETWORK_GROUP", "LOW", ("A1", "A2", "A3")),
        # A value on a single member is kept in the timeline but is not "shared".
        Shared("NETWORK_GROUP", "LOW", ("A3",)),
        Shared("PAYMENT_TOKEN", "HIGH", ("A3",)),
        Shared("STABLE_DEVICE", "HIGH", ("A1", "A2")),
    )
    events = (
        ev(at(5, 9), "A1", "LOGIN"),
        ev(at(5, 10), "A1", "CLAIM_CREDITS", 100),
        ev(at(5, 11), "A1", "CONSUME_CREDITS", 40),
        ev(at(5, 15), "A1", "LOGIN"),
        ev(at(6, 10), "A1", "CLAIM_CREDITS", 100),
        ev(at(6, 11), "A1", "CONSUME_CREDITS", 60),
        ev(at(8, 10), "A1", "CLAIM_CREDITS", 100),
        ev(at(10, 6), "A2", "VERIFICATION_FAILED"),
        ev(at(10, 6), "A2", "VERIFICATION"),
        ev(at(10, 7), "A2", "LOGIN"),
        ev(at(10, 8), "A2", "CLAIM_CREDITS", 100),
        ev(at(10, 9), "A2", "CONSUME_CREDITS", 100),
        ev(at(12, 8), "A2", "CLAIM_CREDITS", 100),
        ev(at(20, 22), "A3", "CLAIM_CREDITS", 50),
        ev(at(21, 20), "A3", "LOGIN"),
        ev(at(21, 21), "A3", "LOGIN"),
    )
    return timeline(accounts, shared, events)


def test_f64_known_answers_on_a_three_account_unit() -> None:
    # Window claims, in hours since 2026-03-03T00:00Z:
    #   A1 Mar 5 10:00 = 58, Mar 6 10:00 = 82, Mar 8 10:00 = 130
    #   A2 Mar 10 08:00 = 176, Mar 12 08:00 = 224;  A3 Mar 20 22:00 = 430
    # Sum 1100.
    expected = {
        "n_accounts": 3.0,
        "age_days_mean": 29 / 3,  # A1 is 29.5 days old, floored to 29; A2, A3 joined later: 0
        "age_days_max": 29.0,
        "age_days_min": 0.0,
        "joined_in_cycle_frac": 2 / 3,  # A2 and A3 were created inside the window
        "domain_consumer_frac": 1 / 3,
        "domain_disposable_frac": 2 / 3,
        "domain_education_frac": 0.0,
        "domain_enterprise_frac": 0.0,
        "verification_none_frac": 1 / 3,
        "verification_email_frac": 1 / 3,
        "verification_phone_frac": 1 / 3,
        "verification_enhanced_frac": 0.0,
        "shared_values_behavior": 0.0,
        "shared_values_network": 1.0,  # the A3-only network value does not count
        "shared_values_payment": 0.0,  # held by one account
        "shared_values_recovery": 0.0,
        "shared_values_device": 1.0,
        "shared_accounts_frac_behavior": 0.0,
        "shared_accounts_frac_network": 1.0,
        "shared_accounts_frac_payment": 0.0,
        "shared_accounts_frac_recovery": 0.0,
        "shared_accounts_frac_device": 2 / 3,  # A1 and A2
        "claims_mean": 2.0,  # 3, 2, 1
        "claims_max": 3.0,
        "claim_credits_mean": 550 / 3,  # 300, 200, 50
        "claim_credits_max": 300.0,
        "consume_credits_mean": 200 / 3,  # 100, 100, 0
        "consume_credits_max": 100.0,
        "consume_uses_mean": 1.0,  # 2, 1, 0
        "consume_uses_max": 2.0,
        "logins_mean": 5 / 3,  # 2, 1, 2
        "logins_max": 2.0,
        "verif_ok_mean": 1 / 3,
        "verif_ok_max": 1.0,
        "verif_failed_mean": 1 / 3,
        "verif_failed_max": 1.0,
        "active_days_mean": 7 / 3,  # A1 Mar 5, 6, 8; A2 Mar 10, 12; A3 Mar 20, 21
        "active_days_max": 3.0,
        "active_hours_mean": 11 / 3,  # A1 {9, 10, 11, 15}; A2 {6, 7, 8, 9}; A3 {20, 21, 22}
        "active_hours_max": 4.0,
        "claim_day_mean": 1100 / 6 / 24,  # mean claim offset 183.33 h
        "claim_span_h": 372.0,  # 430 - 58
        "claim_gap_median_h": 48.0,  # gaps 24, 48, 46, 48, 206
        "claim_gap_min_h": 24.0,
        "hist_claims_mean": 0.0,  # the timeline holds no claims before the window
        "hist_claims_max": 0.0,
        "hist_claim_cycles_mean": 0.0,
        "hist_claim_cycles_max": 0.0,
        "acct_claim_gap_median_d_mean": 1.75,  # A1 median(1 d, 2 d) = 1.5; A2 2 d; A3 has none
        # A1 Feb 1 12:00 -> Mar 5 10:00 = 31 d 22 h = 766 h; A2 2 h; A3 4 h; median 4
        "signup_to_first_claim_h_median": 4.0,
        "history_span_days": 29.0,  # Feb 1 12:00 to Mar 3 00:00 = 29.5 d
        "f_age_1": 0.1,
        "f_con_1": 0.2,
        "f_cyc_1": 0.3,
        "f_cyc_2": 0.4,
        "f_cyc_3": 0.5,
        "f_cyc_4": 0.6,
        "f_dev_1": 0.7,
        "f_dev_2": 0.8,
        "f_dom_1": 0.9,
        "f_dom_2": 1.0,
        "f_pay_1": 1.1,
        "f_ver_1": 1.2,
    }
    assert set(expected) == set(FEATURES)

    got = f64(three_account_timeline(), F12_VALUES)

    assert list(got) == list(FEATURES)
    for name in FEATURES:
        assert got[name] == pytest.approx(expected[name]), name


def test_single_claim_falls_back_to_the_window_length() -> None:
    tl = timeline(
        (Account("A1", START, "ENTERPRISE", "ENHANCED"),),
        (),
        (ev(at(4, 6), "A1", "CLAIM_CREDITS", 10),),
    )
    got = f64(tl, F12_VALUES)

    assert got["n_accounts"] == 1.0
    assert got["joined_in_cycle_frac"] == 1.0  # created at the window start counts as inside
    assert got["age_days_max"] == 0.0
    assert got["history_span_days"] == 0.0  # nothing precedes the window start
    assert got["domain_enterprise_frac"] == 1.0
    assert got["verification_enhanced_frac"] == 1.0
    assert got["shared_values_device"] == 0.0
    assert got["shared_accounts_frac_device"] == 0.0
    assert got["claims_mean"] == 1.0
    assert got["claim_day_mean"] == pytest.approx(1.25)  # 30 h
    assert got["claim_span_h"] == 0.0
    assert got["claim_gap_median_h"] == WINDOW_H
    assert got["claim_gap_min_h"] == WINDOW_H
    assert got["acct_claim_gap_median_d_mean"] == -1.0  # one claim has no gap
    assert got["signup_to_first_claim_h_median"] == pytest.approx(30.0)


def test_no_claims_gives_missing() -> None:
    tl = timeline(
        (Account("A1", at(1, 0, month=1), "CONSUMER", "NONE"),),
        (),
        (ev(at(4, 6), "A1", "LOGIN"),),
    )
    got = f64(tl, F12_VALUES)

    assert got["claims_max"] == 0.0
    assert got["logins_max"] == 1.0
    assert got["claim_day_mean"] == -1.0
    assert got["claim_span_h"] == -1.0
    assert got["claim_gap_median_h"] == WINDOW_H
    assert got["claim_gap_min_h"] == WINDOW_H
    assert got["acct_claim_gap_median_d_mean"] == -1.0
    assert got["signup_to_first_claim_h_median"] == -1.0
    # Created 2026-01-01, so 61 days before 2026-03-03.
    assert got["age_days_mean"] == 61.0
    assert got["history_span_days"] == 61.0
    assert got["joined_in_cycle_frac"] == 0.0


def test_f64_refuses_bad_input() -> None:
    tl = three_account_timeline()
    with pytest.raises(ValueError, match="F-\\* set changed"):
        f64(tl, {k: v for k, v in F12_VALUES.items() if k != "F-VER-1"})
    outside = timeline(tl.accounts, (), (ev(at(1, 0), "A1", "LOGIN"),))
    with pytest.raises(ValueError, match="outside the window"):
        f64(outside, F12_VALUES)
    stranger = timeline(tl.accounts, (), (ev(at(5, 0), "A9", "LOGIN"),))
    with pytest.raises(ValueError, match="unknown account"):
        f64(stranger, F12_VALUES)
    with pytest.raises(ValueError, match="not finite|non-finite"):
        f64(tl, {**F12_VALUES, "F-AGE-1": float("nan")})


def test_feature_names() -> None:
    assert len(FEATURES) == 64
    assert len(set(FEATURES)) == 64
    assert not [name for name in FEATURES if FORBIDDEN.search(name)]
    # Spot-check the vendored order at both ends and around the F-* block.
    assert FEATURES[0] == "n_accounts"
    assert FEATURES[51] == "history_span_days"
    assert FEATURES[52:] == tuple(f.lower().replace("-", "_") for f in F_IDS)


def _unit(latest_contributions: list[dict[str, Any]]) -> Unit:
    older = {"contributions": [{"feature_id": f, "raw_value": 99.0} for f in F_IDS]}
    return Unit(
        snapshot_id="snap-1",
        member_account_ids=("acct-1",),
        detector_score=0.5,
        latest={"contributions": latest_contributions},
        payloads=(older, {"contributions": latest_contributions}),
    )


def _contributions(skip: str | None = None) -> list[dict[str, Any]]:
    # Shuffled order and extra keys, as in a real payload.
    return [
        {"feature_id": f, "raw_value": F12_VALUES[f], "weight": 1.0}
        for f in reversed(F_IDS)
        if f != skip
    ]


def test_f12_reads_the_latest_payload_in_vendored_order() -> None:
    got = f12(_unit(_contributions()))

    assert F_IDS[0] == "F-AGE-1" and F_IDS[-1] == "F-VER-1" and len(F_IDS) == 12
    assert list(got) == list(F_IDS)
    assert got == F12_VALUES


def test_f12_refuses_a_missing_id() -> None:
    with pytest.raises(ValueError, match="F-\\* set changed"):
        f12(_unit(_contributions(skip="F-PAY-1")))


def test_f12_refuses_an_extra_or_repeated_id() -> None:
    extra = [*_contributions(), {"feature_id": "F-NEW-1", "raw_value": 1.0}]
    with pytest.raises(ValueError, match="F-\\* set changed"):
        f12(_unit(extra))
    repeated = [*_contributions(), {"feature_id": "F-AGE-1", "raw_value": 2.0}]
    with pytest.raises(ValueError, match="F-\\* set changed"):
        f12(_unit(repeated))


def test_f12_refuses_a_non_finite_value() -> None:
    bad = _contributions()
    bad[0]["raw_value"] = float("inf")
    with pytest.raises(ValueError, match="not finite"):
        f12(_unit(bad))
