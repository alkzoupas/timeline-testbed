"""Known-answer tests for the world health checks (SPEC §5.4, §5.5, §5.6).

Each row's pass/fail logic is driven with synthetic scores, one input that must pass and one that
must fail. The mechanical SQL runs on a tiny hand-built schema (no keys, so duplicates can be
inserted). Two small persisted worlds (s = 1 and s = 0, the test-only `small` scale) cover the
end-to-end wiring.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.services.generator import WORLD_END

from testbed.eval import health
from testbed.eval.health import (
    ChannelUnit,
    ClassSample,
    PairSample,
    channel_check,
    channel_pair,
    determinism_row,
    fill_undefined,
    gensample_health,
    linkage_row,
    measured_tag,
    mechanical_row,
    negative_row,
    nuisance,
    nuisance_row,
    null_row,
    order_coupling_row,
    parity_row,
    parity_sample,
    positive_oracle_row,
    positive_row,
    render_length_row,
    render_visibility_row,
    undefined_row,
    world_health,
)
from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES
from testbed.pipeline.run_world import run_world
from testbed.pipeline.timeline import WINDOW_START, Account, Event, Shared, UnitTimeline
from testbed.world.families import PAIRS
from testbed.world.generator import GENERATOR_VERSION, WorldSpec


def at(days: float, hours: int = 0) -> datetime:
    return WINDOW_START + timedelta(days=days, hours=hours)


# --- nuisance -------------------------------------------------------------------------------------


def hand_built() -> UnitTimeline:
    """Three accounts created 10 d before, 2 d after and 6 d after the window start.

    Domains consumer, consumer, disposable; levels phone, email, none. One high-reliability device
    held by all, one low-reliability network value held by A1 and A2. Events:
    A1: verify, claim, consume 3, consume 2, login (verified before its first claim);
    A2: signup, verify_fail, claim, verify (verified after its claim);
    A3: signup, claim (never verified)."""
    accounts = (
        Account("A1", at(-10), "CONSUMER", "PHONE"),
        Account("A2", at(2), "CONSUMER", "EMAIL"),
        Account("A3", at(6), "DISPOSABLE", "NONE"),
    )
    shared = (
        Shared("NETWORK_GROUP", "LOW", ("A1", "A2")),
        Shared("STABLE_DEVICE", "HIGH", ("A1", "A2", "A3")),
    )
    events = (
        Event(at(0, 0), "A1", "VERIFICATION", None),
        Event(at(0, 1), "A1", "CLAIM_CREDITS", 25),
        Event(at(0, 2), "A1", "CONSUME_CREDITS", 3),
        Event(at(0, 3), "A1", "CONSUME_CREDITS", 2),
        Event(at(1), "A1", "LOGIN", None),
        Event(at(2), "A2", "SIGNUP", None),
        Event(at(2, 1), "A2", "VERIFICATION_FAILED", None),
        Event(at(2, 2), "A2", "CLAIM_CREDITS", 25),
        Event(at(2, 3), "A2", "VERIFICATION", None),
        Event(at(6), "A3", "SIGNUP", None),
        Event(at(6, 1), "A3", "CLAIM_CREDITS", 25),
    )
    return UnitTimeline(WINDOW_START, WORLD_END, accounts, shared, events)


def test_nuisance_known_answer() -> None:
    # Creation days -10, 2, 6: mean -2/3, population variance 416/9.
    assert nuisance(hand_built()) == pytest.approx(
        {
            "group_size": 3.0,
            "created_mean_d": -2 / 3,
            "created_sd_d": math.sqrt(416 / 9),
            "domain_consumer": 2 / 3,
            "domain_disposable": 1 / 3,
            "domain_education": 0.0,
            "domain_enterprise": 0.0,
            "level_none": 1 / 3,
            "level_email": 1 / 3,
            "level_phone": 1 / 3,
            "level_enhanced": 0.0,
            "edge_stable_device": 1.0,
            "edge_payment_token": 0.0,
            "edge_recovery_contact": 0.0,
            "edge_reliability": 2.0,
            "has_network": 1.0,
            "has_behavior": 0.0,
            "claims": 3.0,
            "consume_events": 2.0,
            "consumed_credits": 5.0,
            "logins": 1.0,
            "verifications": 2.0,
            "failed_verifications": 1.0,
            "events": 11.0,
            "verified_before_first_claim_frac": 1 / 3,
        }
    )


def test_nuisance_without_an_edge_value_marks_reliability_missing() -> None:
    tl = hand_built()
    bare = UnitTimeline(tl.window_start, tl.window_end, tl.accounts, (), tl.events)
    out = nuisance(bare)
    assert out["edge_reliability"] == -1.0
    assert out["edge_stable_device"] == 0.0
    assert out["has_network"] == 0.0


# --- undefined oracle values ----------------------------------------------------------------------


def test_fill_undefined_ranks_none_below_every_defined_value() -> None:
    pos, neg, pos_none, neg_none = fill_undefined([None, 2.0], [1.0, None])
    # The fill is one below the smallest defined value (1.0), i.e. 0.0.
    assert (pos.tolist(), neg.tolist(), pos_none, neg_none) == ([0.0, 2.0], [1.0, 0.0], 1, 1)
    # Pairs: 0 vs 1 lose, 0 vs 0 tie, 2 vs 1 and 2 vs 0 win: (0 + 0.5 + 1 + 1) / 4.
    assert health.auc(pos, neg) == pytest.approx(0.625)


def test_fill_undefined_with_no_defined_value_is_all_ties() -> None:
    pos, neg, pos_none, neg_none = fill_undefined([None, None], [None])
    assert (pos.tolist(), neg.tolist(), pos_none, neg_none) == ([0.0, 0.0], [0.0], 2, 1)
    assert health.auc(pos, neg) == 0.5


# --- the null rule and the threshold rule ---------------------------------------------------------


def test_null_row_passes_identical_distributions_and_fails_separation() -> None:
    same = np.arange(100, dtype=float)
    separated = same + 1000.0
    row = null_row("x", {"fair": (same, same.copy()), "rigged": (separated, same)})
    assert row["passed"] is False
    assert row["failed"] == ["rigged"]
    assert row["results"]["fair"]["passed"] is True
    assert row["results"]["fair"]["auc"] == 0.5
    assert row["results"]["rigged"]["auc"] == 1.0
    assert row["tests"] == 2
    assert null_row("x", {"fair": (same, same.copy())})["passed"] is True


def test_null_row_records_the_permutation_fields_and_no_interval() -> None:
    same = np.arange(100, dtype=float)
    rng = np.random.default_rng(7)
    drift = (rng.normal(0.4, 1, 100), rng.normal(0, 1, 100))
    row = null_row("x", {"fair": (same, same.copy()), "drift": drift}, tests=4)
    assert row["alpha_adj"] == pytest.approx(0.05 / 4) and row["tests"] == 4
    fair = row["results"]["fair"]
    assert set(fair) == {"auc", "z", "p", "alpha_adj", "method", "passed", "n_pos", "n_neg"}
    assert (fair["method"], fair["p"], fair["n_pos"], fair["n_neg"]) == ("screen", 1.0, 100, 100)
    assert row["results"]["drift"]["alpha_adj"] == pytest.approx(0.0125)


def test_null_row_bonferroni_widens_with_tests() -> None:
    rng = np.random.default_rng(7)
    pos, neg = rng.normal(0.4, 1, 100), rng.normal(0, 1, 100)  # AUC ~ 0.61, SE ~ 0.04
    items = {"drift": (pos, neg)}
    assert null_row("x", items, tests=1)["passed"] is False
    assert null_row("x", items, tests=100)["passed"] is True
    assert null_row("x", items, tests=100)["alpha_adj"] < null_row("x", items, tests=1)["alpha_adj"]


def test_world_family_level_on_a_known_p() -> None:
    """The round-1 specimen (24 x +1 and 16 x -1 against 160 zeros) has exact two-sided p
    0.005092447416579459: it fails a 6-test row (alpha 0.05/6 = 0.0083) and passes inside an
    s = 0 world family of 163 (alpha 0.05/163 = 0.00031)."""
    items = {"pair1": (np.array([1.0] * 24 + [-1.0] * 16), np.zeros(160))}
    row, world = null_row("negative_control", items), null_row("negative_control", items, tests=163)
    assert row["tests"] == 1 and row["passed"] is False  # by default an item count of 1
    assert null_row("negative_control", items, tests=6)["passed"] is False
    assert world["passed"] is True and world["alpha_adj"] == pytest.approx(0.05 / 163)
    assert world["results"]["pair1"]["p"] == pytest.approx(0.005092447416579459, abs=0.003)


def test_null_row_results_do_not_depend_on_item_order() -> None:
    rng = np.random.default_rng(9)
    items = {
        f"v{i}": (rng.normal(0.35, 1, 40), rng.normal(0, 1, 160)) for i in range(4)
    }  # borderline, so each item takes the seeded permutation path
    forward = null_row("x", items, tests=6)
    backward = null_row("x", dict(reversed(items.items())), tests=6)
    assert forward["results"] == backward["results"]
    assert any(r["method"] == "permutation" for r in forward["results"].values())


def test_positive_row_threshold() -> None:
    good = (np.array([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 0.0]), np.arange(10.0))
    weak = (np.arange(10.0), np.arange(10.0))
    # `good`: 9 positives beat every negative; the last (0.0) ties one and loses to nine: 0.905
    assert health.auc(*good) == pytest.approx(0.905)
    assert positive_row("p", {"a": good}, minimum=0.90)["passed"] is True
    row = positive_row("p", {"a": good, "b": weak}, minimum=0.90)
    assert row["passed"] is False and row["failed"] == ["b"]
    assert positive_row("p", {"a": good}, minimum=0.91)["passed"] is False


# --- rows on per-pair samples ---------------------------------------------------------------------


def klass(values: dict[str, list[float]], oracle_scores: list[float | None]) -> ClassSample:
    n = len(oracle_scores)
    return ClassSample(
        tuple({k: v[i] for k, v in values.items()} for i in range(n)), tuple(oracle_scores)
    )


def test_nuisance_row_excludes_mechanism_variables_and_counts_tests() -> None:
    created = {"created_mean_d": [1.0, 2.0, 3.0, 4.0], "created_sd_d": [1.0, 2.0, 3.0, 4.0]}
    other = {"claims": [1.0, 2.0, 3.0, 4.0]}
    sample = {
        1: PairSample(
            1, klass({**created, **other}, [0.0] * 4), klass({**created, **other}, [0.0] * 4)
        ),
        5: PairSample(
            5, klass({**created, **other}, [0.0] * 4), klass({**created, **other}, [0.0] * 4)
        ),
    }
    row = nuisance_row(sample)
    # pair 1 tests three variables; pair 5 drops its two creation variables.
    assert sorted(row["results"]) == [
        "pair1/claims",
        "pair1/created_mean_d",
        "pair1/created_sd_d",
        "pair5/claims",
    ]
    assert row["tests"] == 4
    assert row["excluded"] == {"pair5": ["created_mean_d", "created_sd_d"]}
    assert row["passed"] is True


def test_nuisance_row_fails_an_imbalanced_variable() -> None:
    pos = klass({"claims": [5.0] * 30 + [6.0] * 30}, [0.0] * 60)
    neg = klass({"claims": [1.0] * 60}, [0.0] * 60)
    row = nuisance_row({3: PairSample(3, pos, neg)})
    assert row["passed"] is False and row["failed"] == ["pair3/claims"]


def tied(n: int) -> list[float | None]:
    return [float(i % 7) for i in range(n)]


def test_negative_row_passes_a_null_and_fails_a_separated_oracle() -> None:
    fair = {
        p: PairSample(p, klass({"a": [0.0] * 60}, tied(60)), klass({"a": [0.0] * 60}, tied(60)))
        for p in (1, 2)
    }
    row = negative_row(fair)
    assert row["passed"] is True and row["tests"] == 2
    rigged = dict(fair)
    rigged[2] = PairSample(
        2, klass({"a": [0.0] * 60}, [float(i + 100) for i in range(60)]), fair[2].twin
    )
    bad = negative_row(rigged)
    assert bad["passed"] is False and bad["failed"] == ["pair2"]


def test_oracle_rows_report_undefined_counts_by_family() -> None:
    pos = klass({"a": [0.0] * 4}, [None, 1.0, 2.0, None])
    neg = klass({"a": [0.0] * 4}, [None, 0.0, 1.0, 2.0])
    samples = {3: PairSample(3, pos, neg)}
    expected = {"pair3": {"template_operator": 2, "office_card": 1}}
    assert undefined_row(samples)["undefined"] == expected
    assert undefined_row(samples)["passed"] is True
    assert negative_row(samples)["undefined"] == expected
    assert positive_oracle_row(samples)["undefined"] == expected


def test_positive_oracle_row_uses_the_control_threshold() -> None:
    sep = klass({"a": [0.0] * 20}, [float(i + 100) for i in range(20)])
    base = klass({"a": [0.0] * 20}, [float(i) for i in range(20)])
    assert positive_oracle_row({1: PairSample(1, sep, base)})["passed"] is True
    assert positive_oracle_row({1: PairSample(1, base, base)})["passed"] is False


# --- order coupling -------------------------------------------------------------------------------


def test_order_coupling_fails_a_label_that_sorts_with_every_order() -> None:
    n = 40
    positive = [i < 10 for i in range(n)]  # abusive groups come first everywhere
    ids = [f"acct-{i:04d}" for i in range(n)]
    row = order_coupling_row(positive, ids, list(range(n)), [float(i) for i in range(n)])
    assert row["name"] == "order_coupling" and row["tests"] == 3
    assert row["passed"] is False
    assert row["failed"] == ["earliest_created_at", "first_row", "min_account_id"]


def test_order_coupling_passes_an_interleaved_label() -> None:
    n = 40
    positive = [i % 4 == 0 for i in range(n)]
    ids = [f"acct-{i:04d}" for i in range(n)]
    row = order_coupling_row(positive, ids, list(range(n)), [float(i // 3) for i in range(n)])
    assert row["passed"] is True


# --- linkage --------------------------------------------------------------------------------------


def linked(groups: int, *, split: int = 0, mixed: int = 0) -> Any:
    """`groups` groups of two accounts; the first `split` groups are each cut into two units."""
    group_of = {f"a{g}{k}": f"g{g}" for g in range(groups) for k in range(2)}
    units: dict[str, list[str]] = {}
    for g in range(groups):
        if g < split:
            units[f"u{g}x"], units[f"u{g}y"] = [f"a{g}0"], [f"a{g}1"]
        else:
            units[f"u{g}"] = [f"a{g}0", f"a{g}1"]
    return linkage_row(units, group_of, mixed)


def test_linkage_row_threshold_and_mixed_units() -> None:
    assert linked(20)["passed"] is True and linked(20)["share"] == 1.0
    one_split = linked(20, split=1)  # 19 / 20 = 0.95 meets the threshold exactly
    assert one_split["share"] == 0.95 and one_split["passed"] is True
    two_splits = linked(20, split=2)
    assert two_splits["share"] == 0.9 and two_splits["passed"] is False
    assert linked(20, mixed=1)["passed"] is False


def test_linkage_row_fails_a_unit_with_a_stranger() -> None:
    group_of = {"a0": "g0", "a1": "g0", "b0": "g1"}
    # g0's unit also holds b0, and g1's account is in that same unit: neither group is recovered.
    row = linkage_row({"u": ["a0", "a1", "b0"]}, group_of, 1)
    assert row["recovered"] == 0 and row["passed"] is False


# --- render visibility, render length ------------------------------------------------------------


def scores(separated: bool) -> tuple[list[float | None], list[float | None]]:
    return (
        [float(i + 100) if separated else float(i) for i in range(20)],
        [float(i) for i in range(20)],
    )


def test_render_visibility_at_s1_needs_the_render_auc() -> None:
    strong = {1: scores(True)}
    assert render_visibility_row(1.0, strong, strong, 0)["passed"] is True
    weak = {1: scores(False)}
    row = render_visibility_row(1.0, weak, strong, 0)
    assert row["passed"] is False and row["failed"] == ["pair1"]
    assert row["results"]["pair1"]["render_auc"] == pytest.approx(0.5)
    assert row["results"]["pair1"]["store_auc"] == 1.0


def test_render_visibility_at_s075_compares_with_the_store() -> None:
    store = {1: scores(False)}  # AUC 0.5
    close = {1: ([float(i) + (0.5 if i == 19 else 0.0) for i in range(20)], scores(False)[1])}
    far = {1: scores(True)}
    assert render_visibility_row(0.75, close, store, 0)["passed"] is True
    assert render_visibility_row(0.75, far, store, 0)["passed"] is False


def test_render_visibility_elsewhere_reports_only_but_parity_faults_fail() -> None:
    weak = {1: scores(False)}
    assert render_visibility_row(0.5, weak, weak, 0)["passed"] is True
    assert render_visibility_row(0.5, weak, weak, 0)["rule"] == "reported only"
    row = render_visibility_row(0.5, weak, weak, 3)
    assert row["passed"] is False and row["differs"] == 3


def test_render_length_row_null_rule_and_budget() -> None:
    flat = {1: ([float(i) for i in range(50)], [float(i) for i in range(50)])}
    assert render_length_row(flat, 1792)["passed"] is True
    over = render_length_row(flat, 1793)
    assert over["passed"] is False and over["max_tokens"] == 1793
    rigged = {1: ([float(i + 500) for i in range(50)], [float(i) for i in range(50)])}
    assert render_length_row(rigged, 100)["passed"] is False


# --- F12 parity -----------------------------------------------------------------------------------


def test_parity_sample_is_seeded_and_capped_per_family() -> None:
    family_of = {f"u{i:03d}": "fam_a" if i < 30 else "fam_b" for i in range(36)}
    sample = parity_sample(family_of)
    assert {k: len(v) for k, v in sample.items()} == {"fam_a": 10, "fam_b": 6}
    assert sample == parity_sample(family_of)
    assert set(sample["fam_b"]) == {f"u{i:03d}" for i in range(30, 36)}


def test_parity_row_passes_equal_values_and_fails_each_kind_of_mismatch() -> None:
    sample = {"fam": ["u1", "u2", "u3"]}
    expected = {u: {"F-A": 1.0, "F-B": 2.0} for u in sample["fam"]}
    texts = {u: u for u in sample["fam"]}

    def exact(text: str) -> dict[str, float]:
        return {"F-A": 1.0, "F-B": 2.0}

    row = parity_row(sample, texts, expected, exact)
    assert row["passed"] is True and row["checked"] == 3 and row["max_abs_diff"] == 0.0

    def flawed(text: str) -> dict[str, float]:
        if text == "u1":
            return {"F-A": 1.0, "F-B": 2.5}  # a value is off
        if text == "u2":
            return {"F-A": 1.0}  # a feature is missing
        raise ValueError("expected one unit from the render, got 2")

    row = parity_row(sample, texts, expected, flawed)
    assert row["passed"] is False
    assert [(m["unit"], sorted(m)) for m in row["mismatches"]] == [
        ("u1", ["family", "features", "unit"]),
        ("u2", ["error", "family", "unit"]),
        ("u3", ["error", "family", "unit"]),
    ]
    assert row["mismatches"][0]["features"] == ["F-B"]


# --- mechanical validity --------------------------------------------------------------------------

_SCHEMA = """
create table accounts (account_id text, created_at text);
create table account_events (event_id text, account_id text, timestamp text, event_type text,
                             credits_amount integer);
create table linkage_observations (observation_id text, account_id text, timestamp text);
create table entitlement_cycles (cycle_id text, account_id text, cycle_start text, cycle_end text,
                                 credits_granted integer, credits_consumed integer);
"""
START = "2026-03-03 00:00:00.000000"
END = "2026-06-01 00:00:00.000000"


def valid_store() -> sqlite3.Connection:
    """One account created in the window with a claim, 5 credits consumed in its cycle."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    conn.executescript(
        """
        insert into accounts values ('a1', '2026-03-10 00:00:00.000000');
        insert into account_events values
          ('e1', 'a1', '2026-03-10 00:00:00.000000', 'SIGNUP', null),
          ('e2', 'a1', '2026-03-10 01:00:00.000000', 'CLAIM_CREDITS', 25),
          ('e3', 'a1', '2026-03-11 01:00:00.000000', 'CONSUME_CREDITS', 3),
          ('e4', 'a1', '2026-03-12 01:00:00.000000', 'CONSUME_CREDITS', 2);
        insert into linkage_observations values ('o1', 'a1', '2026-03-10 05:00:00.000000');
        insert into entitlement_cycles values
          ('c1', 'a1', '2026-03-10 00:00:00.000000', '2026-04-09 00:00:00.000000', 25, 5);
        """
    )
    return conn


def failed_checks(conn: sqlite3.Connection) -> list[str]:
    row = mechanical_row(conn)
    assert row["passed"] is (not row["failed"])
    return list(row["failed"])


def test_mechanical_passes_a_valid_store() -> None:
    assert failed_checks(valid_store()) == []


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        (
            "update account_events set timestamp = '2026-03-09 00:00:00.000000' "
            "where event_id = 'e1'",
            ["event_before_account_creation"],
        ),
        (
            "update linkage_observations set timestamp = '2026-03-09 00:00:00.000000'",
            ["linkage_before_account_creation"],
        ),
        (
            f"update account_events set timestamp = '{END}' where event_id = 'e4'",
            ["cycle_consumed_not_event_sum", "event_not_before_world_end"],
        ),
        (
            f"update linkage_observations set timestamp = '{END}'",
            ["linkage_not_before_world_end"],
        ),
        (
            f"update accounts set created_at = '{END}'",
            [
                "account_not_before_world_end",
                "event_before_account_creation",
                "linkage_before_account_creation",
            ],
        ),  # fmt: skip
        (
            "update entitlement_cycles set credits_consumed = 26, credits_granted = 25",
            ["cycle_consumed_not_event_sum", "cycle_consumed_over_granted"],
        ),
        (
            "update entitlement_cycles set credits_consumed = 4",
            ["cycle_consumed_not_event_sum"],
        ),
        (
            "update entitlement_cycles set cycle_start = '2025-12-01 00:00:00.000000', "
            "cycle_end = '2025-12-31 00:00:00.000000'",
            ["cycle_consumed_not_event_sum", "cycle_outside_window"],
        ),
        (
            f"update entitlement_cycles set cycle_start = '{END}'",
            [
                "cycle_consumed_not_event_sum",
                "cycle_outside_window",
                "cycle_start_not_before_world_end",
            ],
        ),
        (
            "insert into account_events values "
            "('e1', 'a1', '2026-03-13 00:00:00.000000', 'LOGIN', null)",
            ["duplicate_event_id"],
        ),
        (
            "insert into accounts values ('a1', '2026-03-10 00:00:00.000000')",
            ["duplicate_account_id"],
        ),
        (
            "insert into linkage_observations values ('o1', 'a1', '2026-03-11 00:00:00.000000')",
            ["duplicate_observation_id"],
        ),
        (
            "insert into entitlement_cycles values ('c1', 'a1', '2026-04-09 00:00:00.000000', "
            "'2026-05-09 00:00:00.000000', 25, 0)",
            ["duplicate_cycle_id"],
        ),
    ],
)
def test_mechanical_flags_each_violation(statement: str, expected: list[str]) -> None:
    conn = valid_store()
    conn.execute(statement)
    assert sorted(failed_checks(conn)) == sorted(expected)


# --- channel tag ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uppers", "best", "tag"),
    [
        ([0.55, 0.58, 0.60], 0.58, "T"),  # every bound at most 0.60 (inclusive)
        ([0.55, 0.58, 0.601], 0.58, "M"),  # one bound just over
        ([0.55, 0.58, 0.9], 0.79, "M"),  # F needs 0.80
        ([0.90, 0.95, 0.5], 0.80, "F"),  # exactly 0.80 is F
        ([0.55, 0.58, 0.5], 0.95, "T"),  # the T rule is checked first
    ],
)
def test_measured_tag(uppers: list[float], best: float, tag: str) -> None:
    assert measured_tag(uppers, best) == tag


def channel_units(informative: bool, seed: int = 5) -> list[ChannelUnit]:
    """100 abusive + 400 twin groups; with `informative`, f12 feature 0 carries the label."""
    rng = np.random.default_rng(seed)
    units = []
    for i in range(500):
        positive = i < 100
        f12 = rng.normal(0, 1, len(F_IDS))
        if informative and positive:
            f12[0] += 4.0
        units.append(
            ChannelUnit(
                f"g{i}",
                "fam" if positive else "twin",
                positive,
                tuple(f12),
                tuple(rng.normal(0, 1, len(FEATURES))),
                float(rng.normal(0, 1)),
            )
        )
    return units


def test_channel_pair_tags_a_visible_channel_f_and_a_hidden_one_t() -> None:
    visible = channel_pair(1, channel_units(True))
    assert visible["measured"] == "F" and visible["designed"] == "F" and visible["matches"]
    assert visible["per_feature_auc"]["F-AGE-1"] > 0.95
    assert visible["best_f12_auc"] >= 0.95
    assert (visible["n_pos"], visible["n_neg"]) == (100, 400)
    assert sorted(visible["f12_models"]) == ["hist_gradient_boosting", "logistic_l2"]
    assert sorted(visible["f64_w90_auc"]) == ["hist_gradient_boosting", "logistic_l2"]
    assert len(visible["per_feature_auc"]) == 12

    hidden = channel_pair(2, channel_units(False))
    assert hidden["measured"] == "T" and hidden["designed"] == "T" and hidden["matches"]
    for score in (*hidden["f12_models"].values(), hidden["detector"]):
        assert score["upper"] <= 0.60

    # A pair designed as T whose channel is visible must not match.
    assert channel_pair(2, channel_units(True))["matches"] is False
    # And an M design (pair 5) with a hidden channel measures T, which doesn't match.
    assert channel_pair(5, channel_units(False))["measured"] == "T"


# --- worlds (small scale, test-only) --------------------------------------------------------------


def stub_tokens(texts: Any) -> list[int]:
    return [len(t) // 4 for t in texts]


@pytest.fixture(scope="module")
def world_s1(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("health") / "s1"
    run_world(WorldSpec(3, 1.0, "small"), out, token_counter=stub_tokens)
    return out


@pytest.fixture(scope="module")
def world_s0(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("health") / "s0"
    run_world(WorldSpec(3, 0.0, "small"), out, token_counter=stub_tokens)
    return out


def by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["name"]: r for r in report["rows"]}


# Rows that must pass on any fair small world. Render visibility's AUC threshold is statistical
# (4 abusive groups per pair can miss 0.85 by chance), so only its `differs == 0` is asserted.
STRUCTURAL = (
    "mechanical",
    "determinism",
    "order_coupling",
    "linkage",
    "render_length",
    "parity_f12",
)


def test_world_health_at_s1_has_the_s1_rows(world_s1: Path) -> None:
    report = world_health(world_s1, parity_per_family=1)
    rows = by_name(report)
    assert list(rows) == [
        "mechanical",
        "determinism",
        "order_coupling",
        "nuisance_balance",
        "positive_control",
        "oracle_undefined",
        "linkage",
        "render_visibility",
        "render_length",
        "parity_f12",
    ]
    assert (report["units"], report["mixed"]) == (120, 0)
    assert report["manifest"]["s"] == 1.0 and report["manifest"]["scale"] == "small"
    assert report["passed"] is all(r["passed"] for r in report["rows"])
    for name in STRUCTURAL:
        assert rows[name]["passed"] is True, rows[name]
    assert rows["render_visibility"]["differs"] == 0
    assert rows["parity_f12"]["checked"] == 12  # one unit for each of the 12 families
    assert rows["linkage"]["share"] == 1.0
    # one Bonferroni family per world (owner, 2026-10-21): order 3 + nuisance 148 + render length 6
    assert (
        rows["nuisance_balance"]["row_items"] == 148
    )  # 6 pairs x 25 variables, minus pair 5's two
    family = {"order_coupling": 3, "nuisance_balance": 148, "render_length": 6}
    assert report["null_family"] == {
        "tests": 157,
        "alpha_adj": pytest.approx(0.05 / 157),
        "rows": list(family),
    }
    for name, items in family.items():
        assert rows[name]["family"] == "world" and rows[name]["row_items"] == items
        assert rows[name]["tests"] == 157 and rows[name]["alpha_adj"] == pytest.approx(0.05 / 157)
    assert rows["positive_control"]["minimum"] == 0.90
    assert json.loads((world_s1 / "health.json").read_text()) == report


def test_world_health_at_s0_has_the_negative_control(world_s0: Path) -> None:
    report = world_health(world_s0, parity_per_family=1)
    rows = by_name(report)
    assert "negative_control" in rows and "positive_control" not in rows
    # the s = 0 family adds the six negative-control items: 3 + 148 + 6 + 6
    assert rows["negative_control"]["row_items"] == 6 and rows["negative_control"]["tests"] == 163
    assert report["null_family"]["tests"] == 163
    assert report["null_family"]["rows"] == [
        "order_coupling",
        "nuisance_balance",
        "negative_control",
        "render_length",
    ]
    assert sorted(rows["negative_control"]["results"]) == [f"pair{p}" for p in PAIRS]
    for name in STRUCTURAL:
        assert rows[name]["passed"] is True, rows[name]


def test_determinism_row_detects_a_changed_checksum(world_s1: Path) -> None:
    manifest = json.loads((world_s1 / "manifest.json").read_text())
    assert determinism_row(manifest)["passed"] is True
    assert determinism_row(manifest)["current_generator_version"] == GENERATOR_VERSION
    tampered = {**manifest, "checksum_sha256": "0" * 64}
    assert determinism_row(tampered)["passed"] is False


def test_world_health_mechanical_runs_on_the_persisted_store(world_s1: Path) -> None:
    with sqlite3.connect(f"file:{world_s1 / 'inference.db'}?mode=ro", uri=True) as conn:
        row = mechanical_row(conn)
    assert row["passed"] is True
    assert all(v == 0 for v in row["violations"].values())


def test_channel_check_on_small_worlds(world_s1: Path, world_s0: Path, tmp_path: Path) -> None:
    out = tmp_path / "channel.json"
    report = channel_check(
        world_s1,
        [world_s1, world_s1],
        out,
        channel_scales=("small",),
        standard_scales=("small",),
        folds=2,
    )
    assert sorted(report["pairs"]) == ["1", "2", "3", "4", "5", "6"]
    for pair, record in report["pairs"].items():
        assert record["measured"] in {"T", "F", "M"}
        assert record["designed"] == health.CHANNEL_TAGS[int(pair)]
        assert record["matches"] is (record["measured"] == record["designed"])
        assert record["frozen"] == health.MEASURED_TAGS[int(pair)]
        assert record["matches_frozen"] is (record["measured"] == record["frozen"])
        assert (record["n_pos"], record["n_neg"]) == ((8, 32) if pair in "15" else (4, 16))
    assert report["pairs"]["2"]["worlds"] == "channel"
    assert report["pairs"]["1"]["worlds"] == "standard pooled"
    # SPEC §5.4: only a T-designed pair that doesn't measure T fails the check.
    assert report["passed"] is all(
        r["measured"] == "T" for r in report["pairs"].values() if r["designed"] == "T"
    )
    assert report["measured_tags"] == {p: r["measured"] for p, r in report["pairs"].items()}
    assert json.loads(out.read_text()) == report
    with pytest.raises(ValueError, match="expected s = 1"):
        channel_check(
            world_s0,
            [world_s1],
            tmp_path / "x.json",
            channel_scales=("small",),
            standard_scales=("small",),
        )
    with pytest.raises(ValueError, match="expected s = 1"):
        channel_check(world_s1, [world_s1], tmp_path / "x.json")  # `small` is not a reported scale


# --- generator-only sample ------------------------------------------------------------------------


def test_gensample_health_structure(tmp_path: Path) -> None:
    report = gensample_health(seed=201, n=50, out=tmp_path / "gs")
    rows = by_name(report)
    assert list(rows) == [
        "nuisance_balance_s0",
        "nuisance_balance_s1",
        "negative_control",
        "positive_control",
        "oracle_undefined_s0",
        "oracle_undefined_s1",
    ]
    assert (report["seed"], report["groups_per_side"]) == (201, 50)
    assert report["passed"] is all(r["passed"] for r in report["rows"])
    for name in ("nuisance_balance_s0", "nuisance_balance_s1"):
        assert rows[name]["tests"] == 148 == len(rows[name]["results"])
    assert rows["negative_control"]["tests"] == 6
    # the generator-only sample keeps per-row families (owner, 2026-10-21)
    assert "null_family" not in report and not any("family" in r for r in report["rows"])
    assert sorted(rows["negative_control"]["results"]) == [f"pair{p}" for p in PAIRS]
    assert sorted(rows["positive_control"]["results"]) == [f"pair{p}" for p in PAIRS]
    assert rows["positive_control"]["minimum"] == 0.90
    assert rows["negative_control"]["results"]["pair1"]["n_pos"] == 50
    assert json.loads((tmp_path / "gs" / "health.json").read_text()) == report
