"""Known-answer tests for testbed/eval/r0.py (SPEC §6.3, §6.4, §7: R0).

The scoring and the report are driven with synthetic `R0Unit` lists, so no store is opened. One
test-only `small` world (4 + 16 groups per pair) covers `load_world` and `run` end to end.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from testbed.eval import cv, r0
from testbed.eval.metrics import auc, recall_at_fpr
from testbed.eval.r0 import RUNGS, TEST_PAIRS, R0Unit
from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES as F64_NAMES
from testbed.pipeline.run_world import run_world
from testbed.world.families import MEASURED_TAGS, PAIRS
from testbed.world.generator import WorldSpec

T_A, T_B = (4, 6), (2, 3)  # the T pairs of each test side

Scores = dict[tuple[str, str], NDArray[np.float64]]


def make_units(
    n_pos: int = 8,
    n_neg: int = 16,
    *,
    pairs: Sequence[int] = tuple(PAIRS),
    pair_column: bool = False,
    oracle: bool = True,
) -> list[R0Unit]:
    """Units where feature 0 perfectly separates family from twin in every pair; with
    `pair_column`, feature 1 holds the pair number so a test can see what a fit was given."""
    rng = np.random.default_rng(0)
    units = []
    for pair in pairs:
        for k in range(n_pos + n_neg):
            positive = k < n_pos
            f12, f64 = rng.normal(0, 0.1, len(F_IDS)), rng.normal(0, 0.1, len(F64_NAMES))
            for vector in (f12, f64):
                vector[0] += 4.0 * positive
                if pair_column:
                    vector[1] = pair
            units.append(
                R0Unit(
                    "w",
                    f"w:p{pair}-{k:03d}",
                    PAIRS[pair][0 if positive else 1],
                    pair,
                    positive,
                    tuple(f12),
                    tuple(f64),
                    float(rng.random()),
                    float(k + 100.0 * positive) if oracle and k % 7 else None,
                )
            )
    return units


def same_scores(values: NDArray[np.float64]) -> Scores:
    return {(rung, regime): values.copy() for rung in RUNGS for regime in r0.REGIMES}


# --- Layout ---------------------------------------------------------------------------------------


def test_side_of() -> None:
    assert [r0.side_of(p) for p in (1, 2, 3)] == ["B", "B", "B"]
    assert [r0.side_of(p) for p in (4, 5, 6)] == ["A", "A", "A"]
    with pytest.raises(ValueError):
        r0.side_of(7)


def test_t_pairs_per_side_follow_the_frozen_tags() -> None:
    assert r0.t_pairs("A") == T_A and r0.t_pairs("B") == T_B
    assert all(MEASURED_TAGS[p] == "T" for p in (*T_A, *T_B))
    assert set(TEST_PAIRS["A"]) | set(TEST_PAIRS["B"]) == set(PAIRS)


# --- Scores ---------------------------------------------------------------------------------------


def test_oracle_fill_is_per_pair() -> None:
    def unit(pair: int, value: float | None) -> R0Unit:
        return R0Unit("w", f"w:{pair}-{value}", "f", pair, True, (), (), 0.0, value)

    units = [
        unit(1, 5.0), unit(1, 7.0), unit(1, None),
        unit(2, 100.0), unit(2, 120.0), unit(2, None),
        unit(3, None),
    ]  # fmt: skip
    # one below each pair's smallest defined value; 0 where nothing is defined
    assert r0._oracle_scores(units).tolist() == [5.0, 7.0, 4.0, 100.0, 120.0, 99.0, 0.0]


@pytest.fixture(scope="module")
def pool() -> tuple[list[R0Unit], Scores]:
    units = make_units()
    return units, r0.score_pool(units)


def test_score_pool_holds_every_rung_and_regime_aligned(pool: tuple[list[R0Unit], Scores]) -> None:
    units, scores = pool
    assert set(scores) == {(rung, regime) for rung in RUNGS for regime in r0.REGIMES}
    assert all(v.shape == (len(units),) and np.isfinite(v).all() for v in scores.values())


def test_random_and_detector_are_identical_across_regimes(
    pool: tuple[list[R0Unit], Scores],
) -> None:
    units, scores = pool
    for rung in ("random", "detector", "oracle"):
        assert np.array_equal(scores[(rung, "in_family")], scores[(rung, "held_out")])
    expected = np.random.default_rng(r0.RANDOM_SEED).random(len(units))
    assert np.array_equal(scores[("random", "held_out")], expected)
    assert scores[("detector", "held_out")].tolist() == [u.detector for u in units]


def test_fitted_rungs_separate_every_family_from_its_twin(
    pool: tuple[list[R0Unit], Scores],
) -> None:
    units, scores = pool
    report = r0.evaluate(units, scores)
    for regime in r0.REGIMES:
        for rung in r0.FITTED:
            row = report["regimes"][regime][rung]
            assert [row[f"family:{p}"]["auc"] for p in PAIRS] == [1.0] * 6, (regime, rung)
            assert row["primary"]["auc"] == 1.0 and row["pooled"]["auc"] == 1.0
            assert row["primary"]["lo"] == row["primary"]["hi"] == 1.0  # no resample disagrees
            assert row["primary"]["recall_fpr05"] == 1.0


def test_held_out_never_fits_on_a_test_side_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    units = make_units(4, 8, pair_column=True)
    calls: list[tuple[set[int], set[int]]] = []

    def fake(x_train: Any, y_train: Any, x_test: Any, *, model: str) -> NDArray[np.float64]:
        train, test = set(x_train[:, 1].astype(int)), set(x_test[:, 1].astype(int))
        calls.append((train, test))
        # a score that names the fold that tested the unit
        return np.full(len(x_test), 0.25 if test <= set(TEST_PAIRS["A"]) else 0.75)

    monkeypatch.setattr(cv, "fit_predict", fake)
    monkeypatch.setattr(cv, "oof_scores", lambda x, y, **kw: np.zeros(len(y)))
    scores = r0.score_pool(units)
    assert len(calls) == 6  # three fitted rungs x two folds
    # fold B (tests pairs 1-3) trained on 4-6, fold A (tests 4-6) on 1-3, once per fitted rung
    assert (
        sorted(calls, key=lambda c: min(c[1]))
        == [({4, 5, 6}, {1, 2, 3})] * 3 + [({1, 2, 3}, {4, 5, 6})] * 3
    )
    assert all(not (train & test) for train, test in calls)
    held = scores[("f64_trees", "held_out")]
    for u, value in zip(units, held, strict=True):
        assert value == (0.25 if u.pair in TEST_PAIRS["A"] else 0.75)  # one score, from its fold


def test_in_family_puts_each_group_in_exactly_one_test_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units = make_units(5, 10)
    seen: list[dict[str, Any]] = []

    def fake(x: Any, y: Any, **kwargs: Any) -> NDArray[np.float64]:
        seen.append(kwargs)
        return np.zeros(len(y))

    monkeypatch.setattr(cv, "oof_scores", fake)
    r0.score_pool(units)
    assert [k["model"] for k in seen] == ["logistic_l2", "logistic_l2", "hist_gradient_boosting"]
    for kwargs in seen:
        assert kwargs["seed"] == r0.CV_SEED and kwargs["folds"] == 5
        assert kwargs["groups"] == [u.group for u in units]
        assert kwargs["strata"] == [
            f"{'abusive' if u.positive else 'legitimate'}|{u.family}" for u in units
        ]
    folds = cv.group_folds(seen[0]["strata"], seen[0]["groups"], folds=5, seed=r0.CV_SEED)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(len(units)))  # every unit tested exactly once


# --- Report ---------------------------------------------------------------------------------------


def _two_side_scores(units: Sequence[R0Unit]) -> Scores:
    """Scores whose side AUCs are known. Side A's T pairs (4, 6): positives at 1, and in each
    pair one negative at 2 and the rest at 0, so the AUC is 0.5. Side B's (2, 3): positives at
    0.5 above negatives at 0, so 1.0. Pairs 1 and 5 are perfect, so a primary that leaked them
    would move."""
    s = np.zeros(len(units))
    for i, u in enumerate(units):
        if u.pair in (1, 5):
            s[i] = 10.0 * u.positive
        elif u.pair in T_A:
            s[i] = 1.0 if u.positive else (2.0 if i == _first_negative(units, u.pair) else 0.0)
        elif u.pair in T_B:
            s[i] = 0.5 * u.positive
    return same_scores(s)


def _first_negative(units: Sequence[R0Unit], pair: int) -> int:
    return next(i for i, u in enumerate(units) if u.pair == pair and not u.positive)


def test_primary_is_the_mean_of_the_two_side_aucs_over_t_pairs_only() -> None:
    units = make_units(2, 2, oracle=False)
    report = r0.evaluate(units, _two_side_scores(units))
    for regime in r0.REGIMES:
        row = report["regimes"][regime]["detector"]
        primary = row["primary"]
        # side A: positives 1 against negatives {2, 0, 2, 0}: 2 wins of 4 per positive
        assert primary["sides"]["A"]["auc"] == 0.5
        assert primary["sides"]["B"]["auc"] == 1.0
        assert primary["auc"] == 0.75
        assert (primary["sides"]["A"]["n_pos"], primary["sides"]["A"]["n_neg"]) == (4, 4)
        assert (primary["n_pos"], primary["n_neg"]) == (8, 8)
        assert primary["lo"] <= primary["auc"] <= primary["hi"]
        assert primary["sides"]["B"]["recall_fpr05"] == 1.0  # side B separates perfectly
        # pairs 1 and 5 are perfect on their own, and the primary doesn't see them
        assert row["family:1"]["auc"] == row["family:5"]["auc"] == 1.0


def test_recall_at_fpr_is_reported_per_side_and_pooled() -> None:
    units = make_units(8, 32, oracle=False)
    rng = np.random.default_rng(5)
    values = rng.random(len(units)) + np.array([u.positive for u in units]) * 0.5
    report = r0.evaluate(units, same_scores(values))
    entry = report["regimes"]["held_out"]["detector"]
    a = [i for i, u in enumerate(units) if u.pair in T_A]
    pos = values[[i for i in a if units[i].positive]]
    neg = values[[i for i in a if not units[i].positive]]
    assert entry["primary"]["sides"]["A"]["recall_fpr10"] == recall_at_fpr(pos, neg, 0.10)
    assert entry["primary"]["recall_fpr05"] == pytest.approx(
        np.mean([entry["primary"]["sides"][s]["recall_fpr05"] for s in "AB"])
    )
    everyone = [u.positive for u in units]
    assert entry["pooled"]["recall_fpr05"] == recall_at_fpr(
        values[np.array(everyone)], values[~np.array(everyone)], 0.05
    )


def test_family_channel_and_pooled_comparisons() -> None:
    units = make_units(8, 16)
    rng = np.random.default_rng(2)
    values = rng.random(len(units))
    row = r0.evaluate(units, same_scores(values))["regimes"]["held_out"]["random"]
    mask = np.array([u.pair == 3 for u in units])
    pos = np.array([u.positive for u in units])
    assert row["family:3"]["auc"] == auc(values[mask & pos], values[mask & ~pos])
    assert (row["family:3"]["n_pos"], row["family:3"]["n_neg"]) == (8, 16)
    t_mask = np.isin([u.pair for u in units], [2, 3, 4, 6])
    assert row["channel:T"]["auc"] == auc(values[t_mask & pos], values[t_mask & ~pos])
    assert (row["channel:T"]["n_pos"], row["channel:T"]["n_neg"]) == (32, 64)
    assert (row["channel:F"]["n_pos"], row["channel:M"]["n_pos"]) == (8, 8)  # pairs 5 and 1
    assert row["pooled"]["auc"] == auc(values[pos], values[~pos])
    assert (row["pooled"]["n_pos"], row["pooled"]["n_neg"]) == (48, 96)
    for key in ("family:3", "channel:T", "pooled"):
        assert row[key]["lo"] <= row[key]["auc"] <= row[key]["hi"]


def test_interval_is_the_percentile_interval_of_the_bootstrap() -> None:
    units = make_units(8, 16, oracle=False)
    values = np.random.default_rng(9).random(len(units))
    report = r0.evaluate(units, same_scores(values))
    again = r0.evaluate(units, same_scores(values))
    assert report == again  # seeded, so a rerun replays
    entry = report["regimes"]["held_out"]["detector"]["family:2"]
    assert entry["lo"] < entry["auc"] < entry["hi"]
    assert 0.0 < entry["hi"] - entry["lo"] < 0.5


def test_oracle_reports_family_aucs_and_their_means() -> None:
    units = make_units(8, 16)
    scores = r0.score_pool(units)
    row = r0.evaluate(units, scores)["regimes"]["held_out"]["oracle"]
    by_pair = {p: row[f"family:{p}"] for p in PAIRS}
    assert all("lo" in v and "hi" in v and "auc" in v for v in by_pair.values())
    for key in ("primary", "channel:T", "channel:F", "channel:M", "pooled"):
        assert "auc" not in row[key] and "lo" not in row[key] and "hi" not in row[key], key
    aucs = {p: v["auc"] for p, v in by_pair.items()}
    assert row["primary"]["mean_family_auc"] == pytest.approx(
        np.mean([aucs[p] for p in (2, 3, 4, 6)])
    )
    assert sorted(row["primary"]["families"]) == [2, 3, 4, 6]
    assert row["primary"]["sides"]["A"]["mean_family_auc"] == pytest.approx((aucs[4] + aucs[6]) / 2)
    assert row["channel:T"]["mean_family_auc"] == row["primary"]["mean_family_auc"]
    assert row["channel:F"]["mean_family_auc"] == aucs[5]
    assert row["channel:M"]["mean_family_auc"] == aucs[1]
    assert row["pooled"]["mean_family_auc"] == pytest.approx(np.mean(list(aucs.values())))
    assert row["pooled"]["n_pos"] == 48 and row["pooled"]["n_neg"] == 96
    # the oracle is the same under both regimes
    other = r0.evaluate(units, scores)["regimes"]["in_family"]["oracle"]
    assert other == row


def test_paired_difference_of_a_rung_with_itself_is_exactly_zero() -> None:
    units = make_units(8, 16, oracle=False)
    values = np.random.default_rng(4).random(len(units))
    scores = same_scores(values)
    diffs = r0.evaluate(units, scores)["differences"]
    for regime in r0.REGIMES:
        for key in ("f64_trees_minus_f64_logistic", "stronger_f64_minus_detector"):
            d = diffs[regime][key]
            assert d["point"] == d["lo"] == d["hi"] == 0.0, (regime, key)
            assert (d["n_pos"], d["n_neg"]) == (32, 64)  # four T pairs x (8, 16)


def test_paired_differences_use_the_stronger_f64_and_the_same_resamples() -> None:
    units = make_units(8, 16, oracle=False)
    rng = np.random.default_rng(6)
    pos = np.array([u.positive for u in units])
    scores = same_scores(rng.random(len(units)))
    scores[("f64_trees", "held_out")] = rng.random(len(units)) + 0.8 * pos
    report = r0.evaluate(units, scores)
    d = report["differences"]["held_out"]
    primary = {
        r: report["regimes"]["held_out"][r]["primary"]["auc"] for r in r0.RUNGS if r != "oracle"
    }
    assert d["f64_trees_minus_f64_logistic"]["point"] == pytest.approx(
        primary["f64_trees"] - primary["f64_logistic"]
    )
    assert d["stronger_f64_minus_detector"]["f64"] == "f64_trees"
    assert d["stronger_f64_minus_detector"]["point"] == pytest.approx(
        primary["f64_trees"] - primary["detector"]
    )
    lo, hi = d["f64_trees_minus_f64_logistic"]["lo"], d["f64_trees_minus_f64_logistic"]["hi"]
    assert 0 < lo < d["f64_trees_minus_f64_logistic"]["point"] < hi
    # in the regime where all F64 scores tie, either could be "stronger", and the gap is 0
    assert report["differences"]["in_family"]["f64_trees_minus_f64_logistic"]["point"] == 0.0


def test_primary_comparisons_and_boot_on_known_scores() -> None:
    units = make_units(2, 2, oracle=False)
    comps = r0.primary_comparisons(units)
    assert list(comps) == ["A", "B"]
    pair = np.array([u.pair for u in units])
    for side, pairs in (("A", T_A), ("B", T_B)):  # the side's T units only, split by class
        c = comps[side]
        assert set(pair[c.pos]) == set(pair[c.neg]) == set(pairs)
        assert all(units[i].positive for i in c.pos) and not any(units[i].positive for i in c.neg)
        assert c.w_pos.shape == (r0.N_BOOT, 4) and c.w_neg.shape == (r0.N_BOOT, 4)
    point, boot = r0.primary_boot(_two_side_scores(units)[("detector", "held_out")], comps)
    assert point == 0.75  # side A 0.5, side B 1.0
    assert boot.shape == (r0.N_BOOT,) and 0.5 <= boot.min() <= boot.max() <= 1.0
    assert np.array_equal(  # one draw, from BOOT_SEED, over (side, class) strata
        np.concatenate([comps["A"].w_pos, comps["A"].w_neg], axis=1).sum(axis=1),
        np.full(r0.N_BOOT, 8),
    )
    with pytest.raises(ValueError, match="primary: side"):
        r0.primary_comparisons([u for u in units if u.pair not in T_B])


def test_evaluate_reports_extra_rungs_and_paired_differences() -> None:
    units = make_units(8, 16, oracle=False)
    rng = np.random.default_rng(7)
    pos = np.array([u.positive for u in units])
    scores = same_scores(rng.random(len(units)))
    scores[("probe", "held_out")] = rng.random(len(units)) + 0.8 * pos
    scores[("probe", "in_family")] = scores[("probe", "held_out")].copy()
    rungs = (*RUNGS, "probe")
    plain = r0.evaluate(units, {k: v for k, v in scores.items() if k[0] != "probe"})
    report = r0.evaluate(
        units,
        scores,
        rungs=rungs,
        extra_differences=[("probe", "f64_logistic"), ("probe", "detector")],
    )
    assert report["regimes"]["held_out"]["probe"]["primary"]["lo"] > 0.5
    assert report["regimes"]["held_out"]["detector"] == plain["regimes"]["held_out"]["detector"]
    d = report["differences"]["held_out"]
    assert {"probe_minus_f64_logistic", "probe_minus_detector"} <= set(d)
    probe = report["regimes"]["held_out"]["probe"]["primary"]["auc"]
    base = report["regimes"]["held_out"]["f64_logistic"]["primary"]["auc"]
    assert d["probe_minus_f64_logistic"]["point"] == pytest.approx(probe - base)
    assert 0 < d["probe_minus_f64_logistic"]["lo"] < d["probe_minus_f64_logistic"]["hi"]
    assert (d["probe_minus_detector"]["n_pos"], d["probe_minus_detector"]["n_neg"]) == (32, 64)
    # the default differences are untouched by the extra ones
    assert (
        d["f64_trees_minus_f64_logistic"]
        == plain["differences"]["held_out"]["f64_trees_minus_f64_logistic"]
    )
    for a, b in ((("probe", "oracle")), ("probe", "nope"), ("nope", "probe")):
        with pytest.raises(ValueError, match="two real rungs"):
            r0.evaluate(units, scores, rungs=rungs, extra_differences=[(a, b)])
    with pytest.raises(ValueError, match="rung x regime"):  # scores must match `rungs` exactly
        r0.evaluate(units, scores)
    only_random = {k: v for k, v in scores.items() if k[0] == "random"}
    with pytest.raises(ValueError, match="detector and the fitted"):
        r0.evaluate(units, only_random, rungs=("random",))


def test_a_missing_pair_or_class_raises_instead_of_returning_nan() -> None:
    no_pair = make_units(4, 8, pairs=(1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="family:6"):
        r0.evaluate(no_pair, same_scores(np.zeros(len(no_pair))))
    one_class = [u for u in make_units(4, 8) if not (u.pair == 2 and u.positive)]
    with pytest.raises(ValueError, match="family:2"):
        r0.evaluate(one_class, same_scores(np.zeros(len(one_class))))
    units = make_units(4, 8)
    scores = same_scores(np.zeros(len(units)))
    scores[("detector", "held_out")][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        r0.evaluate(units, scores)
    del scores[("detector", "held_out")]
    with pytest.raises(ValueError, match="rung x regime"):
        r0.evaluate(units, scores)


# --- load_world and run, end to end on the test-only small world ----------------------------------


@pytest.fixture(scope="module")
def small_world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("small") / "world"
    record = run_world(
        WorldSpec(seed=1, s=1.0, scale="small"),
        out,
        token_counter=lambda texts: [len(t) // 4 for t in texts],
    )
    return out, record


def test_load_world_reads_units_labels_and_oracle(small_world: tuple[Path, dict[str, Any]]) -> None:
    path, record = small_world
    s, units, mixed = r0.load_world(path, scales=("small",))
    assert s == 1.0 and mixed == 0
    assert len(units) == record["counts"]["units"] == 120
    assert [u.group for u in units] == sorted(u.group for u in units)
    assert len({u.group for u in units}) == 120 and all(u.world == "world" for u in units)
    assert sum(u.positive for u in units) == 24  # 6 pairs x 4 abusive groups
    assert all(u.positive == (u.family == PAIRS[u.pair][0]) for u in units)
    assert all(len(u.f12) == len(F_IDS) and len(u.f64) == len(F64_NAMES) for u in units)
    assert all(u.oracle is None or np.isfinite(u.oracle) for u in units)
    assert {u.pair for u in units} == set(PAIRS)
    rows = [json.loads(x) for x in (path / "units.jsonl").read_text().splitlines() if x]
    renders = [json.loads(x) for x in (path / "renders.jsonl").read_text().splitlines() if x]
    keys = {r["unit"] for r in rows}
    assert {u.unit for u in units} == keys and len(keys) == 120  # the unit key, one per unit
    assert keys <= {r["unit"] for r in renders}  # which is also the render's (and embedding's) id
    with pytest.raises(ValueError, match="scale"):
        r0.load_world(path)  # the default accepts only the standard scale


@pytest.mark.filterwarnings("ignore:The least populated class")  # 4 abusive groups per family
def test_run_writes_the_four_files(
    small_world: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    path, _ = small_world
    out = tmp_path / "r0"
    report = r0.run([path], out, scales=("small",))
    assert {p.name for p in out.iterdir()} == {
        "r0.json",
        "scores.jsonl",
        "curves.md",
        "auc-vs-s.svg",
    }
    assert json.loads((out / "r0.json").read_text()) == report
    result = report["by_s"]["1.0"]
    assert result["worlds"] == ["world"] and result["mixed"] == {"world": 0}
    assert result["n_units"] == 120 and result["n_boot"] == 2000
    for regime in r0.REGIMES:
        assert set(result["regimes"][regime]) == set(RUNGS)
        pooled = result["regimes"][regime]["f64_trees"]["pooled"]
        assert (pooled["n_pos"], pooled["n_neg"]) == (24, 96)
        assert result["regimes"][regime]["oracle"]["family:1"]["n_pos"] == 4
    primary = result["regimes"]["held_out"]["detector"]["primary"]
    assert (primary["n_pos"], primary["n_neg"]) == (16, 64)  # four T pairs x (4, 16)
    # the oracle reaches the positive-control level even on 4 + 16 groups per pair
    assert result["regimes"]["held_out"]["oracle"]["pooled"]["mean_family_auc"] > 0.9

    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 120 and {r["s"] for r in rows} == {1.0}
    assert all(set(r["scores"]) == set(RUNGS) for r in rows)
    assert all(set(r["scores"]["f64_trees"]) == {"in_family", "held_out"} for r in rows)
    assert all(r["side"] == r0.side_of(r["pair"]) for r in rows)

    assert "pair 1" in (out / "curves.md").read_text()
    svg = ET.fromstring((out / "auc-vs-s.svg").read_text())
    assert len(list(svg.iter("{http://www.w3.org/2000/svg}polyline"))) == 6

    with pytest.raises(FileExistsError):  # a non-empty out is refused, and left alone
        r0.run([path], out, scales=("small",))
    assert json.loads((out / "r0.json").read_text()) == report


def test_run_pools_worlds_that_share_an_s_and_rejects_duplicates(
    small_world: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    path, _ = small_world
    with pytest.raises(ValueError, match="unique"):
        r0.run([path, path], tmp_path / "dup", scales=("small",))
