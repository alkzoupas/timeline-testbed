"""Known-answer tests for testbed/eval/metrics.py and testbed/eval/cv.py."""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from testbed.eval import cv, metrics
from testbed.eval.metrics import (
    PERM_FLOOR,
    SCREEN_MARGIN,
    auc,
    auc_interval,
    auc_upper_bound,
    bootstrap_auc,
    delong_var,
    null_rule,
    null_variance,
    permutation_p,
)

Split = tuple[NDArray[np.intp], NDArray[np.intp]]


def _pairs(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """h[i, j] = 1 if pos_i > neg_j, 1/2 if tied, else 0."""
    diff = pos[:, None] - neg[None, :]
    return (diff > 0) + 0.5 * (diff == 0)


def _brute_delong(pos: np.ndarray, neg: np.ndarray) -> float:
    h = _pairs(pos, neg)
    s10 = h.mean(axis=1)  # per positive
    s01 = h.mean(axis=0)  # per negative
    return float(s10.var(ddof=1) / pos.size + s01.var(ddof=1) / neg.size)


# --- auc ---------------------------------------------------------------------------------


def test_auc_hand_examples() -> None:
    assert auc([3, 2], [1, 2]) == 0.875  # (1 + 1 + 1 + 0.5) / 4: the 2-vs-2 tie counts 1/2
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 3], [2, 4]) == 0.25  # only 3 > 2 wins
    assert auc([5, 5, 5], [5, 5]) == 0.5


def test_auc_matches_pairwise_with_ties() -> None:
    rng = np.random.default_rng(1)
    pos, neg = rng.integers(0, 6, 40).astype(float), rng.integers(0, 8, 70).astype(float)
    assert auc(pos, neg) == pytest.approx(_pairs(pos, neg).mean())


def test_auc_rejects_empty_and_nan() -> None:
    with pytest.raises(ValueError):
        auc([], [1.0])
    with pytest.raises(ValueError):
        auc([1.0, float("nan")], [1.0])


# --- delong_var --------------------------------------------------------------------------


def test_delong_var_hand_computed() -> None:
    # pos [3, 2], neg [1, 2]: V10 = [1, .75], V01 = [1, .75], each sample variance
    # 2 * .125**2 = 1/32, so var = (1/32)/2 + (1/32)/2 = 1/32.
    assert delong_var([3, 2], [1, 2]) == pytest.approx(1 / 32)


def test_delong_var_matches_brute_force_with_ties() -> None:
    rng = np.random.default_rng(2)
    pos, neg = rng.integers(0, 5, 17).astype(float), rng.integers(0, 7, 23).astype(float)
    assert delong_var(pos, neg) == pytest.approx(_brute_delong(pos, neg))


def test_delong_var_zero_at_extremes() -> None:
    assert delong_var([1, 1, 1], [1, 1, 1]) == 0.0
    assert delong_var([5, 6, 7], [1, 2, 3]) == 0.0


def test_delong_var_needs_two_per_class() -> None:
    with pytest.raises(ValueError):
        delong_var([1.0], [1.0, 2.0])


# --- null_variance -----------------------------------------------------------------------


def _all_relabelling_aucs(pooled: np.ndarray, n1: int) -> list[float]:
    """The AUC under every choice of which n1 pooled scores are the positives."""
    out = []
    for chosen in itertools.combinations(range(pooled.size), n1):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(chosen)] = True
        out.append(auc(pooled[mask], pooled[~mask]))
    return out


def test_null_variance_without_ties_is_n_plus_1_over_12_n1_n0() -> None:
    pos, neg = np.arange(0.0, 8.0, 2.0), np.arange(1.0, 12.0, 2.0)  # 4 + 6, all distinct
    assert null_variance(pos, neg) == pytest.approx(11 / (12 * 4 * 6))


def test_null_variance_with_ties_is_the_variance_over_all_relabellings() -> None:
    pos, neg = np.array([1.0, 2, 2, 3]), np.array([2.0, 2, 3, 5, 5, 1])
    exact = np.var(_all_relabelling_aucs(np.concatenate([pos, neg]), pos.size))
    assert null_variance(pos, neg) == pytest.approx(exact, rel=1e-12)
    assert null_variance(pos, neg) < 11 / (12 * 4 * 6)  # ties shrink it


def test_null_variance_is_zero_when_every_score_ties() -> None:
    assert null_variance([2.0] * 5, [2.0] * 9) == 0.0


# --- permutation_p -----------------------------------------------------------------------


def _exact_p(pos: np.ndarray, neg: np.ndarray) -> float:
    aucs = np.array(_all_relabelling_aucs(np.concatenate([pos, neg]), pos.size))
    observed = abs(auc(pos, neg) - 0.5)
    return float((np.abs(aucs - 0.5) >= observed - 1e-12).mean())


def test_permutation_p_matches_the_enumerated_p_value() -> None:
    pos, neg = np.array([5.0, 6, 3]), np.array([1.0, 2, 3, 4, 4])
    exact = _exact_p(pos, neg)  # 3 vs 5: 56 relabellings
    assert 0 < exact < 1
    p = permutation_p(pos, neg, n_perm=20_000, seed=1)
    assert p == pytest.approx(exact, abs=4 * (exact * (1 - exact) / 20_000) ** 0.5)


def test_permutation_p_is_deterministic_for_a_seed_and_never_zero() -> None:
    pos, neg = np.array([5.0, 6, 7]), np.array([1.0, 2, 3, 4, 4])
    assert permutation_p(pos, neg, n_perm=500, seed=3) == permutation_p(
        pos, neg, n_perm=500, seed=3
    )
    # the observed split is the most extreme of 56 relabellings, so p is near 1/56, not 0
    p = permutation_p(pos, neg, n_perm=500, seed=3)
    assert 0 < p < 0.1
    assert permutation_p(np.arange(60.0) + 100, np.arange(60.0), n_perm=50, seed=0) == 1 / 51


def test_permutation_p_is_the_same_whatever_the_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(8)
    pos, neg = rng.integers(0, 4, 30).astype(float), rng.integers(0, 4, 50).astype(float)
    whole = permutation_p(pos, neg, n_perm=999, seed=2)
    monkeypatch.setattr(metrics, "_CHUNK_CELLS", 80 * 7)  # 7 permutations per chunk
    assert permutation_p(pos, neg, n_perm=999, seed=2) == whole


# --- null_rule ---------------------------------------------------------------------------


def test_null_rule_equal_distributions_pass_on_the_screen() -> None:
    rng = np.random.default_rng(3)
    r = null_rule(rng.normal(size=400), rng.normal(size=1600), tests=12)
    assert r.passed and r.method == "screen"
    assert r.alpha_adj == pytest.approx(0.05 / 12)
    assert r.p == pytest.approx(2 * NormalDist().cdf(-abs(r.z)))


def test_null_rule_z_uses_the_exact_null_variance() -> None:
    pos, neg = [3, 2], [1, 2]
    r = null_rule(pos, neg, tests=1)
    assert r.auc == 0.875
    assert r.z == pytest.approx(0.375 / null_variance(pos, neg) ** 0.5)


def test_null_rule_constant_scores_pass_and_perfect_separation_fails() -> None:
    same = null_rule([2.0] * 5, [2.0] * 9, tests=3)
    assert same.passed and (same.auc, same.p, same.method) == (0.5, 1.0, "screen")
    perfect = null_rule(np.arange(40.0) + 1000, np.arange(160.0), tests=6)
    assert not perfect.passed and perfect.auc == 1.0 and perfect.method == "permutation"
    assert perfect.p == pytest.approx(1 / (PERM_FLOOR + 1))


def test_null_rule_borderline_goes_to_the_permutation_test() -> None:
    rng = np.random.default_rng(4)
    pos, neg = rng.normal(0.7, 1.0, 40), rng.normal(0.0, 1.0, 160)
    r = null_rule(pos, neg, tests=6)
    crit = NormalDist().inv_cdf(1 - r.alpha_adj / 2)
    assert abs(r.z) > crit - SCREEN_MARGIN and r.method == "permutation"
    assert r.passed == (r.p >= r.alpha_adj)
    assert r.p == null_rule(pos, neg, tests=6).p  # seeded


def test_null_rule_more_tests_widen_the_rule() -> None:
    rng = np.random.default_rng(4)
    pos, neg = rng.normal(0.7, 1.0, 40), rng.normal(0.0, 1.0, 160)
    assert not null_rule(pos, neg, tests=1).passed
    assert null_rule(pos, neg, tests=100).passed


def test_null_rule_sparse_binary_where_delong_fails_but_the_exact_test_passes() -> None:
    """8 ones, none among 40 positives: a fair draw, but DeLong's variance is too small."""
    pos, neg = np.zeros(40), np.zeros(160)
    neg[:8] = 1.0
    crit = NormalDist().inv_cdf(1 - 0.05 / 12)
    assert abs(auc(pos, neg) - 0.5) > crit * delong_var(pos, neg) ** 0.5  # old rule: fail
    r = null_rule(pos, neg, tests=6)
    assert r.passed and r.method == "permutation" and r.p > 0.05


def test_null_rule_calibration_on_fair_tied_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """400 fair rows of 6 heavily tied comparisons at 40 + 160: a row fails about 5% at most.
    Fewer permutations than production, to keep the test fast."""
    monkeypatch.setattr(metrics, "PERM_FLOOR", 2000)
    monkeypatch.setattr(metrics, "PERM_FACTOR", 10)
    rng = np.random.default_rng(5)
    failures = 0
    for rep in range(400):
        row = []
        for item in range(6):
            v = rng.choice([0.0, 1.0, 2.0], size=200, p=[0.9, 0.08, 0.02])
            row.append(null_rule(v[:40], v[40:], tests=6, seed=rep * 6 + item).passed)
        failures += not all(row)
    assert failures / 400 <= 0.05 + 0.03


def _sparse_exchangeability_case(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Round-1 review specimen: K positives at +1 and 40 - K at -1 against 160 zeros. Each
    +1 beats every zero and each -1 loses to every zero, so AUC = K / 40 (0.5 at K = 20), while the
    distributions of the two samples differ for every K."""
    return np.array([1.0] * k + [-1.0] * (40 - k)), np.zeros(160)


def _exact_sparse_p(k: int) -> float:
    """Exact two-sided permutation p of |AUC - 0.5| for the specimen, summing the multivariate
    hypergeometric over how many +1 (a) and -1 (b) scores fall in the positive class of 40."""
    observed = abs(auc(*_sparse_exchangeability_case(k)) - 0.5)
    total = math.comb(200, 40)
    hits = 0
    for a in range(0, min(k, 40) + 1):
        for b in range(0, min(40 - k, 40 - a) + 1):
            z = 40 - a - b
            if z > 160:
                continue
            pos = np.array([1.0] * a + [-1.0] * b + [0.0] * z)
            neg = np.array([1.0] * (k - a) + [-1.0] * (40 - k - b) + [0.0] * (160 - z))
            if abs(auc(pos, neg) - 0.5) >= observed - 1e-12:
                hits += math.comb(k, a) * math.comb(40 - k, b) * math.comb(160, z)
    return hits / total


def test_null_rule_is_an_exchangeability_test_not_an_auc_interval() -> None:
    """Round-1 finding 4: the rule's p is the exact permutation (exchangeability) p, not an AUC
    interval's. If positives are ±1 with equal probability against zeros, the population AUC is
    0.5 but the distributions differ. The sample 24 × (+1), 16 × (−1) has AUC 0.6 and exact
    two-sided p = 0.005092447416579459 < 0.05 / 6, so the row fails."""
    pos, neg = _sparse_exchangeability_case(24)
    exact = 0.005092447416579459
    assert _exact_sparse_p(24) == pytest.approx(exact, rel=1e-12)
    r = null_rule(pos, neg, tests=6)
    assert r.auc == pytest.approx(0.6)
    assert r.method == "permutation" and not r.passed
    assert exact < 0.05 / 6
    # Monte Carlo p: n_perm = max(PERM_FLOOR, ceil(PERM_FACTOR · tests / alpha)) = 20,000 draws.
    # Its sd is sqrt(p(1-p)/n_perm) ≈ 5.0e-4; allow 4 sd plus the (1 + hits)/(n_perm + 1) bias.
    n_perm = max(PERM_FLOOR, math.ceil(metrics.PERM_FACTOR / r.alpha_adj))
    assert n_perm == PERM_FLOOR
    tol = 4 * math.sqrt(exact * (1 - exact) / n_perm) + 1 / (n_perm + 1)
    assert r.p == pytest.approx(exact, abs=tol)
    # 20 × (+1) and 20 × (−1): AUC is exactly 0.5 and the rule passes
    even = null_rule(*_sparse_exchangeability_case(20), tests=6)
    assert even.auc == 0.5 and even.passed


def test_null_rule_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        null_rule([1, 2], [1, 2], tests=0)
    with pytest.raises(ValueError):
        null_rule([1, 2], [1, 2], tests=1, alpha=1.5)


# --- bootstrap ---------------------------------------------------------------------------


def test_bootstrap_deterministic_for_seed() -> None:
    rng = np.random.default_rng(6)
    pos, neg = rng.normal(0.5, 1, 30), rng.normal(0, 1, 90)
    a = bootstrap_auc(pos, neg, n_boot=200, seed=7)
    assert a.shape == (200,)
    assert np.array_equal(a, bootstrap_auc(pos, neg, n_boot=200, seed=7))
    assert not np.array_equal(a, bootstrap_auc(pos, neg, n_boot=200, seed=8))


def test_bootstrap_degenerate_inputs_have_known_answers() -> None:
    assert np.all(bootstrap_auc([5, 6, 7], [1, 2, 3, 4], n_boot=50, seed=0) == 1.0)
    assert np.all(bootstrap_auc([1, 2, 3], [5, 6, 7, 8], n_boot=50, seed=0) == 0.0)
    assert np.all(bootstrap_auc([2.0] * 4, [2.0] * 6, n_boot=50, seed=0) == 0.5)


def test_bootstrap_matches_direct_resampling() -> None:
    """Same distribution as drawing units with replacement and computing the AUC directly."""
    rng = np.random.default_rng(9)
    pos, neg = rng.integers(0, 6, 25).astype(float), rng.integers(0, 8, 60).astype(float)
    boot = bootstrap_auc(pos, neg, n_boot=4000, seed=10)
    direct = np.array(
        [auc(rng.choice(pos, pos.size), rng.choice(neg, neg.size)) for _ in range(1500)]
    )
    assert boot.mean() == pytest.approx(direct.mean(), abs=0.005)
    assert boot.std() == pytest.approx(direct.std(), rel=0.1)


def test_bootstrap_spread_tracks_delong() -> None:
    rng = np.random.default_rng(11)
    pos, neg = rng.normal(0.4, 1, 200), rng.normal(0, 1, 800)
    boot = bootstrap_auc(pos, neg, n_boot=2000, seed=12)
    assert boot.std() == pytest.approx(delong_var(pos, neg) ** 0.5, rel=0.1)
    assert boot.mean() == pytest.approx(auc(pos, neg), abs=0.005)


def test_interval_contains_auc_and_bounds_order() -> None:
    rng = np.random.default_rng(13)
    pos, neg = rng.normal(0.3, 1, 40), rng.normal(0, 1, 160)
    a, lo, hi = auc_interval(pos, neg, n_boot=1000, seed=14)
    assert a == auc(pos, neg) and lo <= a <= hi
    ub = auc_upper_bound(pos, neg, n_boot=1000, seed=14)
    assert ub == pytest.approx(
        float(np.quantile(bootstrap_auc(pos, neg, n_boot=1000, seed=14), 0.95))
    )
    assert a < ub < 1.0 and ub < hi  # the one-sided 95% bound sits below the two-sided 97.5%


# --- cv ----------------------------------------------------------------------------------


def _problem(signal: float, seed: int = 0) -> tuple[np.ndarray, ...]:
    """400 rows in 100 groups of 4 (a group has one label); 2 label x family strata."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(100), 4)
    label = (group % 2).astype(int)
    family = (group // 2 % 2).astype(int)
    X = rng.normal(size=(400, 5))
    X[:, 0] += signal * label
    strata = np.array([f"{y}-{f}" for y, f in zip(label, family, strict=True)])
    return X, label, group, strata


def test_models_names() -> None:
    assert set(cv.models()) == {"logistic_l2", "hist_gradient_boosting"}


def test_group_folds_put_each_group_in_one_test_fold() -> None:
    _, label, group, strata = _problem(0.0)
    splits = cv.group_folds(strata, group, folds=5, seed=1)
    assert len(splits) == 5
    tests = np.concatenate([t for _, t in splits])
    assert sorted(tests) == list(range(label.size))  # every row tested exactly once
    for train, test in splits:
        assert not set(group[train]) & set(group[test])  # no group on both sides
    home = {g: i for i, (_, t) in enumerate(splits) for g in set(group[t])}
    assert sorted(home) == sorted(set(group))  # each group has exactly one test fold


@pytest.mark.parametrize("model", ["logistic_l2", "hist_gradient_boosting"])
def test_oof_scores_shape_range_and_separable(model: str) -> None:
    X, y, group, strata = _problem(4.0)
    s = cv.oof_scores(X, y, groups=group, strata=strata, model=model, seed=2)
    assert s.shape == y.shape and s.min() >= 0.0 and s.max() <= 1.0
    assert auc(s[y == 1], s[y == 0]) > 0.95


@pytest.mark.parametrize("model", ["logistic_l2", "hist_gradient_boosting"])
def test_oof_scores_pure_noise_is_near_chance(model: str) -> None:
    X, y, group, strata = _problem(0.0, seed=3)
    s = cv.oof_scores(X, y, groups=group, strata=strata, model=model, seed=2)
    assert 0.35 < auc(s[y == 1], s[y == 0]) < 0.65


def test_oof_scores_train_never_sees_test_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y, group, strata = _problem(1.0)
    seen: list[tuple[set[int], set[int]]] = []
    real = cv.group_folds

    def spy(strata: ArrayLike, groups: ArrayLike, *, folds: int, seed: int) -> list[Split]:
        splits = real(strata, groups, folds=folds, seed=seed)
        seen.extend((set(group[a]), set(group[b])) for a, b in splits)
        return splits

    monkeypatch.setattr(cv, "group_folds", spy)
    cv.oof_scores(X, y, groups=group, strata=strata, model="logistic_l2", seed=4)
    assert len(seen) == 5
    assert all(not tr & te for tr, te in seen)
    assert set().union(*(te for _, te in seen)) == set(group.tolist())


def test_oof_scores_deterministic_and_rejects_non_binary() -> None:
    X, y, group, strata = _problem(2.0)
    a = cv.oof_scores(X, y, groups=group, strata=strata, model="logistic_l2", seed=5)
    b = cv.oof_scores(X, y, groups=group, strata=strata, model="logistic_l2", seed=5)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        cv.oof_scores(X, y + 1, groups=group, strata=strata, model="logistic_l2", seed=5)
