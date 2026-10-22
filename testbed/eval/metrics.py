"""AUC, DeLong variance, the permutation null rule (SPEC §5.6) and the group bootstrap (SPEC §6.4).

Pure functions of scores. They take positive and negative scores as arguments and never open a
store. One unit per group in this project, so resampling units is resampling groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

Floats = NDArray[np.float64]

NULL_SEED = 20261021  # default permutation seed of `null_rule`
SCREEN_MARGIN = 1.5  # |z| this far inside the critical value passes without a permutation test
PERM_FLOOR = 20_000  # fewest permutations of the exact test
PERM_FACTOR = 50  # permutations ≥ PERM_FACTOR / alpha_adj, so the smallest p resolves alpha_adj
_CHUNK_CELLS = 4_000_000  # permutations per chunk × pooled size: ~100 MB of working arrays


@dataclass(frozen=True)
class NullRule:
    auc: float
    z: float  # (auc - 0.5) / sd under the permutation null
    p: float  # two-sided: normal tail if method is "screen", Monte Carlo if "permutation"
    alpha_adj: float
    method: str
    passed: bool


def _scores(values: ArrayLike, name: str) -> Floats:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has non-finite scores")
    return arr


def auc(pos: ArrayLike, neg: ArrayLike) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5 P(pos == neg), by binary search on sorted scores."""
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    return float(_placements(p, n)[0].mean())  # share of negatives below, ties 1/2


def _placements(pos: Floats, neg: Floats) -> tuple[Floats, Floats]:
    """DeLong structural components: for each positive, the share of negatives below it (ties
    1/2); for each negative, the share of positives above it (ties 1/2)."""
    sn, sp = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sn, pos, "left") + np.searchsorted(sn, pos, "right")) / (2 * neg.size)
    v01 = (2 * pos.size - np.searchsorted(sp, neg, "right") - np.searchsorted(sp, neg, "left")) / (
        2 * pos.size
    )
    return v10, v01


def delong_var(pos: ArrayLike, neg: ArrayLike) -> float:
    """DeLong variance of the AUC: s10/m + s01/n over the structural components (ties count 1/2,
    which is the midrank treatment)."""
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    if p.size < 2 or n.size < 2:
        raise ValueError("DeLong variance needs at least 2 scores per class")
    v10, v01 = _placements(p, n)
    return float(v10.var(ddof=1) / p.size + v01.var(ddof=1) / n.size)


def null_variance(pos: ArrayLike, neg: ArrayLike) -> float:
    """Exact variance of the AUC over all relabellings of the pooled scores, ties included:
    `[(N + 1) - Σ(t³ - t) / (N(N - 1))] / (12·n1·n0)` over the pooled tie groups of size t."""
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    big_n = p.size + n.size
    if big_n < 2:
        raise ValueError("the null variance needs at least 2 scores")
    _, counts = np.unique(np.concatenate([p, n]), return_counts=True)
    tie_sum = sum(int(t) ** 3 - int(t) for t in counts)  # exact integers, so all-tied is 0
    numerator = (big_n + 1) * big_n * (big_n - 1) - tie_sum
    return numerator / (big_n * (big_n - 1) * 12 * p.size * n.size)


def _midranks(values: Floats) -> Floats:
    """1-based ranks, ties sharing their mean rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return np.asarray((np.cumsum(counts) - (counts - 1) / 2)[inverse], dtype=np.float64)


def permutation_p(pos: ArrayLike, neg: ArrayLike, *, n_perm: int, seed: int) -> float:
    """Two-sided Monte Carlo p-value of |AUC - 0.5| under random relabelling, class sizes fixed:
    `(1 + #{|AUC_b - 0.5| ≥ |AUC - 0.5| - 1e-12}) / (n_perm + 1)`, so it is never 0.

    A permuted AUC is the midrank sum of a random n1-subset of the pooled scores, so the draws
    are vectorised and chunked to bound memory (about 100 MB at N = 4,000).
    """
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1")
    n1, n0 = p.size, n.size
    ranks = _midranks(np.concatenate([p, n]))
    offset = n1 * (n1 + 1) / 2
    observed = abs((float(ranks[:n1].sum()) - offset) / (n1 * n0) - 0.5)
    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_CELLS // ranks.size)
    hits, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        noise = rng.random((b, ranks.size))
        # the n1 smallest of iid uniforms are a uniform random n1-subset
        subset = np.argpartition(noise, n1 - 1, axis=1)[:, :n1]
        sums = ranks[subset].sum(axis=1)
        hits += int((np.abs((sums - offset) / (n1 * n0) - 0.5) >= observed - 1e-12).sum())
        done += b
    return (1 + hits) / (n_perm + 1)


def null_rule(
    pos: ArrayLike, neg: ArrayLike, *, tests: int, alpha: float = 0.05, seed: int = NULL_SEED
) -> NullRule:
    """SPEC §5.6 null rule: a two-sided permutation test of exchangeability of the two samples.

    H0: family and twin scores come from the same distribution (SPEC §5.1(3)). The statistic is
    |AUC − 0.5|, tested at `alpha_adj = alpha / tests` (`tests` is every test in that row on that
    world or sample). A rejection means the distributions differ; it is not an AUC confidence
    interval. When the distributions differ but the population AUC is 0.5, the rule can still
    reject. That is stricter than SPEC §5.6's CI wording, and it is what §5.1(3) asks for. The
    test is exact under the fair-generator null, where family and twin groups are exchangeable.
    A z-screen with the exact permutation variance passes the clear cases; the rest get a Monte
    Carlo permutation test (`permutation_p`), so the pass/fail rule doesn't lean on a normal tail.
    If every score is tied the rule passes with p = 1.
    """
    if tests < 1:
        raise ValueError("tests must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    a = auc(pos, neg)
    alpha_adj = alpha / tests
    variance = null_variance(pos, neg)
    if variance == 0.0:
        return NullRule(auc=a, z=0.0, p=1.0, alpha_adj=alpha_adj, method="screen", passed=True)
    z = (a - 0.5) / math.sqrt(variance)
    normal = NormalDist()
    if abs(z) <= normal.inv_cdf(1 - alpha_adj / 2) - SCREEN_MARGIN:
        p_normal = 2 * normal.cdf(-abs(z))
        return NullRule(auc=a, z=z, p=p_normal, alpha_adj=alpha_adj, method="screen", passed=True)
    n_perm = max(PERM_FLOOR, math.ceil(PERM_FACTOR / alpha_adj))
    p_perm = permutation_p(pos, neg, n_perm=n_perm, seed=seed)
    return NullRule(
        auc=a, z=z, p=p_perm, alpha_adj=alpha_adj, method="permutation", passed=p_perm >= alpha_adj
    )


def bootstrap_auc(pos: ArrayLike, neg: ArrayLike, *, n_boot: int = 2000, seed: int) -> Floats:
    """AUCs of `n_boot` resamples drawn with replacement within each class (sizes fixed).

    Vectorised: a resample is a multinomial count vector over the units. Counts are collapsed
    onto the sorted unique score values, and the AUC is the count-weighted sum of
    (positives above + half the positives tied) over negatives. Deterministic for a seed.
    """
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    rng = np.random.default_rng(seed)
    grid = np.unique(np.concatenate([p, n]))

    def counts(values: Floats) -> Floats:
        """(n_boot, len(grid)) resample counts per grid value."""
        draws = rng.multinomial(values.size, np.full(values.size, 1 / values.size), size=n_boot)
        order = np.argsort(values, kind="stable")
        uniq, starts = np.unique(values[order], return_index=True)
        merged = np.add.reduceat(draws[:, order], starts, axis=1)
        out = np.zeros((n_boot, grid.size))
        out[:, np.searchsorted(grid, uniq)] = merged
        return out

    cp, cn = counts(p), counts(n)
    above = p.size - np.cumsum(cp, axis=1)  # positives strictly above each grid value
    return np.asarray((cn * (above + 0.5 * cp)).sum(axis=1) / (p.size * n.size))


def auc_interval(
    pos: ArrayLike, neg: ArrayLike, *, level: float = 0.95, n_boot: int = 2000, seed: int
) -> tuple[float, float, float]:
    """(AUC, lo, hi): the two-sided percentile bootstrap interval at `level`."""
    boot = bootstrap_auc(pos, neg, n_boot=n_boot, seed=seed)
    tail = (1 - level) / 2
    lo, hi = np.quantile(boot, [tail, 1 - tail])
    return auc(pos, neg), float(lo), float(hi)


def auc_upper_bound(
    pos: ArrayLike, neg: ArrayLike, *, level: float = 0.95, n_boot: int = 2000, seed: int
) -> float:
    """One-sided upper bound: the `level` quantile of the bootstrap AUCs (SPEC §5.4)."""
    return float(np.quantile(bootstrap_auc(pos, neg, n_boot=n_boot, seed=seed), level))


def recall_at_fpr(pos: ArrayLike, neg: ArrayLike, fpr: float) -> float:
    """The largest recall of a flag rule "score ≥ t" whose false-positive rate is at most `fpr`.

    `t` runs over every score and +inf (flag nothing), so tied scores are flagged together and
    never split. The allowed false positives are `floor(fpr · n_neg)`, with a 1e-9 slack so that
    `0.05 · 160` is 8 and not 7.999….
    """
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    if not 0 <= fpr <= 1:
        raise ValueError("fpr must be in [0, 1]")
    allowed = math.floor(fpr * n.size + 1e-9)
    thresholds = np.append(np.unique(np.concatenate([p, n])), np.inf)
    sp, sn = np.sort(p), np.sort(n)
    flagged_neg = n.size - np.searchsorted(sn, thresholds, "left")  # negatives with score ≥ t
    flagged_pos = p.size - np.searchsorted(sp, thresholds, "left")
    return float(flagged_pos[flagged_neg <= allowed].max() / p.size)


def stratified_weights(strata: ArrayLike, *, n_boot: int, seed: int) -> NDArray[np.int64]:
    """(n_boot, n) resample counts: a multinomial draw within each stratum, so every stratum's
    row sum equals its size. Strata are visited in sorted order, so a seed replays."""
    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("strata must be a non-empty 1-D array")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, labels.size), dtype=np.int64)
    for value in np.unique(labels):
        index = np.flatnonzero(labels == value)
        out[:, index] = rng.multinomial(index.size, np.full(index.size, 1 / index.size), n_boot)
    return out


def weighted_auc(pos: ArrayLike, neg: ArrayLike, w_pos: ArrayLike, w_neg: ArrayLike) -> Floats:
    """Mann-Whitney AUC (ties 1/2) of each resample, from integer row weights.

    `w_pos` is (n_boot, n_pos) and `w_neg` is (n_boot, n_neg). The result equals `auc` on the
    rows repeated by their weights; it is vectorised over the resample axis.
    """
    p, n = _scores(pos, "pos"), _scores(neg, "neg")
    wp, wn = np.asarray(w_pos, dtype=np.float64), np.asarray(w_neg, dtype=np.float64)
    if wp.ndim != 2 or wn.ndim != 2 or wp.shape[0] != wn.shape[0]:
        raise ValueError("weights must be (n_boot, n_pos) and (n_boot, n_neg)")
    if wp.shape[1] != p.size or wn.shape[1] != n.size:
        raise ValueError("weights do not match the scores")
    total = wp.sum(axis=1) * wn.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("a resample has no positive or no negative weight")
    diff = p[:, None] - n[None, :]
    wins = (diff > 0) + 0.5 * (diff == 0)  # (n_pos, n_neg)
    return np.asarray(((wp @ wins) * wn).sum(axis=1) / total)


def percentile_interval(values: ArrayLike, level: float = 0.95) -> tuple[float, float]:
    """The two-sided percentile interval of `values` at `level`."""
    arr = _scores(values, "values")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    tail = (1 - level) / 2
    lo, hi = np.quantile(arr, [tail, 1 - tail])
    return float(lo), float(hi)
