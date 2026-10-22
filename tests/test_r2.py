"""Known-answer tests for testbed/eval/r2.py (SPEC §7 R2): the probe's loading, fitting, tuning
and scoring, on synthetic units and embeddings. No store, no model, no real world is read."""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from testbed.eval import cv, r0, r2
from testbed.eval.metrics import auc
from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES as F64_NAMES
from testbed.models import local_llm, prompt
from testbed.world.families import PAIRS

REVISION = "rev-test"
CHILD = "c" * 64  # the "current" mlx_child.py hash
DIM = 6


@pytest.fixture(autouse=True)
def _revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_llm, "model_revision", lambda key: REVISION)
    monkeypatch.setattr(local_llm, "child_sha256", lambda: CHILD)


def make_units(world: str, n_pos: int = 10, n_neg: int = 10) -> list[r0.R0Unit]:
    """Units of one world, `n_pos` abusive and `n_neg` legitimate per pair, with unit keys."""
    rng = np.random.default_rng(zlib.crc32(world.encode()))
    units = []
    for pair in PAIRS:
        for k in range(n_pos + n_neg):
            positive = k < n_pos
            units.append(
                r0.R0Unit(
                    world,
                    f"{world}:p{pair}-{k:03d}",
                    PAIRS[pair][0 if positive else 1],
                    pair,
                    positive,
                    tuple(rng.normal(size=len(F_IDS))),
                    tuple(rng.normal(size=len(F64_NAMES))),
                    float(rng.random()),
                    None,
                    f"{world}-u{pair}{k:03d}",
                )
            )
    units.sort(key=lambda u: u.group)
    return units


def make_world(root: Path, name: str, checksum: str = "sum-1") -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"checksum_sha256": checksum, "s": 0.75}))
    write_renders(path, [f"render {i} of {name}" for i in range(3)])
    return path


def write_renders(world: Path, texts: Sequence[str]) -> None:
    lines = [json.dumps({"unit": f"{world.name}-r{i}", "text": t}) for i, t in enumerate(texts)]
    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")


def rows_sha256_of(world: Path) -> str:
    """The hash of the rows text `r2_embed` would write for `world`, built here by hand."""
    out = ""
    for line in (world / "renders.jsonl").read_text().splitlines():
        r = json.loads(line)
        row = {"id": r["unit"], "before": prompt.BEFORE, "render": r["text"], "after": prompt.AFTER}
        out += json.dumps(row) + "\n"
    return hashlib.sha256(out.encode()).hexdigest()


def planted(
    units: Sequence[r0.R0Unit], *, signal: bool, seed: int = 0, flip_side_a: bool = False
) -> NDArray[np.float32]:
    """(n, DIM) float32 rows; column 0 carries the label (+/- 4) when `signal`; the rest is noise.
    With `flip_side_a`, side A's pairs (4-6) carry the opposite sign."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(units), DIM))
    x[:, 0] *= 0.1  # the label column is clean, so a planted signal separates exactly
    if signal:
        sign = np.array([-1.0 if flip_side_a and u.pair in (4, 5, 6) else 1.0 for u in units])
        x[:, 0] += 4.0 * sign * np.array([u.positive for u in units])
    return x.astype(np.float32)


def write_embedding(
    embed: Path,
    model: str,
    world: Path,
    units: Sequence[r0.R0Unit],
    features: Mapping[str, NDArray[np.float32]],
    *,
    reverse: bool = False,
    extra_ids: Sequence[str] = (),
    prompt_sha: str = prompt.PROMPT_SHA256,
    checksum: str = "sum-1",
    revision: str = REVISION,
    rows_sha: str | None = None,
    child_sha: str = CHILD,
    npz_sha: str | None = None,
    drop: Sequence[str] = (),
) -> None:
    """Write `<embed>/<model>/<world>.npz` + `.json` as `r2_embed` does (rows in render order,
    here the reverse of `units` if `reverse`, plus ids of units that aren't scored). `rows_sha`
    and `npz_sha` default to the true hashes; `drop` leaves fields out of the json."""
    ids = [u.unit for u in units]
    order = list(range(len(units)))[::-1] if reverse else list(range(len(units)))
    arrays: dict[str, Any] = {}
    for name in ("mean_16", "last_16", "mean_final", "last_final"):
        base = features.get(name, np.zeros((len(units), DIM), dtype=np.float32))
        extra = np.full((len(extra_ids), DIM), 99.0, dtype=np.float32)
        arrays[name] = np.concatenate([base[order], extra]) if extra_ids else base[order]
    arrays["ids"] = np.array([ids[i] for i in order] + list(extra_ids))
    dest = embed / model
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / f"{world.name}.npz", **arrays)
    meta = {
        "prompt_sha256": prompt_sha,
        "model_revision": revision,
        "world_checksum_sha256": checksum,
        "layers": ["16", "final"],
        "rows_sha256": rows_sha or rows_sha256_of(world),
        "child_sha256": child_sha,
        "npz_sha256": npz_sha
        or hashlib.sha256((dest / f"{world.name}.npz").read_bytes()).hexdigest(),
    }
    for field in drop:
        del meta[field]
    (dest / f"{world.name}.json").write_text(json.dumps(meta))


# --- Loading --------------------------------------------------------------------------------------


def test_load_features_aligns_by_unit_id_not_by_row_order(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    x = planted(units, signal=True)
    write_embedding(tmp_path / "e", "9b", world, units, {"mean_final": x}, reverse=True)
    got = r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, x.astype(np.float64))
    # the same file read for a shuffled, partial list of units follows that list
    pick = [units[5], units[1], units[100]]
    part = r2.load_features(tmp_path / "e", "9b", world, pick, "mean_final")
    np.testing.assert_array_equal(part, x[[5, 1, 100]].astype(np.float64))


def test_load_features_reads_a_world_whose_name_holds_a_dot(tmp_path: Path) -> None:
    # real world names carry s (m1-dev-s0.75-seed1); `with_suffix` would read m1-dev-s0.json
    world = make_world(tmp_path, "m1-dev-s0.75-seed1")
    units = make_units("m1-dev-s0.75-seed1")
    x = planted(units, signal=True)
    write_embedding(tmp_path / "e", "9b", world, units, {"mean_final": x})
    got = r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")
    np.testing.assert_array_equal(got, x.astype(np.float64))


def test_load_features_ignores_embedded_units_that_are_not_scored(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_embedding(
        tmp_path / "e", "9b", world, units, {"mean_final": planted(units, signal=False)},
        extra_ids=["mixed-cluster"],
    )  # fmt: skip
    got = r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")
    assert got.shape == (len(units), DIM) and (got != 99.0).all()


def test_pooled_features_reads_each_unit_from_its_own_world(tmp_path: Path) -> None:
    w1, w2 = make_world(tmp_path, "w1"), make_world(tmp_path, "w2")
    u1, u2 = make_units("w1"), make_units("w2")
    x1, x2 = planted(u1, signal=False, seed=1), planted(u2, signal=False, seed=2)
    write_embedding(tmp_path / "e", "9b", w1, u1, {"mean_final": x1})
    write_embedding(tmp_path / "e", "9b", w2, u2, {"mean_final": x2}, reverse=True)
    pooled = sorted(u1 + u2, key=lambda u: u.group, reverse=True)  # interleaved by group order
    pooled = pooled[::3] + pooled[1::3] + pooled[2::3]  # and shuffled across worlds
    got = r2.pooled_features(tmp_path / "e", "9b", [w1, w2], pooled, "mean_final")
    by_unit = {u.unit: row for u, row in zip(u1 + u2, np.vstack([x1, x2]), strict=True)}
    expected = np.stack([by_unit[u.unit] for u in pooled]).astype(np.float64)
    np.testing.assert_array_equal(got, expected)
    with pytest.raises(ValueError, match="worlds not given"):
        r2.pooled_features(tmp_path / "e", "9b", [w1], pooled, "mean_final")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"prompt_sha": "0" * 64}, "different prompt"),
        ({"checksum": "other"}, "different world"),
        ({"revision": "rev-old"}, "different 9b revision"),
        ({"rows_sha": "0" * 64}, "rows_sha256 differs"),
        ({"child_sha": "0" * 64}, "different mlx_child.py"),
        ({"npz_sha": "0" * 64}, "npz_sha256 differs"),
        ({"drop": ["rows_sha256"]}, "no rows_sha256"),
        ({"drop": ["child_sha256"]}, "no child_sha256"),
        ({"drop": ["npz_sha256"]}, "no npz_sha256"),
    ],
)
def test_load_features_refuses_each_provenance_mismatch(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_embedding(
        tmp_path / "e", "9b", world, units, {"mean_final": planted(units, signal=False)}, **override
    )
    with pytest.raises(ValueError, match=message):
        r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")


def test_load_features_refuses_rows_changed_after_embedding(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_embedding(
        tmp_path / "e", "9b", world, units, {"mean_final": planted(units, signal=False)}
    )
    r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")  # fine as embedded
    write_renders(world, ["render 0 of w1", "edited", "render 2 of w1"])  # same ids, other text
    with pytest.raises(ValueError, match="w1: embedding rows_sha256 differs"):
        r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")


def test_load_features_refuses_a_replaced_npz(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_embedding(
        tmp_path / "e", "9b", world, units, {"mean_final": planted(units, signal=False)}
    )
    npz = tmp_path / "e" / "9b" / "w1.npz"
    with np.load(npz) as data:
        arrays: dict[str, Any] = {k: np.array(data[k]) for k in data.files}
    arrays["mean_final"] = arrays["mean_final"] + 1.0  # same ids and shapes, other values
    np.savez(npz, **arrays)
    with pytest.raises(ValueError, match="w1: embedding npz_sha256 differs"):
        r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")


def test_load_features_refuses_missing_duplicate_and_non_finite_rows(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    x = planted(units, signal=False)
    embed = tmp_path / "e"

    write_embedding(embed, "9b", world, units[:-1], {"mean_final": x[:-1]})
    with pytest.raises(ValueError, match="1 units have no row"):
        r2.load_features(embed, "9b", world, units, "mean_final")

    write_embedding(embed, "9b", world, units, {"mean_final": x}, extra_ids=[units[3].unit])
    with pytest.raises(ValueError, match="duplicate ids"):
        r2.load_features(embed, "9b", world, units, "mean_final")

    bad = x.copy()
    bad[7, 2] = np.nan
    write_embedding(embed, "9b", world, units, {"mean_final": bad})
    with pytest.raises(ValueError, match="non-finite"):
        r2.load_features(embed, "9b", world, units, "mean_final")
    assert np.isfinite(r2.load_features(embed, "9b", world, units, "mean_16")).all()  # others fine

    with pytest.raises(ValueError, match="no array 'mean_9'"):
        r2.load_features(embed, "9b", world, units, "mean_9")


def test_load_features_refuses_units_of_another_world_or_without_a_key(tmp_path: Path) -> None:
    world = make_world(tmp_path, "w1")
    units = make_units("w1")
    write_embedding(tmp_path / "e", "9b", world, units, {})
    other = make_units("w2")[:2]
    with pytest.raises(ValueError, match="another world"):
        r2.load_features(tmp_path / "e", "9b", world, units[:2] + other, "mean_final")
    keyless = [replace(u, unit="") for u in units[:2]]
    with pytest.raises(ValueError, match="without a unit key"):
        r2.load_features(tmp_path / "e", "9b", world, keyless, "mean_final")


def test_a_row_count_that_disagrees_with_the_ids_is_refused() -> None:
    with pytest.raises(ValueError, match="3 ids for 2 rows"):
        r2.aligned_rows(["a", "b", "c"], np.zeros((2, 4)), ["a"], "x")


# --- Fitting --------------------------------------------------------------------------------------


def test_fit_scores_are_log_odds_that_follow_the_data() -> None:
    x_train = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1])
    hits: list[bool] = []
    got = r2.fit_scores(x_train, y_train, np.array([[-3.0], [0.0], [3.0]]), 1.0, hits)
    assert got.dtype == np.float64
    assert got[0] < got[1] < got[2]
    assert got[1] == pytest.approx(0.0, abs=1e-9)  # symmetric data: no intercept
    assert got[2] == pytest.approx(-got[0])
    assert hits == [False]
    # a smaller C shrinks the weights, and so the log-odds, towards 0
    weak = r2.fit_scores(x_train, y_train, np.array([[3.0]]), 1e-4)
    assert 0 < weak[0] < got[2]


def test_fit_scores_records_a_run_that_stops_at_max_iter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2, "MAX_ITER", 1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4))
    y = (x[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(np.int64)
    hits: list[bool] = []
    scores = r2.fit_scores(x, y, x, 10.0, hits)  # does not raise
    assert hits == [True] and np.isfinite(scores).all()


def test_fit_scores_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        r2.fit_scores(np.zeros((3, 2)), np.array([1, 1, 1]), np.zeros((1, 2)), 1.0)


def test_held_out_fits_each_side_only_on_the_other_sides_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units = make_units("w1", 4, 4)
    y = np.array([int(u.positive) for u in units])
    pairs = np.array([u.pair for u in units])
    x = np.column_stack([pairs, np.arange(len(units))]).astype(np.float64)
    calls: list[tuple[set[int], set[int]]] = []

    def fake(
        x_train: Any, y_train: Any, x_test: Any, C: float, hits: Any = None
    ) -> NDArray[np.float64]:
        train, test = set(x_train[:, 0].astype(int)), set(x_test[:, 0].astype(int))
        calls.append((train, test))
        assert C == 0.5
        return np.full(len(x_test), 0.25 if test <= {4, 5, 6} else 0.75)

    monkeypatch.setattr(r2, "fit_scores", fake)
    out = r2.held_out_scores(x, y, pairs, 0.5)
    assert sorted(calls, key=lambda c: min(c[1])) == [
        ({4, 5, 6}, {1, 2, 3}),
        ({1, 2, 3}, {4, 5, 6}),
    ]
    assert out.tolist() == [0.25 if u.pair in (4, 5, 6) else 0.75 for u in units]


def test_a_signal_that_flips_between_the_sides_is_inverted_by_held_out_only() -> None:
    """Column 0 means 'abusive' in pairs 1-3 and 'legitimate' in pairs 4-6. A held-out model
    learns one side's meaning and meets the other's, so every held-out family AUC is 0. Training
    on a unit's own side (leakage) would give 1."""
    units = make_units("w1")
    x = planted(units, signal=True, flip_side_a=True).astype(np.float64)
    y = np.array([int(u.positive) for u in units])
    pairs = np.array([u.pair for u in units])
    positive = y == 1
    held = r2.held_out_scores(x, y, pairs, 1.0)
    for pair in PAIRS:
        m = pairs == pair
        assert auc(held[m & positive], held[m & ~positive]) == 0.0, pair
    assert r2.primary_auc(units, held) == 0.0


def test_in_family_uses_r0s_strata_and_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    units = make_units("w1")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cv, "oof_scores", lambda x, y, **kw: seen.update(kw) or np.zeros(len(y)))
    r0.score_pool(units)
    assert r2.strata_of(units) == seen["strata"]  # the strata r0's in-family regime uses

    x = planted(units, signal=True).astype(np.float64)
    y = np.array([int(u.positive) for u in units])
    groups = [u.group for u in units]
    got = r2.in_family_scores(x, y, groups, r2.strata_of(units), 1e-2)
    expected = np.full(len(units), np.nan)
    for train, test in cv.group_folds(
        seen["strata"], seen["groups"], folds=seen["folds"], seed=seen["seed"]
    ):
        expected[test] = r2.fit_scores(x[train], y[train], x[test], 1e-2)
    np.testing.assert_array_equal(got, expected)
    assert (seen["folds"], seen["seed"]) == (5, r0.CV_SEED)


def test_a_separable_feature_gives_auc_one_and_noise_gives_one_half() -> None:
    units = make_units("w1", 30, 30)
    y = np.array([int(u.positive) for u in units])
    pairs = np.array([u.pair for u in units])
    groups = [u.group for u in units]
    strata = r2.strata_of(units)
    signal = planted(units, signal=True).astype(np.float64)
    noise = planted(units, signal=False, seed=3).astype(np.float64)
    for scores in (
        r2.held_out_scores(signal, y, pairs, 1e-2),
        r2.in_family_scores(signal, y, groups, strata, 1e-2),
    ):
        assert r2.primary_auc(units, scores) == 1.0
    for scores in (
        r2.held_out_scores(noise, y, pairs, 1e-2),
        r2.in_family_scores(noise, y, groups, strata, 1e-2),
    ):
        assert r2.primary_auc(units, scores) == pytest.approx(0.5, abs=0.1)


# --- Selection and tuning -------------------------------------------------------------------------


def _row(feature: str, C: float, held: float) -> dict[str, Any]:
    return {"feature": feature, "C": C, "held_out_primary": held}


def test_select_takes_the_highest_held_out_primary() -> None:
    grid = [
        _row("mean_final", 1e-2, 0.6),
        _row("mean_16", 1e-1, 0.7),
        _row("last_final", 1.0, 0.65),
    ]
    assert r2.select(grid) == {"feature": "mean_16", "C": 1e-1}


def test_select_breaks_ties_by_smaller_c_then_candidate_order() -> None:
    tied = [
        _row("last_final", 1e-3, 0.8), _row("mean_16", 1e-2, 0.8), _row("mean_final", 1e-2, 0.8),
        _row("mean_16", 1e-3, 0.8), _row("mean_final", 1e-3, 0.8), _row("mean_final", 1.0, 0.7),
    ]  # fmt: skip
    # smallest C among the best is 1e-3; of those, mean_final comes first in CANDIDATES
    assert r2.select(tied) == {"feature": "mean_final", "C": 1e-3}
    assert r2.CANDIDATES == ("mean_final", "mean_16", "last_final")
    assert r2.select(tied[:2]) == {"feature": "last_final", "C": 1e-3}
    # a tie is up to float noise below the 12th decimal, and a higher value still wins
    noisy = [_row("mean_final", 1.0, 0.8), _row("mean_16", 1e-3, 0.8 + 1e-15)]
    assert r2.select(noisy) == {"feature": "mean_16", "C": 1e-3}
    assert r2.select([*noisy, _row("last_final", 10.0, 0.8001)]) == {
        "feature": "last_final",
        "C": 10.0,
    }
    with pytest.raises(ValueError):
        r2.select([])


@pytest.fixture
def tuning_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    world = make_world(tmp_path, "tune")
    units = make_units("tune")
    monkeypatch.setattr(r0, "load_world", lambda w, scales=("standard",): (0.75, units, 0))
    write_embedding(
        tmp_path / "e",
        "9b",
        world,
        units,
        {
            "mean_final": planted(units, signal=True),  # separable
            "mean_16": planted(units, signal=False, seed=5),
            "last_final": planted(units, signal=False, seed=6),
        },
    )
    return world, tmp_path / "e"


def test_tune_finds_the_planted_feature_and_breaks_the_all_way_tie_by_smallest_c(
    tuning_setup: tuple[Path, Path],
) -> None:
    world, embed = tuning_setup
    result = r2.tune(world, embed, "9b")
    grid = {(row["feature"], row["C"]): row for row in result["grid"]}
    assert len(grid) == len(r2.CANDIDATES) * len(r2.C_GRID) == len(result["grid"])
    perfect = [c for c in r2.C_GRID if grid[("mean_final", c)]["held_out_primary"] == 1.0]
    assert len(perfect) > 1  # the scorer ties across C, so the tie-break decides
    assert result["selected"] == {"feature": "mean_final", "C": min(perfect)}
    assert min(perfect) == r2.C_GRID[0]  # separable at every C, so the smallest wins
    row = grid[("mean_final", 1e-2)]
    assert row["in_family_primary"] == 1.0
    assert set(row["held_out_family_auc"]) == {str(p) for p in PAIRS}
    assert all(v == 1.0 for v in row["held_out_family_auc"].values())
    assert row["fits"] == 7 and row["max_iter_hits"] == 0  # 2 held-out + 5 in-family folds
    noise = grid[("mean_16", 1e-2)]["held_out_primary"]
    assert abs(noise - 0.5) < 0.2
    assert result["n_units"] == 120 and result["s"] == 0.75 and result["model"] == "9b"
    assert result["embedding"]["prompt_sha256"] == prompt.PROMPT_SHA256
    assert result["embedding"]["model_revision"] == REVISION
    assert result["embedding"]["child_sha256"] == CHILD
    assert result["embedding"]["rows_sha256"] == rows_sha256_of(world)
    assert (
        result["embedding"]["npz_sha256"]
        == hashlib.sha256((embed / "9b" / f"{world.name}.npz").read_bytes()).hexdigest()
    )
    assert "Selected: `mean_final` at C = 1e-05." in r2.tune_markdown(result)
    json.dumps(result, allow_nan=False)


def test_tune_refuses_an_unknown_model(tuning_setup: tuple[Path, Path]) -> None:
    world, embed = tuning_setup
    with pytest.raises(ValueError, match="unknown model"):
        r2.tune(world, embed, "70b")


# --- Scoring --------------------------------------------------------------------------------------


@pytest.fixture
def two_worlds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], Path]:
    worlds = [make_world(tmp_path, "w1"), make_world(tmp_path, "w2")]
    by_name = {w.name: make_units(w.name) for w in worlds}
    monkeypatch.setattr(
        r0, "load_world", lambda w, scales=("standard",): (0.75, list(by_name[Path(w).name]), 0)
    )
    for w in worlds:
        units = by_name[w.name]
        write_embedding(
            tmp_path / "e",
            "9b",
            w,
            units,
            {"mean_final": planted(units, signal=True, seed=len(w.name)),
             "mean_16": planted(units, signal=False, seed=9)},
        )  # fmt: skip
    return worlds, tmp_path / "e"


def test_score_rungs_adds_the_probe_rung_to_r0s_in_both_regimes(
    two_worlds: tuple[list[Path], Path],
) -> None:
    worlds, embed = two_worlds
    units, scores = r2.score_rungs(worlds, embed, {"9b": ("mean_final", 1e-2)})
    assert [u.group for u in units] == sorted(u.group for u in units)
    assert len(units) == 240 and {u.world for u in units} == {"w1", "w2"}
    expected = {(r, g) for r in (*r0.RUNGS, "r2_9b") for g in r0.REGIMES}
    assert set(scores) == expected
    assert all(v.shape == (240,) and np.isfinite(v).all() for v in scores.values())
    for regime in r0.REGIMES:
        assert r2.primary_auc(units, scores[("r2_9b", regime)]) == 1.0
    baseline = r0.score_pool(units)  # the R0 rungs are unchanged by adding a probe
    for key, value in baseline.items():
        np.testing.assert_array_equal(scores[key], value)


def test_score_rungs_pools_worlds_that_share_one_s(
    two_worlds: tuple[list[Path], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    worlds, embed = two_worlds
    real = r0.load_world
    monkeypatch.setattr(
        r0, "load_world", lambda w, scales=("standard",): (0.5, real(w, scales=scales)[1], 0)
        if Path(w).name == "w2" else real(w, scales=scales),
    )  # fmt: skip
    with pytest.raises(ValueError, match="share one s"):
        r2.score_rungs(worlds, embed, {"9b": ("mean_final", 1e-2)})
    with pytest.raises(ValueError, match="unique"):
        r2.score_rungs([worlds[0], worlds[0]], embed, {})


def test_parse_choice() -> None:
    assert r2.parse_choice("9b:mean_final:1e-3") == ("9b", "mean_final", 1e-3)
    assert r2.parse_choice("4b:last_final:10") == ("4b", "last_final", 10.0)
    for bad in ("9b:mean_final", "70b:mean_final:1", "9b:last_16:1", "9b:mean_final:0.5"):
        with pytest.raises(ValueError):
            r2.parse_choice(bad)


def test_the_score_cli_writes_the_report_and_refuses_a_non_empty_out(
    two_worlds: tuple[list[Path], Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worlds, embed = two_worlds
    out = tmp_path / "out"
    argv = ["score", "--worlds", *map(str, worlds), "--embed", str(embed), "--out", str(out)]
    assert r2.main([*argv, "--choice", "9b:mean_final:1e-2"]) == 0
    assert "r2_9b" in capsys.readouterr().out
    report = json.loads((out / "r2.json").read_text())
    assert report["s"] == 0.75 and report["worlds"] == ["w1", "w2"]
    assert report["choices"] == {"9b": {"feature": "mean_final", "C": 1e-2}}
    assert report["n_units"] == 240 and report["max_iter_hits"] == {"r2_9b": 0}
    for regime in r0.REGIMES:
        assert report["regimes"][regime]["r2_9b"]["primary"]["auc"] == 1.0
        diffs = report["differences"][regime]
        assert {"r2_9b_minus_f64_logistic", "r2_9b_minus_f64_trees"} <= set(diffs)
        assert {"f64_trees_minus_f64_logistic", "stronger_f64_minus_detector"} <= set(diffs)
        gap = diffs["r2_9b_minus_f64_logistic"]
        stable = report["regimes"][regime]["f64_logistic"]["primary"]["auc"]
        assert gap["point"] == pytest.approx(1.0 - stable) and gap["lo"] <= gap["hi"]
        assert (gap["n_pos"], gap["n_neg"]) == (80, 80)  # four T pairs x 10 x two worlds
    assert set(report["embedding"]) == {"9b/w1", "9b/w2"}
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 240 and all("r2_9b" in r["scores"] for r in rows)
    assert all(set(r["scores"]["r2_9b"]) == {"in_family", "held_out"} for r in rows)
    before = (out / "r2.json").read_text()
    with pytest.raises(FileExistsError):
        r2.main([*argv, "--choice", "9b:mean_final:1e-2"])
    assert (out / "r2.json").read_text() == before
    with pytest.raises(ValueError, match="two --choice"):
        r2.main([*argv[:-1], str(tmp_path / "o2"), "--choice", "9b:mean_final:1e-2",
                 "--choice", "9b:mean_16:1e-2"])  # fmt: skip


def test_the_tune_cli_writes_json_and_markdown_and_refuses_a_non_empty_out(
    tuning_setup: tuple[Path, Path], tmp_path: Path
) -> None:
    world, embed = tuning_setup
    out = tmp_path / "tune-out"
    argv = ["tune", "--world", str(world), "--embed", str(embed), "--model", "9b"]
    assert r2.main([*argv, "--out", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"tune.json", "tune.md"}
    assert json.loads((out / "tune.json").read_text())["selected"]["feature"] == "mean_final"
    with pytest.raises(FileExistsError):
        r2.main([*argv, "--out", str(out)])
