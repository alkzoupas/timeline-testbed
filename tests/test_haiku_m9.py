"""Known-answer tests for M9's selector and scorer (testbed/eval/haiku_m9.py).

Units are synthetic `R0Unit` lists: `select` runs with `r0.load_world` replaced, and `score` with
`r0.load_world` and `haiku_m9.f64_held_out` replaced and a hand-written driver output. `claude` is
never run and no world is read."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from testbed.eval import g2, haiku, r0
from testbed.eval import haiku_m9 as m9
from testbed.eval import metrics as mx
from testbed.eval.r0 import R0Unit
from testbed.models import haiku_prompt as hp
from testbed.models import haiku_run as hr
from testbed.world.families import PAIRS

CLI = "9.9.9 (fake)"
W1, W2, CTRL = "m9-s0.75-seed11", "m9-s0.75-seed12", "m9-s0-seed13"
PREREG_TEXT = "prereg v1\n"


def make_units(
    world: str, pairs: Sequence[int] = tuple(PAIRS), n_pos: int = 4, n_neg: int = 6
) -> list[R0Unit]:
    units = []
    for pair in pairs:
        for positive in (True, False):
            for k in range(n_pos if positive else n_neg):
                tag = f"p{pair}{'a' if positive else 'l'}{k:02d}"
                units.append(
                    R0Unit(
                        world,
                        f"{world}:{tag}",
                        PAIRS[pair][0 if positive else 1],
                        pair,
                        positive,
                        (),
                        (),
                        0.0,
                        None,
                        f"snap-{world}-{tag}",  # unique across worlds, as real unit ids are
                    )
                )
    return units


@pytest.fixture(autouse=True)
def prereg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "prereg" / "M9.md"
    path.parent.mkdir()
    path.write_text(PREREG_TEXT)
    monkeypatch.setattr(m9, "PREREG_PATH", path)
    return path


# --- Selection ------------------------------------------------------------------------------------


def test_examples_are_sides_b_and_deterministic() -> None:
    tune = make_units("t")
    first = haiku.pick_examples(tune)
    assert haiku.pick_examples(tune) == first
    shuffled = [tune[i] for i in np.random.default_rng(1).permutation(len(tune))]
    assert haiku.pick_examples(shuffled) == first
    assert m9.SIDE == "B" and r0.side_of(m9.PAIR) == "B"
    assert [e.pair for e in first["B"]] == [4, 5, 6, 4, 5, 6]  # pair 3 is never an example
    assert [e.positive for e in first["B"]] == [True, False, True, False, True, False]


def test_check_examples_match_m8_select_json(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    examples = [{"unit": "snap-a", "pair": 4, "class": "abusive"}]
    meta = {"examples": {"B": examples}}
    m9.check_examples_match_m8(runs, meta)  # no M8 file: nothing to compare
    (runs / "m8-pilot").mkdir(parents=True)
    path = runs / m9.M8_SELECT
    path.write_text(json.dumps({"examples": {"A": [], "B": examples}}))
    m9.check_examples_match_m8(runs, meta)
    path.write_text(json.dumps({"examples": {"A": [], "B": [{**examples[0], "unit": "snap-b"}]}}))
    with pytest.raises(m9.SelectRefusal, match="differ"):
        m9.check_examples_match_m8(runs, meta)


def test_pair3_units_are_every_abusive_and_forty_legitimate() -> None:
    units = make_units("w", n_pos=5, n_neg=50)
    chosen = m9.pair3_units(units, "w")
    assert {u.pair for u in chosen} == {3}
    assert sum(u.positive for u in chosen) == 5
    assert sum(not u.positive for u in chosen) == m9.N_LEGIT == 40
    assert [u.unit for u in chosen] == sorted(u.unit for u in chosen)
    assert chosen == m9.pair3_units(units[::-1], "w") == m9.pair3_units(units, "w")
    assert m9.SUBSET_SEED == 20261091


def test_pair3_legitimate_draw_is_per_world_from_the_name() -> None:
    a = [u.unit.split("-", 2)[2] for u in m9.pair3_units(make_units("w1", n_neg=50), "w1")]
    b = [u.unit.split("-", 2)[2] for u in m9.pair3_units(make_units("w2", n_neg=50), "w2")]
    assert a != b  # the same tags, drawn by different streams
    rng = m9.world_rng(m9.SUBSET_SEED, "w1")
    index = sorted(int(i) for i in rng.choice(50, 40, replace=False))
    legit = sorted(u.unit for u in make_units("w1", n_neg=50) if u.pair == 3 and not u.positive)
    chosen_legit = [
        u.unit for u in m9.pair3_units(make_units("w1", n_neg=50), "w1") if not u.positive
    ]
    assert chosen_legit == [legit[i] for i in index]


def test_pair3_units_refuse_too_few_legitimate_and_no_abusive() -> None:
    with pytest.raises(m9.SelectRefusal, match="need 40"):
        m9.pair3_units(make_units("w", n_neg=39), "w")
    with pytest.raises(m9.SelectRefusal, match="abusive"):
        m9.pair3_units(make_units("w", n_pos=0, n_neg=50), "w")


def test_check_units_two_and_two_never_an_example() -> None:
    tune = make_units("t")
    banned = {e.unit for e in haiku.pick_examples(tune)["B"]}
    chosen = m9.check_units(tune, banned)
    assert len(chosen) == 4
    assert sum(u.positive for u in chosen) == 2
    assert {u.pair for u in chosen} == {3}
    assert not {u.unit for u in chosen} & banned
    assert m9.check_units(tune[::-1], banned) == chosen
    assert m9.CHECK_SEED == 20261094
    with pytest.raises(m9.SelectRefusal):
        m9.check_units(make_units("t", n_pos=1), banned)


def standard_worlds(n_pos: int = 6, n_neg: int = 50) -> dict[str, tuple[float, list[R0Unit]]]:
    return {
        W1: (0.75, make_units(W1, n_pos=n_pos, n_neg=n_neg)),
        W2: (0.75, make_units(W2, n_pos=n_pos, n_neg=n_neg)),
        CTRL: (0.0, make_units(CTRL, n_pos=n_pos, n_neg=n_neg)),
    }


def build(
    mode: str = "confirm", n_pos: int = 6, n_neg: int = 50
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, str]]]:
    tune = make_units(haiku.TUNE_WORLD)
    if mode == "check":
        worlds = {haiku.TUNE_WORLD: (0.75, tune)}
        control = None
    else:
        worlds = standard_worlds(n_pos, n_neg)
        control = CTRL
    renders = {
        w: {u.unit: f"render of {w}/{u.unit}" for u in units}
        for w, units in {**{k: v[1] for k, v in worlds.items()}, haiku.TUNE_WORLD: tune}.items()
    }
    rows, meta = m9.build_selection(mode, tune, worlds, control, renders)
    return rows, meta, renders


def test_job_rows_have_exactly_the_driver_fields() -> None:
    rows, _, _ = build()
    for row in rows:
        assert set(row) == set(hr.JOB_FIELDS)
        assert row["prompt_sha256"] == hp.PROMPT_SHA256
        assert row["user_sha256"] == hr.sha256_text(row["user"])
        assert row["key"] == f"{row['unit']}.{row['variant']}.d{row['draw']}"
        assert hr.KEY_PATTERN.fullmatch(row["key"])


def test_user_messages_carry_side_b_shots_in_presentation_order() -> None:
    rows, meta, renders = build()
    tune = renders[haiku.TUNE_WORLD]
    shots = [(tune[e["unit"]], e["class"] == "abusive") for e in meta["examples"]["B"]]
    assert [e["pair"] for e in meta["examples"]["B"]] == [4, 5, 6, 4, 5, 6]
    assert set(meta["examples"]) == {"B"}
    for row in rows:
        render = renders[row["world"]][row["unit"]]
        if row["variant"] == "zero_shot":
            assert row["user"] == hp.zero_shot_user(render)
        else:
            assert row["user"] == hp.few_shot_user(shots, render)


def test_variants_and_counts_per_world() -> None:
    rows, meta, _ = build()
    assert meta["variants"] == {
        W1: ["zero_shot", "few_shot"],
        W2: ["zero_shot", "few_shot"],
        CTRL: ["few_shot"],
    }
    assert meta["scored_worlds"] == [W1, W2] and meta["control"] == CTRL
    assert meta["worlds"] == {W1: 0.75, W2: 0.75, CTRL: 0.0}
    for world in (W1, W2, CTRL):
        assert meta["units"][world]["counts"] == {"abusive": 6, "legitimate": 40}
        assert len(meta["units"][world]["ids"]) == 46
    assert meta["n_calls"] == len(rows) == 46 * 2 * 3 * 2 + 46 * 3
    assert {(r["world"], r["variant"]) for r in rows} == {
        (W1, "zero_shot"),
        (W1, "few_shot"),
        (W2, "zero_shot"),
        (W2, "few_shot"),
        (CTRL, "few_shot"),
    }


def test_standard_scale_is_1200_calls() -> None:
    rows, meta, _ = build(n_pos=40, n_neg=60)
    assert len(rows) == meta["n_calls"] == 160 * 2 * 3 + 80 * 3 == 1200


def test_order_is_one_seeded_permutation_of_the_sorted_keys() -> None:
    rows, _, _ = build()
    keys = [r["key"] for r in rows]
    assert m9.ORDER_SEED == 20261092
    assert len(set(keys)) == len(keys)
    expect = sorted(keys)
    order = np.random.default_rng(m9.ORDER_SEED).permutation(len(expect))
    assert keys == [expect[int(i)] for i in order]
    assert len({r["world"] for r in rows[:30]}) == 3  # worlds are interleaved
    again, _, _ = build()
    assert [r["key"] for r in again] == keys


def test_check_mode_is_24_calls_on_the_tuning_world() -> None:
    rows, meta, _ = build("check")
    assert meta["n_calls"] == len(rows) == 4 * 2 * 3 == 24
    assert meta["scored_worlds"] == [haiku.TUNE_WORLD] and meta["control"] is None
    assert meta["units"][haiku.TUNE_WORLD]["counts"] == {"abusive": 2, "legitimate": 2}
    assert {r["world"] for r in rows} == {haiku.TUNE_WORLD}
    assert meta["variants"] == {haiku.TUNE_WORLD: ["zero_shot", "few_shot"]}


def fake_runs(
    monkeypatch: pytest.MonkeyPatch,
    runs: Path,
    worlds: Mapping[str, tuple[float, list[R0Unit]]] | None = None,
    health: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, list[R0Unit]]]:
    worlds = dict(worlds or standard_worlds())
    worlds.setdefault(haiku.TUNE_WORLD, (0.75, make_units(haiku.TUNE_WORLD)))
    for name, (_, units) in worlds.items():
        (runs / name).mkdir(parents=True)
        lines = [json.dumps({"unit": u.unit, "text": f"render of {name}/{u.unit}"}) for u in units]
        (runs / name / "renders.jsonl").write_text("\n".join(lines) + "\n")
        record = (health or {}).get(name, {"passed": True})
        if record is not None:
            (runs / name / "health.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        r0,
        "load_world",
        lambda path, **_: (worlds[Path(path).name][0], worlds[Path(path).name][1], 0),
    )
    return worlds


def confirm(
    runs: Path, out: Path, worlds: Sequence[str] = (W1, W2), control: str | None = CTRL
) -> dict[str, Any]:
    return m9.select("confirm", runs, out, worlds, control)


def test_select_writes_the_job_and_select_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prereg: Path
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "job"
    fake_runs(monkeypatch, runs)
    meta = confirm(runs, out)
    assert hr.sha256_text((out / "jobs.jsonl").read_text()) == meta["jobs_sha256"]
    assert json.loads((out / "select.json").read_text()) == meta
    rows, digest = hr.read_jobs(out / "jobs.jsonl")  # the driver accepts what select wrote
    assert len(rows) == meta["n_calls"] and digest == meta["jobs_sha256"]
    assert meta["prereg_sha256"] == m9.prereg_sha256()
    assert meta["prompt_sha256"] == hp.PROMPT_SHA256 and meta["tune_world"] == haiku.TUNE_WORLD
    assert meta["k"] == 3 and meta["seeds"]["subset"] == 20261091
    assert meta["seeds"]["order"] == 20261092 and meta["seeds"]["boot"] == 20261093
    prereg.write_text("amended\n")
    assert m9.prereg_sha256() != meta["prereg_sha256"]


def test_select_does_not_depend_on_the_order_of_worlds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    fake_runs(monkeypatch, runs)
    first = confirm(runs, tmp_path / "a", (W1, W2))
    second = confirm(runs, tmp_path / "b", (W2, W1))
    assert first == second
    assert (tmp_path / "a" / "jobs.jsonl").read_bytes() == (
        tmp_path / "b" / "jobs.jsonl"
    ).read_bytes()


def test_select_check_mode_runs_on_the_tuning_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    fake_runs(monkeypatch, runs)
    meta = m9.select("check", runs, tmp_path / "job")
    assert meta["mode"] == "check" and meta["n_calls"] == 24
    with pytest.raises(m9.SelectRefusal, match="no --worlds"):
        m9.select("check", runs, tmp_path / "job2", [W1], None)


def test_select_refuses_a_non_empty_out_and_a_changed_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "job"
    fake_runs(monkeypatch, runs)
    confirm(runs, out)
    with pytest.raises(m9.SelectRefusal, match="not empty"):
        confirm(runs, out)
    empty = tmp_path / "empty"
    empty.mkdir()
    confirm(runs, empty)  # an empty directory is fine
    monkeypatch.setattr(hp, "PROMPT_SHA256", "0" * 64)
    with pytest.raises(m9.SelectRefusal, match="PROMPT_SHA256"):
        confirm(runs, tmp_path / "other")


def test_select_refuses_confirm_gate_tune_and_pilot_worlds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    worlds = standard_worlds()
    for name in ("confirm-s0.75-seed1", *g2.GATE_WORLDS[0.75], *g2.GATE_WORLDS[0.0]):
        worlds[name] = (0.75 if "0.75" in name else 0.0, make_units(name, n_neg=50))
    worlds[haiku.PILOT_WORLD] = worlds["m1-dev-s0.75-seed1"]
    fake_runs(monkeypatch, runs, worlds)
    for bad in ("confirm-s0.75-seed1", haiku.TUNE_WORLD, haiku.PILOT_WORLD, *g2.GATE_WORLDS[0.75]):
        with pytest.raises(m9.SelectRefusal, match="confirm|tuning|gate"):
            confirm(runs, tmp_path / f"job-{bad}", (W1, bad))
    for bad in ("confirm-s0-seed2", *g2.GATE_WORLDS[0.0]):
        with pytest.raises(m9.SelectRefusal, match="confirm|tuning|gate"):
            confirm(runs, tmp_path / f"job-{bad}", control=bad)
    with pytest.raises(m9.SelectRefusal, match="bare world"):
        confirm(runs, tmp_path / "job-path", (W1, f"../runs/{W2}"))


def test_select_refuses_wrong_world_counts_duplicates_and_a_missing_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    fake_runs(monkeypatch, runs)
    with pytest.raises(m9.SelectRefusal, match="exactly 2"):
        confirm(runs, tmp_path / "a", (W1,))
    with pytest.raises(m9.SelectRefusal, match="exactly 2"):
        confirm(runs, tmp_path / "b", (W1, W2, CTRL))
    with pytest.raises(m9.SelectRefusal, match="needs --control"):
        confirm(runs, tmp_path / "c", control=None)
    with pytest.raises(m9.SelectRefusal, match="distinct"):
        confirm(runs, tmp_path / "d", (W1, W1))
    with pytest.raises(m9.SelectRefusal, match="distinct"):
        confirm(runs, tmp_path / "e", (W1, W2), W1)


def test_select_refuses_the_wrong_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    worlds = standard_worlds()
    worlds[W2] = (0.5, worlds[W2][1])
    for extra in ("m9-s0.75-seed14", "m9-s0.75-seed15"):
        worlds[extra] = (0.75, make_units(extra, n_neg=50))
    fake_runs(monkeypatch, runs, worlds)
    with pytest.raises(m9.SelectRefusal, match=rf"{W2}: s = 0.5, expected 0.75"):
        confirm(runs, tmp_path / "a")
    with pytest.raises(m9.SelectRefusal, match=r"seed14: s = 0.75, expected 0.0"):
        confirm(runs, tmp_path / "b", (W1, "m9-s0.75-seed15"), "m9-s0.75-seed14")


def test_select_refuses_a_world_without_a_recorded_health_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[dict[str, Any] | None] = [
        {"passed": False},
        {"manifest": {}},
        {"passed": "yes"},
        None,
    ]
    for i, record in enumerate(records):
        runs = tmp_path / f"runs{i}"
        fake_runs(monkeypatch, runs, health={CTRL: record})
        with pytest.raises(
            m9.SelectRefusal, match=rf"{CTRL}: (no health.json|health.json does not)"
        ):
            confirm(runs, tmp_path / f"job{i}")


# --- Metric core ----------------------------------------------------------------------------------


def arr(*x: float) -> NDArray[np.float64]:
    return np.array(x, dtype=np.float64)


def flags(*x: int) -> NDArray[np.bool_]:
    return np.array(x, dtype=bool)


def test_auc_hand_computed_cases() -> None:
    # positives {1, 3}, negatives {2, 4}: only (3 > 2) wins, 1 of 4 pairs.
    assert m9.auc(arr(1, 2, 3, 4), flags(1, 0, 1, 0)) == 0.25
    assert m9.auc(arr(1, 2, 3, 4), flags(0, 0, 1, 1)) == 1.0
    assert m9.auc(arr(4, 3, 2, 1), flags(0, 0, 1, 1)) == 0.0
    # positive {2}, negatives {2, 1}: a tie counts 1/2 and a win 1: (0.5 + 1) / 2.
    assert m9.auc(arr(2, 2, 1), flags(1, 0, 0)) == 0.75
    assert m9.auc(arr(7, 7, 7, 7), flags(1, 1, 0, 0)) == 0.5
    assert m9.auc(arr(1, 5, 2, 3, 9, 4), flags(0, 1, 0, 1, 1, 0)) == mx.auc([5, 3, 9], [1, 2, 4])


def test_resample_weights_keep_each_world_class_stratum_size() -> None:
    worlds = ["a"] * 5 + ["b"] * 7
    positive = flags(1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0)
    w = m9.resample_weights(worlds, positive)
    assert w.shape == (m9.N_BOOT, 12) == (2000, 12) and m9.BOOT_SEED == 20261093
    for world in "ab":
        for cls in (True, False):
            cols = [
                i
                for i, (x, p) in enumerate(zip(worlds, positive, strict=True))
                if x == world and p == cls
            ]
            assert (w[:, cols].sum(axis=1) == len(cols)).all()
    assert (w == m9.resample_weights(worlds, positive)).all()  # a seed replays
    assert not (w == m9.resample_weights(worlds, positive, seed=1)).all()
    # the strata are (world, class): another world split gives other draws
    assert not (w == m9.resample_weights(["a"] * 12, positive)).all()


def test_boot_auc_hand_computed_weights() -> None:
    # positives (cols 0, 1) score 3 and 1; the negative (col 2) scores 2.
    score, positive = arr(3, 1, 2), flags(1, 1, 0)
    weights = np.array([[2, 0, 1], [0, 2, 1], [1, 1, 1], [1, 1, 3]])
    got = m9.boot_auc(score, positive, weights)
    assert got.tolist() == [1.0, 0.0, 0.5, 0.5]
    # equal to the AUC of the rows repeated by their weights
    for row, expect in zip(weights, got, strict=True):
        repeated = np.repeat(np.arange(3), row)
        assert m9.auc(score[repeated], positive[repeated]) == expect


def test_boot_auc_is_one_for_a_perfect_ranking_and_paired_across_scores() -> None:
    worlds = ["a"] * 20 + ["b"] * 20
    positive = np.array([i % 2 == 0 for i in range(40)])
    w = m9.resample_weights(worlds, positive)
    perfect = np.where(positive, 1.0, 0.0)
    assert (m9.boot_auc(perfect, positive, w) == 1.0).all()
    e, boot = m9.auc_entry(perfect, positive, w)
    assert e == {"point": 1.0, "lo": 1.0, "hi": 1.0} and (boot == 1.0).all()
    # two score arrays are resampled by the same weights: identical scores, identical resamples
    noise = np.random.default_rng(0).random(40)
    assert (m9.boot_auc(noise, positive, w) == m9.boot_auc(noise.copy(), positive, w)).all()


def test_entry_is_the_2_5_and_97_5_percentiles() -> None:
    boot = np.linspace(0.0, 1.0, 1001)
    e = m9.entry(0.4, boot)
    assert e["point"] == 0.4
    assert e["lo"] == pytest.approx(0.025) and e["hi"] == pytest.approx(0.975)
    assert (e["lo"], e["hi"]) == mx.percentile_interval(boot)


# --- Verdict --------------------------------------------------------------------------------------


def test_verdict_branches() -> None:
    both = {"f64_trees": 0.1, "f64_logistic": 0.2}
    contains = (0.4, 0.6)
    assert m9.verdict(0.55, both, contains)[1] == "CONFIRMED"
    # (a) fails: the few-shot lower bound is not above 0.5 (equal is not above)
    assert m9.verdict(0.5, both, contains)[1] == "NOT CONFIRMED"
    assert m9.verdict(0.45, both, contains)[1] == "NOT CONFIRMED"
    # (b) fails on either F64 model, a lower bound of exactly 0 included
    assert m9.verdict(0.55, {**both, "f64_trees": 0.0}, contains)[1] == "NOT CONFIRMED"
    assert m9.verdict(0.55, {**both, "f64_logistic": -0.01}, contains)[1] == "NOT CONFIRMED"
    # (c) fails: LEAK, whatever (a) and (b) say
    assert m9.verdict(0.55, both, (0.52, 0.7))[1] == "LEAK"
    assert m9.verdict(0.55, both, (0.3, 0.49))[1] == "LEAK"
    assert m9.verdict(0.3, {"f64_trees": -1.0, "f64_logistic": -1.0}, (0.52, 0.7))[1] == "LEAK"
    # the interval's ends count as containing 0.5
    assert m9.verdict(0.55, both, (0.5, 0.7))[1] == "CONFIRMED"
    assert m9.verdict(0.55, both, (0.3, 0.5))[1] == "CONFIRMED"
    criteria, _ = m9.verdict(0.55, {**both, "f64_trees": 0.0}, (0.52, 0.7))
    assert criteria == {
        "a_few_shot_lower_above_half": True,
        "b_beats_both_f64": False,
        "c_control_contains_half": False,
    }
    assert m9.verdict(0.55, both, None)[0]["c_control_contains_half"] is None


def synthetic(
    seed: int = 0,
) -> tuple[list[str], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """(worlds, positive, perfect score, noise) over 2 worlds of 20 + 20 units."""
    worlds = ["a"] * 40 + ["b"] * 40
    positive = np.array([(i % 40) < 20 for i in range(80)])
    return worlds, positive, np.where(positive, 90.0, 10.0), np.random.default_rng(seed).random(80)


def control_of(score: NDArray[np.float64]) -> dict[str, Any]:
    positive = np.array([i < 20 for i in range(40)])
    return {
        "positive": positive,
        "score": score,
        "draw1": score,
        "f64": {rung: np.random.default_rng(3).random(40) for rung in g2.F64_RUNGS},
    }


def run_evaluate(
    few: NDArray[np.float64],
    f64: Mapping[str, NDArray[np.float64]],
    control: NDArray[np.float64] | None,
    mode: str = "confirm",
) -> dict[str, Any]:
    worlds, positive, _, noise = synthetic()
    scores = {"zero_shot": noise, "few_shot": few}
    return m9.evaluate(
        mode,
        worlds,
        positive,
        scores,
        scores,
        f64,
        None if control is None else control_of(control),
    )


def test_evaluate_confirmed() -> None:
    _, _, perfect, noise = synthetic()
    f64 = {"f64_logistic": noise, "f64_trees": 1.0 - noise}
    report = run_evaluate(perfect, f64, np.full(40, 50.0))  # a constant control: AUC 0.5
    assert report["verdict"] == "CONFIRMED"
    assert report["primary"]["few_shot"] == {"point": 1.0, "lo": 1.0, "hi": 1.0}
    assert report["control"]["few_shot"] == {"point": 0.5, "lo": 0.5, "hi": 0.5}
    assert report["criteria"] == {
        "a_few_shot_lower_above_half": True,
        "b_beats_both_f64": True,
        "c_control_contains_half": True,
    }
    for rung in g2.F64_RUNGS:
        assert report["haiku_minus_f64"]["few_shot"][rung]["lo"] > 0
    assert report["n_units"] == {
        "a": {"abusive": 20, "legitimate": 20},
        "b": {"abusive": 20, "legitimate": 20},
    }
    assert set(report["per_world_auc"]) == {"a", "b"}
    assert report["per_world_auc"]["a"]["few_shot"]["point"] == 1.0
    assert report["draw1_primary"]["few_shot"]["point"] == 1.0
    assert report["n_boot"] == 2000 and report["boot_seed"] == 20261093


def test_evaluate_not_confirmed_when_a_fails() -> None:
    _, _, _, noise = synthetic()
    flat = np.full(80, 50.0)  # AUC 0.5: its lower bound is not above 0.5
    f64 = {"f64_logistic": 1.0 - flat, "f64_trees": 1.0 - flat}
    report = run_evaluate(flat, f64, np.full(40, 50.0))
    assert report["criteria"]["a_few_shot_lower_above_half"] is False
    assert report["verdict"] == "NOT CONFIRMED"


def test_evaluate_not_confirmed_when_b_fails() -> None:
    _, _, perfect, noise = synthetic()
    # F64 trees ranks as well as the few-shot score: the paired difference is exactly 0
    f64 = {"f64_logistic": noise, "f64_trees": perfect.copy()}
    report = run_evaluate(perfect, f64, np.full(40, 50.0))
    assert report["criteria"]["a_few_shot_lower_above_half"] is True
    assert report["criteria"]["b_beats_both_f64"] is False
    assert report["haiku_minus_f64"]["few_shot"]["f64_trees"] == {
        "point": 0.0,
        "lo": 0.0,
        "hi": 0.0,
    }
    assert report["haiku_minus_f64"]["few_shot"]["f64_logistic"]["lo"] > 0
    assert report["verdict"] == "NOT CONFIRMED"


def test_evaluate_leak_when_the_control_is_informative() -> None:
    _, _, perfect, noise = synthetic()
    f64 = {"f64_logistic": noise, "f64_trees": 1.0 - noise}
    positive_control = np.array([90.0] * 20 + [10.0] * 20)
    report = run_evaluate(perfect, f64, positive_control)
    assert report["criteria"]["a_few_shot_lower_above_half"] is True
    assert report["criteria"]["b_beats_both_f64"] is True
    assert report["criteria"]["c_control_contains_half"] is False
    assert report["verdict"] == "LEAK"


def test_evaluate_check_mode_reports_a_check_not_a_result() -> None:
    _, _, perfect, noise = synthetic()
    f64 = {"f64_logistic": noise, "f64_trees": 1.0 - noise}
    report = run_evaluate(perfect, f64, None, mode="check")
    assert report["verdict"] == m9.CHECK_VERDICT == "CHECK (not a result)"
    assert report["control"] is None
    assert report["criteria"]["c_control_contains_half"] is None
    assert report["verdict_if_scored"] == "CONFIRMED"  # (a) and (b) alone, no control to leak


def test_differences_are_paired_on_the_shared_resamples() -> None:
    worlds, positive, perfect, noise = synthetic()
    f64 = {"f64_logistic": noise, "f64_trees": noise}
    report = m9.evaluate(
        "confirm",
        worlds,
        positive,
        {"zero_shot": noise, "few_shot": perfect},
        {"zero_shot": noise, "few_shot": perfect},
        f64,
    )
    w = m9.resample_weights(worlds, positive)
    few = m9.boot_auc(perfect, positive, w)
    other = m9.boot_auc(noise, positive, w)
    lo, hi = np.quantile(few - other, [0.025, 0.975])
    diff = report["haiku_minus_f64"]["few_shot"]["f64_trees"]
    assert (diff["lo"], diff["hi"]) == pytest.approx((lo, hi))
    assert diff["point"] == pytest.approx(1.0 - m9.auc(noise, positive))
    zero = report["haiku_minus_f64"]["zero_shot"]["f64_trees"]
    assert (zero["point"], zero["lo"], zero["hi"]) == (0.0, 0.0, 0.0)  # same scores, same resamples


# --- End to end: select, a hand-written driver run, score -----------------------------------------


def write_raw(
    job: Path,
    raw: Path,
    probability: Callable[[Mapping[str, Any]], int | str | list[int | str]],
    version: str = CLI,
) -> None:
    """A driver output for `job`'s rows, scripted by `probability(row)`: one attempt, or a list of
    attempts in order ("unparsed" or a probability)."""
    rows = [json.loads(x) for x in (job / "jobs.jsonl").read_text().splitlines()]
    (raw / "calls").mkdir(parents=True)
    run = {
        "jobs_sha256": json.loads((job / "select.json").read_text())["jobs_sha256"],
        "prompt_sha256": hp.PROMPT_SHA256,
        "route": list(hp.ROUTE_FLAGS),
        "model": hp.MODEL,
        "cli_version": version,
    }
    (raw / "run.json").write_text(json.dumps(run))
    for row in rows:
        specs = probability(row)
        for n, spec in enumerate(specs if isinstance(specs, list) else [specs], 1):
            if spec == "unparsed":
                stdout = json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "modelUsage": {hp.MODEL: {}},
                        "usage": {"input_tokens": 1000, "output_tokens": 7},
                    }
                )
            else:
                stdout = json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "modelUsage": {hp.MODEL: {}},
                        "structured_output": {"probability": spec},
                        "usage": {"input_tokens": 1000, "output_tokens": 7},
                    }
                )
            status, parsed = hr.classify(0, stdout)
            meta = {
                "key": row["key"],
                "attempt": n,
                "status": status,
                "probability": parsed,
                "rc": 0,
                "started": 0.0,
                "wall_s": 3.0,
                "stderr_tail": "",
                "user_sha256": row["user_sha256"],
                "cli_version": version,
                "models": [hp.MODEL],
            }
            (raw / "calls" / f"{row['key']}.a{n}.json").write_text(stdout)
            (raw / "calls" / f"{row['key']}.a{n}.meta.json").write_text(json.dumps(meta))


class Run:
    """select (confirm mode) over synthetic worlds, then a hand-written driver run."""

    def __init__(
        self,
        tmp: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        control_informative: bool = False,
        mode: str = "confirm",
        script: Mapping[str, list[int | str]] | None = None,
    ) -> None:
        self.tmp = tmp
        self.runs = tmp / "runs"
        self.worlds = fake_runs(monkeypatch, self.runs)
        self.positive = {u.unit: u.positive for _, units in self.worlds.values() for u in units}
        self.job = tmp / "job"
        if mode == "check":
            self.meta = m9.select("check", self.runs, self.job)
        else:
            self.meta = confirm(self.runs, self.job)

        def probability(row: Mapping[str, Any]) -> int | list[int | str]:
            if script and row["key"] in script:
                return script[row["key"]]
            if row["world"] == CTRL and not control_informative:
                return 50
            return 85 + row["draw"] if self.positive[row["unit"]] else 15 - row["draw"]

        self.raw = tmp / "job" / "raw"
        write_raw(self.job, self.raw, probability)

        def f64(units: Sequence[R0Unit]) -> dict[str, NDArray[np.float64]]:
            return {
                rung: np.random.default_rng(5 + i).random(len(units))
                for i, rung in enumerate(g2.F64_RUNGS)
            }

        monkeypatch.setattr(m9, "f64_held_out", f64)

    def argv(self, out: Path) -> list[str]:
        return [
            "score", "--job", str(self.job), "--raw", str(self.raw),
            "--runs", str(self.runs), "--out", str(out),
        ]  # fmt: skip


def test_score_confirmed_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    r = Run(tmp_path, monkeypatch)
    report, rows = m9.score(r.job, r.raw, r.runs)
    assert len(rows) == 3 * 46
    assert report["verdict"] == "CONFIRMED"
    assert report["primary"]["few_shot"]["point"] == 1.0
    assert report["primary"]["zero_shot"]["point"] == 1.0
    assert report["control"]["few_shot"] == {"point": 0.5, "lo": 0.5, "hi": 0.5}
    assert report["binding"]["control"] == CTRL
    assert report["binding"]["prereg_sha256"] == r.meta["prereg_sha256"]
    assert report["binding"]["jobs_sha256"] == r.meta["jobs_sha256"]
    assert report["calls"]["cli_version"] == CLI
    assert report["calls"]["attempts"] == r.meta["n_calls"]
    assert report["parse_rate"]["few_shot"]["draws"] == 3 * 3 * 46  # two worlds and the control
    assert report["parse_rate"]["zero_shot"]["draws"] == 2 * 3 * 46
    assert set(report["reliability"]) == {"scope", "zero_shot", "few_shot"}
    by_unit = {x["unit"]: x for x in rows}
    ctrl_rows = [x for x in rows if x["world"] == CTRL]
    assert len(ctrl_rows) == 46 and all("zero_shot" not in x for x in ctrl_rows)
    assert all(set(x) >= {"few_shot", "f64_logistic", "f64_trees"} for x in rows)
    abusive = next(u for u in r.worlds[W1][1] if u.positive and u.unit in by_unit)
    assert by_unit[abusive.unit]["few_shot"]["draws"] == [86, 87, 88]
    assert by_unit[abusive.unit]["few_shot"]["score"] == 87.0
    assert by_unit[abusive.unit]["few_shot"]["draw1_score"] == 86.0
    out = tmp_path / "scored"
    assert m9.main(r.argv(out)) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "M9 CONFIRMED"
    assert "few_shot AUC 1.0000" in printed[1] and "control AUC 0.5000" in printed[1]
    assert json.loads((out / "report.json").read_text())["verdict"] == "CONFIRMED"
    assert len((out / "scores.jsonl").read_text().splitlines()) == 3 * 46
    assert m9.main(r.argv(out)) == 2  # scored once
    assert "scored once" in capsys.readouterr().err


def test_score_leak_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = Run(tmp_path, monkeypatch, control_informative=True)
    report, _ = m9.score(r.job, r.raw, r.runs)
    assert report["control"]["few_shot"]["point"] == 1.0
    assert report["verdict"] == "LEAK"


def test_score_check_mode_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    r = Run(tmp_path, monkeypatch, mode="check")
    out = tmp_path / "scored"
    assert m9.main(r.argv(out)) == 0
    assert capsys.readouterr().out.splitlines()[0] == "M9 CHECK (not a result)"
    report = json.loads((out / "report.json").read_text())
    assert report["control"] is None and report["binding"]["mode"] == "check"
    assert len((out / "scores.jsonl").read_text().splitlines()) == 4
    assert report["calls"]["attempts"] == 24


def test_score_refuses_a_changed_prereg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prereg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    r = Run(tmp_path, monkeypatch)
    prereg.write_text(PREREG_TEXT + "amendment\n")
    with pytest.raises(haiku.ScoreRefusal, match=r"prereg/M9.md differs"):
        m9.score(r.job, r.raw, r.runs)
    out = tmp_path / "scored"
    assert m9.main(r.argv(out)) == 2
    assert "prereg/M9.md differs" in capsys.readouterr().err
    assert not out.exists()  # nothing written on a refusal


def test_score_refuses_incomplete_raw_output_through_load_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = Run(tmp_path, monkeypatch)
    key = json.loads((r.job / "jobs.jsonl").read_text().splitlines()[0])["key"]
    for p in (r.raw / "calls").glob(f"{key}.a1*"):
        p.unlink()
    with pytest.raises(haiku.ScoreRefusal, match="not DONE"):
        m9.score(r.job, r.raw, r.runs)


def test_score_refuses_a_job_that_differs_from_select_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = Run(tmp_path, monkeypatch)
    path = r.job / "select.json"
    meta = json.loads(path.read_text())
    meta["units"][W1]["ids"] = meta["units"][W1]["ids"][1:]
    path.write_text(json.dumps(meta))
    with pytest.raises(haiku.ScoreRefusal, match="differ from select.json"):
        m9.score(r.job, r.raw, r.runs)


def test_score_refuses_a_world_whose_s_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = Run(tmp_path, monkeypatch)
    worlds = dict(r.worlds)
    worlds[CTRL] = (0.75, worlds[CTRL][1])
    monkeypatch.setattr(
        r0,
        "load_world",
        lambda path, **_: (worlds[Path(path).name][0], worlds[Path(path).name][1], 0),
    )
    with pytest.raises(haiku.ScoreRefusal, match=rf"{CTRL}: s = 0.75, expected 0.0"):
        m9.score(r.job, r.raw, r.runs)


def test_score_unparsed_draws_score_50_and_retries_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = m9.pair3_units(standard_worlds()[W1][1], W1)  # the units select will pick
    victim = next(u for u in chosen if u.positive)
    retried = next(u for u in chosen if not u.positive)
    script: dict[str, list[int | str]] = {
        f"{victim.unit}.few_shot.d{d}": ["unparsed"] * 3 for d in (1, 2, 3)
    }
    script[f"{retried.unit}.zero_shot.d1"] = ["unparsed", 12]
    r = Run(tmp_path, monkeypatch, script=script)
    report, rows = m9.score(r.job, r.raw, r.runs)
    by_unit = {x["unit"]: x for x in rows}
    assert by_unit[victim.unit]["few_shot"]["draws"] == ["unparsed"] * 3
    assert by_unit[victim.unit]["few_shot"]["score"] == 50.0
    assert by_unit[retried.unit]["zero_shot"]["draws"][0] == 12
    assert report["parse_rate"]["few_shot"]["unparsed"] == 3
    assert report["parse_rate"]["few_shot"]["units_no_parsed"] == 1
    assert report["parse_rate"]["zero_shot"]["ok_after_retry"] == 1
