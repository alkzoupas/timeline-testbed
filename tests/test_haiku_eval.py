"""Known-answer tests for the M8 pilot's job builder and scorer (testbed/eval/haiku.py).

Units are synthetic `R0Unit` lists, so no store is opened: `select` runs with `r0.load_world`
replaced, and `score` with `load_f64` replaced and a hand-written driver output. `claude` is never
run."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from testbed.eval import g2, haiku, r0
from testbed.eval.r0 import TEST_PAIRS, R0Unit
from testbed.models import haiku_prompt as hp
from testbed.models import haiku_run as hr
from testbed.world.families import PAIRS

T_PAIRS = (2, 3, 4, 6)
CLI = "9.9.9 (fake)"


def make_units(
    world: str, pairs: Sequence[int] = tuple(PAIRS), n_pos: int = 4, n_neg: int = 6
) -> list[R0Unit]:
    units = []
    for pair in pairs:
        for positive in (True, False):
            for k in range(n_pos if positive else n_neg):
                tag = f"p{pair}{'a' if positive else 'l'}{k:02d}"
                family = PAIRS[pair][0 if positive else 1]
                units.append(
                    R0Unit(
                        world,
                        f"{world}:{tag}",
                        family,
                        pair,
                        positive,
                        (),
                        (),
                        0.0,
                        None,
                        f"snap-{tag}",
                    )
                )
    return units


def make_renders(*worlds: list[R0Unit]) -> dict[str, dict[str, str]]:
    return {w[0].world: {u.unit: f"render of {u.world}/{u.unit}" for u in w} for w in worlds}


# --- Selection ------------------------------------------------------------------------------------


def test_examples_per_side_and_fixed_interleaved_order() -> None:
    tune = make_units("t")
    examples = haiku.pick_examples(tune)
    assert set(examples) == {"A", "B"}
    # Side A tests pairs 4-6 and is shown pairs 1-3; side B the reverse.
    assert [e.pair for e in examples["A"]] == [1, 2, 3, 1, 2, 3]
    assert [e.pair for e in examples["B"]] == [4, 5, 6, 4, 5, 6]
    for side in examples:
        assert [e.positive for e in examples[side]] == [True, False, True, False, True, False]
        assert len({e.unit for e in examples[side]}) == 6
        assert not {e.pair for e in examples[side]} & set(TEST_PAIRS[side])


def test_examples_are_deterministic_and_ignore_input_order() -> None:
    tune = make_units("t")
    first = haiku.pick_examples(tune)
    shuffled = [tune[i] for i in np.random.default_rng(1).permutation(len(tune))]
    assert haiku.pick_examples(shuffled) == first
    assert haiku.pick_examples(tune) == first
    # Known answer for EXAMPLE_SEED: the draws are one stream over pairs 1..6, abusive first.
    rng = np.random.default_rng(haiku.EXAMPLE_SEED)
    draws = {
        (p, pos): int(rng.integers(n)) for p in sorted(PAIRS) for pos, n in ((True, 4), (False, 6))
    }

    def expect(p: int, pos: bool) -> str:
        return f"snap-p{p}{'a' if pos else 'l'}{draws[(p, pos)]:02d}"

    assert [e.unit for e in first["A"]] == [
        expect(1, True),
        expect(2, False),
        expect(3, True),
        expect(1, False),
        expect(2, True),
        expect(3, False),
    ]
    assert first["A"][0].unit == "snap-p1a02"  # pinned, so a changed rule or seed is seen


def test_pilot_subset_is_every_abusive_and_forty_legitimate_per_t_pair() -> None:
    units = make_units("p", n_pos=5, n_neg=50)
    chosen = haiku.pilot_units(units)
    assert {u.pair for u in chosen} == set(T_PAIRS)  # none from pairs 1 and 5
    for pair in T_PAIRS:
        assert sum(u.pair == pair and u.positive for u in chosen) == 5
        assert (
            sum(u.pair == pair and not u.positive for u in chosen) == haiku.N_LEGIT_PER_PAIR == 40
        )
    assert len(chosen) == 4 * 45
    assert [u.unit for u in chosen] == sorted(u.unit for u in chosen)
    reversed_input = haiku.pilot_units(units[::-1])
    assert chosen == reversed_input == haiku.pilot_units(units)


def test_check_subset_one_abusive_one_legitimate_per_side_never_an_example() -> None:
    tune = make_units("t")
    examples = haiku.pick_examples(tune)
    banned = {e.unit for side in examples.values() for e in side}
    chosen = haiku.check_units(tune, banned)
    assert len(chosen) == 4
    for side, units in (("A", chosen[:2]), ("B", chosen[2:])):
        assert [u.positive for u in units] == [True, False]
        assert all(u.pair in r0.t_pairs(side) for u in units)
    assert not {u.unit for u in chosen} & banned
    assert haiku.check_units(tune[::-1], banned) == chosen


def build(mode: str = "pilot") -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    tune = make_units(haiku.TUNE_WORLD)
    pilot = make_units(haiku.PILOT_WORLD, n_pos=5, n_neg=50)
    renders = make_renders(tune, pilot)
    rows, meta = haiku.build_selection(mode, tune, pilot, renders)
    return rows, meta, renders


def test_job_rows_have_no_label_field_and_exact_fields() -> None:
    rows, _, _ = build()
    assert hr.JOB_FIELDS == (
        "key",
        "world",
        "unit",
        "variant",
        "draw",
        "prompt_sha256",
        "user",
        "user_sha256",
    )
    for row in rows:
        assert set(row) == set(hr.JOB_FIELDS)
        assert row["prompt_sha256"] == hp.PROMPT_SHA256
        assert row["user_sha256"] == hr.sha256_text(row["user"])
        assert hr.KEY_PATTERN.fullmatch(row["key"])


def test_user_messages_are_the_prompt_builders_with_the_sides_examples() -> None:
    rows, meta, renders = build()
    tune = renders[haiku.TUNE_WORLD]
    units = {u.unit: u for u in make_units(haiku.PILOT_WORLD, n_pos=5, n_neg=50)}
    for row in rows:
        side = r0.side_of(units[row["unit"]].pair)
        render = renders[haiku.PILOT_WORLD][row["unit"]]
        if row["variant"] == "zero_shot":
            assert row["user"] == hp.zero_shot_user(render)
        else:
            shots = [(tune[e["unit"]], e["class"] == "abusive") for e in meta["examples"][side]]
            assert row["user"] == hp.few_shot_user(shots, render)
    assert {r["world"] for r in rows} == {haiku.PILOT_WORLD}


def test_order_is_one_permutation_of_units_variants_draws() -> None:
    rows, meta, _ = build()
    keys = [r["key"] for r in rows]
    assert len(set(keys)) == len(keys) == meta["n_calls"] == 180 * 2 * haiku.K
    expect = {
        f"{unit}.{variant}.d{draw}"
        for unit in meta["units"]
        for variant in haiku.VARIANTS
        for draw in range(1, haiku.K + 1)
    }
    assert set(keys) == expect
    assert keys != sorted(keys)  # shuffled
    assert len({(r["unit"], r["variant"]) for r in rows[:30]}) > 20  # interleaved
    again, _, _ = build()
    assert [r["key"] for r in again] == keys


def test_select_meta_records_choices_and_counts() -> None:
    _, meta, _ = build()
    assert meta["mode"] == "pilot" and meta["world"] == haiku.PILOT_WORLD
    assert meta["f64_pool"] == list(g2.GATE_WORLDS[0.75])
    assert meta["k"] == 3 and meta["variants"] == ["zero_shot", "few_shot"]
    assert meta["prompt_sha256"] == hp.PROMPT_SHA256
    assert meta["counts"]["examples_per_side"] == {"A": 6, "B": 6}
    assert meta["counts"]["units_per_pair"]["2"] == {"abusive": 5, "legitimate": 40}
    assert "1" not in meta["counts"]["units_per_pair"]
    assert [e["pair"] for e in meta["examples"]["A"]] == [1, 2, 3, 1, 2, 3]
    _, check, _ = build("check")
    assert check["f64_pool"] == [haiku.TUNE_WORLD] and check["world"] == haiku.TUNE_WORLD
    assert check["n_calls"] == 4 * 2 * 3 == 24


def fake_worlds(monkeypatch: pytest.MonkeyPatch, runs: Path) -> None:
    worlds = {
        haiku.TUNE_WORLD: make_units(haiku.TUNE_WORLD),
        haiku.PILOT_WORLD: make_units(haiku.PILOT_WORLD, n_pos=5, n_neg=50),
    }
    for name, units in worlds.items():
        (runs / name).mkdir(parents=True)
        lines = [
            json.dumps({"unit": u.unit, "text": f"render of {u.world}/{u.unit}"}) for u in units
        ]
        (runs / name / "renders.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(
        haiku.r0, "load_world", lambda path, **_: (0.75, list(worlds[Path(path).name]), 0)
    )


def test_select_writes_the_job_and_its_hash_and_refuses_a_used_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    fake_worlds(monkeypatch, runs)
    out = tmp_path / "job"
    meta = haiku.select("pilot", runs, out)
    assert hr.sha256_text((out / "jobs.jsonl").read_text()) == meta["jobs_sha256"]
    assert json.loads((out / "select.json").read_text()) == meta
    rows, digest = hr.read_jobs(out / "jobs.jsonl")  # the driver accepts what select wrote
    assert len(rows) == meta["n_calls"] and digest == meta["jobs_sha256"]
    with pytest.raises(FileExistsError):
        haiku.select("pilot", runs, out)


def test_select_refuses_a_changed_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hp, "PROMPT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="PROMPT_SHA256"):
        haiku.select("check", tmp_path / "runs", tmp_path / "job")


# --- Unit scores ----------------------------------------------------------------------------------


def test_unit_scores_mean_of_parsed_draws_and_fallbacks() -> None:
    mean, first = haiku.unit_scores(
        [[10, 20, 60], [None, 40, 80], [None, None, None], [70, None, None]]
    )
    assert mean.tolist() == [30.0, 60.0, 50.0, 70.0]
    assert first.tolist() == [10.0, 50.0, 50.0, 70.0]


# --- Primary and go -------------------------------------------------------------------------------


def eval_units() -> list[R0Unit]:
    return make_units("w", pairs=T_PAIRS, n_pos=6, n_neg=10)


def signal(units: Sequence[R0Unit], strength: float, seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.array([strength * u.positive for u in units]) + rng.normal(0, 1, len(units))


def constant_f64(n: int) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(5)
    return {rung: rng.random(n) for rung in g2.F64_RUNGS}


def run_eval(
    units: Sequence[R0Unit], zero: NDArray[np.float64], few: NDArray[np.float64]
) -> dict[str, Any]:
    scores = {"zero_shot": zero, "few_shot": few}
    return haiku.evaluate(units, scores, scores, constant_f64(len(units)))


def test_perfect_ranking_is_primary_one_and_go() -> None:
    units = eval_units()
    perfect = np.array([100.0 * u.positive for u in units])
    report = run_eval(units, perfect, perfect)
    for variant in haiku.VARIANTS:
        assert report["primary"][variant]["point"] == 1.0
        assert report["primary"][variant]["lower_bound"] == 1.0
        assert report["draw1_primary"][variant]["point"] == 1.0
    assert report["go"]["verdict"] == "GO"
    assert report["n_boot"] == r0.N_BOOT == 2000 and report["boot_seed"] == 20261033


def test_reversed_ranking_is_primary_zero_and_no_go() -> None:
    units = eval_units()
    reversed_scores = np.array([100.0 * (not u.positive) for u in units])
    report = run_eval(units, reversed_scores, reversed_scores)
    assert report["primary"]["zero_shot"]["point"] == 0.0
    assert report["go"]["verdict"] == "NO-GO"


def test_go_when_only_one_variant_clears() -> None:
    units = eval_units()
    perfect = np.array([100.0 * u.positive for u in units])
    noise = np.random.default_rng(3).random(len(units))
    for zero, few in ((perfect, noise), (noise, perfect)):
        report = run_eval(units, zero, few)
        bounds = report["go"]["lower_bounds"]
        assert max(bounds.values()) == 1.0 and min(bounds.values()) < 0.5
        assert report["go"]["verdict"] == "GO"


def test_lower_bound_is_taken_at_the_go_level_on_the_shared_resamples() -> None:
    assert haiku.GO_LEVEL == g2.GATE_LEVEL == pytest.approx(0.975)
    units = eval_units()
    score = signal(units, 1.0)
    report = run_eval(units, score, signal(units, 0.0, seed=1))
    comps = r0.primary_comparisons(units)
    point, boot = r0.primary_boot(score, comps)
    assert report["primary"]["zero_shot"]["point"] == point
    assert report["primary"]["zero_shot"]["lower_bound"] == g2.lower_bound(boot, haiku.GO_LEVEL)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    assert (report["primary"]["zero_shot"]["lo"], report["primary"]["zero_shot"]["hi"]) == (lo, hi)
    assert report["primary"]["zero_shot"]["lower_bound"] < lo  # the 1.25th percentile
    # Paired difference to F64 uses the same resamples.
    f64 = constant_f64(len(units))
    f_point, f_boot = r0.primary_boot(f64["f64_trees"], comps)
    diff = report["haiku_minus_f64"]["zero_shot"]["f64_trees"]
    assert diff["point"] == pytest.approx(point - f_point)
    assert diff["lo"] == pytest.approx(np.quantile(boot - f_boot, 0.025))


def test_per_pair_auc_and_missing_class() -> None:
    units = eval_units()
    perfect = np.array([100.0 * u.positive for u in units])
    entries = haiku.pair_aucs(units, perfect)
    assert sorted(entries) == ["2", "3", "4", "6"]
    assert entries["2"]["auc"] == 1.0 and entries["2"]["n_pos"] == 6 and entries["2"]["n_neg"] == 10
    only_pos = [u for u in units if u.pair != 3 or u.positive]
    got = haiku.pair_aucs(only_pos, np.zeros(len(only_pos)))
    assert got["3"]["auc"] is None and "reason" in got["3"]
    assert got["2"]["auc"] == 0.5


# --- Reliability and parse rate -------------------------------------------------------------------


def test_reliability_identical_draws_is_one() -> None:
    col = np.array([5.0, 20.0, 35.0, 80.0, 90.0])
    out = haiku.reliability(np.stack([col, col, col], axis=1))
    assert out["n_units"] == 5
    assert out["mean_pearson"] == pytest.approx(1.0)
    assert out["mean_spearman"] == pytest.approx(1.0)
    assert out["spearman_brown"] == pytest.approx(1.0)


def test_reliability_hand_computed_case() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0])
    out = haiku.reliability(np.stack([a, a, a[::-1]], axis=1))
    # r(1,2) = 1; r(1,3) = r(2,3) = -1; the mean is -1/3 and SB = 3r / (1 + 2r) = -3.
    assert out["pairs"]["1-2"]["pearson"] == pytest.approx(1.0)
    assert out["pairs"]["1-3"]["pearson"] == pytest.approx(-1.0)
    assert out["mean_pearson"] == pytest.approx(-1 / 3)
    assert out["spearman_brown"] == pytest.approx(-3.0)
    assert haiku.spearman_brown(0.5, 3) == pytest.approx(0.75)
    assert haiku.spearman_brown(-0.5, 3) is None  # 1 + (k - 1) r = 0
    # Monotone but nonlinear: Spearman 1, Pearson below 1.
    x, y = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 4.0, 9.0, 16.0])
    assert haiku.spearman(x, y) == pytest.approx(1.0)
    assert haiku.pearson(x, y) == pytest.approx(25 / 645**0.5)  # sxy 25, sxx 5, syy 129


def test_reliability_undefined_cases_are_null() -> None:
    const = np.array([[1.0, 1.0, 1.0]] * 5)
    assert haiku.reliability(const)["mean_pearson"] is None
    assert haiku.reliability(const)["spearman_brown"] is None
    two = haiku.reliability(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    assert two["mean_pearson"] is None and two["n_units"] == 2
    partial = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0], [2.0, 2.5, 3.0], [9.0, 8.0, 10.0]])
    assert haiku.reliability(partial)["n_units"] == 3  # the row with an unparsed draw is dropped


def test_parse_rate_counts() -> None:
    ok, un, err = hr.OK, hr.UNPARSED, hr.ERROR
    units = [
        [[ok], [un, ok], [err, ok]],  # ok first try; ok after a retry; error attempts don't count
        [[un, un, un], [un, un, un], [un, un, un]],  # a unit with no parsed draw
        [[ok], [err, err, ok], [un, un, ok]],
    ]
    assert haiku.parse_rate(units) == {
        "draws": 9,
        "ok_first_try": 4,
        "ok_after_retry": 2,
        "unparsed": 3,
        "units_no_parsed": 1,
        "error_attempts": 3,
    }


# --- The raw output and score ---------------------------------------------------------------------


def ok_stdout(p: int) -> str:
    return json.dumps(
        {
            "type": "result",
            "is_error": False,
            "modelUsage": {hp.MODEL: {}},
            "structured_output": {"probability": p},
            "usage": {"input_tokens": 1000, "output_tokens": 7},
        }
    )


def outcome(spec: int | str) -> tuple[int, str]:
    """(rc, stdout) of a scripted attempt: a probability, "unparsed" or "error"."""
    if spec == "error":
        return 1, ""
    if spec == "unparsed":
        return 0, json.dumps(
            {
                "type": "result",
                "is_error": False,
                "modelUsage": {hp.MODEL: {}},
                "usage": {"input_tokens": 1000, "output_tokens": 7},
            }
        )
    assert isinstance(spec, int)
    return 0, ok_stdout(spec)


class Scenario:
    """A written job directory and driver output over synthetic units."""

    def __init__(
        self, tmp: Path, script: Mapping[str, Sequence[int | str]] | None = None, version: str = CLI
    ) -> None:
        tune = make_units(haiku.TUNE_WORLD)
        self.units = make_units("w", pairs=T_PAIRS, n_pos=3, n_neg=3)
        extra = make_units("w2", pairs=T_PAIRS, n_pos=1, n_neg=1)  # in the F64 pool, not scored
        self.pool = sorted(self.units + extra, key=lambda u: u.group)
        renders = make_renders(tune, self.units)
        self.rows = haiku.build_jobs(self.units, haiku.pick_examples(tune), renders)
        text = "".join(json.dumps(r) + "\n" for r in self.rows)
        self.sha = hr.sha256_text(text)
        self.job, self.raw = tmp / "job", tmp / "raw"
        (self.raw / "calls").mkdir(parents=True)
        self.job.mkdir()
        (self.job / "jobs.jsonl").write_text(text)
        self.select = {
            "mode": "pilot",
            "prompt_version": hp.PROMPT_VERSION,
            "prompt_sha256": hp.PROMPT_SHA256,
            "jobs_sha256": self.sha,
            "world": "w",
            "f64_pool": ["w", "w2"],
            "units": sorted(u.unit for u in self.units),
        }
        (self.job / "select.json").write_text(json.dumps(self.select))
        self.run = {
            "jobs_sha256": self.sha,
            "prompt_sha256": hp.PROMPT_SHA256,
            "route": list(hp.ROUTE_FLAGS),
            "model": hp.MODEL,
            "cli_version": version,
        }
        (self.raw / "run.json").write_text(json.dumps(self.run))
        positive = {u.unit: u.positive for u in self.units}
        for row in self.rows:
            default: list[int | str] = [
                85 + row["draw"] if positive[row["unit"]] else 15 - row["draw"]
            ]
            for n, spec in enumerate((script or {}).get(row["key"], default), 1):
                self.write_attempt(row, n, *outcome(spec), version=version)

    def write_attempt(
        self, row: Mapping[str, Any], n: int, rc: int, stdout: str, *, version: str = CLI
    ) -> None:
        status, probability = hr.classify(rc, stdout)
        meta = {
            "key": row["key"],
            "attempt": n,
            "status": status,
            "probability": probability,
            "rc": rc,
            "started": 0.0,
            "wall_s": 2.0 + n,
            "stderr_tail": "",
            "user_sha256": row["user_sha256"],
            "cli_version": version,
            "models": [hp.MODEL],
        }
        calls = self.raw / "calls"
        (calls / f"{row['key']}.a{n}.json").write_text(stdout)
        (calls / f"{row['key']}.a{n}.meta.json").write_text(json.dumps(meta))

    def patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        f64 = {rung: np.linspace(0.0, 1.0, len(self.pool)) for rung in g2.F64_RUNGS}
        monkeypatch.setattr(haiku, "load_f64", lambda runs, names: (list(self.pool), f64))


def first_key(s: Scenario, variant: str = "zero_shot", draw: int = 1) -> str:
    return next(r["key"] for r in s.rows if r["variant"] == variant and r["draw"] == draw)


def test_score_end_to_end_on_a_hand_written_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = make_units("w", pairs=T_PAIRS, n_pos=3, n_neg=3)
    victim = next(u for u in units if u.positive)
    # one unit's zero-shot draws all unparsed; one draw ok only after a retry
    script: dict[str, list[int | str]] = {
        f"{victim.unit}.zero_shot.d{d}": ["unparsed", "unparsed", "unparsed"] for d in (1, 2, 3)
    }
    other = next(u for u in units if not u.positive)
    script[f"{other.unit}.few_shot.d1"] = ["error", "unparsed", 12]
    s = Scenario(tmp_path, script)
    s.patch(monkeypatch)
    report, rows = haiku.score(s.job, s.raw, tmp_path)
    assert len(rows) == len(s.units) == 24
    by_unit = {r["unit"]: r for r in rows}
    assert by_unit[victim.unit]["zero_shot"]["draws"] == ["unparsed"] * 3
    assert by_unit[victim.unit]["zero_shot"]["score"] == 50.0
    assert by_unit[victim.unit]["zero_shot"]["draw1_score"] == 50.0
    assert by_unit[other.unit]["few_shot"]["draws"][0] == 12
    assert set(by_unit[victim.unit]) >= {"world", "unit", "group", "family", "pair", "positive"}
    assert report["primary"]["few_shot"]["point"] == 1.0
    assert report["go"]["verdict"] == "GO"
    rate = report["parse_rate"]["zero_shot"]
    assert rate["unparsed"] == 3 and rate["units_no_parsed"] == 1 and rate["draws"] == 72
    few = report["parse_rate"]["few_shot"]
    assert few["ok_after_retry"] == 1 and few["error_attempts"] == 1
    assert report["calls"]["cli_version"] == CLI and report["calls"]["models"] == [hp.MODEL]
    n_usage = report["calls"]["attempts"] - 1  # every attempt but the one error carries a usage
    assert report["calls"]["input_tokens"] == 1000 * n_usage
    assert report["calls"]["output_tokens"] == 7 * n_usage
    assert report["calls"]["input_tokens_total"] == 1000 * n_usage  # no cache fields in the fake
    assert report["binding"]["jobs_sha256"] == s.sha
    assert set(report["reliability"]) == set(haiku.VARIANTS)


def test_main_scores_once_and_prints_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    s = Scenario(tmp_path)
    s.patch(monkeypatch)
    out = tmp_path / "scored"
    argv = ["score", "--job", str(s.job), "--raw", str(s.raw), "--runs", str(tmp_path), "--out"]
    assert haiku.main([*argv, str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "M8 pilot GO"
    assert "zero_shot" in printed and "few_shot" in printed
    report = json.loads((out / "report.json").read_text())
    assert report["go"]["verdict"] == "GO"
    assert len((out / "scores.jsonl").read_text().splitlines()) == 24
    with pytest.raises(SystemExit) as e:
        haiku.main([*argv, str(out)])
    assert e.value.code == 2


# --- Refusals -------------------------------------------------------------------------------------


def test_refuses_an_incomplete_job(tmp_path: Path) -> None:
    key = first_key(s := Scenario(tmp_path, {}))
    for p in (s.raw / "calls").glob(f"{key}.a1*"):
        p.unlink()
    with pytest.raises(haiku.ScoreRefusal, match=r"1 of 144 keys are not DONE"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_keys_with_too_few_unparsed_attempts(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    key = first_key(s)
    for p in (s.raw / "calls").glob(f"{key}.a1*"):
        p.unlink()
    row = next(r for r in s.rows if r["key"] == key)
    for n in (1, 2):
        s.write_attempt(row, n, *outcome("unparsed"))
    with pytest.raises(haiku.ScoreRefusal, match="not DONE"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_a_jobs_sha_mismatch(tmp_path: Path) -> None:
    for where in ("select", "run", "file"):
        s = Scenario(tmp_path / where)
        if where == "select":
            (s.job / "select.json").write_text(json.dumps({**s.select, "jobs_sha256": "0" * 64}))
        elif where == "run":
            (s.raw / "run.json").write_text(json.dumps({**s.run, "jobs_sha256": "0" * 64}))
        else:
            with (s.job / "jobs.jsonl").open("a") as f:
                f.write("\n")
        with pytest.raises(haiku.ScoreRefusal, match="jobs.jsonl sha256"):
            haiku.load_raw(s.job, s.raw)


def test_refuses_a_prompt_hash_mismatch(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    (s.raw / "run.json").write_text(json.dumps({**s.run, "prompt_sha256": "1" * 64}))
    with pytest.raises(haiku.ScoreRefusal, match="prompt_sha256"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_a_meta_that_disagrees_with_classify(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    key = first_key(s)
    path = s.raw / "calls" / f"{key}.a1.meta.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({**meta, "probability": meta["probability"] + 1}))
    with pytest.raises(haiku.ScoreRefusal, match="disagrees with classify"):
        haiku.load_raw(s.job, s.raw)
    path.write_text(json.dumps({**meta, "status": "unparsed", "probability": None}))
    with pytest.raises(haiku.ScoreRefusal, match="disagrees with classify"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_a_user_hash_that_differs_from_the_job(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    path = s.raw / "calls" / f"{first_key(s)}.a1.meta.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "user_sha256": "2" * 64}))
    with pytest.raises(haiku.ScoreRefusal, match="user_sha256"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_two_cli_versions(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    path = s.raw / "calls" / f"{first_key(s)}.a1.meta.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "cli_version": "9.9.10"}))
    with pytest.raises(haiku.ScoreRefusal, match="CLI versions"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_a_run_on_another_route_or_model(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    for field, value in (
        ("route", [*hp.ROUTE_FLAGS, "--verbose"]),
        ("route", None),
        ("model", "claude-other"),
        ("model", None),
    ):
        (s.raw / "run.json").write_text(json.dumps({**s.run, field: value}))
        with pytest.raises(haiku.ScoreRefusal, match="route or model"):
            haiku.load_raw(s.job, s.raw)
    run = {k: v for k, v in s.run.items() if k not in ("route", "model")}
    (s.raw / "run.json").write_text(json.dumps(run))  # absent is not the frozen route either
    with pytest.raises(haiku.ScoreRefusal, match="route or model"):
        haiku.load_raw(s.job, s.raw)


def test_refuses_a_cli_version_that_changed_during_the_run(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    (s.raw / "run.json").write_text(json.dumps({**s.run, "cli_version_after": "9.9.10 (fake)"}))
    with pytest.raises(haiku.ScoreRefusal, match="CLI version changed"):
        haiku.load_raw(s.job, s.raw)
    (s.raw / "run.json").write_text(json.dumps({**s.run, "cli_version_after": ""}))
    with pytest.raises(haiku.ScoreRefusal, match="CLI version changed"):
        haiku.load_raw(s.job, s.raw)
    (s.raw / "run.json").write_text(json.dumps({**s.run, "cli_version_after": CLI}))
    assert haiku.load_raw(s.job, s.raw).run["cli_version_after"] == CLI  # equal is accepted
    (s.raw / "run.json").write_text(json.dumps(s.run))
    assert "cli_version_after" not in haiku.load_raw(s.job, s.raw).run  # absent is accepted


def test_refuses_an_api_error_attempt_recorded_as_ok(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    key = first_key(s)
    path = s.raw / "calls" / f"{key}.a1.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "api_error_status": 429}))  # classify now says "error"
    with pytest.raises(haiku.ScoreRefusal, match="disagrees with classify"):
        haiku.load_raw(s.job, s.raw)


def usage_stdout(usage: Any) -> str:
    return json.dumps({"type": "result", "usage": usage})


def attempt_with(stdout: str, n: int = 1) -> haiku.Attempt:
    return haiku.Attempt("k", n, hr.OK, 50, 1.0, CLI, (hp.MODEL,), haiku.usage_of(stdout))


def test_usage_sums_cover_the_cache_fields() -> None:
    full = usage_stdout(
        {
            "input_tokens": 2,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 5000,
            "output_tokens": 7,
            "service_tier": "standard",  # other fields are ignored
        }
    )
    assert haiku.usage_of(full) == {
        "input_tokens": 2,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 5000,
        "output_tokens": 7,
    }
    partial = usage_stdout(
        {"input_tokens": 3, "cache_read_input_tokens": 400, "output_tokens": True}
    )
    assert haiku.usage_of(partial) == {
        "input_tokens": 3,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 400,
        "output_tokens": 0,  # a bool is not a count
    }
    for stdout in ("not json", "[1]", usage_stdout([1]), json.dumps({"type": "result"})):
        assert set(haiku.usage_of(stdout).values()) == {0}
    attempts = {"k1": [attempt_with(full)], "k2": [attempt_with(partial)]}
    raw = haiku.Raw([], {}, {"cli_version": CLI}, attempts)
    block = haiku.calls_block(raw)
    assert block["input_tokens"] == 2 + 3
    assert block["cache_creation_input_tokens"] == 100
    assert block["cache_read_input_tokens"] == 5000 + 400
    assert block["output_tokens"] == 7
    assert block["input_tokens_total"] == (2 + 100 + 5000) + (3 + 400)
    assert block["attempts"] == 2


def test_refuses_a_job_unit_missing_from_the_f64_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Scenario(tmp_path)
    s.pool = [u for u in s.pool if u.unit != s.units[0].unit]
    s.patch(monkeypatch)
    with pytest.raises(haiku.ScoreRefusal, match="not in the F64 pool"):
        haiku.score(s.job, s.raw, tmp_path)
