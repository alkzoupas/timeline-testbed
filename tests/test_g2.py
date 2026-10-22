"""Known-answer tests for gate G2 (`testbed/eval/g2.py`, prereg/M4.md). Coordinator-owned.

The pure core (bounds, verdict, rows) is driven by hand-made resamples; `score` runs end to end on
synthetic pools with a planted probe signal, with the world and embedding loaders replaced."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from testbed.eval import g2
from testbed.eval.r0 import REGIMES, RUNGS, R0Unit, primary_boot, primary_comparisons
from testbed.world.families import PAIRS

# --- Pure core ------------------------------------------------------------------------------------


def test_gate_level_is_bonferroni_for_two_models() -> None:
    assert g2.GATE_LEVEL == pytest.approx(0.975)
    # np.quantile's linear rule: the 1.25th percentile of 0..1999 is 0.0125 * 1999.
    assert g2.lower_bound(np.arange(2000.0)) == pytest.approx(0.0125 * 1999)


@pytest.mark.parametrize(
    ("beats", "null_ok", "expected"),
    [
        ({"9b": True, "4b": False}, {"9b": True, "4b": True}, "PASS"),
        ({"9b": False, "4b": True}, {"9b": True, "4b": True}, "PASS"),
        ({"9b": True, "4b": False}, {"9b": True, "4b": False}, "LEAK"),
        ({"9b": True, "4b": True}, {"9b": False, "4b": True}, "LEAK"),
        ({"9b": False, "4b": False}, {"9b": True, "4b": True}, "FAIL"),
        ({"9b": False, "4b": False}, {"9b": False, "4b": False}, "FAIL"),
    ],
)
def test_verdict(beats: dict[str, bool], null_ok: dict[str, bool], expected: str) -> None:
    assert g2.verdict(beats, null_ok) == expected


def test_verdict_needs_matching_models() -> None:
    with pytest.raises(ValueError):
        g2.verdict({"9b": True}, {"4b": True})
    with pytest.raises(ValueError):
        g2.verdict({}, {})


def _boot(center: float, spread: float = 0.01, n: int = 2000) -> np.ndarray:
    return center + spread * np.linspace(-1.0, 1.0, n)


def test_decide_rows_and_bounds() -> None:
    f64 = {
        s: {"f64_logistic": (0.46, _boot(0.46)), "f64_trees": (0.52, _boot(0.52))}
        for s in (0.75, 0.0)
    }
    probe = {
        0.75: {"9b": (0.60, _boot(0.60)), "4b": (0.515, _boot(0.515))},
        0.0: {"9b": (0.50, _boot(0.50)), "4b": (0.49, _boot(0.49))},
    }
    out = g2.decide(f64, probe)
    rows = {r["name"]: r for r in out["rows"]}
    a9 = rows["a_beats_f64_9b"]
    assert a9["passed"] and a9["r2_primary"] == 0.60
    assert a9["minus"]["f64_trees"]["point"] == pytest.approx(0.08)
    # Paired differences of two equal-spread linspaces are constant.
    assert a9["minus"]["f64_trees"]["lower_bound"] == pytest.approx(0.08)
    a4 = rows["a_beats_f64_4b"]
    assert a4["minus"]["f64_logistic"]["passed"] and not a4["minus"]["f64_trees"]["passed"]
    assert not a4["passed"]
    assert rows["b_null_s0_9b"]["passed"] and not rows["b_null_s0_4b"]["passed"]
    assert out["verdict"] == "LEAK"  # 9b meets (a); 4b's s = 0 interval tops out at 0.4995
    assert out["f64"]["0.75"] == {"f64_logistic": 0.46, "f64_trees": 0.52}


def test_decide_null_interval_is_the_95_percent_one() -> None:
    f64 = {s: {r: (0.5, _boot(0.5)) for r in g2.F64_RUNGS} for s in (0.75, 0.0)}
    # 0.5 sits just above the 97.5th percentile of the s = 0 resamples: (b) fails.
    z = 0.4 + 0.1 * np.arange(2000) / 1999 * 0.95 / 0.975
    out = g2.decide(f64, {0.75: {"9b": (0.5, _boot(0.5))}, 0.0: {"9b": (0.45, z)}})
    row = next(r for r in out["rows"] if r["name"] == "b_null_s0_9b")
    assert row["hi"] < 0.5 and not row["passed"]


# --- Tune files -----------------------------------------------------------------------------------


def test_check_tune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "9b" / "tune.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"selected": {"feature": "mean_final", "C": 0.001}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(g2, "FROZEN", {"9b": ("mean_final", 0.001)})
    monkeypatch.setattr(g2, "TUNE_SHA256", {"9b": digest})
    assert g2.check_tune(tmp_path) == {"9b": digest}
    monkeypatch.setattr(g2, "FROZEN", {"9b": ("mean_16", 0.001)})
    with pytest.raises(RuntimeError, match="selected"):
        g2.check_tune(tmp_path)
    monkeypatch.setattr(g2, "TUNE_SHA256", {"9b": "0" * 64})
    with pytest.raises(RuntimeError, match="sha256"):
        g2.check_tune(tmp_path)
    monkeypatch.setattr(g2, "FROZEN", {})
    monkeypatch.setattr(g2, "TUNE_SHA256", {})
    with pytest.raises(RuntimeError, match="must be set"):
        g2.check_tune(tmp_path)


# --- Synthetic pools ------------------------------------------------------------------------------

N_POS, N_NEG = 30, 60


def _units(world: str, seed: int) -> list[R0Unit]:
    rng = np.random.default_rng(seed)
    out = []
    for pair, (abusive, twin) in PAIRS.items():
        for i in range(N_POS + N_NEG):
            positive = i < N_POS
            out.append(
                R0Unit(
                    world=world,
                    group=f"{world}:{pair}-{i:03d}",
                    family=abusive if positive else twin,
                    pair=pair,
                    positive=positive,
                    f12=tuple(rng.normal(size=12)),
                    f64=tuple(rng.normal(size=64)),
                    detector=float(rng.random()),
                    oracle=float(rng.random()),
                    unit=f"u-{world}-{pair}-{i:03d}",
                )
            )
    return sorted(out, key=lambda u: u.group)


def _pools() -> dict[float, g2.Pool]:
    out = {}
    for s, seed in ((0.75, 1), (0.0, 2)):
        units = sorted(
            _units(f"w{seed}a", seed) + _units(f"w{seed}b", seed + 10), key=lambda u: u.group
        )
        out[s] = g2.Pool(s, [Path(f"w{seed}a"), Path(f"w{seed}b")], units, {})
    return out


def _fake_score_pool(units: Sequence[R0Unit]) -> dict[tuple[str, str], np.ndarray]:
    """Deterministic chance-level scores for every R0 rung (R0's own fits are tested in
    test_r0.py; here they only need to be fixed and finite)."""
    rng = np.random.default_rng(len(units))
    return {(rung, regime): rng.random(len(units)) for rung in RUNGS for regime in REGIMES}


def _wire(monkeypatch: pytest.MonkeyPatch, signal: dict[float, float]) -> None:
    pools = _pools()
    monkeypatch.setattr(g2, "FROZEN", {"9b": ("mean_final", 1e-2), "4b": ("mean_16", 1e-2)})
    monkeypatch.setattr(g2, "check_tune", lambda tune: {})
    monkeypatch.setattr(g2, "load_pool", lambda runs, s: pools[s])
    monkeypatch.setattr(g2, "score_pool", _fake_score_pool)
    # M2's values are replaced by the synthetic pools' own F64 primaries.
    m2 = {}
    for s, pool in pools.items():
        scores = _fake_score_pool(pool.units)
        comps = primary_comparisons(pool.units)
        m2[s] = {r: primary_boot(scores[(r, "held_out")], comps)[0] for r in g2.F64_RUNGS}
    monkeypatch.setattr(g2, "M2_F64", m2)

    def features(embed: Path, m: str, pool: g2.Pool) -> np.ndarray:
        rng = np.random.default_rng(7 if m == "9b" else 8)
        x = rng.normal(size=(len(pool.units), 16))
        planted = np.array([u.positive for u in pool.units], dtype=np.float64)
        x[:, 0] += signal[pool.s] * planted if m == "9b" else 0.0
        return x

    monkeypatch.setattr(g2, "_features", features)


def test_score_passes_a_planted_probe_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, {0.75: 3.0, 0.0: 0.0})
    gate, report, _ = g2.score(Path("runs"), Path("embed"), Path("tune"))
    rows = {r["name"]: r for r in gate["rows"]}
    assert rows["a_beats_f64_9b"]["passed"] and rows["a_beats_f64_9b"]["r2_primary"] > 0.95
    assert not rows["a_beats_f64_4b"]["passed"]
    assert rows["b_null_s0_9b"]["passed"] and rows["b_null_s0_4b"]["passed"]
    assert gate["verdict"] == "PASS"
    # The full report carries the R2 rungs and their paired differences.
    held = report["0.75"]["regimes"]["held_out"]
    assert held["r2_9b"]["primary"]["auc"] == rows["a_beats_f64_9b"]["r2_primary"]
    assert "r2_9b_minus_f64_trees" in report["0.75"]["differences"]["held_out"]


def test_score_fails_without_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, {0.75: 0.0, 0.0: 0.0})
    gate, _, _ = g2.score(Path("runs"), Path("embed"), Path("tune"))
    assert gate["verdict"] == "FAIL"


def test_score_flags_a_leak_at_s0(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, {0.75: 3.0, 0.0: 3.0})
    gate, _, _ = g2.score(Path("runs"), Path("embed"), Path("tune"))
    assert gate["verdict"] == "LEAK"


def test_f64_must_reproduce_m2(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, {0.75: 0.0, 0.0: 0.0})
    shifted = {s: {r: v + 1e-9 for r, v in d.items()} for s, d in g2.M2_F64.items()}
    monkeypatch.setattr(g2, "M2_F64", shifted)
    with pytest.raises(RuntimeError, match="reproduce M2"):
        g2.score(Path("runs"), Path("embed"), Path("tune"))
