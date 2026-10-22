"""Known-answer tests for testbed/eval/result.py (the M7 figure assembly), on synthetic fixtures."""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from testbed.eval import result
from testbed.eval.figure import Point

NS = "{http://www.w3.org/2000/svg}"
BASE = {"detector": 0.50, "f64_logistic": 0.55, "f64_trees": 0.60}


def _entry(
    s: float, *, r2: tuple[float, float] | None = None, shift: float = 0.0
) -> dict[str, Any]:
    """An `r0.evaluate`-shaped entry; rung values are simple functions of s."""
    held: dict[str, Any] = {
        rung: {"primary": {"auc": auc + s / 10 + shift, "lo": auc + s / 10 - 0.02, "hi": 0.9}}
        for rung, auc in BASE.items()
    }
    held["oracle"] = {"primary": {"mean_family_auc": 0.8 + s / 10 + shift}}
    if r2 is not None:
        for rung, auc in zip(("r2_9b", "r2_4b"), r2, strict=True):
            held[rung] = {"primary": {"auc": auc, "lo": auc - 0.01, "hi": auc + 0.01}}
    return {"s": s, "regimes": {"held_out": held}}


R0: dict[str, Any] = {
    "by_s": {"0.0": _entry(0.0), "0.5": _entry(0.5), "0.75": _entry(0.75)},
    "scales": ["standard"],
}
G2 = {"0.75": _entry(0.75, r2=(0.7, 0.65)), "0.0": _entry(0.0, r2=(0.5, 0.49))}
R2_ONE = _entry(0.5, r2=(0.6, 0.55))


def _approx(points: list[Point]) -> list[Any]:
    return [
        (s, pytest.approx(y), None if lo is None else pytest.approx(lo), hi)
        for s, y, lo, hi in points
    ]


def test_r2_entries_accepts_both_shapes() -> None:
    assert result.r2_entries(R2_ONE) == {0.5: R2_ONE}
    assert result.r2_entries(G2) == {0.75: G2["0.75"], 0.0: G2["0.0"]}


def test_r2_entries_rejects_a_value_without_regimes() -> None:
    with pytest.raises(ValueError, match="no 'regimes'"):
        result.r2_entries({"0.75": {"s": 0.75}})


def test_merge_rejects_a_duplicate_s() -> None:
    with pytest.raises(ValueError, match="s = 0.5"):
        result.merge_r2([R2_ONE, R2_ONE])
    merged = result.merge_r2([G2, R2_ONE])
    assert sorted(merged) == [0.0, 0.5, 0.75]


def test_series_values_and_order() -> None:
    series = result.result_series(R0["by_s"], result.merge_r2([G2, R2_ONE]))
    assert list(series) == [
        "detector",
        "F64-w90 logistic",
        "F64-w90 trees",
        "R2 probe, 9B",
        "R2 probe, 4B",
        "oracle (mean of families)",
    ]
    assert series["detector"] == _approx(
        [(0.0, 0.5, 0.48, 0.9), (0.5, 0.55, 0.53, 0.9), (0.75, 0.575, 0.555, 0.9)]
    )
    assert series["F64-w90 trees"][1] == (0.5, pytest.approx(0.65), pytest.approx(0.63), 0.9)
    assert series["R2 probe, 9B"] == _approx(
        [(0.0, 0.5, 0.49, 0.51), (0.5, 0.6, 0.59, 0.61), (0.75, 0.7, 0.69, 0.71)]
    )
    assert [p[1] for p in series["R2 probe, 4B"]] == pytest.approx([0.49, 0.55, 0.65])
    assert series["oracle (mean of families)"] == [
        (0.0, pytest.approx(0.8), None, None),
        (0.5, pytest.approx(0.85), None, None),
        (0.75, pytest.approx(0.875), None, None),
    ]


def test_r2_s_missing_from_r0_raises() -> None:
    with pytest.raises(ValueError, match=r"R2 s = 0\.25"):
        result.result_series(R0["by_s"], {0.25: _entry(0.25, r2=(0.5, 0.5))})


@pytest.mark.parametrize("rung", ["detector", "f64_logistic", "f64_trees", "oracle"])
def test_baseline_mismatch_raises(rung: str) -> None:
    bad = copy.deepcopy(R2_ONE)
    primary = bad["regimes"]["held_out"][rung]["primary"]
    primary["mean_family_auc" if rung == "oracle" else "auc"] += 1e-6
    with pytest.raises(ValueError, match=f"s = 0.5, rung {rung}"):
        result.result_series(R0["by_s"], {0.5: bad})


def test_baseline_within_tolerance_passes() -> None:
    near = copy.deepcopy(R2_ONE)
    near["regimes"]["held_out"]["detector"]["primary"]["auc"] += 1e-13
    assert "R2 probe, 9B" in result.result_series(R0["by_s"], {0.5: near})


def test_svg_has_one_polyline_per_series_with_two_or_more_points() -> None:
    series = result.result_series(R0["by_s"], result.merge_r2([G2, R2_ONE]))
    root = ET.fromstring(result.result_svg(series))
    assert [ln.get("data-series") for ln in root.iter(f"{NS}polyline")] == list(series)
    assert "T-family AUC against s, held-out regime" in {t.text for t in root.iter(f"{NS}text")}


def test_svg_draws_a_single_point_series_as_a_one_point_polyline() -> None:
    series = result.result_series(R0["by_s"], {0.5: R2_ONE})
    root = ET.fromstring(result.result_svg(series))
    counts = {
        ln.get("data-series"): len((ln.get("points") or "").split())
        for ln in root.iter(f"{NS}polyline")
    }
    assert counts["detector"] == 3 and counts["R2 probe, 9B"] == counts["R2 probe, 4B"] == 1


def test_series_lines_format() -> None:
    series = result.result_series(R0["by_s"], {0.5: R2_ONE})
    lines = result.series_lines(series)
    assert (
        lines[0] == "detector: s=0 0.500 [0.480, 0.900]; s=0.5 0.550 [0.530, 0.900]; "
        "s=0.75 0.575 [0.555, 0.900]"
    )
    assert lines[3] == "R2 probe, 9B: s=0.5 0.600 [0.590, 0.610]"
    assert lines[5] == "oracle (mean of families): s=0 0.800; s=0.5 0.850; s=0.75 0.875"


def _write(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(obj))
    return path


def test_cli_writes_the_svg_and_refuses_an_existing_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    r0, g2, r2 = (
        _write(tmp_path / "r0.json", R0),
        _write(tmp_path / "g2.json", G2),
        _write(tmp_path / "r2.json", R2_ONE),
    )
    out = tmp_path / "fig.svg"
    assert result.main(["--r0", str(r0), "--r2", str(g2), str(r2), "--out", str(out)]) == 0
    ET.fromstring(out.read_text())
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 6 and printed[3].startswith("R2 probe, 9B: s=0 0.500")
    with pytest.raises(SystemExit) as exc:
        result.main(["--r0", str(r0), "--r2", str(g2), "--out", str(out)])
    assert exc.value.code == 2
