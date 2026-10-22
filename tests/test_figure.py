"""Known-answer tests for testbed/eval/figure.py (the stdlib SVG line chart)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from testbed.eval import figure
from testbed.eval.figure import line_chart_svg

NS = "{http://www.w3.org/2000/svg}"
# With WIDTH 760, LEFT 64, RIGHT 190, TOP 44, BOTTOM 56 and HEIGHT 460, the plot box is
# x in [64, 570] and y in [44, 404] pixels.
SERIES = {
    "a": [(0.0, 0.4, None, None), (0.5, 0.7, 0.6, 0.8), (1.0, 1.0, None, None)],
    "b": [(0.0, 0.5, None, None), (1.0, 0.9, 0.85, 0.95)],
}


def _chart(**kwargs: object) -> ET.Element:
    args: dict[str, object] = {"title": "T", "x_label": "s", "y_label": "AUC", **kwargs}
    return ET.fromstring(line_chart_svg(SERIES, **args))  # type: ignore[arg-type]


def _points(root: ET.Element, name: str) -> list[tuple[float, float]]:
    for line in root.iter(f"{NS}polyline"):
        if line.get("data-series") == name:
            pairs = (p.split(",") for p in (line.get("points") or "").split())
            return [(float(x), float(y)) for x, y in pairs]
    raise AssertionError(f"no polyline for {name}")


def test_one_polyline_per_series_with_explicit_colours() -> None:
    root = _chart()
    lines = list(root.iter(f"{NS}polyline"))
    assert [ln.get("data-series") for ln in lines] == ["a", "b"]
    assert all(ln.get("stroke", "").startswith("#") and ln.get("fill") == "none" for ln in lines)
    assert lines[0].get("stroke") != lines[1].get("stroke")
    background = next(root.iter(f"{NS}rect"))
    assert background.get("fill") == "#ffffff"


def test_points_land_on_the_known_axis_mapping() -> None:
    root = _chart()
    # x: 0 -> 64, 0.5 -> 317, 1 -> 570. y over (0.4, 1.0): 0.4 -> 404, 0.7 -> 224, 1.0 -> 44.
    assert _points(root, "a") == [(64.0, 404.0), (317.0, 224.0), (570.0, 44.0)]
    assert _points(root, "b") == [(64.0, 344.0), (570.0, 104.0)]
    assert figure.x_pixel(0.5, (0.0, 1.0)) == 317.0
    assert figure.y_pixel(0.7, (0.4, 1.0)) == pytest.approx(224.0)


def test_whisker_spans_lo_to_hi() -> None:
    root = _chart()
    verticals = [
        ln
        for ln in root.iter(f"{NS}line")
        if ln.get("x1") == ln.get("x2") == "317.00" and ln.get("stroke") != "#dddddd"
    ]
    spans = {(float(ln.get("y1", "")), float(ln.get("y2", ""))) for ln in verticals}
    assert (164.0, 284.0) in spans  # hi 0.8 -> 164, lo 0.6 -> 284


def test_reference_line_is_dashed_at_the_reference_value() -> None:
    dashed = [ln for ln in _chart().iter(f"{NS}line") if ln.get("stroke-dasharray")]
    assert len(dashed) == 1
    assert float(dashed[0].get("y1", "")) == 344.0  # 0.5 on (0.4, 1.0)
    root = ET.fromstring(
        line_chart_svg(SERIES, title="t", x_label="x", y_label="y", reference=None)
    )
    assert not [ln for ln in root.iter(f"{NS}line") if ln.get("stroke-dasharray")]


def test_values_outside_the_range_clamp_to_the_frame() -> None:
    series = {"a": [(0.0, 0.1, None, None), (1.0, 1.5, None, None)]}
    root = ET.fromstring(line_chart_svg(series, title="t", x_label="x", y_label="y"))
    assert _points(root, "a") == [(64.0, 404.0), (570.0, 44.0)]


def test_text_is_escaped_and_a_single_x_is_padded() -> None:
    series = {"a<b": [(0.75, 0.8, 0.7, 0.9)]}
    root = ET.fromstring(line_chart_svg(series, title="A & B", x_label="x", y_label="y"))
    assert "A & B" in {t.text for t in root.iter(f"{NS}text")}
    assert _points(root, "a<b") == [(317.0, 164.0)]  # the pad puts the lone x mid-axis


def test_no_points_is_an_error() -> None:
    with pytest.raises(ValueError):
        line_chart_svg({"a": []}, title="t", x_label="x", y_label="y")


def test_y_ticks_sit_on_round_values() -> None:
    assert figure.y_ticks((0.4, 1.0)) == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert figure.y_ticks((0.0, 1.0)) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert figure.y_ticks((0.45, 0.95)) == [0.5, 0.6, 0.7, 0.8, 0.9]
