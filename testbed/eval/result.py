"""The M7 figure (SPEC §10): AUC against s per rung, T families, held-out regime.

It computes no metric. It reads the points `r0.evaluate` already produced: the R0 curve
(`r0.json`) and the R2 points (`g2` `report.json` or `r2 score`'s `r2.json`), and draws them with
`figure.line_chart_svg`. Each R2 report was scored on its own pooled units, so at every R2 s the
baseline rungs and the oracle must equal R0's to `TOLERANCE`; otherwise the points do not share
units and the figure is refused.

CLI::

    python -m testbed.eval.result --r0 runs/m2-r0/r0.json \
        --r2 runs/m4-g2/report.json runs/m4-r2-s0.5/r2.json --out reports/RESULT-auc-vs-s.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from testbed.eval.figure import Point, line_chart_svg

TOLERANCE = 1e-12
ORACLE_LABEL = "oracle (mean of families)"
# (legend label, rung); the order is the legend order
R0_SERIES = (
    ("detector", "detector"),
    ("F64-w90 logistic", "f64_logistic"),
    ("F64-w90 trees", "f64_trees"),
)
R2_SERIES = (("R2 probe, 9B", "r2_9b"), ("R2 probe, 4B", "r2_4b"))
BASELINES = ("detector", "f64_logistic", "f64_trees", "oracle")


def r2_entries(report: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    """The R2 entries of one report by s: `r2 score` (one entry) or `g2` (entries by s)."""
    if "regimes" in report:
        return {float(report["s"]): report}
    entries: dict[float, Mapping[str, Any]] = {}
    for key, entry in report.items():
        if not isinstance(entry, Mapping) or "regimes" not in entry:
            raise ValueError(f"R2 report entry {key!r} has no 'regimes'")
        entries[float(key)] = entry
    return entries


def merge_r2(reports: Sequence[Mapping[str, Any]]) -> dict[float, Mapping[str, Any]]:
    """All R2 entries by s; the same s in two reports is an error."""
    merged: dict[float, Mapping[str, Any]] = {}
    for report in reports:
        for s, entry in r2_entries(report).items():
            if s in merged:
                raise ValueError(f"s = {s:g} appears in more than one R2 report")
            merged[s] = entry
    return merged


def _primary(entry: Mapping[str, Any], rung: str, s: float) -> Mapping[str, Any]:
    try:
        primary: Mapping[str, Any] = entry["regimes"]["held_out"][rung]["primary"]
    except KeyError as exc:
        raise ValueError(f"s = {s:g}: no held-out primary for rung {rung}") from exc
    return primary


def _point(s: float, rung: str, primary: Mapping[str, Any]) -> Point:
    if rung == "oracle":
        return (s, primary["mean_family_auc"], None, None)
    return (s, primary["auc"], primary["lo"], primary["hi"])


def result_series(
    r0_by_s: Mapping[str, Mapping[str, Any]], r2_by_s: Mapping[float, Mapping[str, Any]]
) -> dict[str, list[Point]]:
    """The legend-ordered series; R2 points are checked against R0's baselines at the same s."""
    r0 = {float(k): v for k, v in r0_by_s.items()}
    for s, entry in r2_by_s.items():
        if s not in r0:
            raise ValueError(f"R2 s = {s:g} is not in the R0 curve")
        for rung in BASELINES:
            key = "mean_family_auc" if rung == "oracle" else "auc"
            ours, theirs = _primary(entry, rung, s)[key], _primary(r0[s], rung, s)[key]
            if abs(ours - theirs) > TOLERANCE:
                raise ValueError(
                    f"s = {s:g}, rung {rung}: R2 report {ours!r} differs from R0 {theirs!r}"
                )
    series: dict[str, list[Point]] = {}
    for label, rung in R0_SERIES:
        series[label] = [_point(s, rung, _primary(r0[s], rung, s)) for s in sorted(r0)]
    for label, rung in R2_SERIES:
        series[label] = [_point(s, rung, _primary(r2_by_s[s], rung, s)) for s in sorted(r2_by_s)]
    series[ORACLE_LABEL] = [_point(s, "oracle", _primary(r0[s], "oracle", s)) for s in sorted(r0)]
    return {label: series[label] for label in (*dict(R0_SERIES), *dict(R2_SERIES), ORACLE_LABEL)}


def result_svg(series: Mapping[str, Sequence[Point]]) -> str:
    return line_chart_svg(
        series,
        title="T-family AUC against s, held-out regime",
        x_label="signal strength s",
        y_label="AUC (mean of sides A and B)",
    )


def series_lines(series: Mapping[str, Sequence[Point]]) -> list[str]:
    """One line per series: `label: s=0 0.500 [0.450, 0.550]; ...` (no interval when absent)."""
    lines = []
    for label, points in series.items():
        cells = []
        for s, y, lo, hi in points:
            cell = f"s={s:g} {y:.3f}"
            if lo is not None and hi is not None:
                cell += f" [{lo:.3f}, {hi:.3f}]"
            cells.append(cell)
        lines.append(f"{label}: " + "; ".join(cells))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r2", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists")
    r0 = json.loads(args.r0.read_text())
    reports = [json.loads(path.read_text()) for path in args.r2]
    series = result_series(r0["by_s"], merge_r2(reports))
    args.out.write_text(result_svg(series))
    print("\n".join(series_lines(series)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
