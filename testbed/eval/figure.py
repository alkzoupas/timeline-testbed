"""A stdlib-only SVG line chart with interval whiskers (no plotting package is installed).

Every fill and stroke is explicit, so the chart reads on a white page in any viewer. One
`<polyline>` per series; each point may carry a (lo, hi) whisker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape, quoteattr

Point = tuple[float, float, float | None, float | None]  # (x, y, lo, hi)

WIDTH, HEIGHT = 760, 460
LEFT, RIGHT, TOP, BOTTOM = 64, 190, 44, 56  # margins; the right one holds the legend
INK, GRID, FRAME = "#222222", "#dddddd", "#888888"
# Okabe-Ito, which stays distinguishable under the common colour-vision deficiencies.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999")
WHISKER = 4  # half-width of a whisker cap, px
MARKER = 3.5  # marker radius, px


def axis_mapping(
    xs: Sequence[float], y_range: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The plotted (x_min, x_max) and the y range. A single x value gets a unit pad."""
    lo, hi = min(xs), max(xs)
    return ((lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)), y_range


def x_pixel(x: float, x_range: tuple[float, float]) -> float:
    left, right = LEFT, WIDTH - RIGHT
    return left + (x - x_range[0]) / (x_range[1] - x_range[0]) * (right - left)


def y_pixel(y: float, y_range: tuple[float, float]) -> float:
    """Values outside `y_range` are clamped to the frame."""
    clamped = min(max(y, y_range[0]), y_range[1])
    top, bottom = TOP, HEIGHT - BOTTOM
    return bottom - (clamped - y_range[0]) / (y_range[1] - y_range[0]) * (bottom - top)


def y_ticks(y_range: tuple[float, float]) -> list[float]:
    """Ticks at a round step (the smallest of 1, 2, 2.5, 5 x 10^k giving at most 7 intervals)."""
    span = y_range[1] - y_range[0]
    step = next(
        m * 10.0**e for e in range(-4, 3) for m in (1, 2, 2.5, 5) if span / (m * 10.0**e) <= 7
    )
    first = round(y_range[0] / step)
    ticks = [round(k * step, 10) for k in range(first, round(y_range[1] / step) + 1)]
    return [t for t in ticks if y_range[0] - 1e-9 <= t <= y_range[1] + 1e-9]


def _num(value: float) -> str:
    return f"{value:.2f}"


def _tick(value: float) -> str:
    return f"{value:g}"


def line_chart_svg(
    series: Mapping[str, Sequence[Point]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    y_range: tuple[float, float] = (0.4, 1.0),
    reference: float | None = 0.5,
) -> str:
    """The chart as an SVG document. `reference` is drawn as a dashed horizontal line."""
    points = [p for pts in series.values() for p in pts]
    if not points:
        raise ValueError("no points to plot")
    if not y_range[0] < y_range[1]:
        raise ValueError("y_range must be increasing")
    x_range, y_range = axis_mapping([p[0] for p in points], y_range)
    left, right, top, bottom = LEFT, WIDTH - RIGHT, TOP, HEIGHT - BOTTOM
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff" stroke="none"/>',
        f'<text x="{left}" y="26" font-size="16" font-weight="bold" fill="{INK}">'
        f"{escape(title)}</text>",
    ]
    for value in y_ticks(y_range):  # y grid and labels
        gy = _num(y_pixel(value, y_range))
        out.append(
            f'<line x1="{left}" y1="{gy}" x2="{right}" y2="{gy}" stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{_num(y_pixel(value, y_range) + 4)}" font-size="11" '
            f'text-anchor="end" fill="{INK}">{_tick(value)}</text>'
        )
    for x in sorted({p[0] for p in points}):  # x ticks at the data's own x values
        px = _num(x_pixel(x, x_range))
        out.append(
            f'<line x1="{px}" y1="{bottom}" x2="{px}" y2="{bottom + 5}" stroke="{FRAME}" '
            'stroke-width="1"/>'
        )
        out.append(
            f'<text x="{px}" y="{bottom + 19}" font-size="11" text-anchor="middle" '
            f'fill="{INK}">{_tick(x)}</text>'
        )
    out.append(
        f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" '
        f'fill="none" stroke="{FRAME}" stroke-width="1"/>'
    )
    if reference is not None and y_range[0] <= reference <= y_range[1]:
        ry = _num(y_pixel(reference, y_range))
        out.append(
            f'<line x1="{left}" y1="{ry}" x2="{right}" y2="{ry}" stroke="{INK}" '
            'stroke-width="1" stroke-dasharray="5,4"/>'
        )
    out.append(
        f'<text x="{(left + right) / 2}" y="{HEIGHT - 12}" font-size="12" text-anchor="middle" '
        f'fill="{INK}">{escape(x_label)}</text>'
    )
    out.append(
        f'<text x="16" y="{(top + bottom) / 2}" font-size="12" text-anchor="middle" '
        f'fill="{INK}" transform="rotate(-90 16 {(top + bottom) / 2})">{escape(y_label)}</text>'
    )
    for index, (name, pts) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        ordered = sorted(pts)
        coords = " ".join(
            f"{_num(x_pixel(x, x_range))},{_num(y_pixel(y, y_range))}" for x, y, _, _ in ordered
        )
        out.append(
            f'<polyline data-series={quoteattr(name)} points={quoteattr(coords)} fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        for x, y, lo, hi in ordered:
            px = _num(x_pixel(x, x_range))
            if lo is not None and hi is not None:
                y1, y2 = _num(y_pixel(hi, y_range)), _num(y_pixel(lo, y_range))
                out.append(
                    f'<line x1="{px}" y1="{y1}" x2="{px}" y2="{y2}" stroke="{color}" '
                    'stroke-width="1.25"/>'
                )
                for cap in (y1, y2):
                    out.append(
                        f'<line x1="{_num(float(px) - WHISKER)}" y1="{cap}" '
                        f'x2="{_num(float(px) + WHISKER)}" y2="{cap}" stroke="{color}" '
                        'stroke-width="1.25"/>'
                    )
            out.append(
                f'<circle cx="{px}" cy="{_num(y_pixel(y, y_range))}" r="{MARKER}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
        ly = top + 14 + 20 * index
        out.append(
            f'<line x1="{right + 14}" y1="{ly}" x2="{right + 38}" y2="{ly}" stroke="{color}" '
            'stroke-width="2"/>'
        )
        out.append(
            f'<text x="{right + 44}" y="{ly + 4}" font-size="12" fill="{INK}">{escape(name)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"
