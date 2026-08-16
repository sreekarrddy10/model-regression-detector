"""Single-file HTML report.

Everything is inlined - CSS, the trend chart, the data. The report is uploaded
as a CI artifact and opened from a laptop with no network, so a single external
reference would break it exactly when someone needs to read it.

The trend chart is hand-generated SVG rather than a charting library, for the
same reason: no CDN, no bundler, no JavaScript at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .model import ReportData

TEMPLATE_DIR = Path(__file__).parent / "templates"


def sparkline(
    values: Sequence[float], *, width: int = 320, height: int = 60, floor: float = 0.90
) -> str:
    """Inline SVG trend line with the drift floor drawn in.

    Returns an empty string for fewer than two points - a one-point "trend" is a
    dot pretending to be information.
    """
    if len(values) < 2:
        return ""

    pad = 6
    lo = min(min(values), floor) - 0.02
    hi = max(max(values), floor) + 0.02
    span = hi - lo or 1.0

    def x(i: int) -> float:
        return pad + i * (width - 2 * pad) / (len(values) - 1)

    def y(v: float) -> float:
        return height - pad - (v - lo) / span * (height - 2 * pad)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    floor_y = y(floor)
    last_x, last_v = x(len(values) - 1), values[-1]
    colour = "#b42318" if last_v < floor else "#067647"

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="accuracy trend over recent runs">'
        f'<line x1="{pad}" y1="{floor_y:.1f}" x2="{width - pad}" y2="{floor_y:.1f}" '
        f'stroke="#d0d5dd" stroke-dasharray="4 3" stroke-width="1"/>'
        f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{y(last_v):.1f}" r="3.5" fill="{colour}"/>'
        f"</svg>"
    )


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: "—" if v is None else f"{v:.1%}"
    env.filters["signed_pct"] = lambda v: "—" if v is None else f"{v:+.1%}"
    env.filters["num"] = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    env.filters["usd"] = lambda v: "—" if v is None else f"${v:.4f}"
    return env


def render(data: ReportData) -> str:
    env = _environment()
    template = env.get_template("report.html.j2")
    return template.render(
        d=data,
        trend_svg=sparkline(data.trend, floor=0.90),
    )


def write(data: ReportData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data), encoding="utf-8")
    return path
