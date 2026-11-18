"""Render docs/before_after.png from the numbers demo.py measured.

Run with:
    python make_chart.py            # reads docs/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e8e7e3"
NEUTRAL = "#c9c8c3"
ACCENT = "#2a78d6"
FLOOR_EDGE = "#a8a7a1"

ROOT = Path(__file__).resolve().parent


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)


def render(results: dict, path: str | Path) -> Path:
    strategies = results["strategies"]
    labels = [s["label"] for s in strategies]
    values = [s["bytes"] / 1e6 for s in strategies]
    keys = [s["key"] for s in strategies]
    baseline = values[keys.index("independent")]
    optimal = values[keys.index("optimal")]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.9), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)

    colors = []
    for key in keys:
        if key == "optimal":
            colors.append(ACCENT)
        elif key == "floor":
            colors.append(SURFACE)
        else:
            colors.append(NEUTRAL)

    bars = left.bar(range(len(values)), values, width=0.62, color=colors, zorder=3)
    floor_bar = bars[keys.index("floor")]
    floor_bar.set_edgecolor(FLOOR_EDGE)
    floor_bar.set_linewidth(1.1)
    floor_bar.set_hatch("///")

    for bar, value, key in zip(bars, values, keys):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + baseline * 0.018,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=INK if key == "optimal" else INK_SOFT,
            fontweight="bold" if key == "optimal" else "normal",
        )
        if key not in ("independent", "floor"):
            left.text(
                bar.get_x() + bar.get_width() / 2,
                value * 0.5,
                f"-{100 * (1 - value / baseline):.1f}%",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#ffffff" if key == "optimal" else INK_SOFT,
            )

    left.set_xticks(range(len(labels)))
    left.set_xticklabels(labels, fontsize=9, color=INK_SOFT, linespacing=1.4)
    left.set_ylabel("distinct layer bytes (MB, gzip)", fontsize=9.5, color=INK_SOFT)
    left.set_ylim(0, baseline * 1.14)
    left.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    left.set_axisbelow(True)
    left.set_title(
        f"Family footprint for {results['images']} service images",
        fontsize=13,
        color=INK,
        loc="left",
        pad=26,
    )
    left.text(
        0.0,
        1.035,
        f"{results['packages']} Python distributions, "
        f"{results['signatures']} co-occurrence signatures, measured compressed sizes",
        transform=left.transAxes,
        fontsize=9.5,
        color=INK_SOFT,
    )
    _style(left)

    bases = results["bases"]
    names = [f"{b['name']}  ({b['images']} images)" for b in bases]
    sizes = [b["bytes"] / 1e6 for b in bases]
    pos = list(range(len(bases)))
    right.barh(pos, sizes, height=0.62, color=ACCENT, zorder=3)
    span = max(sizes) if sizes else 1.0
    for y, (size, base) in enumerate(zip(sizes, bases)):
        shown = f"{size:,.0f}" if size >= 10 else f"{size:,.1f}"
        count = base["packages"]
        right.text(
            size + span * 0.02,
            y,
            f"{shown} MB  {count} package{'' if count == 1 else 's'}",
            va="center",
            fontsize=9,
            color=INK_SOFT,
        )
    right.set_yticks(pos)
    right.set_yticklabels(names, fontsize=9, color=INK_SOFT)
    right.invert_yaxis()
    right.set_xlabel(
        "bytes carried once instead of per image (MB, gzip)", fontsize=9.5, color=INK_SOFT
    )
    right.set_xlim(0, span * 1.42)
    right.xaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    right.set_axisbelow(True)
    right.set_title(
        f"The {len(bases)} shared base images the plan creates",
        fontsize=13,
        color=INK,
        loc="left",
        pad=26,
    )
    right.text(
        0.0,
        1.035,
        f"together they move {baseline - optimal:,.0f} MB out of the per image layers",
        transform=right.transAxes,
        fontsize=9.5,
        color=INK_SOFT,
    )
    _style(right)

    fig.tight_layout(pad=2.0)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    results_path = ROOT / "docs" / "results.json"
    if not results_path.exists():
        print("no results yet: run python demo.py first")
        return 1
    results = json.loads(results_path.read_text(encoding="utf-8"))
    out = render(results, ROOT / "docs" / "before_after.png")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
