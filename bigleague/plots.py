"""Matplotlib renderers — strike zone, hot zones, spray chart, stadium outline.

Every renderer:
  1. Selects only the columns it needs (huge perf win on Statcast frames).
  2. Validates non-empty input.
  3. Returns PNG bytes from an in-memory BytesIO buffer.
  4. Calls `gc.collect()` in a `finally` block — mandatory. A figure is a
     web of reference cycles that holds a C-side bitmap, so without the
     collection each render's memory lingers until the cyclic GC happens to
     run.

matplotlib and numpy are imported on the first render, not at
`import bigleague`: together they cost about 40 MB of resident memory, which a
process that never plots (lookups, standings, leaderboards) shouldn't pay.
Figures are built directly on `matplotlib.figure.Figure` rather than through
pyplot, so there is no global figure registry to leak from and no backend to
select: saving a PNG uses the Agg renderer on its own, and the caller's
matplotlib backend is left alone.
"""

from __future__ import annotations

import gc
from io import BytesIO
from typing import TYPE_CHECKING, Any

import polars as pl

from bigleague.stadiums import LEGACY_TEAM_CODES, STADIUM_DATA, STADIUM_TO_TEAM_CODE
from bigleague.stats import hot_zones_ab_counts, hot_zones_grid

if TYPE_CHECKING:  # annotations only; the real imports happen at first render
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.patches

_PITCH_COLORS: dict[str, str] = {
    "FF": "red",
    "SL": "blue",
    "CH": "green",
    "CU": "purple",
    "SI": "orange",
    "FC": "pink",
    "FS": "brown",
    "KC": "teal",
    "ST": "magenta",
    "SV": "olive",
}
_DEFAULT_PITCH_COLOR = "gray"

# Strike zone outline — feet from plate center.
_ZONE_X_MIN, _ZONE_X_MAX = -0.83, 0.83
_ZONE_Z_MIN, _ZONE_Z_MAX = 1.5, 3.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_png(fig: matplotlib.figure.Figure, dpi: int = 150) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    return buf.read()


def _new_fig(
    figsize: tuple[float, float], dpi: int | None = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """A figure with one axes, built without pyplot (and imported on demand)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize) if dpi is None else Figure(figsize=figsize, dpi=dpi)
    return fig, fig.add_subplot()


def _draw_strike_zone(ax: matplotlib.axes.Axes) -> None:
    """Outer rectangle + 3x3 grid lines for the official MLB strike zone."""
    import matplotlib.patches as mpatches

    rect = mpatches.Rectangle(
        (_ZONE_X_MIN, _ZONE_Z_MIN),
        width=_ZONE_X_MAX - _ZONE_X_MIN,
        height=_ZONE_Z_MAX - _ZONE_Z_MIN,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        zorder=3,
    )
    ax.add_patch(rect)
    third_w = (_ZONE_X_MAX - _ZONE_X_MIN) / 3
    third_h = (_ZONE_Z_MAX - _ZONE_Z_MIN) / 3
    for i in (1, 2):
        ax.plot(
            [_ZONE_X_MIN + i * third_w, _ZONE_X_MIN + i * third_w],
            [_ZONE_Z_MIN, _ZONE_Z_MAX],
            color="black",
            lw=0.8,
            zorder=3,
        )
        ax.plot(
            [_ZONE_X_MIN, _ZONE_X_MAX],
            [_ZONE_Z_MIN + i * third_h, _ZONE_Z_MIN + i * third_h],
            color="black",
            lw=0.8,
            zorder=3,
        )


def _zone_scatter(df: pl.DataFrame, title: str, colorby: str) -> bytes:
    """Pitch-location scatter shared by strike_zone / batter_zone / matchup_zone."""
    needed = df.select([c for c in ("plate_x", "plate_z", colorby) if c in df.columns])
    needed = needed.drop_nulls(subset=["plate_x", "plate_z"])
    if colorby in needed.columns:
        needed = needed.drop_nulls(subset=[colorby])
    if needed.is_empty():
        raise ValueError(f"No plottable pitches after filtering on {colorby!r}.")

    fig, ax = _new_fig((8, 8))
    try:
        # One scatter artist per color group, not per pitch — league-wide
        # frames would otherwise create tens of thousands of artists.
        if colorby in needed.columns:
            groups = list(needed.group_by(colorby, maintain_order=True))
        else:
            groups = [((None,), needed)]
        for (key,), group in groups:
            key_str = str(key or "")
            ax.scatter(
                group["plate_x"].to_numpy(),
                group["plate_z"].to_numpy(),
                c=_PITCH_COLORS.get(key_str, _DEFAULT_PITCH_COLOR),
                label=key_str,
                alpha=0.6,
                s=40,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )
        _draw_strike_zone(ax)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0.0, 5.0)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal Location (ft, catcher's view)")
        ax.set_ylabel("Vertical Location (ft)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        return _save_png(fig)
    finally:
        gc.collect()  # load-bearing: see the module docstring


# ---------------------------------------------------------------------------
# Strike zone family
# ---------------------------------------------------------------------------


def strike_zone(df: pl.DataFrame, title: str, colorby: str = "pitch_type") -> bytes:
    """Pitch-location scatter with the MLB strike zone overlay."""
    return _zone_scatter(df, title, colorby)


def batter_zone(df: pl.DataFrame, title: str) -> bytes:
    """Same rendering as strike_zone, framed as 'pitches received'."""
    return _zone_scatter(df, title, "pitch_type")


def matchup_zone(df: pl.DataFrame, title: str) -> bytes:
    """Strike zone for a pre-filtered pitcher-vs-batter matchup frame."""
    return _zone_scatter(df, title, "pitch_type")


# ---------------------------------------------------------------------------
# Hot zones (3x3 BA grid)
# ---------------------------------------------------------------------------


def hot_zones(df: pl.DataFrame, title: str) -> bytes:
    """Render a 3x3 batting-average heatmap by Statcast zone (1..9).

    Catcher's-POV layout: zones 1-3 top, 4-6 middle, 7-9 bottom.
    """
    grid_dict = hot_zones_grid(df)
    # Check AB counts, not the averages: an 0-for-everything cold slump has
    # all-zero averages but real at-bats and should still render.
    if not any(hot_zones_ab_counts(df).values()):
        raise ValueError("No at-bats recorded in the strike zone for this dataset.")

    import matplotlib.colors as mcolors
    import numpy as np

    grid = np.array(
        [
            [grid_dict[1], grid_dict[2], grid_dict[3]],
            [grid_dict[4], grid_dict[5], grid_dict[6]],
            [grid_dict[7], grid_dict[8], grid_dict[9]],
        ]
    )

    fig, ax = _new_fig((6, 6), dpi=100)
    try:
        norm = mcolors.TwoSlopeNorm(vmin=0.100, vcenter=0.250, vmax=0.400)
        im = ax.imshow(grid, cmap="RdYlBu_r", norm=norm, interpolation="nearest")

        for i in range(3):
            for j in range(3):
                val = float(grid[i, j])
                color = "white" if val > 0.35 or val < 0.15 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}".replace("0.", "."),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                )

        ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(4):
            ax.axhline(i - 0.5, color="black", lw=2)
            ax.axvline(i - 0.5, color="black", lw=2)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
        cbar.ax.set_ylabel("Batting Average", rotation=-90, va="bottom")
        return _save_png(fig)
    finally:
        gc.collect()  # load-bearing: see the module docstring


# ---------------------------------------------------------------------------
# Stadium drawing (also used by spray_chart)
# ---------------------------------------------------------------------------

# Palette for the park diagram.
_STANDS = "#b5b5b5"
_FOUL_GROUND = "#4a7a4a"
_FAIR_GRASS = "darkgreen"
_WALL = "#222222"
_WARNING_TRACK = "sienna"
_DIRT = "#d2b48c"


def _draw_stadium(ax: matplotlib.axes.Axes, stadium_key: str, label: bool = True) -> None:
    """Draw a ballpark from its real outline, in Savant's hc_x / hc_y grid.

    The axes use that grid directly (y grows toward home plate, so the y axis
    is inverted), which is what lets batted balls be scattered on it as they
    come from Statcast. An unknown key draws the generic park.
    """
    from bigleague import stadium_paths

    key = stadium_key if stadium_key in stadium_paths.known_parks() else "generic"
    seg = stadium_paths.paths(key)

    ax.set_facecolor("lightgray")
    if "outfield_inner" in seg:
        _draw_real_park(ax, key, seg, label)
    else:
        _draw_generic_park(ax, seg)

    outer_x, outer_y = seg["outfield_outer"]
    pad = 5.0
    ax.set_xlim(min(outer_x) - pad, max(outer_x) + pad)
    ax.set_ylim(max(outer_y) + pad, min(outer_y) - pad)  # y grows toward home plate
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _add_poly(
    ax: matplotlib.axes.Axes, path: tuple[list[float], list[float]], **kwargs: Any
) -> matplotlib.patches.Polygon:
    import matplotlib.patches as mpatches

    patch = mpatches.Polygon(list(zip(*path, strict=True)), closed=True, **kwargs)
    ax.add_patch(patch)
    return patch


def _draw_real_park(
    ax: matplotlib.axes.Axes,
    key: str,
    seg: dict[str, tuple[list[float], list[float]]],
    label: bool,
) -> None:
    """One of the 30 parks: stands, wall, foul and fair ground, dirt and grass."""
    import matplotlib.patches as mpatches

    _add_poly(
        ax, seg["outfield_outer"], facecolor=_STANDS, edgecolor="#777777", linewidth=0.8, zorder=0
    )

    # The foul-line path is a loop that starts and ends at home plate, out to
    # each pole and back; the poles are its leftmost and rightmost points.
    foul_pts = list(zip(*seg["foul_lines"], strict=True))
    home_x, home_y = max(foul_pts, key=lambda pt: pt[1])
    lx, ly = min(foul_pts)
    rx, ry = max(foul_pts)

    # Everything inside the wall, foul ground included, then the fair-territory
    # wedge (home plate through each foul pole, extended) over it in a brighter
    # green, so foul ground reads as foul ground.
    reach = 10.0
    wedge = mpatches.Polygon(
        [
            (home_x, home_y),
            (home_x + reach * (lx - home_x), home_y + reach * (ly - home_y)),
            (home_x + reach * (rx - home_x), home_y + reach * (ry - home_y)),
        ],
        closed=True,
        transform=ax.transData,
    )
    wall = _add_poly(ax, seg["outfield_inner"], facecolor=_FOUL_GROUND, edgecolor="none", zorder=1)
    fair = _add_poly(ax, seg["outfield_inner"], facecolor=_FAIR_GRASS, edgecolor="none", zorder=1.1)
    fair.set_clip_path(wedge)

    # Warning track: a wide stroke along the wall, clipped to the inside of it.
    xs, ys = seg["outfield_inner"]
    (track,) = ax.plot(
        xs, ys, color=_WARNING_TRACK, lw=9, alpha=0.7, zorder=1.2, solid_capstyle="butt"
    )
    track.set_clip_path(wall)
    ax.plot(xs + xs[:1], ys + ys[:1], color=_WALL, lw=2, zorder=1.3)

    _add_poly(ax, seg["infield_outer"], facecolor=_DIRT, edgecolor="none", zorder=2)
    _add_poly(ax, seg["infield_inner"], facecolor=_FAIR_GRASS, edgecolor="none", zorder=2.1)
    ax.plot([lx, home_x, rx], [ly, home_y, ry], color="white", lw=1.4, zorder=2.5)
    _add_poly(
        ax, seg["home_plate"], facecolor="white", edgecolor="#555555", linewidth=0.6, zorder=2.6
    )

    if label:
        from bigleague.stadiums import STADIUM_DATA

        meta = STADIUM_DATA[key]
        # CF sits at the wall's farthest point from home plate.
        cx, cy = min(zip(xs, ys, strict=True), key=lambda pt: pt[1])
        style: dict[str, Any] = {"color": "white", "fontsize": 9, "zorder": 2.7}
        ax.text(lx + 4, ly - 4, f"LF {int(meta['lf'])}'", ha="left", va="bottom", **style)
        ax.text(cx, cy + 5, f"CF {int(meta['cf'])}'", ha="center", va="top", **style)
        ax.text(rx - 4, ry - 4, f"RF {int(meta['rf'])}'", ha="right", va="bottom", **style)


def _draw_generic_park(
    ax: matplotlib.axes.Axes, seg: dict[str, tuple[list[float], list[float]]]
) -> None:
    """The generic park: a plain fan-shaped field, no stands or foul ground."""
    xs, ys = seg["outfield_outer"]
    pts = list(zip(xs, ys, strict=True))
    home_x, home_y = max(pts, key=lambda pt: pt[1])
    lx, ly = min(pts)
    rx, ry = max(pts)

    wall = _add_poly(ax, seg["outfield_outer"], facecolor=_FAIR_GRASS, edgecolor="none", zorder=1)
    # The wall is the arc beyond the poles; the rest of the path is the foul lines.
    arc = [pt for pt in pts if pt[1] <= max(ly, ry) + 0.5]
    arc_x, arc_y = zip(*arc, strict=True)
    (track,) = ax.plot(
        arc_x, arc_y, color=_WARNING_TRACK, lw=9, alpha=0.7, zorder=1.2, solid_capstyle="butt"
    )
    track.set_clip_path(wall)
    ax.plot(arc_x, arc_y, color=_WALL, lw=2, zorder=1.3)

    _add_poly(ax, seg["infield_outer"], facecolor=_DIRT, edgecolor="none", zorder=2)
    ax.plot([lx, home_x, rx], [ly, home_y, ry], color="white", lw=1.4, zorder=2.5)
    ax.plot(
        [home_x], [home_y], marker="p", color="white", markeredgecolor="#555555", ms=7, zorder=2.6
    )


# ---------------------------------------------------------------------------
# Spray chart
# ---------------------------------------------------------------------------


def _hit_color_expr() -> pl.Expr:
    """Marker color per batted-ball outcome: HR gold, other hits green,
    everything else (outs / unknown) lightcoral."""
    return (
        pl.when(pl.col("events") == "home_run")
        .then(pl.lit("gold"))
        .when(pl.col("events").is_in(["single", "double", "triple"]))
        .then(pl.lit("green"))
        .otherwise(pl.lit("lightcoral"))
    )


def spray_chart(df: pl.DataFrame, title: str, stadium_key: str = "generic") -> bytes:
    """Batted-ball scatter overlaid on a stadium outline.

    Filters to in-play events with hc_x/hc_y coords. If a non-generic
    stadium is selected, also filters to home games at that park. The park is
    drawn from its real outline, in the same pixel grid Savant reports hit
    coordinates in, so the balls are plotted exactly as given.
    """
    needed_cols = [c for c in ("type", "hc_x", "hc_y", "events", "home_team") if c in df.columns]
    data = df.select(needed_cols)
    data = data.filter(pl.col("type") == "X").drop_nulls(subset=["hc_x", "hc_y"])
    if data.is_empty():
        raise ValueError("No batted-ball events with coordinates to plot.")

    team_code = STADIUM_TO_TEAM_CODE.get(stadium_key)
    if team_code and "home_team" in data.columns:
        # Accept any era's code for the park: pre-2025 frames carry OAK
        # where current ones carry ATH.
        codes = [team_code, *LEGACY_TEAM_CODES.get(team_code, ())]
        filtered = data.filter(pl.col("home_team").is_in(codes))
        if filtered.is_empty():
            raise ValueError(f"No batted-ball data at {stadium_key} park (team_code={team_code}).")
        data = filtered

    color_expr = _hit_color_expr() if "events" in data.columns else pl.lit("lightcoral")
    data = data.with_columns(color_expr.alias("_color"))

    fig, ax = _new_fig((8, 8))
    try:
        _draw_stadium(ax, stadium_key)

        # One scatter artist per color bucket, not per batted ball —
        # league-wide frames would otherwise create one artist per pitch.
        for (color,), group in data.group_by("_color", maintain_order=True):
            ax.scatter(
                group["hc_x"].cast(pl.Float64).to_numpy(),
                group["hc_y"].cast(pl.Float64).to_numpy(),
                c=str(color),
                s=30,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
            )

        import matplotlib.patches as mpatches

        legend_elements = [
            mpatches.Patch(color="gold", label="HR"),
            mpatches.Patch(color="green", label="Hit"),
            mpatches.Patch(color="lightcoral", label="Out"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        return _save_png(fig)
    finally:
        gc.collect()  # load-bearing: see the module docstring


# ---------------------------------------------------------------------------
# Stadium outline (no batted balls)
# ---------------------------------------------------------------------------


def stadium_outline(stadium_key: str, title: str = "") -> bytes:
    """Standalone ballpark diagram, drawn from the park's real outline."""
    meta = STADIUM_DATA.get(stadium_key)
    if meta is None:
        raise ValueError(f"Unknown stadium key: {stadium_key!r}")

    fig, ax = _new_fig((8, 8))
    try:
        _draw_stadium(ax, stadium_key)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        else:
            ax.set_title(f"{meta['name']} — {meta['location']}", fontsize=12, fontweight="bold")
        return _save_png(fig)
    finally:
        gc.collect()  # load-bearing: see the module docstring
