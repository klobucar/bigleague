"""Matplotlib renderers — strike zone, hot zones, spray chart, stadium outline.

Every renderer:
  1. Selects only the columns it needs (huge perf win on Statcast frames).
  2. Validates non-empty input.
  3. Returns PNG bytes from an in-memory BytesIO buffer.
  4. Calls `plt.close(fig)` and `gc.collect()` in a `finally` block —
     mandatory. matplotlib holds a C-side bitmap until close().

`gc.collect()` is kept from an earlier pandas-based implementation, where
DataFrames in matplotlib reference cycles caused ~10 MiB RSS leaks
per chart. Polars frames are much smaller in memory; the explicit GC
may be removable. TODO: benchmark on Fly.io and drop if it doesn't move
the needle.
"""

from __future__ import annotations

import gc
from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # headless — must run before pyplot import

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from bigleague.stadiums import LEGACY_TEAM_CODES, STADIUM_DATA, STADIUM_TO_TEAM_CODE
from bigleague.stats import hot_zones_ab_counts, hot_zones_grid

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


def _draw_strike_zone(ax: matplotlib.axes.Axes) -> None:
    """Outer rectangle + 3x3 grid lines for the official MLB strike zone."""
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

    fig, ax = plt.subplots(figsize=(8, 8))
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
        plt.close(fig)
        # TODO: benchmark whether gc.collect() is still needed with polars
        gc.collect()


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

    grid = np.array(
        [
            [grid_dict[1], grid_dict[2], grid_dict[3]],
            [grid_dict[4], grid_dict[5], grid_dict[6]],
            [grid_dict[7], grid_dict[8], grid_dict[9]],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
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
        plt.close(fig)
        # TODO: benchmark whether gc.collect() is still needed with polars
        gc.collect()


# ---------------------------------------------------------------------------
# Stadium drawing primitives (also used by spray_chart)
# ---------------------------------------------------------------------------


def _outfield_arc(lf: float, cf: float, rf: float, n: int = 80) -> np.ndarray:
    """Smooth arc from left-foul corner through CF to right-foul corner.

    Polar interpolation: angle theta runs -45° -> 0° -> +45° (where 0
    is straight CF). Radius interpolates linearly between LF->CF on
    the left half and CF->RF on the right half.
    """
    angles = np.linspace(-np.pi / 4, np.pi / 4, n)
    radii = np.empty_like(angles)
    for i, t in enumerate(angles):
        if t < 0:
            frac = (t - (-np.pi / 4)) / (np.pi / 4)  # 0 at LF, 1 at CF
            radii[i] = lf + frac * (cf - lf)
        else:
            frac = t / (np.pi / 4)  # 0 at CF, 1 at RF
            radii[i] = cf + frac * (rf - cf)
    xs = radii * np.sin(angles)
    ys = radii * np.cos(angles)
    return np.column_stack([xs, ys])


def _draw_field(
    ax: matplotlib.axes.Axes,
    lf: float,
    cf: float,
    rf: float,
    label: bool = True,
) -> None:
    """Draw a ballpark: foul territory, fair grass, infield diamond, walls."""
    # Foul territory background
    ax.set_facecolor("lightgray")

    # Outfield wall + foul lines polygon (closed)
    arc = _outfield_arc(lf, cf, rf)
    fair = np.vstack([[0.0, 0.0], arc, [0.0, 0.0]])
    field_patch = mpatches.Polygon(
        fair,
        closed=True,
        facecolor="darkgreen",
        edgecolor="black",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(field_patch)

    # Warning track: shrink arc by ~15ft toward home
    track_inner = arc * (1 - 15.0 / np.linalg.norm(arc, axis=1, keepdims=True))
    track_poly = np.vstack([arc, track_inner[::-1]])
    track_patch = mpatches.Polygon(
        track_poly,
        closed=True,
        facecolor="sienna",
        edgecolor="none",
        alpha=0.7,
        zorder=2,
    )
    ax.add_patch(track_patch)

    # Foul lines
    ax.plot([0, arc[0, 0]], [0, arc[0, 1]], color="white", lw=1.2, zorder=3)
    ax.plot([0, arc[-1, 0]], [0, arc[-1, 1]], color="white", lw=1.2, zorder=3)

    # Infield diamond (90 ft bases) — square rotated 45°
    base = 90.0
    diamond = np.array(
        [
            [0.0, 0.0],
            [-base * np.sin(np.pi / 4), base * np.cos(np.pi / 4)],
            [0.0, base * np.sqrt(2)],
            [base * np.sin(np.pi / 4), base * np.cos(np.pi / 4)],
        ]
    )
    ax.add_patch(
        mpatches.Polygon(
            diamond,
            closed=True,
            facecolor="tan",
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    )

    if label:
        ax.text(
            arc[0, 0] - 5, arc[0, 1] + 8, f"LF {int(lf)}'", color="white", fontsize=9, ha="right"
        )
        ax.text(0, cf + 8, f"CF {int(cf)}'", color="white", fontsize=9, ha="center")
        ax.text(
            arc[-1, 0] + 5, arc[-1, 1] + 8, f"RF {int(rf)}'", color="white", fontsize=9, ha="left"
        )

    ax.set_xlim(-400, 400)
    ax.set_ylim(0, 450)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Spray chart
# ---------------------------------------------------------------------------


# Savant hc_x/hc_y come in a ~0-250 pixel grid with home plate at
# (125.42, 198.27); the field is drawn in feet, so scale by ~2.495 ft/px.
_SAVANT_PX_TO_FT = 2.495


def _savant_to_feet(
    hc_x: float | np.ndarray, hc_y: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert Savant pixel-grid hit coordinates to feet from home plate.

    Pure arithmetic, so it works elementwise on numpy arrays as well as
    scalars.
    """
    return (hc_x - 125.42) * _SAVANT_PX_TO_FT, (198.27 - hc_y) * _SAVANT_PX_TO_FT


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
    stadium is selected, also filters to home games at that park.
    Coordinate translation lifts Savant's pixel grid to feet from home.
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

    meta = STADIUM_DATA.get(stadium_key, STADIUM_DATA["generic"])
    lf, cf, rf = float(meta["lf"]), float(meta["cf"]), float(meta["rf"])

    color_expr = _hit_color_expr() if "events" in data.columns else pl.lit("lightcoral")
    data = data.with_columns(color_expr.alias("_color"))

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        _draw_field(ax, lf, cf, rf)

        # One scatter artist per color bucket, not per batted ball —
        # league-wide frames would otherwise create one artist per pitch.
        for (color,), group in data.group_by("_color", maintain_order=True):
            field_x, field_y = _savant_to_feet(
                group["hc_x"].cast(pl.Float64).to_numpy(),
                group["hc_y"].cast(pl.Float64).to_numpy(),
            )
            ax.scatter(
                field_x,
                field_y,
                c=str(color),
                s=30,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
            )

        legend_elements = [
            mpatches.Patch(color="gold", label="HR"),
            mpatches.Patch(color="green", label="Hit"),
            mpatches.Patch(color="lightcoral", label="Out"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        return _save_png(fig)
    finally:
        plt.close(fig)
        # TODO: benchmark whether gc.collect() is still needed with polars
        gc.collect()


# ---------------------------------------------------------------------------
# Stadium outline (no batted balls)
# ---------------------------------------------------------------------------


def stadium_outline(stadium_key: str, title: str = "") -> bytes:
    """Standalone ballpark outline using STADIUM_DATA dimensions."""
    meta = STADIUM_DATA.get(stadium_key)
    if meta is None:
        raise ValueError(f"Unknown stadium key: {stadium_key!r}")

    lf, cf, rf = float(meta["lf"]), float(meta["cf"]), float(meta["rf"])
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        _draw_field(ax, lf, cf, rf)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        else:
            ax.set_title(f"{meta['name']} — {meta['location']}", fontsize=12, fontweight="bold")
        return _save_png(fig)
    finally:
        plt.close(fig)
        # TODO: benchmark whether gc.collect() is still needed with polars
        gc.collect()
