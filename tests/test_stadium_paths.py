"""Park outlines: the data, and that spray charts are drawn on it correctly."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import matplotlib.axes
import polars as pl
import pytest
from bigleague import plots, stadium_paths
from bigleague.stadiums import STADIUM_DATA

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PARKS = sorted(STADIUM_DATA)


def _area(xs: list[float], ys: list[float]) -> float:
    """Shoelace area of a closed path, in square pixels."""
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2


# ── the data ─────────────────────────────────────────────────────────────────


def test_every_park_bigleague_knows_has_an_outline() -> None:
    assert stadium_paths.known_parks() == set(STADIUM_DATA)


@pytest.mark.parametrize("park", _PARKS)
def test_each_outline_is_complete_and_in_the_savant_grid(park: str) -> None:
    seg = stadium_paths.paths(park)
    if park == "generic":  # a plain fan-shaped field: just a wall and the dirt
        assert set(seg) == {"outfield_outer", "infield_outer"}
    else:
        assert set(seg) == set(stadium_paths.SEGMENTS)
    for name, (xs, ys) in seg.items():
        assert len(xs) == len(ys) >= 3, name
        # Savant's hc_x / hc_y grid is roughly 0-250 on both axes.
        assert min(xs) > 0
        assert max(xs) < 260
        assert min(ys) > 0
        assert max(ys) < 260
    if park != "generic":
        # Home plate sits near (125, 199) in that grid.
        hx, hy = seg["home_plate"]
        assert sum(hx) / len(hx) == pytest.approx(125.4, abs=2.0)
        assert sum(hy) / len(hy) == pytest.approx(200.5, abs=3.0)


def test_unknown_park_has_no_outline() -> None:
    with pytest.raises(ValueError, match="No outline"):
        stadium_paths.paths("not_a_park")


def test_parks_really_do_have_different_shapes() -> None:
    """The point of the data: each park is its own shape, not one arc rescaled."""
    walls = {k: _area(*stadium_paths.paths(k)["outfield_inner"]) for k in ("yankees", "red_sox")}
    assert abs(walls["yankees"] - walls["red_sox"]) > 1_000  # px², of roughly 25,000
    # Yankee Stadium's short right-field porch: the right pole is nearer the
    # wall's apex than at Fenway's. Compare the wall's depth in right field.
    fenway = stadium_paths.paths("red_sox")["outfield_inner"]
    yankee = stadium_paths.paths("yankees")["outfield_inner"]
    assert fenway != yankee


# ── drawing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("park", _PARKS)
def test_every_park_renders_as_an_outline_and_a_spray_chart(park: str) -> None:
    assert plots.stadium_outline(park).startswith(_PNG_MAGIC)
    df = pl.DataFrame({"type": ["X"], "hc_x": [125.0], "hc_y": [120.0], "events": ["single"]})
    assert plots.spray_chart(df, "t", stadium_key=park).startswith(_PNG_MAGIC)


def test_unknown_park_key_draws_the_generic_park_instead_of_failing() -> None:
    df = pl.DataFrame({"type": ["X"], "hc_x": [125.0], "hc_y": [120.0], "events": ["single"]})
    assert plots.spray_chart(df, "t", stadium_key="nowhere").startswith(_PNG_MAGIC)


def test_hits_are_plotted_at_their_raw_savant_coordinates() -> None:
    """The outline is in Savant's grid, so hits go on it unconverted. (They
    used to be rescaled to feet and drawn on an idealized arc.)"""
    df = pl.DataFrame(
        {
            "type": ["X", "X", "X"],
            "hc_x": [100.5, 125.0, 150.25],
            "hc_y": [80.0, 90.5, 100.0],
            "events": ["single", "single", "single"],
            "home_team": ["NYY"] * 3,
        }
    )
    captured: list[tuple[Any, ...]] = []
    real = matplotlib.axes.Axes.scatter

    def _spy(self: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append((list(x), list(y)))
        return real(self, x, y, *args, **kwargs)

    with patch.object(matplotlib.axes.Axes, "scatter", _spy):
        plots.spray_chart(df, "t", stadium_key="yankees")

    assert captured == [([100.5, 125.0, 150.25], [80.0, 90.5, 100.0])]


def test_distance_labels_sit_at_the_foul_poles_not_at_home_plate() -> None:
    """Regression: the foul-line path loops out and back from home plate, so
    taking its first/last points put the LF/RF labels beside home plate."""
    texts: dict[str, tuple[float, float]] = {}
    real = matplotlib.axes.Axes.text

    def _spy(self: Any, x: float, y: float, s: str, *args: Any, **kwargs: Any) -> Any:
        texts[s] = (x, y)
        return real(self, x, y, s, *args, **kwargs)

    with patch.object(matplotlib.axes.Axes, "text", _spy):
        plots.stadium_outline("yankees")

    seg = stadium_paths.paths("yankees")
    left_pole_x = min(seg["foul_lines"][0])
    right_pole_x = max(seg["foul_lines"][0])
    assert texts["LF 318'"][0] == pytest.approx(left_pole_x + 4)
    assert texts["RF 314'"][0] == pytest.approx(right_pole_x - 4)
    assert texts["LF 318'"][1] < 150  # out in the field, not down by home plate (y ~ 200)
    assert texts["RF 314'"][1] < 150
    assert texts["CF 408'"][1] < texts["LF 318'"][1]  # center field is the deepest point


def test_fair_territory_is_clipped_to_the_wedge_between_the_foul_lines() -> None:
    """Regression: with the wrong poles the wedge was degenerate and the whole
    field drew as foul ground."""
    _fig, ax = plots._new_fig((6, 6))
    plots._draw_stadium(ax, "yankees")
    fair = [
        p
        for p in ax.patches
        if p.get_facecolor()[:3] == (0.0, 100 / 255, 0.0) and p.get_zorder() == 1.1
    ]
    assert len(fair) == 1
    assert fair[0].get_clip_path() is not None
