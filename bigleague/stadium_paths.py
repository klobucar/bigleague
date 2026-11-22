"""Real ballpark outlines, in Baseball Savant's hit-coordinate grid.

`plots.spray_chart` and `plots.stadium_outline` draw each park from these
paths, so Yankee Stadium's short right-field porch or Fenway's Green Monster
look like themselves rather than like a smooth arc scaled to three wall
distances.

The paths are the `MLBStadiumsPathData` table from Ben Dilday's
GeomMLBStadiums (MIT), as redistributed in pybaseball. They are bundled in
`data/stadium_paths.csv.gz` (96 KB for all 30 parks plus a generic one); see
`data/LICENSE-stadium-paths` for the notice and `docs/build_stadium_paths.py`
for how the file is built. The x/y values are in the same pixel grid as
Statcast's `hc_x` / `hc_y`, with home plate near (125.4, 199) and y growing
toward home plate, so batted balls can be plotted on them as they are.

Each of the 30 parks has six segments, each an ordered path of points. (The
"generic" park is a plain fan-shaped field and has only `outfield_outer` and
`infield_outer`.)

- `outfield_outer`: the back of the stands, the outside edge of the park
- `outfield_inner`: the outfield wall, around to behind home plate
- `foul_lines`: left pole, home plate, right pole, as one polyline
- `infield_outer`: the edge of the infield dirt
- `infield_inner`: the infield grass inside the dirt
- `home_plate`: the plate
"""

from __future__ import annotations

import functools
import gzip
import io
from importlib import resources

import polars as pl

SEGMENTS: tuple[str, ...] = (
    "outfield_outer",
    "outfield_inner",
    "infield_outer",
    "infield_inner",
    "foul_lines",
    "home_plate",
)

# One segment: (x values, y values), in path order.
Path = tuple[list[float], list[float]]


@functools.cache
def _table() -> pl.DataFrame:
    """All parks' points. Loaded once per process (about 1 MB)."""
    raw = (resources.files("bigleague") / "data" / "stadium_paths.csv.gz").read_bytes()
    return pl.read_csv(io.BytesIO(gzip.decompress(raw)))


@functools.cache
def _park(stadium_key: str) -> dict[str, Path]:
    df = _table().filter(pl.col("team") == stadium_key)
    if df.is_empty():
        raise ValueError(f"No outline for stadium {stadium_key!r}.")
    out: dict[str, Path] = {}
    for segment in SEGMENTS:
        pts = df.filter(pl.col("segment") == segment)
        if not pts.is_empty():
            out[segment] = (pts["x"].to_list(), pts["y"].to_list())
    return out


def paths(stadium_key: str) -> dict[str, Path]:
    """The outline segments a park has, by segment name.

    Raises ValueError for a key with no outline (use `known_parks()`). The
    returned lists are shared between calls; don't modify them.
    """
    return _park(stadium_key)


def known_parks() -> frozenset[str]:
    """Stadium keys that have an outline (the 30 parks plus "generic")."""
    return frozenset(_table()["team"].unique().to_list())
