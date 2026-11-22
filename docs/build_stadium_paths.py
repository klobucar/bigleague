"""Rebuild bigleague/data/stadium_paths.csv.gz from the upstream outline data.

The outlines are Ben Dilday's GeomMLBStadiums `MLBStadiumsPathData` (MIT), as
bundled by pybaseball in `pybaseball/data/mlbstadiums.csv`. This keeps only
what bigleague draws (team, segment, x, y), rounds to 2 decimals, and keeps the
original point order, which is the path order. The output is deterministic.

    curl -sLO https://raw.githubusercontent.com/jldbc/pybaseball/master/pybaseball/data/mlbstadiums.csv
    uv run python docs/build_stadium_paths.py mlbstadiums.csv
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parent.parent / "bigleague" / "data" / "stadium_paths.csv.gz"


def main(src: Path) -> None:
    df = pl.read_csv(src).select(
        "team",
        "segment",
        pl.col("x").round(2),
        pl.col("y").round(2),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(OUT, "wb", compresslevel=9, mtime=0) as fh:
        fh.write(df.write_csv().encode())
    print(
        f"{df.height} points, {df['team'].n_unique()} parks -> {OUT} ({OUT.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
