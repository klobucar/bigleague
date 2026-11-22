"""Regenerate the README gallery images in docs/img/.

Pulls Shohei Ohtani's 2025 Statcast data from Baseball Savant, so it needs
network access and isn't run in CI. Run it from the repo root:

    uv run python -m docs.make_gallery
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import bigleague as bl

OHTANI = 660271  # MLBAM id
YEAR = 2025
OUT = Path(__file__).parent / "img"


async def main() -> None:
    season = await bl.statcast_season(OHTANI, YEAR, "batter")
    # A full season of pitches is an unreadable blob on a strike-zone plot.
    april = await bl.statcast_batter(OHTANI, date(YEAR, 4, 1), date(YEAR, 4, 30))
    print(f"{season.height} pitches in {YEAR}, {april.height} in April")

    images = {
        "strike_zone.png": bl.strike_zone(april, f"Shohei Ohtani, April {YEAR}"),
        "hot_zones.png": bl.hot_zones(season, f"Shohei Ohtani, {YEAR}"),
        "spray_chart.png": bl.spray_chart(season, f"Shohei Ohtani, {YEAR}", stadium_key="dodgers"),
        "stadium_outline.png": bl.stadium_outline("dodgers", "Dodger Stadium"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for name, png in images.items():
        (OUT / name).write_bytes(png)
        print(f"wrote {OUT / name} ({len(png) / 1024:.0f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
