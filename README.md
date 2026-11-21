# bigleague

[![PyPI](https://img.shields.io/pypi/v/bigleague)](https://pypi.org/project/bigleague/)
[![Python](https://img.shields.io/pypi/pyversions/bigleague)](https://pypi.org/project/bigleague/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Async, [polars](https://pola.rs)-native baseball data and plotting. Pull
Statcast pitches, FanGraphs stats, MLB Stats API schedules and rosters, and
Baseball Savant leaderboards into polars DataFrames, then render strike
zones, hot zones and spray charts to PNG. No pandas, no pybaseball.

```python
import asyncio
from datetime import date

import bigleague as bl


async def main() -> None:
    ohtani = await bl.resolve_id("Shohei", "Ohtani")  # MLBAM id
    df = await bl.statcast_batter(ohtani, date(2025, 4, 1), date(2025, 4, 30))
    print(bl.hot_cold_batter(df))  # {pa: 87, ab: 76, h: 22, avg: 0.289, ...}
    png = bl.strike_zone(df, "Shohei Ohtani, April 2025")  # PNG bytes


asyncio.run(main())
```

## Why bigleague

- **Polars everywhere.** DataFrames are polars end to end: smaller in memory,
  faster joins, cleaner null handling. pandas is not a dependency.
- **Async-first.** Built on httpx and asyncio, so you can fan out requests
  without threads. Baseball Savant fetches are capped at 5 concurrent
  requests, the rate Savant tolerates, so a big `asyncio.gather` won't get
  you blocked.
- **Plots without pybaseball.** Strike zones, hot-zone heatmaps, spray charts
  and stadium outlines are drawn with matplotlib patches. All renderers
  return PNG `bytes` and never touch the filesystem.
- **A cache that respects fresh data.** Statcast chunks are cached as parquet.
  Chunks that include today's date are deliberately not cached, so you never
  get stale partial data.
- **Typed pitch schema.** `StatcastPitch` and `STATCAST_SCHEMA` give you the
  Statcast column set with polars dtypes.

## Install

```bash
pip install bigleague
# or
uv add bigleague
```

Requires Python 3.12 or newer. The test suite runs on 3.12, 3.13 and 3.14.

Optional extra: `bigleague[bloom]` installs
[bouquet-bloom](https://pypi.org/project/bouquet-bloom/), which puts a bloom
filter in front of the cache so lookups for never-cached keys skip the disk.
The cache behaves the same without it.

## What's in the box

| Area | Functions | Source |
| --- | --- | --- |
| Pitch-level data | `statcast`, `statcast_batter`, `statcast_pitcher`, `statcast_season` | Baseball Savant |
| Savant leaderboards | `batter_percentile_ranks`, `pitcher_percentile_ranks`, `batter_pitch_arsenal`, `pitcher_pitch_arsenal`, `batter_exit_velo`, `pitcher_exit_velo`, `batter_expected_stats`, `pitcher_expected_stats`, `sprint_speed`, `pitcher_pitch_movement` | Baseball Savant |
| ABS challenges | `bl.abs.challenges`, `team_summary`, `team_top_challengers`, `player`, `league_leaders` | Baseball Savant |
| Season stats | `batting_stats`, `pitching_stats`, `fielding_stats`, `team_batting`, `team_pitching`, `*_range` variants, `player_stats`, `career_stats`, `top_leaderboard`, `playoff_odds` | FanGraphs |
| Schedules and rosters | `schedule`, `live_scores`, `next_game`, `recent_results`, `standings`, `roster`, `injuries`, `transactions` | MLB Stats API |
| Player lookup | `lookup`, `resolve_id`, `reverse_lookup`, `mlb_years` | Chadwick Bureau register |
| Plots | `strike_zone`, `batter_zone`, `matchup_zone`, `hot_zones`, `spray_chart`, `stadium_outline` | matplotlib |
| Pure stats helpers | `hot_cold_batter`, `hot_cold_pitcher`, `matchup_stats`, `hot_zones_grid`, `aggregate_career`, `season_range`, `parse_*` | computed locally |
| Stadiums | `normalize_stadium`, `stadium_metadata`, `stadium_team_code`, `known_stadiums` | bundled data |

Everything is importable from the top level (`import bigleague as bl`) except
`mlb.players()` and `mlb.players_multi_season()`, which would collide with the
`bigleague.players` submodule. Use `bl.mlb.players(...)` for those.

## Examples

### Statcast

```python
# One player, any date range. Player queries are split by calendar year, so
# each request stays under Savant's ~25k-row cap and past years cache cleanly.
df = await bl.statcast_pitcher(543037, date(2025, 4, 1), date(2025, 9, 30))

# League-wide or one team. These are split into 7-day chunks because a
# single day league-wide exceeds Savant's ~25k-row CSV cap.
week = await bl.statcast(date(2025, 4, 1), date(2025, 4, 7), team="NYY")

# Player ids are MLBAM ids. Look them up from a name:
await bl.resolve_id("Gerrit", "Cole")  # -> 543037
await bl.lookup("Cole", "Gerrit")  # -> polars DataFrame of register rows
```

### FanGraphs

```python
stats = await bl.player_stats("Shohei", "Ohtani", 2025)
leaders = await bl.top_leaderboard("WAR", 2025, n=10)
table = await bl.batting_stats(2024, 2025, qual=100)  # polars DataFrame

try:
    stats = await bl.player_stats("Gerrit", "Cole", 2025)
except bl.FangraphsBlockedError:
    pass  # transient Cloudflare block; retry later
```

### MLB Stats API

```python
games = await bl.live_scores("NYY")
game = await bl.next_game("NYY")
# game = {date, time, home_team, away_team, home_probable, away_probable}

season = await bl.schedule("NYY", 2025)
# One row per game in play order:
# {date, start_time, game_pk, is_home, opponent, status, score, record}
# status is "W", "L", "P" (postponed) or "S" (scheduled); record is the
# running "W-L" after that game.
last5 = [g for g in season if g["status"] in ("W", "L")][-5:]
```

Team arguments take standard abbreviations (`"NYY"`, `"LAD"`, ...) and raise
`ValueError` for anything unknown.

### Plots

Every renderer takes a Statcast DataFrame and a title and returns PNG bytes:

```python
png = bl.strike_zone(df, title)  # pitch locations, colored by pitch type
png = bl.batter_zone(df, title)
png = bl.matchup_zone(df, title)
png = bl.hot_zones(df, title)  # 3x3 batting-average heatmap by zone
png = bl.spray_chart(df, title, stadium_key="yankees")
png = bl.stadium_outline("yankees")

Path("zone.png").write_bytes(png)
```

Matplotlib figures are closed in a `finally` block on every render, so
long-running services don't leak bitmaps.

### Stadiums

```python
bl.normalize_stadium("Guardians")  # -> "indians"  (the Statcast key)
bl.normalize_stadium("Wizards")  # -> "generic"
bl.stadium_metadata("yankees")  # -> {"name": "Yankee Stadium", ...}
```

### ABS challenges

`bigleague.abs` wraps Baseball Savant's Automated Ball-Strike challenge
leaderboards (batter, pitcher and catcher roles, plus a team summary feed).

```python
row = await bl.abs.team_summary("NYY", 2025)
leaders = await bl.abs.league_leaders(2025, limit=10)
```

## Cache

| Aspect | Behavior |
| --- | --- |
| Backend | parquet on disk, optional bloom filter (`bloom` extra) |
| Location | `~/.bigleague/cache`; override with `BIGLEAGUE_CACHE_DIR=/path` |
| Disable | `BIGLEAGUE_NO_CACHE=1` |
| Fresh data | chunks where `end_date >= today` are never cached |
| Eviction | none; manage manually with `bl.cache.clear()` and `bl.cache.size()` |

A request that includes today always re-fetches until its chunk ages into the
past. To force fresh data for a backfill, set `BIGLEAGUE_NO_CACHE=1` for that
run. The player register from the Chadwick Bureau is cached in the same
directory.

## Benchmarks

The same Ohtani (660271) Statcast pulls against real Savant, each library in
its own subprocess for clean RSS. These are the author's runs; your numbers
will vary with your network and machine.

| lib            | scenario   |  rows |  frame |  peak heap |  peak RSS |  wall |
| -------------- | ---------- | ----: | -----: | ---------: | --------: | ----: |
| bigleague      | month cold |   426 | 194 KB |      52 MB |    161 MB | 1.35s |
| pybaseball     | month cold |   426 | 433 KB |      85 MB |    260 MB | 2.69s |
| pandas+pyarrow | month cold |   426 | 375 KB |      48 MB |    167 MB | 1.27s |
| bigleague      | month warm |   426 | 193 KB |      48 MB |    151 MB | 1.20s |
| pybaseball     | month warm |   426 | 433 KB |      85 MB |    260 MB | 2.60s |
| pandas+pyarrow | month warm |   426 | 375 KB |      48 MB |    166 MB | 1.27s |
| bigleague      | season     |  2859 | 1.3 MB |      55 MB |    168 MB | 1.40s |
| pybaseball     | season     |  2850 | 2.8 MB |      96 MB |    281 MB | 2.71s |
| pandas+pyarrow | season     |  2859 | 2.5 MB |      51 MB |    176 MB | 1.29s |
| bigleague      | hotzones   |   426 | 193 KB |      48 MB |    158 MB | 1.23s |
| pybaseball     | hotzones   |   426 | 433 KB |      99 MB |    290 MB | 2.93s |
| pandas+pyarrow | hotzones   |   426 | 375 KB |      80 MB |    250 MB | 2.21s |

- **In-memory frames are about 2x smaller** than pybaseball and about 1.5 to
  2x smaller than pandas with the pyarrow backend, across every scenario.
- **Peak heap and RSS** track frame size: bigleague and pandas+pyarrow are
  comparable (Arrow under both); pybaseball is about 1.7x heavier on heap and
  about 70% heavier on RSS.
- **Warm reads** drop from 1.35s to 1.20s on the same window via the parquet
  cache. pybaseball and pandas+pyarrow have no cache here, so their "warm"
  timing is just Savant CDN noise.
- **Player-scoped queries** chunk by calendar year rather than every 7 days,
  so a single-season pull is one request and takes about 1.4s cold instead of
  about 6s.

Reproduce with `uv run python bench/compare_pybaseball.py`. It needs network
access and extra packages (`uv pip install pybaseball pandas pyarrow`), and
it isn't run in CI.

## Data sources and responsible use

bigleague is an independent project. It is not affiliated with or endorsed by
Major League Baseball, Baseball Savant, FanGraphs or the Chadwick Bureau.

It reads public endpoints that those sites don't document as a stable API:

- **Baseball Savant** CSV endpoints, rate-limited by the library to 5
  concurrent requests.
- **FanGraphs** JSON endpoints, fetched with
  [curl_cffi](https://github.com/lexiforest/curl_cffi) Chrome impersonation,
  because Cloudflare blocks plain TLS clients. Expect the occasional
  `FangraphsBlockedError`; it is transient.
- **MLB Stats API** (`statsapi.mlb.com`).
- **Chadwick Bureau register** for player ids, fetched from a pinned commit.

Endpoints and terms can change without notice, so check each provider's terms
of use before building anything commercial on top. Keep the cache on, since
repeat queries shouldn't hit these servers again.

## Development

```bash
git clone https://github.com/klobucar/bigleague
cd bigleague
uv sync --extra dev                  # add `--extra bloom` to test the bloom path
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

All HTTP in the test suite is mocked (`respx` for httpx, monkey-patching for
curl_cffi), and `tests/conftest.py` sets `BIGLEAGUE_NO_CACHE=1`. A VCR layer
(`pytest-recording`) is wired up but no cassettes are checked in; to record
one against a live endpoint:

```bash
uv run pytest tests/test_savant.py --record-mode=once
```

### Layout

```
bigleague/
├── models.py     # StatcastPitch dataclass, the schema source of truth
├── schema.py     # polars dtype map derived from models
├── http.py       # shared async client, semaphore, retries
├── cache.py      # parquet cache, optional bloom filter
├── stadiums.py   # stadium data and name normalization
├── stats.py      # pure polars computation, no I/O
├── players.py    # Chadwick Bureau lookup
├── mlb.py        # MLB Stats API
├── savant.py     # Baseball Savant: pitch chunks and leaderboards
├── abs.py        # Baseball Savant: ABS challenge leaderboards
├── fangraphs.py  # FanGraphs JSON API (curl_cffi)
└── plots.py      # matplotlib renderers
```

## Used by

[harry-bot](https://github.com/klobucar/harry-bot), a Discord baseball bot.

## License

[MIT](LICENSE)
